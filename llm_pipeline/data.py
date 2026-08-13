"""Dataset loading, cleaning, statistics, token caching, and collators."""

from __future__ import annotations

import glob
import hashlib
import json
import math
import pickle
import random
import shutil
import statistics
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_write_json
from .data_governance import (
    FILTER_VERSION,
    POLICY_VERSION,
    load_benchmark_denylist,
    preference_rejection_categories,
    sample_rejection_categories,
)
from .data_reader import (
    PreferenceSample,
    TextSample,
    clean_text,
    escape_special_tokens,
    field_value,
    get_source_value,
    iter_rows,
    normalize_messages,
    read_rows,
    render_messages,
    stable_hash,
)
from .errors import DataPolicyError

STATS_RESERVOIR_SIZE = 100_000


def instruction_to_messages(
    row: dict[str, Any], config: dict[str, Any], source: dict[str, Any] | None
) -> list[dict[str, str]] | None:
    """Convert instruction/input/output rows into chat messages."""

    data_cfg = config["data"]
    instruction = clean_text(
        field_value(row, get_source_value(source, "instruction_field", ["instruction", "question", "prompt"])),
        data_cfg["normalize_nfkc"],
    )
    input_text = clean_text(
        field_value(row, get_source_value(source, "input_field", ["input", "context"])), data_cfg["normalize_nfkc"]
    )
    output = clean_text(
        field_value(row, get_source_value(source, "output_field", ["output", "answer", "chosen", "response"])),
        data_cfg["normalize_nfkc"],
    )
    if not instruction or not output:
        return None
    user_content = instruction if not input_text else f"{instruction}\n\n{input_text}"
    return [{"role": "user", "content": user_content}, {"role": "assistant", "content": output}]


def translation_to_sample(
    row: dict[str, Any], config: dict[str, Any], dataset_type: str, source: dict[str, Any] | None
) -> TextSample | None:
    """Convert an explicitly described parallel row into pretraining text or translation SFT."""

    data_cfg = config["data"]
    source_field = get_source_value(source, "source_lang_field", "source")
    target_field = get_source_value(source, "target_lang_field", "target")
    source_lang = str(get_source_value(source, "source_lang", source_field))
    target_lang = str(get_source_value(source, "target_lang", target_field))
    source_text = clean_text(row.get(source_field, ""), data_cfg["normalize_nfkc"])
    target_text = clean_text(row.get(target_field, ""), data_cfg["normalize_nfkc"])
    if not source_text or not target_text:
        return None
    # Both translation directions must land in the same dataset split. Hashing
    # each rendered direction independently could leak a held-out answer into
    # training through its reversed pair.
    canonical_pair = "\0".join(sorted((source_text, target_text)))
    split_key = f"translation:{canonical_pair}"
    endpoint_splits = {
        split_name_for_key(f"translation-endpoint:{source_field}:{source_text}", config),
        split_name_for_key(f"translation-endpoint:{target_field}:{target_text}", config),
    }
    split_meta: dict[str, Any] = {"_split_key": split_key}
    if len(endpoint_splits) == 1:
        split_meta["_forced_split"] = next(iter(endpoint_splits))
    else:
        # Keeping an edge whose two endpoint identities hash to different
        # splits would let a repeated source or target sentence cross the
        # evaluation boundary.  Dropping only conflicts is streaming-safe and
        # provides the same isolation as a global bipartite component split.
        split_meta["_split_conflict"] = True
    if dataset_type == "sft" or get_source_value(source, "as_messages", False):
        prompt = get_source_value(
            source,
            "prompt_template",
            "Translate this {source_lang} text into {target_lang}.\n{source_text}",
        )
        user_content = str(prompt).format(
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=source_text,
            target_text=target_text,
        )
        messages = [{"role": "user", "content": user_content}, {"role": "assistant", "content": target_text}]
        text, mask = render_messages(messages, config["tokenizer"]["special_tokens"])
        return TextSample(
            text=clean_text(text, data_cfg["normalize_nfkc"]),
            kind="sft",
            meta={**row, "messages": messages, **split_meta},
            labels_mask=mask,
        )
    text = get_source_value(source, "text_template", "{source_lang}: {source_text}\n{target_lang}: {target_text}")
    rendered = str(text).format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        target_text=target_text,
    )
    return TextSample(
        text=clean_text(rendered, data_cfg["normalize_nfkc"]),
        kind=dataset_type,
        meta={**row, **split_meta},
    )


def row_to_text_sample(
    row: Any,
    config: dict[str, Any],
    dataset_type: str | None = None,
    source: dict[str, Any] | None = None,
) -> TextSample | None:
    """Convert heterogeneous row formats into a normalized text sample."""

    data_cfg = config["data"]
    tok_cfg = config["tokenizer"]
    dataset_type = dataset_type or data_cfg.get("dataset_type", "pretrain")
    schema = str(get_source_value(source, "schema", "auto")).lower()

    if isinstance(row, str):
        text = clean_text(row, data_cfg["normalize_nfkc"])
        return TextSample(text=text, kind=dataset_type, meta={})

    if not isinstance(row, dict):
        return None

    messages_field = get_source_value(source, "messages_field", data_cfg["messages_field"])
    if messages_field not in row and "conversations" in row:
        messages_field = "conversations"
    text_field = get_source_value(source, "text_field", data_cfg["text_field"])
    prompt_field = get_source_value(source, "prompt_field", data_cfg["prompt_field"])
    chosen_field = get_source_value(source, "chosen_field", data_cfg["chosen_field"])

    source_field = str(get_source_value(source, "source_lang_field", "source"))
    target_field = str(get_source_value(source, "target_lang_field", "target"))
    explicit_parallel_fields = bool(
        source
        and ("source_lang_field" in source or "target_lang_field" in source)
        and source_field in row
        and target_field in row
    )
    if schema == "translation" or (schema == "auto" and explicit_parallel_fields):
        return translation_to_sample(row, config, dataset_type, source)

    if schema == "instruction" or (
        schema == "auto" and "instruction" in row and any(field in row for field in ("output", "answer", "chosen"))
    ):
        messages = instruction_to_messages(row, config, source)
        if not messages:
            return None
        text, mask = render_messages(messages, tok_cfg["special_tokens"])
        text = clean_text(text, data_cfg["normalize_nfkc"])
        return TextSample(text=text, kind="sft", meta={**row, data_cfg["messages_field"]: messages}, labels_mask=mask)

    if messages_field in row:
        messages = normalize_messages(row.get(messages_field) or [])
        text, mask = render_messages(messages, tok_cfg["special_tokens"])
        text = clean_text(text, data_cfg["normalize_nfkc"])
        meta = row if messages_field == data_cfg["messages_field"] else {**row, data_cfg["messages_field"]: messages}
        return TextSample(text=text, kind="sft", meta=meta, labels_mask=mask)

    if text_field in row:
        text = clean_text(row.get(text_field), data_cfg["normalize_nfkc"])
        return TextSample(text=text, kind=dataset_type, meta=row)

    if prompt_field in row and chosen_field in row:
        prompt = clean_text(row.get(prompt_field), data_cfg["normalize_nfkc"])
        chosen = clean_text(row.get(chosen_field), data_cfg["normalize_nfkc"])
        return TextSample(text=f"{prompt}\n{chosen}", kind="sft", meta=row)

    return None


def row_to_preference_sample(row: Any, config: dict[str, Any]) -> PreferenceSample | None:
    """Convert a row into a DPO prompt/chosen/rejected sample."""

    if not isinstance(row, dict):
        return None
    data_cfg = config["data"]
    prompt_field = data_cfg["prompt_field"]
    chosen_field = data_cfg["chosen_field"]
    rejected_field = data_cfg["rejected_field"]
    if all(field in row for field in (prompt_field, chosen_field, rejected_field)):
        return PreferenceSample(
            prompt=clean_text(row[prompt_field], data_cfg["normalize_nfkc"]),
            chosen=clean_text(row[chosen_field], data_cfg["normalize_nfkc"]),
            rejected=clean_text(row[rejected_field], data_cfg["normalize_nfkc"]),
            meta=row,
        )
    if data_cfg["messages_field"] in row and chosen_field in row and rejected_field in row:
        prompt, _ = render_messages(row[data_cfg["messages_field"]], config["tokenizer"]["special_tokens"])
        return PreferenceSample(
            prompt=clean_text(prompt, data_cfg["normalize_nfkc"]),
            chosen=clean_text(row[chosen_field], data_cfg["normalize_nfkc"]),
            rejected=clean_text(row[rejected_field], data_cfg["normalize_nfkc"]),
            meta=row,
        )
    return None


def load_text_samples(
    path: str | Path,
    config: dict[str, Any],
    dataset_type: str | None = None,
    source: dict[str, Any] | None = None,
) -> list[TextSample]:
    """Load and filter text/SFT samples."""

    samples = iter_text_samples(path, config, dataset_type=dataset_type, source=source)
    return list(iter_deduped_samples(samples, config, db_path=None))


def expand_source_paths(source: dict[str, Any]) -> list[Path]:
    """Expand a source path or glob into concrete files."""

    value = source.get("path") or source.get("paths")
    if value is None:
        return []
    patterns = value if isinstance(value, list) else [value]
    paths: list[Path] = []
    for pattern in patterns:
        pattern_text = str(pattern)
        if any(token in pattern_text for token in ("*", "?", "[")):
            paths.extend(Path(match) for match in glob.glob(pattern_text, recursive=True))
        else:
            paths.append(Path(pattern_text))
    return sorted(dict.fromkeys(paths))


def source_matches_stage(source: dict[str, Any], stage: str | None) -> bool:
    """Return whether a source participates in a stage."""

    purpose = str(source.get("purpose", "training")).lower()
    if purpose == "evaluation" and stage != "eval":
        return False
    if purpose != "evaluation" and stage == "eval":
        return False
    if stage == "tokenizer":
        return bool(source.get("tokenizer", True))
    stages = source.get("stages")
    if stages is None:
        return True
    if isinstance(stages, str):
        stages = [stages]
    return "all" in stages or stage in stages


def source_sample_is_selected(sample: TextSample, source: dict[str, Any], path: str | Path) -> bool:
    """Apply reproducible per-source downsampling without depending on input order."""

    rate = float(source.get("sample_rate", 1.0))
    if not 0.0 < rate <= 1.0:
        raise ValueError("A source sample_rate must be greater than zero and no larger than one.")
    if rate == 1.0:
        return True
    identity = f"{source.get('name', '<unnamed>')}\0{Path(path)}\0{split_key_for_sample(sample)}"
    return stable_hash(identity) < int(rate * (1 << 256))


def declared_languages_for_sample(source: dict[str, Any], sample: TextSample) -> tuple[str, ...]:
    """Return normalized language tags declared by a source or one of its rows."""

    values: list[Any] = []
    language_field = source.get("language_field")
    if language_field and sample.meta:
        row_value = sample.meta.get(str(language_field))
        values.extend(row_value if isinstance(row_value, (list, tuple, set)) else [row_value])

    configured = source.get("languages", [])
    values.extend(configured if isinstance(configured, (list, tuple, set)) else [configured])

    if not values and str(source.get("schema", "auto")).lower() == "translation":
        values.extend(
            [
                source.get("source_lang_field", "source"),
                source.get("target_lang_field", "target"),
            ]
        )

    normalized = [str(value).strip().lower() for value in values if value is not None and str(value).strip()]
    return tuple(dict.fromkeys(normalized))


def select_split(samples: list[TextSample], split: str, config: dict[str, Any]) -> list[TextSample]:
    """Select a stable hash split from one source."""

    if split not in {"train", "valid", "test"} or not config["data"].get("hash_split", True):
        return samples
    splits = split_samples(samples, config)
    return splits[split]


def load_text_samples_from_config(
    config: dict[str, Any],
    split: str,
    dataset_type: str | None = None,
    fallback_path: str | Path | None = None,
) -> list[TextSample]:
    """Load text samples from data.sources when present, otherwise a single fallback file."""

    samples = iter_text_samples_from_config(config, split, dataset_type=dataset_type, fallback_path=fallback_path)
    return list(iter_deduped_samples(samples, config, db_path=None))


def iter_text_samples(
    path: str | Path,
    config: dict[str, Any],
    dataset_type: str | None = None,
    source: dict[str, Any] | None = None,
) -> Iterator[TextSample]:
    """Stream filtered text/SFT samples from one source file."""

    rows = iter_rows(path, get_source_value(source, "format", config["data"]["format"]))
    data_cfg = config["data"]
    removed = {"too_short": 0, "too_long": 0, "duplicate": 0, "invalid": 0}
    max_samples = get_source_value(source, "max_samples", data_cfg.get("max_samples_per_source"))
    max_samples = int(max_samples) if max_samples is not None else None
    emitted = 0

    for row in rows:
        sample = row_to_text_sample(row, config, dataset_type, source)
        if sample is None or not sample.text:
            removed["invalid"] += 1
            continue
        size = len(sample.text)
        if size < int(data_cfg["min_chars"]):
            removed["too_short"] += 1
            continue
        if size > int(data_cfg["max_chars"]):
            removed["too_long"] += 1
            continue
        sample.meta = {**(sample.meta or {}), "_removed_counts": dict(removed)}
        yield sample
        emitted += 1
        if max_samples is not None and emitted >= max_samples:
            break


def split_key_for_sample(sample: TextSample) -> str:
    """Return the leakage-safe identity used for deterministic splitting."""

    if sample.meta and sample.meta.get("_split_key") is not None:
        return str(sample.meta["_split_key"])
    return sample.text


def split_name_for_key(key: str, config: dict[str, Any]) -> str:
    """Return train/valid/test for one stable identity key."""

    ratios = config["data"]
    train_ratio = float(ratios["train_ratio"])
    valid_ratio = float(ratios["valid_ratio"])
    total = train_ratio + valid_ratio + float(ratios["test_ratio"])
    train_cut = train_ratio / total
    valid_cut = (train_ratio + valid_ratio) / total
    value = (stable_hash(key) % 10_000_000) / 10_000_000
    if value < train_cut:
        return "train"
    if value < valid_cut:
        return "valid"
    return "test"


def split_name_for_sample(sample: TextSample, config: dict[str, Any]) -> str:
    """Return train/valid/test, or discard for a conflicting parallel edge."""

    if sample.meta:
        if sample.meta.get("_split_conflict"):
            return "discard"
        forced = sample.meta.get("_forced_split")
        if forced in {"train", "valid", "test"}:
            return str(forced)
    return split_name_for_key(split_key_for_sample(sample), config)


def iter_text_samples_from_config(
    config: dict[str, Any],
    split: str,
    dataset_type: str | None = None,
    fallback_path: str | Path | None = None,
) -> Iterator[TextSample]:
    """Stream text samples from data.sources or one fallback file."""

    sources = config["data"].get("sources") or []
    denylist = load_benchmark_denylist(config)
    if not sources:
        path = fallback_path or config["data"]["train_file"]
        for sample in iter_text_samples(path, config, dataset_type=dataset_type):
            if sample_rejection_categories(sample, config, denylist):
                continue
            yield sample
        return

    for source in sources:
        if not isinstance(source, dict) or not source_matches_stage(source, dataset_type):
            continue
        source_split = str(source.get("split", "all")).lower()
        if source_split not in {"all", "auto", split}:
            continue
        paths = expand_source_paths(source)
        if not paths:
            if config["data"].get("strict_sources", True):
                raise FileNotFoundError(
                    f"Data source '{source.get('name', '<unnamed>')}' matched no files: "
                    f"{source.get('path', source.get('paths'))}"
                )
            continue
        if dataset_type == "tokenizer" and source.get("tokenizer_max_samples") is not None:
            source_limit = source["tokenizer_max_samples"]
        else:
            source_limit = get_source_value(source, "max_samples", config["data"].get("max_samples_per_source"))
        source_limit = int(source_limit) if source_limit is not None else None
        source_seen = 0
        stop_source = False
        for path in paths:
            # Enforce max_samples across the logical source, not once per file
            # in a glob.  The old behavior silently multiplied caps by the
            # number of shards and could turn a 10k cap into millions of rows.
            path_source = {**source, "path": str(path), "max_samples": None}
            for sample in iter_text_samples(path, config, dataset_type=dataset_type, source=path_source):
                if not source_sample_is_selected(sample, source, path):
                    continue
                if source_limit is not None and source_seen >= source_limit:
                    stop_source = True
                    break
                source_seen += 1
                languages = declared_languages_for_sample(source, sample)
                sample.meta = {
                    **(sample.meta or {}),
                    "_source_name": str(source.get("name", "<unnamed>")),
                    "_source_path": str(path),
                    "_languages": list(languages),
                }
                if sample_rejection_categories(sample, config, denylist):
                    continue
                if (
                    source_split in {"all", "auto"}
                    and config["data"].get("hash_split", True)
                    and split_name_for_sample(sample, config) != split
                ):
                    continue
                yield sample
            if stop_source:
                break


def load_preference_samples(
    path: str | Path, config: dict[str, Any], *, purpose: str = "training"
) -> list[PreferenceSample]:
    """Load DPO rows, applying the central safety boundary to training data."""

    rows = read_rows(path, config["data"]["format"])
    samples = [row_to_preference_sample(row, config) for row in rows]
    valid = [sample for sample in samples if sample is not None]
    if purpose not in {"training", "evaluation"}:
        raise ValueError("Preference data purpose must be training or evaluation.")
    if purpose == "evaluation":
        return valid
    denylist = load_benchmark_denylist(config)
    for index, sample in enumerate(valid):
        categories = preference_rejection_categories(sample, config, denylist)
        if categories:
            raise DataPolicyError(
                f"DPO training row {index + 1} was rejected by the shared data boundary: {', '.join(categories)}"
            )
    return valid


def split_samples(samples: list[TextSample], config: dict[str, Any]) -> dict[str, list[TextSample]]:
    """Split samples into train/valid/test, optionally by stable hash."""

    splits = {"train": [], "valid": [], "test": []}
    for index, sample in enumerate(samples):
        if config["data"].get("hash_split", True):
            split = split_name_for_sample(sample, config)
            if split in splits:
                splits[split].append(sample)
        else:
            ratios = config["data"]
            train_ratio = float(ratios["train_ratio"])
            valid_ratio = float(ratios["valid_ratio"])
            total = train_ratio + valid_ratio + float(ratios["test_ratio"])
            value = index / max(1, len(samples))
            if value < train_ratio / total:
                splits["train"].append(sample)
            elif value < (train_ratio + valid_ratio) / total:
                splits["valid"].append(sample)
            else:
                splits["test"].append(sample)
    return splits


def _script_bucket(character: str) -> str | None:
    """Classify one visible character by writing system, not by language."""

    code = ord(character)
    if 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F or 0xA960 <= code <= 0xA97F or 0xAC00 <= code <= 0xD7AF:
        return "hangul"
    if 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF or 0xFF66 <= code <= 0xFF9D:
        return "kana"
    if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
        return "han"
    if "a" <= character.lower() <= "z":
        return "latin"
    return None if character.isspace() else "other"


def script_ratios(texts: Iterable[str]) -> dict[str, float]:
    """Estimate writing-system ratios without pretending that scripts are languages."""

    counts = {"hangul": 0, "kana": 0, "han": 0, "latin": 0, "other": 0}
    for text in texts:
        for character in text:
            bucket = _script_bucket(character)
            if bucket is not None:
                counts[bucket] += 1
    total = max(1, sum(counts.values()))
    return {key: round(value / total, 4) for key, value in counts.items()}


def analyze_samples(samples: list[TextSample], tokenizer: Any | None = None) -> dict[str, Any]:
    """Produce dataset statistics, including optional token counts."""

    return analyze_sample_stream(samples, tokenizer)


def analyze_sample_stream(samples: Iterable[TextSample], tokenizer: Any | None = None) -> dict[str, Any]:
    """Analyze samples with bounded memory, even for multi-million-row corpora."""

    length_reservoir: list[int] = []
    token_length_reservoir: list[int] = []
    script_counts = {"hangul": 0, "kana": 0, "han": 0, "latin": 0, "other": 0}
    language_stats: dict[str, dict[str, int]] = {}
    multiturn_count = 0
    sample_count = 0
    total_chars = 0
    total_tokens = 0
    max_chars = 0
    max_tokens = 0
    rng = random.Random(0)

    def add_to_reservoir(values: list[int], value: int, seen: int) -> None:
        if len(values) < STATS_RESERVOIR_SIZE:
            values.append(value)
            return
        replacement = rng.randrange(seen)
        if replacement < STATS_RESERVOIR_SIZE:
            values[replacement] = value

    for sample in samples:
        sample_count += 1
        char_length = len(sample.text)
        total_chars += char_length
        max_chars = max(max_chars, char_length)
        add_to_reservoir(length_reservoir, char_length, sample_count)
        if tokenizer is not None:
            token_length = len(tokenizer.encode(sample.text, add_special_tokens=False))
            total_tokens += token_length
            max_tokens = max(max_tokens, token_length)
            add_to_reservoir(token_length_reservoir, token_length, sample_count)
        else:
            token_length = 0
        for language in (sample.meta or {}).get("_languages", []):
            tag = str(language).strip().lower()
            if not tag:
                continue
            stats = language_stats.setdefault(
                tag,
                {"sample_count": 0, "total_chars": 0, "total_utf8_bytes": 0, "total_tokens": 0},
            )
            stats["sample_count"] += 1
            stats["total_chars"] += char_length
            stats["total_utf8_bytes"] += len(sample.text.encode("utf-8"))
            stats["total_tokens"] += token_length
        if sample.meta and isinstance(sample.meta.get("messages"), list) and len(sample.meta["messages"]) >= 4:
            multiturn_count += 1
        for character in sample.text:
            bucket = _script_bucket(character)
            if bucket is not None:
                script_counts[bucket] += 1
    total_script = max(1, sum(script_counts.values()))
    declared_language_stats = {}
    for language, values in sorted(language_stats.items()):
        language_tokens = values["total_tokens"]
        declared_language_stats[language] = {
            **values,
            "sample_ratio": round(values["sample_count"] / max(1, sample_count), 4),
            "chars_per_token": (
                round(values["total_chars"] / language_tokens, 4) if tokenizer is not None and language_tokens else None
            ),
            "utf8_bytes_per_token": (
                round(values["total_utf8_bytes"] / language_tokens, 4)
                if tokenizer is not None and language_tokens
                else None
            ),
        }
    return {
        "sample_count": sample_count,
        "total_chars": total_chars,
        "avg_chars": round(total_chars / sample_count, 2) if sample_count else 0,
        "median_chars": statistics.median(length_reservoir) if length_reservoir else 0,
        "max_chars": max_chars,
        "total_tokens": total_tokens if tokenizer is not None else None,
        "avg_tokens": round(total_tokens / sample_count, 2) if tokenizer is not None and sample_count else None,
        "median_tokens": statistics.median(token_length_reservoir) if token_length_reservoir else None,
        "max_tokens": max_tokens if tokenizer is not None else None,
        "median_is_approximate": sample_count > STATS_RESERVOIR_SIZE,
        "median_sample_size": min(sample_count, STATS_RESERVOIR_SIZE),
        "script_ratio": {key: round(value / total_script, 4) for key, value in script_counts.items()},
        "declared_language_stats": declared_language_stats,
        "multiturn_2plus_turn_ratio": round(multiturn_count / max(1, sample_count), 4),
        "recommended_vocab_sizes": recommend_vocab_sizes_from_chars(total_chars),
    }


def recommend_vocab_sizes(samples: list[TextSample]) -> list[int]:
    """Recommend practical SentencePiece vocab sizes from corpus size and scripts."""

    char_count = sum(len(sample.text) for sample in samples)
    if char_count < 1_000_000:
        return [8000, 12000, 16000]
    if char_count < 50_000_000:
        return [16000, 24000, 32000]
    return [32000, 48000, 64000]


def recommend_vocab_sizes_from_chars(char_count: int) -> list[int]:
    """Recommend vocab sizes from total character count."""

    if char_count < 1_000_000:
        return [8000, 12000, 16000]
    if char_count < 50_000_000:
        return [16000, 24000, 32000]
    return [32000, 48000, 64000]


def save_json(path: str | Path, data: Any) -> None:
    """Write standards-compliant UTF-8 JSON atomically."""

    atomic_write_json(path, data)


def chunk_tokens(tokens: list[int], max_seq_len: int, eos_id: int) -> list[list[int]]:
    """Split long documents into max_seq_len chunks ending with EOS when possible."""

    chunks = []
    for start in range(0, len(tokens), max_seq_len):
        chunk = tokens[start : start + max_seq_len]
        if chunk and chunk[-1] != eos_id:
            chunk = [*chunk[: max_seq_len - 1], eos_id]
        if len(chunk) > 1:
            chunks.append(chunk)
    return chunks


def pack_sequences(sequences: list[list[int]], max_seq_len: int, eos_id: int) -> list[list[int]]:
    """Pack short sequences to reduce padding waste."""

    packed: list[list[int]] = []
    current: list[int] = []
    for seq in sequences:
        if len(seq) > max_seq_len:
            packed.extend(chunk_tokens(seq, max_seq_len, eos_id))
            continue
        if len(current) + len(seq) <= max_seq_len:
            current.extend(seq)
        else:
            if current:
                packed.append(current)
            current = list(seq)
    if current:
        packed.append(current)
    return packed


def tokenize_samples(samples: list[TextSample], tokenizer: Any, config: dict[str, Any]) -> list[list[int]]:
    """Tokenize, chunk, and optionally pack samples."""

    max_seq_len = int(config["model"]["max_seq_len"])
    eos_id = tokenizer.eos_id
    sequences: list[list[int]] = []
    for sample in samples:
        ids = tokenizer.encode(sample.text, add_special_tokens=True)
        if len(ids) > max_seq_len:
            sequences.extend(chunk_tokens(ids, max_seq_len, eos_id))
        else:
            sequences.append(ids)
    if config["data"].get("sequence_packing", True):
        return pack_sequences(sequences, max_seq_len, eos_id)
    return sequences


def iter_tokenized_training_samples(
    samples: Iterable[TextSample],
    tokenizer: Any,
    config: dict[str, Any],
    assistant_only_loss: bool = False,
) -> Iterator[dict[str, list[int]]]:
    """Stream tokenized training rows with chunking and optional sequence packing."""

    max_seq_len = int(config["model"]["max_seq_len"])
    eos_id = tokenizer.eos_id
    bos_id = tokenizer.bos_id
    specials = config["tokenizer"]["special_tokens"]

    def has_loss_labels(labels: list[int]) -> bool:
        return any(label != -100 for label in labels[1:])

    def iter_chunks(ids: list[int], labels: list[int]) -> Iterator[dict[str, list[int]]]:
        for start in range(0, len(ids), max_seq_len):
            chunk_ids = ids[start : start + max_seq_len]
            chunk_labels = labels[start : start + max_seq_len]
            if len(chunk_ids) > 1 and has_loss_labels(chunk_labels):
                yield {"input_ids": chunk_ids, "labels": chunk_labels, "seq_lens": [len(chunk_ids)]}

    pending_ids: list[int] = []
    pending_labels: list[int] = []
    pending_seq_lens: list[int] = []
    for sample in samples:
        if assistant_only_loss and sample.meta and isinstance(sample.meta.get(config["data"]["messages_field"]), list):
            ids = [bos_id]
            labels = [-100]
            messages = sample.meta[config["data"]["messages_field"]]
            for message in messages:
                role = str(message.get("role", "user")).lower()
                role_token = specials.get(role, f"<{role}>")
                content = clean_text(message.get("content", ""), config["data"]["normalize_nfkc"])
                content = escape_special_tokens(content, specials)
                segment_text = f"{role_token}\n{content}\n"
                segment_ids = tokenizer.encode(segment_text, add_special_tokens=False)
                ids.extend(segment_ids)
                labels.extend(segment_ids if role == "assistant" else [-100] * len(segment_ids))
            ids.append(eos_id)
            labels.append(eos_id if messages and messages[-1].get("role") == "assistant" else -100)
        else:
            ids = tokenizer.encode(sample.text, add_special_tokens=True)
            labels = list(ids)

        if config["data"].get("sequence_packing", True) and len(ids) <= max_seq_len:
            if len(pending_ids) + len(ids) <= max_seq_len:
                pending_ids.extend(ids)
                pending_labels.extend(labels)
                pending_seq_lens.append(len(ids))
                continue
            if pending_ids and has_loss_labels(pending_labels):
                yield {"input_ids": pending_ids, "labels": pending_labels, "seq_lens": pending_seq_lens}
            pending_ids, pending_labels, pending_seq_lens = list(ids), list(labels), [len(ids)]
        else:
            yield from iter_chunks(ids, labels)

    if pending_ids and has_loss_labels(pending_labels):
        yield {"input_ids": pending_ids, "labels": pending_labels, "seq_lens": pending_seq_lens}


def tokenize_training_samples(
    samples: list[TextSample],
    tokenizer: Any,
    config: dict[str, Any],
    assistant_only_loss: bool = False,
) -> list[dict[str, list[int]]]:
    """Tokenize samples into input_ids and labels.

    For SFT chat rows, assistant_only_loss masks user/system/control text with
    -100 so PyTorch cross entropy ignores those positions.  This teaches the
    model to answer as the assistant without penalizing it for not predicting
    the user's prompt.
    """

    max_seq_len = int(config["model"]["max_seq_len"])
    cache_dir_value = config["data"].get("token_cache_dir")
    cache_path: Path | None = None
    if cache_dir_value:
        cache_dir = Path(cache_dir_value)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_path = Path(str(getattr(tokenizer, "model_path", "")))
        tokenizer_stat = tokenizer_path.stat() if tokenizer_path.exists() else None
        cache_settings = {
            "tokenization_version": 3,
            "assistant_only_loss": assistant_only_loss,
            "max_seq_len": max_seq_len,
            "sequence_packing": config["data"].get("sequence_packing", True),
            "tokenizer_model": str(tokenizer_path),
            "tokenizer_model_size": tokenizer_stat.st_size if tokenizer_stat else None,
            "tokenizer_model_mtime_ns": tokenizer_stat.st_mtime_ns if tokenizer_stat else None,
            "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
            "special_tokens": config["tokenizer"]["special_tokens"],
        }
        hasher = hashlib.sha256(json.dumps(cache_settings, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        for sample in samples:
            hasher.update(b"\0kind:")
            hasher.update(sample.kind.encode("utf-8"))
            hasher.update(b"\0text:")
            hasher.update(sample.text.encode("utf-8"))
        digest = hasher.hexdigest()
        cache_path = cache_dir / f"tokens_{digest}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as handle:
                return pickle.load(handle)

    try:
        from tqdm import tqdm

        iterable = tqdm(samples, desc="tokenizing", unit="sample")
    except Exception:
        iterable = samples
    output = list(iter_tokenized_training_samples(iterable, tokenizer, config, assistant_only_loss))
    if cache_path is not None:
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
    return output


def collate_token_batch(batch: list[dict[str, Any]], pad_id: int):
    """Pad one token batch and add packing-aware position/document ids.

    When a row packs several documents together, ``position_ids`` restart at 0
    for each document and ``document_ids`` label the document each token belongs
    to.  The model uses these to reset RoPE positions per document and to build
    a block-diagonal attention mask so packed documents never attend to one
    another.  For the common case of a single document per row and no padding,
    all auxiliary tensors are omitted so the fast ``is_causal`` attention path
    stays active.
    """

    import torch

    lengths = [int(item["input_ids"].shape[0]) for item in batch]
    max_len = max(lengths)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    needs_padding = any(length != max_len for length in lengths)

    def seq_lens_for(item: dict[str, Any], length: int) -> list[int]:
        raw = item.get("seq_lens")
        if raw is None:
            return [length]
        return [int(value) for value in (raw.tolist() if hasattr(raw, "tolist") else raw)]

    needs_documents = any(len(seq_lens_for(item, lengths[row])) > 1 for row, item in enumerate(batch))

    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool) if needs_padding else None
    position_ids = None
    document_ids = None
    if needs_padding or needs_documents:
        position_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
        document_ids = torch.full((len(batch), max_len), -1, dtype=torch.long)

    for row, item in enumerate(batch):
        length = lengths[row]
        input_ids[row, :length] = item["input_ids"]
        labels[row, :length] = item["labels"]
        if attention_mask is not None:
            attention_mask[row, :length] = True
        if position_ids is not None:
            cursor = 0
            for doc_index, doc_len in enumerate(seq_lens_for(item, length)):
                if doc_len <= 0:
                    continue
                end = min(cursor + doc_len, length)
                position_ids[row, cursor:end] = torch.arange(end - cursor, dtype=torch.long)
                document_ids[row, cursor:end] = doc_index
                cursor = end
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "document_ids": document_ids,
    }


class TokenCollator:
    """Pickle-friendly token batch collator."""

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, Any]]):
        return collate_token_batch(batch, self.pad_id)


def make_torch_dataset(sequences: list[list[int]] | list[dict[str, list[int]]], pad_id: int):
    """Create a small PyTorch Dataset lazily so analyze/tokenizer modes do not need torch."""

    import torch
    from torch.utils.data import Dataset

    class TokenDataset(Dataset):
        def __len__(self) -> int:
            return len(sequences)

        def __getitem__(self, index: int) -> dict[str, Any]:
            row = sequences[index]
            if isinstance(row, dict):
                ids = torch.tensor(row["input_ids"], dtype=torch.long)
                labels = torch.tensor(row["labels"], dtype=torch.long)
                seq_lens = list(row.get("seq_lens") or [int(ids.shape[0])])
            else:
                ids = torch.tensor(row, dtype=torch.long)
                labels = ids.clone()
                seq_lens = [int(ids.shape[0])]
            return {"input_ids": ids, "labels": labels, "seq_lens": seq_lens}

    return TokenDataset(), TokenCollator(pad_id)


class TokenShardDataset:
    """Map-style Dataset backed by flat memory-mapped token arrays.

    Tokens and labels for the whole corpus live in two contiguous binary files;
    per-sequence offsets and packed-document boundaries live in small index
    arrays.  ``__getitem__`` slices the memmaps directly, so random access from
    a shuffling DataLoader is O(sequence length) instead of reloading a whole
    shard from disk on every index.  Memmaps are opened lazily so each
    DataLoader worker process maps the files independently.
    """

    def __init__(self, metadata_path: str | Path) -> None:
        self.metadata_path = Path(metadata_path)
        self.root = self.metadata_path.parent
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.total_sequences = int(self.metadata["total_sequences"])
        self.offsets = np.load(self.root / "offsets.npy")
        self.doc_offsets = np.load(self.root / "doc_offsets.npy")
        self._tokens: np.memmap | None = None
        self._labels: np.memmap | None = None
        self._doc_lens: np.memmap | None = None

    def _ensure_open(self) -> None:
        if self._tokens is None:
            self._tokens = np.memmap(self.root / "tokens.bin", dtype=np.int32, mode="r")
            self._labels = np.memmap(self.root / "labels.bin", dtype=np.int32, mode="r")
            self._doc_lens = np.memmap(self.root / "doc_lens.bin", dtype=np.int32, mode="r")

    def __len__(self) -> int:
        return self.total_sequences

    def __getitem__(self, index: int):
        import torch

        if index < 0 or index >= self.total_sequences:
            raise IndexError(index)
        self._ensure_open()
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        doc_start, doc_end = int(self.doc_offsets[index]), int(self.doc_offsets[index + 1])
        ids = torch.from_numpy(self._tokens[start:end].astype(np.int64))
        labels = torch.from_numpy(self._labels[start:end].astype(np.int64))
        seq_lens = [int(value) for value in self._doc_lens[doc_start:doc_end]]
        return {"input_ids": ids, "labels": labels, "seq_lens": seq_lens}


def token_shard_cache_key(
    config: dict[str, Any],
    tokenizer: Any,
    split: str,
    dataset_type: str,
    assistant_only_loss: bool,
) -> str:
    """Build a cache key from source files, tokenizer, and tokenization settings."""

    tokenizer_path = Path(str(getattr(tokenizer, "model_path", "")))
    tokenizer_stat = tokenizer_path.stat() if tokenizer_path.exists() else None
    sources_payload = []
    sources = config["data"].get("sources") or [{"path": config["data"]["train_file"], "schema": "auto"}]
    for source in sources:
        if not isinstance(source, dict) or not source_matches_stage(source, dataset_type):
            continue
        expanded = []
        for path in expand_source_paths(source):
            if not path.exists():
                continue
            stat = path.stat()
            expanded.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        payload = dict(source)
        payload["expanded"] = expanded
        sources_payload.append(payload)
    policy_cfg = config.get("data_policy", {})

    def recorded_digest(path_value: Any, field: str) -> str | None:
        if not path_value:
            return None
        try:
            artifact = json.loads(Path(path_value).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        value = artifact.get(field)
        return str(value) if value else None

    payload = {
        "version": 6,
        "policy_version": POLICY_VERSION,
        "filter_version": FILTER_VERSION,
        "source_lock_digest": recorded_digest(policy_cfg.get("source_lock_path"), "lock_digest"),
        "audit_digest": recorded_digest(policy_cfg.get("audit_path"), "audit_digest"),
        "split": split,
        "dataset_type": dataset_type,
        "assistant_only_loss": assistant_only_loss,
        "max_seq_len": int(config["model"]["max_seq_len"]),
        "sequence_packing": bool(config["data"].get("sequence_packing", True)),
        "train_ratio": config["data"].get("train_ratio"),
        "valid_ratio": config["data"].get("valid_ratio"),
        "test_ratio": config["data"].get("test_ratio"),
        "normalize_nfkc": config["data"].get("normalize_nfkc"),
        "min_chars": config["data"].get("min_chars"),
        "max_chars": config["data"].get("max_chars"),
        "max_samples_per_source": config["data"].get("max_samples_per_source"),
        "dedup": config["data"].get("dedup"),
        "dedup_level": config["data"].get("dedup_level"),
        "dedup_backend": config["data"].get("dedup_backend"),
        "truncation_policy": config["data"].get("truncation_policy"),
        "tokenizer_model": str(tokenizer_path),
        "tokenizer_model_size": tokenizer_stat.st_size if tokenizer_stat else None,
        "tokenizer_model_mtime_ns": tokenizer_stat.st_mtime_ns if tokenizer_stat else None,
        "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
        "special_tokens": config["tokenizer"]["special_tokens"],
        "sources": sources_payload,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def source_manifest_fingerprint(config: dict[str, Any], stage: str) -> str:
    """Return an opaque hash of sources/settings for artifact freshness.

    The source manifest itself is never stored in the artifact.  The digest is
    safe to retain even when checkpoint configuration redaction is enabled.
    """

    sources_payload = []
    sources = config["data"].get("sources") or [{"path": config["data"]["train_file"], "schema": "auto"}]
    for source in sources:
        if not isinstance(source, dict) or not source_matches_stage(source, stage):
            continue
        expanded = []
        for path in expand_source_paths(source):
            if not path.exists():
                continue
            stat = path.stat()
            expanded.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        payload = dict(source)
        payload["expanded"] = expanded
        sources_payload.append(payload)
    payload = {
        "stage": stage,
        "policy_version": POLICY_VERSION,
        "filter_version": FILTER_VERSION,
        "format": config["data"].get("format"),
        "text_field": config["data"].get("text_field"),
        "messages_field": config["data"].get("messages_field"),
        "normalize_nfkc": config["data"].get("normalize_nfkc"),
        "min_chars": config["data"].get("min_chars"),
        "max_chars": config["data"].get("max_chars"),
        "max_samples_per_source": config["data"].get("max_samples_per_source"),
        "dedup": config["data"].get("dedup"),
        "hash_split": config["data"].get("hash_split"),
        "train_ratio": config["data"].get("train_ratio"),
        "valid_ratio": config["data"].get("valid_ratio"),
        "test_ratio": config["data"].get("test_ratio"),
        "data_policy": config.get("data_policy"),
        "sources": sources_payload,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def iter_deduped_samples(
    samples: Iterable[TextSample],
    config: dict[str, Any],
    db_path: Path | None,
) -> Iterator[TextSample]:
    """Optionally deduplicate samples with an on-disk SQLite hash set."""

    if not config["data"].get("dedup", True):
        yield from samples
        return
    if db_path is None:
        seen: set[str] = set()
        for sample in samples:
            digest = hashlib.sha1(sample.text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            yield sample
        return

    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute("create table if not exists seen (hash text primary key)")
    inserted = 0
    try:
        for sample in samples:
            digest = hashlib.sha1(sample.text.encode("utf-8")).hexdigest()
            try:
                conn.execute("insert into seen(hash) values (?)", (digest,))
                inserted += 1
                if inserted % 100_000 == 0:
                    conn.commit()
                yield sample
            except sqlite3.IntegrityError:
                continue
    finally:
        conn.commit()
        conn.close()


def _parallel_tokenization_worker_func(args):
    """Worker process task for tokenizing a chunk of samples."""
    (
        samples_chunk_meta,
        tokenizer_path,
        special_tokens,
        messages_field,
        normalize_nfkc,
        assistant_only_loss,
        _max_seq_len,
        bos_id,
        eos_id,
    ) = args
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

    results = []
    for text, meta in samples_chunk_meta:
        if assistant_only_loss and meta and isinstance(meta.get(messages_field), list):
            ids = [bos_id]
            labels = [-100]
            messages = meta[messages_field]
            for message in messages:
                role = str(message.get("role", "user")).lower()
                role_token = special_tokens.get(role, f"<{role}>")
                content = escape_special_tokens(clean_text(message.get("content", ""), normalize_nfkc), special_tokens)
                segment_text = f"{role_token}\n{content}\n"
                segment_ids = list(sp.encode(segment_text, out_type=int))
                ids.extend(segment_ids)
                labels.extend(segment_ids if role == "assistant" else [-100] * len(segment_ids))
            ids.append(eos_id)
            labels.append(eos_id if messages and messages[-1].get("role") == "assistant" else -100)
        else:
            ids = [bos_id, *list(sp.encode(text, out_type=int)), eos_id]
            labels = list(ids)
        results.append((ids, labels))
    return results


def build_token_shard_dataset(
    config: dict[str, Any],
    tokenizer: Any,
    split: str,
    dataset_type: str,
    assistant_only_loss: bool,
    logger: Any | None = None,
):
    """Create or load a bounded-memory, memory-mapped token dataset.

    Samples and tokenized batches are consumed incrementally.  At no point do
    we retain the full source corpus or all tokenized rows in RAM; this is
    essential for the multi-gigabyte corpora this path is intended to support.
    """

    import multiprocessing

    cache_root_value = config["data"].get("token_cache_dir") or (Path(config["data"]["processed_dir"]) / "token_cache")
    cache_root = Path(cache_root_value) / "shards"
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = token_shard_cache_key(config, tokenizer, split, dataset_type, assistant_only_loss)
    shard_dir = cache_root / f"{dataset_type}_{split}_{digest}"
    metadata_path = shard_dir / "metadata.json"
    if metadata_path.exists():
        wrapper = TokenShardDataset(metadata_path)
        return wrapper, TokenCollator(tokenizer.pad_id), wrapper.total_sequences

    tmp_dir = shard_dir.with_name(shard_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    dedup_db = (
        tmp_dir / "dedup.sqlite" if str(config["data"].get("dedup_backend", "sqlite")).lower() == "sqlite" else None
    )

    if logger:
        logger.info("Streaming, deduplicating, tokenizing, and packing samples...")
    loaded_iter = iter_text_samples_from_config(
        config,
        split=split,
        dataset_type=dataset_type,
        fallback_path=config["data"].get(f"{split}_file") or config["data"]["train_file"],
    )
    samples = iter_deduped_samples(loaded_iter, config, dedup_db)

    # Configuration values needed by workers.
    tokenizer_path = str(getattr(tokenizer, "model_path", ""))
    special_tokens = config["tokenizer"]["special_tokens"]
    messages_field = config["data"]["messages_field"]
    normalize_nfkc = config["data"]["normalize_nfkc"]
    max_seq_len = int(config["model"]["max_seq_len"])
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id

    num_workers = min(multiprocessing.cpu_count(), int(config["hardware"].get("num_workers", 0)))
    batch_size = int(config["data"].get("token_cache_shard_size", 4096))
    source_samples: dict[str, int] = {}
    if logger:
        mode = f"{num_workers} worker processes" if num_workers > 1 else "the main process"
        logger.info(f"Tokenizing with {mode}; bounded batch size={batch_size:,} samples.")

    def worker_batches() -> Iterator[tuple[Any, ...]]:
        batch: list[tuple[str, dict[str, Any] | None]] = []
        for sample in samples:
            source_name = str((sample.meta or {}).get("_source_name", "<fallback>"))
            source_samples[source_name] = source_samples.get(source_name, 0) + 1
            # Pretraining does not need the original row after rendering.  For
            # SFT keep only normalized messages instead of duplicating every
            # source column through multiprocessing queues.
            worker_meta = None
            if assistant_only_loss and sample.meta and isinstance(sample.meta.get(messages_field), list):
                worker_meta = {messages_field: sample.meta[messages_field]}
            batch.append((sample.text, worker_meta))
            if len(batch) < batch_size:
                continue
            yield (
                batch,
                tokenizer_path,
                special_tokens,
                messages_field,
                normalize_nfkc,
                assistant_only_loss,
                max_seq_len,
                bos_id,
                eos_id,
            )
            batch = []
        if batch:
            yield (
                batch,
                tokenizer_path,
                special_tokens,
                messages_field,
                normalize_nfkc,
                assistant_only_loss,
                max_seq_len,
                bos_id,
                eos_id,
            )

    # Flat binary token store: tokens/labels are appended contiguously while
    # small index arrays record per-sequence and per-document boundaries.  This
    # gives the DataLoader O(1) random access through np.memmap instead of
    # reloading whole shards on every shuffled index.
    offsets: list[int] = [0]
    doc_offsets: list[int] = [0]
    total_samples = 0
    token_total = 0
    doc_total = 0
    total_sequences = 0

    tokens_file = None
    labels_file = None
    doc_lens_file = None

    def emit(ids: list[int], labels: list[int], seq_lens: list[int]) -> None:
        nonlocal token_total, doc_total, total_sequences
        if tokens_file is None or labels_file is None or doc_lens_file is None:
            raise RuntimeError("Token cache output streams are not open.")
        np.asarray(ids, dtype=np.int32).tofile(tokens_file)
        np.asarray(labels, dtype=np.int32).tofile(labels_file)
        np.asarray(seq_lens, dtype=np.int32).tofile(doc_lens_file)
        token_total += len(ids)
        doc_total += len(seq_lens)
        offsets.append(token_total)
        doc_offsets.append(doc_total)
        total_sequences += 1
        if (
            logger
            and total_sequences % max(1, int(config["data"].get("token_cache_log_interval_shards", 25)) * 1000) == 0
        ):
            logger.info(f"Wrote {total_sequences} packed sequences.")

    sequence_packing = config["data"].get("sequence_packing", True)

    def has_loss_labels(labels: list[int]) -> bool:
        return any(label != -100 for label in labels[1:])

    pending_ids: list[int] = []
    pending_labels: list[int] = []
    pending_seq_lens: list[int] = []

    def consume_result_chunk(result_chunk: list[tuple[list[int], list[int]]]) -> None:
        nonlocal pending_ids, pending_labels, pending_seq_lens, total_samples
        total_samples += len(result_chunk)
        for ids, labels in result_chunk:
            if sequence_packing and len(ids) <= max_seq_len:
                if len(pending_ids) + len(ids) <= max_seq_len:
                    pending_ids.extend(ids)
                    pending_labels.extend(labels)
                    pending_seq_lens.append(len(ids))
                    continue
                if pending_ids and has_loss_labels(pending_labels):
                    emit(pending_ids, pending_labels, pending_seq_lens)
                pending_ids, pending_labels, pending_seq_lens = list(ids), list(labels), [len(ids)]
            else:
                for start in range(0, len(ids), max_seq_len):
                    chunk_ids = ids[start : start + max_seq_len]
                    chunk_labels = labels[start : start + max_seq_len]
                    if len(chunk_ids) > 1 and has_loss_labels(chunk_labels):
                        emit(chunk_ids, chunk_labels, [len(chunk_ids)])

    progress = None
    try:
        try:
            from tqdm import tqdm

            progress = tqdm(desc=f"tokenizing {dataset_type}/{split}", unit="sample", disable=(logger is None))
        except ImportError:
            pass

        with (
            (tmp_dir / "tokens.bin").open("wb") as opened_tokens,
            (tmp_dir / "labels.bin").open("wb") as opened_labels,
            (tmp_dir / "doc_lens.bin").open("wb") as opened_doc_lens,
        ):
            tokens_file = opened_tokens
            labels_file = opened_labels
            doc_lens_file = opened_doc_lens
            batches = worker_batches()
            if num_workers > 1:
                with multiprocessing.Pool(processes=num_workers) as pool:
                    results = pool.imap(_parallel_tokenization_worker_func, batches, chunksize=1)
                    for result_chunk in results:
                        consume_result_chunk(result_chunk)
                        if progress is not None:
                            progress.update(len(result_chunk))
            else:
                for batch_args in batches:
                    result_chunk = _parallel_tokenization_worker_func(batch_args)
                    consume_result_chunk(result_chunk)
                    if progress is not None:
                        progress.update(len(result_chunk))

            if pending_ids and has_loss_labels(pending_labels):
                emit(pending_ids, pending_labels, pending_seq_lens)
            tokens_file = labels_file = doc_lens_file = None

        if total_sequences <= 0:
            raise RuntimeError(f"No tokenized {dataset_type}/{split} sequences were produced from configured sources.")
        if (
            split == "train"
            and dataset_type == "pretrain"
            and bool(config["data"].get("require_all_training_sources", False))
        ):
            expected_sources = {
                str(source.get("name", "<unnamed>"))
                for source in config["data"].get("sources") or []
                if source_matches_stage(source, dataset_type)
                and str(source.get("split", "all")).lower() in {"all", "auto", "train"}
            }
            missing_sources = sorted(expected_sources - source_samples.keys())
            if missing_sources:
                raise RuntimeError(
                    "Configured pretraining sources produced no accepted training samples: "
                    + ", ".join(missing_sources)
                )

        np.save(tmp_dir / "offsets.npy", np.asarray(offsets, dtype=np.int64))
        np.save(tmp_dir / "doc_offsets.npy", np.asarray(doc_offsets, dtype=np.int64))
        atomic_write_json(
            tmp_dir / "metadata.json",
            {
                "dataset_type": dataset_type,
                "split": split,
                "assistant_only_loss": assistant_only_loss,
                "total_samples": total_samples,
                "total_sequences": total_sequences,
                "total_tokens": token_total,
                "source_samples": dict(sorted(source_samples.items())),
                "format": "memmap_int32_v1",
            },
        )
        if dedup_db and dedup_db.exists():
            dedup_db.unlink()
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        tmp_dir.rename(shard_dir)
    except BaseException:
        tokens_file = labels_file = doc_lens_file = None
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        if progress is not None:
            progress.close()
        close_samples = getattr(samples, "close", None)
        if close_samples is not None:
            close_samples()

    if logger:
        logger.info(
            f"Built token memmap cache {shard_dir} from {total_samples:,} samples: "
            f"{total_sequences:,} sequences / {token_total:,} tokens."
        )

    wrapper = TokenShardDataset(shard_dir / "metadata.json")
    return wrapper, TokenCollator(tokenizer.pad_id), wrapper.total_sequences


def estimate_token_count(samples: list[TextSample], tokenizer: Any) -> int:
    """Count tokens with a progress bar when tqdm is installed."""

    try:
        from tqdm import tqdm

        iterator = tqdm(samples, desc="counting tokens")
    except Exception:
        iterator = samples
    return sum(len(tokenizer.encode(sample.text, add_special_tokens=False)) for sample in iterator)


def safe_perplexity(loss: float) -> float:
    """Return a finite perplexity without inventing an arbitrary cutoff."""

    if not math.isfinite(loss):
        raise ValueError(f"Perplexity requires a finite loss, got {loss!r}.")
    if loss < 0:
        raise ValueError(f"Cross-entropy loss must be non-negative, got {loss!r}.")
    max_log = math.log(sys.float_info.max) - 1.0
    return math.exp(min(loss, max_log))
