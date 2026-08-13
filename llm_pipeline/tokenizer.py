"""SentencePiece tokenizer training/loading with explicit special-token metadata."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from .artifacts import (
    TOKENIZER_BEHAVIOR_VERSION,
    atomic_replace_file_bundle,
    atomic_write_json,
    bundle_lock_path,
    exclusive_file_lock,
    recover_file_bundle,
    tokenizer_training_fingerprint,
)
from .data import analyze_sample_stream, iter_text_samples_from_config, save_json, source_manifest_fingerprint
from .data_governance import (
    FILTER_VERSION,
    POLICY_VERSION,
    benchmark_denylist_digest,
    load_and_verify_data_audit,
    load_and_verify_source_lock,
)

TOKENIZER_CORPUS_MANIFEST_VERSION = 2
TOKENIZER_VALIDATION_FILENAME = "tokenizer_validation.json"

# These probes are code-owned and deliberately independent of any private
# training corpus. The expected strings make normalization changes explicit
# instead of silently accepting whatever a tokenizer happens to return.
NUMERIC_VALIDATION_PROBES: tuple[tuple[str, str, str], ...] = (
    ("integer", "0 7 42 1234567890", "0 7 42 1234567890"),
    ("leading_zeroes", "0000 000123 001234567890", "0000 000123 001234567890"),
    ("signed", "+17 -42 +0007 -0007", "+17 -42 +0007 -0007"),
    ("decimal", "0.0 -3.14159 +0012.500", "0.0 -3.14159 +0012.500"),
    ("exponent", "6.022e23 -1.25E-09 +9e+99", "6.022e23 -1.25E-09 +9e+99"),
    ("percent", "0% 12.5% -100%", "0% 12.5% -100%"),
    ("date_time", "2026-08-13 14:05:09 2026/08/13", "2026-08-13 14:05:09 2026/08/13"),
    ("currency", "$1,234.56 ¥7890 ₩12,000", "$1,234.56 ¥7890 ₩12,000"),
    ("radix", "0xDEADBEEF 0b101011 0o755", "0xDEADBEEF 0b101011 0o755"),
    (
        "long_identifier",
        "000123456789012345678901234567890123456789",
        "000123456789012345678901234567890123456789",
    ),
    (
        "korean_context",
        "\uc8fc\ubb38\ubc88\ud638 001234, \uae08\uc561\uc740 \u20a912,345.67\uc785\ub2c8\ub2e4.",
        "\uc8fc\ubb38\ubc88\ud638 001234, \uae08\uc561\uc740 \u20a912,345.67\uc785\ub2c8\ub2e4.",
    ),
    (
        "japanese_context",
        "\u6ce8\u6587\u756a\u53f7 001234\u3001\u4fa1\u683c\u306f\u00a512,345.67\u3067\u3059\u3002",
        "\u6ce8\u6587\u756a\u53f7 001234\u3001\u4fa1\u683c\u306f\u00a512,345.67\u3067\u3059\u3002",
    ),
    (
        "full_width_nfkc",
        "\uff29\uff24\uff1a\uff10\uff10\uff11\uff12\uff13\uff14\uff15\uff16"
        "\u3000\u4fa1\u683c\uff1a\uffe5\uff11\uff12\uff0c\uff13\uff14\uff15\uff0e\uff16\uff17\uff05",
        "ID:00123456 \u4fa1\u683c:\u00a512,345.67%",
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_probe_suite_sha256() -> str:
    """Return an opaque digest for the code-owned numeric validation suite."""

    encoded = json.dumps(NUMERIC_VALIDATION_PROBES, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sentencepiece_trainer_kwargs(
    config: dict[str, Any],
    *,
    input_path: str | Path,
    model_prefix: str | Path,
    num_threads: int | None = None,
) -> dict[str, Any]:
    """Build the single canonical SentencePiece trainer configuration."""

    tok_cfg = config["tokenizer"]
    specials = tok_cfg["special_tokens"]
    user_defined = [
        specials["user"],
        specials["assistant"],
        specials["system"],
        specials["reasoning_off"],
        specials["reasoning_low"],
        specials["reasoning_medium"],
        specials["reasoning_high"],
    ]
    if "reasoning_max" in specials:
        user_defined.append(specials["reasoning_max"])
    user_defined.append(specials["mask"])
    kwargs: dict[str, Any] = {
        "input": str(input_path),
        "model_prefix": str(model_prefix),
        "model_type": tok_cfg["model_type"],
        "vocab_size": int(tok_cfg["vocab_size"]),
        "character_coverage": float(tok_cfg["character_coverage"]),
        "input_sentence_size": int(tok_cfg["input_sentence_size"]),
        "shuffle_input_sentence": bool(tok_cfg["shuffle_input_sentence"]),
        "byte_fallback": bool(tok_cfg["byte_fallback"]),
        "split_digits": bool(tok_cfg.get("split_digits", True)),
        "normalization_rule_name": str(tok_cfg.get("normalization_rule_name", "nmt_nfkc")),
        "hard_vocab_limit": False,
        "unk_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 3,
        "pad_piece": specials["pad"],
        "unk_piece": specials["unk"],
        "bos_piece": specials["bos"],
        "eos_piece": specials["eos"],
        "user_defined_symbols": ",".join(user_defined),
    }
    if num_threads is not None:
        kwargs["num_threads"] = max(1, int(num_threads))
    return kwargs


def _corpus_signature(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_count = 0
    line_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
            line_count += chunk.count(b"\n")
    return {"corpus_bytes": byte_count, "corpus_lines": line_count, "corpus_sha256": digest.hexdigest()}


def build_tokenizer_corpus_manifest(config: dict[str, Any], corpus: Path) -> dict[str, Any]:
    """Record enough evidence to prove exactly how a tokenizer corpus was built."""

    policy_enforced = bool(config.get("data_policy", {}).get("enforce", True))
    source_lock_digest = None
    audit_digest = None
    if policy_enforced:
        lock = load_and_verify_source_lock(config, verify_hashes=True)
        audit = load_and_verify_data_audit(config, lock)
        source_lock_digest = lock["lock_digest"]
        audit_digest = audit["audit_digest"]
    payload = {
        "format_version": TOKENIZER_CORPUS_MANIFEST_VERSION,
        "corpus_file": corpus.name,
        **_corpus_signature(corpus),
        "source_fingerprint": source_manifest_fingerprint(config, "tokenizer"),
        "policy_enforced": policy_enforced,
        "policy_version": POLICY_VERSION,
        "filter_version": FILTER_VERSION,
        "source_lock_digest": source_lock_digest,
        "audit_digest": audit_digest,
        "benchmark_denylist_digest": benchmark_denylist_digest(config),
        "finalized_from_existing_corpus": False,
    }
    atomic_write_json(corpus.parent / "corpus_manifest.json", payload)
    return payload


def verify_tokenizer_corpus_manifest(config: dict[str, Any], corpus: Path) -> dict[str, Any]:
    """Fail unless an existing corpus still matches all current provenance evidence."""

    manifest_path = corpus.parent / "corpus_manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Tokenizer corpus provenance is missing or invalid: {manifest_path}") from exc
    if payload.get("format_version") != TOKENIZER_CORPUS_MANIFEST_VERSION:
        raise RuntimeError("Tokenizer corpus provenance uses an obsolete format and cannot be trusted.")
    actual_signature = _corpus_signature(corpus)
    if any(payload.get(key) != value for key, value in actual_signature.items()):
        raise RuntimeError("Tokenizer corpus bytes, line count, or SHA-256 changed after provenance was recorded.")
    expected = {
        "source_fingerprint": source_manifest_fingerprint(config, "tokenizer"),
        "policy_version": POLICY_VERSION,
        "filter_version": FILTER_VERSION,
        "benchmark_denylist_digest": benchmark_denylist_digest(config),
    }
    policy_enforced = bool(config.get("data_policy", {}).get("enforce", True))
    expected["policy_enforced"] = policy_enforced
    if policy_enforced:
        lock = load_and_verify_source_lock(config, verify_hashes=True)
        audit = load_and_verify_data_audit(config, lock)
        expected["source_lock_digest"] = lock["lock_digest"]
        expected["audit_digest"] = audit["audit_digest"]
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Tokenizer corpus provenance is stale relative to current data or policy evidence.")
    return payload


class SentencePieceTokenizer:
    """Thin adapter around sentencepiece so the rest of the code has stable ids."""

    def __init__(self, model_path: str | Path, special_tokens: dict[str, str]) -> None:
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise RuntimeError(
                "sentencepiece is required for tokenizer.type=sentencepiece. "
                "Install dependencies with `python -m pip install -r requirements.txt`."
            ) from exc

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Tokenizer model not found: {self.model_path}")
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.special_tokens = special_tokens
        missing = [
            name
            for name, token in special_tokens.items()
            if self.sp.id_to_piece(int(self.sp.piece_to_id(token))) != token
        ]
        if missing:
            raise ValueError(f"Tokenizer model is missing configured special tokens: {missing}")
        resolved_ids = {name: int(self.sp.piece_to_id(token)) for name, token in special_tokens.items()}
        if len(set(resolved_ids.values())) != len(resolved_ids):
            raise ValueError("Configured tokenizer special tokens do not resolve to unique ids.")
        self.pad_id = self.piece_to_id(special_tokens["pad"])
        self.unk_id = self.piece_to_id(special_tokens["unk"])
        self.bos_id = self.piece_to_id(special_tokens["bos"])
        self.eos_id = self.piece_to_id(special_tokens["eos"])
        self.mask_id = self.piece_to_id(special_tokens["mask"])
        self.special_ids = frozenset(resolved_ids.values())
        self.vocab_size = self.sp.get_piece_size()
        self.validation_metadata: dict[str, Any] | None = None

    def piece_to_id(self, piece: str) -> int:
        idx = int(self.sp.piece_to_id(piece))
        return self.sp.unk_id() if idx < 0 else idx

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = list(self.sp.encode(text, out_type=int))
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        filtered = [int(i) for i in ids if int(i) not in self.special_ids]
        return str(self.sp.decode(filtered))

    def save_metadata(
        self,
        save_dir: str | Path,
        config: dict[str, Any],
        validation: dict[str, Any] | None = None,
    ) -> None:
        output = Path(save_dir)
        output.mkdir(parents=True, exist_ok=True)
        output_model = output / "tokenizer.model"
        if self.model_path.resolve() != output_model.resolve():
            shutil.copy2(self.model_path, output_model)
        model_sha256 = _sha256_file(output_model)

        clean_validation = validation or self.validation_metadata
        if clean_validation is None:
            source_validation = self.model_path.parent / TOKENIZER_VALIDATION_FILENAME
            try:
                candidate = json.loads(source_validation.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                candidate = None
            if (
                isinstance(candidate, dict)
                and candidate.get("status") == "passed"
                and candidate.get("model_sha256") == model_sha256
            ):
                clean_validation = candidate

        validation_summary: dict[str, Any]
        if clean_validation is not None:
            if clean_validation.get("status") != "passed":
                raise RuntimeError("Refusing to save tokenizer metadata for a validation result that did not pass.")
            if clean_validation.get("model_sha256") != model_sha256:
                raise RuntimeError("Tokenizer validation metadata does not match the tokenizer model SHA-256.")
            if clean_validation.get("behavior_version") != TOKENIZER_BEHAVIOR_VERSION:
                raise RuntimeError("Tokenizer validation metadata uses an unsupported behavior version.")
            atomic_write_json(output / TOKENIZER_VALIDATION_FILENAME, clean_validation)
            validation_summary = {
                "status": "passed",
                "path": TOKENIZER_VALIDATION_FILENAME,
                "probe_count": int(clean_validation["probe_count"]),
                "corpus_samples_checked": int(clean_validation["corpus_samples_checked"]),
                "unk_count": int(clean_validation["unk_count"]),
            }
        else:
            validation_summary = {
                "status": "not_available",
                "path": None,
                "probe_count": 0,
                "corpus_samples_checked": 0,
                "unk_count": None,
            }

        tok_cfg = config["tokenizer"]
        token_map = {
            name: {"token": token, "id": self.piece_to_id(token)} for name, token in self.special_tokens.items()
        }
        save_json(output / "special_tokens_map.json", token_map)
        atomic_write_json(
            output / "tokenizer_config.json",
            {
                "type": "sentencepiece",
                "model_type": tok_cfg["model_type"],
                "vocab_size": self.vocab_size,
                "target_vocab_size": int(tok_cfg["vocab_size"]),
                "byte_fallback": tok_cfg["byte_fallback"],
                "behavior_version": TOKENIZER_BEHAVIOR_VERSION,
                "split_digits": bool(tok_cfg.get("split_digits", True)),
                "normalization_rule_name": str(tok_cfg.get("normalization_rule_name", "nmt_nfkc")),
                "normalization": "sentencepiece_nmt_nfkc",
                "model_sha256": model_sha256,
                "validation_path": validation_summary["path"],
                "validation": validation_summary,
                "data_sources_fingerprint": source_manifest_fingerprint(config, "tokenizer"),
                "training_fingerprint": tokenizer_training_fingerprint(config),
            },
        )
        # A lightweight tokenizer.json manifest is enough for downstream tooling
        # to discover the canonical model file and special ids.
        save_json(output / "tokenizer.json", {"model_file": "tokenizer.model", "special_tokens": token_map})


def validate_tokenizer_candidate(
    tokenizer: SentencePieceTokenizer,
    config: dict[str, Any],
    corpus_path: str | Path,
) -> dict[str, Any]:
    """Validate numeric behavior and a bounded corpus sample without exposing text."""

    tok_cfg = config["tokenizer"]
    split_digits = bool(tok_cfg.get("split_digits", True))
    normalization_rule_name = str(tok_cfg.get("normalization_rule_name", "nmt_nfkc"))
    numeric_validation = bool(tok_cfg.get("numeric_validation", True))
    probe_count = 0
    unk_count = 0

    if numeric_validation:
        for probe_index, (_name, source, expected) in enumerate(NUMERIC_VALIDATION_PROBES, start=1):
            if unicodedata.normalize("NFKC", source) != expected:
                raise RuntimeError(f"Numeric probe {probe_index} has an invalid code-owned NFKC expectation.")
            ids = tokenizer.encode(source, add_special_tokens=False)
            probe_unk_count = ids.count(tokenizer.unk_id)
            unk_count += probe_unk_count
            if probe_unk_count:
                raise RuntimeError(f"Numeric probe {probe_index} produced an unknown token id.")
            if tokenizer.decode(ids) != expected:
                raise RuntimeError(f"Numeric probe {probe_index} failed exact canonical round-trip validation.")
            for piece_id in ids:
                if tokenizer.sp.is_byte(piece_id):
                    continue
                piece = tokenizer.sp.id_to_piece(piece_id)
                if sum(character.isdecimal() for character in piece) > 1:
                    raise RuntimeError(
                        f"Numeric probe {probe_index} encoded multiple decimal digits in one vocabulary piece."
                    )
            probe_count += 1

    if split_digits:
        for piece_id in range(tokenizer.vocab_size):
            if (
                tokenizer.sp.is_byte(piece_id)
                or tokenizer.sp.is_control(piece_id)
                or tokenizer.sp.is_unknown(piece_id)
                or tokenizer.sp.is_unused(piece_id)
            ):
                continue
            piece = tokenizer.sp.id_to_piece(piece_id)
            if sum(character.isdecimal() for character in piece) > 1:
                raise RuntimeError(
                    "Tokenizer split_digits validation found a non-byte vocabulary piece with multiple decimal digits."
                )

    corpus_limit = max(0, int(tok_cfg.get("numeric_validation_corpus_samples", 256)))
    corpus_samples_checked = 0
    if corpus_limit:
        try:
            with Path(corpus_path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    sample = line.rstrip("\r\n")
                    if not sample:
                        continue
                    ids = tokenizer.encode(sample, add_special_tokens=False)
                    sample_unk_count = ids.count(tokenizer.unk_id)
                    unk_count += sample_unk_count
                    corpus_samples_checked += 1
                    if sample_unk_count:
                        raise RuntimeError(
                            f"Tokenizer corpus sample {corpus_samples_checked} produced an unknown token id."
                        )
                    if not tokenizer.decode(ids).strip():
                        raise RuntimeError(
                            f"Tokenizer corpus sample {corpus_samples_checked} decoded to an empty string."
                        )
                    if corpus_samples_checked >= corpus_limit:
                        break
        except UnicodeDecodeError:
            raise RuntimeError("Tokenizer validation corpus is not valid UTF-8.") from None
        if corpus_samples_checked == 0:
            raise RuntimeError("Tokenizer validation corpus has no non-empty samples.")

    metadata = {
        "behavior_version": TOKENIZER_BEHAVIOR_VERSION,
        "status": "passed",
        "split_digits": split_digits,
        "normalization_rule_name": normalization_rule_name,
        "probe_count": probe_count,
        "probe_suite_sha256": numeric_probe_suite_sha256(),
        "corpus_samples_checked": corpus_samples_checked,
        "unk_count": unk_count,
        "model_sha256": _sha256_file(tokenizer.model_path),
    }
    tokenizer.validation_metadata = metadata
    return metadata


def verify_tokenizer_artifacts(tokenizer: SentencePieceTokenizer, config: dict[str, Any]) -> None:
    """Reject a tokenizer whose numeric-integrity evidence is missing or stale."""

    root = tokenizer.model_path.parent
    try:
        tokenizer_config = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
        validation = json.loads((root / TOKENIZER_VALIDATION_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(
            "Tokenizer integrity metadata is missing or invalid; retrain the tokenizer with the current code."
        ) from exc
    tok_cfg = config["tokenizer"]
    model_sha256 = _sha256_file(tokenizer.model_path)
    expected = {
        "behavior_version": TOKENIZER_BEHAVIOR_VERSION,
        "split_digits": bool(tok_cfg["split_digits"]),
        "normalization_rule_name": str(tok_cfg["normalization_rule_name"]),
        "model_sha256": model_sha256,
    }
    if any(tokenizer_config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Tokenizer configuration metadata is stale or does not match the model bytes.")
    if int(tokenizer_config.get("target_vocab_size", 0)) != int(tok_cfg["vocab_size"]):
        raise RuntimeError("Tokenizer target vocabulary size does not match the active configuration.")
    if any(validation.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Tokenizer validation metadata is stale or does not match the model bytes.")
    if (
        validation.get("status") != "passed"
        or validation.get("probe_suite_sha256") != numeric_probe_suite_sha256()
        or int(validation.get("probe_count", -1)) != len(NUMERIC_VALIDATION_PROBES)
        or int(validation.get("corpus_samples_checked", 0)) <= 0
        or int(validation.get("unk_count", -1)) != 0
    ):
        raise RuntimeError("Tokenizer numeric-integrity evidence is incomplete or obsolete.")


def publish_tokenizer_bundle(
    candidate: SentencePieceTokenizer,
    config: dict[str, Any],
    validation: dict[str, Any],
    target_dir: str | Path,
    *,
    vocab_path: str | Path,
) -> SentencePieceTokenizer:
    """Publish validated tokenizer artifacts under one rollback-safe lock."""

    target = Path(target_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-bundle-", dir=target.parent) as temporary:
        temporary_root = Path(temporary)
        staged = temporary_root / "staged"
        staged.mkdir()
        candidate.save_metadata(staged, config, validation)
        shutil.copy2(vocab_path, staged / "tokenizer.vocab")
        artifact_names = (
            "tokenizer.model",
            "tokenizer.vocab",
            TOKENIZER_VALIDATION_FILENAME,
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
        )
        atomic_replace_file_bundle(staged, target, artifact_names)
    published = SentencePieceTokenizer(target / "tokenizer.model", config["tokenizer"]["special_tokens"])
    verify_tokenizer_artifacts(published, config)
    return published


def load_tokenizer(config: dict[str, Any]) -> SentencePieceTokenizer:
    """Load the configured tokenizer."""

    tok_cfg = config["tokenizer"]
    if tok_cfg["type"] != "sentencepiece":
        raise ValueError("Only tokenizer.type=sentencepiece is implemented for production training.")
    model_path = Path(tok_cfg["model_path"])
    with exclusive_file_lock(bundle_lock_path(model_path.parent)):
        recover_file_bundle(model_path.parent)
        tokenizer = SentencePieceTokenizer(model_path, tok_cfg["special_tokens"])
        verify_tokenizer_artifacts(tokenizer, config)
    return tokenizer


def train_tokenizer(config: dict[str, Any], logger: Any) -> SentencePieceTokenizer:
    """Train SentencePiece BPE/Unigram and save all tokenizer artifacts."""

    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError(
            "train_tokenizer requires sentencepiece. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    logger.info("Loading samples for tokenizer training.")
    stats_iter = iter_text_samples_from_config(
        config,
        split="train",
        dataset_type="tokenizer",
        fallback_path=config["data"]["train_file"],
    )
    try:
        from tqdm import tqdm

        stats_iter = tqdm(stats_iter, desc="analyzing tokenizer corpus", unit="sample")
    except Exception:
        pass
    stats = analyze_sample_stream(stats_iter)
    logger.info(f"Tokenizer corpus stats: {json.dumps(stats, ensure_ascii=False)}")
    logger.info(f"Recommended vocab sizes for this corpus: {stats['recommended_vocab_sizes']}")

    tok_cfg = config["tokenizer"]
    save_dir = Path(tok_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    input_path = save_dir / "tokenizer_corpus.txt"
    corpus_iter = iter_text_samples_from_config(
        config,
        split="train",
        dataset_type="tokenizer",
        fallback_path=config["data"]["train_file"],
    )
    try:
        from tqdm import tqdm

        corpus_iter = tqdm(corpus_iter, desc="writing tokenizer corpus", unit="sample")
    except Exception:
        pass
    with input_path.open("w", encoding="utf-8") as handle:
        for sample in corpus_iter:
            handle.write(sample.text.replace("\n", " ") + "\n")

    build_tokenizer_corpus_manifest(config, input_path)

    specials = tok_cfg["special_tokens"]
    validation: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="tokenizer-build-", dir=save_dir) as tmp:
        temporary_prefix = Path(tmp) / "tokenizer"
        spm.SentencePieceTrainer.train(
            **sentencepiece_trainer_kwargs(
                config,
                input_path=input_path,
                model_prefix=temporary_prefix,
            )
        )
        candidate = SentencePieceTokenizer(temporary_prefix.with_suffix(".model"), specials)
        if candidate.vocab_size != int(tok_cfg["vocab_size"]):
            raise RuntimeError(
                f"Tokenizer produced vocab_size={candidate.vocab_size}, expected {tok_cfg['vocab_size']}."
            )
        validation = validate_tokenizer_candidate(candidate, config, input_path)
        tokenizer = publish_tokenizer_bundle(
            candidate,
            config,
            validation,
            save_dir,
            vocab_path=temporary_prefix.with_suffix(".vocab"),
        )
    logger.info(
        "Tokenizer validation passed: "
        f"probes={validation['probe_count']}, "
        f"corpus_samples={validation['corpus_samples_checked']}, "
        f"unknown_tokens={validation['unk_count']}, "
        f"model_sha256={validation['model_sha256']}"
    )
    return tokenizer
