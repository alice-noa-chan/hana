"""SentencePiece tokenizer training/loading with explicit special-token metadata."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, tokenizer_training_fingerprint
from .data import analyze_sample_stream, iter_text_samples_from_config, save_json, source_manifest_fingerprint
from .data_governance import (
    FILTER_VERSION,
    POLICY_VERSION,
    benchmark_denylist_digest,
    load_and_verify_data_audit,
    load_and_verify_source_lock,
)

TOKENIZER_CORPUS_MANIFEST_VERSION = 2


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
        lock = load_and_verify_source_lock(config)
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
        lock = load_and_verify_source_lock(config)
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

    def save_metadata(self, save_dir: str | Path, config: dict[str, Any]) -> None:
        output = Path(save_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.model_path.resolve() != (output / "tokenizer.model").resolve():
            shutil.copy2(self.model_path, output / "tokenizer.model")
        token_map = {
            name: {"token": token, "id": self.piece_to_id(token)} for name, token in self.special_tokens.items()
        }
        save_json(output / "special_tokens_map.json", token_map)
        save_json(
            output / "tokenizer_config.json",
            {
                "type": "sentencepiece",
                "model_type": config["tokenizer"]["model_type"],
                "vocab_size": self.vocab_size,
                "target_vocab_size": int(config["tokenizer"]["vocab_size"]),
                "byte_fallback": config["tokenizer"]["byte_fallback"],
                "normalization": "external_nfkc_plus_sentencepiece_normalizer",
                "data_sources_fingerprint": source_manifest_fingerprint(config, "tokenizer"),
                "training_fingerprint": tokenizer_training_fingerprint(config),
            },
        )
        # A lightweight tokenizer.json manifest is enough for downstream tooling
        # to discover the canonical model file and special ids.
        save_json(output / "tokenizer.json", {"model_file": "tokenizer.model", "special_tokens": token_map})


def load_tokenizer(config: dict[str, Any]) -> SentencePieceTokenizer:
    """Load the configured tokenizer."""

    tok_cfg = config["tokenizer"]
    if tok_cfg["type"] != "sentencepiece":
        raise ValueError("Only tokenizer.type=sentencepiece is implemented for production training.")
    return SentencePieceTokenizer(tok_cfg["model_path"], tok_cfg["special_tokens"])


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
    sample_text = None
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
            if sample_text is None:
                sample_text = sample.text
            handle.write(sample.text.replace("\n", " ") + "\n")

    build_tokenizer_corpus_manifest(config, input_path)

    specials = tok_cfg["special_tokens"]
    user_defined = [
        specials["user"],
        specials["assistant"],
        specials["system"],
        specials["reasoning_off"],
        specials["reasoning_low"],
        specials["reasoning_medium"],
        specials["reasoning_high"],
        specials["mask"],
    ]
    prefix = save_dir / "tokenizer"
    with tempfile.TemporaryDirectory(prefix="tokenizer-build-", dir=save_dir) as tmp:
        temporary_prefix = Path(tmp) / "tokenizer"
        spm.SentencePieceTrainer.train(
            input=str(input_path),
            model_prefix=str(temporary_prefix),
            model_type=tok_cfg["model_type"],
            vocab_size=int(tok_cfg["vocab_size"]),
            character_coverage=float(tok_cfg["character_coverage"]),
            input_sentence_size=int(tok_cfg["input_sentence_size"]),
            shuffle_input_sentence=bool(tok_cfg["shuffle_input_sentence"]),
            byte_fallback=bool(tok_cfg["byte_fallback"]),
            hard_vocab_limit=False,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            pad_piece=specials["pad"],
            unk_piece=specials["unk"],
            bos_piece=specials["bos"],
            eos_piece=specials["eos"],
            user_defined_symbols=",".join(user_defined),
        )
        temporary_prefix.with_suffix(".model").replace(prefix.with_suffix(".model"))
        temporary_prefix.with_suffix(".vocab").replace(prefix.with_suffix(".vocab"))

    tokenizer = SentencePieceTokenizer(prefix.with_suffix(".model"), specials)
    tokenizer.save_metadata(save_dir, config)
    sample_text = sample_text or "Tokenizer smoke test text with ordinary words and numbers 123."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    logger.info(f"Tokenizer sample encode: {encoded[:32]}")
    logger.info(f"Tokenizer sample decode: {decoded[:200]}")
    if not decoded:
        raise RuntimeError("Tokenizer decode smoke test failed: decoded string is empty.")
    return tokenizer
