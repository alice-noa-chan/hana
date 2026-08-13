from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm_pipeline.artifacts import tokenizer_training_fingerprint
from llm_pipeline.config import load_config
from llm_pipeline.data import source_manifest_fingerprint
from llm_pipeline.progress import mode_is_complete
from llm_pipeline.tokenizer import (
    NUMERIC_VALIDATION_PROBES,
    TOKENIZER_BEHAVIOR_VERSION,
    numeric_probe_suite_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def test_tokenizer_progress_invalidates_changed_special_tokens(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").mutable_copy()
    config["tokenizer"]["save_dir"] = str(tmp_path)
    for name in ("tokenizer.model", "tokenizer.json", "special_tokens_map.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    model_sha256 = hashlib.sha256((tmp_path / "tokenizer.model").read_bytes()).hexdigest()
    validation = {
        "behavior_version": TOKENIZER_BEHAVIOR_VERSION,
        "status": "passed",
        "split_digits": config["tokenizer"]["split_digits"],
        "normalization_rule_name": config["tokenizer"]["normalization_rule_name"],
        "probe_count": len(NUMERIC_VALIDATION_PROBES),
        "probe_suite_sha256": numeric_probe_suite_sha256(),
        "corpus_samples_checked": 1,
        "unk_count": 0,
        "model_sha256": model_sha256,
    }
    (tmp_path / "tokenizer_validation.json").write_text(json.dumps(validation), encoding="utf-8")
    metadata = {
        "target_vocab_size": config["tokenizer"]["vocab_size"],
        "behavior_version": TOKENIZER_BEHAVIOR_VERSION,
        "model_sha256": model_sha256,
        "split_digits": config["tokenizer"]["split_digits"],
        "normalization_rule_name": config["tokenizer"]["normalization_rule_name"],
        "data_sources_fingerprint": source_manifest_fingerprint(config, "tokenizer"),
        "training_fingerprint": tokenizer_training_fingerprint(config),
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert mode_is_complete("train_tokenizer", config)

    malformed = dict(validation)
    malformed["probe_count"] = {"not": "an integer"}
    (tmp_path / "tokenizer_validation.json").write_text(json.dumps(malformed), encoding="utf-8")
    assert not mode_is_complete("train_tokenizer", config)
    (tmp_path / "tokenizer_validation.json").write_text(json.dumps(validation), encoding="utf-8")
    assert mode_is_complete("train_tokenizer", config)

    config["tokenizer"]["special_tokens"]["mask"] = "<different-mask>"
    assert not mode_is_complete("train_tokenizer", config)


def test_tokenizer_progress_invalidates_changed_model_bytes(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").mutable_copy()
    config["tokenizer"]["save_dir"] = str(tmp_path)
    for name in ("tokenizer.model", "tokenizer.json", "special_tokens_map.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    model_sha256 = hashlib.sha256((tmp_path / "tokenizer.model").read_bytes()).hexdigest()
    validation = {
        "behavior_version": TOKENIZER_BEHAVIOR_VERSION,
        "status": "passed",
        "split_digits": True,
        "normalization_rule_name": "nmt_nfkc",
        "probe_count": len(NUMERIC_VALIDATION_PROBES),
        "probe_suite_sha256": numeric_probe_suite_sha256(),
        "corpus_samples_checked": 1,
        "unk_count": 0,
        "model_sha256": model_sha256,
    }
    (tmp_path / "tokenizer_validation.json").write_text(json.dumps(validation), encoding="utf-8")
    metadata = {
        "target_vocab_size": config["tokenizer"]["vocab_size"],
        "behavior_version": TOKENIZER_BEHAVIOR_VERSION,
        "model_sha256": model_sha256,
        "split_digits": True,
        "normalization_rule_name": "nmt_nfkc",
        "data_sources_fingerprint": source_manifest_fingerprint(config, "tokenizer"),
        "training_fingerprint": tokenizer_training_fingerprint(config),
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert mode_is_complete("train_tokenizer", config)

    (tmp_path / "tokenizer.model").write_text("changed", encoding="utf-8")

    assert not mode_is_complete("train_tokenizer", config)
