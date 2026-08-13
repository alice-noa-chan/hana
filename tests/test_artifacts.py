from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import pytest

from llm_pipeline.artifacts import (
    BUNDLE_TRANSACTION_VERSION,
    atomic_write_json,
    build_rejects_fingerprint,
    checkpoint_fingerprint,
    evaluation_fingerprint,
    inference_fingerprint,
    recover_file_bundle,
    tokenizer_training_fingerprint,
    training_fingerprint,
)
from llm_pipeline.config import DEFAULT_CONFIG
from llm_pipeline.logging_utils import RunLogger


def test_atomic_json_is_standard_compliant(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    atomic_write_json(path, {"value": 3.5})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 3.5}

    with pytest.raises(ValueError, match="JSON compliant"):
        atomic_write_json(path, {"value": math.inf})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 3.5}


def test_checkpoint_fingerprint_changes_with_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"first")
    before = checkpoint_fingerprint(checkpoint)
    weights.write_bytes(b"second-version")
    assert checkpoint_fingerprint(checkpoint) != before


def test_dpo_training_fingerprint_tracks_direct_preference_file(tmp_path: Path) -> None:
    preference_path = tmp_path / "preferences.jsonl"
    preference_path.write_text('{"prompt":"a","chosen":"b","rejected":"c"}\n', encoding="utf-8")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dpo"]["train_file"] = str(preference_path)

    before = training_fingerprint(config, "dpo")
    preference_path.write_text('{"prompt":"changed","chosen":"b","rejected":"c"}\n', encoding="utf-8")

    assert training_fingerprint(config, "dpo") != before


def test_evaluation_fingerprint_tracks_prompt_rendering_inputs(tmp_path: Path) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    before = evaluation_fingerprint(config, checkpoint)
    config["inference"]["model_system_prompt"] = "different private system instruction"
    after_prompt = evaluation_fingerprint(config, checkpoint)
    config["tokenizer"]["special_tokens"]["assistant"] = "<different-assistant>"

    assert after_prompt != before
    assert evaluation_fingerprint(config, checkpoint) != after_prompt


def test_rejected_generation_fingerprint_tracks_reasoning_policy(tmp_path: Path) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    before = build_rejects_fingerprint(config, checkpoint)
    config["reasoning"]["max_reasoning_tokens"] += 1

    assert build_rejects_fingerprint(config, checkpoint) != before


def test_inference_fingerprint_tracks_seed_controls_and_prompt_normalization(tmp_path: Path) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    baseline = inference_fingerprint(config, checkpoint)
    config["run"]["seed"] += 1
    changed_seed = inference_fingerprint(config, checkpoint)
    config["tokenizer"]["special_tokens"]["reasoning_max"] = "<different-reasoning-max>"
    changed_control = inference_fingerprint(config, checkpoint)
    config["data"]["normalize_nfkc"] = not config["data"]["normalize_nfkc"]
    changed_normalization = inference_fingerprint(config, checkpoint)

    assert len({baseline, changed_seed, changed_control, changed_normalization}) == 4


def test_training_fingerprint_tracks_tokenizer_bytes_with_unchanged_metadata(tmp_path: Path) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    tokenizer_model = tmp_path / "tokenizer.model"
    tokenizer_model.write_bytes(b"token-one")
    original_stat = tokenizer_model.stat()
    config["tokenizer"]["model_path"] = str(tokenizer_model)

    before = training_fingerprint(config, "pretrain")
    tokenizer_model.write_bytes(b"token-two")
    os.utime(tokenizer_model, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert tokenizer_model.stat().st_size == original_stat.st_size
    assert tokenizer_model.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert training_fingerprint(config, "pretrain") != before


def test_tokenizer_training_fingerprint_tracks_numeric_contract() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    before = tokenizer_training_fingerprint(config)
    config["tokenizer"]["split_digits"] = False

    assert tokenizer_training_fingerprint(config) != before


def test_tokenizer_training_fingerprint_tracks_source_bytes_with_unchanged_metadata(tmp_path: Path) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    source = tmp_path / "source.jsonl"
    source.write_bytes(b'{"text":"row-one"}\n')
    original_stat = source.stat()
    config["data"]["sources"] = [{"name": "fixture", "path": str(source), "schema": "text"}]

    before = tokenizer_training_fingerprint(config)
    source.write_bytes(b'{"text":"row-two"}\n')
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert source.stat().st_size == original_stat.st_size
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert tokenizer_training_fingerprint(config) != before


def test_file_bundle_recovery_rolls_back_interrupted_promotion(tmp_path: Path) -> None:
    target = tmp_path / "bundle"
    target.mkdir()
    (target / "model.bin").write_bytes(b"new-model")
    (target / "config.json").write_bytes(b"old-config")
    (target / "new-evidence.json").write_bytes(b"partial-new-evidence")
    transaction_root = tmp_path / ".bundle.bundle-synthetic"
    backup = transaction_root / "backup"
    backup.mkdir(parents=True)
    (transaction_root / "incoming").mkdir()
    (backup / "model.bin").write_bytes(b"old-model")
    journal = {
        "version": BUNDLE_TRANSACTION_VERSION,
        "target": str(target.resolve()),
        "transaction_root": str(transaction_root.resolve()),
        "names": ["model.bin", "config.json", "new-evidence.json"],
        "previous_names": ["model.bin", "config.json"],
    }
    journal_path = tmp_path / ".bundle.bundle.transaction.json"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    assert recover_file_bundle(target)
    assert (target / "model.bin").read_bytes() == b"old-model"
    assert (target / "config.json").read_bytes() == b"old-config"
    assert not (target / "new-evidence.json").exists()
    assert not journal_path.exists()
    assert not transaction_root.exists()


def test_tensorboard_setting_writes_real_event_data(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path, tensorboard_enabled=True)
    logger.metric({"stage": "pretrain", "step": 3, "loss": 1.25, "message": "ignored"})
    logger.close()

    events = list((tmp_path / "tensorboard").glob("events.out.tfevents.*"))
    assert events and events[0].stat().st_size > 0
