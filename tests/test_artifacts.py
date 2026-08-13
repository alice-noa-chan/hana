from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from llm_pipeline.artifacts import atomic_write_json, checkpoint_fingerprint, training_fingerprint
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


def test_tensorboard_setting_writes_real_event_data(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path, tensorboard_enabled=True)
    logger.metric({"stage": "pretrain", "step": 3, "loss": 1.25, "message": "ignored"})
    logger.close()

    events = list((tmp_path / "tensorboard").glob("events.out.tfevents.*"))
    assert events and events[0].stat().st_size > 0
