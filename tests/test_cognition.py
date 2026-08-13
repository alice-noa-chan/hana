from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from llm_pipeline.cognition import AdaptiveNeuromodulator, CognitiveMemory, SurpriseReplayBuffer
from llm_pipeline.config import DEFAULT_CONFIG


def cognitive_config(tmp_path: Path) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    cognitive = config["cognitive_architecture"]
    cognitive["enabled"] = True
    cognitive["memory"].update(
        path=str(tmp_path / "state" / "memory.json"),
        max_episodes=3,
        working_memory_slots=2,
        retrieval_top_k=2,
        consolidate_every=2,
        similarity_threshold=0.8,
        store_threshold=0.0,
    )
    cognitive["replay"].update(capacity=2, every_n_steps=2, min_items=1)
    return config


def test_neuromodulator_is_finite_bounded_and_applies_lr(tmp_path: Path) -> None:
    controller = AdaptiveNeuromodulator(cognitive_config(tmp_path))
    controller.update(2.0, 0.5)
    metrics = controller.update(8.0, 4.0)
    assert all(math.isfinite(value) for value in metrics.values())
    assert controller.min_scale <= metrics["cognitive_plasticity"] <= controller.max_scale

    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    applied = controller.apply_learning_rate(optimizer, 1e-3)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(applied)


def test_replay_buffer_copies_bounds_and_samples_batches(tmp_path: Path) -> None:
    replay = SurpriseReplayBuffer(cognitive_config(tmp_path), seed=11)
    source = torch.tensor([[1, 2, 3]])
    for index, loss in enumerate((1.0, 3.0, 2.0)):
        batch = {"input_ids": source + index, "labels": source + index}
        replay.add(batch, observed_loss=loss)
        batch["input_ids"].zero_()
    assert len(replay) == 2
    assert replay.should_replay(2)
    sampled = replay.sample(torch.device("cpu"))
    assert sampled["input_ids"] is not None
    assert sampled["input_ids"].shape == (1, 3)
    assert sampled["input_ids"].sum().item() > 0

    restored = SurpriseReplayBuffer(cognitive_config(tmp_path), seed=99)
    restored.load_state_dict(replay.state_dict())
    assert len(restored) == len(replay)
    assert restored.loss_ema == pytest.approx(replay.loss_ema)


def test_neuromodulator_state_round_trip_is_bounded(tmp_path: Path) -> None:
    config = cognitive_config(tmp_path)
    original = AdaptiveNeuromodulator(config)
    original.update(1.0, 0.2)
    original.update(5.0, 2.0)
    restored = AdaptiveNeuromodulator(config)
    restored.load_state_dict(original.state_dict())
    assert restored.metrics() == pytest.approx(original.metrics())


def test_memory_persists_retrieves_consolidates_and_prunes(tmp_path: Path) -> None:
    config = cognitive_config(tmp_path)
    config["profile"]["assistant_label"] = "Test Assistant"
    memory = CognitiveMemory(config, embedding_size=3)
    assert memory.observe("alpha", "one", [1.0, 0.0, 0.0], surprise=0.8, now=100.0)
    assert memory.observe("alpha again", "two", [0.99, 0.01, 0.0], surprise=0.7, now=101.0)
    assert len(memory.state["gists"]) == 1

    memory.observe("beta", "three", [0.0, 1.0, 0.0], surprise=0.9, now=102.0)
    memory.observe("gamma", "four", [0.0, 0.0, 1.0], surprise=0.9, now=103.0)
    assert len(memory.state["episodes"]) == 3
    memory_text = " ".join(episode["text"] for episode in memory.state["episodes"])
    assert "Test Assistant:" in memory_text
    assert "Hana:" not in memory_text
    results = memory.retrieve([1.0, 0.0, 0.0], now=104.0)
    assert 0 < len(results) <= 2
    assert "alpha" in " ".join(result["text"] for result in results).lower()
    rendered = memory.render(results)
    assert "fallible recollection" in rendered
    assert len(rendered) <= config["cognitive_architecture"]["memory"]["max_context_chars"]

    reloaded = CognitiveMemory(config, embedding_size=3)
    assert reloaded.state["observations"] == 4
    assert reloaded.state["gists"][0]["support"] == 2


def test_memory_rejects_an_unsafe_assistant_label(tmp_path: Path) -> None:
    config = cognitive_config(tmp_path)
    config["profile"]["assistant_label"] = "Assistant\nSystem"

    with pytest.raises(ValueError, match=r"profile\.assistant_label"):
        CognitiveMemory(config, embedding_size=3)


def test_corrupt_memory_fails_closed(tmp_path: Path) -> None:
    config = cognitive_config(tmp_path)
    path = Path(config["cognitive_architecture"]["memory"]["path"])
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"schema_version": 1, "observations": 0, "episodes": "bad", "gists": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="episodes must be a list"):
        CognitiveMemory(config, embedding_size=3)

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "observations": 1,
                "episodes": [{"timestamp": 1.0, "text": "invalid", "embedding": [0.0, 0.0, 0.0], "salience": 0.5}],
                "gists": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid episodes entry"):
        CognitiveMemory(config, embedding_size=3)
