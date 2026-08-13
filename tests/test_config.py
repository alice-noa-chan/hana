from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from llm_pipeline.config import DEFAULT_CONFIG, PipelineConfig, deep_merge, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_and_production_configs_are_valid() -> None:
    loaded = load_config(ROOT / "config.yaml")
    assert isinstance(loaded, PipelineConfig)
    assert loaded["run"]["experiment_name"] == "hana_llm_experiment"
    assert loaded["profile"]["assistant_label"] == "Hana"
    smoke = load_config(ROOT / "configs/smoke.yaml")
    assert smoke["tokenizer"]["vocab_size"] == 1024
    assert smoke["profile"]["id"] == "smoke_test"


def test_default_profile_and_inference_are_identity_neutral() -> None:
    assert DEFAULT_CONFIG["profile"] == {
        "id": "generic",
        "display_name": "Assistant",
        "assistant_label": "Assistant",
    }
    inference = DEFAULT_CONFIG["inference"]
    assert inference["model_system_prompt_files"] == []
    assert inference["prompt"] == "Hello, who are you?"
    serialized = json.dumps(DEFAULT_CONFIG, ensure_ascii=False).casefold()
    assert "hana" not in serialized
    assert "personas/" not in serialized


def test_source_manifest_loads_data_pack_metadata(tmp_path: Path) -> None:
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        """
pack:
  name: spanish-pack
  version: 2
  languages: [es]
  description: Spanish portability fixture.
sources:
  - name: spanish-text
    path: spanish.jsonl
    schema: text
    split: train
    languages: [es]
    sample_rate: 0.5
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "schema_version: 2\ndata:\n  sources_file: sources.yaml\n",
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded["data"]["pack"] == {
        "name": "spanish-pack",
        "version": 2,
        "languages": ("es",),
        "description": "Spanish portability fixture.",
    }
    assert loaded["data"]["sources"][0]["sample_rate"] == 0.5


def test_training_and_evaluation_sources_cannot_share_a_path(tmp_path: Path) -> None:
    shared = tmp_path / "shared.jsonl"
    shared.write_text('{"text":"private test item"}\n', encoding="utf-8")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["sources"] = [
        {"name": "training", "path": str(shared), "schema": "text", "split": "train"},
        {
            "name": "evaluation",
            "path": str(shared),
            "schema": "text",
            "split": "test",
            "stages": ["eval"],
            "tokenizer": False,
            "purpose": "evaluation",
        },
    ]

    with pytest.raises(ValueError, match="same file"):
        validate_config(config)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"tokenizer": True}, "tokenizer: false"),
        ({"stages": ["pretrain"]}, r"stages: \[eval\]"),
        ({"split": "valid"}, "split: test"),
    ],
)
def test_evaluation_source_boundary_is_fail_closed(patch: dict[str, object], message: str) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    source = {
        "name": "evaluation",
        "path": "evaluation.jsonl",
        "schema": "text",
        "split": "test",
        "stages": ["eval"],
        "tokenizer": False,
        "purpose": "evaluation",
    }
    source.update(patch)
    config["data"]["sources"] = [source]

    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_enabled_dpo_requires_explicit_prompt_sources() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dpo"]["enabled"] = True
    config["dpo"]["prompt_sources"] = []

    with pytest.raises(ValueError, match=r"dpo\.prompt_sources"):
        validate_config(config)


def test_enabled_memory_evaluation_requires_a_private_file() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["eval"]["multiturn_memory"]["enabled"] = True

    with pytest.raises(ValueError, match=r"eval\.multiturn_memory\.file"):
        validate_config(config)


def test_enabled_knowledge_pilot_requires_a_private_file() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["eval"]["knowledge_pilot"]["enabled"] = True

    with pytest.raises(ValueError, match=r"eval\.knowledge_pilot\.file"):
        validate_config(config)


def test_knowledge_pilot_cannot_share_a_training_path(tmp_path: Path) -> None:
    shared = tmp_path / "shared.jsonl"
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["train_file"] = str(shared)
    config["eval"]["knowledge_pilot"]["file"] = str(shared)

    with pytest.raises(ValueError, match="same file"):
        validate_config(config)


def test_loaded_config_is_deeply_immutable_and_local_overlay_has_precedence(tmp_path: Path) -> None:
    primary = tmp_path / "config.yaml"
    local = tmp_path / "local.yaml"
    primary.write_text("schema_version: 2\nrun:\n  seed: 7\n", encoding="utf-8")
    local.write_text("schema_version: 2\nrun:\n  seed: 9\n", encoding="utf-8")

    loaded = load_config(primary, local_path=local, run_mode="eval")

    assert loaded["run"]["seed"] == 9
    assert loaded["run"]["mode"] == "eval"
    with pytest.raises(TypeError):
        loaded["run"]["seed"] = 10
    mutable = loaded.mutable_copy()
    mutable["run"]["seed"] = 11
    assert loaded["run"]["seed"] == 9


def test_schema_version_two_is_required(tmp_path: Path) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text("run:\n  mode: auto\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version: 2"):
        load_config(path)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("long_context_experimental", "activation_beacon", "enabled"), "Unimplemented long-context"),
        (("inference", "use_speculative_decoding"), "Speculative decoding"),
        (("quantization", "method"), "quantization.method"),
    ],
)
def test_unimplemented_features_fail_instead_of_silently_falling_back(path: tuple[str, ...], message: str) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "mxfp4" if path[-1] == "method" else True

    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_unknown_optimizer_fails_instead_of_falling_back() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["train"]["optimizer"] = "wishful-gradient-descent"

    with pytest.raises(ValueError, match=r"Unsupported train\.optimizer"):
        validate_config(config)


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"nested": {"left": 1}, "items": [1]}
    override = {"nested": {"right": 2}, "items": [2]}
    result = deep_merge(base, override)
    result["nested"]["left"] = 99
    assert base == {"nested": {"left": 1}, "items": [1]}
    assert override == {"nested": {"right": 2}, "items": [2]}


def test_unknown_yaml_key_fails_fast(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 2\nrun:\n  mod: pretrain\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Unknown configuration key: run\.mod"):
        load_config(path)


def test_rope_requires_even_head_dimension() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["model"].update(
        hidden_size=15,
        num_attention_heads=3,
        num_key_value_heads=1,
    )
    with pytest.raises(ValueError, match="head dimension must be even"):
        validate_config(config)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("train", "learning_rate", 0, "learning rates"),
        ("eval", "batch_size", 0, "eval.batch_size"),
        ("inference", "top_p", 0, "inference.top_p"),
        ("hardware", "target_vram_usage", 1.1, "target_vram_usage"),
    ],
)
def test_invalid_numeric_settings_are_rejected(section: str, key: str, value: object, message: str) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config[section][key] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_invalid_cognitive_architecture_is_rejected() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["cognitive_architecture"]["workspace"]["bottleneck_size"] = config["model"]["hidden_size"] + 1
    with pytest.raises(ValueError, match="workspace bottleneck_size"):
        validate_config(config)

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["cognitive_architecture"]["replay"]["priority_alpha"] = -0.1
    with pytest.raises(ValueError, match="priority_alpha"):
        validate_config(config)

    config["cognitive_architecture"]["replay"]["priority_alpha"] = float("inf")
    with pytest.raises(ValueError, match="priority_alpha"):
        validate_config(config)
