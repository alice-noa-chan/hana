from __future__ import annotations

import copy
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_pipeline.config import (
    DEFAULT_CONFIG,
    PipelineConfig,
    config_path,
    redacted_config_for_artifact,
    save_redacted_config,
    validate_config,
)


def assign(path: tuple[str, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(config: dict[str, Any]) -> None:
        target = config
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def changes(**values: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(config: dict[str, Any]) -> None:
        for dotted, value in values.items():
            target = config
            parts = dotted.split("__")
            for key in parts[:-1]:
                target = target[key]
            target[parts[-1]] = value

    return mutate


INVALID_CONFIGS: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
    ("schema", assign(("schema_version",), 1), "schema_version"),
    ("mode", assign(("run", "mode"), "bad"), "Unsupported run.mode"),
    ("sequence-empty", assign(("run", "sequence"), []), "non-empty list"),
    ("sequence-invalid", assign(("run", "sequence"), ["auto"]), "unsupported stages"),
    ("sequence-duplicate", assign(("run", "sequence"), ["pretrain", "pretrain"]), "duplicate"),
    ("format", assign(("data", "format"), "parquet"), "Unsupported data.format"),
    ("dataset", assign(("data", "dataset_type"), "eval"), "data.dataset_type"),
    ("chars", assign(("data", "min_chars"), -1), "min_chars/max_chars"),
    ("source-cap", assign(("data", "max_samples_per_source"), 0), "max_samples_per_source"),
    ("dedup", assign(("data", "dedup_backend"), "redis"), "dedup_backend"),
    ("shard", assign(("data", "token_cache_shard_size"), 0), "token_cache_shard_size"),
    ("tokenizer", assign(("tokenizer", "type"), "other"), "sentencepiece"),
    ("tokenizer-model", assign(("tokenizer", "model_type"), "bad"), "model_type"),
    ("vocab", assign(("tokenizer", "vocab_size"), 0), "vocab_size"),
    ("architecture", assign(("model", "architecture"), "encoder"), "architecture"),
    ("norm", assign(("model", "norm"), "layernorm"), "model.norm"),
    ("activation", assign(("model", "activation"), "relu"), "model.activation"),
    ("sequence-positive", assign(("model", "max_seq_len"), 0), "must be positive"),
    ("sequence-limit", changes(model__max_seq_len=8192), "must not exceed"),
    ("heads-positive", assign(("model", "num_attention_heads"), 0), "head counts"),
    ("hidden-divisible", assign(("model", "hidden_size"), 1001), "must be divisible"),
    (
        "head-even",
        changes(model__hidden_size=12, model__num_attention_heads=4, model__num_key_value_heads=2),
        "head dimension must be even",
    ),
    ("kv-divisible", assign(("model", "num_key_value_heads"), 5), "divisible by model.num_key_value_heads"),
    ("mha", changes(model__attention_type="mha", model__num_key_value_heads=1), "attention_type=mha"),
    ("mqa", changes(model__attention_type="mqa", model__num_key_value_heads=2), "attention_type=mqa"),
    ("layers", assign(("model", "num_layers"), 0), "num_layers"),
    ("dropout", assign(("model", "dropout"), 1), "model.dropout"),
    ("attention-dropout", assign(("model", "attention_dropout"), -0.1), "attention_dropout"),
    ("residual-dropout", assign(("model", "residual_dropout"), 1), "residual_dropout"),
    ("embedding-dropout", assign(("model", "embedding_dropout"), -1), "embedding_dropout"),
    (
        "rope-scale",
        changes(model__rope_scaling__enabled=True, model__rope_scaling__factor=0),
        "rope_scaling.factor",
    ),
    ("hybrid-loss", assign(("hybrid_diffusion", "loss_weight"), math.inf), "loss_weight"),
    ("hybrid-mask", assign(("hybrid_diffusion", "mask_probability"), 0), "mask_probability"),
    ("hybrid-block", assign(("hybrid_diffusion", "block_size"), 0), "block_size"),
    ("hybrid-steps", assign(("hybrid_diffusion", "denoise_steps"), 0), "denoise_steps"),
    (
        "workspace-cadence",
        assign(("cognitive_architecture", "workspace", "every_n_layers"), 0),
        "workspace cadence",
    ),
    (
        "predictive",
        assign(("cognitive_architecture", "predictive_coding", "loss_weight"), -1),
        "predictive_coding",
    ),
    (
        "homeostasis",
        assign(("cognitive_architecture", "homeostasis", "target_rms"), 0),
        "homeostasis",
    ),
    (
        "neuromod-decay",
        assign(("cognitive_architecture", "neuromodulation", "ema_decay"), 1),
        "ema_decay",
    ),
    (
        "neuromod-gain",
        assign(("cognitive_architecture", "neuromodulation", "fatigue_rate"), -1),
        "gains/rates",
    ),
    (
        "neuromod-scale",
        assign(("cognitive_architecture", "neuromodulation", "min_lr_scale"), 2),
        "LR scales",
    ),
    ("replay-capacity", assign(("cognitive_architecture", "replay", "capacity"), 0), "replay capacity"),
    (
        "replay-min",
        changes(cognitive_architecture__replay__capacity=2, cognitive_architecture__replay__min_items=3),
        "must not exceed",
    ),
    ("replay-weight", assign(("cognitive_architecture", "replay", "weight"), 0), "replay weight"),
    (
        "memory-capacity",
        assign(("cognitive_architecture", "memory", "max_episodes"), 0),
        "memory capacities",
    ),
    (
        "memory-retrieval",
        changes(
            cognitive_architecture__memory__working_memory_slots=1,
            cognitive_architecture__memory__retrieval_top_k=2,
        ),
        "retrieval_top_k",
    ),
    (
        "memory-similarity",
        assign(("cognitive_architecture", "memory", "similarity_threshold"), 0),
        "similarity_threshold",
    ),
    (
        "memory-recency",
        assign(("cognitive_architecture", "memory", "recency_half_life_hours"), 0),
        "recency/store",
    ),
    ("activation-modules", assign(("experiments", "activation_monitor", "modules"), []), "modules"),
    (
        "activation-cadence",
        assign(("experiments", "activation_monitor", "every_n_calls"), 0),
        "every_n_calls",
    ),
    (
        "activation-sample",
        assign(("experiments", "activation_monitor", "sample_values"), 257),
        "sample_values",
    ),
    ("gradient-params", assign(("experiments", "gradient_monitor", "parameters"), []), "parameters"),
    (
        "gradient-cadence",
        assign(("experiments", "gradient_monitor", "every_n_steps"), 0),
        "every_n_steps",
    ),
    ("ratios", changes(data__train_ratio=0, data__valid_ratio=0, data__test_ratio=0), "split ratios"),
    ("micro", assign(("train", "micro_batch_size"), 0), "micro_batch_size"),
    ("accum", assign(("train", "gradient_accumulation_steps"), 0), "gradient_accumulation"),
    ("batch-range", assign(("hardware", "auto_micro_batch_min"), 0), "auto_micro_batch"),
    ("batch-growth", assign(("hardware", "auto_batch_growth_factor"), 1), "growth_factor"),
    ("epochs", assign(("train", "epochs"), 0), "train.epochs"),
    ("max-steps", assign(("train", "max_steps"), 0), "train.max_steps"),
    ("learning-rate", assign(("train", "learning_rate"), 0), "learning rates"),
    ("weight-decay", assign(("train", "weight_decay"), -1), "weight_decay"),
    ("smoothing", assign(("train", "label_smoothing"), 1), "label_smoothing"),
    ("grad-norm", assign(("train", "max_grad_norm"), 0), "max_grad_norm"),
    ("sft-path", assign(("train", "sft_init_checkpoint"), 42), "sft_init_checkpoint"),
    ("save-interval", assign(("train", "save_interval_steps"), 0), "save_interval_steps"),
    ("vram", assign(("hardware", "target_vram_usage"), 0), "target_vram_usage"),
    ("workers", assign(("hardware", "num_workers"), -1), "num_workers"),
    ("eval-batch", assign(("eval", "batch_size"), 0), "eval.batch_size"),
    ("eval-checkpoints", assign(("eval", "checkpoints"), []), "eval.checkpoints"),
    ("temperature", assign(("inference", "temperature"), -1), "temperature"),
    ("top-p", assign(("inference", "top_p"), 0), "top_p"),
    ("top-k", assign(("inference", "top_k"), -1), "top_k"),
    ("min-output", assign(("inference", "min_output_chars"), -1), "min_output_chars"),
    ("repetition", assign(("inference", "repetition_penalty"), 0), "repetition_penalty"),
    ("generation", assign(("inference", "generation_strategy"), "beam"), "generation_strategy"),
    ("dpo-beta", assign(("dpo", "beta"), 0), "dpo.beta"),
    ("dpo-policy", assign(("dpo", "policy_model_path"), ""), "policy_model_path"),
    ("dpo-file", assign(("dpo", "train_file"), ""), "dpo.train_file"),
    ("reject-temp", assign(("dpo", "generate_rejected", "temperature"), -1), "temperature/top_p"),
    ("reject-top-k", assign(("dpo", "generate_rejected", "top_k"), -1), "top_k/max_new_tokens"),
    ("reasoning", assign(("reasoning", "modes"), []), "reasoning.modes"),
    ("policy-use", assign(("data_policy", "use_case"), "commercial"), "data_policy.use_case"),
    ("policy-hashes", assign(("data_policy", "max_rejection_hashes_per_source"), -1), "non-negative"),
]


@pytest.mark.parametrize(("_name", "mutate", "message"), INVALID_CONFIGS, ids=[row[0] for row in INVALID_CONFIGS])
def test_invalid_configuration_matrix(_name: str, mutate: Callable[[dict[str, Any]], None], message: str) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    mutate(config)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def base_source() -> dict[str, Any]:
    return {"name": "fixture", "path": "fixture.jsonl", "schema": "text", "split": "train"}


SOURCE_CASES: list[tuple[str, Any, str]] = [
    ("not-list", {"unexpected": True}, "must be a list"),
    ("not-mapping", ["path"], "must be a mapping"),
    ("unknown", [{**base_source(), "mystery": True}], "unknown keys"),
    ("both-paths", [{**base_source(), "paths": ["other"]}], "exactly one"),
    ("empty-paths", [{"name": "x", "paths": []}], "non-empty paths"),
    ("bad-path", [{"name": "x", "path": ""}], "non-empty path"),
    ("schema", [{**base_source(), "schema": "bad"}], "schema is unsupported"),
    ("split", [{**base_source(), "split": "dev"}], "split is unsupported"),
    ("cap", [{**base_source(), "max_samples": 0}], "max_samples"),
    ("stages-empty", [{**base_source(), "stages": []}], "stages must be"),
    ("stages-bad", [{**base_source(), "stages": ["eval"]}], "training source"),
    ("provenance", [{**base_source(), "provenance": "bad"}], "provenance must be"),
]


@pytest.mark.parametrize(("_name", "sources", "message"), SOURCE_CASES, ids=[row[0] for row in SOURCE_CASES])
def test_source_schema_matrix(_name: str, sources: Any, message: str) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["sources"] = sources
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_source_auto_split_requires_hash_splitting() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["hash_split"] = False
    config["data"]["sources"] = [{**base_source(), "split": "auto"}]
    with pytest.raises(ValueError, match="hash_split=true"):
        validate_config(config)


def test_tokenizer_special_token_invariants() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["tokenizer"]["special_tokens"]["user"] = ""
    with pytest.raises(ValueError, match="non-empty"):
        validate_config(config)

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["tokenizer"]["special_tokens"]["user"] = config["tokenizer"]["special_tokens"]["assistant"]
    with pytest.raises(ValueError, match="unique"):
        validate_config(config)

    config = copy.deepcopy(DEFAULT_CONFIG)
    del config["tokenizer"]["special_tokens"]["mask"]
    with pytest.raises(ValueError, match="mask is required"):
        validate_config(config)

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["tokenizer"]["vocab_size"] = 10
    with pytest.raises(ValueError, match="must be at least"):
        validate_config(config)


INTERVENTION_CASES = [
    ("not-list", "bad", "must be a list"),
    ("missing-module", [{}], "requires a module"),
    ("kind", [{"module": "x", "kind": "bad"}], "Unsupported activation"),
    ("vector", [{"module": "x", "kind": "add_vector"}], "inline vector"),
    ("start", [{"module": "x", "kind": "zero", "start_step": -1}], "start_step"),
    ("end", [{"module": "x", "kind": "zero", "start_step": 2, "end_step": 1}], "end_step"),
    ("positions", [{"module": "x", "kind": "zero", "token_positions": ["x"]}], "token_positions"),
    ("value", [{"module": "x", "kind": "scale", "value": math.inf}], "value must be finite"),
]


@pytest.mark.parametrize(("_name", "interventions", "message"), INTERVENTION_CASES)
def test_intervention_validation(_name: str, interventions: Any, message: str) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["experiments"]["interventions"] = interventions
    with pytest.raises(ValueError, match=message):
        validate_config(config)


PATCH_CASES = [
    ("not-list", "bad", "must be a list"),
    ("missing-step", [{}], "non-negative at_step"),
    ("missing-changes", [{"at_step": 0}], "non-empty changes"),
    ("unsafe", [{"at_step": 0, "changes": {"model.hidden_size": 1}}], "not shape-safe"),
    ("finite", [{"at_step": 0, "changes": {"train.learning_rate": math.inf}}], "must be finite"),
    ("dropout", [{"at_step": 0, "changes": {"model.attention_dropout": 1}}], "must be in"),
    (
        "mask",
        [{"at_step": 0, "changes": {"hybrid_diffusion.mask_probability": 0}}],
        "mask_probability",
    ),
    ("positive", [{"at_step": 0, "changes": {"train.max_grad_norm": 0}}], "invalid value"),
    (
        "lr-order",
        [{"at_step": 0, "changes": {"train.learning_rate": 0.00001, "train.min_learning_rate": 0.00002}}],
        "preserve min_learning_rate",
    ),
]


@pytest.mark.parametrize(("_name", "patches", "message"), PATCH_CASES)
def test_runtime_patch_validation(_name: str, patches: Any, message: str) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["experiments"]["runtime_patches"] = patches
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_redacted_artifact_removes_local_paths(tmp_path: Path) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(__config_path__=str(tmp_path / "config.yaml"), __base_dir__=str(tmp_path))
    config["data"]["sources"] = [base_source()]
    config["dpo"]["train_file"] = "private.jsonl"
    config["eval"]["instruction_file"] = "private-eval.jsonl"
    config["eval"]["multiturn_memory"]["file"] = "private-memory.jsonl"
    config["cognitive_architecture"]["memory"]["path"] = "private-memory.json"

    redacted = redacted_config_for_artifact(config)
    assert config_path(config) == tmp_path / "config.yaml"
    assert redacted["data"]["sources"] == "<redacted-local-data>"
    assert redacted["dpo"]["train_file"] == "<redacted-local-data>"
    assert redacted["eval"]["instruction_file"] == "<redacted-local-data>"
    assert redacted["cognitive_architecture"]["memory"]["path"] == "<redacted-local-state>"

    output = tmp_path / "artifact.yaml"
    save_redacted_config(output, config)
    saved = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert saved["data"]["data_provenance_redacted"] is True


def test_pipeline_config_mapping_and_digest_are_stable(tmp_path: Path) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(__config_path__=str(tmp_path / "config.yaml"), __base_dir__=str(tmp_path))
    loaded = PipelineConfig.from_dict(config)
    assert len(loaded) == len(config)
    assert set(iter(loaded)) == set(config)
    assert loaded.path == tmp_path / "config.yaml"
    assert loaded.base_dir == tmp_path
    assert loaded.digest == PipelineConfig.from_dict(copy.deepcopy(config)).digest
