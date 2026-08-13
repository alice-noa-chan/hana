"""Schema-v2 configuration loading, validation, and immutable public views."""

from __future__ import annotations

import copy
import glob
import hashlib
import json
import math
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .artifacts import atomic_write_text
from .errors import ConfigError

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 2,
    "profile": {
        "id": "generic",
        "display_name": "Assistant",
        "assistant_label": "Assistant",
    },
    "run": {
        "mode": "auto",
        "sequence": [
            "train_tokenizer",
            "analyze_data",
            "pretrain",
            "sft",
            "build_rejects",
            "dpo",
            "eval",
            "inference",
            "export",
            "quantize",
        ],
        "seed": 42,
        "deterministic": False,
        "output_dir": "./checkpoints",
        "resume": True,
        "experiment_name": "llm_experiment",
        "require_clean_git": False,
    },
    "data": {
        "pack": {
            "name": "default",
            "version": 1,
            "languages": [],
            "description": "",
        },
        "raw_dir": "./train_data",
        "processed_dir": "./train_data/processed",
        "train_file": "./train_data/train.local.jsonl",
        "valid_file": "./train_data/valid.local.jsonl",
        "test_file": "./train_data/test.local.jsonl",
        "format": "jsonl",
        "dataset_type": "pretrain",
        "text_field": "text",
        "prompt_field": "prompt",
        "chosen_field": "chosen",
        "rejected_field": "rejected",
        "messages_field": "messages",
        "reasoning_field": "reasoning",
        "reasoning_mode_field": "reasoning_mode",
        "normalize_nfkc": True,
        "dedup": True,
        "dedup_level": "sample",
        "dedup_backend": "sqlite",
        "hash_split": True,
        "train_ratio": 0.98,
        "valid_ratio": 0.01,
        "test_ratio": 0.01,
        "min_chars": 10,
        "max_chars": 200000,
        "max_samples_per_source": None,
        "strict_sources": True,
        "require_all_training_sources": False,
        "token_cache_dir": "./train_data/token_cache",
        "sequence_packing": True,
        "streaming": False,
        "token_cache_shard_size": 4096,
        "token_cache_log_interval_shards": 25,
        "truncation_policy": "recent",
        "sources_file": "./train_data/sources.local.yaml",
        "sources": [],
    },
    "tokenizer": {
        "type": "sentencepiece",
        "model_type": "bpe",
        "vocab_size": 32000,
        "byte_fallback": True,
        "split_digits": True,
        "normalization_rule_name": "nmt_nfkc",
        "numeric_validation": True,
        "numeric_validation_corpus_samples": 256,
        "character_coverage": 0.9995,
        "input_sentence_size": 10000000,
        "shuffle_input_sentence": True,
        "save_dir": "./train_data/tokenizer",
        "model_path": "./train_data/tokenizer/tokenizer.model",
        "special_tokens": {
            "pad": "<pad>",
            "unk": "<unk>",
            "bos": "<s>",
            "eos": "</s>",
            "user": "<user>",
            "assistant": "<assistant>",
            "system": "<system>",
            "reasoning_off": "<reasoning:off>",
            "reasoning_low": "<reasoning:low>",
            "reasoning_medium": "<reasoning:medium>",
            "reasoning_high": "<reasoning:high>",
            "reasoning_max": "<reasoning:max>",
            "mask": "<mask>",
        },
    },
    "model": {
        "architecture": "decoder_only",
        "vocab_size": 32000,
        "hidden_size": 768,
        "num_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "max_position_embeddings": 4096,
        "max_seq_len": 2048,
        "rope": True,
        "rope_scaling": {"enabled": False, "type": "linear", "factor": 1.0},
        "rope_theta": 10000.0,
        "norm": "rmsnorm",
        "activation": "swiglu",
        "attention_type": "gqa",
        "attention_backend": "auto",
        "sliding_window": {"enabled": False, "window_size": 4096, "layer_pattern": []},
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "residual_dropout": 0.0,
        "embedding_dropout": 0.0,
        "tie_embeddings": True,
        "use_bias": False,
        "gradient_checkpointing": "auto",
        "qk_norm": True,
        "attention_output_gate": False,
        "attention_output_gate_bias": 2.0,
        "logit_softcap": 0.0,
        "initializer_range": 0.02,
    },
    "long_context_experimental": {
        "activation_beacon": {"enabled": False},
        "ring_attention": {"enabled": False},
        "index_share": {"enabled": False, "reuse_group_size": 4},
        "csa": {"enabled": False, "top_k": 256, "compression_ratio": 4},
        "compare_with_dense": True,
    },
    "mtp": {"enabled": False, "num_future_tokens": 2, "loss_weight": 0.2},
    "hybrid_diffusion": {
        "enabled": False,
        "loss_weight": 0.25,
        "mask_probability": 0.3,
        "block_size": 16,
        "denoise_steps": 4,
        "ar_warmup_tokens": 4,
    },
    "experiments": {
        "enabled": False,
        "output_dir": "./experiments",
        "activation_monitor": {
            "enabled": False,
            "modules": ["layers.*"],
            "every_n_calls": 1,
            "max_records": 10000,
            "sample_values": 0,
            "output_file": "activations.jsonl",
        },
        "gradient_monitor": {
            "enabled": False,
            "parameters": ["layers.*"],
            "every_n_steps": 10,
            "max_records": 10000,
            "output_file": "gradients.jsonl",
        },
        "interventions": [],
        "runtime_patches": [],
    },
    "cognitive_architecture": {
        "enabled": False,
        "workspace": {
            "enabled": True,
            "bottleneck_size": 128,
            "every_n_layers": 2,
            "gate_bias": -2.0,
        },
        "predictive_coding": {"enabled": True, "loss_weight": 0.05},
        "homeostasis": {"enabled": True, "target_rms": 0.05, "loss_weight": 0.001},
        "neuromodulation": {
            "enabled": True,
            "ema_decay": 0.95,
            "plasticity_gain": 0.15,
            "fatigue_rate": 0.02,
            "recovery_rate": 0.01,
            "min_lr_scale": 0.7,
            "max_lr_scale": 1.3,
        },
        "replay": {
            "enabled": True,
            "capacity": 64,
            "every_n_steps": 10,
            "min_items": 4,
            "weight": 0.5,
            "priority_alpha": 0.7,
        },
        "memory": {
            "enabled": True,
            "path": "./cognitive_state/memory.json",
            "max_episodes": 256,
            "working_memory_slots": 4,
            "retrieval_top_k": 4,
            "similarity_threshold": 0.78,
            "consolidate_every": 8,
            "max_context_chars": 2000,
            "recency_half_life_hours": 168.0,
            "store_threshold": 0.15,
        },
    },
    "reasoning": {
        "enabled": True,
        "default_mode": "medium",
        "modes": ["off", "low", "medium", "high", "max"],
        "max_reasoning_tokens": 1024,
        "mode_budget_ratios": {"off": 0.0, "low": 0.25, "medium": 0.5, "high": 0.75, "max": 1.0},
        "test_time_compute": {
            "enabled": True,
            "mode": "max",
            "candidates": 3,
            "candidate_temperature": 1.0,
            "candidate_top_p": 0.95,
            "candidate_top_k": 50,
            "selector_max_new_tokens": 8,
            "selector_candidate_max_tokens": 256,
        },
        "scratchpad_instruction": (
            "Reason privately and produce a concise internal scratchpad. Do not write the final response "
            "until the reasoning-off boundary appears. After that boundary, return only the final response "
            "and do not quote or mention the scratchpad."
        ),
        "scratchpad_instruction_file": None,
        "expose_reasoning_trace": False,
        "save_reasoning_trace": False,
    },
    "train": {
        "epochs": 3,
        "max_steps": None,
        "micro_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
        "target_tokens_per_step": 32768,
        "learning_rate": 0.0003,
        "min_learning_rate": 0.00003,
        "weight_decay": 0.1,
        "optimizer": "adamw",
        "fused_adamw": "auto",
        "foreach_optimizer": "auto",
        "compile": "auto",
        "compile_mode": "default",
        "scheduler": "cosine",
        "warmup_steps": None,
        "warmup_ratio": 0.03,
        "max_grad_norm": 1.0,
        "label_smoothing": 0.0,
        "z_loss_weight": 0.0,
        "mixed_precision": "auto",
        "early_stopping": {"enabled": True, "patience": 3, "metric": "valid_loss", "mode": "min"},
        "save_interval_steps": 1000,
        "eval_interval_steps": 1000,
        "log_interval_steps": 10,
        "top_k_checkpoints": 3,
        "nan_inf_policy": "stop",
        "assistant_only_loss": True,
        # ``auto`` initializes a fresh SFT stage from pretrain/best.  Set null
        # only when intentionally training SFT from random weights.
        "sft_init_checkpoint": "auto",
    },
    "hardware": {
        "device": "auto",
        "target_vram_usage": 0.9,
        "auto_batch_size": True,
        "find_executable_batch_size": True,
        "auto_micro_batch_min": 1,
        "auto_micro_batch_max": 1024,
        "auto_batch_growth_factor": 2.0,
        "oom_retry": True,
        "auto_shrink_seq_len": False,
        "distributed": "auto",
        "ddp_backend": "auto",
        "ddp_timeout_minutes": 60,
        "ddp_find_unused_parameters": False,
        "data_parallel": "auto",
        "num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
        "non_blocking_transfer": True,
        "allow_tf32": True,
        "float32_matmul_precision": "high",
    },
    "dpo": {
        "enabled": False,
        "beta": 0.1,
        "policy_model_path": "auto",
        "reference_model_path": "auto",
        "train_file": "./train_data/processed/dpo_rejected.jsonl",
        "valid_file": None,
        "test_file": None,
        "prompt_sources": [],
        "max_prompt_samples": None,
        "loss_type": "sigmoid",
        "best_metric": "valid_dpo_loss",
        "generate_rejected": {"temperature": 0.8, "top_p": 0.95, "top_k": 50, "max_new_tokens": 512},
    },
    "eval": {
        "calculate_perplexity": True,
        "calculate_token_accuracy": True,
        "batch_size": 8,
        "instruction_file": None,
        "long_context_file": None,
        "multiturn_memory": {"enabled": False, "n_turns": 10, "file": None},
        "knowledge_pilot": {
            "enabled": False,
            "file": None,
            "prompt_file": None,
            "item_count": 10,
            "required_correct": 10,
            "choice_labels": ["A", "B", "C", "D"],
            "reasoning_mode": "max",
            "max_new_tokens": 8,
            "require_denylist_coverage": True,
        },
        "long_context": {"enabled": False},
        "checkpoints": ["latest", "best"],
    },
    "inference": {
        "model_path": "./checkpoints/sft/best",
        "model_system_prompt": "",
        "model_system_prompt_files": [],
        "user_system_prompt": "",
        "user_system_prompt_file": None,
        "prompt": "Hello, who are you?",
        "interactive": False,
        "reasoning_mode": "medium",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "max_new_tokens": 1024,
        "min_output_chars": 1,
        "use_kv_cache": True,
        "use_speculative_decoding": False,
        "generation_strategy": "ar",
        "token_trace_file": None,
    },
    "export": {
        "save_safetensors": True,
        "source_checkpoint": "best",
        "export_non_it_dir": "./exports/non_it",
        "export_it_dir": "./exports/it",
        "export_quantized_dir": "./exports/quantized",
    },
    "quantization": {
        "enabled": False,
        "method": "none",
    },
    "data_policy": {
        "enforce": True,
        "use_case": "internal_noncommercial_research",
        "allow_audit_gated_sources": False,
        "source_lock_path": "./train_data/sources.lock.json",
        "audit_path": "./train_data/data_audit.json",
        "benchmark_denylist_path": "./train_data/benchmark_denylist.txt",
        "require_benchmark_denylist": False,
        "max_rejection_hashes_per_source": 25,
    },
    "logging": {
        "log_dir": "./logs",
        "jsonl_log": True,
        "tensorboard": True,
    },
}


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class PipelineConfig(Mapping[str, Any]):
    """Deeply immutable validated configuration exposed by the public API."""

    _data: Mapping[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        return cls(_freeze(copy.deepcopy(data)))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def path(self) -> Path:
        return Path(self._data["__config_path__"])

    @property
    def base_dir(self) -> Path:
        return Path(self._data["__base_dir__"])

    @property
    def digest(self) -> str:
        payload = self.mutable_copy()
        payload.pop("__config_path__", None)
        payload.pop("__base_dir__", None)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def mutable_copy(self) -> dict[str, Any]:
        """Return an explicit working copy for one pipeline invocation."""

        return _thaw(self._data)


@dataclass
class RunContext:
    """Mutable values resolved at runtime without rewriting source configuration."""

    mode: str
    config_digest: str
    tokenizer_vocab_size: int | None = None
    micro_batch_size: int | None = None
    micro_batch_size_per_device: int | None = None
    gradient_accumulation_steps: int | None = None
    parallel_world_size: int = 1
    attention_backend: str | None = None
    precision: str | None = None
    quantization_backend: str | None = None


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""

    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_known_keys(
    user_config: dict[str, Any],
    defaults: dict[str, Any],
    prefix: str = "",
) -> None:
    """Reject misspelled settings instead of silently ignoring them."""

    for key, value in user_config.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if key not in defaults:
            raise ValueError(f"Unknown configuration key: {dotted}")
        default = defaults[key]
        if isinstance(value, dict) and isinstance(default, dict):
            validate_known_keys(value, default, dotted)


def load_sources_file(config: dict[str, Any], base_dir: Path) -> None:
    """Load a local data pack and its sources when the manifest is present."""

    data_cfg = config["data"]
    sources_file = data_cfg.get("sources_file")
    if not sources_file:
        return
    source_path = Path(str(sources_file))
    if not source_path.is_absolute():
        source_path = (base_dir / source_path).resolve()
    if not source_path.exists():
        return

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read data.sources_file manifests.") from exc

    with source_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if isinstance(payload, dict) and "pack" in payload:
        pack = payload["pack"]
        if not isinstance(pack, dict):
            raise ValueError(f"data.sources_file pack must be a mapping: {source_path}")
        unknown_pack_keys = sorted(set(pack) - set(data_cfg["pack"]))
        if unknown_pack_keys:
            raise ValueError(f"data.sources_file pack contains unknown keys: {unknown_pack_keys}")
        data_cfg["pack"] = deep_merge(data_cfg["pack"], pack)
    sources = payload.get("sources", payload) if isinstance(payload, dict) else payload
    if sources is None:
        sources = []
    if not isinstance(sources, list):
        raise ValueError(f"data.sources_file must contain a list or a mapping with a sources list: {source_path}")
    data_cfg["sources"] = copy.deepcopy(sources)


def discover_project_base(config_path: Path) -> Path:
    """Find the project root so config paths never depend on the caller CWD."""

    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "llm_pipeline").is_dir():
            return candidate.resolve()
    return config_path.parent.resolve()


def resolve_config_paths(config: dict[str, Any], base_dir: Path) -> None:
    """Resolve local artifact/data paths against the discovered project root."""

    aliases = {"auto", "best", "latest", "none"}

    def resolved(value: Any, *, allow_alias: bool = False) -> Any:
        if value is None or not isinstance(value, (str, os.PathLike)):
            return value
        text = str(value)
        if not text or (allow_alias and text.lower() in aliases):
            return value
        path = Path(text).expanduser()
        if path.is_absolute():
            return str(path)
        # os.path.abspath preserves glob metacharacters used by source paths.
        return os.path.abspath(base_dir / path)

    path_fields = {
        "run": ("output_dir",),
        "data": (
            "raw_dir",
            "processed_dir",
            "train_file",
            "valid_file",
            "test_file",
            "token_cache_dir",
            "sources_file",
        ),
        "tokenizer": ("save_dir", "model_path"),
        "experiments": ("output_dir",),
    }
    for section, keys in path_fields.items():
        for key in keys:
            config[section][key] = resolved(config[section].get(key))

    for source in config["data"].get("sources", []):
        if not isinstance(source, dict):
            continue
        if "path" in source:
            source["path"] = resolved(source["path"])
        if "paths" in source:
            values = source["paths"] if isinstance(source["paths"], list) else [source["paths"]]
            source["paths"] = [resolved(value) for value in values]
        provenance = source.get("provenance")
        if isinstance(provenance, dict) and provenance.get("evidence_path"):
            provenance["evidence_path"] = resolved(provenance["evidence_path"])

    config["cognitive_architecture"]["memory"]["path"] = resolved(
        config["cognitive_architecture"]["memory"].get("path")
    )
    config["reasoning"]["scratchpad_instruction_file"] = resolved(
        config["reasoning"].get("scratchpad_instruction_file")
    )
    for key in ("train_file", "valid_file", "test_file"):
        config["dpo"][key] = resolved(config["dpo"].get(key))
    for key in ("policy_model_path", "reference_model_path"):
        config["dpo"][key] = resolved(config["dpo"].get(key), allow_alias=True)
    config["train"]["sft_init_checkpoint"] = resolved(config["train"].get("sft_init_checkpoint"), allow_alias=True)
    for key in ("instruction_file", "long_context_file"):
        config["eval"][key] = resolved(config["eval"].get(key))
    config["eval"]["multiturn_memory"]["file"] = resolved(config["eval"]["multiturn_memory"].get("file"))
    config["eval"]["knowledge_pilot"]["file"] = resolved(config["eval"]["knowledge_pilot"].get("file"))
    config["eval"]["knowledge_pilot"]["prompt_file"] = resolved(config["eval"]["knowledge_pilot"].get("prompt_file"))
    config["inference"]["model_path"] = resolved(config["inference"].get("model_path"), allow_alias=True)
    config["inference"]["model_system_prompt_files"] = [
        resolved(value) for value in config["inference"].get("model_system_prompt_files", [])
    ]
    for key in ("user_system_prompt_file", "token_trace_file"):
        config["inference"][key] = resolved(config["inference"].get(key))
    for key in ("export_non_it_dir", "export_it_dir", "export_quantized_dir"):
        config["export"][key] = resolved(config["export"].get(key))
    config["export"]["source_checkpoint"] = resolved(config["export"].get("source_checkpoint"), allow_alias=True)
    for key in ("source_lock_path", "audit_path", "benchmark_denylist_path"):
        config["data_policy"][key] = resolved(config["data_policy"].get(key))


def load_config(
    path: str | os.PathLike[str] | None = None,
    run_mode: str | None = None,
    local_path: str | os.PathLike[str] | None = None,
) -> PipelineConfig:
    """Load schema-v2 YAML, optional local overlay, defaults, and CLI mode."""

    config_path = Path(path or "config.yaml").resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"YAML config not found: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read YAML configs. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}

    if not isinstance(user_config, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {config_path}")
    # Runtime metadata can appear when a resolved in-memory config is saved as
    # a new experiment config.  Recompute it for the new location.
    user_config.pop("__config_path__", None)
    user_config.pop("__base_dir__", None)
    base_dir = discover_project_base(config_path)
    overlay_path: Path | None
    if local_path is not None:
        overlay_path = Path(local_path).resolve()
    elif config_path.name == "config.yaml" and (base_dir / "config.local.yaml").is_file():
        overlay_path = (base_dir / "config.local.yaml").resolve()
    else:
        overlay_path = None
    if overlay_path is not None:
        if not overlay_path.is_file():
            raise FileNotFoundError(f"Local YAML overlay not found: {overlay_path}")
        with overlay_path.open("r", encoding="utf-8") as handle:
            local_config = yaml.safe_load(handle) or {}
        if not isinstance(local_config, dict):
            raise ConfigError(f"Top-level local YAML value must be a mapping: {overlay_path}")
        local_config.pop("__config_path__", None)
        local_config.pop("__base_dir__", None)
        validate_known_keys(local_config, DEFAULT_CONFIG)
        user_config = deep_merge(user_config, local_config)

    if int(user_config.get("schema_version", 0)) != 2:
        raise ConfigError("config.yaml must declare schema_version: 2")
    validate_known_keys(user_config, DEFAULT_CONFIG)

    config = deep_merge(DEFAULT_CONFIG, user_config)
    config["__config_path__"] = str(config_path)
    config["__base_dir__"] = str(base_dir)
    load_sources_file(config, base_dir)
    resolve_config_paths(config, base_dir)

    if run_mode:
        config["run"]["mode"] = run_mode

    validate_config(config)
    return PipelineConfig.from_dict(config)


def validate_config(config: dict[str, Any]) -> None:
    """Fail fast for settings that would otherwise create obscure tensor errors."""

    if int(config.get("schema_version", 0)) != 2:
        raise ConfigError("schema_version must be 2")

    profile = config["profile"]
    for key in ("id", "display_name", "assistant_label"):
        value = profile.get(key)
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError(f"profile.{key} must be a non-empty, trimmed, single-line string.")
    if len(profile["assistant_label"]) > 64:
        raise ValueError("profile.assistant_label must contain at most 64 characters.")

    modes = {
        "train_tokenizer",
        "analyze_data",
        "pretrain",
        "sft",
        "build_rejects",
        "dpo",
        "eval",
        "inference",
        "export",
        "quantize",
        "auto",
    }
    mode = config["run"]["mode"]
    if mode not in modes:
        raise ValueError(f"Unsupported run.mode '{mode}'. Expected one of: {sorted(modes)}")

    sequence = config["run"].get("sequence")
    if not isinstance(sequence, list) or not sequence:
        raise ValueError("run.sequence must be a non-empty list of concrete stage names.")
    invalid_sequence = [item for item in sequence if item not in modes or item == "auto"]
    if invalid_sequence:
        raise ValueError(f"run.sequence contains unsupported stages: {invalid_sequence}")
    if len(sequence) != len(set(sequence)):
        raise ValueError("run.sequence must not contain duplicate stages.")

    data = config["data"]
    pack = data.get("pack")
    if not isinstance(pack, dict):
        raise ValueError("data.pack must be a mapping.")
    if not isinstance(pack.get("name"), str) or not str(pack["name"]).strip():
        raise ValueError("data.pack.name must be a non-empty string.")
    if isinstance(pack.get("version"), bool) or not isinstance(pack.get("version"), int) or pack["version"] <= 0:
        raise ValueError("data.pack.version must be a positive integer.")
    if not isinstance(pack.get("description"), str):
        raise ValueError("data.pack.description must be a string.")
    pack_languages = pack.get("languages")
    if not isinstance(pack_languages, list) or any(
        not isinstance(language, str) or not language.strip() or language != language.strip()
        for language in pack_languages
    ):
        raise ValueError("data.pack.languages must be a list of non-empty, trimmed language tags.")
    if len({language.casefold() for language in pack_languages}) != len(pack_languages):
        raise ValueError("data.pack.languages must not contain duplicate language tags.")
    if str(data["format"]).lower() not in {"jsonl", "jl", "json", "txt", "csv", "tsv"}:
        raise ValueError(f"Unsupported data.format '{data['format']}'.")
    if data.get("dataset_type") not in {"pretrain", "sft", "dpo"}:
        raise ValueError("data.dataset_type must be one of: pretrain, sft, dpo.")
    if not isinstance(data.get("require_all_training_sources", False), bool):
        raise ValueError("data.require_all_training_sources must be true or false.")
    min_chars = int(data["min_chars"])
    max_chars = int(data["max_chars"])
    if min_chars < 0 or max_chars <= 0 or min_chars > max_chars:
        raise ValueError("data.min_chars/max_chars must define a valid non-negative range.")
    if data.get("max_samples_per_source") is not None and int(data["max_samples_per_source"]) <= 0:
        raise ValueError("data.max_samples_per_source must be null or a positive integer.")
    if str(data.get("dedup_backend", "sqlite")).lower() not in {"memory", "sqlite"}:
        raise ValueError("data.dedup_backend must be 'memory' or 'sqlite'.")
    if str(data.get("truncation_policy", "recent")).lower() not in {"recent", "tail", "left"}:
        raise ValueError("data.truncation_policy must preserve recent tokens: recent, tail, or left.")
    if int(data.get("token_cache_shard_size", 4096)) <= 0:
        raise ValueError("data.token_cache_shard_size must be positive.")
    for field_name in ("text_field", "messages_field", "reasoning_field", "reasoning_mode_field"):
        field_value = data.get(field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"data.{field_name} must be a non-empty string.")

    sources = data.get("sources") or []
    if not isinstance(sources, list):
        raise ValueError("data.sources must be a list.")
    allowed_source_keys = {
        "name",
        "path",
        "paths",
        "format",
        "schema",
        "stages",
        "split",
        "tokenizer",
        "tokenizer_max_samples",
        "max_samples",
        "sample_rate",
        "purpose",
        "languages",
        "language_field",
        "domain",
        "text_field",
        "messages_field",
        "reasoning_field",
        "reasoning_mode_field",
        "prompt_field",
        "chosen_field",
        "instruction_field",
        "input_field",
        "output_field",
        "source_lang_field",
        "target_lang_field",
        "source_lang",
        "target_lang",
        "prompt_template",
        "text_template",
        "as_messages",
        # Provenance fields are retained in the local manifest but never
        # embedded in redacted checkpoints.
        "license",
        "license_url",
        "quality_tier",
        "notes",
        "provenance",
        "license_status",
        "allowed_uses",
        "pii_status",
        "child_safety_status",
    }
    source_names: set[str] = set()
    training_paths: set[str] = set()
    evaluation_paths: set[str] = set()
    for index, source in enumerate(sources):
        label = f"data.sources[{index}]"
        if not isinstance(source, dict):
            raise ValueError(f"{label} must be a mapping.")
        unknown = sorted(set(source) - allowed_source_keys)
        if unknown:
            raise ValueError(f"{label} contains unknown keys: {unknown}")
        name = str(source.get("name", "")).strip()
        if not name:
            raise ValueError(f"{label}.name must be a non-empty string.")
        if name in source_names:
            raise ValueError(f"data.sources contains duplicate name: {name}")
        source_names.add(name)
        if ("path" in source) == ("paths" in source):
            raise ValueError(f"{label} must define exactly one of path or paths.")
        paths = source.get("path", source.get("paths"))
        if isinstance(paths, list):
            if not paths or any(not isinstance(item, (str, os.PathLike)) or not str(item) for item in paths):
                raise ValueError(f"{label}.paths must contain non-empty paths.")
        elif not isinstance(paths, (str, os.PathLike)) or not str(paths):
            raise ValueError(f"{label}.path must be a non-empty path.")
        schema = str(source.get("schema", "auto")).lower()
        if schema not in {"auto", "text", "messages", "instruction", "translation"}:
            raise ValueError(f"{label}.schema is unsupported: {schema}")
        split = str(source.get("split", "all")).lower()
        if split not in {"all", "auto", "train", "valid", "test"}:
            raise ValueError(f"{label}.split is unsupported: {split}")
        if not data.get("hash_split", True) and split in {"all", "auto"}:
            raise ValueError(
                f"{label}.split={split} requires data.hash_split=true; "
                "with hash splitting disabled, assign every source an explicit train/valid/test split."
            )
        if source.get("max_samples") is not None and int(source["max_samples"]) <= 0:
            raise ValueError(f"{label}.max_samples must be a positive integer.")
        sample_rate = source.get("sample_rate", 1.0)
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, float)):
            raise ValueError(f"{label}.sample_rate must be a number in (0, 1].")
        if not math.isfinite(float(sample_rate)) or not 0 < float(sample_rate) <= 1:
            raise ValueError(f"{label}.sample_rate must be a finite number in (0, 1].")
        languages = source.get("languages", [])
        language_values = [languages] if isinstance(languages, str) else languages
        if not isinstance(language_values, list) or any(
            not isinstance(language, str) or not language.strip() or language != language.strip()
            for language in language_values
        ):
            raise ValueError(f"{label}.languages must be a language tag or a list of trimmed language tags.")
        if len({language.casefold() for language in language_values}) != len(language_values):
            raise ValueError(f"{label}.languages must not contain duplicates.")
        if source.get("language_field") is not None and (
            not isinstance(source["language_field"], str) or not source["language_field"].strip()
        ):
            raise ValueError(f"{label}.language_field must be a non-empty string.")
        for field_name in ("reasoning_field", "reasoning_mode_field"):
            if source.get(field_name) is not None and (
                not isinstance(source[field_name], str) or not source[field_name].strip()
            ):
                raise ValueError(f"{label}.{field_name} must be a non-empty string.")
        if source.get("domain") is not None and (not isinstance(source["domain"], str) or not source["domain"].strip()):
            raise ValueError(f"{label}.domain must be a non-empty string.")
        stages = source.get("stages")
        stage_values = [stages] if isinstance(stages, str) else stages
        if stages is not None:
            if not isinstance(stage_values, list) or not stage_values:
                raise ValueError(f"{label}.stages must be a non-empty string or list.")
            invalid_stages = sorted(set(stage_values) - {"all", "pretrain", "sft", "dpo", "eval"})
            if invalid_stages:
                raise ValueError(f"{label}.stages contains unsupported values: {invalid_stages}")
        if "provenance" in source and not isinstance(source["provenance"], dict):
            raise ValueError(f"{label}.provenance must be a mapping.")
        provenance = source.get("provenance") if isinstance(source.get("provenance"), dict) else {}
        purpose = str(provenance.get("purpose", source.get("purpose", "training"))).strip().lower()
        if "EVALUATION_ONLY" in str(source.get("license", "")).upper():
            purpose = "evaluation"
        if purpose not in {"training", "evaluation"}:
            raise ValueError(f"{label}.purpose must be training or evaluation.")
        effective_stages = set(stage_values or ["all"])
        if purpose == "evaluation":
            if source.get("tokenizer", False):
                raise ValueError(f"{label} is evaluation-only and must set tokenizer: false.")
            if effective_stages != {"eval"}:
                raise ValueError(f"{label} is evaluation-only and must set stages: [eval].")
            if split != "test":
                raise ValueError(f"{label} is evaluation-only and must set split: test.")
        elif "eval" in effective_stages:
            raise ValueError(f"{label} is a training source and cannot use the eval stage.")
        if schema == "translation":
            for field_name in ("source_lang_field", "target_lang_field"):
                if not isinstance(source.get(field_name), str) or not str(source[field_name]).strip():
                    raise ValueError(f"{label}.{field_name} is required for a translation source.")
            if ("sft" in effective_stages or "all" in effective_stages) and (
                not isinstance(source.get("prompt_template"), str) or not str(source["prompt_template"]).strip()
            ):
                raise ValueError(f"{label}.prompt_template is required for translation SFT.")

        concrete_paths: set[str] = set()
        path_values = paths if isinstance(paths, list) else [paths]
        for value in path_values:
            pattern = str(value)
            matches = (
                glob.glob(pattern, recursive=True) if any(token in pattern for token in ("*", "?", "[")) else [pattern]
            )
            concrete_paths.update(os.path.normcase(os.path.realpath(match)) for match in matches)
        (evaluation_paths if purpose == "evaluation" else training_paths).update(concrete_paths)

    for key in ("train_file", "valid_file", "test_file"):
        value = data.get(key)
        if value:
            training_paths.add(os.path.normcase(os.path.realpath(value)))
        dpo_value = config["dpo"].get(key)
        if dpo_value:
            training_paths.add(os.path.normcase(os.path.realpath(dpo_value)))
    for value in (
        config["eval"].get("instruction_file"),
        config["eval"].get("long_context_file"),
        config["eval"].get("multiturn_memory", {}).get("file"),
        config["eval"].get("knowledge_pilot", {}).get("file"),
        config["eval"].get("knowledge_pilot", {}).get("prompt_file"),
    ):
        if value:
            evaluation_paths.add(os.path.normcase(os.path.realpath(value)))

    overlap = sorted(training_paths & evaluation_paths)
    if overlap:
        raise ValueError(f"Training and evaluation sources resolve to the same file: {overlap[0]}")

    tokenizer = config["tokenizer"]
    if tokenizer["type"] != "sentencepiece":
        raise ValueError("Only tokenizer.type=sentencepiece is supported.")
    if tokenizer["model_type"] not in {"bpe", "unigram", "char", "word"}:
        raise ValueError("tokenizer.model_type must be bpe, unigram, char, or word.")
    if int(tokenizer["vocab_size"]) <= 0:
        raise ValueError("tokenizer.vocab_size must be positive.")
    for key in ("byte_fallback", "split_digits", "numeric_validation", "shuffle_input_sentence"):
        if not isinstance(tokenizer.get(key), bool):
            raise ValueError(f"tokenizer.{key} must be true or false.")
    for key in ("byte_fallback", "split_digits", "numeric_validation"):
        if tokenizer[key] is not True:
            raise ValueError(f"tokenizer.{key} must remain true for the current fail-closed tokenizer contract.")
    normalizer = tokenizer.get("normalization_rule_name")
    if normalizer != "nmt_nfkc":
        raise ValueError("tokenizer.normalization_rule_name must remain nmt_nfkc for the current numeric contract.")
    character_coverage = tokenizer.get("character_coverage")
    if (
        isinstance(character_coverage, bool)
        or not isinstance(character_coverage, (int, float))
        or not math.isfinite(float(character_coverage))
        or not 0.98 <= float(character_coverage) <= 1
    ):
        raise ValueError("tokenizer.character_coverage must be a finite number in [0.98, 1.0].")
    input_sentence_size = tokenizer.get("input_sentence_size")
    if (
        isinstance(input_sentence_size, bool)
        or not isinstance(input_sentence_size, int)
        or input_sentence_size < 0
        or 0 < input_sentence_size <= 100
    ):
        raise ValueError("tokenizer.input_sentence_size must be zero or an integer greater than 100.")
    numeric_samples = tokenizer.get("numeric_validation_corpus_samples")
    if isinstance(numeric_samples, bool) or not isinstance(numeric_samples, int) or numeric_samples < 0:
        raise ValueError("tokenizer.numeric_validation_corpus_samples must be a non-negative integer.")
    if tokenizer["numeric_validation"] and numeric_samples == 0:
        raise ValueError(
            "tokenizer.numeric_validation_corpus_samples must be positive when numeric validation is enabled."
        )
    special_values = list(tokenizer["special_tokens"].values())
    if any(not isinstance(value, str) or not value.strip() for value in special_values):
        raise ValueError("Every tokenizer.special_tokens value must be a non-empty string.")
    if len(special_values) != len(set(special_values)):
        raise ValueError("tokenizer.special_tokens values must be unique.")
    if "mask" not in tokenizer["special_tokens"]:
        raise ValueError("tokenizer.special_tokens.mask is required for hybrid diffusion experiments.")
    minimum_vocab = len(special_values) + (256 if tokenizer.get("byte_fallback") else 0)
    if int(tokenizer["vocab_size"]) < minimum_vocab:
        raise ValueError(
            f"tokenizer.vocab_size must be at least {minimum_vocab} for the configured special tokens/byte fallback."
        )

    model = config["model"]
    if model.get("architecture") != "decoder_only":
        raise ConfigError("Only model.architecture=decoder_only is implemented.")
    if model.get("norm") != "rmsnorm":
        raise ConfigError("Only model.norm=rmsnorm is implemented.")
    if model.get("activation") != "swiglu":
        raise ConfigError("Only model.activation=swiglu is implemented.")
    heads = int(model["num_attention_heads"])
    kv_heads = int(model["num_key_value_heads"])
    hidden = int(model["hidden_size"])
    max_seq_len = int(model["max_seq_len"])
    max_positions = int(model["max_position_embeddings"])
    if max_seq_len <= 0 or max_positions <= 0:
        raise ValueError("model.max_seq_len and model.max_position_embeddings must be positive.")
    if max_seq_len > max_positions:
        raise ValueError("model.max_seq_len must not exceed model.max_position_embeddings.")
    if heads <= 0 or kv_heads <= 0 or hidden <= 0:
        raise ValueError("model hidden size and attention head counts must be positive.")
    if hidden % heads != 0:
        raise ValueError("model.hidden_size must be divisible by model.num_attention_heads.")
    if (hidden // heads) % 2 != 0:
        raise ValueError("The attention head dimension must be even for RoPE.")
    if heads % kv_heads != 0:
        raise ValueError("model.num_attention_heads must be divisible by model.num_key_value_heads.")
    if model["attention_type"] == "mha" and kv_heads != heads:
        raise ValueError("attention_type=mha requires num_key_value_heads == num_attention_heads.")
    if model["attention_type"] == "mqa" and kv_heads != 1:
        raise ValueError("attention_type=mqa requires num_key_value_heads == 1.")
    if int(model["num_layers"]) <= 0:
        raise ValueError("model.num_layers must be positive.")
    for key in ("qk_norm", "attention_output_gate"):
        if not isinstance(model.get(key), bool):
            raise ValueError(f"model.{key} must be true or false.")
    if model["attention_output_gate"] and not model["qk_norm"]:
        raise ValueError("model.attention_output_gate requires model.qk_norm=true.")
    gate_bias = model.get("attention_output_gate_bias")
    if isinstance(gate_bias, bool) or not isinstance(gate_bias, (int, float)) or not math.isfinite(float(gate_bias)):
        raise ValueError("model.attention_output_gate_bias must be a finite number.")
    sliding_window = model.get("sliding_window")
    if not isinstance(sliding_window, dict):
        raise ValueError("model.sliding_window must be a mapping.")
    if not isinstance(sliding_window.get("enabled"), bool):
        raise ValueError("model.sliding_window.enabled must be true or false.")
    window_size = sliding_window.get("window_size")
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("model.sliding_window.window_size must be a positive integer.")
    if sliding_window["enabled"] and window_size > max_positions:
        raise ValueError("model.sliding_window.window_size must not exceed model.max_position_embeddings when enabled.")
    layer_pattern = sliding_window.get("layer_pattern")
    if not isinstance(layer_pattern, list) or any(item not in {"full", "sliding"} for item in layer_pattern):
        raise ValueError("model.sliding_window.layer_pattern must be a list containing only full and sliding.")
    if layer_pattern and not sliding_window["enabled"]:
        raise ValueError("model.sliding_window.layer_pattern requires sliding_window.enabled=true.")
    if layer_pattern and set(layer_pattern) != {"full", "sliding"}:
        raise ValueError("model.sliding_window.layer_pattern must contain both full and sliding when configured.")
    for name in ("dropout", "attention_dropout", "residual_dropout", "embedding_dropout"):
        value = float(model[name])
        if not 0.0 <= value < 1.0:
            raise ValueError(f"model.{name} must be in [0, 1).")
    if model["rope_scaling"].get("enabled") and float(model["rope_scaling"].get("factor", 0)) <= 0:
        raise ValueError("model.rope_scaling.factor must be positive when scaling is enabled.")

    unavailable_long_context = [
        name
        for name in ("activation_beacon", "ring_attention", "index_share", "csa")
        if config["long_context_experimental"][name].get("enabled", False)
    ]
    if unavailable_long_context:
        raise ConfigError(
            "Unimplemented long-context features cannot be enabled: " + ", ".join(unavailable_long_context)
        )

    hybrid = config["hybrid_diffusion"]
    if not math.isfinite(float(hybrid["loss_weight"])) or float(hybrid["loss_weight"]) < 0:
        raise ValueError("hybrid_diffusion.loss_weight must be non-negative.")
    if not 0 < float(hybrid["mask_probability"]) <= 1:
        raise ValueError("hybrid_diffusion.mask_probability must be in (0, 1].")
    if int(hybrid["block_size"]) <= 0 or int(hybrid["block_size"]) > max_seq_len:
        raise ValueError("hybrid_diffusion.block_size must be positive and no larger than model.max_seq_len.")
    if int(hybrid["denoise_steps"]) <= 0 or int(hybrid["ar_warmup_tokens"]) < 0:
        raise ValueError("hybrid diffusion denoise_steps must be positive and ar_warmup_tokens non-negative.")

    cognitive = config["cognitive_architecture"]
    workspace = cognitive["workspace"]
    if int(workspace["bottleneck_size"]) <= 0 or int(workspace["bottleneck_size"]) > hidden:
        raise ValueError("cognitive workspace bottleneck_size must be in [1, model.hidden_size].")
    if int(workspace["every_n_layers"]) <= 0 or not math.isfinite(float(workspace["gate_bias"])):
        raise ValueError("cognitive workspace cadence must be positive and gate_bias finite.")
    predictive = cognitive["predictive_coding"]
    homeostasis = cognitive["homeostasis"]
    if not math.isfinite(float(predictive["loss_weight"])) or float(predictive["loss_weight"]) < 0:
        raise ValueError("cognitive predictive_coding.loss_weight must be finite and non-negative.")
    if (
        not math.isfinite(float(homeostasis["target_rms"]))
        or float(homeostasis["target_rms"]) <= 0
        or not math.isfinite(float(homeostasis["loss_weight"]))
        or float(homeostasis["loss_weight"]) < 0
    ):
        raise ValueError("cognitive homeostasis target_rms must be positive and loss_weight non-negative.")
    neuromodulation = cognitive["neuromodulation"]
    if not 0 <= float(neuromodulation["ema_decay"]) < 1:
        raise ValueError("cognitive neuromodulation.ema_decay must be in [0, 1).")
    if any(
        not math.isfinite(float(neuromodulation[name])) or float(neuromodulation[name]) < 0
        for name in ("plasticity_gain", "fatigue_rate", "recovery_rate")
    ):
        raise ValueError("cognitive neuromodulation gains/rates must be finite and non-negative.")
    min_lr_scale = float(neuromodulation["min_lr_scale"])
    max_lr_scale = float(neuromodulation["max_lr_scale"])
    if not 0 < min_lr_scale <= 1 <= max_lr_scale:
        raise ValueError("cognitive neuromodulation LR scales must satisfy 0 < min <= 1 <= max.")
    replay = cognitive["replay"]
    if int(replay["capacity"]) <= 0 or int(replay["every_n_steps"]) <= 0 or int(replay["min_items"]) <= 0:
        raise ValueError("cognitive replay capacity, cadence, and min_items must be positive.")
    if int(replay["min_items"]) > int(replay["capacity"]):
        raise ValueError("cognitive replay.min_items must not exceed capacity.")
    if (
        not 0 < float(replay["weight"]) <= 1
        or not math.isfinite(float(replay["priority_alpha"]))
        or float(replay["priority_alpha"]) < 0
    ):
        raise ValueError("cognitive replay weight must be in (0, 1] and priority_alpha non-negative.")
    memory = cognitive["memory"]
    if any(
        int(memory[name]) <= 0
        for name in (
            "max_episodes",
            "working_memory_slots",
            "retrieval_top_k",
            "consolidate_every",
            "max_context_chars",
        )
    ):
        raise ValueError("cognitive memory capacities/cadences must be positive.")
    if int(memory["retrieval_top_k"]) > int(memory["working_memory_slots"]):
        raise ValueError("cognitive memory retrieval_top_k must not exceed working_memory_slots.")
    if not 0 < float(memory["similarity_threshold"]) <= 1:
        raise ValueError("cognitive memory similarity_threshold must be in (0, 1].")
    if float(memory["recency_half_life_hours"]) <= 0 or not 0 <= float(memory["store_threshold"]) <= 1:
        raise ValueError("cognitive memory recency/store thresholds are invalid.")

    experiments = config["experiments"]
    monitor = experiments["activation_monitor"]
    if not isinstance(monitor["modules"], list) or not monitor["modules"]:
        raise ValueError("experiments.activation_monitor.modules must be a non-empty list.")
    if int(monitor["every_n_calls"]) <= 0 or int(monitor["max_records"]) <= 0:
        raise ValueError("activation monitor every_n_calls/max_records must be positive.")
    if not 0 <= int(monitor["sample_values"]) <= 256:
        raise ValueError("experiments.activation_monitor.sample_values must be in [0, 256].")
    gradient_monitor = experiments["gradient_monitor"]
    if not isinstance(gradient_monitor["parameters"], list) or not gradient_monitor["parameters"]:
        raise ValueError("experiments.gradient_monitor.parameters must be a non-empty list.")
    if int(gradient_monitor["every_n_steps"]) <= 0 or int(gradient_monitor["max_records"]) <= 0:
        raise ValueError("gradient monitor every_n_steps/max_records must be positive.")
    interventions = experiments.get("interventions") or []
    if not isinstance(interventions, list):
        raise ValueError("experiments.interventions must be a list.")
    for intervention in interventions:
        if not isinstance(intervention, dict) or not intervention.get("module"):
            raise ValueError("Every activation intervention requires a module pattern.")
        kind = intervention.get("kind")
        if kind not in {"zero", "scale", "clamp", "noise", "add_vector", "project_out"}:
            raise ValueError(f"Unsupported activation intervention kind: {kind}")
        if kind in {"add_vector", "project_out"} and not isinstance(intervention.get("vector"), list):
            raise ValueError(f"Activation intervention {kind} requires an inline vector list.")
        if int(intervention.get("start_step", 0)) < 0:
            raise ValueError("Activation intervention start_step must be non-negative.")
        if intervention.get("end_step") is not None and int(intervention["end_step"]) < int(
            intervention.get("start_step", 0)
        ):
            raise ValueError("Activation intervention end_step must not precede start_step.")
        positions = intervention.get("token_positions")
        if positions is not None and (
            not isinstance(positions, list) or any(not isinstance(position, int) for position in positions)
        ):
            raise ValueError("Activation intervention token_positions must be a list of integers.")
        if "value" in intervention and not math.isfinite(float(intervention["value"])):
            raise ValueError("Activation intervention value must be finite.")
    patches = experiments.get("runtime_patches") or []
    if not isinstance(patches, list):
        raise ValueError("experiments.runtime_patches must be a list.")
    from .experiments import RUNTIME_PATCH_KEYS

    runtime_learning_rate = float(config["train"]["learning_rate"])
    runtime_min_learning_rate = float(config["train"]["min_learning_rate"])
    for patch in sorted(patches, key=lambda item: int(item.get("at_step", -1)) if isinstance(item, dict) else -1):
        if not isinstance(patch, dict) or int(patch.get("at_step", -1)) < 0:
            raise ValueError("Every runtime patch requires a non-negative at_step.")
        changes = patch.get("changes")
        if not isinstance(changes, dict) or not changes:
            raise ValueError("Every runtime patch requires a non-empty changes mapping.")
        unsafe = sorted(set(changes) - RUNTIME_PATCH_KEYS)
        if unsafe:
            raise ValueError(f"Runtime patch keys are not shape-safe: {unsafe}")
        for key, value in changes.items():
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Runtime patch {key} must be finite.")
            if key.endswith("dropout") or key == "train.label_smoothing":
                if not 0 <= number < 1:
                    raise ValueError(f"Runtime patch {key} must be in [0, 1).")
            elif key == "hybrid_diffusion.mask_probability":
                if not 0 < number <= 1:
                    raise ValueError("Runtime diffusion mask_probability must be in (0, 1].")
            elif number < 0 or (key in {"train.learning_rate", "train.max_grad_norm"} and number == 0):
                raise ValueError(f"Runtime patch {key} has an invalid value: {value}")
        runtime_learning_rate = float(changes.get("train.learning_rate", runtime_learning_rate))
        runtime_min_learning_rate = float(changes.get("train.min_learning_rate", runtime_min_learning_rate))
        if runtime_min_learning_rate > runtime_learning_rate:
            raise ValueError("Runtime learning rates must preserve min_learning_rate <= learning_rate.")

    ratios = [float(config["data"][name]) for name in ("train_ratio", "valid_ratio", "test_ratio")]
    if any(r < 0 for r in ratios) or sum(ratios) <= 0:
        raise ValueError("data split ratios must be non-negative and have positive sum.")

    train = config["train"]
    hardware = config["hardware"]
    if train["micro_batch_size"] != "auto" and int(train["micro_batch_size"]) <= 0:
        raise ValueError("train.micro_batch_size must be 'auto' or a positive integer.")
    if train["gradient_accumulation_steps"] != "auto" and int(train["gradient_accumulation_steps"]) <= 0:
        raise ValueError("train.gradient_accumulation_steps must be 'auto' or a positive integer.")
    min_batch = int(hardware.get("auto_micro_batch_min", 1))
    max_batch = int(hardware.get("auto_micro_batch_max", 1024))
    if min_batch <= 0 or max_batch < min_batch:
        raise ValueError("hardware.auto_micro_batch_min/max must define a positive increasing range.")
    if float(hardware.get("auto_batch_growth_factor", 2.0)) <= 1.0:
        raise ValueError("hardware.auto_batch_growth_factor must be greater than 1.0.")
    if int(train["epochs"]) <= 0:
        raise ValueError("train.epochs must be positive.")
    if train.get("max_steps") is not None and int(train["max_steps"]) <= 0:
        raise ValueError("train.max_steps must be null or a positive integer.")
    learning_rate = float(train["learning_rate"])
    min_learning_rate = float(train["min_learning_rate"])
    if learning_rate <= 0 or not 0 <= min_learning_rate <= learning_rate:
        raise ValueError("train learning rates must satisfy 0 <= min_learning_rate <= learning_rate.")
    if float(train["weight_decay"]) < 0:
        raise ValueError("train.weight_decay must be non-negative.")
    if not 0 <= float(train["label_smoothing"]) < 1:
        raise ValueError("train.label_smoothing must be in [0, 1).")
    if float(train["max_grad_norm"]) <= 0:
        raise ValueError("train.max_grad_norm must be positive.")
    if train.get("sft_init_checkpoint") is not None and not isinstance(train["sft_init_checkpoint"], str):
        raise ValueError("train.sft_init_checkpoint must be 'auto', a path, or null.")
    for name in ("save_interval_steps", "eval_interval_steps", "log_interval_steps"):
        if int(train[name]) <= 0:
            raise ValueError(f"train.{name} must be positive.")
    if not 0 < float(hardware["target_vram_usage"]) <= 1:
        raise ValueError("hardware.target_vram_usage must be in (0, 1].")
    if int(hardware["num_workers"]) < 0:
        raise ValueError("hardware.num_workers must be non-negative.")

    evaluation = config["eval"]
    if int(evaluation["batch_size"]) <= 0:
        raise ValueError("eval.batch_size must be positive.")
    if not isinstance(evaluation.get("checkpoints"), list) or not evaluation["checkpoints"]:
        raise ValueError("eval.checkpoints must be a non-empty list.")
    memory_eval = evaluation.get("multiturn_memory", {})
    if memory_eval.get("enabled", False) and not memory_eval.get("file"):
        raise ValueError("eval.multiturn_memory.file is required when multiturn memory evaluation is enabled.")
    knowledge_pilot = evaluation.get("knowledge_pilot", {})
    if not isinstance(knowledge_pilot.get("enabled", False), bool):
        raise ValueError("eval.knowledge_pilot.enabled must be true or false.")
    if knowledge_pilot.get("enabled", False) and not knowledge_pilot.get("file"):
        raise ValueError("eval.knowledge_pilot.file is required when the knowledge pilot is enabled.")
    for key in ("file", "prompt_file"):
        value = knowledge_pilot.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"eval.knowledge_pilot.{key} must be a non-empty path or null.")
    raw_item_count = knowledge_pilot.get("item_count", 10)
    raw_required_correct = knowledge_pilot.get("required_correct", raw_item_count)
    if isinstance(raw_item_count, bool) or not isinstance(raw_item_count, int):
        raise ValueError("eval.knowledge_pilot.item_count must be an integer.")
    if isinstance(raw_required_correct, bool) or not isinstance(raw_required_correct, int):
        raise ValueError("eval.knowledge_pilot.required_correct must be an integer.")
    item_count = raw_item_count
    required_correct = raw_required_correct
    if item_count <= 0:
        raise ValueError("eval.knowledge_pilot.item_count must be positive.")
    if not 0 <= required_correct <= item_count:
        raise ValueError("eval.knowledge_pilot.required_correct must be between zero and item_count.")
    choice_labels = knowledge_pilot.get("choice_labels")
    if (
        not isinstance(choice_labels, list)
        or len(choice_labels) < 2
        or len(choice_labels) != len(set(choice_labels))
        or any(
            not isinstance(label, str)
            or len(label) != 1
            or not label.isascii()
            or not label.isalpha()
            or label != label.upper()
            for label in choice_labels
        )
    ):
        raise ValueError("eval.knowledge_pilot.choice_labels must contain unique single-letter ASCII labels.")
    raw_max_new_tokens = knowledge_pilot.get("max_new_tokens", 8)
    if isinstance(raw_max_new_tokens, bool) or not isinstance(raw_max_new_tokens, int):
        raise ValueError("eval.knowledge_pilot.max_new_tokens must be an integer.")
    if raw_max_new_tokens <= 0:
        raise ValueError("eval.knowledge_pilot.max_new_tokens must be positive.")
    if not isinstance(knowledge_pilot.get("require_denylist_coverage", True), bool):
        raise ValueError("eval.knowledge_pilot.require_denylist_coverage must be true or false.")
    if knowledge_pilot.get("enabled", False) and not knowledge_pilot.get("require_denylist_coverage", True):
        raise ValueError("eval.knowledge_pilot.require_denylist_coverage must remain true when the pilot is enabled.")

    inference = config["inference"]
    if inference.get("use_speculative_decoding", False):
        raise ConfigError("Speculative decoding is not implemented; set inference.use_speculative_decoding=false.")
    if float(inference["temperature"]) < 0:
        raise ValueError("inference.temperature must be non-negative.")
    if not 0 < float(inference["top_p"]) <= 1:
        raise ValueError("inference.top_p must be in (0, 1].")
    if int(inference["top_k"]) < 0 or int(inference["max_new_tokens"]) < 0:
        raise ValueError("inference.top_k and max_new_tokens must be non-negative.")
    if int(inference.get("min_output_chars", 1)) < 0:
        raise ValueError("inference.min_output_chars must be non-negative.")
    if float(inference["repetition_penalty"]) <= 0:
        raise ValueError("inference.repetition_penalty must be positive.")
    if inference["generation_strategy"] not in {"ar", "hybrid"}:
        raise ValueError("inference.generation_strategy must be 'ar' or 'hybrid'.")

    dpo = config["dpo"]
    if float(dpo["beta"]) <= 0 or dpo["loss_type"] not in {"sigmoid", "hinge"}:
        raise ValueError("dpo.beta must be positive and dpo.loss_type must be sigmoid or hinge.")
    prompt_sources = dpo.get("prompt_sources") or []
    if not isinstance(prompt_sources, list) or any(not isinstance(value, str) or not value for value in prompt_sources):
        raise ValueError("dpo.prompt_sources must be a list of non-empty source names.")
    if dpo.get("enabled", False) and not prompt_sources:
        raise ValueError("dpo.prompt_sources must explicitly list approved SFT sources when DPO is enabled.")
    configured_source_names = {
        str(source.get("name", "<unnamed>"))
        for source in config["data"].get("sources") or []
        if isinstance(source, dict)
    }
    missing_prompt_sources = sorted(set(prompt_sources) - configured_source_names)
    if dpo.get("enabled", False) and missing_prompt_sources:
        raise ValueError("dpo.prompt_sources contains unknown sources: " + ", ".join(missing_prompt_sources))
    if dpo.get("max_prompt_samples") is not None and int(dpo["max_prompt_samples"]) <= 0:
        raise ValueError("dpo.max_prompt_samples must be null or a positive integer.")
    for key in ("policy_model_path", "reference_model_path"):
        if not isinstance(dpo.get(key), str) or not dpo[key].strip():
            raise ValueError(f"dpo.{key} must be 'auto' or a non-empty checkpoint path.")
    for key in ("train_file", "valid_file", "test_file"):
        if dpo.get(key) is not None and (not isinstance(dpo[key], str) or not dpo[key].strip()):
            raise ValueError(f"dpo.{key} must be a non-empty path or null.")
    rejected = dpo["generate_rejected"]
    if float(rejected["temperature"]) < 0 or not 0 < float(rejected["top_p"]) <= 1:
        raise ValueError("dpo.generate_rejected temperature/top_p settings are invalid.")
    if int(rejected["top_k"]) < 0 or int(rejected["max_new_tokens"]) <= 0:
        raise ValueError("dpo.generate_rejected top_k/max_new_tokens settings are invalid.")

    reasoning = config["reasoning"]
    if not isinstance(reasoning.get("enabled", True), bool):
        raise ValueError("reasoning.enabled must be true or false.")
    modes_list = reasoning.get("modes")
    if (
        not isinstance(modes_list, list)
        or modes_list != ["off", "low", "medium", "high", "max"]
        or any(not isinstance(mode, str) or not mode.strip() for mode in modes_list)
        or len(modes_list) != len(set(modes_list))
        or reasoning.get("default_mode") not in modes_list
    ):
        raise ValueError("reasoning.modes must be exactly: off, low, medium, high, max, and include the default mode.")
    missing_reasoning_tokens = [
        value for value in modes_list if f"reasoning_{value}" not in tokenizer["special_tokens"]
    ]
    if missing_reasoning_tokens:
        raise ValueError(f"Missing special tokens for reasoning modes: {missing_reasoning_tokens}")
    max_reasoning_tokens = reasoning.get("max_reasoning_tokens", 0)
    if isinstance(max_reasoning_tokens, bool) or not isinstance(max_reasoning_tokens, int):
        raise ValueError("reasoning.max_reasoning_tokens must be an integer.")
    if max_reasoning_tokens < 0:
        raise ValueError("reasoning.max_reasoning_tokens must be non-negative.")
    budget_ratios = reasoning.get("mode_budget_ratios")
    if not isinstance(budget_ratios, dict) or set(budget_ratios) != set(modes_list):
        raise ValueError("reasoning.mode_budget_ratios must define every reasoning mode exactly once.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
        for value in budget_ratios.values()
    ):
        raise ValueError("reasoning.mode_budget_ratios values must be finite numbers in [0, 1].")
    if float(budget_ratios.get("off", -1)) != 0:
        raise ValueError("reasoning.mode_budget_ratios.off must be zero.")
    ordered_ratios = [float(budget_ratios[mode]) for mode in modes_list]
    if ordered_ratios != sorted(ordered_ratios) or float(budget_ratios["max"]) != 1.0:
        raise ValueError("reasoning.mode_budget_ratios must increase from off through max, and max must equal one.")
    test_time_compute = reasoning.get("test_time_compute")
    if not isinstance(test_time_compute, dict):
        raise ValueError("reasoning.test_time_compute must be a mapping.")
    if not isinstance(test_time_compute.get("enabled"), bool):
        raise ValueError("reasoning.test_time_compute.enabled must be true or false.")
    if test_time_compute.get("mode") != "max":
        raise ValueError("reasoning.test_time_compute.mode must be max.")
    candidates = test_time_compute.get("candidates")
    if isinstance(candidates, bool) or not isinstance(candidates, int) or not 2 <= candidates <= 26:
        raise ValueError("reasoning.test_time_compute.candidates must be an integer in [2, 26].")
    candidate_temperature = test_time_compute.get("candidate_temperature")
    if (
        isinstance(candidate_temperature, bool)
        or not isinstance(candidate_temperature, (int, float))
        or not math.isfinite(float(candidate_temperature))
        or float(candidate_temperature) < 0
    ):
        raise ValueError("reasoning.test_time_compute.candidate_temperature must be finite and non-negative.")
    candidate_top_p = test_time_compute.get("candidate_top_p")
    if (
        isinstance(candidate_top_p, bool)
        or not isinstance(candidate_top_p, (int, float))
        or not math.isfinite(float(candidate_top_p))
        or not 0 < float(candidate_top_p) <= 1
    ):
        raise ValueError("reasoning.test_time_compute.candidate_top_p must be finite and in (0, 1].")
    for key in ("candidate_top_k", "selector_max_new_tokens", "selector_candidate_max_tokens"):
        value = test_time_compute.get(key)
        minimum = 0 if key == "candidate_top_k" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            qualifier = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"reasoning.test_time_compute.{key} must be a {qualifier} integer.")
    for key in ("scratchpad_instruction",):
        value = reasoning.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"reasoning.{key} must be a non-empty string.")
    reasoning_prompt = reasoning.get("scratchpad_instruction_file")
    if reasoning_prompt is not None and (not isinstance(reasoning_prompt, str) or not reasoning_prompt.strip()):
        raise ValueError("reasoning.scratchpad_instruction_file must be a non-empty path or null.")
    for key in ("expose_reasoning_trace", "save_reasoning_trace"):
        if not isinstance(reasoning.get(key), bool):
            raise ValueError(f"reasoning.{key} must be true or false.")
    if inference.get("reasoning_mode") not in modes_list:
        raise ValueError("inference.reasoning_mode must be one of reasoning.modes.")
    if knowledge_pilot.get("reasoning_mode") not in modes_list:
        raise ValueError("eval.knowledge_pilot.reasoning_mode must be one of reasoning.modes.")

    quantization = config["quantization"]
    if str(quantization.get("method", "none")).lower() not in {"none", "int8"}:
        raise ConfigError("Only quantization.method=none or int8 is implemented.")
    optimizer = str(config["train"].get("optimizer", "")).lower()
    if optimizer not in {"adamw", "muon", "muon_adamw", "hybrid_muon_adamw"}:
        raise ConfigError(f"Unsupported train.optimizer: {optimizer}")
    policy = config["data_policy"]
    if policy.get("use_case") != "internal_noncommercial_research":
        raise ConfigError("Only data_policy.use_case=internal_noncommercial_research is supported.")
    if not isinstance(policy.get("allow_audit_gated_sources", False), bool):
        raise ConfigError("data_policy.allow_audit_gated_sources must be true or false.")
    if not isinstance(policy.get("require_benchmark_denylist", False), bool):
        raise ConfigError("data_policy.require_benchmark_denylist must be true or false.")
    if int(policy.get("max_rejection_hashes_per_source", 0)) < 0:
        raise ConfigError("data_policy.max_rejection_hashes_per_source must be non-negative.")


def config_path(config: Mapping[str, Any]) -> Path:
    """Return the authoritative YAML path used by this run."""

    return Path(config["__config_path__"])


def redacted_config_for_artifact(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a checkpoint-safe config with local data provenance removed."""

    redacted = _thaw(config)
    redacted.pop("__config_path__", None)
    redacted.pop("__base_dir__", None)
    data_cfg = redacted.get("data", {})
    for key in (
        "raw_dir",
        "processed_dir",
        "train_file",
        "valid_file",
        "test_file",
        "token_cache_dir",
        "sources_file",
    ):
        if key in data_cfg:
            data_cfg[key] = "<redacted-local-data>"
    if "sources" in data_cfg:
        data_cfg["sources"] = "<redacted-local-data>"
    if "pack" in data_cfg:
        data_cfg["pack"] = "<redacted-local-data-pack>"
    data_cfg["data_provenance_redacted"] = True
    memory_cfg = redacted.get("cognitive_architecture", {}).get("memory", {})
    if "path" in memory_cfg:
        memory_cfg["path"] = "<redacted-local-state>"
    for key in ("train_file", "valid_file", "test_file"):
        if redacted.get("dpo", {}).get(key) is not None:
            redacted["dpo"][key] = "<redacted-local-data>"
    if redacted.get("dpo", {}).get("prompt_sources"):
        redacted["dpo"]["prompt_sources"] = "<redacted-local-source-names>"
    eval_cfg = redacted.get("eval", {})
    for key in ("instruction_file", "long_context_file"):
        if eval_cfg.get(key) is not None:
            eval_cfg[key] = "<redacted-local-data>"
    if eval_cfg.get("multiturn_memory", {}).get("file") is not None:
        eval_cfg["multiturn_memory"]["file"] = "<redacted-local-data>"
    if eval_cfg.get("knowledge_pilot", {}).get("file") is not None:
        eval_cfg["knowledge_pilot"]["file"] = "<redacted-local-data>"
    if eval_cfg.get("knowledge_pilot", {}).get("prompt_file") is not None:
        eval_cfg["knowledge_pilot"]["prompt_file"] = "<redacted-local-data>"
    policy_cfg = redacted.get("data_policy", {})
    for key in ("source_lock_path", "audit_path", "benchmark_denylist_path"):
        if policy_cfg.get(key) is not None:
            policy_cfg[key] = "<redacted-local-evidence>"
    reasoning_cfg = redacted.get("reasoning", {})
    if reasoning_cfg.get("scratchpad_instruction"):
        reasoning_cfg["scratchpad_instruction"] = "<redacted-local-prompt>"
    if reasoning_cfg.get("scratchpad_instruction_file") is not None:
        reasoning_cfg["scratchpad_instruction_file"] = "<redacted-local-prompt>"
    inference_cfg = redacted.get("inference", {})
    for key in ("model_system_prompt", "user_system_prompt", "prompt"):
        if inference_cfg.get(key):
            inference_cfg[key] = "<redacted-local-prompt>"
    for key in ("model_system_prompt_files", "user_system_prompt_file"):
        value = inference_cfg.get(key)
        if value:
            inference_cfg[key] = "<redacted-local-prompt-paths>"
    return redacted


def save_redacted_config(path: str | os.PathLike[str], config: dict[str, Any]) -> None:
    """Write an artifact config without training-data paths or source names."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to save checkpoint configs.") from exc

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output,
        yaml.safe_dump(redacted_config_for_artifact(config), allow_unicode=True, sort_keys=False),
    )
