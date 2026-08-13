"""Immutable-at-construction decoder architecture configuration."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecoderConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    max_seq_len: int
    rope: bool
    rope_scaling: dict[str, Any]
    rope_theta: float
    attention_type: str
    attention_backend: str
    sliding_window: dict[str, Any]
    dropout: float
    attention_dropout: float
    residual_dropout: float
    embedding_dropout: float
    tie_embeddings: bool
    use_bias: bool
    gradient_checkpointing: bool
    qk_norm: bool
    attention_output_gate: bool
    attention_output_gate_bias: float
    logit_softcap: float
    z_loss_weight: float
    initializer_range: float
    mtp_enabled: bool
    mtp_num_future_tokens: int
    mtp_loss_weight: float
    cognitive_enabled: bool
    workspace_enabled: bool
    workspace_bottleneck_size: int
    workspace_every_n_layers: int
    workspace_gate_bias: float
    predictive_coding_enabled: bool
    predictive_coding_loss_weight: float
    homeostasis_enabled: bool
    homeostasis_target_rms: float
    homeostasis_loss_weight: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DecoderConfig:
        model = config["model"]
        mtp = config["mtp"]
        cognitive = config["cognitive_architecture"]
        workspace = cognitive["workspace"]
        predictive = cognitive["predictive_coding"]
        homeostasis = cognitive["homeostasis"]
        train = config.get("train", {})
        return cls(
            vocab_size=int(model["vocab_size"]),
            hidden_size=int(model["hidden_size"]),
            num_layers=int(model["num_layers"]),
            num_attention_heads=int(model["num_attention_heads"]),
            num_key_value_heads=int(model["num_key_value_heads"]),
            max_position_embeddings=int(model["max_position_embeddings"]),
            max_seq_len=int(model["max_seq_len"]),
            rope=bool(model["rope"]),
            rope_scaling=dict(model["rope_scaling"]),
            rope_theta=float(model.get("rope_theta", 10000.0)),
            attention_type=str(model["attention_type"]),
            attention_backend=str(model["attention_backend"]),
            sliding_window=dict(model["sliding_window"]),
            dropout=float(model["dropout"]),
            attention_dropout=float(model["attention_dropout"]),
            residual_dropout=float(model["residual_dropout"]),
            embedding_dropout=float(model.get("embedding_dropout", model["dropout"])),
            tie_embeddings=bool(model["tie_embeddings"]),
            use_bias=bool(model["use_bias"]),
            gradient_checkpointing=resolve_gradient_checkpointing(model),
            qk_norm=bool(model.get("qk_norm", False)),
            attention_output_gate=bool(model.get("attention_output_gate", False)),
            attention_output_gate_bias=float(model.get("attention_output_gate_bias", 2.0)),
            logit_softcap=float(model.get("logit_softcap", 0.0)),
            z_loss_weight=float(train.get("z_loss_weight", 0.0)),
            initializer_range=float(model.get("initializer_range", 0.02)),
            mtp_enabled=bool(mtp["enabled"]),
            mtp_num_future_tokens=int(mtp["num_future_tokens"]),
            mtp_loss_weight=float(mtp["loss_weight"]),
            cognitive_enabled=bool(cognitive["enabled"]),
            workspace_enabled=bool(workspace["enabled"]),
            workspace_bottleneck_size=int(workspace["bottleneck_size"]),
            workspace_every_n_layers=int(workspace["every_n_layers"]),
            workspace_gate_bias=float(workspace["gate_bias"]),
            predictive_coding_enabled=bool(predictive["enabled"]),
            predictive_coding_loss_weight=float(predictive["loss_weight"]),
            homeostasis_enabled=bool(homeostasis["enabled"]),
            homeostasis_target_rms=float(homeostasis["target_rms"]),
            homeostasis_loss_weight=float(homeostasis["loss_weight"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def resolve_gradient_checkpointing(model: dict[str, Any]) -> bool:
    """Resolve the explicit value or the documented architecture heuristic."""

    value = model.get("gradient_checkpointing", False)
    if isinstance(value, str) and value.strip().lower() == "auto":
        hidden = int(model["hidden_size"])
        layers = int(model["num_layers"])
        return hidden >= 1024 or layers >= 16
    return bool(value)


def with_tokenizer_vocab(config: dict[str, Any], vocab_size: int) -> dict[str, Any]:
    """Return a model-build snapshot without mutating the source configuration."""

    runtime_config = copy.deepcopy(config)
    runtime_config["model"]["vocab_size"] = int(vocab_size)
    return runtime_config
