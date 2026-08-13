"""Checkpoint serialization for decoder models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .artifacts import atomic_write_json
from .model import DecoderOnlyTransformer, build_model


def save_model_config(path: str | Path, model: DecoderOnlyTransformer) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, model.cfg.to_dict())


def load_model_from_checkpoint(
    checkpoint_dir: str | Path, config: dict[str, Any], map_location: str | torch.device = "cpu"
) -> DecoderOnlyTransformer:
    """Load model weights from safetensors when present, otherwise PyTorch bin."""

    model = build_model(config)
    checkpoint = Path(checkpoint_dir)
    safe_path = checkpoint / "model.safetensors"
    bin_path = checkpoint / "pytorch_model.bin"
    if safe_path.exists():
        try:
            from safetensors.torch import load_model

            load_model(model, str(safe_path), strict=True, device=str(map_location))
            return model
        except ImportError as exc:
            raise RuntimeError("safetensors is required to load model.safetensors.") from exc
    if bin_path.exists():
        state = torch.load(bin_path, map_location=map_location)
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {checkpoint}")
    model.load_state_dict(state, strict=True)
    return model
