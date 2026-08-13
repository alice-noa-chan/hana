"""Post-training quantization entrypoint."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from .artifacts import atomic_replace_directory, atomic_write_json, checkpoint_fingerprint
from .evaluation import resolve_checkpoint
from .model import build_model
from .model_config import with_tokenizer_vocab
from .model_io import load_model_from_checkpoint, save_model_config
from .tokenizer import load_tokenizer


def dynamically_quantize(model: torch.nn.Module) -> torch.nn.Module:
    """Apply the portable CPU int8 dynamic quantization backend."""

    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def load_dynamic_int8_export(export_dir: str | Path, config: dict[str, Any]) -> torch.nn.Module:
    """Reconstruct and strictly load a dynamic-int8 export."""

    root = Path(export_dir)
    weights = root / "pytorch_model_int8.bin"
    if not weights.is_file():
        raise FileNotFoundError(f"Dynamic int8 weights not found: {weights}")
    model = dynamically_quantize(build_model(config).cpu().eval())
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


def run_quantize(config: dict[str, Any], logger: Any) -> Path | None:
    """Run the explicitly selected portable post-training quantization."""

    q_cfg = config["quantization"]
    if not q_cfg.get("enabled", False) or q_cfg.get("method") == "none":
        logger.info("Quantization disabled by config; nothing to do.")
        return None
    method = str(q_cfg["method"]).lower()
    if method != "int8":
        raise ValueError(f"Unsupported quantization method: {method}")

    tokenizer = load_tokenizer(config)
    model_config = with_tokenizer_vocab(config, tokenizer.vocab_size)
    source = resolve_checkpoint(config, config["export"].get("source_checkpoint", "best"))
    model = load_model_from_checkpoint(source, model_config, map_location="cpu")
    quantized = dynamically_quantize(model.eval())

    target = Path(config["export"]["export_quantized_dir"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-build-", dir=target.parent) as temporary:
        staged = Path(temporary) / target.name
        staged.mkdir()
        torch.save(quantized.state_dict(), staged / "pytorch_model_int8.bin")
        save_model_config(staged / "model_config.json", model)
        if not (source / "tokenizer").is_dir():
            raise RuntimeError(f"Source checkpoint is missing tokenizer artifacts: {source}")
        shutil.copytree(source / "tokenizer", staged / "tokenizer")
        if (source / "config.yaml").is_file():
            shutil.copy2(source / "config.yaml", staged / "config.yaml")
        atomic_write_json(
            staged / "quantization_report.json",
            {
                "format_version": 1,
                "source_checkpoint": str(source),
                "source_fingerprint": checkpoint_fingerprint(source),
                "method": method,
                "loader": "llm_pipeline.quantization.load_dynamic_int8_export",
            },
        )
        # A saved file is not an export until a fresh model can load it.
        loaded = load_dynamic_int8_export(staged, model_config)
        with torch.no_grad():
            probe = loaded(torch.tensor([[tokenizer.bos_id]], dtype=torch.long))["logits"]
        if not torch.isfinite(probe).all():
            raise FloatingPointError("Dynamic int8 round-trip produced non-finite logits.")
        atomic_replace_directory(staged, target)
    logger.info(f"Saved quantized model to {target}.")
    return target
