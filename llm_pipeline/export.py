"""Checkpoint export utilities."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from .artifacts import (
    atomic_replace_directory,
    atomic_write_json,
    checkpoint_dataset_type,
    checkpoint_stage,
    export_fingerprint,
)
from .evaluation import resolve_checkpoint


def run_export(config: dict[str, Any], logger: Any) -> Path:
    """Copy a trained checkpoint into an export directory with tokenizer/config."""

    source = resolve_checkpoint(config, config["export"].get("source_checkpoint", "best"))
    dataset_type = checkpoint_dataset_type(source, config["data"].get("dataset_type", "pretrain"))
    target = Path(
        config["export"]["export_it_dir"] if dataset_type in {"sft", "dpo"} else config["export"]["export_non_it_dir"]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-build-", dir=target.parent) as temporary:
        staged = Path(temporary) / target.name
        staged.mkdir()
        for name in ("model.safetensors", "pytorch_model.bin", "model_config.json", "config.yaml"):
            src = source / name
            if src.exists():
                shutil.copy2(src, staged / name)
        if (source / "tokenizer").exists():
            shutil.copytree(source / "tokenizer", staged / "tokenizer")
        if not (staged / "model.safetensors").exists() and not (staged / "pytorch_model.bin").exists():
            raise RuntimeError(f"Checkpoint has no exportable model weights: {source}")
        if not (staged / "model_config.json").exists() or not (staged / "tokenizer").exists():
            raise RuntimeError(f"Checkpoint is missing model config or tokenizer artifacts: {source}")
        atomic_write_json(
            staged / "export_manifest.json",
            {
                "format_version": 1,
                "source_checkpoint": source.name,
                "source_stage": checkpoint_stage(source),
                "dataset_type": dataset_type,
                "export_fingerprint": export_fingerprint(config, source),
            },
        )
        atomic_replace_directory(staged, target)
    logger.info(f"Exported checkpoint from {source} to {target}.")
    return target
