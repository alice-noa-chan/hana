"""Artifact fingerprints and crash-safe file writes.

Pipeline stages use these helpers to distinguish a current artifact from a
stale file that merely happens to exist at the expected path.
"""

from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA_VERSION = 1


def atomic_write_text(path: str | Path, text: str) -> None:
    """Replace a UTF-8 text file only after the complete payload is written."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def strict_json_dumps(data: Any, *, indent: int | None = None) -> str:
    """Serialize standards-compliant JSON and reject NaN/Infinity."""

    return json.dumps(data, ensure_ascii=False, indent=indent, allow_nan=False)


def atomic_write_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, strict_json_dumps(data, indent=indent))


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(strict_json_dumps(row) + "\n" for row in rows)
    atomic_write_text(path, payload)


def atomic_replace_directory(staged: str | Path, target: str | Path) -> None:
    """Swap a fully prepared directory into place, restoring the old one on failure."""

    staged_path = Path(staged)
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    backup = target_path.with_name(f".{target_path.name}.{uuid.uuid4().hex}.backup")
    had_target = target_path.exists()
    try:
        if had_target:
            target_path.replace(backup)
        staged_path.replace(target_path)
    except Exception:
        if had_target and backup.exists() and not target_path.exists():
            backup.replace(target_path)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def _clean_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _clean_config(item)
            for key, item in value.items()
            if not str(key).startswith("_") and key != "__config_path__"
        }
    if isinstance(value, (list, tuple)):
        return [_clean_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _fingerprint(payload: Any) -> str:
    encoded = strict_json_dumps(
        {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "payload": _clean_config(payload)}
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_signature(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {"path": str(file_path), "exists": False}
    stat = file_path.stat()
    return {
        "path": str(file_path.resolve()),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _file_content_signature(path: str | Path) -> dict[str, Any]:
    """Return a content-based signature without retaining any file contents."""

    file_path = Path(path)
    if not file_path.exists():
        return {"path": str(file_path), "exists": False}
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(file_path.resolve()),
        "exists": True,
        "size": file_path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _expand_paths(values: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        pattern = str(value)
        if any(marker in pattern for marker in ("*", "?", "[")):
            paths.extend(Path(match) for match in glob.glob(pattern, recursive=True))
        else:
            paths.append(Path(pattern))
    return sorted(dict.fromkeys(paths), key=lambda item: str(item))


def data_signature(config: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    """Return cheap metadata signatures for files that can feed a stage."""

    data = config["data"]
    configured: list[str | Path] = []
    for key in ("train_file", "valid_file", "test_file", "sources_file"):
        if data.get(key):
            configured.append(data[key])
    for source in data.get("sources") or []:
        if not isinstance(source, dict):
            continue
        purpose = str(source.get("purpose", "training")).lower()
        if purpose == "evaluation" and stage != "eval":
            continue
        if purpose != "evaluation" and stage == "eval":
            continue
        stages = source.get("stages")
        stages = [stages] if isinstance(stages, str) else stages
        if stage == "tokenizer" and not source.get("tokenizer", True):
            continue
        if stage != "tokenizer" and stages and "all" not in stages and stage not in stages:
            continue
        values = source.get("paths", source.get("path", []))
        configured.extend(values if isinstance(values, list) else [values])
    return [_file_signature(path) for path in _expand_paths(value for value in configured if value)]


def checkpoint_fingerprint(checkpoint: str | Path) -> str:
    checkpoint_path = Path(checkpoint)
    files = [
        checkpoint_path / name
        for name in ("model.safetensors", "pytorch_model.bin", "model_config.json", "training_state.pt")
    ]
    return _fingerprint([_file_signature(path) for path in files])


def checkpoint_content_fingerprint(checkpoint: str | Path) -> str:
    """Fingerprint checkpoint files without embedding their directory path."""

    checkpoint_path = Path(checkpoint)
    payload = []
    for name in ("model.safetensors", "pytorch_model.bin", "model_config.json", "training_state.pt"):
        path = checkpoint_path / name
        if not path.exists():
            payload.append({"name": name, "exists": False})
            continue
        stat = path.stat()
        payload.append({"name": name, "exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return _fingerprint(payload)


def checkpoint_stage(checkpoint: str | Path) -> str | None:
    """Read the training stage recorded by a checkpoint.

    New checkpoints carry an explicit manifest.  The path fallback keeps
    legacy ``.../<stage>/best`` checkpoints usable without trusting mutable
    runtime ``data.dataset_type`` state.
    """

    checkpoint_path = Path(checkpoint)
    manifest_path = checkpoint_path / "checkpoint_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        manifest = {}
    stage = manifest.get("stage")
    if stage in {"pretrain", "sft", "dpo"}:
        return str(stage)
    for part in reversed(checkpoint_path.parts):
        if part in {"pretrain", "sft", "dpo"}:
            return part
    return None


def checkpoint_dataset_type(checkpoint: str | Path, default: str = "pretrain") -> str:
    """Return the text-evaluation/export kind implied by model provenance."""

    stage = checkpoint_stage(checkpoint)
    return "sft" if stage in {"sft", "dpo"} else (stage or default)


def configured_checkpoint_path(config: dict[str, Any], value: str | Path, auto_stage: str) -> Path:
    """Resolve an explicit path or the experiment-local ``<stage>/best`` path."""

    if str(value).lower() == "auto":
        return Path(config["run"]["output_dir"]) / str(config["run"]["experiment_name"]) / auto_stage / "best"
    return Path(value)


def checkpoint_is_loadable(checkpoint: str | Path) -> bool:
    checkpoint_path = Path(checkpoint)
    has_weights = (checkpoint_path / "model.safetensors").is_file() or (checkpoint_path / "pytorch_model.bin").is_file()
    basic = checkpoint_path.is_dir() and has_weights and (checkpoint_path / "model_config.json").is_file()
    if not basic:
        return False
    manifest_path = checkpoint_path / "checkpoint_manifest.json"
    if not manifest_path.exists():
        return True
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    recorded = manifest.get("checkpoint_fingerprint")
    return recorded is None or recorded == checkpoint_content_fingerprint(checkpoint_path)


def training_fingerprint(config: dict[str, Any], stage: str) -> str:
    model_config = copy.deepcopy(config["model"])
    # Runtime loading replaces the requested vocabulary size with the actual
    # SentencePiece size, so the tokenizer file signature is authoritative.
    model_config.pop("vocab_size", None)
    upstream: dict[str, Any] | None = None
    if stage == "sft" and config["train"].get("sft_init_checkpoint") is not None:
        init_checkpoint = configured_checkpoint_path(
            config, config["train"]["sft_init_checkpoint"], auto_stage="pretrain"
        )
        upstream = {"sft_init_checkpoint": checkpoint_fingerprint(init_checkpoint)}
    elif stage == "dpo":
        policy_checkpoint = configured_checkpoint_path(config, config["dpo"]["policy_model_path"], "sft")
        reference_checkpoint = configured_checkpoint_path(config, config["dpo"]["reference_model_path"], "sft")
        upstream = {
            "policy_checkpoint": checkpoint_fingerprint(policy_checkpoint),
            "reference_checkpoint": checkpoint_fingerprint(reference_checkpoint),
        }

    dpo_data_files = None
    if stage == "dpo":
        dpo_data_files = [
            _file_signature(config["dpo"][key])
            for key in ("train_file", "valid_file", "test_file")
            if config["dpo"].get(key)
        ]

    payload = {
        "stage": stage,
        "run": {key: config["run"].get(key) for key in ("seed", "deterministic")},
        "data": config["data"],
        "data_files": data_signature(config, stage),
        "tokenizer": config["tokenizer"],
        "tokenizer_model": _file_signature(config["tokenizer"]["model_path"]),
        "model": model_config,
        "mtp": config["mtp"],
        "hybrid_diffusion": config["hybrid_diffusion"],
        "experiments": config["experiments"],
        "cognitive_architecture": config["cognitive_architecture"],
        "train": config["train"],
        "dpo": config["dpo"] if stage == "dpo" else None,
        "dpo_data_files": dpo_data_files,
        "upstream_checkpoints": upstream,
    }
    return _fingerprint(payload)


def completed_stage_checkpoint(config: dict[str, Any], stage: str, name: str = "best") -> Path | None:
    """Return a checkpoint only when the stage's completion proof is current."""

    root = Path(config["run"]["output_dir"]) / str(config["run"]["experiment_name"]) / stage
    candidate = root / name
    best = root / "best"
    if not checkpoint_is_loadable(candidate) or not checkpoint_is_loadable(best):
        return None
    try:
        completion = json.loads((root / "stage_complete.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        completion.get("stage") != stage
        or completion.get("training_fingerprint") != training_fingerprint(config, stage)
        or completion.get("best_checkpoint_fingerprint") != checkpoint_fingerprint(best)
    ):
        return None
    return candidate


def tokenizer_training_fingerprint(config: dict[str, Any]) -> str:
    """Fingerprint every setting/input that can change the trained tokenizer."""

    tokenizer_config = copy.deepcopy(config["tokenizer"])
    tokenizer_config.pop("save_dir", None)
    tokenizer_config.pop("model_path", None)
    return _fingerprint(
        {
            "tokenizer": tokenizer_config,
            "data_files": data_signature(config, "tokenizer"),
            "data_sources": config["data"].get("sources") or [],
        }
    )


def analysis_fingerprint(config: dict[str, Any]) -> str:
    return _fingerprint(
        {
            "data": config["data"],
            "data_files": data_signature(config, "pretrain"),
            "tokenizer_model": _file_signature(config["tokenizer"]["model_path"]),
        }
    )


def evaluation_fingerprint(config: dict[str, Any], checkpoint: str | Path) -> str:
    dataset_type = checkpoint_dataset_type(checkpoint, config["data"].get("dataset_type", "pretrain"))
    knowledge_pilot = config["eval"].get("knowledge_pilot", {})
    knowledge_pilot_file = knowledge_pilot.get("file")
    knowledge_prompt_file = knowledge_pilot.get("prompt_file")
    reasoning_prompt_file = config["reasoning"].get("scratchpad_instruction_file")
    inference_prompt_files = list(config["inference"].get("model_system_prompt_files") or [])
    if config["inference"].get("user_system_prompt_file"):
        inference_prompt_files.append(config["inference"]["user_system_prompt_file"])
    payload = {
        "checkpoint": checkpoint_fingerprint(checkpoint),
        "checkpoint_stage": checkpoint_stage(checkpoint),
        "evaluation_dataset_type": dataset_type,
        "data": config["data"],
        "data_files": data_signature(config, dataset_type),
        "tokenizer_model": _file_signature(config["tokenizer"]["model_path"]),
        "model_limits": {
            "max_seq_len": config["model"]["max_seq_len"],
            "max_position_embeddings": config["model"]["max_position_embeddings"],
        },
        "assistant_only_loss": config["train"].get("assistant_only_loss"),
        "eval": config["eval"],
        "knowledge_pilot_file": (_file_content_signature(knowledge_pilot_file) if knowledge_pilot_file else None),
        "knowledge_prompt_file": (_file_content_signature(knowledge_prompt_file) if knowledge_prompt_file else None),
        "benchmark_denylist": (
            _file_content_signature(config["data_policy"]["benchmark_denylist_path"])
            if config["data_policy"].get("benchmark_denylist_path")
            else None
        ),
        "knowledge_generation": {
            "protocol_version": 1,
            "reasoning": config["reasoning"],
            "reasoning_prompt_file": (
                _file_content_signature(reasoning_prompt_file) if reasoning_prompt_file else None
            ),
            "inference": {
                "generation_strategy": config["inference"].get("generation_strategy"),
                "use_kv_cache": config["inference"].get("use_kv_cache"),
                "model_system_prompt": config["inference"].get("model_system_prompt"),
                "user_system_prompt": config["inference"].get("user_system_prompt"),
                "prompt_files": [_file_content_signature(path) for path in _expand_paths(inference_prompt_files)],
            },
            "special_tokens": config["tokenizer"].get("special_tokens"),
            "hybrid_diffusion": config["hybrid_diffusion"],
        },
    }
    return _fingerprint(payload)


def inference_fingerprint(config: dict[str, Any], checkpoint: str | Path) -> str:
    prompt_files = list(config["inference"].get("model_system_prompt_files") or [])
    if config["inference"].get("user_system_prompt_file"):
        prompt_files.append(config["inference"]["user_system_prompt_file"])
    if config["reasoning"].get("scratchpad_instruction_file"):
        prompt_files.append(config["reasoning"]["scratchpad_instruction_file"])
    payload = {
        "checkpoint": checkpoint_fingerprint(checkpoint),
        "tokenizer_model": _file_signature(config["tokenizer"]["model_path"]),
        "inference": config["inference"],
        "reasoning": config["reasoning"],
        "hybrid_diffusion": config["hybrid_diffusion"],
        "experiments": config["experiments"],
        "cognitive_architecture": config["cognitive_architecture"],
        "prompt_files": [_file_signature(path) for path in _expand_paths(prompt_files)],
        "truncation_policy": config["data"].get("truncation_policy"),
        "max_position_embeddings": config["model"]["max_position_embeddings"],
    }
    return _fingerprint(payload)


def build_rejects_fingerprint(config: dict[str, Any], checkpoint: str | Path) -> str:
    """Fingerprint every input that determines generated preference rejects."""

    prompt_sources = config["dpo"].get("prompt_sources") or []
    input_signature = data_signature(config, "sft") if prompt_sources else _file_signature(config["data"]["train_file"])
    reasoning_prompt_file = config["reasoning"].get("scratchpad_instruction_file")
    prompt_files = list(config["inference"].get("model_system_prompt_files") or [])
    if config["inference"].get("user_system_prompt_file"):
        prompt_files.append(config["inference"]["user_system_prompt_file"])
    return _fingerprint(
        {
            "input": input_signature,
            "prompt_sources": prompt_sources,
            "max_prompt_samples": config["dpo"].get("max_prompt_samples"),
            "checkpoint": checkpoint_fingerprint(checkpoint),
            "prompt_field": config["data"]["prompt_field"],
            "chosen_field": config["data"]["chosen_field"],
            "generation": config["dpo"].get("generate_rejected"),
            "inference": {
                key: value
                for key, value in config["inference"].items()
                if key not in {"model_path", "prompt", "interactive", "token_trace_file"}
            },
            "prompt_files": [_file_content_signature(path) for path in _expand_paths(prompt_files)],
            "reasoning": config["reasoning"],
            "reasoning_prompt_file": (
                _file_content_signature(reasoning_prompt_file) if reasoning_prompt_file else None
            ),
            "hybrid_diffusion": config["hybrid_diffusion"],
            "experiments": config["experiments"],
            "special_tokens": config["tokenizer"].get("special_tokens"),
            "seed": config["run"].get("seed"),
        }
    )


def export_fingerprint(config: dict[str, Any], checkpoint: str | Path) -> str:
    return _fingerprint(
        {
            "checkpoint": checkpoint_fingerprint(checkpoint),
            "checkpoint_stage": checkpoint_stage(checkpoint),
            "dataset_type": checkpoint_dataset_type(checkpoint, config["data"].get("dataset_type", "pretrain")),
            "export": config["export"],
        }
    )


def copy_config_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a safe copy for artifacts that must not share mutable config state."""

    return copy.deepcopy(config[key])
