"""Pipeline progress detection for RUN_MODE='auto'.

The runner uses artifact evidence rather than an in-memory flag.  That means a
stopped job can be resumed later: completed stages are skipped, and the next
incomplete runnable stage is selected from run.sequence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import (
    analysis_fingerprint,
    build_rejects_fingerprint,
    checkpoint_dataset_type,
    checkpoint_fingerprint,
    checkpoint_is_loadable,
    completed_stage_checkpoint,
    evaluation_fingerprint,
    export_fingerprint,
    inference_fingerprint,
    tokenizer_training_fingerprint,
    training_fingerprint,
)
from .data import source_manifest_fingerprint

DEFAULT_MODE_SEQUENCE = [
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
]


def experiment_dir(config: dict[str, Any]) -> Path:
    """Return the experiment directory used by checkpoints/logs."""

    return Path(config["run"]["output_dir"]) / str(config["run"]["experiment_name"])


def run_log_dir(config: dict[str, Any]) -> Path:
    """Return the effective log directory, matching setup_logger behavior."""

    log_dir = Path(config["logging"]["log_dir"])
    return log_dir if log_dir.is_absolute() else experiment_dir(config) / log_dir


def checkpoint_dir(config: dict[str, Any], stage: str, name: str = "best") -> Path:
    """Return a stage checkpoint folder."""

    return experiment_dir(config) / stage / name


def source_matches_stage(source: dict[str, Any], stage: str) -> bool:
    purpose = str(source.get("purpose", "training")).lower()
    if purpose == "evaluation" and stage != "eval":
        return False
    if purpose != "evaluation" and stage == "eval":
        return False
    stages = source.get("stages")
    if stages is None:
        return True
    if isinstance(stages, str):
        stages = [stages]
    return "all" in stages or stage in stages


def source_paths(source: dict[str, Any]) -> list[Path]:
    value = source.get("path") or source.get("paths")
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [Path(str(item)) for item in values]


def has_source_for_stage(config: dict[str, Any], stage: str) -> bool:
    sources = config["data"].get("sources") or []
    for source in sources:
        if not isinstance(source, dict) or not source_matches_stage(source, stage):
            continue
        if any(path.is_file() for path in _expanded_source_paths(source)):
            return True
    return False


def has_source_schema(config: dict[str, Any], stage: str, schemas: set[str]) -> bool:
    sources = config["data"].get("sources") or []
    for source in sources:
        if not isinstance(source, dict) or not source_matches_stage(source, stage):
            continue
        if str(source.get("schema", "auto")).lower() in schemas and any(
            path.is_file() for path in _expanded_source_paths(source)
        ):
            return True
    return False


def has_model_weights(path: Path) -> bool:
    """Check whether a checkpoint/export directory contains loadable weights."""

    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()


def _expanded_source_paths(source: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for path in source_paths(source):
        pattern = str(path)
        if any(marker in pattern for marker in ("*", "?", "[")):
            import glob

            paths.extend(Path(match) for match in glob.glob(pattern, recursive=True))
        else:
            paths.append(path)
    return paths


def resolve_checkpoint_artifact(config: dict[str, Any], name: str) -> Path | None:
    path = Path(name)
    if checkpoint_is_loadable(path):
        return path
    for stage in ("dpo", "sft", "pretrain"):
        candidate = completed_stage_checkpoint(config, stage, name)
        if candidate is not None:
            return candidate
    return None


def read_first_json_row(path: str | Path) -> dict[str, Any] | None:
    """Read the first object row cheaply for format/runnability checks."""

    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        return row if isinstance(row, dict) else None
        if suffix == ".json":
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data[0] if isinstance(data[0], dict) else None
            return data if isinstance(data, dict) else None
    except Exception:
        return None
    return None


def mode_is_complete(mode: str, config: dict[str, Any]) -> bool:
    """Return True when current artifacts prove a mode is already complete."""

    if mode == "train_tokenizer":
        tokenizer_dir = Path(config["tokenizer"]["save_dir"])
        if not all(
            (tokenizer_dir / name).exists()
            for name in (
                "tokenizer.model",
                "tokenizer.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
            )
        ):
            return False
        try:
            tokenizer_config = json.loads((tokenizer_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
        except Exception:
            return False
        return (
            tokenizer_config.get("data_sources_fingerprint") == source_manifest_fingerprint(config, "tokenizer")
            and tokenizer_config.get("training_fingerprint") == tokenizer_training_fingerprint(config)
            and int(tokenizer_config.get("target_vocab_size", tokenizer_config.get("vocab_size", 0)))
            == int(config["tokenizer"]["vocab_size"])
        )
    if mode == "analyze_data":
        stats_path = run_log_dir(config) / "data_stats.json"
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return stats.get("artifact_fingerprint") == analysis_fingerprint(config)
    if mode in {"pretrain", "sft", "dpo"}:
        checkpoint = checkpoint_dir(config, mode, "best")
        if not (checkpoint / "training_state.pt").exists() or not has_model_weights(checkpoint):
            return False
        completion_path = checkpoint.parent / "stage_complete.json"
        try:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A checkpoint proves resumability, not completion.  Only the
            # marker written after the normal training-loop exit may advance
            # auto mode to the next stage.
            return False
        if (
            completion.get("stage") != mode
            or completion.get("training_fingerprint") != training_fingerprint(config, mode)
            or completion.get("best_checkpoint_fingerprint") != checkpoint_fingerprint(checkpoint)
        ):
            return False
        manifest_path = checkpoint / "checkpoint_manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return manifest.get("training_fingerprint") == completion.get("training_fingerprint")
    if mode == "build_rejects":
        output = Path(config["dpo"].get("train_file") or (Path(config["data"]["processed_dir"]) / "dpo_rejected.jsonl"))
        checkpoint = resolve_checkpoint_artifact(config, str(config["inference"].get("model_path", "best")))
        if checkpoint is None or not output.is_file():
            return False
        try:
            completion = json.loads(output.with_suffix(output.suffix + ".complete.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return int(completion.get("completed_pairs", 0)) > 0 and completion.get(
            "generation_fingerprint"
        ) == build_rejects_fingerprint(config, checkpoint)
    if mode == "eval":
        summary = run_log_dir(config) / "eval_summary.md"
        results_path = run_log_dir(config) / "eval_results.jsonl"
        if not summary.exists() or not results_path.exists():
            return False
        try:
            rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            return False
        by_checkpoint = {str(row.get("checkpoint", "")): row for row in rows}
        for name in config["eval"].get("checkpoints", ["latest", "best"]):
            checkpoint = resolve_checkpoint_artifact(config, str(name))
            if checkpoint is None:
                return False
            row = by_checkpoint.get(str(checkpoint))
            if not row or row.get("evaluation_fingerprint") != evaluation_fingerprint(config, checkpoint):
                return False
        return True
    if mode == "inference":
        artifact = run_log_dir(config) / "inference.json"
        checkpoint = resolve_checkpoint_artifact(config, str(config["inference"]["model_path"]))
        if checkpoint is None:
            return False
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return payload.get("status") == "ok" and payload.get("inference_fingerprint") == inference_fingerprint(
            config, checkpoint
        )
    if mode == "export":
        checkpoint = resolve_checkpoint_artifact(config, str(config["export"].get("source_checkpoint", "best")))
        if checkpoint is None:
            return False
        dataset_type = checkpoint_dataset_type(checkpoint, config["data"].get("dataset_type", "pretrain"))
        export_dir = Path(
            config["export"]["export_it_dir"]
            if dataset_type in {"sft", "dpo"}
            else config["export"]["export_non_it_dir"]
        )
        if not has_model_weights(export_dir) or not (export_dir / "model_config.json").exists():
            return False
        try:
            manifest = json.loads((export_dir / "export_manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return manifest.get("export_fingerprint") == export_fingerprint(config, checkpoint)
    if mode == "quantize":
        if not config["quantization"].get("enabled", False) or config["quantization"].get("method") == "none":
            return True
        export_dir = Path(config["export"]["export_quantized_dir"])
        checkpoint = resolve_checkpoint_artifact(config, str(config["export"].get("source_checkpoint", "best")))
        if checkpoint is None or not (
            (export_dir / "pytorch_model_int8.bin").exists() or any(export_dir.glob("*.safetensors"))
        ):
            return False
        try:
            report = json.loads((export_dir / "quantization_report.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return report.get("source_fingerprint") == checkpoint_fingerprint(checkpoint)
    return False


def mode_is_runnable(mode: str, config: dict[str, Any]) -> tuple[bool, str]:
    """Return whether auto mode should attempt a stage now."""

    train_file = Path(config["data"]["train_file"])
    row = read_first_json_row(train_file)
    if mode in {"train_tokenizer", "analyze_data", "pretrain"}:
        if config["data"].get("sources"):
            return has_source_for_stage(config, "pretrain"), "no pretrain sources are configured"
        return train_file.exists(), f"missing data.train_file: {train_file}"
    if mode == "sft":
        if config["data"].get("sources"):
            schemas = {"messages", "instruction", "translation", "auto"}
            return has_source_schema(config, "sft", schemas), "no SFT-capable sources are configured"
        has_messages = bool(row and config["data"]["messages_field"] in row)
        dataset_type = config["data"].get("dataset_type") == "sft"
        return has_messages or dataset_type, "training file does not look like SFT messages data"
    if mode == "build_rejects":
        if not config["dpo"].get("enabled", False):
            return False, "DPO disabled"
        prompt_sources = set(config["dpo"].get("prompt_sources") or [])
        if prompt_sources:
            configured = {
                str(source.get("name", "<unnamed>"))
                for source in config["data"].get("sources") or []
                if source_matches_stage(source, "sft")
            }
            missing = sorted(prompt_sources - configured)
            return not missing and has_any_checkpoint(config), (
                "missing DPO prompt sources: " + ", ".join(missing) if missing else "needs an existing checkpoint"
            )
        has_prompt = bool(row and config["data"]["prompt_field"] in row)
        return has_prompt and has_any_checkpoint(config), "needs prompt data and an existing checkpoint"
    if mode == "dpo":
        if not config["dpo"].get("enabled", False):
            return False, "DPO disabled"
        dpo_path = Path(config["dpo"].get("train_file") or config["data"]["train_file"])
        dpo_row = read_first_json_row(dpo_path)
        has_preference = bool(
            dpo_row
            and config["data"]["prompt_field"] in dpo_row
            and config["data"]["chosen_field"] in dpo_row
            and config["data"]["rejected_field"] in dpo_row
        )
        return has_preference, f"DPO file is missing or lacks prompt/chosen/rejected rows: {dpo_path}"
    if mode in {"eval", "inference", "export"}:
        return has_any_checkpoint(config), "needs a trained checkpoint"
    if mode == "quantize":
        enabled = bool(config["quantization"].get("enabled", False)) and config["quantization"].get("method") != "none"
        return enabled and has_any_checkpoint(config), "quantization disabled or no checkpoint exists"
    return True, ""


def has_any_checkpoint(config: dict[str, Any]) -> bool:
    """Check for any best/latest checkpoint across train stages."""

    for stage in ("dpo", "sft", "pretrain"):
        for name in ("best", "latest"):
            path = checkpoint_dir(config, stage, name)
            if (path / "training_state.pt").exists() and has_model_weights(path):
                return True
    return False


def select_next_mode(
    config: dict[str, Any],
    requested_mode: str,
    logger: Any | None = None,
    force_run: bool = False,
) -> str | None:
    """Pick the next incomplete mode.

    If requested_mode is a concrete stage and that stage is complete, scanning
    continues from the following stage.  If requested_mode='auto', scanning
    starts at the beginning of run.sequence.
    """

    sequence = list(config["run"].get("sequence") or DEFAULT_MODE_SEQUENCE)
    if requested_mode == "auto":
        start_index = 0
    else:
        if requested_mode not in sequence:
            raise ValueError(f"RUN_MODE '{requested_mode}' is not in run.sequence: {sequence}")
        start_index = sequence.index(requested_mode)

    for index, mode in enumerate(sequence[start_index:], start=start_index):
        if not force_run and mode_is_complete(mode, config):
            if logger:
                logger.info(f"Auto progress: '{mode}' is already complete; checking next stage.")
            continue
        runnable, reason = mode_is_runnable(mode, config)
        if runnable:
            return mode
        if requested_mode != "auto" and index == start_index and not mode_is_complete(mode, config):
            return mode
        if logger:
            logger.info(f"Auto progress: skipping '{mode}' for now ({reason}).")
    return None
