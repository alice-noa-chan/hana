"""Mode dispatcher for the YAML-driven LLM pipeline."""

from __future__ import annotations

import json
import os
from copy import deepcopy

from .artifacts import analysis_fingerprint
from .config import PipelineConfig, load_config
from .data import analyze_sample_stream, iter_text_samples_from_config, save_json
from .data_governance import enforce_data_policy
from .logging_utils import get_git_commit, hardware_summary, make_experiment_dir, set_seed, setup_logger
from .operations import assert_runtime_preconditions, write_run_manifest
from .progress import select_next_mode


def analyze_data(config: dict, logger) -> None:
    """Analyze dataset statistics and save them beside logs."""

    tokenizer = None
    try:
        from .tokenizer import load_tokenizer

        tokenizer = load_tokenizer(config)
    except Exception as exc:
        logger.info(f"Tokenizer unavailable for token counts; char-level stats only. Reason: {exc}")
    samples = iter_text_samples_from_config(
        config,
        split="train",
        dataset_type=config["data"].get("dataset_type", "pretrain"),
        fallback_path=config["data"]["train_file"],
    )
    try:
        from tqdm import tqdm

        samples = tqdm(samples, desc="analyzing data", unit="sample")
    except Exception:
        pass
    stats = analyze_sample_stream(samples, tokenizer)
    if stats["sample_count"] <= 0:
        raise RuntimeError("Data analysis produced zero usable samples. Check source paths and filters.")
    stats["artifact_fingerprint"] = analysis_fingerprint(config)
    log_dir = logger.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    save_json(log_dir / "data_stats.json", stats)
    logger.info(f"Data statistics: {json.dumps(stats, ensure_ascii=False)}")


def run_mode_once(config: dict, logger) -> None:
    """Execute one concrete mode from the current config."""

    mode = config["run"]["mode"]
    sequence = config["run"].get("sequence", [])
    stage_offset = sequence.index(mode) if mode in sequence else 0
    set_seed(int(config["run"]["seed"]) + stage_offset, bool(config["run"].get("deterministic", False)))
    logger.info(f"Executing mode: {mode}")
    if mode == "train_tokenizer":
        from .tokenizer import train_tokenizer

        train_tokenizer(config, logger)
    elif mode == "analyze_data":
        analyze_data(config, logger)
    elif mode in {"pretrain", "sft"}:
        from .training import train_language_model

        train_language_model(config, logger, mode)
    elif mode == "build_rejects":
        from .inference import build_rejected_responses

        build_rejected_responses(config, logger)
    elif mode == "dpo":
        from .training import train_dpo

        train_dpo(config, logger)
    elif mode == "eval":
        from .evaluation import run_eval

        run_eval(config, logger)
    elif mode == "inference":
        from .inference import run_inference

        run_inference(config, logger)
    elif mode == "export":
        from .export import run_export

        run_export(config, logger)
    elif mode == "quantize":
        from .quantization import run_quantize

        run_quantize(config, logger)
    else:
        raise ValueError(f"Unsupported run mode: {mode}")


def main(
    config_path: str | None = None,
    run_mode: str | None = None,
    auto_continue: bool = False,
    force_run: bool = False,
) -> None:
    """Load config, initialize logs, select progress-aware mode, and run."""

    loaded_config = load_config(config_path, run_mode=run_mode)
    run_pipeline_config(loaded_config, auto_continue=auto_continue, force_run=force_run)


def run_pipeline_config(
    loaded_config: PipelineConfig,
    *,
    auto_continue: bool = False,
    force_run: bool = False,
) -> None:
    """Run an already validated immutable configuration."""

    config = loaded_config.mutable_copy()
    requested_mode = config["run"]["mode"]
    data_modes = {"auto", "train_tokenizer", "analyze_data", "pretrain", "sft", "build_rejects", "dpo", "eval"}
    excluded_sources = enforce_data_policy(config, require_artifacts=requested_mode in data_modes)
    if config["data_policy"].get("enforce", True):
        assert_runtime_preconditions(loaded_config)
    experiment_dir = make_experiment_dir(config)
    logger = setup_logger(config, experiment_dir)
    manifest_path = write_run_manifest(loaded_config, logger.log_dir)
    logger.info(f"Requested run mode: {requested_mode}")
    logger.info(f"Run manifest: {manifest_path}")
    if excluded_sources:
        logger.info(
            f"Data policy excluded {len(excluded_sources)} source(s): {json.dumps(excluded_sources, ensure_ascii=False)}"
        )
    logger.info(f"Config path: {config['__config_path__']}")
    logger.info(f"Git commit: {get_git_commit()}")
    logger.info(f"Initial hardware: {json.dumps(hardware_summary(), ensure_ascii=False)}")

    try:
        while True:
            selected_mode = select_next_mode(config, requested_mode, logger=logger, force_run=force_run)
            if selected_mode is None:
                logger.info("No pending runnable mode remains in run.sequence.")
                return
            if int(os.environ.get("WORLD_SIZE", "1")) > 1 and selected_mode not in {"pretrain", "sft", "dpo"}:
                raise RuntimeError(
                    "torchrun/DDP is supported for training modes only. "
                    "Use --mode pretrain, sft, or dpo, or run non-training stages with one process."
                )
            stage_config = deepcopy(config)
            stage_config["run"]["mode"] = selected_mode
            logger.info(f"Selected run mode: {selected_mode}")
            run_mode_once(stage_config, logger)
            if not auto_continue:
                return
            requested_mode = "auto"
    finally:
        logger.close()


if __name__ == "__main__":
    main()
