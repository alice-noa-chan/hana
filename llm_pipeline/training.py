"""Training loops for pretraining, SFT, and DPO."""

from __future__ import annotations

import contextlib
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from .artifacts import (
    atomic_write_json,
    checkpoint_content_fingerprint,
    checkpoint_fingerprint,
    checkpoint_is_loadable,
    configured_checkpoint_path,
    training_fingerprint,
)
from .cognition import AdaptiveNeuromodulator, SurpriseReplayBuffer
from .config import save_redacted_config
from .data import (
    build_token_shard_dataset,
    load_preference_samples,
    load_text_samples,
    load_text_samples_from_config,
    make_torch_dataset,
    safe_perplexity,
    split_samples,
    stable_hash,
    tokenize_training_samples,
)
from .experiments import (
    ActivationExperiment,
    GradientExperiment,
    RuntimePatchController,
    build_masked_diffusion_batch,
)
from .logging_utils import make_experiment_dir, set_seed
from .model import build_model
from .model_config import with_tokenizer_vocab
from .model_io import load_model_from_checkpoint, save_model_config
from .tokenizer import load_tokenizer
from .training_optimizer import OptimizerBundle, WarmupScheduler, build_optimizer
from .training_runtime import (
    ParallelContext,
    choose_amp,
    choose_amp_dtype,
    choose_device,
    configure_torch_performance,
    distributed_any_bool,
    distributed_barrier,
    distributed_mean_float,
    distributed_min_int,
    distributed_sum_float,
    init_parallel_context,
    maybe_destroy_parallel_context,
    runtime_setting_enabled,
)


class EpochRandomSampler(Sampler[int]):
    """Deterministic epoch-addressable sampler for exact mid-epoch resume."""

    def __init__(self, data_source: Any, seed: int) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


def is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def scalar_loss(loss: torch.Tensor) -> torch.Tensor:
    return loss.mean() if loss.ndim > 0 else loss


def loss_sum_and_tokens(loss: torch.Tensor, num_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (sum-of-token-losses, token-count) handling DataParallel vectors.

    The model returns a per-token mean loss and the number of contributing
    tokens.  Multiplying them recovers the token-weighted sum so gradient
    accumulation and distributed reduction stay exact regardless of how many
    tokens each micro-batch or replica contributed.
    """

    if loss.ndim > 0:
        total = (loss * num_tokens.to(loss.dtype)).sum()
        count = num_tokens.sum()
    else:
        count = num_tokens.sum() if num_tokens.ndim > 0 else num_tokens
        total = loss * count.to(loss.dtype)
    return total, count


def total_training_steps(num_batches: int, config: dict[str, Any]) -> int:
    max_steps = config["train"].get("max_steps")
    if max_steps:
        return int(max_steps)
    accum = int(resolve_gradient_accumulation(config, fallback=1))
    return max(1, int(config["train"]["epochs"]) * math.ceil(max(1, num_batches) / accum))


def resolve_gradient_accumulation(config: dict[str, Any], fallback: int = 1) -> int:
    value = config["train"]["gradient_accumulation_steps"]
    if value != "auto":
        return max(1, int(value))
    target_tokens = int(config["train"].get("target_tokens_per_step", 32768))
    micro = config["train"].get("_resolved_micro_batch_size", config["train"]["micro_batch_size"])
    micro_batch = 1 if micro == "auto" else int(micro)
    parallel_world_size = int(config["train"].get("_gradient_parallel_world_size", 1))
    seq_len = int(config["model"]["max_seq_len"])
    tokens_per_micro_step = max(1, micro_batch * seq_len * max(1, parallel_world_size))
    return max(fallback, math.ceil(target_tokens / tokens_per_micro_step))


def resolve_micro_batch_size(config: dict[str, Any], device: torch.device, logger: Any) -> int:
    resolved = config["train"].get("_resolved_micro_batch_size")
    if resolved is not None:
        return max(1, int(resolved))
    value = config["train"]["micro_batch_size"]
    if value != "auto":
        return max(1, int(value))
    if device.type != "cuda":
        logger.info("Auto micro_batch_size selected 1 on CPU.")
        return 1
    free, _total = torch.cuda.mem_get_info(device)
    free_gb = free / (1024**3)
    hidden = int(config["model"]["hidden_size"])
    layers = int(config["model"]["num_layers"])
    seq = int(config["model"]["max_seq_len"])
    # Conservative activation-memory estimate; the OOM retry path still guards
    # real training if the estimate is too optimistic.
    bytes_per_sample = seq * hidden * layers * 12
    target_bytes = free * float(config["hardware"]["target_vram_usage"])
    cap = int(config["hardware"].get("auto_micro_batch_max", 1024))
    batch = max(1, min(cap, int(target_bytes / max(1, bytes_per_sample))))
    logger.info(f"Auto micro_batch_size selected {batch} using free VRAM {free_gb:.2f} GB.")
    return batch


def estimate_micro_batch_size(config: dict[str, Any], device: torch.device, logger: Any) -> int:
    value = config["train"]["micro_batch_size"]
    if value != "auto":
        return max(1, int(value))
    if device.type != "cuda":
        return 1
    free, _ = torch.cuda.mem_get_info(device)
    free_gb = free / (1024**3)
    hidden = int(config["model"]["hidden_size"])
    layers = int(config["model"]["num_layers"])
    seq = int(config["model"]["max_seq_len"])
    bytes_per_sample = seq * hidden * layers * 12
    target_bytes = free * float(config["hardware"]["target_vram_usage"])
    cap = int(config["hardware"].get("auto_micro_batch_max", 1024))
    batch = max(1, min(cap, int(target_bytes / max(1, bytes_per_sample))))
    logger.info(f"Estimated auto micro_batch_size={batch} from free VRAM {free_gb:.2f} GB before executable probe.")
    return batch


def probe_batch_fits(
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    batch_size: int,
    dtype: torch.dtype,
    amp_enabled: bool,
) -> bool:
    """Return whether a max-length training step can backpropagate at batch_size."""

    seq_len = int(config["model"]["max_seq_len"])
    vocab = int(config["model"]["vocab_size"])
    try:
        input_ids = torch.randint(0, max(1, vocab), (batch_size, seq_len), dtype=torch.long, device=device)
        labels = input_ids.clone()
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
            out = model(
                input_ids,
                labels=labels,
                attention_mask=None,
                label_smoothing=float(config["train"]["label_smoothing"]),
            )
            loss = out["loss"]
            if loss is None:
                raise RuntimeError("Model did not return a training loss during batch-size probe.")
            loss = scalar_loss(loss)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return True
    except RuntimeError as exc:
        if is_oom_error(exc):
            return False
        raise
    finally:
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def find_executable_micro_batch_size(
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    logger: Any,
    parallel_context: ParallelContext,
) -> int:
    """Find the largest per-device micro batch that completes forward/backward."""

    train_cfg = config["train"]
    hardware = config["hardware"]
    explicit = train_cfg["micro_batch_size"]
    if explicit != "auto":
        return max(1, int(explicit))
    estimated = estimate_micro_batch_size(config, device, logger)
    if device.type != "cuda" or not bool(hardware.get("auto_batch_size", True)):
        return estimated
    if not bool(hardware.get("find_executable_batch_size", True)):
        return estimated

    dtype, amp_enabled = choose_amp_dtype(device, config)
    cap = max(1, int(hardware.get("auto_micro_batch_max", 1024)))
    min_batch = max(1, int(hardware.get("auto_micro_batch_min", 1)))
    growth = max(2.0, float(hardware.get("auto_batch_growth_factor", 2.0)))
    start = min(cap, max(min_batch, estimated))
    model.train()

    def fits(candidate: int) -> bool:
        ok = probe_batch_fits(model, config, device, candidate, dtype, amp_enabled)
        status = "ok" if ok else "oom"
        logger.info(f"Executable batch probe {status}: per_device_micro_batch={candidate}.")
        return ok

    if not fits(start):
        hi = start
        lo = min_batch
        success = 0
        candidate = max(min_batch, start // 2)
        while candidate >= min_batch:
            if fits(candidate):
                success = candidate
                lo = candidate + 1
                break
            hi = candidate
            if candidate == min_batch:
                break
            candidate = max(min_batch, candidate // 2)
        if success == 0:
            raise RuntimeError(
                "No executable CUDA micro batch size was found. Lower model.max_seq_len/model size or free more VRAM."
            )
    else:
        success = start
        hi = min(cap + 1, max(success + 1, math.ceil(success * growth)))
        while hi <= cap and fits(hi):
            success = hi
            hi = min(cap + 1, max(success + 1, math.ceil(success * growth)))
        lo = success + 1

    high = min(cap, hi - 1)
    while lo <= high:
        mid = (lo + high) // 2
        if fits(mid):
            success = mid
            lo = mid + 1
        else:
            high = mid - 1

    success = distributed_min_int(success, parallel_context, device)
    if parallel_context.is_main:
        logger.info(f"Auto executable per-device micro_batch_size selected {success}.")
    return max(1, success)


def configure_resolved_batching(
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    logger: Any,
    parallel_context: ParallelContext,
) -> None:
    """Resolve micro batch, data parallel scaling, and accumulation together."""

    per_device_micro = find_executable_micro_batch_size(model, config, device, logger, parallel_context)
    data_parallel_size = parallel_context.data_parallel_size
    if data_parallel_size > 1:
        micro_batch = per_device_micro * data_parallel_size
        logger.info(
            f"DataParallel global micro_batch_size={micro_batch} "
            f"({per_device_micro} per GPU x {data_parallel_size} GPUs)."
        )
    else:
        micro_batch = per_device_micro
    config["train"]["_resolved_micro_batch_size"] = micro_batch
    config["train"]["_resolved_micro_batch_size_per_device"] = per_device_micro
    config["train"]["_gradient_parallel_world_size"] = parallel_context.training_world_size
    accum = resolve_gradient_accumulation(config)
    config["train"]["_resolved_gradient_accumulation_steps"] = accum
    seq_len = int(config["model"]["max_seq_len"])
    effective_tokens = micro_batch * seq_len * accum * max(1, parallel_context.training_world_size)
    logger.info(
        "Resolved training batch: "
        f"micro_batch_size={micro_batch}, gradient_accumulation_steps={accum}, "
        f"ddp_world_size={parallel_context.training_world_size}, "
        f"effective_tokens_per_optimizer_step={effective_tokens}."
    )


def configure_single_process_data_parallel(
    config: dict[str, Any],
    device: torch.device,
    logger: Any,
    parallel_context: ParallelContext,
) -> None:
    """Use every visible CUDA device when the run was not launched by torchrun."""

    if parallel_context.distributed or device.type != "cuda":
        return
    if not runtime_setting_enabled(config["hardware"].get("data_parallel", "auto"), default=True):
        return
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return
    devices = list(range(device_count))
    parallel_context.data_parallel_devices = devices
    logger.info(
        "Multiple CUDA devices detected without torchrun; enabling torch.nn.DataParallel "
        f"across devices={devices}. For best scaling, launch with torchrun to use DDP."
    )


def wrap_parallel_model(
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    logger: Any,
    parallel_context: ParallelContext,
) -> torch.nn.Module:
    if parallel_context.distributed:
        if device.type == "cuda":
            wrapped = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=bool(config["hardware"].get("ddp_find_unused_parameters", False)),
            )
        else:
            wrapped = torch.nn.parallel.DistributedDataParallel(
                model,
                find_unused_parameters=bool(config["hardware"].get("ddp_find_unused_parameters", False)),
            )
        if parallel_context.is_main:
            logger.info("Wrapped model with DistributedDataParallel.")
        return wrapped
    if parallel_context.data_parallel_size > 1:
        logger.info(f"Wrapped model with DataParallel on {parallel_context.data_parallel_size} GPUs.")
        return torch.nn.DataParallel(model, device_ids=parallel_context.data_parallel_devices)
    return model


def maybe_compile_model(
    model: torch.nn.Module, config: dict[str, Any], device: torch.device, logger: Any
) -> torch.nn.Module:
    """Optionally wrap the model with torch.compile for faster execution.

    Compilation is requested with train.compile (true/false/auto).  'auto'
    enables it on CUDA only, since CPU compile gains are usually small and the
    first-step compile latency is more disruptive there.  Any failure falls
    back to eager so a missing/old compiler never blocks training.
    """

    experiment_cfg = config.get("experiments", {})
    if experiment_cfg.get("enabled") and (
        experiment_cfg.get("activation_monitor", {}).get("enabled") or experiment_cfg.get("interventions")
    ):
        logger.info("torch.compile disabled because activation hooks/interventions require eager module boundaries.")
        return model
    pref = config["train"].get("compile", "auto")
    lowered = str(pref).lower()
    if lowered in {"false", "0", "no", "off", "none"} or pref is False:
        return model
    if lowered == "auto" and device.type != "cuda":
        return model
    if not hasattr(torch, "compile"):
        logger.info("torch.compile is unavailable in this PyTorch build; using eager model.")
        return model
    try:
        mode = str(config["train"].get("compile_mode", "default"))
        compiled = torch.compile(model, mode=mode)
        logger.info(f"Enabled torch.compile (mode={mode}).")
        return compiled
    except Exception as exc:
        logger.info(f"torch.compile failed ({exc}); continuing with eager model.")
        return model


def same_data_path(left: str | Path, right: str | Path) -> bool:
    """Compare data paths after resolving relative segments."""

    return Path(left).resolve() == Path(right).resolve()


def load_train_valid_samples(config: dict[str, Any], stage: str, logger: Any):
    """Load train/valid samples, splitting one shared file when requested."""

    if config["data"].get("sources"):
        train_samples = load_text_samples_from_config(
            config,
            split="train",
            dataset_type=stage,
            fallback_path=config["data"]["train_file"],
        )
        valid_samples = load_text_samples_from_config(
            config,
            split="valid",
            dataset_type=stage,
            fallback_path=config["data"]["valid_file"],
        )
        if not valid_samples and len(train_samples) > 1:
            splits = split_samples(train_samples, config)
            train_samples = splits["train"] or train_samples[:-1]
            valid_samples = splits["valid"] or splits["test"]
            if not valid_samples:
                valid_samples = [train_samples.pop()]
        logger.info(f"Loaded source manifest for {stage}: train={len(train_samples)}, valid={len(valid_samples)}.")
        return train_samples, valid_samples

    train_file = config["data"]["train_file"]
    valid_file = config["data"]["valid_file"]
    if config["data"].get("hash_split", True) and same_data_path(train_file, valid_file):
        samples = load_text_samples(train_file, config, dataset_type=stage)
        splits = split_samples(samples, config)
        train_samples = list(splits["train"])
        valid_samples = list(splits["valid"])
        if not valid_samples and splits["test"]:
            valid_samples = list(splits["test"])
        if not valid_samples and len(train_samples) > 1:
            valid_samples = [train_samples.pop()]
        if not train_samples and valid_samples:
            train_samples = list(valid_samples)
        logger.info(f"Using hash split from shared data file: train={len(train_samples)}, valid={len(valid_samples)}.")
        return train_samples, valid_samples
    return (
        load_text_samples(train_file, config, dataset_type=stage),
        load_text_samples(valid_file, config, dataset_type=stage),
    )


def build_dataloader(
    samples,
    tokenizer,
    config: dict[str, Any],
    mode: str,
    split: str,
    device: torch.device,
    logger: Any,
    parallel_context: ParallelContext | None = None,
):
    assistant_only = mode == "sft" and bool(config["train"].get("assistant_only_loss", True))
    tokenized = tokenize_training_samples(samples, tokenizer, config, assistant_only_loss=assistant_only)
    if not tokenized:
        raise RuntimeError("No tokenized training samples were produced after filtering.")
    dataset, collate = make_torch_dataset(tokenized, tokenizer.pad_id)
    micro_batch = resolve_micro_batch_size(config, device, logger)
    config["train"]["_resolved_micro_batch_size"] = micro_batch
    num_workers = int(config["hardware"]["num_workers"])
    shuffle = split == "train" and mode in {"pretrain", "sft"}
    sampler = None
    if parallel_context and parallel_context.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=parallel_context.world_size,
            rank=parallel_context.rank,
            shuffle=shuffle,
            drop_last=False,
        )
    elif shuffle:
        sampler = EpochRandomSampler(dataset, int(config["run"]["seed"]))
    kwargs = {
        "batch_size": micro_batch,
        "shuffle": False,
        "sampler": sampler,
        "collate_fn": collate,
        "num_workers": num_workers,
        "pin_memory": bool(config["hardware"]["pin_memory"]) and device.type == "cuda",
        "generator": torch.Generator().manual_seed(int(config["run"]["seed"])),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(config["hardware"]["prefetch_factor"])
        kwargs["persistent_workers"] = bool(config["hardware"].get("persistent_workers", True))
    return DataLoader(dataset, **kwargs), len(tokenized)


def build_dataloader_from_config(
    tokenizer,
    config: dict[str, Any],
    mode: str,
    split: str,
    device: torch.device,
    logger: Any,
    parallel_context: ParallelContext | None = None,
):
    """Build a full-corpus DataLoader from token shards."""

    assistant_only = mode == "sft" and bool(config["train"].get("assistant_only_loss", True))
    build_args = {
        "config": config,
        "tokenizer": tokenizer,
        "split": split,
        "dataset_type": mode,
        "assistant_only_loss": assistant_only,
        "logger": logger,
    }
    if parallel_context and parallel_context.distributed and not parallel_context.is_main:
        # Rank 0 owns cache creation.  Other ranks wait and then open the
        # completed immutable memmaps, preventing concurrent .tmp deletion and
        # corruption on shared storage.
        distributed_barrier(parallel_context)
        dataset, collate, count = build_token_shard_dataset(**build_args)
    else:
        dataset, collate, count = build_token_shard_dataset(**build_args)
        if parallel_context and parallel_context.distributed:
            distributed_barrier(parallel_context)
    micro_batch = resolve_micro_batch_size(config, device, logger)
    config["train"]["_resolved_micro_batch_size"] = micro_batch
    num_workers = int(config["hardware"]["num_workers"])
    shuffle = split == "train"
    sampler = None
    if parallel_context and parallel_context.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=parallel_context.world_size,
            rank=parallel_context.rank,
            shuffle=shuffle,
            drop_last=False,
        )
    elif shuffle:
        sampler = EpochRandomSampler(dataset, int(config["run"]["seed"]))
    kwargs = {
        "batch_size": micro_batch,
        "shuffle": False,
        "sampler": sampler,
        "collate_fn": collate,
        "num_workers": num_workers,
        "pin_memory": bool(config["hardware"]["pin_memory"]) and device.type == "cuda",
        "generator": torch.Generator().manual_seed(int(config["run"]["seed"])),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(config["hardware"]["prefetch_factor"])
        kwargs["persistent_workers"] = bool(config["hardware"].get("persistent_workers", True))
    return DataLoader(dataset, **kwargs), count


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
    non_blocking: bool,
) -> dict[str, torch.Tensor | None]:
    """Move a collated batch (including packing tensors) to the target device."""

    moved: dict[str, torch.Tensor | None] = {}
    for key in ("input_ids", "labels", "attention_mask", "position_ids", "document_ids"):
        value = batch.get(key)
        moved[key] = value.to(device, non_blocking=non_blocking) if value is not None else None
    return moved


def count_batch_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> int:
    """Count non-padding tokens for logging."""

    if attention_mask is None:
        return int(input_ids.numel())
    return int(attention_mask.sum().item())


def save_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: OptimizerBundle | None,
    scheduler: WarmupScheduler | None,
    scaler: torch.cuda.amp.GradScaler | None,
    tokenizer: Any,
    config: dict[str, Any],
    epoch: int,
    step: int,
    metric: float,
    logger: Any,
    cognitive_state: dict[str, Any] | None = None,
    next_batch_index: int = 0,
    best_metric: float | None = None,
    stale_evals: int = 0,
) -> None:
    """Save model weights, training state, config, tokenizer, and RNG state."""

    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    model_to_save = unwrap_model(model)
    save_safe = bool(config["export"].get("save_safetensors", True))
    if save_safe:
        temporary_safe = path / ".model.tmp.safetensors"
        try:
            from safetensors.torch import save_model

            save_model(model_to_save, str(temporary_safe))
            temporary_safe.replace(path / "model.safetensors")
            stale_bin = path / "pytorch_model.bin"
            if stale_bin.exists():
                stale_bin.unlink()
        except Exception as exc:
            temporary_safe.unlink(missing_ok=True)
            raise RuntimeError("export.save_safetensors=true, but safetensors checkpoint writing failed") from exc
    else:
        temporary_bin = path / ".pytorch_model.tmp.bin"
        torch.save(model_to_save.state_dict(), temporary_bin)
        temporary_bin.replace(path / "pytorch_model.bin")
        (path / "model.safetensors").unlink(missing_ok=True)

    numpy_rng_state = np.random.get_state()
    state = {
        "epoch": epoch,
        "next_batch_index": int(next_batch_index),
        "step": step,
        "metric": metric,
        "best_metric": metric if best_metric is None else best_metric,
        "stale_evals": int(stale_evals),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": (
            numpy_rng_state[0],
            numpy_rng_state[1].tolist(),
            int(numpy_rng_state[2]),
            int(numpy_rng_state[3]),
            float(numpy_rng_state[4]),
        ),
        "cognitive_state": cognitive_state,
    }
    temporary_state = path / ".training_state.tmp.pt"
    torch.save(state, temporary_state)
    temporary_state.replace(path / "training_state.pt")
    save_model_config(path / "model_config.json", model_to_save)
    save_redacted_config(path / "config.yaml", config)
    tokenizer.save_metadata(path / "tokenizer", config)
    content_fingerprint = checkpoint_content_fingerprint(path)
    atomic_write_json(
        path / "checkpoint_manifest.json",
        {
            "format_version": 1,
            "stage": path.parent.name,
            "epoch": epoch,
            "step": step,
            "metric": metric if math.isfinite(metric) else None,
            "training_fingerprint": training_fingerprint(config, path.parent.name),
            "checkpoint_fingerprint": content_fingerprint,
        },
    )
    logger.info(f"Saved checkpoint {path} with metric={metric:.6f}.")


def mark_stage_complete(checkpoint_root: Path, config: dict[str, Any], stage: str, step: int) -> None:
    """Atomically distinguish a finished stage from a resumable checkpoint."""

    best = checkpoint_root / "best"
    atomic_write_json(
        checkpoint_root / "stage_complete.json",
        {
            "format_version": 1,
            "stage": stage,
            "step": int(step),
            "training_fingerprint": training_fingerprint(config, stage),
            "best_checkpoint_fingerprint": checkpoint_fingerprint(best),
        },
    )


def resume_checkpoint_is_compatible(path: Path, config: dict[str, Any], stage: str) -> bool:
    """Reject stale or partially-written optimizer checkpoints before loading."""

    if not checkpoint_is_loadable(path) or not (path / "training_state.pt").is_file():
        return False
    try:
        manifest = json.loads((path / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("stage") == stage and manifest.get("training_fingerprint") == training_fingerprint(
        config, stage
    )


def load_resume_state(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: OptimizerBundle,
    scheduler: WarmupScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    logger: Any,
    device: torch.device,
    cognitive_components: dict[str, Any] | None = None,
) -> tuple[int, int, int, float, int]:
    """Restore model, optimizer, scheduler, scaler, and RNG state."""

    path = Path(checkpoint_dir)
    state_path = path / "training_state.pt"
    if not state_path.exists():
        return 0, 0, 0, float("inf"), 0
    if (path / "model.safetensors").exists():
        try:
            from safetensors.torch import load_model

            load_model(model, str(path / "model.safetensors"), strict=True, device=str(device))
        except Exception as exc:
            raise RuntimeError(f"Failed to load safetensors checkpoint {path}: {exc}") from exc
    elif (path / "pytorch_model.bin").exists():
        model.load_state_dict(torch.load(path / "pytorch_model.bin", map_location=device))
    state = torch.load(state_path, map_location=device)
    if state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    if state.get("torch_rng_state") is not None:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if device.type == "cuda" and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda_rng_state"]])
    if state.get("python_rng_state") is not None:
        random.setstate(state["python_rng_state"])
    if state.get("numpy_rng_state") is not None:
        numpy_state = state["numpy_rng_state"]
        np.random.set_state(
            (
                str(numpy_state[0]),
                np.asarray(numpy_state[1], dtype=np.uint32),
                int(numpy_state[2]),
                int(numpy_state[3]),
                float(numpy_state[4]),
            )
        )
    saved_cognitive = state.get("cognitive_state")
    if isinstance(saved_cognitive, dict) and cognitive_components:
        for name, component in cognitive_components.items():
            component_state = saved_cognitive.get(name)
            if isinstance(component_state, dict) and hasattr(component, "load_state_dict"):
                component.load_state_dict(component_state)
    logger.info(
        f"Resumed from {path} at epoch={state.get('epoch')} "
        f"batch={state.get('next_batch_index', 0)} step={state.get('step')}."
    )
    return (
        int(state.get("epoch", 0)),
        int(state.get("next_batch_index", 0)),
        int(state.get("step", 0)),
        float(state.get("best_metric", state.get("metric", float("inf")))),
        int(state.get("stale_evals", 0)),
    )


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    parallel_context: ParallelContext | None = None,
    desc: str = "eval",
) -> dict[str, float]:
    """Compute validation loss, perplexity, and token accuracy.

    Loss is accumulated as a token-weighted sum so the result is independent of
    batch boundaries.  Under DDP every rank evaluates its shard and the partial
    sums are all-reduced, which keeps every process busy instead of idling
    while rank 0 evaluates alone.
    """

    model.eval()
    dtype, amp_enabled = choose_amp_dtype(device, config)
    non_blocking = bool(config["hardware"].get("non_blocking_transfer", True)) and device.type == "cuda"
    is_main = parallel_context.is_main if parallel_context else True
    objective_sum = 0.0
    ce_sum = 0.0
    token_count = 0.0
    correct = 0
    total = 0
    try:
        from tqdm import tqdm

        iterator = tqdm(dataloader, desc=desc, disable=not is_main, leave=False)
    except Exception:
        iterator = dataloader
    for batch in iterator:
        moved = move_batch_to_device(batch, device, non_blocking)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
            out = model(
                moved["input_ids"],
                labels=moved["labels"],
                attention_mask=moved["attention_mask"],
                position_ids=moved["position_ids"],
                document_ids=moved["document_ids"],
            )
        if out["loss"] is not None:
            objective_batch_sum, batch_tokens = loss_sum_and_tokens(
                out["loss"].detach(), out["num_loss_tokens"].detach()
            )
            ce_batch_sum, _ = loss_sum_and_tokens(out["ce_loss"].detach(), out["num_loss_tokens"].detach())
            objective_sum += float(objective_batch_sum.cpu())
            ce_sum += float(ce_batch_sum.cpu())
            token_count += float(batch_tokens.cpu())
        logits = out["logits"][:, :-1, :]
        shifted = moved["labels"][:, 1:]
        mask = shifted.ne(-100)
        preds = logits.argmax(dim=-1)
        correct += int((preds.eq(shifted) & mask).sum().item())
        total += int(mask.sum().item())

    if parallel_context is not None and parallel_context.distributed:
        reduce_device = device if device.type == "cuda" else torch.device("cpu")
        tensor = torch.tensor(
            [objective_sum, ce_sum, token_count, float(correct), float(total)],
            dtype=torch.float64,
            device=reduce_device,
        )
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        objective_sum, ce_sum, token_count, correct_f, total_f = tensor.tolist()
        correct, total = int(correct_f), int(total_f)
    if token_count <= 0 or total <= 0:
        raise RuntimeError(
            "Evaluation produced zero supervised tokens. Check the evaluation split, "
            "assistant_only_loss, and data filters."
        )
    objective = objective_sum / max(1.0, token_count)
    ce_loss = ce_sum / max(1.0, token_count)
    if not math.isfinite(objective) or not math.isfinite(ce_loss):
        raise FloatingPointError(
            f"Evaluation produced non-finite metrics: objective={objective!r}, cross_entropy={ce_loss!r}."
        )
    return {
        "valid_loss": ce_loss,
        "valid_objective": objective,
        "perplexity": safe_perplexity(ce_loss),
        "token_accuracy": correct / max(1, total),
    }


def train_language_model(config: dict[str, Any], logger: Any, mode: str) -> Path:
    """Run pretraining or SFT."""

    set_seed(int(config["run"]["seed"]), bool(config["run"].get("deterministic", False)))
    parallel_context = init_parallel_context(config, logger)
    experiment_dir = make_experiment_dir(config)
    stage = "sft" if mode == "sft" else "pretrain"
    checkpoint_root = experiment_dir / stage
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    latest_dir = checkpoint_root / "latest"
    best_dir = checkpoint_root / "best"
    device = choose_device(config, logger, parallel_context)
    configure_torch_performance(config, device, logger)
    configure_single_process_data_parallel(config, device, logger, parallel_context)

    activation_experiment: ActivationExperiment | None = None
    gradient_experiment: GradientExperiment | None = None
    try:
        tokenizer = load_tokenizer(config)
        model_config = with_tokenizer_vocab(config, tokenizer.vocab_size)
        resume_dir = None
        if config["run"].get("resume"):
            for candidate in (latest_dir, best_dir):
                if resume_checkpoint_is_compatible(candidate, config, stage):
                    resume_dir = candidate
                    break
        has_resumable_latest = resume_dir is not None
        init_setting = config["train"].get("sft_init_checkpoint") if stage == "sft" else None
        if init_setting is not None and not has_resumable_latest:
            init_checkpoint = configured_checkpoint_path(config, init_setting, auto_stage="pretrain")
            if not checkpoint_is_loadable(init_checkpoint):
                raise FileNotFoundError(
                    f"Fresh SFT requires a loadable initialization checkpoint at {init_checkpoint}. "
                    "Run pretrain first, set train.sft_init_checkpoint to a valid path, or set it to null "
                    "only when intentionally training SFT from scratch."
                )
            base_model = load_model_from_checkpoint(init_checkpoint, model_config, map_location=device).to(device)
            logger.info(f"Initialized fresh SFT policy from pretrain checkpoint: {init_checkpoint}")
        else:
            base_model = build_model(model_config).to(device)
        logger.info(f"Attention backend selected: {base_model.backend}")
        for note in base_model.backend_notes:
            logger.info(note)
        if config["mtp"]["enabled"]:
            logger.info("MTP enabled: auxiliary future-token heads contribute to training loss.")
        if config["hybrid_diffusion"]["enabled"]:
            logger.info("Hybrid objective enabled: causal next-token loss + block masked-denoising loss.")

        experiment_output = Path(config["experiments"]["output_dir"]) / stage
        if parallel_context.distributed:
            experiment_output = experiment_output / f"rank_{parallel_context.rank}"
        activation_experiment = ActivationExperiment(base_model, config, experiment_output)
        gradient_experiment = GradientExperiment(base_model, config, experiment_output)

        configure_resolved_batching(base_model, config, device, logger, parallel_context)
        model = wrap_parallel_model(base_model, config, device, logger, parallel_context)
        model = maybe_compile_model(model, config, device, logger)

        if config["data"].get("streaming", False) and config["data"].get("sources"):
            train_loader, train_count = build_dataloader_from_config(
                tokenizer, config, mode, "train", device, logger, parallel_context
            )
            valid_loader, valid_count = build_dataloader_from_config(
                tokenizer, config, mode, "valid", device, logger, parallel_context
            )
        else:
            train_samples, valid_samples = load_train_valid_samples(config, stage, logger)
            train_loader, train_count = build_dataloader(
                train_samples, tokenizer, config, mode, "train", device, logger, parallel_context
            )
            valid_loader, valid_count = build_dataloader(
                valid_samples, tokenizer, config, mode, "valid", device, logger, parallel_context
            )
        logger.info(f"Prepared {train_count} training sequences and {valid_count} validation sequences.")

        accum = int(
            config["train"].get("_resolved_gradient_accumulation_steps") or resolve_gradient_accumulation(config)
        )
        total_steps = total_training_steps(len(train_loader), config)
        warmup_steps = config["train"].get("warmup_steps")
        if warmup_steps is None:
            warmup_steps = int(float(config["train"]["warmup_ratio"]) * total_steps)
        optimizer = build_optimizer(model, config, logger)
        scheduler = WarmupScheduler(
            optimizer,
            str(config["train"]["scheduler"]).lower(),
            total_steps,
            int(warmup_steps),
            float(config["train"]["learning_rate"]),
            float(config["train"]["min_learning_rate"]),
        )
        dtype, amp_enabled, scaler = choose_amp(device, config)
        logger.info(f"Mixed precision: dtype={dtype}, enabled={amp_enabled}, grad_scaler={scaler is not None}")
        neuromodulator = AdaptiveNeuromodulator(config)
        replay_buffer = SurpriseReplayBuffer(config, int(config["run"]["seed"]) + parallel_context.rank)
        if config["cognitive_architecture"]["enabled"]:
            logger.info(
                "Cognitive architecture enabled: causal workspace, predictive/homeostatic loss, "
                "adaptive plasticity, and surprise replay follow their individual gates."
            )

        start_epoch, start_batch_index, global_step, best_metric, stale_evals = 0, 0, 0, float("inf"), 0
        runtime_patches = RuntimePatchController(config)
        if resume_dir is not None:
            start_epoch, start_batch_index, global_step, best_metric, stale_evals = load_resume_state(
                resume_dir,
                base_model,
                optimizer,
                scheduler,
                scaler,
                logger,
                device,
                {"neuromodulator": neuromodulator, "replay_buffer": replay_buffer},
            )

        patience = int(config["train"]["early_stopping"]["patience"])
        max_steps = config["train"].get("max_steps")
        max_steps = int(max_steps) if max_steps else None
        optimizer.zero_grad(set_to_none=True)
        for change in runtime_patches.apply_due(global_step, optimizer, scheduler, model):
            if parallel_context.is_main:
                logger.metric({"stage": stage, "step": global_step, "event": "runtime_patch", **change})
                logger.info(
                    f"Runtime patch at step {global_step}: {change['key']} {change['before']} -> {change['after']}"
                )
        non_blocking = bool(config["hardware"].get("non_blocking_transfer", True)) and device.type == "cuda"
        log_tokens = 0
        log_samples = 0
        log_loss_sum = 0.0
        log_token_count = 0.0
        log_diffusion_loss_sum = 0.0
        log_diffusion_correct = 0
        log_diffusion_tokens = 0
        log_predictive_loss_sum = 0.0
        log_homeostatic_loss_sum = 0.0
        log_cognitive_batches = 0
        log_replay_loss_sum = 0.0
        log_replay_count = 0
        window_tokens = 0.0
        cognitive_metrics = neuromodulator.metrics()
        start_time = time.time()
        last_epoch = start_epoch
        last_eval_step: int | None = None
        last_eval_metrics: dict[str, float] | None = None
        last_checkpoint_step: int | None = None

        try:
            from tqdm import tqdm
        except Exception:

            def tqdm(iterable, **_):  # type: ignore[no-redef]
                return iterable

        for epoch in range(start_epoch, int(config["train"]["epochs"])):
            last_epoch = epoch
            model.train()
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            num_batches = len(train_loader)
            iterator = tqdm(train_loader, desc=f"{stage} epoch {epoch + 1}", disable=not parallel_context.is_main)
            for batch_idx, batch in enumerate(iterator):
                if epoch == start_epoch and batch_idx < start_batch_index:
                    continue
                if max_steps is not None and global_step >= max_steps:
                    break
                window_start = (batch_idx // accum) * accum
                window_end = min(window_start + accum, num_batches)
                should_step = batch_idx + 1 >= window_end
                sync_context = (
                    model.no_sync()
                    if parallel_context.distributed and hasattr(model, "no_sync") and not should_step
                    else contextlib.nullcontext()
                )
                try:
                    moved = move_batch_to_device(batch, device, non_blocking)
                    if activation_experiment is not None:
                        activation_experiment.set_step(global_step)
                        activation_experiment.set_phase("ar_train")
                    with sync_context:
                        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
                            out = model(
                                moved["input_ids"],
                                labels=moved["labels"],
                                attention_mask=moved["attention_mask"],
                                position_ids=moved["position_ids"],
                                document_ids=moved["document_ids"],
                                label_smoothing=float(
                                    runtime_patches.value("train.label_smoothing", config["train"]["label_smoothing"])
                                ),
                            )
                            raw_loss = out["loss"]
                            if raw_loss is None:
                                raise RuntimeError("Model did not return a training loss.")
                            diffusion_loss = None
                            diffusion_correct = 0
                            diffusion_tokens = 0
                            if config["hybrid_diffusion"]["enabled"]:
                                diffusion_batch = build_masked_diffusion_batch(
                                    moved["input_ids"],
                                    moved["labels"],
                                    moved["attention_mask"],
                                    tokenizer.mask_id,
                                    int(config["hybrid_diffusion"]["block_size"]),
                                    float(
                                        runtime_patches.value(
                                            "hybrid_diffusion.mask_probability",
                                            config["hybrid_diffusion"]["mask_probability"],
                                        )
                                    ),
                                    moved["document_ids"],
                                )
                                if activation_experiment is not None:
                                    activation_experiment.set_phase("diffusion_train")
                                diffusion_out = model(
                                    diffusion_batch["input_ids"],
                                    labels=diffusion_batch["labels"],
                                    attention_mask=diffusion_batch["attention_mask"],
                                    position_ids=moved["position_ids"],
                                    document_ids=moved["document_ids"],
                                    attention_mode="prefix_block",
                                    prefix_lengths=diffusion_batch["prefix_lengths"],
                                    block_size=int(config["hybrid_diffusion"]["block_size"]),
                                    loss_mode="same_token",
                                )
                                diffusion_loss = diffusion_out["loss"]
                                if diffusion_loss is None:
                                    raise RuntimeError("Model did not return a diffusion training loss.")
                                weight = float(
                                    runtime_patches.value(
                                        "hybrid_diffusion.loss_weight",
                                        config["hybrid_diffusion"]["loss_weight"],
                                    )
                                )
                                raw_loss = raw_loss + weight * diffusion_loss
                                diffusion_mask = diffusion_batch["labels"].ne(-100)
                                diffusion_correct = int(
                                    (
                                        diffusion_out["logits"].argmax(dim=-1).eq(diffusion_batch["labels"])
                                        & diffusion_mask
                                    )
                                    .sum()
                                    .item()
                                )
                                diffusion_tokens = int(diffusion_mask.sum().item())
                            mean_loss = scalar_loss(raw_loss)
                            # Token-weighted accumulation: back-propagate the summed
                            # token loss so each token contributes equally regardless
                            # of how micro-batches are split.  Gradients are divided
                            # by the window's total token count before the step.
                            batch_sum, batch_tokens = loss_sum_and_tokens(raw_loss, out["num_loss_tokens"])
                        if not torch.isfinite(mean_loss).item():
                            policy = config["train"].get("nan_inf_policy", "stop")
                            message = f"NaN/Inf loss detected at step {global_step}."
                            if policy == "stop":
                                raise FloatingPointError(message)
                            logger.info(message + " Skipping batch due to nan_inf_policy.")
                            optimizer.zero_grad(set_to_none=True)
                            window_tokens = 0.0
                            continue
                        log_tokens += count_batch_tokens(moved["input_ids"], moved["attention_mask"])
                        log_samples += int(moved["input_ids"].size(0))
                        log_loss_sum += float(batch_sum.detach().cpu())
                        log_token_count += float(batch_tokens.detach().cpu())
                        if diffusion_loss is not None:
                            log_diffusion_loss_sum += float(diffusion_loss.detach().cpu())
                            log_diffusion_correct += diffusion_correct
                            log_diffusion_tokens += diffusion_tokens
                        predictive_loss = out.get("predictive_loss")
                        homeostatic_loss = out.get("homeostatic_loss")
                        if predictive_loss is not None:
                            log_predictive_loss_sum += float(predictive_loss.detach().cpu())
                        if homeostatic_loss is not None:
                            log_homeostatic_loss_sum += float(homeostatic_loss.detach().cpu())
                        if predictive_loss is not None or homeostatic_loss is not None:
                            log_cognitive_batches += 1
                        window_tokens += float(batch_tokens.detach().cpu())
                        if scaler is not None:
                            scaler.scale(batch_sum).backward()
                        else:
                            batch_sum.backward()
                    if should_step and replay_buffer.should_replay(global_step + 1):
                        if activation_experiment is not None:
                            activation_experiment.set_phase("surprise_replay")
                        replayed = replay_buffer.sample(device, non_blocking)
                        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
                            replay_out = model(
                                replayed["input_ids"],
                                labels=replayed["labels"],
                                attention_mask=replayed["attention_mask"],
                                position_ids=replayed["position_ids"],
                                document_ids=replayed["document_ids"],
                                label_smoothing=float(
                                    runtime_patches.value("train.label_smoothing", config["train"]["label_smoothing"])
                                ),
                            )
                            replay_loss = replay_out["loss"]
                            if replay_loss is None or not torch.isfinite(replay_loss).item():
                                raise FloatingPointError("Replay produced a missing or non-finite loss.")
                            replay_sum, replay_tokens = loss_sum_and_tokens(replay_loss, replay_out["num_loss_tokens"])
                            weighted_replay_sum = replay_buffer.weight * replay_sum
                        if scaler is not None:
                            scaler.scale(weighted_replay_sum).backward()
                        else:
                            weighted_replay_sum.backward()
                        weighted_replay_tokens = replay_buffer.weight * float(replay_tokens.detach().cpu())
                        window_tokens += weighted_replay_tokens
                        log_replay_loss_sum += float(replay_loss.detach().cpu())
                        log_replay_count += 1
                        log_tokens += count_batch_tokens(replayed["input_ids"], replayed["attention_mask"])
                    replay_buffer.add(moved, float(mean_loss.detach().cpu()))
                except RuntimeError as exc:
                    if is_oom_error(exc) and config["hardware"].get("oom_retry", True):
                        if parallel_context.distributed:
                            raise RuntimeError(
                                "CUDA OOM occurred during DDP training after executable batch probing. "
                                "Lower hardware.target_vram_usage or hardware.auto_micro_batch_max and resume."
                            ) from exc
                        logger.info(
                            f"OOM at step {global_step}; clearing cache. "
                            "Reduce micro_batch_size and resume if repeated."
                        )
                        optimizer.zero_grad(set_to_none=True)
                        window_tokens = 0.0
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    raise

                if should_step:
                    if scaler is not None:
                        for optim in optimizer.optimizers:
                            scaler.unscale_(optim)
                    # Convert accumulated summed gradients into the token-weighted
                    # mean gradient.  Under DDP, gradients are averaged across the
                    # world, so multiply by world_size before dividing by the
                    # globally-summed token count to recover the exact mean.
                    total_window_tokens = window_tokens
                    if parallel_context.distributed:
                        total_window_tokens = distributed_sum_float(window_tokens, parallel_context, device)
                    grad_scale = parallel_context.training_world_size / max(1.0, total_window_tokens)
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.mul_(grad_scale)
                    window_tokens = 0.0
                    if gradient_experiment is not None:
                        gradient_experiment.record(global_step, scheduler.get_last_lr()[0])
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(runtime_patches.value("train.max_grad_norm", config["train"]["max_grad_norm"])),
                    )
                    if scaler is not None:
                        for optim in optimizer.optimizers:
                            scaler.step(optim)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    cognitive_metrics = neuromodulator.update(float(mean_loss.detach().cpu()), float(grad_norm))
                    neuromodulator.apply_learning_rate(optimizer, scheduler.get_last_lr()[0])
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    for change in runtime_patches.apply_due(global_step, optimizer, scheduler, model):
                        if parallel_context.is_main:
                            logger.metric({"stage": stage, "step": global_step, "event": "runtime_patch", **change})
                            logger.info(
                                f"Runtime patch at step {global_step}: "
                                f"{change['key']} {change['before']} -> {change['after']}"
                            )

                    window_loss = log_loss_sum / max(1.0, log_token_count)
                    if parallel_context.is_main and hasattr(iterator, "set_postfix"):
                        iterator.set_postfix(
                            loss=f"{window_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}", refresh=False
                        )

                    if global_step % int(config["train"]["log_interval_steps"]) == 0:
                        elapsed = max(1e-6, time.time() - start_time)
                        if parallel_context.is_main:
                            metrics = {
                                "stage": stage,
                                "epoch": epoch,
                                "step": global_step,
                                "loss": window_loss,
                                "lr": scheduler.get_last_lr()[0],
                                "grad_norm": float(grad_norm),
                                "tokens_per_sec": round(log_tokens / elapsed, 2),
                                "samples_per_sec": round(log_samples / elapsed, 2),
                                "vram_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3)
                                if device.type == "cuda"
                                else 0,
                            }
                            if log_diffusion_tokens:
                                metrics["diffusion_loss"] = log_diffusion_loss_sum / max(1, log_samples)
                                metrics["diffusion_token_accuracy"] = log_diffusion_correct / log_diffusion_tokens
                            if log_cognitive_batches:
                                metrics["predictive_loss"] = log_predictive_loss_sum / log_cognitive_batches
                                metrics["homeostatic_loss"] = log_homeostatic_loss_sum / log_cognitive_batches
                            if log_replay_count:
                                metrics["replay_loss"] = log_replay_loss_sum / log_replay_count
                            metrics["replay_buffer_items"] = len(replay_buffer)
                            metrics.update(cognitive_metrics)
                            logger.metric(metrics)
                            logger.info(json.dumps(metrics, ensure_ascii=False))
                        log_tokens = 0
                        log_samples = 0
                        log_loss_sum = 0.0
                        log_token_count = 0.0
                        log_diffusion_loss_sum = 0.0
                        log_diffusion_correct = 0
                        log_diffusion_tokens = 0
                        log_predictive_loss_sum = 0.0
                        log_homeostatic_loss_sum = 0.0
                        log_cognitive_batches = 0
                        log_replay_loss_sum = 0.0
                        log_replay_count = 0
                        start_time = time.time()

                    should_eval = global_step % int(config["train"]["eval_interval_steps"]) == 0
                    if should_eval:
                        stop_training = False
                        # Every rank participates so DDP eval shards the work and
                        # the all-reduced metric is identical on all processes.
                        if activation_experiment is not None:
                            activation_experiment.set_phase("ar_validation")
                        metrics = evaluate_loss(
                            model, valid_loader, device, config, parallel_context, desc=f"{stage} valid"
                        )
                        model.train()
                        # Set on every rank so the final-eval cache check stays
                        # collective (all ranks take the same branch in DDP).
                        last_eval_step = global_step
                        last_eval_metrics = metrics
                        if parallel_context.is_main:
                            logger.metric({"stage": stage, "step": global_step, **metrics})
                            metric = float(metrics["valid_loss"])
                            logger.info(f"Validation metrics: {json.dumps(metrics, ensure_ascii=False)}")
                            save_checkpoint(
                                latest_dir,
                                base_model,
                                optimizer,
                                scheduler,
                                scaler,
                                tokenizer,
                                config,
                                epoch,
                                global_step,
                                metric,
                                logger,
                                {
                                    "neuromodulator": neuromodulator.state_dict(),
                                    "replay_buffer": replay_buffer.state_dict(),
                                },
                                next_batch_index=batch_idx + 1,
                                best_metric=min(best_metric, metric),
                                stale_evals=0 if metric < best_metric else stale_evals + 1,
                            )
                            last_checkpoint_step = global_step
                            save_top_k_snapshot(
                                checkpoint_root,
                                latest_dir,
                                metric,
                                global_step,
                                int(config["train"]["top_k_checkpoints"]),
                                logger,
                            )
                            if metric < best_metric:
                                best_metric = metric
                                stale_evals = 0
                                if best_dir.exists():
                                    shutil.rmtree(best_dir)
                                shutil.copytree(latest_dir, best_dir)
                                logger.info(f"New best checkpoint by valid_loss={best_metric:.6f}.")
                            else:
                                stale_evals += 1
                            if config["train"]["early_stopping"]["enabled"] and stale_evals >= patience:
                                logger.info("Early stopping triggered.")
                                stop_training = True
                            if stop_training:
                                mark_stage_complete(checkpoint_root, config, stage, global_step)
                        distributed_barrier(parallel_context)
                        if distributed_any_bool(stop_training, parallel_context, device):
                            return best_dir if best_dir.exists() else latest_dir
                        model.train()

                    if global_step % int(config["train"]["save_interval_steps"]) == 0:
                        if parallel_context.is_main and last_checkpoint_step != global_step:
                            save_checkpoint(
                                latest_dir,
                                base_model,
                                optimizer,
                                scheduler,
                                scaler,
                                tokenizer,
                                config,
                                epoch,
                                global_step,
                                best_metric,
                                logger,
                                {
                                    "neuromodulator": neuromodulator.state_dict(),
                                    "replay_buffer": replay_buffer.state_dict(),
                                },
                                next_batch_index=batch_idx + 1,
                                best_metric=best_metric,
                                stale_evals=stale_evals,
                            )
                            last_checkpoint_step = global_step
                        distributed_barrier(parallel_context)
                    if max_steps is not None and global_step >= max_steps:
                        break
            if max_steps is not None and global_step >= max_steps:
                break

        if last_eval_step == global_step and last_eval_metrics is not None:
            final_metrics = last_eval_metrics
        else:
            if activation_experiment is not None:
                activation_experiment.set_phase("ar_validation")
            final_metrics = evaluate_loss(model, valid_loader, device, config, parallel_context, desc=f"{stage} final")
        if parallel_context.is_main:
            metric = float(final_metrics["valid_loss"])
            if last_checkpoint_step != global_step:
                save_checkpoint(
                    latest_dir,
                    base_model,
                    optimizer,
                    scheduler,
                    scaler,
                    tokenizer,
                    config,
                    last_epoch + 1,
                    global_step,
                    metric,
                    logger,
                    {
                        "neuromodulator": neuromodulator.state_dict(),
                        "replay_buffer": replay_buffer.state_dict(),
                    },
                    best_metric=min(best_metric, metric),
                    stale_evals=stale_evals,
                )
                last_checkpoint_step = global_step
                save_top_k_snapshot(
                    checkpoint_root,
                    latest_dir,
                    metric,
                    global_step,
                    int(config["train"]["top_k_checkpoints"]),
                    logger,
                )
            else:
                logger.info(f"Skipping duplicate final checkpoint save at step {global_step}.")
            if metric <= best_metric or not best_dir.exists():
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                shutil.copytree(latest_dir, best_dir)
                logger.info(f"Final checkpoint is best by valid_loss={metric:.6f}.")
            mark_stage_complete(checkpoint_root, config, stage, global_step)
            logger.info(f"Training complete: {json.dumps(final_metrics, ensure_ascii=False)}")
        distributed_barrier(parallel_context)
        return best_dir if best_dir.exists() else latest_dir
    finally:
        if activation_experiment is not None:
            activation_experiment.close()
        if gradient_experiment is not None:
            gradient_experiment.close()
        maybe_destroy_parallel_context(parallel_context)


def save_top_k_snapshot(
    checkpoint_root: Path,
    latest_dir: Path,
    metric: float,
    step: int,
    top_k: int,
    logger: Any,
) -> None:
    """Copy latest into a metric-named snapshot and prune to top_k best."""

    if top_k <= 0 or not latest_dir.exists():
        return
    top_root = checkpoint_root / "top_k"
    top_root.mkdir(parents=True, exist_ok=True)
    destination = top_root / f"step_{step:08d}_metric_{metric:.6f}"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(latest_dir, destination)

    snapshots: list[tuple[float, Path]] = []
    for child in top_root.iterdir():
        state_path = child / "training_state.pt"
        if not child.is_dir() or not state_path.exists():
            continue
        try:
            state = torch.load(state_path, map_location="cpu")
            snapshots.append((float(state.get("metric", float("inf"))), child))
        except Exception:
            snapshots.append((float("inf"), child))
    snapshots.sort(key=lambda item: item[0])
    for _, child in snapshots[top_k:]:
        shutil.rmtree(child)
        logger.info(f"Pruned top-k checkpoint snapshot {child}.")


def sequence_logprob(
    model, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Return summed log-probability over non--100 labels."""

    out = model(input_ids, attention_mask=attention_mask)
    logits = out["logits"][:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    logp = torch.log_softmax(logits, dim=-1)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logp = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=-1)


@torch.no_grad()
def evaluate_dpo_loss(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    tokenized: list[dict[str, list[int]]],
    pad_id: int,
    beta: float,
    config: dict[str, Any],
    device: torch.device,
    parallel_context: ParallelContext | None = None,
) -> float:
    """Average DPO loss over a held-out preference set (token-count agnostic)."""

    policy.eval()
    dtype, amp_enabled = choose_amp_dtype(device, config)
    batch_size = max(1, int(config["train"].get("_resolved_micro_batch_size") or 1))
    loss_type = config["dpo"].get("loss_type", "sigmoid")
    is_main = parallel_context.is_main if parallel_context else True
    loss_sum = 0.0
    pair_count = 0
    starts = range(0, len(tokenized), batch_size)
    try:
        from tqdm import tqdm

        starts = tqdm(starts, desc="dpo valid", disable=not is_main, leave=False)
    except Exception:
        pass
    for start in starts:
        chosen_ids, chosen_labels, chosen_mask, rejected_ids, rejected_labels, rejected_mask = make_dpo_batch(
            tokenized, pad_id, start, batch_size, device
        )
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
            chosen_logp = sequence_logprob(policy, chosen_ids, chosen_labels, chosen_mask)
            rejected_logp = sequence_logprob(policy, rejected_ids, rejected_labels, rejected_mask)
            ref_chosen = sequence_logprob(reference, chosen_ids, chosen_labels, chosen_mask)
            ref_rejected = sequence_logprob(reference, rejected_ids, rejected_labels, rejected_mask)
            logits = beta * ((chosen_logp - rejected_logp) - (ref_chosen - ref_rejected))
            if loss_type == "hinge":
                batch_loss = torch.relu(1 - logits).sum()
            else:
                batch_loss = -torch.nn.functional.logsigmoid(logits).sum()
        loss_sum += float(batch_loss.detach().cpu())
        pair_count += int(logits.numel())
    if parallel_context is not None and parallel_context.distributed:
        reduce_device = device if device.type == "cuda" else torch.device("cpu")
        tensor = torch.tensor([loss_sum, float(pair_count)], dtype=torch.float64, device=reduce_device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        loss_sum, pair_count_f = tensor.tolist()
        pair_count = int(pair_count_f)
    return loss_sum / max(1, pair_count)


def tokenize_preference_samples(samples, tokenizer, config: dict[str, Any]) -> list[dict[str, list[int]]]:
    """Tokenize DPO prompt/chosen/rejected rows once for reuse across epochs.

    Re-encoding the same preference data every batch and every epoch is wasted
    CPU work; tokenizing up front keeps the training loop GPU-bound.
    """

    length_limit = min(
        int(config["model"].get("max_seq_len", config["model"]["max_position_embeddings"])),
        int(config["model"]["max_position_embeddings"]),
    )

    def make_sequence(prompt_ids: list[int], completion_ids: list[int]) -> tuple[list[int], list[int]]:
        ids = prompt_ids + completion_ids + [tokenizer.eos_id]
        labels = [-100] * len(prompt_ids) + completion_ids + [tokenizer.eos_id]
        if len(ids) <= length_limit:
            return ids, labels
        overflow = len(ids) - length_limit
        if overflow < len(prompt_ids):
            kept_prompt = prompt_ids[overflow:]
            return kept_prompt + completion_ids + [tokenizer.eos_id], [-100] * len(kept_prompt) + completion_ids + [
                tokenizer.eos_id
            ]
        return ids[-length_limit:], labels[-length_limit:]

    try:
        from tqdm import tqdm

        iterator = tqdm(samples, desc="dpo tokenizing", unit="pair")
    except Exception:
        iterator = samples
    tokenized: list[dict[str, list[int]]] = []
    for sample in iterator:
        prompt_ids = [tokenizer.bos_id, *tokenizer.encode(sample.prompt, add_special_tokens=False)]
        chosen_ids, chosen_labels = make_sequence(prompt_ids, tokenizer.encode(sample.chosen, add_special_tokens=False))
        rejected_ids, rejected_labels = make_sequence(
            prompt_ids, tokenizer.encode(sample.rejected, add_special_tokens=False)
        )
        tokenized.append(
            {
                "chosen_ids": chosen_ids,
                "chosen_labels": chosen_labels,
                "rejected_ids": rejected_ids,
                "rejected_labels": rejected_labels,
            }
        )
    return tokenized


def make_dpo_batch(
    tokenized: list[dict[str, list[int]]],
    pad_id: int,
    start: int,
    batch_size: int,
    device: torch.device,
    indices: list[int] | None = None,
):
    """Pad pre-tokenized chosen/rejected sequences for one DPO batch."""

    if indices is not None:
        batch_rows = [tokenized[idx] for idx in indices[start : start + batch_size]]
    else:
        batch_rows = tokenized[start : start + batch_size]
    max_len = 1
    for row in batch_rows:
        max_len = max(max_len, len(row["chosen_ids"]), len(row["rejected_ids"]))

    def pad(key_ids: str, key_labels: str):
        input_ids = torch.full((len(batch_rows), max_len), pad_id, dtype=torch.long, device=device)
        labels = torch.full((len(batch_rows), max_len), -100, dtype=torch.long, device=device)
        mask = torch.zeros((len(batch_rows), max_len), dtype=torch.bool, device=device)
        for idx, row in enumerate(batch_rows):
            ids = row[key_ids]
            row_labels = row[key_labels]
            length = min(max_len, len(ids))
            input_ids[idx, :length] = torch.tensor(ids[:length], dtype=torch.long, device=device)
            labels[idx, :length] = torch.tensor(row_labels[:length], dtype=torch.long, device=device)
            mask[idx, :length] = True
        return input_ids, labels, mask

    return (*pad("chosen_ids", "chosen_labels"), *pad("rejected_ids", "rejected_labels"))


def probe_dpo_batch_fits(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    batch_size: int,
    dtype: torch.dtype,
    amp_enabled: bool,
) -> bool:
    """Return whether a max-length DPO step can backpropagate at batch_size."""

    seq_len = int(config["model"]["max_seq_len"])
    vocab = int(config["model"]["vocab_size"])
    beta = float(config["dpo"]["beta"])
    try:
        chosen_ids = torch.randint(0, max(1, vocab), (batch_size, seq_len), dtype=torch.long, device=device)
        rejected_ids = torch.randint(0, max(1, vocab), (batch_size, seq_len), dtype=torch.long, device=device)
        chosen_labels = chosen_ids.clone()
        rejected_labels = rejected_ids.clone()
        chosen_mask = torch.ones_like(chosen_ids, dtype=torch.bool)
        rejected_mask = torch.ones_like(rejected_ids, dtype=torch.bool)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
            chosen_logp = sequence_logprob(policy, chosen_ids, chosen_labels, chosen_mask)
            rejected_logp = sequence_logprob(policy, rejected_ids, rejected_labels, rejected_mask)
            with torch.no_grad():
                ref_chosen = sequence_logprob(reference, chosen_ids, chosen_labels, chosen_mask)
                ref_rejected = sequence_logprob(reference, rejected_ids, rejected_labels, rejected_mask)
            logits = beta * ((chosen_logp - rejected_logp) - (ref_chosen - ref_rejected))
            loss = -torch.nn.functional.logsigmoid(logits).mean()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return True
    except RuntimeError as exc:
        if is_oom_error(exc):
            return False
        raise
    finally:
        policy.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def find_executable_dpo_micro_batch_size(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    logger: Any,
    parallel_context: ParallelContext,
) -> int:
    """Find the largest per-device DPO micro batch that completes forward/backward."""

    explicit = config["train"]["micro_batch_size"]
    if explicit != "auto":
        return max(1, int(explicit))
    estimated = estimate_micro_batch_size(config, device, logger)
    if device.type != "cuda" or not bool(config["hardware"].get("auto_batch_size", True)):
        return estimated
    if not bool(config["hardware"].get("find_executable_batch_size", True)):
        return estimated

    dtype, amp_enabled = choose_amp_dtype(device, config)
    cap = max(1, int(config["hardware"].get("auto_micro_batch_max", 1024)))
    min_batch = max(1, int(config["hardware"].get("auto_micro_batch_min", 1)))
    growth = max(2.0, float(config["hardware"].get("auto_batch_growth_factor", 2.0)))
    start = min(cap, max(min_batch, estimated))
    policy.train()
    reference.eval()

    def fits(candidate: int) -> bool:
        ok = probe_dpo_batch_fits(policy, reference, config, device, candidate, dtype, amp_enabled)
        status = "ok" if ok else "oom"
        logger.info(f"DPO executable batch probe {status}: per_device_micro_batch={candidate}.")
        return ok

    if not fits(start):
        hi = start
        lo = min_batch
        success = 0
        candidate = max(min_batch, start // 2)
        while candidate >= min_batch:
            if fits(candidate):
                success = candidate
                lo = candidate + 1
                break
            hi = candidate
            if candidate == min_batch:
                break
            candidate = max(min_batch, candidate // 2)
        if success == 0:
            raise RuntimeError(
                "No executable CUDA DPO micro batch size was found. "
                "Lower model.max_seq_len/model size or free more VRAM."
            )
    else:
        success = start
        hi = min(cap + 1, max(success + 1, math.ceil(success * growth)))
        while hi <= cap and fits(hi):
            success = hi
            hi = min(cap + 1, max(success + 1, math.ceil(success * growth)))
        lo = success + 1

    high = min(cap, hi - 1)
    while lo <= high:
        mid = (lo + high) // 2
        if fits(mid):
            success = mid
            lo = mid + 1
        else:
            high = mid - 1

    success = distributed_min_int(success, parallel_context, device)
    if parallel_context.is_main:
        logger.info(f"Auto executable per-device DPO micro_batch_size selected {success}.")
    return max(1, success)


def configure_resolved_dpo_batching(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    logger: Any,
    parallel_context: ParallelContext,
) -> None:
    per_device_micro = find_executable_dpo_micro_batch_size(policy, reference, config, device, logger, parallel_context)
    data_parallel_size = parallel_context.data_parallel_size
    micro_batch = per_device_micro * data_parallel_size if data_parallel_size > 1 else per_device_micro
    config["train"]["_resolved_micro_batch_size"] = micro_batch
    config["train"]["_resolved_micro_batch_size_per_device"] = per_device_micro
    config["train"]["_gradient_parallel_world_size"] = parallel_context.training_world_size
    accum = resolve_gradient_accumulation(config)
    config["train"]["_resolved_gradient_accumulation_steps"] = accum
    logger.info(
        "Resolved DPO batch: "
        f"micro_batch_size={micro_batch}, gradient_accumulation_steps={accum}, "
        f"ddp_world_size={parallel_context.training_world_size}."
    )


def train_dpo(config: dict[str, Any], logger: Any) -> Path:
    """Direct Preference Optimization after SFT."""

    set_seed(int(config["run"]["seed"]), bool(config["run"].get("deterministic", False)))
    parallel_context = init_parallel_context(config, logger)
    experiment_dir = make_experiment_dir(config)
    checkpoint_root = experiment_dir / "dpo"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    latest_dir = checkpoint_root / "latest"
    best_dir = checkpoint_root / "best"
    device = choose_device(config, logger, parallel_context)
    configure_torch_performance(config, device, logger)
    configure_single_process_data_parallel(config, device, logger, parallel_context)

    try:
        tokenizer = load_tokenizer(config)
        model_config = with_tokenizer_vocab(config, tokenizer.vocab_size)
        policy_checkpoint = configured_checkpoint_path(config, config["dpo"]["policy_model_path"], "sft")
        reference_checkpoint = configured_checkpoint_path(config, config["dpo"]["reference_model_path"], "sft")
        for label, checkpoint in (("policy", policy_checkpoint), ("reference", reference_checkpoint)):
            if not checkpoint_is_loadable(checkpoint):
                raise FileNotFoundError(f"DPO {label} checkpoint is not loadable: {checkpoint}")
        resume_dir = None
        if config["run"].get("resume"):
            for candidate in (latest_dir, best_dir):
                if resume_checkpoint_is_compatible(candidate, config, "dpo"):
                    resume_dir = candidate
                    break
        has_resumable_latest = resume_dir is not None
        if has_resumable_latest:
            base_policy = build_model(model_config).to(device)
        else:
            base_policy = load_model_from_checkpoint(policy_checkpoint, model_config, map_location=device).to(device)
            logger.info(f"Initialized fresh DPO policy from: {policy_checkpoint}")
        base_reference = load_model_from_checkpoint(reference_checkpoint, model_config, map_location=device).to(device)
        logger.info(f"Loaded frozen DPO reference from: {reference_checkpoint}")
        base_reference.eval()
        for param in base_reference.parameters():
            param.requires_grad_(False)

        configure_resolved_dpo_batching(base_policy, base_reference, config, device, logger, parallel_context)
        policy = wrap_parallel_model(base_policy, config, device, logger, parallel_context)
        policy = maybe_compile_model(policy, config, device, logger)
        if parallel_context.data_parallel_size > 1:
            reference = torch.nn.DataParallel(base_reference, device_ids=parallel_context.data_parallel_devices)
        else:
            reference = base_reference

        dpo_train_file = Path(config["dpo"].get("train_file") or config["data"]["train_file"])
        raw_samples = load_preference_samples(dpo_train_file, config)
        if not raw_samples:
            raise RuntimeError("No DPO preference samples found. Expected prompt/chosen/rejected fields.")
        # Hold out a small validation split so "best" tracks generalization
        # rather than a noisy per-step training loss.
        raw_valid = []
        valid_file = config["dpo"].get("valid_file")
        if valid_file and not same_data_path(valid_file, dpo_train_file):
            raw_valid = load_preference_samples(valid_file, config)
        if not raw_valid and len(raw_samples) >= 20:
            holdout = max(1, len(raw_samples) // 50)
            # Input files are commonly concatenated by source.  A tail slice
            # therefore makes a biased holdout; stable hashes give an exact,
            # reproducible sample without depending on source order.
            ranked = sorted(
                enumerate(raw_samples),
                key=lambda item: stable_hash(f"{item[1].prompt}\0{item[1].chosen}"),
            )
            valid_indices = {index for index, _ in ranked[:holdout]}
            raw_valid = [sample for index, sample in enumerate(raw_samples) if index in valid_indices]
            raw_samples = [sample for index, sample in enumerate(raw_samples) if index not in valid_indices]
        samples = tokenize_preference_samples(raw_samples, tokenizer, config)
        valid_samples = tokenize_preference_samples(raw_valid, tokenizer, config) if raw_valid else []
        if parallel_context.is_main:
            logger.info(f"DPO data: train_pairs={len(samples)}, valid_pairs={len(valid_samples)}.")

        batch_size = resolve_micro_batch_size(config, device, logger)
        accum = int(
            config["train"].get("_resolved_gradient_accumulation_steps") or resolve_gradient_accumulation(config)
        )

        sampler = None
        if parallel_context.distributed:
            sampler = DistributedSampler(
                samples,
                num_replicas=parallel_context.world_size,
                rank=parallel_context.rank,
                shuffle=True,
                drop_last=False,
            )
            samples_per_epoch = int(sampler.num_samples)
        else:
            samples_per_epoch = len(samples)
        num_batches_per_epoch = max(1, math.ceil(samples_per_epoch / batch_size))
        total_steps = int(
            config["train"].get("max_steps")
            or int(config["train"]["epochs"]) * math.ceil(num_batches_per_epoch / max(1, accum))
        )
        optimizer = build_optimizer(policy, config, logger)
        dpo_warmup_steps = config["train"].get("warmup_steps")
        if dpo_warmup_steps is None:
            dpo_warmup_steps = int(float(config["train"]["warmup_ratio"]) * total_steps)
        scheduler = WarmupScheduler(
            optimizer,
            str(config["train"]["scheduler"]).lower(),
            total_steps,
            int(dpo_warmup_steps),
            float(config["train"]["learning_rate"]),
            float(config["train"]["min_learning_rate"]),
        )
        dtype, amp_enabled, scaler = choose_amp(device, config)
        beta = float(config["dpo"]["beta"])
        global_step = 0
        best_metric = float("inf")
        ema_loss: float | None = None
        start_epoch = 0
        start_batch_index = 0
        if resume_dir is not None:
            # DPO can resume mid-stage from the latest checkpoint instead of
            # restarting the whole stage.
            start_epoch, start_batch_index, global_step, best_metric, _ = load_resume_state(
                resume_dir, base_policy, optimizer, scheduler, scaler, logger, device
            )
        max_steps = config["train"].get("max_steps")
        max_steps = int(max_steps) if max_steps else None
        optimizer.zero_grad(set_to_none=True)
        last_epoch = start_epoch
        eval_interval = max(1, int(config["train"].get("eval_interval_steps", 1000)))

        try:
            from tqdm import tqdm
        except Exception:

            def tqdm(iterable, **_):  # type: ignore[no-redef]
                return iterable

        for epoch in range(start_epoch, int(config["train"]["epochs"])):
            last_epoch = epoch
            policy.train()
            if sampler is not None:
                sampler.set_epoch(epoch)
                indices = list(sampler)
                item_count = len(indices)
            else:
                generator = torch.Generator().manual_seed(int(config["run"]["seed"]) + epoch)
                indices = torch.randperm(len(samples), generator=generator).tolist()
                item_count = len(indices)
            starts = range(0, item_count, batch_size)
            num_batches = max(1, math.ceil(item_count / batch_size))
            iterator = tqdm(starts, desc=f"dpo epoch {epoch + 1}", disable=not parallel_context.is_main)
            for batch_idx, start in enumerate(iterator):
                if epoch == start_epoch and batch_idx < start_batch_index:
                    continue
                if max_steps is not None and global_step >= max_steps:
                    break
                window_start = (batch_idx // accum) * accum
                window_end = min(window_start + accum, num_batches)
                loss_divisor = max(1, window_end - window_start)
                should_step = batch_idx + 1 >= window_end
                sync_context = (
                    policy.no_sync()
                    if parallel_context.distributed and hasattr(policy, "no_sync") and not should_step
                    else contextlib.nullcontext()
                )
                chosen_ids, chosen_labels, chosen_mask, rejected_ids, rejected_labels, rejected_mask = make_dpo_batch(
                    samples, tokenizer.pad_id, start, batch_size, device, indices=indices
                )
                with sync_context:
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
                        chosen_logp = sequence_logprob(policy, chosen_ids, chosen_labels, chosen_mask)
                        rejected_logp = sequence_logprob(policy, rejected_ids, rejected_labels, rejected_mask)
                        with torch.no_grad():
                            ref_chosen = sequence_logprob(reference, chosen_ids, chosen_labels, chosen_mask)
                            ref_rejected = sequence_logprob(reference, rejected_ids, rejected_labels, rejected_mask)
                        logits = beta * ((chosen_logp - rejected_logp) - (ref_chosen - ref_rejected))
                        if config["dpo"].get("loss_type", "sigmoid") == "hinge":
                            raw_loss = torch.relu(1 - logits).mean()
                        else:
                            raw_loss = -torch.nn.functional.logsigmoid(logits).mean()
                        loss = raw_loss / loss_divisor
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                if should_step:
                    if scaler is not None:
                        for optim in optimizer.optimizers:
                            scaler.unscale_(optim)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), float(config["train"]["max_grad_norm"])
                    )
                    if scaler is not None:
                        for optim in optimizer.optimizers:
                            scaler.step(optim)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    loss_value = distributed_mean_float(float(raw_loss.detach().cpu()), parallel_context, device)
                    acc = distributed_mean_float(float((logits > 0).float().mean().item()), parallel_context, device)
                    margin = distributed_mean_float(
                        float((chosen_logp - rejected_logp).mean().item()), parallel_context, device
                    )
                    metrics = {
                        "stage": "dpo",
                        "step": global_step,
                        "dpo_loss": loss_value,
                        "chosen_rejected_accuracy": acc,
                        "reward_margin": margin,
                        "grad_norm": float(grad_norm),
                        "lr": scheduler.get_last_lr()[0],
                    }
                    ema_loss = loss_value if ema_loss is None else 0.9 * ema_loss + 0.1 * loss_value
                    if parallel_context.is_main and hasattr(iterator, "set_postfix"):
                        iterator.set_postfix(loss=f"{loss_value:.4f}", acc=f"{acc:.3f}", refresh=False)
                    if parallel_context.is_main:
                        logger.metric(metrics)
                        if global_step % int(config["train"]["log_interval_steps"]) == 0:
                            logger.info(json.dumps(metrics, ensure_ascii=False))

                    # Periodically checkpoint by validation loss (or smoothed
                    # training loss when no validation pairs are available) so the
                    # "best" model is not chosen from a single noisy step.
                    if global_step % eval_interval == 0:
                        if valid_samples:
                            candidate = evaluate_dpo_loss(
                                policy,
                                reference,
                                valid_samples,
                                tokenizer.pad_id,
                                beta,
                                config,
                                device,
                                parallel_context,
                            )
                            policy.train()
                        else:
                            candidate = ema_loss if ema_loss is not None else loss_value
                        if parallel_context.is_main:
                            save_checkpoint(
                                latest_dir,
                                base_policy,
                                optimizer,
                                scheduler,
                                scaler,
                                tokenizer,
                                config,
                                epoch,
                                global_step,
                                candidate,
                                logger,
                                next_batch_index=batch_idx + 1,
                                best_metric=min(best_metric, candidate),
                            )
                            if candidate < best_metric:
                                best_metric = candidate
                                if best_dir.exists():
                                    shutil.rmtree(best_dir)
                                shutil.copytree(latest_dir, best_dir)
                                logger.info(f"New best DPO checkpoint by metric={best_metric:.6f}.")
                        distributed_barrier(parallel_context)
                    if max_steps is not None and global_step >= max_steps:
                        break
            if max_steps is not None and global_step >= max_steps:
                break
        if valid_samples:
            final_candidate = evaluate_dpo_loss(
                policy,
                reference,
                valid_samples,
                tokenizer.pad_id,
                beta,
                config,
                device,
                parallel_context,
            )
        elif ema_loss is not None:
            final_candidate = ema_loss
        else:
            raise RuntimeError("DPO finished without producing a train or validation metric.")
        if parallel_context.is_main:
            save_checkpoint(
                latest_dir,
                base_policy,
                optimizer,
                scheduler,
                scaler,
                tokenizer,
                config,
                last_epoch + 1,
                global_step,
                final_candidate,
                logger,
                best_metric=min(best_metric, final_candidate),
            )
            if final_candidate <= best_metric or not best_dir.exists():
                best_metric = final_candidate
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                shutil.copytree(latest_dir, best_dir)
            mark_stage_complete(checkpoint_root, config, "dpo", global_step)
        distributed_barrier(parallel_context)
        return best_dir if best_dir.exists() else latest_dir
    finally:
        maybe_destroy_parallel_context(parallel_context)
