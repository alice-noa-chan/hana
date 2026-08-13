"""Device, precision, and distributed-process runtime services."""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from typing import Any

import torch

from .logging_utils import hardware_summary, require_torch


@dataclass
class ParallelContext:
    """Runtime parallelism discovered from CUDA and torchrun environment."""

    distributed: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    data_parallel_devices: list[int] | None = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def training_world_size(self) -> int:
        return max(1, self.world_size) if self.distributed else 1

    @property
    def data_parallel_size(self) -> int:
        return len(self.data_parallel_devices or [])


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def runtime_setting_enabled(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered == "auto":
        return default
    return lowered in {"1", "true", "yes", "on", "enabled"}


def init_parallel_context(config: dict[str, Any], logger: Any) -> ParallelContext:
    """Initialize DDP when launched by torchrun and record rank metadata."""

    hardware = config["hardware"]
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", rank)
    distributed_requested = runtime_setting_enabled(hardware.get("distributed", "auto"), default=world_size > 1)
    if world_size <= 1 or not distributed_requested:
        return ParallelContext()
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available, but WORLD_SIZE > 1 was detected.")

    requested = str(hardware.get("ddp_backend", "auto")).lower()
    cuda_ok = torch.cuda.is_available()
    backend = requested
    if requested == "auto":
        backend = "nccl" if cuda_ok and torch.distributed.is_nccl_available() else "gloo"
    if cuda_ok:
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            raise RuntimeError("CUDA is reported available, but no CUDA devices were found.")
        torch.cuda.set_device(local_rank % device_count)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=dt.timedelta(minutes=int(hardware.get("ddp_timeout_minutes", 60))),
        )
    context = ParallelContext(distributed=True, rank=rank, local_rank=local_rank, world_size=world_size)
    if context.is_main:
        logger.info(f"Distributed training initialized: backend={backend}, world_size={world_size}.")
    return context


def maybe_destroy_parallel_context(context: ParallelContext) -> None:
    if context.distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def distributed_barrier(context: ParallelContext) -> None:
    if context.distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def distributed_min_int(value: int, context: ParallelContext, device: torch.device) -> int:
    if not context.distributed:
        return value
    tensor_device = device if device.type == "cuda" else torch.device("cpu")
    tensor = torch.tensor([int(value)], dtype=torch.long, device=tensor_device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
    return int(tensor.item())


def distributed_mean_float(value: float, context: ParallelContext, device: torch.device) -> float:
    if not context.distributed:
        return float(value)
    tensor_device = device if device.type == "cuda" else torch.device("cpu")
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=tensor_device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.AVG)
    return float(tensor.item())


def distributed_sum_float(value: float, context: ParallelContext, device: torch.device) -> float:
    if not context.distributed:
        return float(value)
    tensor_device = device if device.type == "cuda" else torch.device("cpu")
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=tensor_device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item())


def distributed_any_bool(value: bool, context: ParallelContext, device: torch.device) -> bool:
    if not context.distributed:
        return bool(value)
    tensor_device = device if device.type == "cuda" else torch.device("cpu")
    tensor = torch.tensor([1 if value else 0], dtype=torch.long, device=tensor_device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return bool(tensor.item())


def choose_device(config: dict[str, Any], logger: Any, parallel_context: ParallelContext | None = None) -> torch.device:
    """Choose the requested CUDA device or the explicit CPU fallback."""

    torch_mod = require_torch()
    requested = config["hardware"]["device"]
    logger.info(f"Hardware summary: {json.dumps(hardware_summary(), ensure_ascii=False)}")
    if parallel_context and parallel_context.distributed and torch_mod.cuda.is_available():
        return torch.device("cuda", parallel_context.local_rank % torch_mod.cuda.device_count())
    if requested == "auto":
        if torch_mod.cuda.is_available():
            return torch.device("cuda")
        logger.info("WARNING: CUDA is not available. CPU training is supported but will be very slow.")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch_mod.cuda.is_available():
        raise RuntimeError("hardware.device requests CUDA, but torch.cuda.is_available() is false.")
    return device


def configure_torch_performance(config: dict[str, Any], device: torch.device, logger: Any) -> None:
    """Enable CUDA performance knobs that preserve training semantics."""

    if device.type != "cuda":
        return
    allow_tf32 = bool(config["hardware"].get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
    precision = str(config["hardware"].get("float32_matmul_precision", "high"))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(precision)
    logger.info(f"CUDA performance: allow_tf32={allow_tf32}, float32_matmul_precision={precision}.")


def choose_amp_dtype(device: torch.device, config: dict[str, Any]) -> tuple[torch.dtype, bool]:
    """Pick bf16, fp16, or fp32 from hardware capability and config."""

    mode = config["train"]["mixed_precision"]
    if device.type != "cuda":
        return torch.float32, False
    if mode == "bf16" or (mode == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16, True
    if mode in {"fp16", "auto"}:
        return torch.float16, True
    return torch.float32, False


def choose_amp(
    device: torch.device, config: dict[str, Any]
) -> tuple[torch.dtype, bool, torch.cuda.amp.GradScaler | None]:
    dtype, enabled = choose_amp_dtype(device, config)
    if dtype == torch.float16:
        return torch.float16, True, torch.cuda.amp.GradScaler()
    return dtype, enabled, None
