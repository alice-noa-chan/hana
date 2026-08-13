"""Optimizer construction and learning-rate scheduling."""

from __future__ import annotations

import math
from typing import Any

import torch


class MuonLike(torch.optim.Optimizer):
    """Small built-in Muon-style optimizer for two-dimensional hidden weights."""

    def __init__(self, params, lr: float, momentum: float = 0.95, weight_decay: float = 0.0):
        super().__init__(params, {"lr": lr, "momentum": momentum, "weight_decay": weight_decay})

    @staticmethod
    def _orthogonalize(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
        original_shape = grad.shape
        x = grad.float().reshape(grad.shape[0], -1)
        transposed = x.shape[0] > x.shape[1]
        if transposed:
            x = x.T
        x = x / (x.norm() + 1e-7)
        for _ in range(steps):
            x = 1.5 * x - 0.5 * x @ (x.T @ x)
        if transposed:
            x = x.T
        return x.reshape(original_shape).to(dtype=grad.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if weight_decay:
                    param.mul_(1 - lr * weight_decay)
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param.grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(param.grad)
                update = self._orthogonalize(buf) if param.ndim >= 2 else buf
                param.add_(update, alpha=-lr)
        return loss


class OptimizerBundle:
    """Treat one or more optimizers as a single optimizer-like object."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers

    @property
    def param_groups(self):
        groups = []
        for optimizer in self.optimizers:
            groups.extend(optimizer.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for optimizer, saved in zip(self.optimizers, state.get("optimizers", []), strict=False):
            optimizer.load_state_dict(saved)


class WarmupScheduler:
    """Cosine or linear scheduler with warmup and explicit state."""

    def __init__(
        self,
        optimizer: OptimizerBundle,
        scheduler_type: str,
        total_steps: int,
        warmup_steps: int,
        base_lr: float,
        min_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(0, warmup_steps)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.step_num = 0
        self._set_lr(self._lr_at(0))

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _lr_at(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)))
        if self.scheduler_type == "linear":
            return self.base_lr - (self.base_lr - self.min_lr) * progress
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def step(self) -> None:
        self.step_num += 1
        self._set_lr(self._lr_at(self.step_num))

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {"step_num": self.step_num}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step_num = int(state.get("step_num", 0))
        self._set_lr(self._lr_at(self.step_num))


def split_parameters_for_muon(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Route only 2D hidden weights to Muon-like updates."""

    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower = name.lower()
        if param.ndim == 2 and not any(skip in lower for skip in ("embed", "lm_head", "norm", "bias")):
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


def make_adamw(
    params,
    lr: float,
    weight_decay: float,
    config: dict[str, Any],
    logger: Any,
) -> torch.optim.AdamW:
    """Create AdamW with fused/foreach acceleration when available."""

    param_list = list(params)
    train_cfg = config["train"]
    fused_pref = train_cfg.get("fused_adamw", "auto")
    foreach_pref = train_cfg.get("foreach_optimizer", "auto")
    has_cuda_params = any(param.is_cuda for param in param_list)

    def is_enabled(value: Any) -> bool:
        return value is True or str(value).lower() == "auto"

    if has_cuda_params and is_enabled(fused_pref):
        try:
            optimizer = torch.optim.AdamW(param_list, lr=lr, weight_decay=weight_decay, fused=True)
            logger.info("Using fused AdamW.")
            return optimizer
        except TypeError:
            if fused_pref is True:
                raise
        except RuntimeError as exc:
            if fused_pref is True:
                raise
            logger.info(f"Fused AdamW unavailable, falling back to foreach/default AdamW: {exc}")

    if is_enabled(foreach_pref):
        try:
            optimizer = torch.optim.AdamW(param_list, lr=lr, weight_decay=weight_decay, foreach=True)
            logger.info("Using foreach AdamW.")
            return optimizer
        except TypeError:
            if foreach_pref is True:
                raise
        except RuntimeError as exc:
            if foreach_pref is True:
                raise
            logger.info(f"Foreach AdamW unavailable, falling back to default AdamW: {exc}")

    return torch.optim.AdamW(param_list, lr=lr, weight_decay=weight_decay)


def build_optimizer(model: torch.nn.Module, config: dict[str, Any], logger: Any) -> OptimizerBundle:
    """Build one of the optimizer modes accepted by config validation."""

    train_cfg = config["train"]
    lr = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg["weight_decay"])
    opt_name = str(train_cfg["optimizer"]).lower()
    if opt_name in {"muon", "muon_adamw", "hybrid_muon_adamw"}:
        muon_params, adamw_params = split_parameters_for_muon(model)
        logger.info(
            f"Using Muon-like optimizer for {len(muon_params)} matrix tensors and AdamW for "
            f"{len(adamw_params)} embedding/norm/bias/head tensors."
        )
        optimizers: list[torch.optim.Optimizer] = []
        if muon_params:
            optimizers.append(MuonLike(muon_params, lr=lr, weight_decay=weight_decay))
        if adamw_params:
            optimizers.append(make_adamw(adamw_params, lr=lr, weight_decay=weight_decay, config=config, logger=logger))
        return OptimizerBundle(optimizers)
    return OptimizerBundle(
        [make_adamw(model.parameters(), lr=lr, weight_decay=weight_decay, config=config, logger=logger)]
    )
