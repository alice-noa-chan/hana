"""Runtime observability, activation interventions, and AR/diffusion experiments.

The features in this module are deliberately opt-in.  They expose useful
research controls without making ordinary training slower or retaining full
activation tensors by default.
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .artifacts import atomic_write_jsonl


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if isinstance(item, torch.Tensor)), None)
    return None


def _replace_first_tensor(value: Any, tensor: torch.Tensor) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor
    if isinstance(value, tuple):
        items = list(value)
        index = next(i for i, item in enumerate(items) if isinstance(item, torch.Tensor))
        items[index] = tensor
        return tuple(items)
    if isinstance(value, list):
        items = list(value)
        index = next(i for i, item in enumerate(items) if isinstance(item, torch.Tensor))
        items[index] = tensor
        return items
    return value


def _safe_scalar(value: torch.Tensor) -> float | None:
    number = float(value.cpu())
    return number if math.isfinite(number) else None


def _activation_stats(tensor: torch.Tensor) -> dict[str, float | None]:
    values = tensor.detach().float()
    finite_mask = torch.isfinite(values)
    finite_values = values[finite_mask]
    if finite_values.numel() == 0:
        return {
            "mean": None,
            "std": None,
            "rms": None,
            "abs_max": None,
            "zero_fraction": None,
            "finite_fraction": 0.0,
        }
    return {
        "mean": _safe_scalar(finite_values.mean()),
        "std": _safe_scalar(finite_values.std(unbiased=False)),
        "rms": _safe_scalar(finite_values.square().mean().sqrt()),
        "abs_max": _safe_scalar(finite_values.abs().max()),
        "zero_fraction": _safe_scalar((finite_values == 0).float().mean()),
        "finite_fraction": _safe_scalar(finite_mask.float().mean()),
    }


def _sample_tensor_values(tensor: torch.Tensor, count: int) -> list[float | None]:
    return [_safe_scalar(value) for value in tensor.detach().flatten()[:count].float().cpu()]


class ActivationExperiment:
    """Record bounded activation statistics and apply explicit forward hooks."""

    def __init__(self, model: nn.Module, config: dict[str, Any], output_dir: str | Path) -> None:
        experiment_cfg = config["experiments"]
        monitor_cfg = experiment_cfg["activation_monitor"]
        self.monitoring = bool(experiment_cfg["enabled"] and monitor_cfg["enabled"])
        self.enabled = bool(experiment_cfg["enabled"] and (self.monitoring or experiment_cfg.get("interventions")))
        self.step = 0
        self.phase = "unspecified"
        self.call_count = 0
        self.every_n_calls = max(1, int(monitor_cfg["every_n_calls"]))
        self.max_records = max(1, int(monitor_cfg["max_records"]))
        self.sample_values = max(0, int(monitor_cfg["sample_values"]))
        self.records: list[dict[str, Any]] = []
        self.interventions = list(experiment_cfg.get("interventions") or [])
        self.path = Path(output_dir) / str(monitor_cfg["output_file"])
        self.handles: list[Any] = []
        if not self.enabled:
            return

        patterns = list(monitor_cfg["modules"]) if self.monitoring else []
        patterns.extend(str(item["module"]) for item in self.interventions)
        matched: list[str] = []
        for name, module in model.named_modules():
            if name and any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
                matched.append(name)
                self.handles.append(module.register_forward_hook(self._make_hook(name)))
        if not matched:
            raise ValueError(f"Activation monitor patterns matched no modules: {patterns}")

    def set_step(self, step: int) -> None:
        self.step = int(step)

    def set_phase(self, phase: str) -> None:
        self.phase = str(phase)

    def has_active_stochastic_interventions(self) -> bool:
        """Return whether an active intervention consumes an untracked RNG stream."""

        if not self.enabled:
            return False
        for intervention in self.interventions:
            start = int(intervention.get("start_step", 0))
            end = intervention.get("end_step")
            if self.step < start or (end is not None and self.step > int(end)):
                continue
            if str(intervention["kind"]) == "noise":
                return True
        return False

    def _active_interventions(self, module_name: str) -> list[dict[str, Any]]:
        active = []
        for intervention in self.interventions:
            start = int(intervention.get("start_step", 0))
            end = intervention.get("end_step")
            if self.step < start or (end is not None and self.step > int(end)):
                continue
            if fnmatch.fnmatchcase(module_name, str(intervention["module"])):
                active.append(intervention)
        return active

    @staticmethod
    def _position_mask(tensor: torch.Tensor, positions: list[int] | None) -> torch.Tensor | None:
        if not positions or tensor.ndim < 3:
            return None
        seq_len = tensor.shape[-2]
        resolved = [position if position >= 0 else seq_len + position for position in positions]
        resolved = [position for position in resolved if 0 <= position < seq_len]
        mask = torch.zeros(seq_len, dtype=torch.bool, device=tensor.device)
        if resolved:
            mask[resolved] = True
        shape = [1] * tensor.ndim
        shape[-2] = seq_len
        return mask.view(shape)

    @staticmethod
    def _apply(tensor: torch.Tensor, spec: dict[str, Any]) -> torch.Tensor:
        kind = str(spec["kind"])
        strength = float(spec.get("value", 1.0))
        mask = ActivationExperiment._position_mask(tensor, spec.get("token_positions"))
        selected = tensor
        if kind == "zero":
            changed = torch.zeros_like(tensor)
        elif kind == "scale":
            changed = tensor * strength
        elif kind == "clamp":
            changed = tensor.clamp(min=-abs(strength), max=abs(strength))
        elif kind == "noise":
            changed = tensor + torch.randn_like(tensor) * strength
        elif kind in {"add_vector", "project_out"}:
            vector = torch.as_tensor(spec["vector"], dtype=tensor.dtype, device=tensor.device)
            if vector.ndim != 1 or vector.numel() != tensor.shape[-1]:
                raise ValueError(f"{kind} vector has {vector.numel()} values; expected hidden size {tensor.shape[-1]}.")
            if kind == "add_vector":
                changed = tensor + strength * vector
            else:
                direction = vector / vector.float().norm().clamp_min(1e-12).to(vector.dtype)
                projection = (tensor * direction).sum(dim=-1, keepdim=True) * direction
                changed = tensor - strength * projection
        else:
            raise ValueError(f"Unsupported activation intervention kind: {kind}")
        return torch.where(mask, changed, selected) if mask is not None else changed

    def _make_hook(self, module_name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            self.call_count += 1
            tensor = _first_tensor(output)
            if tensor is None:
                return output
            changed = tensor
            active = self._active_interventions(module_name)
            for intervention in active:
                changed = self._apply(changed, intervention)
            if self.monitoring and self.call_count % self.every_n_calls == 0 and len(self.records) < self.max_records:
                record: dict[str, Any] = {
                    "step": self.step,
                    "phase": self.phase,
                    "call": self.call_count,
                    "module": module_name,
                    "shape": list(tensor.shape),
                    "pre": _activation_stats(tensor),
                    "interventions": [str(item["kind"]) for item in active],
                }
                if active:
                    record["post"] = _activation_stats(changed)
                    record["delta_rms"] = _safe_scalar(
                        (changed.detach().float() - tensor.detach().float()).square().mean().sqrt()
                    )
                if self.sample_values:
                    record["pre_values"] = _sample_tensor_values(tensor, self.sample_values)
                    if active:
                        record["post_values"] = _sample_tensor_values(changed, self.sample_values)
                self.records.append(record)
            return _replace_first_tensor(output, changed) if active else output

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.flush()

    def flush(self) -> None:
        if self.enabled:
            atomic_write_jsonl(self.path, self.records)


class GradientExperiment:
    """Capture bounded per-parameter gradient/weight/update ratios before clipping."""

    def __init__(self, model: nn.Module, config: dict[str, Any], output_dir: str | Path) -> None:
        experiment_cfg = config["experiments"]
        gradient_cfg = experiment_cfg["gradient_monitor"]
        self.enabled = bool(experiment_cfg["enabled"] and gradient_cfg["enabled"])
        self.model = model
        self.patterns = list(gradient_cfg["parameters"])
        self.every_n_steps = max(1, int(gradient_cfg["every_n_steps"]))
        self.max_records = max(1, int(gradient_cfg["max_records"]))
        self.path = Path(output_dir) / str(gradient_cfg["output_file"])
        self.records: list[dict[str, Any]] = []

    def record(self, step: int, learning_rate: float) -> None:
        if not self.enabled or step % self.every_n_steps or len(self.records) >= self.max_records:
            return
        for name, parameter in self.model.named_parameters():
            if len(self.records) >= self.max_records:
                break
            if parameter.grad is None or not any(fnmatch.fnmatchcase(name, pattern) for pattern in self.patterns):
                continue
            weight = parameter.detach().float()
            gradient = parameter.grad.detach().float()
            finite_gradient = gradient[torch.isfinite(gradient)]
            weight_norm = weight.norm()
            gradient_norm = gradient.norm()
            self.records.append(
                {
                    "step": int(step),
                    "parameter": name,
                    "shape": list(parameter.shape),
                    "weight_norm": _safe_scalar(weight_norm),
                    "gradient_norm": _safe_scalar(gradient_norm),
                    "gradient_mean": _safe_scalar(finite_gradient.mean()) if finite_gradient.numel() else None,
                    "gradient_std": (
                        _safe_scalar(finite_gradient.std(unbiased=False)) if finite_gradient.numel() else None
                    ),
                    "gradient_finite_fraction": _safe_scalar(torch.isfinite(gradient).float().mean()),
                    "estimated_update_ratio": _safe_scalar(
                        float(learning_rate) * gradient_norm / weight_norm.clamp_min(1e-12)
                    ),
                }
            )

    def close(self) -> None:
        if self.enabled:
            atomic_write_jsonl(self.path, self.records)


RUNTIME_PATCH_KEYS = {
    "train.learning_rate",
    "train.min_learning_rate",
    "train.label_smoothing",
    "train.max_grad_norm",
    "model.attention_dropout",
    "model.residual_dropout",
    "model.embedding_dropout",
    "model.logit_softcap",
    "hybrid_diffusion.loss_weight",
    "hybrid_diffusion.mask_probability",
}


def _base_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module"):
        model = model.module  # type: ignore[assignment]
    return model


class RuntimePatchController:
    """Apply whitelisted, shape-preserving hyperparameter changes at step boundaries."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.events = sorted(config["experiments"].get("runtime_patches") or [], key=lambda item: int(item["at_step"]))
        self.applied: set[int] = set()
        self.current: dict[str, Any] = {}

    def value(self, key: str, default: Any) -> Any:
        return self.current.get(key, default)

    def apply_due(self, step: int, optimizer: Any, scheduler: Any, model: nn.Module) -> list[dict[str, Any]]:
        changes_applied: list[dict[str, Any]] = []
        for index, event in enumerate(self.events):
            if index in self.applied or int(event["at_step"]) > int(step):
                continue
            for key, value in event["changes"].items():
                if key not in RUNTIME_PATCH_KEYS:
                    raise ValueError(f"Runtime patch key is not shape-safe: {key}")
                section, name = key.split(".", 1)
                before = self.current.get(key, self.config[section][name])
                self.current[key] = value
                self._apply_one(key, value, optimizer, scheduler, model)
                changes_applied.append(
                    {
                        "scheduled_step": int(event["at_step"]),
                        "key": key,
                        "before": before,
                        "after": value,
                    }
                )
            self.applied.add(index)
        return changes_applied

    @staticmethod
    def _apply_one(key: str, value: Any, optimizer: Any, scheduler: Any, model: nn.Module) -> None:
        core = _base_model(model)
        number = float(value)
        if key == "train.learning_rate":
            scheduler.base_lr = number
            for group in optimizer.param_groups:
                group["lr"] = number
        elif key == "train.min_learning_rate":
            scheduler.min_lr = number
        elif key == "model.logit_softcap":
            core.cfg = replace(core.cfg, logit_softcap=number)
        elif key == "model.embedding_dropout":
            core.cfg = replace(core.cfg, embedding_dropout=number)
            core.embed_dropout.p = number
        elif key == "model.attention_dropout":
            core.cfg = replace(core.cfg, attention_dropout=number)
            for layer in core.layers:
                layer.attn.cfg = replace(layer.attn.cfg, attention_dropout=number)
                layer.attn.dropout.p = number
        elif key == "model.residual_dropout":
            core.cfg = replace(core.cfg, residual_dropout=number)
            for layer in core.layers:
                layer.residual_dropout.p = number
                layer.ffn.dropout.p = number


def build_masked_diffusion_batch(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    mask_id: int,
    block_size: int,
    mask_probability: float,
    document_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Corrupt one supervised block per row for same-token denoising."""

    if block_size <= 0 or not 0 < mask_probability <= 1:
        raise ValueError("block_size must be positive and mask_probability must be in (0, 1].")
    corrupted = input_ids.clone()
    diffusion_labels = torch.full_like(labels, -100)
    diffusion_attention = (
        torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask is None
        else attention_mask.clone().to(torch.bool)
    )
    prefixes = torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)
    total_masked = 0

    for row in range(input_ids.size(0)):
        candidates = torch.nonzero(diffusion_attention[row] & labels[row].ne(-100), as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        chosen = int(candidates[torch.randint(candidates.numel(), (1,), device=input_ids.device)].item())
        if document_ids is None:
            document_start = 0
            document_end = input_ids.size(1)
        else:
            same_document = torch.nonzero(
                diffusion_attention[row] & document_ids[row].eq(document_ids[row, chosen]), as_tuple=False
            ).flatten()
            document_start = int(same_document[0].item())
            document_end = int(same_document[-1].item()) + 1
        start = document_start + ((chosen - document_start) // block_size) * block_size
        end = min(document_end, start + block_size)
        in_block = candidates[(candidates >= start) & (candidates < end)]
        if document_ids is not None:
            in_block = in_block[document_ids[row, in_block] == document_ids[row, chosen]]
        if in_block.numel() == 0:
            in_block = torch.tensor([chosen], device=input_ids.device)
        selected = in_block[torch.rand(in_block.numel(), device=input_ids.device) < mask_probability]
        if selected.numel() == 0:
            selected = in_block[torch.randint(in_block.numel(), (1,), device=input_ids.device)]
        corrupted[row, selected] = int(mask_id)
        diffusion_labels[row, selected] = input_ids[row, selected]
        diffusion_attention[row, end:] = False
        prefixes[row] = start
        total_masked += int(selected.numel())

    if total_masked == 0:
        raise ValueError("Masked diffusion batch contains zero supervised tokens.")
    return {
        "input_ids": corrupted,
        "labels": diffusion_labels,
        "attention_mask": diffusion_attention,
        "prefix_lengths": prefixes,
        "num_masked_tokens": torch.tensor(total_masked, device=input_ids.device),
    }


def _sample_block_logits(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    suppress_ids: set[int] | frozenset[int] | None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = logits.float().clone()
    if suppress_ids:
        valid = [index for index in suppress_ids if 0 <= index < scores.size(-1)]
        if valid:
            scores[..., valid] = -float("inf")
    if temperature <= 0:
        probabilities = torch.softmax(scores, dim=-1)
        tokens = scores.argmax(dim=-1)
    else:
        scores = scores / max(temperature, 1e-5)
        if top_k > 0:
            threshold = torch.topk(scores, min(top_k, scores.size(-1)), dim=-1).values[..., [-1]]
            scores.masked_fill_(scores < threshold, -float("inf"))
        if top_p < 1.0:
            sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
            cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_scores.masked_fill_(remove, -float("inf"))
            scores = torch.full_like(scores, -float("inf")).scatter(-1, sorted_indices, sorted_scores)
        probabilities = torch.softmax(scores, dim=-1)
        tokens = torch.multinomial(
            probabilities.reshape(-1, probabilities.size(-1)),
            1,
            generator=generator,
        ).reshape(probabilities.shape[:-1])
    confidence = probabilities.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    return tokens, confidence, entropy


@torch.no_grad()
def hybrid_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_id: int,
    mask_id: int,
    block_size: int,
    denoise_steps: int,
    ar_warmup_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    suppress_ids: set[int] | frozenset[int] | None = None,
    trace: list[dict[str, Any]] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate causally across blocks and denoise tokens in each block in parallel."""

    model.eval()
    if input_ids.size(0) != 1:
        raise ValueError("hybrid_generate currently supports batch size 1.")
    if block_size <= 0 or denoise_steps <= 0:
        raise ValueError("block_size and denoise_steps must be positive.")
    position_limit = int(_base_model(model).cfg.max_position_embeddings)
    budget = min(max(0, int(max_new_tokens)), max(0, position_limit - input_ids.size(1)))
    warmup = min(budget, max(0, int(ar_warmup_tokens)))
    generated = model.generate(
        input_ids,
        max_new_tokens=warmup,
        eos_id=eos_id,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        use_cache=True,
        suppress_ids=suppress_ids,
        trace=trace,
        generator=generator,
    )
    if generated[0, input_ids.size(1) :].eq(eos_id).any():
        suffix_eos = torch.nonzero(generated[0, input_ids.size(1) :].eq(eos_id), as_tuple=False)[0]
        return generated[:, : input_ids.size(1) + int(suffix_eos.item()) + 1]

    remaining = budget - (generated.size(1) - input_ids.size(1))
    while remaining > 0:
        width = min(block_size, remaining, position_limit - generated.size(1))
        if width <= 0:
            break
        prefix_len = generated.size(1)
        block = torch.full((1, width), int(mask_id), dtype=generated.dtype, device=generated.device)
        committed = torch.zeros(width, dtype=torch.bool, device=generated.device)
        confidence = torch.zeros(width, device=generated.device)
        for denoise_step in range(denoise_steps):
            candidate = torch.cat([generated, block], dim=1)
            output = model(
                candidate,
                attention_mode="prefix_block",
                prefix_lengths=torch.tensor([prefix_len], device=generated.device),
                block_size=block_size,
            )
            block_logits = output["logits"][:, prefix_len:, :]
            if repetition_penalty != 1.0:
                penalty = float(repetition_penalty)
                seen = torch.unique(candidate[candidate.ne(mask_id)])
                if seen.numel():
                    seen_scores = block_logits[..., seen]
                    seen_scores = torch.where(
                        seen_scores < 0,
                        seen_scores * penalty,
                        seen_scores / penalty,
                    )
                    block_logits[..., seen] = seen_scores
            tokens, proposed_confidence, entropy = _sample_block_logits(
                block_logits,
                temperature,
                top_p,
                top_k,
                suppress_ids,
                generator,
            )
            target_count = min(width, math.ceil(width * (denoise_step + 1) / denoise_steps))
            add_count = target_count - int(committed.sum().item())
            if add_count > 0:
                scores = proposed_confidence[0].masked_fill(committed, -1)
                chosen = torch.topk(scores, add_count).indices
                block[0, chosen] = tokens[0, chosen]
                confidence[chosen] = proposed_confidence[0, chosen]
                committed[chosen] = True
            if trace is not None:
                trace.append(
                    {
                        "phase": "diffusion",
                        "block_start": prefix_len,
                        "denoise_step": denoise_step,
                        "committed": int(committed.sum().item()),
                        "token_ids": block[0].detach().cpu().tolist(),
                        "confidence": confidence.detach().cpu().tolist(),
                        "entropy": entropy[0].detach().cpu().tolist(),
                    }
                )
        if block.eq(mask_id).any():
            raise RuntimeError("Hybrid denoising ended with unresolved mask tokens.")
        generated = torch.cat([generated, block], dim=1)
        eos_positions = torch.nonzero(generated[0, prefix_len:].eq(eos_id), as_tuple=False)
        if eos_positions.numel():
            return generated[:, : prefix_len + int(eos_positions[0].item()) + 1]
        remaining = budget - (generated.size(1) - input_ids.size(1))
    return generated
