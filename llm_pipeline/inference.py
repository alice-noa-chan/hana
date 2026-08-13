"""Inference and rejected-response generation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .artifacts import (
    REASONING_GENERATION_PROTOCOL_VERSION,
    atomic_write_json,
    atomic_write_jsonl,
    build_rejects_fingerprint,
    inference_fingerprint,
    strict_json_dumps,
)
from .cognition import CognitiveMemory
from .data import iter_text_samples_from_config
from .data_reader import clean_text, render_messages
from .evaluation import resolve_checkpoint
from .experiments import ActivationExperiment, hybrid_generate
from .model_config import with_tokenizer_vocab
from .model_io import load_model_from_checkpoint
from .tokenizer import load_tokenizer
from .training_runtime import choose_device, configure_torch_performance


@dataclass(frozen=True)
class GenerationResult:
    """Final answer plus separately gated private reasoning metadata."""

    answer: str
    reasoning_mode: str
    reasoning_tokens: int = 0
    reasoning_trace: str | None = None
    candidate_count: int = 1
    reasoning_compute_tokens: int = 0
    answer_compute_tokens: int = 0
    selector_used: bool = False
    selector_compute_tokens: int = 0


@dataclass(frozen=True)
class _DecodedPhase:
    """Internal token-level result used to join reasoning and answer phases."""

    text: str
    input_ids: torch.Tensor
    token_ids: tuple[int, ...]
    trace: list[dict[str, Any]] | None


@dataclass(frozen=True)
class _ReasoningCandidate:
    """One ephemeral reasoning path and its final answer."""

    answer: str
    scratchpad: str
    scratchpad_ids: tuple[int, ...]
    final_phase: _DecodedPhase
    reasoning_generated_tokens: int


def apply_reasoning_control(prompt: str, config: dict[str, Any], mode: str | None = None) -> str:
    """Prefix one assistant cue with the configured reasoning control token."""

    reasoning = config["reasoning"]
    if not reasoning.get("enabled", True):
        return prompt
    mode = mode or config["inference"].get("reasoning_mode") or reasoning["default_mode"]
    if mode not in reasoning["modes"]:
        raise ValueError(f"Unsupported reasoning mode '{mode}'. Expected {reasoning['modes']}")
    if mode == "off":
        return prompt
    token = config["tokenizer"]["special_tokens"][f"reasoning_{mode}"]
    return f"{token}\n{prompt}"


def read_prompt_file(path: str | Path | None) -> str:
    """Read an optional prompt file."""

    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configured prompt file not found: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()


def read_prompt_files(paths: list[str] | tuple[str, ...] | None) -> str:
    """Read prompt files in priority order and join them."""

    if not paths:
        return ""
    parts = [read_prompt_file(path) for path in paths]
    return "\n\n".join(part for part in parts if part)


def build_inference_prompt(
    config: dict[str, Any],
    prompt: str,
    user_system_prompt: str | None = None,
    reasoning_mode: str | None = None,
) -> str:
    """Render reasoning control, system prompts, user prompt, and assistant cue."""

    inference_cfg = config["inference"]
    model_system = "\n\n".join(
        part
        for part in (
            inference_cfg.get("model_system_prompt", ""),
            read_prompt_files(inference_cfg.get("model_system_prompt_files")),
        )
        if part
    )
    user_system = "\n\n".join(
        part
        for part in (
            inference_cfg.get("user_system_prompt", ""),
            read_prompt_file(inference_cfg.get("user_system_prompt_file")),
            user_system_prompt or "",
        )
        if part
    )

    messages: list[dict[str, str]] = []
    if model_system:
        messages.append({"role": "system", "content": model_system})
    if user_system:
        messages.append({"role": "system", "content": user_system})
    messages.append({"role": "user", "content": prompt})
    rendered, _ = render_messages(messages, config["tokenizer"]["special_tokens"])
    assistant_cue = f"{config['tokenizer']['special_tokens']['assistant']}\n"
    return f"{rendered}{apply_reasoning_control(assistant_cue, config, reasoning_mode)}"


def reasoning_token_budget(config: dict[str, Any], mode: str) -> int:
    """Return the bounded scratchpad budget for one configured reasoning mode."""

    reasoning = config["reasoning"]
    if not reasoning.get("enabled", True) or mode == "off":
        return 0
    ratio = float(reasoning["mode_budget_ratios"][mode])
    return max(0, int(int(reasoning["max_reasoning_tokens"]) * ratio))


class TextGenerator:
    """Reusable inference context that keeps model/tokenizer loaded across prompts."""

    def __init__(
        self,
        config: dict[str, Any],
        logger: Any,
        checkpoint: Path | None = None,
        enable_memory: bool = True,
    ) -> None:
        self.config = config
        self.logger = logger
        self.device = choose_device(config, logger)
        configure_torch_performance(config, self.device, logger)
        self.tokenizer = load_tokenizer(config)
        model_config = with_tokenizer_vocab(config, self.tokenizer.vocab_size)
        self.checkpoint = checkpoint or resolve_checkpoint(config, config["inference"]["model_path"])
        self.model = load_model_from_checkpoint(self.checkpoint, model_config, map_location=self.device).to(self.device)
        self.model.eval()
        self.memory = (
            CognitiveMemory(config, self.model.cfg.hidden_size)
            if enable_memory
            and config["cognitive_architecture"]["enabled"]
            and config["cognitive_architecture"]["memory"]["enabled"]
            else None
        )
        self.activation_experiment = ActivationExperiment(
            self.model,
            config,
            Path(config["experiments"]["output_dir"]) / "inference",
        )
        self._warning_count = 0
        self._saved_reasoning_records: list[dict[str, Any]] = []
        if config["inference"].get("use_speculative_decoding"):
            logger.info(
                "Speculative/MTP decoding requested but no verifier/draft engine is configured; "
                "using autoregressive fallback."
            )

    @torch.no_grad()
    def _embed_text(self, text: str, phase: str) -> list[float]:
        limit = min(int(self.model.cfg.max_seq_len), int(self.model.cfg.max_position_embeddings))
        token_ids = [self.tokenizer.bos_id, *self.tokenizer.encode(text, add_special_tokens=False)]
        if len(token_ids) > limit:
            token_ids = [token_ids[0], *token_ids[-(limit - 1) :]] if limit > 1 else token_ids[:1]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        self.activation_experiment.set_phase(phase)
        output = self.model(input_ids, return_hidden_states=True)
        hidden_states = output["hidden_states"]
        if not hidden_states:
            raise RuntimeError("Model did not expose hidden states for cognitive memory.")
        embedding = hidden_states[-1][0, -1].detach().float()
        embedding = embedding / embedding.norm().clamp_min(1e-12)
        return embedding.cpu().tolist()

    def _warn_limited(self, message: str) -> None:
        if self._warning_count < 5:
            self.logger.info(message)
        elif self._warning_count == 5:
            self.logger.info("Further generation budget warnings suppressed.")
        self._warning_count += 1

    def _encode_prompt(
        self,
        prompt: str,
        settings: dict[str, Any],
        *,
        reserve_new_tokens: int = 0,
    ) -> tuple[torch.Tensor, int]:
        requested_new_tokens = max(0, int(settings["max_new_tokens"]))
        reserve_new_tokens = max(0, int(reserve_new_tokens))
        position_limit = int(self.config["model"].get("max_position_embeddings", self.config["model"]["max_seq_len"]))
        if position_limit <= 0:
            raise ValueError("model.max_position_embeddings must be positive for inference.")

        reserved = min(reserve_new_tokens, max(0, position_limit - 1))
        max_new_tokens = min(requested_new_tokens, max(0, position_limit - 1 - reserved))
        if max_new_tokens < requested_new_tokens:
            if settings.get("strict_context_fit", False):
                raise RuntimeError("The prompt and requested generation budgets do not fit the model context window.")
            self._warn_limited(
                f"Clamped max_new_tokens from {requested_new_tokens} to {max_new_tokens} "
                f"to fit max_position_embeddings={position_limit}."
            )
        prompt_budget = max(1, position_limit - max_new_tokens - reserved)

        token_ids = [self.tokenizer.bos_id, *self.tokenizer.encode(prompt, add_special_tokens=False)]
        if len(token_ids) > prompt_budget:
            if settings.get("strict_context_fit", False):
                raise RuntimeError("The prompt and requested generation budgets do not fit the model context window.")
            original_len = len(token_ids)
            policy = str(self.config["data"].get("truncation_policy", "recent")).lower()
            if policy in {"recent", "tail", "left"} and prompt_budget > 1:
                token_ids = [token_ids[0], *token_ids[-(prompt_budget - 1) :]]
            else:
                token_ids = token_ids[:prompt_budget]
            self._warn_limited(
                f"Truncated inference prompt from {original_len} to {len(token_ids)} tokens to fit generation budget."
            )

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        return input_ids, max_new_tokens

    def _decode(
        self,
        rendered_prompt: str,
        settings: dict[str, Any],
        *,
        phase: str,
        allow_token_trace: bool,
        reserve_new_tokens: int = 0,
        allow_reasoning_boundary: bool = False,
        generator: torch.Generator | None = None,
    ) -> _DecodedPhase:
        """Decode one phase while keeping scratchpad and final traces separate."""

        input_ids, max_new_tokens = self._encode_prompt(
            rendered_prompt,
            settings,
            reserve_new_tokens=reserve_new_tokens,
        )
        return self._decode_from_ids(
            input_ids,
            max_new_tokens,
            settings,
            phase=phase,
            allow_token_trace=allow_token_trace,
            allow_reasoning_boundary=allow_reasoning_boundary,
            generator=generator,
        )

    def _decode_from_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        settings: dict[str, Any],
        *,
        phase: str,
        allow_token_trace: bool,
        allow_reasoning_boundary: bool = False,
        generator: torch.Generator | None = None,
    ) -> _DecodedPhase:
        """Decode from an existing token prefix without re-tokenizing it."""

        position_limit = int(self.config["model"].get("max_position_embeddings", self.config["model"]["max_seq_len"]))
        max_new_tokens = min(max(0, int(max_new_tokens)), max(0, position_limit - input_ids.size(1)))
        trace: list[dict[str, Any]] | None = [] if allow_token_trace else None
        allowed_special_ids = {self.tokenizer.eos_id}
        generation_stop_id = self.tokenizer.eos_id
        if allow_reasoning_boundary:
            generation_stop_id = self.tokenizer.piece_to_id(self.config["tokenizer"]["special_tokens"]["reasoning_off"])
            allowed_special_ids = {generation_stop_id}
        suppress_ids = self.tokenizer.special_ids - allowed_special_ids
        if settings["generation_strategy"] == "hybrid":
            self.activation_experiment.set_phase(f"hybrid_{phase}")
            hybrid = self.config["hybrid_diffusion"]
            output = hybrid_generate(
                self.model,
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_id=generation_stop_id,
                mask_id=self.tokenizer.mask_id,
                block_size=int(hybrid["block_size"]),
                denoise_steps=int(hybrid["denoise_steps"]),
                ar_warmup_tokens=int(hybrid["ar_warmup_tokens"]),
                temperature=float(settings["temperature"]),
                top_p=float(settings["top_p"]),
                top_k=int(settings["top_k"]),
                repetition_penalty=float(settings["repetition_penalty"]),
                suppress_ids=suppress_ids,
                trace=trace,
                generator=generator,
            )
        else:
            self.activation_experiment.set_phase(f"ar_{phase}")
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_id=generation_stop_id,
                temperature=float(settings["temperature"]),
                top_p=float(settings["top_p"]),
                top_k=int(settings["top_k"]),
                repetition_penalty=float(settings["repetition_penalty"]),
                use_cache=bool(settings["use_kv_cache"]),
                suppress_ids=suppress_ids,
                trace=trace,
                generator=generator,
            )
        generated_ids = output[0].tolist()[input_ids.size(1) :]
        if trace is not None:
            for record in trace:
                ids = record.get("token_ids", [])
                flat_ids = ids[0] if ids and isinstance(ids[0], list) else ids
                record["token_pieces"] = [self.tokenizer.sp.id_to_piece(int(token_id)) for token_id in flat_ids]
        return _DecodedPhase(
            text=self.tokenizer.decode(generated_ids).strip(),
            input_ids=input_ids,
            token_ids=tuple(generated_ids),
            trace=trace,
        )

    def _candidate_generator(self, prompt_text: str, candidate_index: int) -> torch.Generator:
        """Create a stable per-candidate RNG without changing global RNG state."""

        configured_system = "\n\n".join(
            part
            for part in (
                self.config["inference"].get("model_system_prompt", ""),
                *(read_prompt_file(path) for path in self.config["inference"].get("model_system_prompt_files", [])),
            )
            if part
        )
        payload = "\0".join(
            (
                str(REASONING_GENERATION_PROTOCOL_VERSION),
                str(int(self.config["run"]["seed"])),
                configured_system,
                prompt_text,
                str(candidate_index),
            )
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
        return torch.Generator(device=self.device).manual_seed(seed)

    def _run_reasoning_candidate(
        self,
        prompt_text: str,
        effective_user_system: str,
        mode: str,
        settings: dict[str, Any],
        *,
        generator: torch.Generator | None = None,
    ) -> _ReasoningCandidate:
        """Generate one scratchpad and final answer without external side effects."""

        scratchpad_instruction = (
            read_prompt_file(self.config["reasoning"].get("scratchpad_instruction_file"))
            or self.config["reasoning"]["scratchpad_instruction"]
        )
        scratchpad_system = "\n\n".join(part for part in (effective_user_system, scratchpad_instruction) if part)
        reasoning_prompt = build_inference_prompt(
            self.config,
            prompt_text,
            user_system_prompt=scratchpad_system or None,
            reasoning_mode=mode,
        )
        reasoning_settings = {
            **settings,
            "max_new_tokens": reasoning_token_budget(self.config, mode),
            "token_trace_file": None,
        }
        boundary_token = self.config["tokenizer"]["special_tokens"]["reasoning_off"]
        boundary_id = self.tokenizer.piece_to_id(boundary_token)
        forced_boundary_ids = self.tokenizer.encode(
            f"\n{boundary_token}\n",
            add_special_tokens=False,
        )
        if forced_boundary_ids.count(boundary_id) != 1:
            raise RuntimeError("Tokenizer did not encode exactly one reasoning-off boundary.")
        requested_final_tokens = max(0, int(settings["max_new_tokens"]))
        reasoning_phase = self._decode(
            reasoning_prompt,
            reasoning_settings,
            phase="reasoning",
            allow_token_trace=False,
            reserve_new_tokens=len(forced_boundary_ids) + requested_final_tokens,
            allow_reasoning_boundary=True,
            generator=generator,
        )
        generated_reasoning_ids = list(reasoning_phase.token_ids)
        if boundary_id in generated_reasoning_ids:
            boundary_index = generated_reasoning_ids.index(boundary_id)
            scratchpad_ids = generated_reasoning_ids[:boundary_index]
            continuation_ids = generated_reasoning_ids[: boundary_index + 1]
        else:
            scratchpad_ids = generated_reasoning_ids
            continuation_ids = []
        while scratchpad_ids and scratchpad_ids[-1] == self.tokenizer.eos_id:
            scratchpad_ids.pop()
        if not continuation_ids:
            continuation_ids = [*scratchpad_ids, *forced_boundary_ids]
        final_input_ids = torch.cat(
            (
                reasoning_phase.input_ids,
                torch.tensor([continuation_ids], dtype=torch.long, device=self.device),
            ),
            dim=1,
        )
        final_phase = self._decode_from_ids(
            final_input_ids,
            requested_final_tokens,
            settings,
            phase="inference",
            allow_token_trace=bool(settings.get("token_trace_file") or self.memory is not None),
            generator=generator,
        )
        return _ReasoningCandidate(
            answer=final_phase.text.strip(),
            scratchpad=self.tokenizer.decode(scratchpad_ids).strip(),
            scratchpad_ids=tuple(scratchpad_ids),
            final_phase=final_phase,
            reasoning_generated_tokens=len(reasoning_phase.token_ids),
        )

    def _normalized_candidate_answer(self, answer: str) -> str:
        return clean_text(answer, self.config["data"]["normalize_nfkc"]).casefold()

    def _selector_answer(self, answer: str, max_tokens: int) -> str:
        ids = self.tokenizer.encode(answer, add_special_tokens=False)[:max_tokens]
        return self.tokenizer.decode(ids).strip() or "[empty answer]"

    def _select_reasoning_candidate(
        self,
        prompt_text: str,
        effective_user_system: str,
        candidates: list[_ReasoningCandidate],
        settings: dict[str, Any],
        tta: dict[str, Any],
    ) -> tuple[int, bool, int]:
        """Choose by strict majority, then by a private deterministic selector."""

        normalized = [self._normalized_candidate_answer(candidate.answer) for candidate in candidates]
        counts = Counter(normalized)
        majority_answer, majority_count = counts.most_common(1)[0]
        if majority_answer and majority_count > len(candidates) // 2:
            return normalized.index(majority_answer), False, 0

        if len(candidates) > 26:
            self._warn_limited("Reasoning selector supports at most 26 candidates; using the earliest candidate.")
            return 0, False, 0
        labels = [chr(ord("A") + index) for index in range(len(candidates))]
        candidate_token_limit = int(tta["selector_candidate_max_tokens"])
        rendered_candidates = "\n\n".join(
            f"{label}. {self._selector_answer(candidate.answer, candidate_token_limit)}"
            for label, candidate in zip(labels, candidates, strict=True)
        )
        selector_prompt = f"Original request:\n{prompt_text}\n\nCandidate final answers:\n{rendered_candidates}"
        selector_instruction = (
            "Privately select the candidate that best answers the original request. "
            f"Return exactly one ASCII letter from {', '.join(labels)} and no other text."
        )
        rendered_selector = build_inference_prompt(
            self.config,
            selector_prompt,
            user_system_prompt="\n\n".join(part for part in (effective_user_system, selector_instruction) if part),
            reasoning_mode="off",
        )
        selector_settings = {
            **settings,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "max_new_tokens": int(tta["selector_max_new_tokens"]),
            "token_trace_file": None,
            "strict_context_fit": True,
        }
        try:
            selector_phase = self._decode(
                rendered_selector,
                selector_settings,
                phase="reasoning_selector",
                allow_token_trace=False,
            )
            selector = selector_phase.text.strip()
        except RuntimeError as exc:
            if "context" not in str(exc).lower() and "fit" not in str(exc).lower():
                raise
            self._warn_limited("Reasoning selector did not fit the context; using the earliest candidate.")
            return 0, True, 0
        return (labels.index(selector) if selector in labels else 0), True, len(selector_phase.token_ids)

    @torch.no_grad()
    def generate_result(
        self,
        prompt: str | None = None,
        user_system_prompt: str | None = None,
        generation_settings: dict[str, Any] | None = None,
    ) -> GenerationResult:
        """Generate a hidden scratchpad first, then a separately decoded final answer."""

        settings = {**self.config["inference"], **(generation_settings or {})}
        prompt_text = clean_text(
            prompt if prompt is not None else self.config["inference"]["prompt"],
            self.config["data"]["normalize_nfkc"],
        )
        retrieved: list[dict[str, Any]] = []
        memory_context = ""
        if self.memory is not None:
            query_embedding = self._embed_text(prompt_text, "memory_query")
            retrieved = self.memory.retrieve(query_embedding)
            memory_context = self.memory.render(retrieved)
        effective_user_system = "\n\n".join(part for part in (user_system_prompt or "", memory_context) if part)
        mode = settings.get("reasoning_mode") or self.config["reasoning"]["default_mode"]
        if not self.config["reasoning"].get("enabled", True):
            mode = "off"
        if mode not in self.config["reasoning"]["modes"]:
            raise ValueError(f"Unsupported reasoning mode '{mode}'. Expected {self.config['reasoning']['modes']}")

        scratchpad = ""
        scratchpad_ids: tuple[int, ...] = ()
        candidate_count = 1
        reasoning_compute_tokens = 0
        answer_compute_tokens = 0
        selector_compute_tokens = 0
        selector_used = False
        budget = reasoning_token_budget(self.config, mode)
        if mode != "off" and budget <= 0:
            mode = "off"
        final_phase: _DecodedPhase
        if budget > 0:
            tta = self.config["reasoning"].get("test_time_compute") or {}
            if tta.get("enabled", False) and mode == tta.get("mode", "max"):
                if self.activation_experiment.has_active_stochastic_interventions():
                    raise RuntimeError(
                        "Seeded maximum-effort reasoning cannot run with an active noise activation intervention. "
                        "Disable the noise intervention or use a single-candidate reasoning mode."
                    )
                candidate_settings = {
                    **settings,
                    "temperature": float(tta["candidate_temperature"]),
                    "top_p": float(tta["candidate_top_p"]),
                    "top_k": int(tta["candidate_top_k"]),
                }
                activation_records = getattr(self.activation_experiment, "records", None)
                baseline_records = list(activation_records) if isinstance(activation_records, list) else None
                candidate_records: list[list[dict[str, Any]]] = []
                candidates = []
                try:
                    for candidate_index in range(int(tta["candidates"])):
                        if baseline_records is not None:
                            activation_records[:] = baseline_records
                        candidates.append(
                            self._run_reasoning_candidate(
                                prompt_text,
                                effective_user_system,
                                mode,
                                candidate_settings,
                                generator=self._candidate_generator(prompt_text, candidate_index),
                            )
                        )
                        if baseline_records is not None:
                            candidate_records.append(list(activation_records[len(baseline_records) :]))
                    if baseline_records is not None:
                        activation_records[:] = baseline_records
                    selected_index, selector_used, selector_compute_tokens = self._select_reasoning_candidate(
                        prompt_text,
                        effective_user_system,
                        candidates,
                        settings,
                        tta,
                    )
                finally:
                    if baseline_records is not None:
                        activation_records[:] = baseline_records
                if baseline_records is not None:
                    capacity = max(
                        0,
                        int(getattr(self.activation_experiment, "max_records", len(activation_records)))
                        - len(activation_records),
                    )
                    activation_records.extend(candidate_records[selected_index][:capacity])
                selected = candidates[selected_index]
                candidate_count = len(candidates)
                reasoning_compute_tokens = sum(candidate.reasoning_generated_tokens for candidate in candidates)
                answer_compute_tokens = sum(len(candidate.final_phase.token_ids) for candidate in candidates)
            else:
                selected = self._run_reasoning_candidate(
                    prompt_text,
                    effective_user_system,
                    mode,
                    settings,
                )
                reasoning_compute_tokens = selected.reasoning_generated_tokens
                answer_compute_tokens = len(selected.final_phase.token_ids)
            scratchpad = selected.scratchpad
            scratchpad_ids = selected.scratchpad_ids
            final_phase = selected.final_phase
        else:
            final_prompt = build_inference_prompt(
                self.config,
                prompt_text,
                user_system_prompt=effective_user_system or None,
                reasoning_mode=mode,
            )
            final_phase = self._decode(
                final_prompt,
                settings,
                phase="inference",
                allow_token_trace=bool(settings.get("token_trace_file") or self.memory is not None),
            )
            answer_compute_tokens = len(final_phase.token_ids)
        if settings.get("token_trace_file") and final_phase.trace is not None:
            atomic_write_jsonl(settings["token_trace_file"], final_phase.trace)

        generated = final_phase.text.strip()
        if self.memory is not None and generated:
            entropies = [
                float(value)
                for record in final_phase.trace or []
                for value in record.get("entropy", [])
                if isinstance(value, (int, float))
            ]
            entropy_scale = max(1e-12, math.log(max(2, self.tokenizer.vocab_size)))
            surprise = min(1.0, max(0.0, (sum(entropies) / max(1, len(entropies))) / entropy_scale))
            outcome_embedding = self._embed_text(f"{prompt_text}\n{generated}", "memory_encoding")
            stored = self.memory.observe(prompt_text, generated, outcome_embedding, surprise)
            self.logger.info(
                f"Cognitive memory: retrieved={len(retrieved)}, stored={stored}, "
                f"episodes={len(self.memory.state['episodes'])}, gists={len(self.memory.state['gists'])}."
            )
        self.activation_experiment.flush()
        exposed = scratchpad if self.config["reasoning"].get("expose_reasoning_trace", False) else None
        if scratchpad and self.config["reasoning"].get("save_reasoning_trace", False):
            self._saved_reasoning_records.append(
                {
                    "reasoning_mode": mode,
                    "reasoning_tokens": len(scratchpad_ids),
                    "reasoning_trace": scratchpad,
                }
            )
            atomic_write_jsonl(self.logger.log_dir / "reasoning_trace.jsonl", self._saved_reasoning_records)
        return GenerationResult(
            answer=generated,
            reasoning_mode=mode,
            reasoning_tokens=len(scratchpad_ids),
            reasoning_trace=exposed,
            candidate_count=candidate_count,
            reasoning_compute_tokens=reasoning_compute_tokens,
            answer_compute_tokens=answer_compute_tokens,
            selector_compute_tokens=selector_compute_tokens,
            selector_used=selector_used,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str | None = None,
        user_system_prompt: str | None = None,
        generation_settings: dict[str, Any] | None = None,
    ) -> str:
        """Generate a final answer while keeping any scratchpad private."""

        return self.generate_result(
            prompt=prompt,
            user_system_prompt=user_system_prompt,
            generation_settings=generation_settings,
        ).answer


def generate_text(
    config: dict[str, Any],
    logger: Any,
    prompt: str | None = None,
    checkpoint: Path | None = None,
    user_system_prompt: str | None = None,
) -> str:
    """Generate text from one prompt using a short-lived inference context."""

    return TextGenerator(config, logger, checkpoint=checkpoint).generate(
        prompt=prompt,
        user_system_prompt=user_system_prompt,
    )


def run_inference(config: dict[str, Any], logger: Any) -> None:
    """Run single-prompt or interactive inference."""

    generator = TextGenerator(config, logger)
    if config["inference"].get("interactive"):
        logger.info("Starting interactive inference. Empty input exits.")
        while True:
            prompt = input("prompt> ").strip()
            if not prompt:
                break
            print(generator.generate(prompt=prompt))
    else:
        result = generator.generate_result()
        output = result.answer
        minimum = int(config["inference"].get("min_output_chars", 1))
        if len(output) < minimum:
            raise RuntimeError(
                f"Inference output is too short ({len(output)} chars; minimum={minimum}). "
                "The checkpoint may be undertrained or generation settings may be invalid."
            )
        payload = {
            "status": "ok",
            "prompt": config["inference"]["prompt"],
            "output": output,
            "output_chars": len(output),
            "reasoning_mode": result.reasoning_mode,
            "reasoning_tokens": result.reasoning_tokens,
            "candidate_count": result.candidate_count,
            "reasoning_compute_tokens": result.reasoning_compute_tokens,
            "answer_compute_tokens": result.answer_compute_tokens,
            "selector_compute_tokens": result.selector_compute_tokens,
            "selector_used": result.selector_used,
            "checkpoint": str(generator.checkpoint),
            "inference_fingerprint": inference_fingerprint(config, generator.checkpoint),
            "settings": config["inference"],
        }
        if result.reasoning_trace is not None:
            payload["reasoning_trace"] = result.reasoning_trace
        logger.log_dir.mkdir(parents=True, exist_ok=True)
        out_path = logger.log_dir / "inference.json"
        atomic_write_json(out_path, payload)
        logger.info(f"Inference output: {output}")
        logger.info(f"Wrote inference log to {out_path}.")


def build_rejected_responses(config: dict[str, Any], logger: Any) -> Path:
    """Generate DPO rejected answers from prompts using the configured model.

    The output file is resumable: prompts already present in a partial run are
    skipped so an interrupted generation continues instead of restarting.
    """

    output_path = Path(
        config["dpo"].get("train_file") or (Path(config["data"]["processed_dir"]) / "dpo_rejected.jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completion_path = output_path.with_suffix(output_path.suffix + ".complete.json")
    generator = TextGenerator(config, logger, enable_memory=False)
    generation_fingerprint = build_rejects_fingerprint(config, generator.checkpoint)

    done_pairs: set[tuple[str, str]] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Partial DPO output contains invalid JSON and cannot be resumed safely: {output_path}"
                    ) from exc
                if existing.get("generation_fingerprint") != generation_fingerprint:
                    raise RuntimeError(
                        f"Existing DPO rejects were generated from different inputs/settings: {output_path}. "
                        "Move that generated artifact aside before rebuilding it."
                    )
                done_pairs.add((str(existing.get("prompt", "")), str(existing.get("chosen", ""))))
        if done_pairs:
            logger.info(f"Resuming rejected generation; {len(done_pairs)} preference pairs already complete.")

    rejected_settings = config["dpo"].get("generate_rejected") or {}
    rows = iter_rejection_seed_pairs(config)
    try:
        from tqdm import tqdm

        rows_iter = tqdm(rows, desc="build_rejects", unit="prompt")
    except Exception:
        rows_iter = rows

    written = 0
    eligible_pairs: set[tuple[str, str]] = set()
    with output_path.open("a", encoding="utf-8") as handle:
        for prompt, chosen in rows_iter:
            pair_key = (prompt, chosen)
            eligible_pairs.add(pair_key)
            if pair_key in done_pairs:
                continue
            rejected = generator.generate(prompt=prompt, generation_settings=rejected_settings)
            if not rejected:
                raise RuntimeError(f"Generated an empty rejected response for prompt: {prompt[:120]!r}")
            payload = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source": "generated_rejected",
                "generation_fingerprint": generation_fingerprint,
            }
            handle.write(strict_json_dumps(payload) + "\n")
            handle.flush()
            done_pairs.add(pair_key)
            written += 1
    if not eligible_pairs:
        raise RuntimeError("No eligible prompt/chosen pairs were found for rejected-response generation.")
    missing = eligible_pairs - done_pairs
    if missing:
        raise RuntimeError(f"Rejected generation ended with {len(missing)} preference pairs incomplete.")
    atomic_write_json(
        completion_path,
        {
            "format_version": 1,
            "generation_fingerprint": generation_fingerprint,
            "completed_pairs": len(eligible_pairs),
            "output_file": str(output_path),
        },
    )
    logger.info(f"Wrote {written} new rejected responses to {output_path}.")
    return output_path


def iter_rejection_seed_pairs(config: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Yield deterministic prompt/chosen pairs from selected, policy-approved SFT sources."""

    requested = tuple(str(value) for value in config["dpo"].get("prompt_sources") or ())
    if not requested:
        raise RuntimeError(
            "dpo.prompt_sources must explicitly list policy-approved SFT sources; "
            "the ungoverned data.train_file fallback is disabled."
        )
    limit_value = config["dpo"].get("max_prompt_samples")
    limit = int(limit_value) if limit_value is not None else None
    seen: set[tuple[str, str]] = set()

    selected = [
        source for source in config["data"].get("sources") or [] if str(source.get("name", "<unnamed>")) in requested
    ]
    selected_names = {str(source.get("name", "<unnamed>")) for source in selected}
    missing = sorted(set(requested) - selected_names)
    if missing:
        raise RuntimeError("DPO prompt sources are missing or excluded by policy: " + ", ".join(missing))
    source_config = deepcopy(config)
    source_config["data"]["sources"] = selected
    samples = iter_text_samples_from_config(source_config, split="train", dataset_type="sft")
    for sample in samples:
        messages = (sample.meta or {}).get(config["data"]["messages_field"])
        if not isinstance(messages, list):
            continue
        latest_user = ""
        for message in messages:
            role = str(message.get("role", "")).lower()
            content = clean_text(message.get("content", ""), config["data"]["normalize_nfkc"])
            if role == "user":
                latest_user = content
            elif role == "assistant" and latest_user and content:
                pair = (latest_user, content)
                if pair not in seen:
                    seen.add(pair)
                    yield pair
                    if limit is not None and len(seen) >= limit:
                        return
