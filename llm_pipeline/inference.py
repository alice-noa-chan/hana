"""Inference and rejected-response generation."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from .artifacts import (
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


def apply_reasoning_control(prompt: str, config: dict[str, Any]) -> str:
    """Prefix the prompt with the configured reasoning control token."""

    reasoning = config["reasoning"]
    if not reasoning.get("enabled", True):
        return f"{config['tokenizer']['special_tokens']['reasoning_off']}\n{prompt}"
    mode = config["inference"].get("reasoning_mode") or reasoning["default_mode"]
    if mode not in reasoning["modes"]:
        raise ValueError(f"Unsupported reasoning mode '{mode}'. Expected {reasoning['modes']}")
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
    rendered = f"{rendered}{config['tokenizer']['special_tokens']['assistant']}\n"
    return apply_reasoning_control(rendered, config)


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

    def _encode_prompt(self, prompt: str, settings: dict[str, Any]) -> tuple[torch.Tensor, int]:
        requested_new_tokens = max(0, int(settings["max_new_tokens"]))
        position_limit = int(self.config["model"].get("max_position_embeddings", self.config["model"]["max_seq_len"]))
        if position_limit <= 0:
            raise ValueError("model.max_position_embeddings must be positive for inference.")

        max_new_tokens = min(requested_new_tokens, max(0, position_limit - 1))
        if max_new_tokens < requested_new_tokens:
            self._warn_limited(
                f"Clamped max_new_tokens from {requested_new_tokens} to {max_new_tokens} "
                f"to fit max_position_embeddings={position_limit}."
            )
        prompt_budget = max(1, position_limit - max_new_tokens)

        token_ids = [self.tokenizer.bos_id, *self.tokenizer.encode(prompt, add_special_tokens=False)]
        if len(token_ids) > prompt_budget:
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

    @torch.no_grad()
    def generate(
        self,
        prompt: str | None = None,
        user_system_prompt: str | None = None,
        generation_settings: dict[str, Any] | None = None,
    ) -> str:
        """Generate text from one prompt using KV-cache autoregressive decoding."""

        settings = {**self.config["inference"], **(generation_settings or {})}
        prompt_text = clean_text(
            prompt if prompt is not None else self.config["inference"]["prompt"],
            self.config["data"]["normalize_nfkc"],
        )
        retrieved: list[dict[str, Any]] = []
        query_embedding: list[float] | None = None
        memory_context = ""
        if self.memory is not None:
            query_embedding = self._embed_text(prompt_text, "memory_query")
            retrieved = self.memory.retrieve(query_embedding)
            memory_context = self.memory.render(retrieved)
        effective_user_system = "\n\n".join(part for part in (user_system_prompt or "", memory_context) if part)
        rendered_prompt = build_inference_prompt(
            self.config,
            prompt_text,
            user_system_prompt=effective_user_system or None,
        )
        input_ids, max_new_tokens = self._encode_prompt(rendered_prompt, settings)
        trace: list[dict[str, Any]] | None = [] if settings.get("token_trace_file") or self.memory is not None else None
        suppress_ids = self.tokenizer.special_ids - {self.tokenizer.eos_id}
        if settings["generation_strategy"] == "hybrid":
            self.activation_experiment.set_phase("hybrid_inference")
            hybrid = self.config["hybrid_diffusion"]
            output = hybrid_generate(
                self.model,
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_id=self.tokenizer.eos_id,
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
            )
        else:
            self.activation_experiment.set_phase("ar_inference")
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_id=self.tokenizer.eos_id,
                temperature=float(settings["temperature"]),
                top_p=float(settings["top_p"]),
                top_k=int(settings["top_k"]),
                repetition_penalty=float(settings["repetition_penalty"]),
                use_cache=bool(settings["use_kv_cache"]),
                suppress_ids=suppress_ids,
                trace=trace,
            )
        if trace is not None:
            for record in trace:
                ids = record.get("token_ids", [])
                flat_ids = ids[0] if ids and isinstance(ids[0], list) else ids
                record["token_pieces"] = [self.tokenizer.sp.id_to_piece(int(token_id)) for token_id in flat_ids]
            if settings.get("token_trace_file"):
                atomic_write_jsonl(settings["token_trace_file"], trace)
        generated = self.tokenizer.decode(output[0].tolist()[input_ids.size(1) :])
        if not self.config["reasoning"].get("expose_reasoning_trace", False):
            for token in ("<reasoning>", "</reasoning>"):
                generated = generated.replace(token, "")
        generated = generated.strip()
        if self.memory is not None and generated:
            entropies = [
                float(value)
                for record in trace or []
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
        return generated


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
        output = generator.generate()
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
            "checkpoint": str(generator.checkpoint),
            "inference_fingerprint": inference_fingerprint(config, generator.checkpoint),
            "settings": config["inference"],
        }
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
