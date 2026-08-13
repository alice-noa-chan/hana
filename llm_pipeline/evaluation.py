"""Evaluation entrypoints for LM loss, DPO metrics, and memory probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .artifacts import (
    atomic_write_jsonl,
    atomic_write_text,
    checkpoint_dataset_type,
    checkpoint_is_loadable,
    checkpoint_stage,
    completed_stage_checkpoint,
    evaluation_fingerprint,
)
from .data import (
    build_token_shard_dataset,
    load_preference_samples,
    load_text_samples,
    load_text_samples_from_config,
    make_torch_dataset,
    split_samples,
    tokenize_training_samples,
)
from .model_config import with_tokenizer_vocab
from .model_io import load_model_from_checkpoint
from .multiple_choice import preflight_knowledge_pilot, run_knowledge_pilot
from .tokenizer import load_tokenizer
from .training import evaluate_loss, sequence_logprob
from .training_runtime import choose_device, configure_torch_performance

_SAFE_CACHED_EVALUATION_FIELDS = frozenset(
    {
        "checkpoint",
        "valid_loss",
        "valid_objective",
        "perplexity",
        "token_accuracy",
        "dpo_sample_count",
        "chosen_rejected_accuracy",
        "reward_margin",
        "multiturn_memory_n_turn_accuracy",
        "memory_probe_count",
        "correct_count",
        "item_count",
        "accuracy",
        "parse_rate",
        "passed",
        "evaluation_fingerprint",
    }
)


def _safe_cached_evaluation_row(row: Any) -> dict[str, Any] | None:
    """Accept only aggregate fields that this evaluator writes itself."""

    if not isinstance(row, dict) or not set(row).issubset(_SAFE_CACHED_EVALUATION_FIELDS):
        return None
    if not isinstance(row.get("checkpoint"), str) or not isinstance(row.get("evaluation_fingerprint"), str):
        return None
    return row


def resolve_checkpoint(config: dict[str, Any], name: str) -> Path:
    """Resolve latest/best or an explicit checkpoint path."""

    path = Path(name)
    if checkpoint_is_loadable(path):
        return path
    for stage in ("dpo", "sft", "pretrain"):
        candidate = completed_stage_checkpoint(config, stage, name)
        if candidate is not None:
            return candidate
    root = Path(config["run"]["output_dir"]) / config["run"]["experiment_name"]
    raise FileNotFoundError(f"Could not resolve checkpoint '{name}' under {root}.")


def same_data_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def load_eval_text_samples(config: dict[str, Any], dataset_type: str | None = None) -> list[Any]:
    """Use the configured test file, or the hash-split test partition for one shared file."""

    dataset_type = dataset_type or config["data"].get("dataset_type", "pretrain")
    if config["data"].get("sources"):
        return load_text_samples_from_config(
            config,
            split="test",
            dataset_type=dataset_type,
            fallback_path=config["data"].get("test_file") or config["data"]["valid_file"],
        )
    data_cfg = config["data"]
    test_file = data_cfg.get("test_file") or data_cfg["valid_file"]
    if data_cfg.get("hash_split", True) and same_data_path(test_file, data_cfg["train_file"]):
        samples = load_text_samples(data_cfg["train_file"], config, dataset_type=dataset_type)
        splits = split_samples(samples, config)
        return splits["test"] or splits["valid"] or splits["train"]
    return load_text_samples(test_file, config, dataset_type=dataset_type)


def evaluate_checkpoint(config: dict[str, Any], logger: Any, checkpoint: Path) -> dict[str, Any]:
    """Evaluate one checkpoint on the configured test/valid file."""

    device = choose_device(config, logger)
    configure_torch_performance(config, device, logger)
    tokenizer = load_tokenizer(config)
    model_config = with_tokenizer_vocab(config, tokenizer.vocab_size)
    model = load_model_from_checkpoint(checkpoint, model_config, map_location=device).to(device)
    dataset_type = checkpoint_dataset_type(checkpoint, config["data"].get("dataset_type", "pretrain"))
    assistant_only = dataset_type == "sft" and bool(config["train"].get("assistant_only_loss", True))
    eval_batch_size = max(1, int(config["eval"].get("batch_size", 8)))
    if config["data"].get("streaming", False) and config["data"].get("sources"):
        dataset, collate, _ = build_token_shard_dataset(
            config,
            tokenizer,
            split="test",
            dataset_type=dataset_type,
            assistant_only_loss=assistant_only,
            logger=logger,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=eval_batch_size, collate_fn=collate)
    else:
        samples = load_eval_text_samples(config, dataset_type)
        tokenized = tokenize_training_samples(samples, tokenizer, config, assistant_only_loss=assistant_only)
        dataset, collate = make_torch_dataset(tokenized, tokenizer.pad_id)
        loader = torch.utils.data.DataLoader(dataset, batch_size=eval_batch_size, collate_fn=collate)
    metrics = evaluate_loss(model, loader, device, config)
    metrics["checkpoint"] = str(checkpoint)
    return metrics


def evaluate_dpo(config: dict[str, Any], logger: Any, checkpoint: Path) -> dict[str, Any] | None:
    """Evaluate DPO chosen/rejected accuracy when preference data is available."""

    if checkpoint_stage(checkpoint) != "dpo" and not config["dpo"].get("enabled", False):
        return None
    preference_path = Path(
        config["dpo"].get("test_file")
        or config["dpo"].get("valid_file")
        or config["dpo"].get("train_file")
        or config["data"].get("test_file")
        or config["data"]["valid_file"]
    )
    if not preference_path.exists():
        raise FileNotFoundError(f"DPO evaluation data not found: {preference_path}")
    samples = load_preference_samples(preference_path, config, purpose="evaluation")
    if not samples:
        return None
    device = choose_device(config, logger)
    configure_torch_performance(config, device, logger)
    tokenizer = load_tokenizer(config)
    model_config = with_tokenizer_vocab(config, tokenizer.vocab_size)
    model = load_model_from_checkpoint(checkpoint, model_config, map_location=device).to(device)
    model.eval()
    correct = 0
    margins = []
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

        sample_iter = tqdm(samples, desc="dpo eval", unit="pair", leave=False)
    except Exception:
        sample_iter = samples
    with torch.no_grad():
        for sample in sample_iter:
            prompt_ids = [tokenizer.bos_id, *tokenizer.encode(sample.prompt, add_special_tokens=False)]
            chosen_ids, chosen_labels = make_sequence(
                prompt_ids, tokenizer.encode(sample.chosen, add_special_tokens=False)
            )
            rejected_ids, rejected_labels = make_sequence(
                prompt_ids, tokenizer.encode(sample.rejected, add_special_tokens=False)
            )
            max_len = max(len(chosen_ids), len(rejected_ids))

            def pad(ids, row_labels, padded_length=max_len):
                input_ids = torch.full((1, padded_length), tokenizer.pad_id, dtype=torch.long, device=device)
                labels = torch.full((1, padded_length), -100, dtype=torch.long, device=device)
                mask = torch.zeros((1, padded_length), dtype=torch.bool, device=device)
                input_ids[0, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                labels[0, : len(row_labels)] = torch.tensor(row_labels, dtype=torch.long, device=device)
                mask[0, : len(ids)] = True
                return input_ids, labels, mask

            c = sequence_logprob(model, *pad(chosen_ids, chosen_labels))
            r = sequence_logprob(model, *pad(rejected_ids, rejected_labels))
            margin = float((c - r).item())
            margins.append(margin)
            correct += int(margin > 0)
    return {
        "dpo_sample_count": len(samples),
        "chosen_rejected_accuracy": correct / max(1, len(samples)),
        "reward_margin": sum(margins) / max(1, len(margins)),
    }


def run_memory_eval(config: dict[str, Any], logger: Any, checkpoint: Path) -> dict[str, Any] | None:
    """Run operator-supplied, file-backed multiturn memory probes."""

    mem_cfg = config["eval"]["multiturn_memory"]
    if not mem_cfg.get("enabled", False):
        return None
    from .inference import TextGenerator

    probe_file = mem_cfg.get("file")
    if not probe_file:
        raise ValueError("Multiturn memory evaluation requires a private eval.multiturn_memory.file.")
    with Path(probe_file).open("r", encoding="utf-8") as handle:
        probes = [json.loads(line) for line in handle if line.strip()]
    if not probes:
        raise ValueError("Multiturn memory evaluation file is empty.")
    generator = TextGenerator(config, logger, checkpoint=checkpoint, enable_memory=False)
    correct = 0
    try:
        from tqdm import tqdm

        probes = tqdm(probes, desc="memory eval", unit="probe", leave=False)
    except Exception:
        pass
    for probe in probes:
        rendered = "\n".join(f"{m['role']}: {m['content']}" for m in probe["messages"])
        answer = str(probe["answer"])
        output = generator.generate(prompt=rendered)
        correct += int(answer in output)
    return {"multiturn_memory_n_turn_accuracy": correct / max(1, len(probes)), "memory_probe_count": len(probes)}


def run_eval(config: dict[str, Any], logger: Any) -> None:
    """Evaluate configured latest/best checkpoints and write JSONL + markdown summary."""

    preflight_knowledge_pilot(config)
    output_dir = logger.log_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "eval_results.jsonl"
    summary_path = output_dir / "eval_summary.md"

    # Cache entries are reusable only when both the checkpoint and every
    # evaluation-relevant setting/data signature are unchanged.
    cache: dict[str, dict[str, Any]] = {}
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                safe_row = _safe_cached_evaluation_row(row)
                if safe_row is not None:
                    cache[safe_row["checkpoint"]] = safe_row

    checkpoint_names = config["eval"].get("checkpoints", ["latest", "best"])
    results: list[dict[str, Any]] = []
    try:
        from tqdm import tqdm

        names_iter = tqdm(checkpoint_names, desc="eval checkpoints", unit="ckpt")
    except Exception:
        names_iter = checkpoint_names
    for name in names_iter:
        checkpoint = resolve_checkpoint(config, name)
        checkpoint_name = str(checkpoint)
        fingerprint = evaluation_fingerprint(config, checkpoint)
        cached = cache.get(checkpoint_name)
        if cached and cached.get("evaluation_fingerprint") == fingerprint:
            logger.info(f"Skipping already-evaluated checkpoint {checkpoint}.")
            results.append(cached)
            continue
        metrics = evaluate_checkpoint(config, logger, checkpoint)
        dpo_metrics = evaluate_dpo(config, logger, checkpoint)
        memory_metrics = run_memory_eval(config, logger, checkpoint)
        knowledge_metrics = run_knowledge_pilot(config, logger, checkpoint)
        if dpo_metrics:
            metrics.update(dpo_metrics)
        if memory_metrics:
            metrics.update(memory_metrics)
        if knowledge_metrics:
            metrics.update(knowledge_metrics)
        metrics["evaluation_fingerprint"] = fingerprint
        results.append(metrics)
        cache[checkpoint_name] = metrics
        atomic_write_jsonl(jsonl_path, cache.values())
        logger.info(f"Evaluation metrics: {json.dumps(metrics, ensure_ascii=False)}")

    # Drop cache entries for checkpoints no longer requested and normalize the
    # order for deterministic diffs and downstream parsing.
    atomic_write_jsonl(jsonl_path, results)

    lines = ["# Evaluation Summary", ""]
    for metrics in results:
        lines.append(f"## {metrics['checkpoint']}")
        lines.append(f"- validation/test loss: {metrics.get('valid_loss'):.6f}")
        lines.append(f"- perplexity: {metrics.get('perplexity')}")
        lines.append(f"- token accuracy: {metrics.get('token_accuracy'):.4f}")
        if "chosen_rejected_accuracy" in metrics:
            lines.append(f"- DPO chosen/rejected accuracy: {metrics['chosen_rejected_accuracy']:.4f}")
            lines.append(f"- DPO reward margin: {metrics['reward_margin']:.4f}")
        if "multiturn_memory_n_turn_accuracy" in metrics:
            lines.append(f"- multiturn memory accuracy: {metrics['multiturn_memory_n_turn_accuracy']:.4f}")
        if "correct_count" in metrics:
            lines.append(f"- private knowledge pilot correct: {metrics['correct_count']}/{metrics['item_count']}")
            lines.append(f"- private knowledge pilot accuracy: {metrics['accuracy']:.4f}")
            lines.append(f"- private knowledge pilot parse rate: {metrics['parse_rate']:.4f}")
            lines.append(f"- private knowledge pilot passed: {str(metrics['passed']).lower()}")
        lines.append("")
    atomic_write_text(summary_path, "\n".join(lines))
    logger.info(f"Wrote evaluation summary to {summary_path}.")
