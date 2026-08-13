from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_pipeline.artifacts import evaluation_fingerprint
from llm_pipeline.config import DEFAULT_CONFIG
from llm_pipeline.data_governance import content_hash
from llm_pipeline.data_reader import clean_text
from llm_pipeline.errors import DataPolicyError
from llm_pipeline.evaluation import run_eval
from llm_pipeline.multiple_choice import (
    load_multiple_choice_items,
    load_prompt_template,
    parse_choice_label,
    preflight_knowledge_pilot,
    quarantine_knowledge_pilot,
    render_multiple_choice_prompt,
    required_denylist_hashes,
    run_knowledge_pilot,
    verify_denylist_coverage,
)


def _rows() -> list[dict]:
    return [
        {
            "id": "synthetic-1",
            "question": "Synthetic prompt alpha?",
            "choices": {"B": "Second option", "A": "First option"},
            "answer": "A",
        },
        {
            "id": "synthetic-2",
            "question": "Synthetic prompt beta?",
            "choices": {"A": "Left option", "B": "Right option"},
            "answer": "B",
        },
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _pilot_config(pilot_file: Path, denylist_file: Path) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["eval"]["knowledge_pilot"].update(
        enabled=True,
        file=str(pilot_file),
        item_count=2,
        required_correct=1,
        choice_labels=["A", "B"],
        reasoning_mode="high",
        max_new_tokens=4,
        require_denylist_coverage=True,
    )
    config["data_policy"]["benchmark_denylist_path"] = str(denylist_file)
    return config


def test_loader_preserves_file_order_and_uses_configured_choice_order(tmp_path: Path) -> None:
    pilot_file = tmp_path / "pilot.jsonl"
    _write_jsonl(pilot_file, _rows())

    items = load_multiple_choice_items(pilot_file, choice_labels=["A", "B"], item_count=2)

    assert [item.item_id for item in items] == ["synthetic-1", "synthetic-2"]
    assert items[0].choices == (("A", "First option"), ("B", "Second option"))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: rows[0].update(extra="rejected"), "contain exactly"),
        (lambda rows: rows[1].update(id=rows[0]["id"]), "repeats an earlier id"),
        (lambda rows: rows[0].update(answer="C"), "one configured choice label"),
        (lambda rows: rows[0].update(choices={"A": "Only one"}), "exactly these labels"),
    ],
)
def test_loader_rejects_nonconforming_rows(tmp_path: Path, change, message: str) -> None:
    rows = _rows()
    change(rows)
    pilot_file = tmp_path / "pilot.jsonl"
    _write_jsonl(pilot_file, rows)

    with pytest.raises(ValueError, match=message):
        load_multiple_choice_items(pilot_file, choice_labels=["A", "B"], item_count=2)


def test_loader_requires_exact_item_count_and_rejects_blank_lines(tmp_path: Path) -> None:
    pilot_file = tmp_path / "pilot.jsonl"
    _write_jsonl(pilot_file, _rows())
    with pytest.raises(ValueError, match="requires exactly 3 items"):
        load_multiple_choice_items(pilot_file, choice_labels=["A", "B"], item_count=3)

    pilot_file.write_text(json.dumps(_rows()[0]) + "\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strict JSONL"):
        load_multiple_choice_items(pilot_file, choice_labels=["A", "B"], item_count=1)


def test_quarantine_rejects_duplicate_canonical_questions(tmp_path: Path) -> None:
    rows = _rows()
    rows[1]["question"] = "  SYNTHETIC   PROMPT ALPHA?  "
    pilot_file = tmp_path / "pilot.jsonl"
    denylist_file = tmp_path / "denylist.txt"
    _write_jsonl(pilot_file, rows)
    config = _pilot_config(pilot_file, denylist_file)

    with pytest.raises(ValueError, match="duplicate canonical"):
        quarantine_knowledge_pilot(config)


def test_choice_parser_accepts_only_one_nfkc_normalized_label() -> None:
    assert parse_choice_label("\uff21", ["A", "B"]) == "A"
    assert parse_choice_label("  B\n", ["A", "B"]) == "B"
    assert parse_choice_label("A.", ["A", "B"]) is None
    assert parse_choice_label("Answer: A", ["A", "B"]) is None
    assert parse_choice_label("A\nB", ["A", "B"]) is None


def test_private_prompt_template_supports_the_evaluation_language(tmp_path: Path) -> None:
    pilot_file = tmp_path / "pilot.jsonl"
    prompt_file = tmp_path / "prompt.txt"
    _write_jsonl(pilot_file, _rows())
    prompt_file.write_text(
        "Allowed labels: {labels}\nPrivate question: {question}\nPrivate choices:\n{choices}\nLabel:",
        encoding="utf-8",
    )
    item = load_multiple_choice_items(pilot_file, choice_labels=["A", "B"], item_count=2)[0]

    template = load_prompt_template(prompt_file)

    rendered = render_multiple_choice_prompt(item, template)
    assert "Allowed labels: A, B" in rendered
    assert item.question in rendered
    assert "A. First option" in rendered

    prompt_file.write_text("Missing fields: {question}", encoding="utf-8")
    with pytest.raises(ValueError, match="labels, question, and choices"):
        load_prompt_template(prompt_file)


def test_prompt_hash_matches_nfkc_normalization_used_for_generation(tmp_path: Path) -> None:
    pilot_file = tmp_path / "pilot.jsonl"
    prompt_file = tmp_path / "prompt.txt"
    _write_jsonl(pilot_file, _rows())
    prompt_file.write_text(
        "\uff2c\uff41\uff42\uff45\uff4c\uff53: {labels}\n"
        "\uff31\uff55\uff45\uff53\uff54\uff49\uff4f\uff4e: {question}\n"
        "\uff23\uff48\uff4f\uff49\uff43\uff45\uff53:\n{choices}",
        encoding="utf-8",
    )
    items = load_multiple_choice_items(pilot_file, choice_labels=["A", "B"], item_count=2)
    template = load_prompt_template(prompt_file)
    actual_prompt = clean_text(render_multiple_choice_prompt(items[0], template), True)

    required = required_denylist_hashes(items, normalize_nfkc=True, prompt_template=template)

    assert content_hash(actual_prompt) in required


def test_denylist_requires_questions_and_full_prompts_only(tmp_path: Path) -> None:
    pilot_file = tmp_path / "pilot.jsonl"
    denylist_file = tmp_path / "denylist.txt"
    _write_jsonl(pilot_file, _rows())
    config = _pilot_config(pilot_file, denylist_file)
    items = load_multiple_choice_items(pilot_file, choice_labels=["A", "B"], item_count=2)
    required = required_denylist_hashes(items, normalize_nfkc=True)
    denylist_file.write_text("\n".join(sorted(required)) + "\n", encoding="utf-8")

    verify_denylist_coverage(config, items, normalize_nfkc=True)

    assert content_hash(items[0].question) in required
    assert content_hash(render_multiple_choice_prompt(items[0])) in required
    assert content_hash(items[0].choices[0][1]) not in required
    assert content_hash(items[0].answer) not in required


def test_missing_denylist_coverage_does_not_reveal_private_content(tmp_path: Path) -> None:
    pilot_file = tmp_path / "pilot.jsonl"
    denylist_file = tmp_path / "denylist.txt"
    _write_jsonl(pilot_file, _rows())
    denylist_file.write_text(content_hash("unrelated") + "\n", encoding="utf-8")
    config = _pilot_config(pilot_file, denylist_file)
    items = load_multiple_choice_items(pilot_file, choice_labels=["A", "B"], item_count=2)

    with pytest.raises(DataPolicyError) as caught:
        verify_denylist_coverage(config, items, normalize_nfkc=True)

    message = str(caught.value)
    assert "Synthetic prompt" not in message
    assert "First option" not in message


def test_quarantine_atomically_merges_required_hashes(tmp_path: Path) -> None:
    pilot_file = tmp_path / "pilot.jsonl"
    denylist_file = tmp_path / "denylist.txt"
    _write_jsonl(pilot_file, _rows())
    existing = content_hash("unrelated")
    denylist_file.write_text(existing + "\n", encoding="utf-8")
    config = _pilot_config(pilot_file, denylist_file)

    result = quarantine_knowledge_pilot(config)

    items = load_multiple_choice_items(pilot_file, choice_labels=["A", "B"], item_count=2)
    expected = required_denylist_hashes(items, normalize_nfkc=True) | {existing}
    assert set(denylist_file.read_text(encoding="utf-8").splitlines()) == expected
    assert result["added_hashes"] == len(expected) - 1
    assert result["item_count"] == 2


def test_pilot_checks_denylist_before_loading_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pilot_file = tmp_path / "pilot.jsonl"
    denylist_file = tmp_path / "denylist.txt"
    _write_jsonl(pilot_file, _rows())
    denylist_file.write_text(content_hash("unrelated") + "\n", encoding="utf-8")
    config = _pilot_config(pilot_file, denylist_file)

    class ModelMustNotLoad:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("model loaded before denylist validation")

    monkeypatch.setattr("llm_pipeline.inference.TextGenerator", ModelMustNotLoad)

    with pytest.raises(DataPolicyError, match="coverage is incomplete"):
        run_knowledge_pilot(config, SimpleNamespace(), Path("checkpoint"))


def test_eval_preflight_rejects_missing_coverage_before_checkpoint_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pilot_file = tmp_path / "pilot.jsonl"
    denylist_file = tmp_path / "denylist.txt"
    _write_jsonl(pilot_file, _rows())
    denylist_file.write_text(content_hash("unrelated") + "\n", encoding="utf-8")
    config = _pilot_config(pilot_file, denylist_file)

    def checkpoint_work_must_not_start(*_args):
        raise AssertionError("checkpoint work started before preflight")

    monkeypatch.setattr("llm_pipeline.evaluation.resolve_checkpoint", checkpoint_work_must_not_start)

    with pytest.raises(DataPolicyError, match="coverage is incomplete"):
        run_eval(config, SimpleNamespace(log_dir=tmp_path / "logs", info=lambda _message: None))

    with pytest.raises(DataPolicyError, match="coverage is incomplete"):
        preflight_knowledge_pilot(config)


def test_pilot_uses_isolated_deterministic_generation_and_returns_aggregates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pilot_file = tmp_path / "pilot.jsonl"
    denylist_file = tmp_path / "denylist.txt"
    _write_jsonl(pilot_file, _rows())
    config = _pilot_config(pilot_file, denylist_file)
    items = load_multiple_choice_items(pilot_file, choice_labels=["A", "B"], item_count=2)
    required = required_denylist_hashes(items, normalize_nfkc=True)
    denylist_file.write_text("\n".join(sorted(required)) + "\n", encoding="utf-8")
    calls: list[dict] = []
    outputs = iter(["A", "not-a-label"])

    class FakeGenerator:
        def __init__(self, isolated, _logger, checkpoint, enable_memory):
            calls.append(
                {
                    "checkpoint": checkpoint,
                    "enable_memory": enable_memory,
                    "trace_file": isolated["inference"]["token_trace_file"],
                    "save_reasoning": isolated["reasoning"]["save_reasoning_trace"],
                    "expose_reasoning": isolated["reasoning"]["expose_reasoning_trace"],
                    "experiments": isolated["experiments"]["enabled"],
                    "reasoning_mode": isolated["inference"]["reasoning_mode"],
                }
            )

        def generate(self, prompt, generation_settings):
            calls.append({"prompt": prompt, "settings": generation_settings})
            return next(outputs)

    monkeypatch.setattr("llm_pipeline.inference.TextGenerator", FakeGenerator)

    metrics = run_knowledge_pilot(config, SimpleNamespace(), Path("checkpoint"))

    assert metrics == {
        "correct_count": 1,
        "item_count": 2,
        "accuracy": 0.5,
        "parse_rate": 0.5,
        "passed": True,
    }
    assert calls[0] == {
        "checkpoint": Path("checkpoint"),
        "enable_memory": False,
        "trace_file": None,
        "save_reasoning": False,
        "expose_reasoning": False,
        "experiments": False,
        "reasoning_mode": "high",
    }
    assert calls[1]["settings"] == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 4,
        "reasoning_mode": "high",
        "token_trace_file": None,
        "strict_context_fit": True,
    }
    assert "Synthetic prompt alpha?" in calls[1]["prompt"]
    assert "Synthetic prompt" not in json.dumps(metrics)


def test_evaluation_fingerprint_uses_private_file_content_hash(tmp_path: Path) -> None:
    pilot_file = tmp_path / "pilot.jsonl"
    denylist_file = tmp_path / "denylist.txt"
    pilot_file.write_text("first-content\n", encoding="utf-8")
    config = _pilot_config(pilot_file, denylist_file)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    original_stat = pilot_file.stat()
    before = evaluation_fingerprint(config, checkpoint)

    pilot_file.write_text("other-content\n", encoding="utf-8")
    os.utime(pilot_file, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert pilot_file.stat().st_size == original_stat.st_size
    assert evaluation_fingerprint(config, checkpoint) != before


def test_run_eval_merges_only_aggregate_pilot_metrics_into_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["eval"]["checkpoints"] = ["best"]
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    logger = SimpleNamespace(log_dir=tmp_path / "logs", info=lambda _message: None)
    monkeypatch.setattr("llm_pipeline.evaluation.resolve_checkpoint", lambda _config, _name: checkpoint)
    monkeypatch.setattr("llm_pipeline.evaluation.evaluation_fingerprint", lambda _config, _checkpoint: "fingerprint")
    monkeypatch.setattr(
        "llm_pipeline.evaluation.evaluate_checkpoint",
        lambda _config, _logger, _checkpoint: {
            "checkpoint": str(checkpoint),
            "valid_loss": 1.0,
            "perplexity": 2.0,
            "token_accuracy": 0.5,
        },
    )
    monkeypatch.setattr("llm_pipeline.evaluation.evaluate_dpo", lambda *_args: None)
    monkeypatch.setattr("llm_pipeline.evaluation.run_memory_eval", lambda *_args: None)
    monkeypatch.setattr(
        "llm_pipeline.evaluation.run_knowledge_pilot",
        lambda *_args: {
            "correct_count": 7,
            "item_count": 10,
            "accuracy": 0.7,
            "parse_rate": 0.9,
            "passed": True,
        },
    )

    run_eval(config, logger)

    results = (logger.log_dir / "eval_results.jsonl").read_text(encoding="utf-8")
    summary = (logger.log_dir / "eval_summary.md").read_text(encoding="utf-8")
    assert "private knowledge pilot correct: 7/10" in summary
    assert "private knowledge pilot parse rate: 0.9000" in summary
    assert "correct_count" in results
    assert "question" not in results
    assert "answer" not in results
    assert "output" not in results
    assert "trace" not in results


def test_run_eval_does_not_reuse_cache_rows_with_private_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["eval"]["checkpoints"] = ["best"]
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    logger = SimpleNamespace(log_dir=tmp_path / "logs", info=lambda _message: None)
    logger.log_dir.mkdir()
    poisoned = {
        "checkpoint": str(checkpoint),
        "valid_loss": 9.0,
        "perplexity": 9.0,
        "token_accuracy": 0.0,
        "evaluation_fingerprint": "fingerprint",
        "question": "must not survive",
    }
    _write_jsonl(logger.log_dir / "eval_results.jsonl", [poisoned])
    calls = {"evaluate": 0}

    monkeypatch.setattr("llm_pipeline.evaluation.resolve_checkpoint", lambda _config, _name: checkpoint)
    monkeypatch.setattr("llm_pipeline.evaluation.evaluation_fingerprint", lambda *_args: "fingerprint")

    def fresh_metrics(*_args):
        calls["evaluate"] += 1
        return {
            "checkpoint": str(checkpoint),
            "valid_loss": 1.0,
            "valid_objective": 1.0,
            "perplexity": 2.0,
            "token_accuracy": 0.5,
        }

    monkeypatch.setattr("llm_pipeline.evaluation.evaluate_checkpoint", fresh_metrics)
    monkeypatch.setattr("llm_pipeline.evaluation.evaluate_dpo", lambda *_args: None)
    monkeypatch.setattr("llm_pipeline.evaluation.run_memory_eval", lambda *_args: None)
    monkeypatch.setattr("llm_pipeline.evaluation.run_knowledge_pilot", lambda *_args: None)

    run_eval(config, logger)

    output = (logger.log_dir / "eval_results.jsonl").read_text(encoding="utf-8")
    assert calls["evaluate"] == 1
    assert "must not survive" not in output
    assert "question" not in output
