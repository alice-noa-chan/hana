"""Private, contamination-aware multiple-choice evaluation helpers.

This module intentionally treats evaluation questions as local-only inputs.
It validates them without logging their contents and returns aggregate metrics
that are safe to place in ordinary run artifacts.
"""

from __future__ import annotations

import json
import os
import re
import string
import unicodedata
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_text
from .data_governance import content_hash, load_benchmark_denylist
from .data_reader import clean_text
from .errors import DataPolicyError

_REQUIRED_FIELDS = frozenset({"id", "question", "choices", "answer"})
_PROMPT_FIELDS = frozenset({"labels", "question", "choices"})


@dataclass(frozen=True)
class MultipleChoiceItem:
    """One validated private evaluation item in deterministic label order."""

    item_id: str
    question: str
    choices: tuple[tuple[str, str], ...]
    answer: str


def _nonempty_string(value: Any, *, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Knowledge pilot line {line_number} field '{field}' must be a non-empty string.")
    return value.strip()


def load_multiple_choice_items(
    path: str | Path,
    *,
    choice_labels: list[str] | tuple[str, ...],
    item_count: int,
) -> list[MultipleChoiceItem]:
    """Load a strict JSONL pilot without reordering, sampling, or logging rows."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError("The configured private knowledge pilot file does not exist.")
    labels = tuple(choice_labels)
    expected_choice_fields = set(labels)
    items: list[MultipleChoiceItem] = []
    identifiers: set[str] = set()

    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("The private knowledge pilot file must be valid UTF-8.") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"Knowledge pilot line {line_number} is blank; strict JSONL does not allow blank lines.")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Knowledge pilot line {line_number} is not valid JSON.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Knowledge pilot line {line_number} must be a JSON object.")
        if set(row) != _REQUIRED_FIELDS:
            raise ValueError(f"Knowledge pilot line {line_number} must contain exactly: answer, choices, id, question.")

        item_id = _nonempty_string(row["id"], field="id", line_number=line_number)
        if item_id in identifiers:
            raise ValueError(f"Knowledge pilot line {line_number} repeats an earlier id.")
        identifiers.add(item_id)
        question = _nonempty_string(row["question"], field="question", line_number=line_number)

        choices = row["choices"]
        if not isinstance(choices, dict) or set(choices) != expected_choice_fields:
            expected = ", ".join(labels)
            raise ValueError(
                f"Knowledge pilot line {line_number} choices must be an object with exactly these labels: {expected}."
            )
        ordered_choices = tuple(
            (label, _nonempty_string(choices[label], field=f"choices.{label}", line_number=line_number))
            for label in labels
        )
        answer = _nonempty_string(row["answer"], field="answer", line_number=line_number)
        if answer not in expected_choice_fields:
            raise ValueError(f"Knowledge pilot line {line_number} answer must be one configured choice label.")
        items.append(MultipleChoiceItem(item_id, question, ordered_choices, answer))

    if len(items) != item_count:
        raise ValueError(
            f"Knowledge pilot requires exactly {item_count} items; the private file contains {len(items)}."
        )
    return items


def load_prompt_template(path: str | Path | None) -> str | None:
    """Load and validate an optional private, language-natural prompt template."""

    if not path:
        return None
    template_path = Path(path)
    if not template_path.is_file():
        raise FileNotFoundError("The configured private knowledge-pilot prompt file does not exist.")
    template = template_path.read_text(encoding="utf-8").strip()
    if not template:
        raise ValueError("The private knowledge-pilot prompt template must not be empty.")
    fields = {
        field_name
        for _literal, field_name, _format_spec, _conversion in string.Formatter().parse(template)
        if field_name is not None
    }
    if fields != _PROMPT_FIELDS:
        raise ValueError(
            "The private knowledge-pilot prompt template must use labels, question, and choices once or more."
        )
    return template


def render_multiple_choice_prompt(item: MultipleChoiceItem, prompt_template: str | None = None) -> str:
    """Render one private item with a generic or operator-supplied instruction."""

    labels = ", ".join(label for label, _choice in item.choices)
    choices = "\n".join(f"{label}. {choice}" for label, choice in item.choices)
    template = prompt_template or (
        "Select the best answer to the multiple-choice question below.\n"
        "Return exactly one label from: {labels}.\n"
        "Do not include an explanation or any other text.\n\n"
        "Question:\n{question}\n\n"
        "Choices:\n{choices}\n\n"
        "Answer:"
    )
    return template.format(labels=labels, question=item.question, choices=choices)


def parse_choice_label(output: str, choice_labels: list[str] | tuple[str, ...]) -> str | None:
    """Parse only a single, exact label after Unicode compatibility normalization."""

    normalized = unicodedata.normalize("NFKC", output).strip()
    pattern = "(?:" + "|".join(re.escape(label) for label in choice_labels) + ")"
    return normalized if re.fullmatch(pattern, normalized) else None


def required_denylist_hashes(
    items: list[MultipleChoiceItem],
    *,
    normalize_nfkc: bool,
    prompt_template: str | None = None,
) -> set[str]:
    """Hash full questions and rendered prompts, but not short choices or answers."""

    required: set[str] = set()
    question_hashes: set[str] = set()
    prompt_hashes: set[str] = set()
    for item in items:
        question = clean_text(item.question, normalize_nfkc)
        normalized_item = MultipleChoiceItem(
            item_id=item.item_id,
            question=question,
            choices=tuple((label, clean_text(choice, normalize_nfkc)) for label, choice in item.choices),
            answer=item.answer,
        )
        question_hash = content_hash(question)
        rendered_prompt = render_multiple_choice_prompt(normalized_item, prompt_template)
        prompt_hash = content_hash(clean_text(rendered_prompt, normalize_nfkc))
        if question_hash in question_hashes or prompt_hash in prompt_hashes:
            raise ValueError("The private knowledge pilot contains duplicate canonical questions or prompts.")
        question_hashes.add(question_hash)
        prompt_hashes.add(prompt_hash)
        required.update((question_hash, prompt_hash))
    return required


@contextmanager
def _exclusive_file_lock(path: Path):
    """Hold a cross-process lock for one private registry update."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def verify_denylist_coverage(
    config: dict[str, Any],
    items: list[MultipleChoiceItem],
    *,
    normalize_nfkc: bool,
    prompt_template: str | None = None,
) -> None:
    """Fail before model loading when any required canonical hash is absent."""

    required = required_denylist_hashes(
        items,
        normalize_nfkc=normalize_nfkc,
        prompt_template=prompt_template,
    )
    denylist = load_benchmark_denylist(config)
    missing_count = len(required - denylist)
    if missing_count:
        raise DataPolicyError(
            "Private knowledge pilot denylist coverage is incomplete: "
            f"{missing_count} required canonical hash(es) are missing."
        )


def quarantine_knowledge_pilot(config: dict[str, Any]) -> dict[str, Any]:
    """Atomically add pilot hashes to the private benchmark denylist."""

    pilot = config["eval"].get("knowledge_pilot", {})
    if not pilot.get("file"):
        raise ValueError("eval.knowledge_pilot.file is required before quarantining the pilot.")
    items = load_multiple_choice_items(
        pilot["file"],
        choice_labels=pilot["choice_labels"],
        item_count=int(pilot["item_count"]),
    )
    prompt_template = load_prompt_template(pilot.get("prompt_file"))
    required = required_denylist_hashes(
        items,
        normalize_nfkc=bool(config["data"].get("normalize_nfkc", True)),
        prompt_template=prompt_template,
    )
    target_value = config.get("data_policy", {}).get("benchmark_denylist_path")
    if not target_value:
        raise ValueError("data_policy.benchmark_denylist_path is required before quarantining the pilot.")
    target = Path(target_value)
    lock_path = target.with_name(f".{target.name}.quarantine.lock")
    with _exclusive_file_lock(lock_path):
        existing: set[str] = set()
        if target.exists():
            relaxed = deepcopy(config)
            relaxed["data_policy"]["require_benchmark_denylist"] = False
            existing = load_benchmark_denylist(relaxed)
        merged = existing | required
        atomic_write_text(target, "".join(f"{digest}\n" for digest in sorted(merged)))
    return {
        "added_hashes": len(merged - existing),
        "total_hashes": len(merged),
        "item_count": len(items),
        "path": str(target),
    }


def preflight_knowledge_pilot(config: dict[str, Any]) -> None:
    """Validate the enabled pilot and its quarantine before any model loads."""

    pilot = config["eval"].get("knowledge_pilot", {})
    if not pilot.get("enabled", False):
        return
    items = load_multiple_choice_items(
        pilot["file"],
        choice_labels=pilot["choice_labels"],
        item_count=int(pilot["item_count"]),
    )
    prompt_template = load_prompt_template(pilot.get("prompt_file"))
    if pilot.get("require_denylist_coverage", True):
        verify_denylist_coverage(
            config,
            items,
            normalize_nfkc=bool(config["data"].get("normalize_nfkc", True)),
            prompt_template=prompt_template,
        )


def run_knowledge_pilot(
    config: dict[str, Any],
    logger: Any,
    checkpoint: Path,
) -> dict[str, int | float | bool] | None:
    """Evaluate a fixed private pilot and return aggregate-only metrics."""

    pilot = config["eval"].get("knowledge_pilot", {})
    if not pilot.get("enabled", False):
        return None

    preflight_knowledge_pilot(config)
    items = load_multiple_choice_items(
        pilot["file"],
        choice_labels=pilot["choice_labels"],
        item_count=int(pilot["item_count"]),
    )
    prompt_template = load_prompt_template(pilot.get("prompt_file"))
    normalize_nfkc = bool(config["data"].get("normalize_nfkc", True))
    isolated_config = config.mutable_copy() if hasattr(config, "mutable_copy") else deepcopy(config)
    isolated_config["inference"]["reasoning_mode"] = str(pilot["reasoning_mode"])
    isolated_config["inference"]["token_trace_file"] = None
    isolated_config["reasoning"]["expose_reasoning_trace"] = False
    isolated_config["reasoning"]["save_reasoning_trace"] = False
    isolated_config["experiments"]["enabled"] = False

    from .inference import TextGenerator

    generator = TextGenerator(isolated_config, logger, checkpoint=checkpoint, enable_memory=False)
    generation_settings = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": int(pilot["max_new_tokens"]),
        "reasoning_mode": str(pilot["reasoning_mode"]),
        "token_trace_file": None,
        "strict_context_fit": True,
    }
    correct_count = 0
    parsed_count = 0
    for item in items:
        normalized_item = MultipleChoiceItem(
            item_id=item.item_id,
            question=clean_text(item.question, normalize_nfkc),
            choices=tuple((label, clean_text(choice, normalize_nfkc)) for label, choice in item.choices),
            answer=item.answer,
        )
        prompt = render_multiple_choice_prompt(normalized_item, prompt_template)
        output = generator.generate(prompt=prompt, generation_settings=generation_settings)
        parsed = parse_choice_label(output, pilot["choice_labels"])
        parsed_count += int(parsed is not None)
        correct_count += int(parsed == item.answer)

    item_count = len(items)
    return {
        "correct_count": correct_count,
        "item_count": item_count,
        "accuracy": correct_count / item_count,
        "parse_rate": parsed_count / item_count,
        "passed": correct_count >= int(pilot["required_correct"]),
    }
