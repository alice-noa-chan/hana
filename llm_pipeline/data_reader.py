"""Streaming readers and deterministic text/schema normalization."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
PRESERVED_TAGS = {
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
    "<user>",
    "<assistant>",
    "<system>",
    "<reasoning:off>",
    "<reasoning:low>",
    "<reasoning:medium>",
    "<reasoning:high>",
    "<mask>",
}
REASONING_MODES = frozenset({"low", "medium", "high"})
DEFAULT_REASONING_MODE = "medium"


@dataclass
class TextSample:
    """Normalized text sample with optional metadata."""

    text: str
    kind: str = "pretrain"
    meta: dict[str, Any] | None = None
    labels_mask: list[int] | None = None


@dataclass
class PreferenceSample:
    """DPO preference row."""

    prompt: str
    chosen: str
    rejected: str
    meta: dict[str, Any] | None = None


def clean_text(text: str, normalize_nfkc: bool = True) -> str:
    """Normalize text consistently before tokenizer and model training."""

    text = "" if text is None else str(text)
    if normalize_nfkc:
        text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = TAG_RE.sub(lambda match: match.group(0) if match.group(0) in PRESERVED_TAGS else " ", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\ufffd", "")
    return SPACE_RE.sub(" ", text).strip()


def escape_special_tokens(text: str, special_tokens: dict[str, str]) -> str:
    """Prevent untrusted chat content from impersonating control/role tokens."""

    for token in sorted(special_tokens.values(), key=len, reverse=True):
        if token in text:
            escaped = token.replace("<", "\u2039").replace(">", "\u203a")
            text = text.replace(token, escaped)
    return text


def stable_hash(value: str) -> int:
    """Stable integer hash used for reproducible split decisions."""

    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def read_rows(path: str | Path, data_format: str) -> list[Any]:
    return list(iter_rows(path, data_format))


def iter_rows(path: str | Path, data_format: str) -> Iterator[Any]:
    """Yield JSONL, JSON, TXT, CSV, or TSV rows without loading JSONL eagerly."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    fmt = (data_format or file_path.suffix.lstrip(".")).lower()
    if fmt in {"jsonl", "jl"}:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {file_path}:{line_no}: {exc}") from exc
        return
    if fmt == "json":
        with file_path.open("r", encoding="utf-8") as handle:
            yield from iter_json_records(json.load(handle))
        return
    if fmt == "txt":
        yield {"text": file_path.read_text(encoding="utf-8")}
        return
    if fmt in {"csv", "tsv"}:
        delimiter = "\t" if fmt == "tsv" else ","
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle, delimiter=delimiter)
        return
    raise ValueError(f"Unsupported data format: {fmt}")


def iter_json_records(data: Any) -> Iterator[Any]:
    """Flatten common JSON dataset envelopes into row-like records."""

    if isinstance(data, list):
        for item in data:
            yield from iter_json_records(item)
        return
    if not isinstance(data, dict):
        yield data
        return
    if isinstance(data.get("paragraphs"), list):
        for paragraph in data["paragraphs"]:
            context = paragraph.get("context", "") if isinstance(paragraph, dict) else ""
            for qa in paragraph.get("qas", []) if isinstance(paragraph, dict) else []:
                answers = qa.get("answers") or []
                answer = answers[0].get("text", "") if answers and isinstance(answers[0], dict) else ""
                yield {
                    "instruction": qa.get("question", ""),
                    "input": context,
                    "output": answer,
                    "meta": {"id": qa.get("id"), "title": data.get("title")},
                }
        return
    if any(key in data for key in ("text", "messages", "instruction", "prompt", "ko", "ja")):
        yield data
        return
    for key in ("data", "rows", "records", "items", "examples"):
        value = data.get(key)
        if isinstance(value, list):
            for item in value:
                yield from iter_json_records(item)
            return
    yield data


def render_message_segments(
    messages: list[dict[str, Any]],
    special_tokens: dict[str, str],
    normalize_nfkc: bool = True,
    default_reasoning_mode: str = DEFAULT_REASONING_MODE,
) -> list[tuple[str, bool]]:
    """Render messages into text segments and assistant-loss flags.

    An assistant message with a non-empty ``reasoning`` field places its
    reasoning-mode control token before the assistant role cue.  The
    reasoning-off token separates the private reasoning target from the final
    answer target.  Messages without reasoning retain the original rendering.
    """

    segments: list[tuple[str, bool]] = []
    role_tokens = {
        "system": special_tokens.get("system", "<system>"),
        "user": special_tokens.get("user", "<user>"),
        "assistant": special_tokens.get("assistant", "<assistant>"),
    }
    for message in messages:
        role = str(message.get("role", "user")).lower()
        content = escape_special_tokens(
            clean_text(message.get("content", ""), normalize_nfkc),
            special_tokens,
        )
        role_token = role_tokens.get(role, f"<{role}>")
        reasoning = ""
        if role == "assistant":
            reasoning = escape_special_tokens(
                clean_text(message.get("reasoning", ""), normalize_nfkc),
                special_tokens,
            )
        if reasoning:
            requested_mode = str(message.get("reasoning_mode", default_reasoning_mode)).strip().lower()
            if requested_mode not in REASONING_MODES:
                raise ValueError(f"Unsupported reasoning_mode in assistant message: {requested_mode!r}.")
            mode = requested_mode
            reasoning_token = special_tokens.get(f"reasoning_{mode}", f"<reasoning:{mode}>")
            reasoning_off = special_tokens.get("reasoning_off", "<reasoning:off>")
            rendered = f"{reasoning_token}\n{role_token}\n{reasoning}\n{reasoning_off}\n{content}\n"
        else:
            rendered = f"{role_token}\n{content}\n"
        segments.append((rendered, role == "assistant"))
    return segments


def render_messages(
    messages: list[dict[str, Any]],
    special_tokens: dict[str, str],
    *,
    default_reasoning_mode: str = DEFAULT_REASONING_MODE,
) -> tuple[str, list[int]]:
    """Render chat messages and a character-level assistant-only loss mask."""

    parts: list[str] = []
    mask: list[int] = []
    for rendered, is_assistant in render_message_segments(
        messages,
        special_tokens,
        default_reasoning_mode=default_reasoning_mode,
    ):
        parts.append(rendered)
        mask.extend([1 if is_assistant else 0] * len(rendered))
    return "".join(parts), mask


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    """Normalize chat rows from role/content or from/value formats."""

    if not isinstance(messages, list):
        return []
    role_map = {
        "human": "user",
        "user": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "bot": "assistant",
        "system": "system",
    }
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        raw_role = message.get("role", message.get("from", "user"))
        raw_content = message.get("content", message.get("value", message.get("text", "")))
        role = role_map.get(str(raw_role).lower(), str(raw_role).lower())
        normalized_message = {"role": role, "content": str(raw_content)}
        if role == "assistant":
            if "reasoning" in message:
                normalized_message["reasoning"] = str(message.get("reasoning") or "")
            if "reasoning_mode" in message:
                requested_mode = str(message.get("reasoning_mode") or "").strip().lower()
                if requested_mode not in REASONING_MODES:
                    raise ValueError(f"Unsupported reasoning_mode in assistant message: {requested_mode!r}.")
                normalized_message["reasoning_mode"] = requested_mode
        normalized.append(normalized_message)
    return normalized


def get_source_value(source: dict[str, Any] | None, key: str, default: Any) -> Any:
    if source and key in source:
        return source[key]
    return default


def field_value(row: dict[str, Any], names: str | list[str] | tuple[str, ...]) -> Any:
    if isinstance(names, str):
        names = [names]
    for name in names:
        if name in row:
            return row[name]
    return None
