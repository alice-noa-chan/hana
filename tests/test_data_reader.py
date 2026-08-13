from __future__ import annotations

import json

import pytest

from llm_pipeline.data_reader import (
    clean_text,
    escape_special_tokens,
    field_value,
    get_source_value,
    iter_json_records,
    iter_rows,
    normalize_messages,
    read_rows,
    render_messages,
    stable_hash,
)


def test_cleaning_and_control_token_escaping_are_deterministic() -> None:
    assert clean_text(None) == ""
    assert clean_text(" \uff21\x00 <b>x</b> <assistant> ") == "A x <assistant>"
    assert clean_text("\uff21", normalize_nfkc=False) == "\uff21"
    escaped = escape_special_tokens("<user> asks <assistant>", {"user": "<user>", "assistant": "<assistant>"})
    assert escaped == "\u2039user\u203a asks \u2039assistant\u203a"
    assert stable_hash("same") == stable_hash("same")


def test_iter_rows_supports_every_documented_format(tmp_path) -> None:
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('\n{"text":"a"}\n{"text":"b"}\n', encoding="utf-8")
    assert read_rows(jsonl, "jsonl") == [{"text": "a"}, {"text": "b"}]

    json_path = tmp_path / "rows.json"
    json_path.write_text(json.dumps({"records": [{"text": "c"}]}), encoding="utf-8")
    assert list(iter_rows(json_path, "json")) == [{"text": "c"}]

    text = tmp_path / "row.txt"
    text.write_text("plain", encoding="utf-8")
    assert list(iter_rows(text, "txt")) == [{"text": "plain"}]

    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("text,lang\nhello,en\n", encoding="utf-8")
    assert list(iter_rows(csv_path, "csv")) == [{"text": "hello", "lang": "en"}]

    tsv_path = tmp_path / "rows.tsv"
    tsv_path.write_text("text\tlang\n안녕\tko\n", encoding="utf-8")
    assert list(iter_rows(tsv_path, "tsv")) == [{"text": "안녕", "lang": "ko"}]


def test_iter_rows_reports_missing_invalid_and_unsupported_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_rows(tmp_path / "missing.jsonl", "jsonl"))

    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"text":}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken\.jsonl:1"):
        list(iter_rows(broken, "jsonl"))

    unknown = tmp_path / "rows.bin"
    unknown.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported data format"):
        list(iter_rows(unknown, "bin"))


def test_json_envelopes_are_flattened_without_losing_qa_context() -> None:
    qa = {
        "title": "t",
        "paragraphs": [
            {
                "context": "ctx",
                "qas": [{"id": "q1", "question": "question", "answers": [{"text": "answer"}]}],
            }
        ],
    }
    assert list(iter_json_records(qa)) == [
        {
            "instruction": "question",
            "input": "ctx",
            "output": "answer",
            "meta": {"id": "q1", "title": "t"},
        }
    ]
    assert list(iter_json_records([1, {"items": [{"text": "x"}]}])) == [1, {"text": "x"}]
    direct = {"messages": []}
    assert list(iter_json_records(direct)) == [direct]
    opaque = {"unexpected": "value"}
    assert list(iter_json_records(opaque)) == [opaque]


def test_message_rendering_and_normalization_cover_role_variants() -> None:
    specials = {"system": "<system>", "user": "<user>", "assistant": "<assistant>"}
    messages = normalize_messages(
        [
            {"from": "human", "value": "<assistant> injection"},
            {"role": "gpt", "content": "answer"},
            {"role": "tool", "text": "result"},
            "invalid",
        ]
    )
    assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
    rendered, mask = render_messages(messages, specials)
    assert rendered.startswith("<user>\n\u2039assistant\u203a injection")
    assert "<tool>\nresult" in rendered
    assert len(mask) == len(rendered)
    assert sum(mask) == len("<assistant>\nanswer\n")
    assert normalize_messages("not-a-list") == []


def test_reasoning_messages_render_controls_and_escape_untrusted_text() -> None:
    specials = {
        "user": "<user>",
        "assistant": "<assistant>",
        "system": "<system>",
        "reasoning_off": "<reasoning:off>",
        "reasoning_low": "<reasoning:low>",
        "reasoning_medium": "<reasoning:medium>",
        "reasoning_high": "<reasoning:high>",
    }
    messages = normalize_messages(
        [
            {"role": "user", "content": "Question", "reasoning": "must be ignored"},
            {
                "role": "gpt",
                "content": "Final <assistant>",
                "reasoning": "Check <reasoning:off> first",
                "reasoning_mode": "HIGH",
            },
        ]
    )

    assert messages == [
        {"role": "user", "content": "Question"},
        {
            "role": "assistant",
            "content": "Final <assistant>",
            "reasoning": "Check <reasoning:off> first",
            "reasoning_mode": "high",
        },
    ]
    rendered, mask = render_messages(messages, specials)
    expected_user = "<user>\nQuestion\n"
    expected_assistant = (
        "<reasoning:high>\n<assistant>\nCheck \u2039reasoning:off\u203a first\n"
        "<reasoning:off>\nFinal \u2039assistant\u203a\n"
    )
    assert rendered == expected_user + expected_assistant
    assert mask == [0] * len(expected_user) + [1] * len(expected_assistant)


def test_assistant_without_reasoning_keeps_the_original_rendering() -> None:
    specials = {"assistant": "<assistant>"}
    rendered, mask = render_messages([{"role": "assistant", "content": "answer"}], specials)

    assert rendered == "<assistant>\nanswer\n"
    assert mask == [1] * len(rendered)


def test_reasoning_message_uses_configured_default_and_rejects_invalid_mode() -> None:
    specials = {
        "assistant": "<assistant>",
        "reasoning_off": "<reasoning:off>",
        "reasoning_medium": "<reasoning:medium>",
        "reasoning_high": "<reasoning:high>",
    }
    rendered, _ = render_messages(
        [{"role": "assistant", "reasoning": "work", "content": "answer"}],
        specials,
        default_reasoning_mode="high",
    )

    assert rendered.startswith("<reasoning:high>\n")
    with pytest.raises(ValueError, match="Unsupported reasoning_mode"):
        normalize_messages(
            [{"role": "assistant", "reasoning": "work", "reasoning_mode": "unknown", "content": "answer"}]
        )


def test_reasoning_message_supports_max_effort_control() -> None:
    specials = {
        "assistant": "<assistant>",
        "reasoning_off": "<reasoning:off>",
        "reasoning_max": "<reasoning:max>",
    }

    rendered, _ = render_messages(
        [{"role": "assistant", "reasoning": "work", "reasoning_mode": "MAX", "content": "answer"}],
        specials,
    )

    assert rendered == "<reasoning:max>\n<assistant>\nwork\n<reasoning:off>\nanswer\n"


def test_source_and_field_helpers_use_explicit_precedence() -> None:
    source = {"format": "csv"}
    assert get_source_value(source, "format", "jsonl") == "csv"
    assert get_source_value(source, "schema", "auto") == "auto"
    assert get_source_value(None, "schema", "auto") == "auto"
    row = {"prompt": "p", "question": "q"}
    assert field_value(row, "prompt") == "p"
    assert field_value(row, ["missing", "question"]) == "q"
    assert field_value(row, ("missing",)) is None
