from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, local

import pytest
import torch

from llm_pipeline.config import load_config
from llm_pipeline.data import (
    analyze_sample_stream,
    collate_token_batch,
    iter_deduped_samples,
    iter_text_samples_from_config,
    load_preference_samples,
    row_to_text_sample,
    safe_perplexity,
    source_matches_stage,
    token_shard_cache_key,
    tokenize_training_samples,
)
from llm_pipeline.data_governance import content_hash
from llm_pipeline.data_reader import TextSample, clean_text, iter_rows
from llm_pipeline.errors import DataPolicyError

ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    bos_id = 1
    eos_id = 2
    pad_id = 3
    vocab_size = 256
    model_path = Path("missing-test-tokenizer.model")

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [20 + (ord(char) % 200) for char in text]
        return [self.bos_id, *ids, self.eos_id] if add_special_tokens else ids


class ConcurrentFakeTokenizer(FakeTokenizer):
    def __init__(self, barrier: Barrier) -> None:
        self.barrier = barrier
        self.thread_state = local()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if not getattr(self.thread_state, "synchronized", False):
            self.thread_state.synchronized = True
            self.barrier.wait(timeout=5)
        return super().encode(text, add_special_tokens)


def test_clean_text_normalizes_and_preserves_pipeline_tokens() -> None:
    value = clean_text("  \uff21\x00 <b>hello</b> <assistant>  ")
    assert value == "A hello <assistant>"


def test_invalid_jsonl_reports_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text":"ok"}\n{"text":}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        list(iter_rows(path, "jsonl"))


def test_token_cache_key_changes_when_audit_digest_changes(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").mutable_copy()
    source = tmp_path / "source.jsonl"
    source.write_text('{"text":"안전한 학습 문장입니다."}\n', encoding="utf-8")
    tokenizer_model = tmp_path / "tokenizer.model"
    tokenizer_model.write_bytes(b"tokenizer")
    lock = tmp_path / "sources.lock.json"
    audit = tmp_path / "data_audit.json"
    lock.write_text(json.dumps({"lock_digest": "lock-a"}), encoding="utf-8")
    audit.write_text(json.dumps({"audit_digest": "audit-a"}), encoding="utf-8")
    config["data"]["sources"] = [{"name": "fixture", "path": str(source), "schema": "text"}]
    config["data_policy"].update(source_lock_path=str(lock), audit_path=str(audit))
    tokenizer = FakeTokenizer()
    tokenizer.model_path = tokenizer_model

    before = token_shard_cache_key(config, tokenizer, "train", "pretrain", False)
    audit.write_text(json.dumps({"audit_digest": "audit-b"}), encoding="utf-8")
    after = token_shard_cache_key(config, tokenizer, "train", "pretrain", False)

    assert before != after


def test_token_cache_publish_is_safe_for_concurrent_ddp_style_writers(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").mutable_copy()
    cache_dir = tmp_path / "token_cache"
    config["data"]["token_cache_dir"] = str(cache_dir)
    samples = [TextSample("synthetic concurrent cache record")]
    tokenizer = ConcurrentFakeTokenizer(Barrier(2))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: tokenize_training_samples(samples, tokenizer, config), range(2)))

    assert results[0] == results[1]
    assert len(list(cache_dir.glob("tokens_*.pkl"))) == 1
    assert list(cache_dir.glob("*.tmp")) == []


def test_in_memory_deduplication_keeps_first_sample() -> None:
    config = load_config(ROOT / "configs/smoke.yaml").mutable_copy()
    rows = [TextSample("same"), TextSample("same"), TextSample("different")]
    assert [row.text for row in iter_deduped_samples(rows, config, None)] == ["same", "different"]


def test_configured_samples_carry_source_coverage_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"text":"source coverage sentence"}\n', encoding="utf-8")
    config = load_config(ROOT / "configs/smoke.yaml").mutable_copy()
    config["data"].update(
        sources=[{"name": "coverage-source", "path": str(source), "schema": "text", "stages": ["pretrain"]}],
        min_chars=0,
        hash_split=False,
    )

    samples = list(iter_text_samples_from_config(config, "train", dataset_type="pretrain"))

    assert samples[0].meta["_source_name"] == "coverage-source"
    assert samples[0].meta["_source_path"] == str(source)


def test_assistant_only_labels_and_packed_positions() -> None:
    config = load_config(ROOT / "configs/smoke.yaml").mutable_copy()
    config["data"]["token_cache_dir"] = None
    config["model"]["max_seq_len"] = 64
    messages = [
        {"role": "user", "content": "질문"},
        {"role": "assistant", "content": "대답"},
    ]
    samples = [TextSample("unused", kind="sft", meta={"messages": messages})]
    rows = tokenize_training_samples(samples, FakeTokenizer(), config, assistant_only_loss=True)
    assert len(rows) == 1
    labels = rows[0]["labels"]
    assert -100 in labels
    assert any(label != -100 for label in labels[1:])

    batch = [
        {
            "input_ids": torch.tensor([1, 10, 2, 1, 11, 2]),
            "labels": torch.tensor([1, 10, 2, 1, 11, 2]),
            "seq_lens": [3, 3],
        }
    ]
    collated = collate_token_batch(batch, pad_id=3)
    assert collated["position_ids"].tolist() == [[0, 1, 2, 0, 1, 2]]
    assert collated["document_ids"].tolist() == [[0, 0, 0, 1, 1, 1]]


def test_perplexity_is_finite_for_large_finite_loss() -> None:
    perplexity = safe_perplexity(60.0)
    assert math.isfinite(perplexity)
    assert perplexity == pytest.approx(math.exp(60.0))
    with pytest.raises(ValueError, match="finite loss"):
        safe_perplexity(float("inf"))


def test_stream_stats_use_bounded_median_reservoir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm_pipeline.data.STATS_RESERVOIR_SIZE", 3)
    stats = analyze_sample_stream(TextSample("x" * length) for length in range(1, 11))
    assert stats["sample_count"] == 10
    assert stats["total_chars"] == 55
    assert stats["max_chars"] == 10
    assert stats["median_is_approximate"] is True
    assert stats["median_sample_size"] == 3


def test_explicit_translation_fields_support_any_language_pair() -> None:
    config = load_config(ROOT / "configs/smoke.yaml").mutable_copy()
    source = {
        "schema": "translation",
        "source_lang_field": "spanish_text",
        "target_lang_field": "english_text",
        "source_lang": "Spanish",
        "target_lang": "English",
        "prompt_template": "Traduce este texto al inglés.\n{source_text}",
    }

    sample = row_to_text_sample(
        {"spanish_text": "Buenos días.", "english_text": "Good morning."},
        config,
        dataset_type="sft",
        source=source,
    )

    assert sample is not None
    assert sample.meta["messages"] == [
        {"role": "user", "content": "Traduce este texto al inglés.\nBuenos días."},
        {"role": "assistant", "content": "Good morning."},
    ]


def test_source_sampling_and_language_metadata_are_reproducible(tmp_path: Path) -> None:
    source_path = tmp_path / "spanish.jsonl"
    source_path.write_text(
        "".join(json.dumps({"text": f"ejemplo número {index}"}) + "\n" for index in range(50)),
        encoding="utf-8",
    )
    config = load_config(ROOT / "configs/smoke.yaml").mutable_copy()
    config["data"].update(min_chars=0, hash_split=False)
    config["data"]["sources"] = [
        {
            "name": "spanish-example",
            "path": str(source_path),
            "schema": "text",
            "split": "train",
            "languages": ["es"],
            "sample_rate": 0.4,
        }
    ]

    first = list(iter_text_samples_from_config(config, "train", dataset_type="pretrain"))
    second = list(iter_text_samples_from_config(config, "train", dataset_type="pretrain"))

    assert 0 < len(first) < 50
    assert [sample.text for sample in first] == [sample.text for sample in second]
    assert all(sample.meta["_languages"] == ["es"] for sample in first)


def test_stats_separate_scripts_from_declared_languages() -> None:
    samples = [
        TextSample("한국어", meta={"_languages": ["ko"]}),
        TextSample("かな", meta={"_languages": ["ja"]}),
        TextSample("中文", meta={"_languages": ["zh"]}),
    ]

    stats = analyze_sample_stream(samples, FakeTokenizer())

    assert stats["script_ratio"]["hangul"] > 0
    assert stats["script_ratio"]["kana"] > 0
    assert stats["script_ratio"]["han"] > 0
    assert set(stats["declared_language_stats"]) == {"ja", "ko", "zh"}
    assert stats["declared_language_stats"]["zh"]["chars_per_token"] == 1.0
    assert "language_ratio" not in stats


def test_evaluation_sources_never_match_training_or_tokenizer_stages() -> None:
    source = {
        "purpose": "evaluation",
        "tokenizer": True,
        "stages": ["all"],
    }

    assert not source_matches_stage(source, "tokenizer")
    assert not source_matches_stage(source, "pretrain")
    assert not source_matches_stage(source, "sft")
    assert source_matches_stage({**source, "stages": ["eval"]}, "eval")


def test_dpo_training_rows_cannot_bypass_benchmark_filter(tmp_path: Path) -> None:
    held_out = "private benchmark answer"
    preference_path = tmp_path / "preferences.jsonl"
    preference_path.write_text(
        json.dumps({"prompt": "question", "chosen": held_out, "rejected": "other"}) + "\n",
        encoding="utf-8",
    )
    denylist = tmp_path / "denylist.txt"
    denylist.write_text(content_hash(held_out) + "\n", encoding="utf-8")
    config = load_config(ROOT / "configs/smoke.yaml").mutable_copy()
    config["data_policy"]["benchmark_denylist_path"] = str(denylist)

    with pytest.raises(DataPolicyError, match="DPO training row 1"):
        load_preference_samples(preference_path, config)
    assert len(load_preference_samples(preference_path, config, purpose="evaluation")) == 1
