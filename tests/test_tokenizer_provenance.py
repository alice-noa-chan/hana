from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import sentencepiece as spm

from llm_pipeline.config import DEFAULT_CONFIG
from llm_pipeline.data_governance import content_hash
from llm_pipeline.tokenizer import (
    SentencePieceTokenizer,
    build_tokenizer_corpus_manifest,
    verify_tokenizer_corpus_manifest,
)


def ungoverned_test_config() -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data_policy"]["enforce"] = False
    config["data_policy"]["benchmark_denylist_path"] = None
    config["data"]["sources"] = []
    return config


def test_tokenizer_corpus_manifest_detects_changed_bytes(tmp_path: Path) -> None:
    config = ungoverned_test_config()
    corpus = tmp_path / "tokenizer_corpus.txt"
    corpus.write_text("first row\nsecond row\n", encoding="utf-8")

    manifest = build_tokenizer_corpus_manifest(config, corpus)

    assert manifest["format_version"] == 2
    assert manifest["corpus_lines"] == 2
    assert verify_tokenizer_corpus_manifest(config, corpus) == manifest

    corpus.write_text("changed row\nsecond row\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="corpus bytes"):
        verify_tokenizer_corpus_manifest(config, corpus)


def test_tokenizer_corpus_manifest_tracks_benchmark_registry(tmp_path: Path) -> None:
    config = ungoverned_test_config()
    corpus = tmp_path / "tokenizer_corpus.txt"
    corpus.write_text("training row\n", encoding="utf-8")
    denylist = tmp_path / "benchmark-denylist.txt"
    denylist.write_text(content_hash("benchmark one") + "\n", encoding="utf-8")
    config["data_policy"]["benchmark_denylist_path"] = str(denylist)
    build_tokenizer_corpus_manifest(config, corpus)

    denylist.write_text(content_hash("benchmark two") + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="provenance is stale"):
        verify_tokenizer_corpus_manifest(config, corpus)


def test_obsolete_tokenizer_corpus_manifest_is_rejected(tmp_path: Path) -> None:
    config = ungoverned_test_config()
    corpus = tmp_path / "tokenizer_corpus.txt"
    corpus.write_text("legacy corpus\n", encoding="utf-8")
    (tmp_path / "corpus_manifest.json").write_text(
        json.dumps({"format_version": 1, "corpus_file": corpus.name}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="obsolete format"):
        verify_tokenizer_corpus_manifest(config, corpus)


def test_reasoning_boundary_encoding_preserves_sentencepiece_separator(tmp_path: Path) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    corpus = tmp_path / "synthetic-corpus.txt"
    corpus.write_text(
        "synthetic reasoning answer alpha beta gamma\nanother ordinary training sentence for tokenizer coverage\n",
        encoding="utf-8",
    )
    prefix = tmp_path / "tokenizer"
    specials = config["tokenizer"]["special_tokens"]
    user_defined = [
        specials["user"],
        specials["assistant"],
        specials["system"],
        specials["reasoning_off"],
        specials["reasoning_low"],
        specials["reasoning_medium"],
        specials["reasoning_high"],
        specials["mask"],
    ]
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=300,
        byte_fallback=True,
        hard_vocab_limit=False,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        pad_piece=specials["pad"],
        unk_piece=specials["unk"],
        bos_piece=specials["bos"],
        eos_piece=specials["eos"],
        user_defined_symbols=",".join(user_defined),
    )
    tokenizer = SentencePieceTokenizer(prefix.with_suffix(".model"), specials)
    boundary_id = tokenizer.piece_to_id(specials["reasoning_off"])

    encoded = tokenizer.encode(f"\n{specials['reasoning_off']}\n", add_special_tokens=False)

    assert encoded.count(boundary_id) == 1
    assert len(encoded) > 1
