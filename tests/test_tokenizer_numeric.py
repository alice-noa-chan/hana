from __future__ import annotations

import copy
import json
from pathlib import Path
from types import MethodType

import pytest
import sentencepiece as spm

from llm_pipeline.config import DEFAULT_CONFIG
from llm_pipeline.tokenizer import (
    NUMERIC_VALIDATION_PROBES,
    TOKENIZER_BEHAVIOR_VERSION,
    SentencePieceTokenizer,
    load_tokenizer,
    publish_tokenizer_bundle,
    sentencepiece_trainer_kwargs,
    validate_tokenizer_candidate,
)


def numeric_test_config(tmp_path: Path) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data_policy"]["enforce"] = False
    config["data_policy"]["benchmark_denylist_path"] = None
    config["data"]["sources"] = []
    config["data"]["train_file"] = str(tmp_path / "unused-private-corpus.jsonl")
    config["tokenizer"].update(
        {
            "vocab_size": 512,
            "input_sentence_size": 0,
            "shuffle_input_sentence": False,
            "split_digits": True,
            "normalization_rule_name": "nmt_nfkc",
            "numeric_validation": True,
            "numeric_validation_corpus_samples": 4,
            "save_dir": str(tmp_path),
            "model_path": str(tmp_path / "tokenizer.model"),
        }
    )
    return config


def write_code_owned_numeric_corpus(path: Path) -> None:
    rows = [source for _name, source, _expected in NUMERIC_VALIDATION_PROBES]
    rows.extend(
        [
            "ordinary tokenizer coverage alpha beta gamma delta",
            "한국어 숫자 문맥과 일본語の数値文脈을 함께 확인합니다",
            "repeatable deterministic test corpus with punctuation and symbols",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def train_test_tokenizer(config: dict, corpus: Path, prefix: Path) -> SentencePieceTokenizer:
    spm.SentencePieceTrainer.train(**sentencepiece_trainer_kwargs(config, input_path=corpus, model_prefix=prefix))
    return SentencePieceTokenizer(prefix.with_suffix(".model"), config["tokenizer"]["special_tokens"])


def test_numeric_validation_round_trips_nfkc_and_saves_aggregate_metadata(tmp_path: Path) -> None:
    config = numeric_test_config(tmp_path)
    corpus = tmp_path / "numeric-corpus.txt"
    write_code_owned_numeric_corpus(corpus)
    tokenizer = train_test_tokenizer(config, corpus, tmp_path / "candidate")

    validation = validate_tokenizer_candidate(tokenizer, config, corpus)
    tokenizer.save_metadata(tmp_path / "published", config, validation)

    assert set(validation) == {
        "behavior_version",
        "status",
        "split_digits",
        "normalization_rule_name",
        "probe_count",
        "probe_suite_sha256",
        "corpus_samples_checked",
        "unk_count",
        "model_sha256",
    }
    assert validation["behavior_version"] == TOKENIZER_BEHAVIOR_VERSION == 2
    assert validation["status"] == "passed"
    assert validation["split_digits"] is True
    assert validation["normalization_rule_name"] == "nmt_nfkc"
    assert validation["probe_count"] == len(NUMERIC_VALIDATION_PROBES)
    assert validation["corpus_samples_checked"] == 4
    assert validation["unk_count"] == 0
    assert len(validation["probe_suite_sha256"]) == 64
    assert len(validation["model_sha256"]) == 64

    _name, full_width, expected_nfkc = NUMERIC_VALIDATION_PROBES[-1]
    full_width_ids = tokenizer.encode(full_width, add_special_tokens=False)
    assert tokenizer.decode(full_width_ids) == expected_nfkc

    long_identifier = NUMERIC_VALIDATION_PROBES[9][1]
    identifier_ids = tokenizer.encode(long_identifier, add_special_tokens=False)
    assert all(
        tokenizer.sp.is_byte(piece_id)
        or sum(character.isdecimal() for character in tokenizer.sp.id_to_piece(piece_id)) <= 1
        for piece_id in identifier_ids
    )

    saved_validation = json.loads((tmp_path / "published" / "tokenizer_validation.json").read_text("utf-8"))
    tokenizer_config = json.loads((tmp_path / "published" / "tokenizer_config.json").read_text("utf-8"))
    assert saved_validation == validation
    assert tokenizer_config["behavior_version"] == 2
    assert tokenizer_config["split_digits"] is True
    assert tokenizer_config["normalization_rule_name"] == "nmt_nfkc"
    assert tokenizer_config["model_sha256"] == validation["model_sha256"]
    assert tokenizer_config["validation_path"] == "tokenizer_validation.json"
    assert tokenizer_config["validation"] == {
        "status": "passed",
        "path": "tokenizer_validation.json",
        "probe_count": len(NUMERIC_VALIDATION_PROBES),
        "corpus_samples_checked": 4,
        "unk_count": 0,
    }


def test_shared_trainer_kwargs_include_numeric_behavior_and_reasoning_max(tmp_path: Path) -> None:
    config = numeric_test_config(tmp_path)
    kwargs = sentencepiece_trainer_kwargs(
        config,
        input_path=tmp_path / "corpus.txt",
        model_prefix=tmp_path / "tokenizer",
    )

    assert kwargs["split_digits"] is True
    assert kwargs["normalization_rule_name"] == "nmt_nfkc"
    assert config["tokenizer"]["special_tokens"]["reasoning_max"] in kwargs["user_defined_symbols"].split(",")


def test_numeric_validation_rejects_model_with_multi_digit_pieces(tmp_path: Path) -> None:
    training_config = numeric_test_config(tmp_path)
    training_config["tokenizer"]["split_digits"] = False
    corpus = tmp_path / "unsplit-corpus.txt"
    corpus.write_text(
        ("12345678901234567890 repeated numeric identifier\n" * 20),
        encoding="utf-8",
    )
    tokenizer = train_test_tokenizer(training_config, corpus, tmp_path / "unsplit")
    validation_config = copy.deepcopy(training_config)
    validation_config["tokenizer"]["split_digits"] = True

    with pytest.raises(RuntimeError, match="multiple decimal digits"):
        validate_tokenizer_candidate(tokenizer, validation_config, corpus)


def test_load_tokenizer_requires_current_numeric_integrity_evidence(tmp_path: Path) -> None:
    config = numeric_test_config(tmp_path)
    corpus = tmp_path / "numeric-corpus.txt"
    write_code_owned_numeric_corpus(corpus)
    tokenizer = train_test_tokenizer(config, corpus, tmp_path / "candidate")
    validation = validate_tokenizer_candidate(tokenizer, config, corpus)
    published = tmp_path / "published"
    tokenizer.save_metadata(published, config, validation)
    config["tokenizer"]["model_path"] = str(published / "tokenizer.model")
    config["tokenizer"]["save_dir"] = str(published)

    assert load_tokenizer(config).vocab_size == int(config["tokenizer"]["vocab_size"])

    validation_path = published / "tokenizer_validation.json"
    stale = json.loads(validation_path.read_text(encoding="utf-8"))
    stale["probe_count"] -= 1
    validation_path.write_text(json.dumps(stale), encoding="utf-8")

    with pytest.raises(RuntimeError, match="incomplete or obsolete"):
        load_tokenizer(config)


def test_tokenizer_bundle_publish_restores_previous_files_on_failure(tmp_path: Path, monkeypatch) -> None:
    config = numeric_test_config(tmp_path)
    corpus = tmp_path / "numeric-corpus.txt"
    write_code_owned_numeric_corpus(corpus)
    candidate = train_test_tokenizer(config, corpus, tmp_path / "candidate")
    validation = validate_tokenizer_candidate(candidate, config, corpus)
    published = tmp_path / "published"
    publish_tokenizer_bundle(
        candidate,
        config,
        validation,
        published,
        vocab_path=tmp_path / "candidate.vocab",
    )
    artifact_names = (
        "tokenizer.model",
        "tokenizer.vocab",
        "tokenizer_validation.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
    )
    previous = {name: (published / name).read_bytes() for name in artifact_names}
    original_replace = Path.replace

    def fail_during_promotion(path: Path, target: Path) -> Path:
        if path.parent.name == "incoming" and path.name == "tokenizer.vocab":
            raise OSError("synthetic promotion failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_during_promotion)
    with pytest.raises(OSError, match="synthetic promotion failure"):
        publish_tokenizer_bundle(
            candidate,
            config,
            validation,
            published,
            vocab_path=tmp_path / "candidate.vocab",
        )

    assert {name: (published / name).read_bytes() for name in artifact_names} == previous


def test_corpus_validation_rejects_nonempty_input_that_decodes_empty(tmp_path: Path) -> None:
    config = numeric_test_config(tmp_path)
    config["tokenizer"]["numeric_validation"] = False
    config["tokenizer"]["numeric_validation_corpus_samples"] = 1
    training_corpus = tmp_path / "training-corpus.txt"
    write_code_owned_numeric_corpus(training_corpus)
    tokenizer = train_test_tokenizer(config, training_corpus, tmp_path / "empty-round-trip")
    validation_corpus = tmp_path / "validation-corpus.txt"
    validation_corpus.write_text("synthetic visible test row\n", encoding="utf-8")
    tokenizer.decode = MethodType(lambda _self, _ids: "", tokenizer)

    with pytest.raises(RuntimeError, match="decoded to an empty string"):
        validate_tokenizer_candidate(tokenizer, config, validation_corpus)
