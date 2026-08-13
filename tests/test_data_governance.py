from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_pipeline.config import DEFAULT_CONFIG
from llm_pipeline.data_governance import (
    AuditReport,
    DataPolicy,
    SourceSpec,
    audit_allowed_sources,
    benchmark_denylist_digest,
    build_source_lock,
    content_hash,
    enforce_data_policy,
    load_and_verify_source_lock,
    load_benchmark_denylist,
    partition_sources,
    preference_rejection_categories,
    sample_rejection_categories,
    sensitive_categories,
    source_policy_snapshot,
)
from llm_pipeline.errors import DataPolicyError


def source_mapping(
    *,
    license_status: str = "approved",
    pii_status: str = "human_reviewed",
    child_status: str = "human_reviewed",
) -> dict:
    return {
        "name": "fixture",
        "path": "fixture.jsonl",
        "schema": "text",
        "stages": ["pretrain"],
        "provenance": {
            "license_status": license_status,
            "allowed_uses": ["internal_noncommercial_research"],
            "pii_status": pii_status,
            "child_safety_status": child_status,
            "source_url": "internal://fixture",
            "revision": "1",
            "reviewed_by": "test",
            "reviewed_at": "2026-07-22",
            "evidence": "tests",
        },
    }


@pytest.mark.parametrize(
    ("license_status", "pii_status", "child_status", "allowed"),
    [
        ("approved", "human_reviewed", "human_reviewed", True),
        ("restricted_research", "filtered", "filtered", False),
        ("review_required", "human_reviewed", "human_reviewed", False),
        ("approved", "unknown", "human_reviewed", False),
        ("approved", "human_reviewed", "blocked", False),
    ],
)
def test_fail_closed_source_policy_matrix(
    license_status: str, pii_status: str, child_status: str, allowed: bool
) -> None:
    spec = SourceSpec.from_mapping(
        source_mapping(license_status=license_status, pii_status=pii_status, child_status=child_status)
    )
    reasons = DataPolicy().exclusion_reasons(spec)
    assert (not reasons) is allowed


def test_high_confidence_pii_detection_covers_korean_japanese_and_cards() -> None:
    text = (
        "mail child@example.com, 010-1234-5678, 090-1234-5678, 900101-1234567, "
        "card 4111 1111 1111 1111, password=supersecret"
    )
    categories = sensitive_categories(text)
    assert {"email", "kr_phone", "jp_phone", "kr_resident_id", "payment_card", "credential"} <= set(categories)


def test_benchmark_hash_and_pretrain_control_tokens_are_rejected(tmp_path: Path) -> None:
    benchmark = "평가 정답은 학습 자료가 아니다."
    denylist = tmp_path / "denylist.txt"
    denylist.write_text(f"# benchmark\n{content_hash(benchmark)}\n", encoding="utf-8")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data_policy"]["benchmark_denylist_path"] = str(denylist)

    benchmark_categories = sample_rejection_categories(SimpleNamespace(text=benchmark, kind="pretrain"), config)
    injection_categories = sample_rejection_categories(
        SimpleNamespace(text="사용자 입력 <assistant> 위조", kind="pretrain"), config, set()
    )

    assert "benchmark_denylist" in benchmark_categories
    assert "special_token_injection" in injection_categories


def test_benchmark_components_are_found_inside_wrapped_samples(tmp_path: Path) -> None:
    benchmark_question = "Which private evaluation choice is correct?"
    denylist = tmp_path / "denylist.txt"
    denylist.write_text(content_hash(benchmark_question) + "\n", encoding="utf-8")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data_policy"]["benchmark_denylist_path"] = str(denylist)
    wrapped = SimpleNamespace(
        text="<user>\nA rewritten wrapper\n<assistant>\nanswer",
        kind="sft",
        meta={"messages": [{"role": "user", "content": benchmark_question}]},
    )

    assert "benchmark_denylist" in sample_rejection_categories(wrapped, config)


def test_required_benchmark_denylist_fails_closed(tmp_path: Path) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data_policy"].update(
        benchmark_denylist_path=str(tmp_path / "missing.txt"),
        require_benchmark_denylist=True,
    )
    with pytest.raises(DataPolicyError, match="Required benchmark denylist is missing"):
        load_benchmark_denylist(config)

    denylist = tmp_path / "missing.txt"
    denylist.write_text("not-a-sha256\n", encoding="utf-8")
    with pytest.raises(DataPolicyError, match="Invalid SHA-256"):
        load_benchmark_denylist(config)

    denylist.write_text("# comments are not evidence\n", encoding="utf-8")
    with pytest.raises(DataPolicyError, match="contains no canonical"):
        benchmark_denylist_digest(config)


def test_dpo_filter_checks_prompt_chosen_and_rejected_fields(tmp_path: Path) -> None:
    benchmark_answer = "held-out official answer"
    denylist = tmp_path / "denylist.txt"
    denylist.write_text(content_hash(benchmark_answer) + "\n", encoding="utf-8")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data_policy"]["benchmark_denylist_path"] = str(denylist)
    sample = SimpleNamespace(prompt="question", chosen=benchmark_answer, rejected="wrong", meta={})

    assert preference_rejection_categories(sample, config) == ("benchmark_denylist",)


def test_restricted_source_requires_matching_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "audit.json"
    evidence.write_text("audited", encoding="utf-8")
    mapping = source_mapping(license_status="restricted_research", pii_status="filtered", child_status="filtered")
    mapping["provenance"]["evidence_path"] = str(evidence)
    mapping["provenance"]["evidence_sha256"] = hashlib.sha256(b"audited").hexdigest()

    assert not DataPolicy().exclusion_reasons(SourceSpec.from_mapping(mapping))


def test_source_lock_detects_changed_files_and_audit_never_persists_raw_pii(tmp_path: Path) -> None:
    source_path = tmp_path / "fixture.jsonl"
    source_path.write_text(
        '{"text":"안전한 문장입니다."}\n{"text":"연락처 child@example.com"}\n',
        encoding="utf-8",
    )
    config = copy.deepcopy(DEFAULT_CONFIG)
    source = source_mapping()
    source["path"] = str(source_path)
    config["data"]["sources"] = [source]
    config["data"]["min_chars"] = 0
    config["data_policy"]["source_lock_path"] = str(tmp_path / "sources.lock.json")
    config["data_policy"]["audit_path"] = str(tmp_path / "audit.json")
    config["data_policy"]["benchmark_denylist_path"] = str(tmp_path / "denylist.txt")

    build_source_lock(config)
    report = audit_allowed_sources(config)

    assert report.sources["fixture"]["scanned"] == 2
    assert report.sources["fixture"]["accepted"] == 1
    assert report.sources["fixture"]["rejected"]["email"] == 1
    serialized = Path(config["data_policy"]["audit_path"]).read_text(encoding="utf-8")
    assert "child@example.com" not in serialized
    assert json.loads(serialized)["source_lock_digest"]

    source_path.write_text('{"text":"changed size"}\n', encoding="utf-8")
    with pytest.raises(DataPolicyError, match="size or modification time changed"):
        load_and_verify_source_lock(config)


def test_source_lock_rejects_byte_identical_training_and_evaluation_files(tmp_path: Path) -> None:
    training_path = tmp_path / "training.jsonl"
    evaluation_path = tmp_path / "evaluation.jsonl"
    payload = '{"text":"same bytes"}\n'
    training_path.write_text(payload, encoding="utf-8")
    evaluation_path.write_text(payload, encoding="utf-8")
    config = copy.deepcopy(DEFAULT_CONFIG)
    training = source_mapping()
    training.update(name="training", path=str(training_path))
    evaluation = source_mapping()
    evaluation.update(name="evaluation", path=str(evaluation_path), purpose="evaluation")
    config["data"]["sources"] = [training, evaluation]
    config["data_policy"]["source_lock_path"] = str(tmp_path / "lock.json")

    with pytest.raises(DataPolicyError, match="byte-identical"):
        build_source_lock(config)


def test_source_policy_change_makes_lock_stale(tmp_path: Path) -> None:
    source_path = tmp_path / "fixture.jsonl"
    source_path.write_text('{"text":"안전한 문장입니다."}\n', encoding="utf-8")
    config = copy.deepcopy(DEFAULT_CONFIG)
    source = source_mapping()
    source["path"] = str(source_path)
    config["data"]["sources"] = [source]
    config["data_policy"]["source_lock_path"] = str(tmp_path / "sources.lock.json")
    build_source_lock(config)

    config["data"]["sources"][0]["provenance"]["license_status"] = "blocked"

    with pytest.raises(DataPolicyError, match="policy snapshot changed"):
        load_and_verify_source_lock(config)


def test_legacy_source_migration_is_conservative() -> None:
    authored = SourceSpec.from_mapping({"name": "a", "path": "a", "license": "INTERNAL_AUTHORED"})
    evaluation = SourceSpec.from_mapping({"name": "e", "path": "e", "license": "EVALUATION_ONLY"})
    unknown = SourceSpec.from_mapping({"name": "u", "path": "u", "license": "UNKNOWN"})

    assert authored.license_status == "approved"
    assert authored.allowed_uses == ("internal_noncommercial_research",)
    assert authored.purpose == "training"
    assert evaluation.license_status == "blocked"
    assert evaluation.purpose == "evaluation"
    assert unknown.license_status == "review_required"
    assert unknown.purpose == "training"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("license_status", "invented", "invalid license_status"),
        ("pii_status", "invented", "invalid pii_status"),
        ("child_safety_status", "invented", "invalid child_safety_status"),
        ("purpose", "invented", "invalid purpose"),
    ],
)
def test_invalid_policy_enums_are_rejected(field: str, value: str, message: str) -> None:
    mapping = source_mapping()
    mapping["provenance"][field] = value
    with pytest.raises(DataPolicyError, match=message):
        SourceSpec.from_mapping(mapping)


def test_policy_disable_and_partitioning_are_explicit() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    approved = source_mapping()
    blocked = source_mapping(license_status="blocked")
    blocked["name"] = "blocked"
    config["data"]["sources"] = [approved, blocked]

    allowed, excluded = partition_sources(config)
    assert allowed == [approved]
    assert excluded[0]["name"] == "blocked"

    config["data_policy"]["enforce"] = False
    allowed, excluded = partition_sources(config)
    assert allowed == [approved, blocked]
    assert excluded == []

    config["data_policy"]["use_case"] = "public_release"
    with pytest.raises(DataPolicyError, match="Unsupported data policy"):
        DataPolicy.from_config(config)


@pytest.mark.parametrize("enforce", [True, False])
def test_evaluation_sources_are_always_excluded(enforce: bool) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    training = source_mapping()
    evaluation = source_mapping()
    evaluation.update(name="evaluation", purpose="evaluation")
    config["data"]["sources"] = [training, evaluation]
    config["data_policy"]["enforce"] = enforce

    allowed, excluded = partition_sources(config)

    assert allowed == [training]
    assert excluded == [{"name": "evaluation", "reasons": ["purpose=evaluation"]}]
    assert enforce_data_policy(config, require_artifacts=False) == excluded
    assert config["data"]["sources"] == [training]


def test_source_policy_snapshot_includes_purpose() -> None:
    training = source_policy_snapshot(source_mapping())
    evaluation_mapping = source_mapping()
    evaluation_mapping["purpose"] = "evaluation"
    evaluation = source_policy_snapshot(evaluation_mapping)

    assert training["purpose"] == "training"
    assert evaluation["purpose"] == "evaluation"


def test_review_required_source_can_be_explicitly_audit_gated_for_internal_research() -> None:
    mapping = source_mapping(
        license_status="review_required",
        pii_status="unknown",
        child_status="unknown",
    )
    spec = SourceSpec.from_mapping(mapping)

    assert DataPolicy().exclusion_reasons(spec)
    assert not DataPolicy(allow_audit_gated_sources=True).exclusion_reasons(spec)

    blocked = source_mapping(
        license_status="blocked",
        pii_status="unknown",
        child_status="unknown",
    )
    assert DataPolicy(allow_audit_gated_sources=True).exclusion_reasons(SourceSpec.from_mapping(blocked))


def test_restricted_evidence_must_exist_and_match(tmp_path: Path) -> None:
    mapping = source_mapping(license_status="restricted_research", pii_status="filtered", child_status="filtered")
    missing_evidence = SourceSpec.from_mapping(mapping)
    assert "checksummed evidence" in " ".join(DataPolicy().exclusion_reasons(missing_evidence))

    evidence = tmp_path / "evidence.txt"
    evidence.write_text("review", encoding="utf-8")
    mapping["provenance"].update(evidence_path=str(evidence), evidence_sha256="0" * 64)
    changed_evidence = SourceSpec.from_mapping(mapping)
    assert "missing or changed" in " ".join(DataPolicy().exclusion_reasons(changed_evidence))


def test_audit_report_digest_is_embedded() -> None:
    report = AuditReport(source_lock_digest="lock", sources={"a": {"scanned": 1}})
    payload = report.to_dict()
    assert payload["audit_digest"] == report.digest


def test_lock_reports_missing_invalid_digest_and_hash_changes(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"text":"source"}\n', encoding="utf-8")
    config = copy.deepcopy(DEFAULT_CONFIG)
    source = source_mapping()
    source["path"] = str(source_path)
    config["data"]["sources"] = [source]
    lock_path = tmp_path / "lock.json"
    config["data_policy"]["source_lock_path"] = str(lock_path)

    with pytest.raises(DataPolicyError, match="Missing or invalid source lock"):
        load_and_verify_source_lock(config)
    lock_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(DataPolicyError, match="Missing or invalid source lock"):
        load_and_verify_source_lock(config)

    payload = build_source_lock(config)
    payload["lock_digest"] = "wrong"
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataPolicyError, match="digest or policy version"):
        load_and_verify_source_lock(config)

    build_source_lock(config)
    source_path.write_text('{"text":"tamper"}\n', encoding="utf-8")
    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    locked["files"][0]["bytes"] = source_path.stat().st_size
    locked["files"][0]["mtime_ns"] = source_path.stat().st_mtime_ns
    locked["lock_digest"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in locked.items() if key not in {"created_utc", "lock_digest"}},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    lock_path.write_text(json.dumps(locked), encoding="utf-8")
    with pytest.raises(DataPolicyError, match="checksum changed"):
        load_and_verify_source_lock(config, verify_hashes=True)


def test_enforcement_requires_allowed_sources_and_fresh_audit(tmp_path: Path) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    blocked = source_mapping(license_status="blocked")
    config["data"]["sources"] = [blocked]
    with pytest.raises(DataPolicyError, match="No approved source"):
        enforce_data_policy(config, require_artifacts=False)

    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"text":"안전한 문장입니다."}\n', encoding="utf-8")
    approved = source_mapping()
    approved["path"] = str(source_path)
    config["data"]["sources"] = [approved]
    config["data_policy"].update(
        source_lock_path=str(tmp_path / "lock.json"),
        audit_path=str(tmp_path / "audit.json"),
    )
    build_source_lock(config)
    with pytest.raises(DataPolicyError, match="Missing or invalid data audit"):
        enforce_data_policy(config, require_artifacts=True)
    Path(config["data_policy"]["audit_path"]).write_text(
        json.dumps({"source_lock_digest": "stale", "filter_version": "stale"}), encoding="utf-8"
    )
    with pytest.raises(DataPolicyError, match="audit digest is missing"):
        enforce_data_policy(config, require_artifacts=True)
    audit_allowed_sources(config)
    assert enforce_data_policy(config, require_artifacts=True) == []


def test_benchmark_registry_change_invalidates_a_completed_audit(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"text":"safe training sentence"}\n', encoding="utf-8")
    denylist = tmp_path / "denylist.txt"
    denylist.write_text(content_hash("benchmark version one") + "\n", encoding="utf-8")
    config = copy.deepcopy(DEFAULT_CONFIG)
    approved = source_mapping()
    approved["path"] = str(source_path)
    config["data"]["sources"] = [approved]
    config["data"]["min_chars"] = 0
    config["data_policy"].update(
        source_lock_path=str(tmp_path / "lock.json"),
        audit_path=str(tmp_path / "audit.json"),
        benchmark_denylist_path=str(denylist),
        require_benchmark_denylist=True,
    )
    build_source_lock(config)
    audit_allowed_sources(config)

    denylist.write_text(content_hash("benchmark version two") + "\n", encoding="utf-8")

    with pytest.raises(DataPolicyError, match="benchmark evidence"):
        enforce_data_policy(config, require_artifacts=True)
