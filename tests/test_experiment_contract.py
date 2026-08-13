from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from llm_pipeline.experiment_contract import (
    ExperimentContractError,
    config_diff_keys,
    evaluate_candidate_results,
    load_registry,
    parse_registry,
    verify_arm_config_diff,
    verify_changed_keys,
)

ROOT = Path(__file__).resolve().parents[1]


def valid_registry_payload() -> dict:
    return {
        "schema_version": 1,
        "studies": [
            {
                "id": "workspace_v1",
                "kind": "architecture_ablation",
                "hypothesis": "A small causal workspace may improve validation loss at the same token budget.",
                "claim_status": "unverified_hypothesis",
                "baseline_arm": "dense_control",
                "controls": {
                    "seeds": [42, 43],
                    "budget": {"basis": "train_tokens", "value": 100_000},
                },
                "metrics": {
                    "primary": {
                        "name": "valid_loss",
                        "direction": "min",
                        "minimum_improvement": 0.02,
                        "maximum_regression": 0.01,
                    },
                    "required": ["valid_loss", "token_accuracy", "train_tokens_per_sec_global"],
                    "guardrails": {
                        "token_accuracy": {"direction": "max", "maximum_regression": 0.01},
                        "train_tokens_per_sec_global": {"direction": "max", "maximum_regression": 100.0},
                    },
                },
                "arms": [
                    {
                        "id": "dense_control",
                        "config": "workspace_control.yaml",
                        "changed_keys": [],
                    },
                    {
                        "id": "workspace_on",
                        "config": "workspace_on.yaml",
                        "changed_keys": [
                            "cognitive_architecture.enabled",
                            "cognitive_architecture.workspace.enabled",
                        ],
                    },
                ],
            }
        ],
    }


def result_rows(
    *,
    candidate_losses: tuple[float, float] = (0.96, 0.98),
    candidate_accuracy: tuple[float, float] = (0.71, 0.73),
    candidate_speed: tuple[float, float] = (920.0, 940.0),
) -> list[dict]:
    rows = []
    baseline_values = {
        42: {"valid_loss": 1.00, "token_accuracy": 0.70, "train_tokens_per_sec_global": 1000.0},
        43: {"valid_loss": 1.02, "token_accuracy": 0.72, "train_tokens_per_sec_global": 1020.0},
    }
    for seed, metrics in baseline_values.items():
        rows.append({"arm_id": "dense_control", "seed": seed, "metrics": metrics})
    for index, seed in enumerate((42, 43)):
        rows.append(
            {
                "arm_id": "workspace_on",
                "seed": seed,
                "metrics": {
                    "valid_loss": candidate_losses[index],
                    "token_accuracy": candidate_accuracy[index],
                    "train_tokens_per_sec_global": candidate_speed[index],
                },
            }
        )
    return rows


def test_valid_registry_is_immutable_and_lookup_helpers_are_clear(tmp_path: Path) -> None:
    payload = valid_registry_payload()
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    registry = load_registry(path)

    assert registry.schema_version == 1
    study = registry.study("workspace_v1")
    assert study.baseline_arm == "dense_control"
    assert study.controls.seeds == (42, 43)
    assert study.controls.budget.value == 100_000
    assert study.arm("workspace_on").changed_keys == (
        "cognitive_architecture.enabled",
        "cognitive_architecture.workspace.enabled",
    )
    with pytest.raises(ExperimentContractError, match="does not contain study"):
        registry.study("missing")


def test_repository_example_is_a_valid_registry() -> None:
    registry = load_registry(ROOT / "configs/experiments/registry.example.yaml")

    assert registry.study("workspace_v1").claim_status == "unverified_hypothesis"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(schema_version=2), "schema_version"),
        (lambda payload: payload.update(schema_version=1.0), "schema_version"),
        (lambda payload: payload["studies"][0].update(kind="unknown"), "kind must be one of"),
        (lambda payload: payload["studies"][0].update(claim_status="novel"), "claim_status must be one of"),
        (lambda payload: payload["studies"][0].update(hypothesis="   "), "hypothesis must be a non-empty"),
        (lambda payload: payload["studies"][0].update(baseline_arm="missing"), "exactly one arm"),
        (
            lambda payload: payload["studies"][0]["arms"].append(copy.deepcopy(payload["studies"][0]["arms"][1])),
            "unique arm IDs",
        ),
        (
            lambda payload: payload["studies"][0]["arms"][0].update(changed_keys=["model.hidden_size"]),
            "baseline arm.*empty changed_keys",
        ),
        (
            lambda payload: payload["studies"][0]["arms"][1].update(changed_keys=[]),
            "candidate arm.*at least one changed key",
        ),
    ],
)
def test_registry_rejects_ambiguous_identity_and_unsupported_claims(mutate, message: str) -> None:
    payload = valid_registry_payload()
    mutate(payload)

    with pytest.raises(ExperimentContractError, match=message):
        parse_registry(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload["studies"][0]["controls"].update(seeds=[]), "at least one seed"),
        (lambda payload: payload["studies"][0]["controls"].update(seeds=[42, 42]), "must not contain duplicates"),
        (
            lambda payload: payload["studies"][0]["controls"]["budget"].update(value=0),
            "must be greater than zero",
        ),
        (
            lambda payload: payload["studies"][0]["controls"]["budget"].update(basis="train_tokens", value=1.5),
            "must be a whole number",
        ),
        (
            lambda payload: payload["studies"][0]["metrics"]["primary"].update(minimum_improvement=0),
            "must be greater than zero",
        ),
        (
            lambda payload: payload["studies"][0]["metrics"].update(required=["token_accuracy"]),
            "must include the primary metric",
        ),
        (
            lambda payload: payload["studies"][0]["metrics"].update(required=["valid_loss"]),
            "must include every guardrail metric",
        ),
    ],
)
def test_registry_requires_controlled_budget_seeds_and_metrics(mutate, message: str) -> None:
    payload = valid_registry_payload()
    mutate(payload)

    with pytest.raises(ExperimentContractError, match=message):
        parse_registry(payload)


def test_config_diff_is_sorted_strict_and_treats_lists_as_one_value() -> None:
    baseline = {
        "model": {"enabled": False, "layers": 4},
        "data": {"sources": ["ko.jsonl"]},
        "run": {"output_dir": "control"},
    }
    candidate = {
        "model": {"enabled": True, "layers": 4},
        "data": {"sources": ["ja.jsonl"]},
        "run": {"output_dir": "candidate"},
    }

    assert config_diff_keys(baseline, candidate) == (
        "data.sources",
        "model.enabled",
        "run.output_dir",
    )
    assert config_diff_keys(baseline, candidate, ignored_keys=["run"]) == (
        "data.sources",
        "model.enabled",
    )

    # Python normally considers True equal to 1.  Config comparison is stricter
    # because changing a Boolean switch into an integer should be visible.
    assert config_diff_keys({"feature": True}, {"feature": 1}) == ("feature",)


def test_declared_changed_keys_must_match_the_actual_diff_exactly() -> None:
    baseline = {"model": {"hidden_size": 64, "workspace": False}, "run": {"output_dir": "a"}}
    candidate = {"model": {"hidden_size": 64, "workspace": True}, "run": {"output_dir": "b"}}

    assert verify_changed_keys(
        baseline,
        candidate,
        ["model.workspace"],
        ignored_keys=["run.output_dir"],
    ) == ("model.workspace",)

    with pytest.raises(ExperimentContractError, match=r"changed but not declared.*run.output_dir"):
        verify_changed_keys(baseline, candidate, ["model.workspace"])
    with pytest.raises(ExperimentContractError, match=r"declared but unchanged.*model.hidden_size"):
        verify_changed_keys(
            baseline,
            candidate,
            ["model.hidden_size", "model.workspace"],
            ignored_keys=["run.output_dir"],
        )


def test_arm_diff_helper_uses_the_registry_declaration() -> None:
    study = parse_registry(valid_registry_payload()).study("workspace_v1")
    baseline = {"cognitive_architecture": {"enabled": False, "workspace": {"enabled": False}}}
    candidate = {"cognitive_architecture": {"enabled": True, "workspace": {"enabled": True}}}

    assert verify_arm_config_diff(study, "workspace_on", baseline, candidate) == (
        "cognitive_architecture.enabled",
        "cognitive_architecture.workspace.enabled",
    )
    with pytest.raises(ExperimentContractError, match="baseline arm cannot"):
        verify_arm_config_diff(study, "dense_control", baseline, baseline)


def test_complete_results_promote_only_after_primary_and_guardrails_pass() -> None:
    study = parse_registry(valid_registry_payload()).study("workspace_v1")

    assert evaluate_candidate_results(study, "workspace_on", result_rows()) == "promote"


def test_complete_results_reject_a_primary_or_guardrail_regression() -> None:
    study = parse_registry(valid_registry_payload()).study("workspace_v1")

    assert (
        evaluate_candidate_results(
            study,
            "workspace_on",
            result_rows(candidate_losses=(1.04, 1.06)),
        )
        == "reject"
    )
    assert (
        evaluate_candidate_results(
            study,
            "workspace_on",
            result_rows(candidate_accuracy=(0.67, 0.68)),
        )
        == "reject"
    )


def test_complete_but_small_primary_change_is_inconclusive() -> None:
    study = parse_registry(valid_registry_payload()).study("workspace_v1")

    assert (
        evaluate_candidate_results(
            study,
            "workspace_on",
            result_rows(candidate_losses=(0.995, 1.015)),
        )
        == "inconclusive"
    )


@pytest.mark.parametrize("break_rows", ["missing", "duplicate", "metric", "non_finite"])
def test_missing_or_ambiguous_evidence_is_incomplete(break_rows: str) -> None:
    study = parse_registry(valid_registry_payload()).study("workspace_v1")
    rows = result_rows()
    if break_rows == "missing":
        rows.pop()
    elif break_rows == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    elif break_rows == "metric":
        del rows[-1]["metrics"]["valid_loss"]
    else:
        rows[-1]["metrics"]["valid_loss"] = float("nan")

    assert evaluate_candidate_results(study, "workspace_on", rows) == "incomplete"


def test_unknown_or_baseline_candidate_id_is_an_error() -> None:
    study = parse_registry(valid_registry_payload()).study("workspace_v1")

    with pytest.raises(ExperimentContractError, match="does not contain arm"):
        evaluate_candidate_results(study, "missing", result_rows())
    with pytest.raises(ExperimentContractError, match="baseline arm cannot"):
        evaluate_candidate_results(study, "dense_control", result_rows())
