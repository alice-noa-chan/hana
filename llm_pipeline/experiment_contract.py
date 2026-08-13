"""Validate small, explicit research contracts before comparing model runs.

This module is intentionally separate from the training pipeline.  It does not
start a run, read a checkpoint, or decide that an idea is novel.  Its job is
smaller: make the planned comparison precise enough that another person can
check it.

The public entry points are:

``load_registry``
    Read and validate a schema-v1 YAML registry.
``parse_registry``
    Validate an already-loaded Python mapping.
``config_diff_keys``
    List the exact leaf settings that differ between two configurations.
``verify_changed_keys``
    Check that an experiment arm declared every changed setting, and no extra
    setting.
``evaluate_candidate_results``
    Return one conservative decision for a complete baseline/candidate pair.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

SCHEMA_VERSION = 1

ALLOWED_STUDY_KINDS = frozenset(
    {
        "architecture_ablation",
        "data_portability",
        "systems_speed",
    }
)

# These names describe the evidence honestly.  The registry deliberately does
# not provide a "novel" or "first" status.  Those claims require a separate,
# human-reviewed search of prior work.
ALLOWED_CLAIM_STATUSES = frozenset(
    {
        "unverified_hypothesis",
        "replication",
        "prior_art_extension",
        "negative_result",
    }
)

ALLOWED_BUDGET_BASES = frozenset(
    {
        "train_tokens",
        "optimizer_steps",
        "wall_clock_seconds",
        "evaluation_examples",
    }
)
ALLOWED_DIRECTIONS = frozenset({"min", "max"})

Decision = Literal["promote", "reject", "inconclusive", "incomplete"]

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]*$")
_MISSING = object()


class ExperimentContractError(ValueError):
    """Raised when a registry or declared configuration difference is invalid."""


@dataclass(frozen=True)
class Budget:
    """One shared resource limit used by every arm in a study."""

    basis: str
    value: float | int


@dataclass(frozen=True)
class Controls:
    """Values that make repeated baseline and candidate runs comparable."""

    seeds: tuple[int, ...]
    budget: Budget


@dataclass(frozen=True)
class PrimaryMetric:
    """The pre-registered metric that can justify promoting a candidate."""

    name: str
    direction: str
    minimum_improvement: float
    maximum_regression: float


@dataclass(frozen=True)
class GuardrailMetric:
    """A metric that may reject a candidate when it becomes too much worse."""

    name: str
    direction: str
    maximum_regression: float


@dataclass(frozen=True)
class Metrics:
    """All measurements required before a result is considered complete."""

    primary: PrimaryMetric
    required: tuple[str, ...]
    guardrails: tuple[GuardrailMetric, ...]


@dataclass(frozen=True)
class Arm:
    """One baseline or candidate configuration in a study."""

    id: str
    config: str
    changed_keys: tuple[str, ...]


@dataclass(frozen=True)
class Study:
    """A validated comparison plan with exactly one referenced baseline."""

    id: str
    kind: str
    hypothesis: str
    claim_status: str
    baseline_arm: str
    controls: Controls
    metrics: Metrics
    arms: tuple[Arm, ...]

    def arm(self, arm_id: str) -> Arm:
        """Return one arm, or raise a clear error for an unknown identifier."""

        for arm in self.arms:
            if arm.id == arm_id:
                return arm
        raise ExperimentContractError(f"Study '{self.id}' does not contain arm '{arm_id}'.")


@dataclass(frozen=True)
class ExperimentRegistry:
    """A complete schema-v1 experiment registry."""

    schema_version: int
    studies: tuple[Study, ...]

    def study(self, study_id: str) -> Study:
        """Return one study, or raise a clear error for an unknown identifier."""

        for study in self.studies:
            if study.id == study_id:
                return study
        raise ExperimentContractError(f"Registry does not contain study '{study_id}'.")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentContractError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExperimentContractError(f"{label} must be a list.")
    return value


def _known_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ExperimentContractError(f"{label} contains unknown keys: {unknown}")


def _required_keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ExperimentContractError(f"{label} is missing required keys: {missing}")


def _identifier(value: Any, label: str) -> str:
    identifier = str(value or "").strip()
    if not _IDENTIFIER.fullmatch(identifier):
        raise ExperimentContractError(
            f"{label} must start with a lowercase letter and contain only lowercase letters, digits, '_' or '-'."
        )
    return identifier


def _non_empty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentContractError(f"{label} must be a non-empty string.")
    return value.strip()


def _finite_number(value: Any, label: str, *, positive: bool = False, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentContractError(f"{label} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ExperimentContractError(f"{label} must be finite.")
    if positive and number <= 0:
        raise ExperimentContractError(f"{label} must be greater than zero.")
    if non_negative and number < 0:
        raise ExperimentContractError(f"{label} must be zero or greater.")
    return number


def _metric_name(value: Any, label: str) -> str:
    name = _non_empty_text(value, label)
    if any(character.isspace() for character in name):
        raise ExperimentContractError(f"{label} must not contain whitespace.")
    return name


def _direction(value: Any, label: str) -> str:
    direction = str(value or "").strip()
    if direction not in ALLOWED_DIRECTIONS:
        raise ExperimentContractError(f"{label} must be one of: {sorted(ALLOWED_DIRECTIONS)}")
    return direction


def _parse_budget(value: Any, label: str) -> Budget:
    mapping = _mapping(value, label)
    _known_keys(mapping, {"basis", "value"}, label)
    _required_keys(mapping, {"basis", "value"}, label)
    basis = str(mapping["basis"] or "").strip()
    if basis not in ALLOWED_BUDGET_BASES:
        raise ExperimentContractError(f"{label}.basis must be one of: {sorted(ALLOWED_BUDGET_BASES)}")
    number = _finite_number(mapping["value"], f"{label}.value", positive=True)
    if basis in {"train_tokens", "optimizer_steps", "evaluation_examples"} and not number.is_integer():
        raise ExperimentContractError(f"{label}.value must be a whole number when basis is '{basis}'.")
    return Budget(basis=basis, value=int(number) if number.is_integer() else number)


def _parse_controls(value: Any, label: str) -> Controls:
    mapping = _mapping(value, label)
    _known_keys(mapping, {"seeds", "budget"}, label)
    _required_keys(mapping, {"seeds", "budget"}, label)
    raw_seeds = _sequence(mapping["seeds"], f"{label}.seeds")
    if not raw_seeds:
        raise ExperimentContractError(f"{label}.seeds must contain at least one seed.")
    seeds: list[int] = []
    for index, value in enumerate(raw_seeds):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ExperimentContractError(f"{label}.seeds[{index}] must be a non-negative integer.")
        seeds.append(value)
    if len(seeds) != len(set(seeds)):
        raise ExperimentContractError(f"{label}.seeds must not contain duplicates.")
    return Controls(seeds=tuple(seeds), budget=_parse_budget(mapping["budget"], f"{label}.budget"))


def _parse_primary_metric(value: Any, label: str) -> PrimaryMetric:
    mapping = _mapping(value, label)
    allowed = {"name", "direction", "minimum_improvement", "maximum_regression"}
    _known_keys(mapping, allowed, label)
    _required_keys(mapping, allowed, label)
    return PrimaryMetric(
        name=_metric_name(mapping["name"], f"{label}.name"),
        direction=_direction(mapping["direction"], f"{label}.direction"),
        # A positive threshold prevents an unchanged result from being promoted.
        minimum_improvement=_finite_number(
            mapping["minimum_improvement"], f"{label}.minimum_improvement", positive=True
        ),
        maximum_regression=_finite_number(
            mapping["maximum_regression"], f"{label}.maximum_regression", non_negative=True
        ),
    )


def _parse_guardrails(value: Any, label: str) -> tuple[GuardrailMetric, ...]:
    mapping = _mapping(value, label)
    guardrails: list[GuardrailMetric] = []
    for name, raw_rule in mapping.items():
        metric_name = _metric_name(name, f"{label} metric name")
        rule = _mapping(raw_rule, f"{label}.{metric_name}")
        _known_keys(rule, {"direction", "maximum_regression"}, f"{label}.{metric_name}")
        _required_keys(rule, {"direction", "maximum_regression"}, f"{label}.{metric_name}")
        guardrails.append(
            GuardrailMetric(
                name=metric_name,
                direction=_direction(rule["direction"], f"{label}.{metric_name}.direction"),
                maximum_regression=_finite_number(
                    rule["maximum_regression"],
                    f"{label}.{metric_name}.maximum_regression",
                    non_negative=True,
                ),
            )
        )
    return tuple(guardrails)


def _parse_metrics(value: Any, label: str) -> Metrics:
    mapping = _mapping(value, label)
    _known_keys(mapping, {"primary", "required", "guardrails"}, label)
    _required_keys(mapping, {"primary", "required", "guardrails"}, label)
    primary = _parse_primary_metric(mapping["primary"], f"{label}.primary")

    raw_required = _sequence(mapping["required"], f"{label}.required")
    if not raw_required:
        raise ExperimentContractError(f"{label}.required must contain at least one metric name.")
    required = tuple(_metric_name(item, f"{label}.required[{index}]") for index, item in enumerate(raw_required))
    if len(required) != len(set(required)):
        raise ExperimentContractError(f"{label}.required must not contain duplicates.")

    guardrails = _parse_guardrails(mapping["guardrails"], f"{label}.guardrails")
    guardrail_names = {guardrail.name for guardrail in guardrails}
    if primary.name not in required:
        raise ExperimentContractError(f"{label}.required must include the primary metric '{primary.name}'.")
    missing_guardrails = sorted(guardrail_names - set(required))
    if missing_guardrails:
        raise ExperimentContractError(f"{label}.required must include every guardrail metric: {missing_guardrails}")
    if primary.name in guardrail_names:
        raise ExperimentContractError(f"Primary metric '{primary.name}' must not also be a guardrail.")
    return Metrics(primary=primary, required=required, guardrails=guardrails)


def _valid_changed_key(value: Any, label: str) -> str:
    key = _non_empty_text(value, label)
    if key.startswith(".") or key.endswith(".") or ".." in key:
        raise ExperimentContractError(f"{label} must be a dot-separated configuration path.")
    return key


def _parse_arm(value: Any, label: str) -> Arm:
    mapping = _mapping(value, label)
    _known_keys(mapping, {"id", "config", "changed_keys"}, label)
    _required_keys(mapping, {"id", "config", "changed_keys"}, label)
    changed_values = _sequence(mapping["changed_keys"], f"{label}.changed_keys")
    changed_keys = tuple(
        _valid_changed_key(item, f"{label}.changed_keys[{index}]") for index, item in enumerate(changed_values)
    )
    if len(changed_keys) != len(set(changed_keys)):
        raise ExperimentContractError(f"{label}.changed_keys must not contain duplicates.")
    return Arm(
        id=_identifier(mapping["id"], f"{label}.id"),
        config=_non_empty_text(mapping["config"], f"{label}.config"),
        changed_keys=changed_keys,
    )


def _parse_study(value: Any, label: str) -> Study:
    mapping = _mapping(value, label)
    allowed = {
        "id",
        "kind",
        "hypothesis",
        "claim_status",
        "baseline_arm",
        "controls",
        "metrics",
        "arms",
    }
    _known_keys(mapping, allowed, label)
    _required_keys(mapping, allowed, label)

    study_id = _identifier(mapping["id"], f"{label}.id")
    kind = str(mapping["kind"] or "").strip()
    if kind not in ALLOWED_STUDY_KINDS:
        raise ExperimentContractError(f"{label}.kind must be one of: {sorted(ALLOWED_STUDY_KINDS)}")
    claim_status = str(mapping["claim_status"] or "").strip()
    if claim_status not in ALLOWED_CLAIM_STATUSES:
        raise ExperimentContractError(f"{label}.claim_status must be one of: {sorted(ALLOWED_CLAIM_STATUSES)}")

    raw_arms = _sequence(mapping["arms"], f"{label}.arms")
    if len(raw_arms) < 2:
        raise ExperimentContractError(f"{label}.arms must contain one baseline and at least one candidate.")
    arms = tuple(_parse_arm(item, f"{label}.arms[{index}]") for index, item in enumerate(raw_arms))
    arm_ids = [arm.id for arm in arms]
    if len(arm_ids) != len(set(arm_ids)):
        raise ExperimentContractError(f"{label}.arms must use unique arm IDs.")

    baseline_arm = _identifier(mapping["baseline_arm"], f"{label}.baseline_arm")
    baseline_matches = [arm for arm in arms if arm.id == baseline_arm]
    if len(baseline_matches) != 1:
        raise ExperimentContractError(
            f"{label}.baseline_arm must reference exactly one arm; found {len(baseline_matches)} matches for '{baseline_arm}'."
        )
    if baseline_matches[0].changed_keys:
        raise ExperimentContractError(f"{label} baseline arm '{baseline_arm}' must have an empty changed_keys list.")
    for arm in arms:
        if arm.id != baseline_arm and not arm.changed_keys:
            raise ExperimentContractError(f"{label} candidate arm '{arm.id}' must declare at least one changed key.")

    return Study(
        id=study_id,
        kind=kind,
        hypothesis=_non_empty_text(mapping["hypothesis"], f"{label}.hypothesis"),
        claim_status=claim_status,
        baseline_arm=baseline_arm,
        controls=_parse_controls(mapping["controls"], f"{label}.controls"),
        metrics=_parse_metrics(mapping["metrics"], f"{label}.metrics"),
        arms=arms,
    )


def parse_registry(payload: Mapping[str, Any]) -> ExperimentRegistry:
    """Validate an already-loaded registry mapping and return immutable objects."""

    mapping = _mapping(payload, "registry")
    _known_keys(mapping, {"schema_version", "studies"}, "registry")
    _required_keys(mapping, {"schema_version", "studies"}, "registry")
    version = mapping["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
        raise ExperimentContractError(f"registry.schema_version must be {SCHEMA_VERSION}.")

    raw_studies = _sequence(mapping["studies"], "registry.studies")
    if not raw_studies:
        raise ExperimentContractError("registry.studies must contain at least one study.")
    studies = tuple(_parse_study(item, f"registry.studies[{index}]") for index, item in enumerate(raw_studies))
    study_ids = [study.id for study in studies]
    if len(study_ids) != len(set(study_ids)):
        raise ExperimentContractError("registry.studies must use unique study IDs.")
    return ExperimentRegistry(schema_version=SCHEMA_VERSION, studies=studies)


def load_registry(path: str | Path) -> ExperimentRegistry:
    """Read one UTF-8 YAML file and validate it as a schema-v1 registry."""

    registry_path = Path(path)
    try:
        payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExperimentContractError(f"Could not read experiment registry: {registry_path}") from exc
    except yaml.YAMLError as exc:
        raise ExperimentContractError(f"Experiment registry is not valid YAML: {registry_path}") from exc
    if payload is None:
        raise ExperimentContractError(f"Experiment registry is empty: {registry_path}")
    return parse_registry(_mapping(payload, "registry"))


def _flatten_config(value: Any, prefix: str, output: dict[str, Any]) -> None:
    """Flatten mappings while treating lists as one intentional config value."""

    if isinstance(value, Mapping) and value:
        for key in sorted(value, key=lambda item: str(item)):
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            _flatten_config(value[key], path, output)
        return
    # Empty mappings and all sequences are values in their own right.  Treating
    # a list as one value keeps paths readable, for example ``data.sources``.
    output[prefix] = value


def _strictly_equal(left: Any, right: Any) -> bool:
    if left is _MISSING or right is _MISSING:
        return left is right
    if type(left) is not type(right):
        return False
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _ignored(path: str, ignored_keys: frozenset[str]) -> bool:
    return any(path == ignored or path.startswith(ignored + ".") for ignored in ignored_keys)


def config_diff_keys(
    baseline_config: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    *,
    ignored_keys: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return the exact sorted leaf paths that differ between two configs.

    Mapping values are compared recursively.  Lists are compared as one value,
    because list indices are usually not stable experiment controls.  Callers
    may explicitly ignore operational fields such as output directories.
    """

    baseline_flat: dict[str, Any] = {}
    candidate_flat: dict[str, Any] = {}
    _flatten_config(_mapping(baseline_config, "baseline_config"), "", baseline_flat)
    _flatten_config(_mapping(candidate_config, "candidate_config"), "", candidate_flat)
    ignored = frozenset(_valid_changed_key(key, "ignored_keys item") for key in ignored_keys)
    keys = sorted(set(baseline_flat) | set(candidate_flat))
    return tuple(
        key
        for key in keys
        if key
        and not _ignored(key, ignored)
        and not _strictly_equal(baseline_flat.get(key, _MISSING), candidate_flat.get(key, _MISSING))
    )


def verify_changed_keys(
    baseline_config: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    declared_changed_keys: Iterable[str],
    *,
    ignored_keys: Iterable[str] = (),
) -> tuple[str, ...]:
    """Verify that the declared changed keys exactly match the actual config diff.

    The function returns the sorted actual paths when the declaration is exact.
    It raises :class:`ExperimentContractError` when a setting was changed but
    not declared, or declared but not changed.
    """

    declared_values = tuple(_valid_changed_key(key, "declared_changed_keys item") for key in declared_changed_keys)
    if len(declared_values) != len(set(declared_values)):
        raise ExperimentContractError("declared_changed_keys must not contain duplicates.")
    actual = config_diff_keys(baseline_config, candidate_config, ignored_keys=ignored_keys)
    declared = tuple(sorted(declared_values))
    if actual != declared:
        undeclared = sorted(set(actual) - set(declared))
        unchanged = sorted(set(declared) - set(actual))
        details = []
        if undeclared:
            details.append(f"changed but not declared: {undeclared}")
        if unchanged:
            details.append(f"declared but unchanged: {unchanged}")
        raise ExperimentContractError("Changed-key declaration does not match the config diff; " + "; ".join(details))
    return actual


def verify_arm_config_diff(
    study: Study,
    candidate_arm_id: str,
    baseline_config: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    *,
    ignored_keys: Iterable[str] = (),
) -> tuple[str, ...]:
    """Verify one candidate using the changed_keys declared by its study arm."""

    if candidate_arm_id == study.baseline_arm:
        raise ExperimentContractError("A baseline arm cannot be verified as its own candidate.")
    arm = study.arm(candidate_arm_id)
    return verify_changed_keys(
        baseline_config,
        candidate_config,
        arm.changed_keys,
        ignored_keys=ignored_keys,
    )


def _metric_improvement(baseline: float, candidate: float, direction: str) -> float:
    return candidate - baseline if direction == "max" else baseline - candidate


def _complete_metric_rows(
    study: Study,
    candidate_arm_id: str,
    results: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, float]] | None:
    expected_arms = {study.baseline_arm, candidate_arm_id}
    expected_seeds = set(study.controls.seeds)
    required_metrics = set(study.metrics.required)
    collected: dict[tuple[str, int], dict[str, float]] = {}

    for raw_row in results:
        if not isinstance(raw_row, Mapping):
            return None
        arm_id = raw_row.get("arm_id")
        seed = raw_row.get("seed")
        if arm_id not in expected_arms or seed not in expected_seeds:
            # Results for other arms or seeds do not affect this pair.
            continue
        if isinstance(seed, bool) or not isinstance(seed, int):
            return None
        key = (str(arm_id), seed)
        if key in collected:
            return None
        raw_metrics = raw_row.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            return None
        row: dict[str, float] = {}
        for name in required_metrics:
            value = raw_metrics.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            number = float(value)
            if not math.isfinite(number):
                return None
            row[name] = number
        collected[key] = row

    expected = {(arm_id, seed) for arm_id in expected_arms for seed in expected_seeds}
    return collected if set(collected) == expected else None


def evaluate_candidate_results(
    study: Study,
    candidate_arm_id: str,
    results: Iterable[Mapping[str, Any]],
) -> Decision:
    """Evaluate one baseline/candidate pair using only pre-registered rules.

    Each result row must contain ``arm_id``, ``seed``, and a ``metrics`` mapping.
    Every required seed and metric must exist once for both arms.  Missing,
    duplicate, non-numeric, or non-finite evidence returns ``incomplete``.

    A complete result is ``reject`` when the primary metric or any guardrail
    regresses beyond its allowed amount.  It is ``promote`` only when the
    primary metric reaches its pre-registered improvement and every guardrail
    passes.  All other complete results are ``inconclusive``.
    """

    if candidate_arm_id == study.baseline_arm:
        raise ExperimentContractError("A baseline arm cannot be evaluated as its own candidate.")
    study.arm(candidate_arm_id)
    collected = _complete_metric_rows(study, candidate_arm_id, results)
    if collected is None:
        return "incomplete"

    def mean(arm_id: str, metric: str) -> float:
        values = [collected[(arm_id, seed)][metric] for seed in study.controls.seeds]
        return sum(values) / len(values)

    primary = study.metrics.primary
    primary_improvement = _metric_improvement(
        mean(study.baseline_arm, primary.name),
        mean(candidate_arm_id, primary.name),
        primary.direction,
    )
    if primary_improvement < -primary.maximum_regression:
        return "reject"

    for guardrail in study.metrics.guardrails:
        improvement = _metric_improvement(
            mean(study.baseline_arm, guardrail.name),
            mean(candidate_arm_id, guardrail.name),
            guardrail.direction,
        )
        if improvement < -guardrail.maximum_regression:
            return "reject"

    if primary_improvement >= primary.minimum_improvement:
        return "promote"
    return "inconclusive"


__all__ = [
    "ALLOWED_BUDGET_BASES",
    "ALLOWED_CLAIM_STATUSES",
    "ALLOWED_STUDY_KINDS",
    "SCHEMA_VERSION",
    "Arm",
    "Budget",
    "Controls",
    "Decision",
    "ExperimentContractError",
    "ExperimentRegistry",
    "GuardrailMetric",
    "Metrics",
    "PrimaryMetric",
    "Study",
    "config_diff_keys",
    "evaluate_candidate_results",
    "load_registry",
    "parse_registry",
    "verify_arm_config_diff",
    "verify_changed_keys",
]
