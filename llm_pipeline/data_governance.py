"""Fail-closed source policy, source locking, and privacy audit helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json
from .errors import DataPolicyError

POLICY_VERSION = "internal-research-v3"
FILTER_VERSION = "pii-benchmark-v2"
ALLOWED_LICENSE_STATUSES = {"approved", "restricted_research"}
ALLOWED_REVIEW_STATUSES = {"author_controlled", "filtered", "human_reviewed"}
LICENSE_STATUSES = ALLOWED_LICENSE_STATUSES | {"review_required", "blocked"}
REVIEW_STATUSES = ALLOWED_REVIEW_STATUSES | {"unknown", "blocked"}
SOURCE_PURPOSES = {"training", "evaluation"}

_EMAIL = re.compile(r"(?i)(?<![\w.+-])[\w.+-]{1,64}@[\w.-]{1,253}\.[a-z]{2,24}(?![\w.-])")
_KR_PHONE = re.compile(r"(?<!\d)(?:\+?82[- .]?)?0?1[016789][- .]?\d{3,4}[- .]?\d{4}(?!\d)")
_JP_PHONE = re.compile(r"(?<!\d)(?:\+?81[- .]?)?0(?:70|80|90)[- .]?\d{4}[- .]?\d{4}(?!\d)")
_KR_RESIDENT_ID = re.compile(r"(?<!\d)\d{6}[- ]?[1-4]\d{6}(?!\d)")
_CREDENTIAL = re.compile(
    r"(?i)\b(?:api[_ -]?key|access[_ -]?token|client[_ -]?secret|password|passwd)\b\s*[:=]\s*['\"]?[A-Za-z0-9_./+~-]{8,}"
)
_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_SHA256_LINE = re.compile(r"[0-9a-fA-F]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


EMPTY_BENCHMARK_DENYLIST_DIGEST = _json_digest([])


def _luhn_valid(candidate: str) -> bool:
    digits = [int(value) for value in re.sub(r"\D", "", candidate)]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, value in enumerate(digits):
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        checksum += value
    return checksum % 10 == 0


@dataclass(frozen=True)
class SourceSpec:
    """Normalized source policy used by locking and runtime selection."""

    name: str
    license_status: str
    allowed_uses: tuple[str, ...]
    pii_status: str
    child_safety_status: str
    purpose: str = "training"
    source_url: str | None = None
    revision: str | None = None
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    evidence: str | None = None
    evidence_path: str | None = None
    evidence_sha256: str | None = None

    @classmethod
    def from_mapping(cls, source: dict[str, Any]) -> SourceSpec:
        provenance = source.get("provenance") if isinstance(source.get("provenance"), dict) else {}
        legacy_license = str(source.get("license", "")).strip()
        name = str(source.get("name", "<unnamed>"))

        license_status = str(provenance.get("license_status", source.get("license_status", ""))).lower()
        allowed_uses = provenance.get("allowed_uses", source.get("allowed_uses", []))
        pii_status = str(provenance.get("pii_status", source.get("pii_status", "unknown"))).lower()
        child_status = str(provenance.get("child_safety_status", source.get("child_safety_status", "unknown"))).lower()
        purpose_value = provenance.get("purpose", source.get("purpose", "training"))
        purpose = str(purpose_value).strip().lower()

        # This legacy marker is an immutable data-boundary signal. It must not
        # be overridden by a permissive license or a disabled license policy.
        if "EVALUATION_ONLY" in legacy_license.upper():
            purpose = "evaluation"

        # One-time conservative migration for the local v1 manifest.  Only
        # internally authored material is promoted automatically; every other
        # ambiguous legacy label remains excluded.
        if not license_status:
            if legacy_license == "INTERNAL_AUTHORED":
                license_status = "approved"
                allowed_uses = ["internal_noncommercial_research"]
                pii_status = "author_controlled"
                child_status = "author_controlled"
            else:
                license_status = "blocked" if "EVALUATION_ONLY" in legacy_license else "review_required"

        uses = (allowed_uses,) if isinstance(allowed_uses, str) else tuple(str(value) for value in allowed_uses)
        if purpose not in SOURCE_PURPOSES:
            raise DataPolicyError(f"Source {name!r} has invalid purpose={purpose!r}")
        if license_status not in LICENSE_STATUSES:
            raise DataPolicyError(f"Source {name!r} has invalid license_status={license_status!r}")
        if pii_status not in REVIEW_STATUSES:
            raise DataPolicyError(f"Source {name!r} has invalid pii_status={pii_status!r}")
        if child_status not in REVIEW_STATUSES:
            raise DataPolicyError(f"Source {name!r} has invalid child_safety_status={child_status!r}")
        return cls(
            name=name,
            license_status=license_status,
            allowed_uses=uses,
            pii_status=pii_status,
            child_safety_status=child_status,
            purpose=purpose,
            source_url=provenance.get("source_url", source.get("license_url")),
            revision=provenance.get("revision"),
            reviewed_by=provenance.get("reviewed_by"),
            reviewed_at=provenance.get("reviewed_at"),
            evidence=provenance.get("evidence"),
            evidence_path=provenance.get("evidence_path"),
            evidence_sha256=provenance.get("evidence_sha256"),
        )


@dataclass(frozen=True)
class DataPolicy:
    """The only supported production policy: internal noncommercial research."""

    use_case: str = "internal_noncommercial_research"
    enforce: bool = True
    allow_audit_gated_sources: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DataPolicy:
        policy = config.get("data_policy", {})
        use_case = str(policy.get("use_case", "internal_noncommercial_research"))
        if use_case != "internal_noncommercial_research":
            raise DataPolicyError(f"Unsupported data policy use_case: {use_case}")
        return cls(
            use_case=use_case,
            enforce=bool(policy.get("enforce", True)),
            allow_audit_gated_sources=bool(policy.get("allow_audit_gated_sources", False)),
        )

    def exclusion_reasons(self, source: SourceSpec) -> tuple[str, ...]:
        reasons = []
        if source.purpose == "evaluation":
            reasons.append("purpose=evaluation")
        if not self.enforce:
            return tuple(reasons)
        audit_gated = self.allow_audit_gated_sources and source.license_status == "review_required"
        if source.license_status not in ALLOWED_LICENSE_STATUSES and not audit_gated:
            reasons.append(f"license_status={source.license_status}")
        if self.use_case not in source.allowed_uses:
            reasons.append(f"use_case={self.use_case} not allowed")
        if source.pii_status not in ALLOWED_REVIEW_STATUSES and not audit_gated:
            reasons.append(f"pii_status={source.pii_status}")
        if source.child_safety_status not in ALLOWED_REVIEW_STATUSES and not audit_gated:
            reasons.append(f"child_safety_status={source.child_safety_status}")
        if source.license_status == "restricted_research":
            if not source.evidence_path or not source.evidence_sha256:
                reasons.append("restricted source lacks checksummed evidence")
            else:
                evidence_path = Path(source.evidence_path)
                if not evidence_path.is_file() or _sha256_file(evidence_path) != source.evidence_sha256.lower():
                    reasons.append("restricted source evidence is missing or changed")
        return tuple(reasons)


@dataclass
class AuditReport:
    """Privacy/contamination counters without retaining matched raw text."""

    source_lock_digest: str
    benchmark_denylist_digest: str = EMPTY_BENCHMARK_DENYLIST_DIGEST
    policy_version: str = POLICY_VERSION
    filter_version: str = FILTER_VERSION
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return _json_digest(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["audit_digest"] = self.digest
        return payload


def source_policy_snapshot(source: dict[str, Any]) -> dict[str, Any]:
    return asdict(SourceSpec.from_mapping(source))


def partition_sources(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy = DataPolicy.from_config(config)
    allowed: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for source in config["data"].get("sources") or []:
        spec = SourceSpec.from_mapping(source)
        reasons = policy.exclusion_reasons(spec)
        if reasons:
            excluded.append({"name": spec.name, "reasons": list(reasons)})
        else:
            allowed.append(source)
    return allowed, excluded


def sensitive_categories(
    text: str, special_tokens: dict[str, str] | None = None, kind: str = "pretrain"
) -> tuple[str, ...]:
    """Return high-confidence categories; never return or log matched text."""

    categories = []
    for name, pattern in (
        ("email", _EMAIL),
        ("kr_phone", _KR_PHONE),
        ("jp_phone", _JP_PHONE),
        ("kr_resident_id", _KR_RESIDENT_ID),
        ("credential", _CREDENTIAL),
    ):
        if pattern.search(text):
            categories.append(name)
    if any(_luhn_valid(match.group(0)) for match in _CARD_CANDIDATE.finditer(text)):
        categories.append("payment_card")
    if kind == "pretrain" and special_tokens and any(token in text for token in special_tokens.values()):
        categories.append("special_token_injection")
    return tuple(categories)


def content_hash(text: str) -> str:
    normalized = " ".join(text.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_benchmark_denylist(config: dict[str, Any]) -> set[str]:
    """Load canonical benchmark component hashes and reject malformed registries."""

    required = bool(config.get("data_policy", {}).get("require_benchmark_denylist", False))
    value = config.get("data_policy", {}).get("benchmark_denylist_path")
    if not value:
        if required:
            raise DataPolicyError("A benchmark denylist is required, but benchmark_denylist_path is empty.")
        return set()
    path = Path(value)
    if not path.is_file():
        if required:
            raise DataPolicyError(
                f"Required benchmark denylist is missing: {path}. "
                "Create it from evaluation-only canonical components before training."
            )
        return set()
    hashes: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if not _SHA256_LINE.fullmatch(value):
            raise DataPolicyError(f"Invalid SHA-256 value in benchmark denylist {path}:{line_number}")
        hashes.add(value.lower())
    if required and not hashes:
        raise DataPolicyError(f"Required benchmark denylist contains no canonical component hashes: {path}")
    return hashes


def benchmark_denylist_digest(config: dict[str, Any]) -> str:
    """Return the stable digest embedded in audits and training fingerprints."""

    return _json_digest(sorted(load_benchmark_denylist(config)))


def _atomic_text_values(value: Any, *, depth: int = 0) -> list[str]:
    """Collect raw row components so chat or translation wrappers cannot hide a benchmark item."""

    if depth > 8:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        texts: list[str] = []
        for key, item in value.items():
            if not str(key).startswith("_"):
                texts.extend(_atomic_text_values(item, depth=depth + 1))
        return texts
    if isinstance(value, (list, tuple)):
        texts = []
        for item in value:
            texts.extend(_atomic_text_values(item, depth=depth + 1))
        return texts
    return []


def sample_rejection_categories(
    sample: Any, config: dict[str, Any], denylist: set[str] | None = None
) -> tuple[str, ...]:
    categories = list(
        sensitive_categories(
            str(sample.text),
            special_tokens=config.get("tokenizer", {}).get("special_tokens"),
            kind=str(getattr(sample, "kind", "pretrain")),
        )
    )
    benchmark_hashes = denylist if denylist is not None else load_benchmark_denylist(config)
    candidate_texts = [str(sample.text), *_atomic_text_values(getattr(sample, "meta", None))]
    if any(content_hash(text) in benchmark_hashes for text in candidate_texts if text.strip()):
        categories.append("benchmark_denylist")
    return tuple(categories)


def preference_rejection_categories(
    sample: Any, config: dict[str, Any], denylist: set[str] | None = None
) -> tuple[str, ...]:
    """Apply the shared training filter to every DPO text field."""

    benchmark_hashes = denylist if denylist is not None else load_benchmark_denylist(config)
    categories: set[str] = set()
    for text in (sample.prompt, sample.chosen, sample.rejected):
        categories.update(
            sensitive_categories(
                str(text),
                special_tokens=config.get("tokenizer", {}).get("special_tokens"),
                kind="sft",
            )
        )
        if content_hash(str(text)) in benchmark_hashes:
            categories.add("benchmark_denylist")
    for text in _atomic_text_values(getattr(sample, "meta", None)):
        if content_hash(text) in benchmark_hashes:
            categories.add("benchmark_denylist")
    return tuple(sorted(categories))


def build_source_lock(config: dict[str, Any], output_path: str | Path | None = None) -> dict[str, Any]:
    """Resolve and checksum every configured file with its policy snapshot."""

    from .data import expand_source_paths

    records = []
    checksum_cache: dict[tuple[str, int, int], str] = {}
    for source in config["data"].get("sources") or []:
        policy = source_policy_snapshot(source)
        for path in expand_source_paths(source):
            if not path.is_file():
                continue
            stat = path.stat()
            resolved = str(path.resolve())
            cache_key = (resolved, stat.st_size, stat.st_mtime_ns)
            checksum = checksum_cache.get(cache_key)
            if checksum is None:
                checksum = _sha256_file(path)
                checksum_cache[cache_key] = checksum
            records.append(
                {
                    "source": policy["name"],
                    "path": resolved,
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": checksum,
                    "policy": policy,
                }
            )
    records.sort(key=lambda item: (item["source"], item["path"]))
    purposes_by_checksum: dict[str, set[str]] = {}
    for record in records:
        purposes_by_checksum.setdefault(record["sha256"], set()).add(record["policy"]["purpose"])
    if any(purposes == {"training", "evaluation"} for purposes in purposes_by_checksum.values()):
        raise DataPolicyError("Training and evaluation sources contain a byte-identical file.")
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy_version": POLICY_VERSION,
        "files": records,
    }
    payload["lock_digest"] = _json_digest({key: value for key, value in payload.items() if key != "created_utc"})
    destination = Path(output_path or config["data_policy"]["source_lock_path"])
    atomic_write_json(destination, payload)
    return payload


def load_and_verify_source_lock(config: dict[str, Any], *, verify_hashes: bool = False) -> dict[str, Any]:
    from .data import expand_source_paths

    path = Path(config["data_policy"]["source_lock_path"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataPolicyError(f"Missing or invalid source lock: {path}. Run `hana data lock`.") from exc
    expected = _json_digest({key: value for key, value in payload.items() if key not in {"created_utc", "lock_digest"}})
    if payload.get("lock_digest") != expected or payload.get("policy_version") != POLICY_VERSION:
        raise DataPolicyError("Source lock digest or policy version is stale; rebuild it with `hana data lock`.")
    locked_records = payload.get("files", [])
    current_manifest = []
    for source in config["data"].get("sources") or []:
        policy = source_policy_snapshot(source)
        for source_path in expand_source_paths(source):
            if source_path.is_file():
                current_manifest.append(
                    {"source": policy["name"], "path": str(source_path.resolve()), "policy": policy}
                )
    locked_manifest = [
        {
            "source": str(record.get("source")),
            "path": str(Path(record["path"]).resolve()),
            "policy": record.get("policy"),
        }
        for record in locked_records
    ]

    def sort_key(item: dict[str, Any]) -> tuple[str, str]:
        return item["source"], item["path"]

    if _json_digest(sorted(current_manifest, key=sort_key)) != _json_digest(sorted(locked_manifest, key=sort_key)):
        raise DataPolicyError(
            "Source paths or policy snapshot changed after locking; run `hana data lock` and audit again."
        )

    for record in locked_records:
        file_path = Path(record["path"])
        if not file_path.is_file():
            raise DataPolicyError(f"Locked source is missing: {file_path}")
        stat = file_path.stat()
        if stat.st_size != int(record["bytes"]) or stat.st_mtime_ns != int(record["mtime_ns"]):
            raise DataPolicyError(f"Locked source size or modification time changed: {file_path}")
        if verify_hashes and _sha256_file(file_path) != record["sha256"]:
            raise DataPolicyError(f"Locked source checksum changed: {file_path}")
    return payload


def audit_allowed_sources(config: dict[str, Any], output_path: str | Path | None = None) -> AuditReport:
    """Scan allowed samples and write aggregate-only rejection evidence."""

    from .data import expand_source_paths, iter_text_samples

    lock = load_and_verify_source_lock(config)
    allowed, _excluded = partition_sources(config)
    if not allowed and DataPolicy.from_config(config).enforce:
        raise DataPolicyError("No approved source remains under the internal research policy.")
    denylist = load_benchmark_denylist(config)
    report = AuditReport(
        source_lock_digest=str(lock["lock_digest"]),
        benchmark_denylist_digest=_json_digest(sorted(denylist)),
    )
    max_hashes = int(config["data_policy"].get("max_rejection_hashes_per_source", 25))
    for source in allowed:
        name = str(source.get("name", "<unnamed>"))
        stats: dict[str, Any] = {"scanned": 0, "accepted": 0, "rejected": {}, "rejection_hashes": []}
        for path in expand_source_paths(source):
            path_source = {**source, "path": str(path), "max_samples": None}
            for sample in iter_text_samples(path, config, dataset_type=None, source=path_source):
                stats["scanned"] += 1
                categories = sample_rejection_categories(sample, config, denylist)
                if not categories:
                    stats["accepted"] += 1
                    continue
                for category in categories:
                    stats["rejected"][category] = int(stats["rejected"].get(category, 0)) + 1
                if len(stats["rejection_hashes"]) < max_hashes:
                    stats["rejection_hashes"].append(content_hash(sample.text))
        report.sources[name] = stats
    atomic_write_json(output_path or config["data_policy"]["audit_path"], report.to_dict())
    return report


def load_and_verify_data_audit(config: dict[str, Any], lock: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a complete audit and verify every input that gives it meaning."""

    source_lock = lock or load_and_verify_source_lock(config)
    audit_path = Path(config["data_policy"]["audit_path"])
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataPolicyError(f"Missing or invalid data audit: {audit_path}. Run `hana data audit`.") from exc
    expected_audit_digest = _json_digest({key: value for key, value in audit.items() if key != "audit_digest"})
    if audit.get("audit_digest") != expected_audit_digest:
        raise DataPolicyError("Data audit digest is missing or invalid; run `hana data audit` again.")
    expected_benchmark_digest = benchmark_denylist_digest(config)
    if (
        audit.get("source_lock_digest") != source_lock.get("lock_digest")
        or audit.get("policy_version") != POLICY_VERSION
        or audit.get("filter_version") != FILTER_VERSION
        or audit.get("benchmark_denylist_digest") != expected_benchmark_digest
    ):
        raise DataPolicyError("Data audit is stale relative to source, policy, filter, or benchmark evidence.")
    allowed, _excluded = partition_sources(config)
    expected_sources = {str(source.get("name", "<unnamed>")) for source in allowed}
    audited_sources = set(audit.get("sources", {})) if isinstance(audit.get("sources"), dict) else set()
    if audited_sources != expected_sources:
        raise DataPolicyError("Data audit does not cover exactly the currently allowed training sources.")
    return audit


def enforce_data_policy(config: dict[str, Any], *, require_artifacts: bool) -> list[dict[str, Any]]:
    """Replace runtime sources with allowed sources after freshness checks."""

    policy = DataPolicy.from_config(config)
    allowed, excluded = partition_sources(config)
    if not policy.enforce:
        config["data"]["sources"] = allowed
        return excluded
    if not allowed:
        raise DataPolicyError("No approved source remains under the internal research policy.")
    if require_artifacts:
        lock = load_and_verify_source_lock(config)
        load_and_verify_data_audit(config, lock)
    config["data"]["sources"] = allowed
    return excluded
