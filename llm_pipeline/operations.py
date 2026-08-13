"""Environment diagnostics, reproducibility manifests, and GPU transfer plans."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, atomic_write_text
from .config import PipelineConfig
from .data_governance import (
    DataPolicy,
    load_and_verify_data_audit,
    load_and_verify_source_lock,
    partition_sources,
)
from .errors import DataPolicyError, PreflightError
from .logging_utils import get_git_commit, hardware_summary

DIRECT_DEPENDENCIES = ("numpy", "psutil", "PyYAML", "safetensors", "sentencepiece", "tensorboard", "torch", "tqdm")
HIGH_WRITE_PATHS = (
    ("run", "output_dir"),
    ("data", "processed_dir"),
    ("data", "token_cache_dir"),
    ("dpo", "train_file"),
    ("experiments", "output_dir"),
    ("export", "export_non_it_dir"),
    ("export", "export_it_dir"),
    ("export", "export_quantized_dir"),
    ("logging", "log_dir"),
)


def _git_status(project_root: Path) -> tuple[str, bool]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return get_git_commit(), bool(result.stdout.strip()) if result.returncode == 0 else True


def _inside_cloud_sync(path: Path) -> bool:
    resolved = str(path.resolve()).casefold()
    roots = [os.environ.get("ONEDRIVE"), os.environ.get("ONEDRIVECONSUMER"), os.environ.get("ONEDRIVECOMMERCIAL")]
    normalized_roots = [str(Path(value).resolve()).casefold() for value in roots if value]
    return "\\onedrive\\" in resolved or any(
        resolved == root or resolved.startswith(root + os.sep) for root in normalized_roots
    )


def _nearest_existing(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _audit_payload(config: dict[str, Any]) -> dict[str, Any]:
    return load_and_verify_data_audit(config)


def build_doctor_report(
    config: PipelineConfig, *, allow_cpu: bool = False, verify_hashes: bool = False
) -> dict[str, Any]:
    """Collect actionable readiness errors and warnings without changing state."""

    import torch

    mutable = config.mutable_copy()
    errors: list[str] = []
    warnings: list[str] = []
    allowed, excluded = partition_sources(mutable)
    policy = DataPolicy.from_config(mutable)
    if policy.enforce and not allowed:
        errors.append("No approved source remains after data policy filtering.")
    audit_gated = [
        str(source.get("name", "<unnamed>"))
        for source in allowed
        if str((source.get("provenance") or {}).get("license_status", source.get("license_status", ""))).lower()
        == "review_required"
    ]
    if audit_gated:
        warnings.append(
            f"{len(audit_gated)} review-required source(s) are enabled only for audit-gated internal research."
        )
    try:
        if policy.enforce:
            lock = load_and_verify_source_lock(mutable, verify_hashes=verify_hashes)
            audit = _audit_payload(mutable)
            if audit.get("source_lock_digest") != lock.get("lock_digest"):
                errors.append("Data audit is stale relative to sources.lock.json.")
        else:
            lock = {}
            audit = {}
    except DataPolicyError as exc:
        lock = {}
        audit = {}
        errors.append(str(exc))

    source_paths = []
    from .data import expand_source_paths

    for source in allowed:
        source_paths.extend(path for path in expand_source_paths(source) if path.is_file())
    source_paths = sorted(set(source_paths))
    source_bytes = sum(path.stat().st_size for path in source_paths)
    if any(_inside_cloud_sync(path) for path in source_paths):
        warnings.append("Read-only source corpora are inside a cloud-synchronized directory.")

    high_write = []
    for section, key in HIGH_WRITE_PATHS:
        value = mutable[section].get(key)
        if not value:
            continue
        path = Path(value)
        high_write.append(str(path))
        if _inside_cloud_sync(path):
            errors.append(f"High-write path must be outside cloud sync: {section}.{key}={path}")
    memory_path = Path(mutable["cognitive_architecture"]["memory"]["path"])
    trace_path = mutable["inference"].get("token_trace_file")
    for label, path in (("cognitive memory", memory_path), ("token trace", Path(trace_path) if trace_path else None)):
        if path is not None and _inside_cloud_sync(path):
            errors.append(f"High-write {label} path must be outside cloud sync: {path}")

    output_root = Path(mutable["run"]["output_dir"])
    free_bytes = shutil.disk_usage(_nearest_existing(output_root)).free
    if free_bytes < 20 * 1024**3:
        errors.append(f"Less than 20 GiB free at runtime output root: {free_bytes / 1024**3:.2f} GiB")

    from .model import build_model
    from .tokenizer import load_tokenizer

    try:
        tokenizer = load_tokenizer(mutable)
        tokenizer_vocab = int(tokenizer.vocab_size)
        if tokenizer_vocab != int(mutable["model"]["vocab_size"]):
            errors.append(f"Tokenizer/model vocabulary mismatch: {tokenizer_vocab} != {mutable['model']['vocab_size']}")
    except Exception as exc:
        tokenizer_vocab = None
        errors.append(f"Tokenizer is unavailable or invalid: {exc}")

    with torch.device("meta"):
        parameter_count = sum(parameter.numel() for parameter in build_model(mutable).parameters())
    estimated_training_gib = parameter_count * 16 / 1024**3
    estimated_source_tokens = int(source_bytes / 3.5)
    tokens_per_parameter = estimated_source_tokens / max(1, parameter_count)
    estimated_token_cache_gib = estimated_source_tokens * 8 * 1.25 / 1024**3
    retained_checkpoints = (int(mutable["train"].get("top_k_checkpoints", 3)) + 2) * 3
    estimated_checkpoint_gib = estimated_training_gib * retained_checkpoints
    recommended_free_gib = estimated_token_cache_gib + estimated_checkpoint_gib + 20
    if free_bytes / 1024**3 < recommended_free_gib:
        errors.append(
            "Runtime disk is too small for token caches, retained pretrain/SFT/DPO checkpoints, and exports: "
            f"{free_bytes / 1024**3:.2f} GiB free; approximately {recommended_free_gib:.2f} GiB recommended."
        )

    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if cuda_count == 0 and not allow_cpu:
        errors.append("CUDA is unavailable; pass --allow-cpu only for local validation.")
    gpus = []
    for index in range(cuda_count):
        properties = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": properties.name,
                "vram_gib": round(properties.total_memory / 1024**3, 2),
                "bf16": bool(torch.cuda.is_bf16_supported()),
            }
        )

    git_commit, git_dirty = _git_status(config.base_dir)
    if mutable["run"].get("require_clean_git", False) and git_dirty:
        errors.append("Tracked files are dirty; commit or stash them before a production run.")

    return {
        "status": "error" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
        "config_digest": config.digest,
        "git": {"commit": git_commit, "dirty": git_dirty},
        "policy": {
            "allowed_sources": [str(source.get("name", "<unnamed>")) for source in allowed],
            "audit_gated_sources": audit_gated,
            "excluded_sources": excluded,
            "source_lock_digest": lock.get("lock_digest"),
            "audit_digest": audit.get("audit_digest"),
        },
        "data": {
            "files": len(source_paths),
            "gib": round(source_bytes / 1024**3, 3),
            "estimated_unique_tokens": estimated_source_tokens,
            "estimated_tokens_per_parameter": round(tokens_per_parameter, 3),
            "estimated_token_cache_gib": round(estimated_token_cache_gib, 2),
        },
        "model": {
            "parameters": parameter_count,
            "tokenizer_vocab_size": tokenizer_vocab,
            "estimated_training_state_gib": round(estimated_training_gib, 2),
            "max_seq_len": int(mutable["model"]["max_seq_len"]),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": bool(cuda_count),
            "gpus": gpus,
            "free_disk_gib": round(free_bytes / 1024**3, 2),
            "recommended_free_disk_gib": round(recommended_free_gib, 2),
            "high_write_paths": high_write,
        },
    }


def run_doctor(config: PipelineConfig, *, allow_cpu: bool = False, verify_hashes: bool = False) -> dict[str, Any]:
    report = build_doctor_report(config, allow_cpu=allow_cpu, verify_hashes=verify_hashes)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise PreflightError("; ".join(report["errors"]))
    return report


def assert_runtime_preconditions(config: PipelineConfig) -> None:
    """Enforce checks that must never depend on whether doctor was run manually."""

    mutable = config.mutable_copy()
    for section, key in HIGH_WRITE_PATHS:
        value = mutable[section].get(key)
        if value and _inside_cloud_sync(Path(value)):
            raise PreflightError(f"High-write path must be outside cloud sync: {section}.{key}={value}")
    for value in (
        mutable["cognitive_architecture"]["memory"].get("path"),
        mutable["inference"].get("token_trace_file"),
    ):
        if value and _inside_cloud_sync(Path(value)):
            raise PreflightError(f"High-write path must be outside cloud sync: {value}")
    _commit, dirty = _git_status(config.base_dir)
    if mutable["run"].get("require_clean_git", False) and dirty:
        raise PreflightError("Tracked files are dirty; commit or stash them before a production run.")


def write_run_manifest(config: PipelineConfig, log_dir: Path) -> Path:
    """Persist exact code/config/data/environment provenance for one run."""

    mutable = config.mutable_copy()
    git_commit, git_dirty = _git_status(config.base_dir)
    packages = {}
    for name in DIRECT_DEPENDENCIES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    lock_digest = None
    audit_digest = None
    try:
        lock_digest = load_and_verify_source_lock(mutable).get("lock_digest")
        audit_digest = _audit_payload(mutable).get("audit_digest")
    except DataPolicyError:
        if mutable["data_policy"].get("enforce", True):
            raise

    try:
        import torch

        torch_runtime = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        }
        precision = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp32"
    except ImportError:
        torch_runtime = {"version": None, "cuda": None, "cudnn": None}
        precision = "unavailable"

    from .model import select_attention_backend

    attention_backend, fallback_reasons = select_attention_backend(str(mutable["model"]["attention_backend"]))
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "git": {"commit": git_commit, "dirty": git_dirty},
        "config": {"path": str(config.path), "digest": config.digest},
        "data": {"source_lock_digest": lock_digest, "audit_digest": audit_digest},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": packages,
            "torch": torch_runtime,
            "hardware": hardware_summary(),
        },
        "backends": {
            "attention": attention_backend,
            "attention_fallback_reasons": fallback_reasons,
            "precision": precision,
            "quantization": mutable["quantization"]["method"] if mutable["quantization"]["enabled"] else "none",
        },
    }
    path = log_dir / "run_manifest.json"
    atomic_write_json(path, payload)
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_transfer_manifest(config: PipelineConfig, output_dir: str | Path) -> dict[str, Any]:
    """Create a checksummed transfer plan containing only approved locked data."""

    mutable = config.mutable_copy()
    lock = load_and_verify_source_lock(mutable)
    audit = _audit_payload(mutable)
    if audit.get("source_lock_digest") != lock.get("lock_digest"):
        raise DataPolicyError("Cannot transfer with a stale data audit.")
    allowed, _excluded = partition_sources(mutable)
    allowed_names = {str(source.get("name", "<unnamed>")) for source in allowed}
    root = config.base_dir.resolve()
    files: dict[Path, str | None] = {}

    tracked = (
        subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
        .stdout.decode("utf-8")
        .split("\0")
    )
    for relative in tracked:
        if relative:
            path = root / relative
            if path.is_file():
                files[path.resolve()] = None
    for key in ("sources_file",):
        path = Path(mutable["data"][key])
        if path.is_file():
            files[path.resolve()] = None
    for key in ("source_lock_path", "audit_path"):
        path = Path(mutable["data_policy"][key])
        if path.is_file():
            files[path.resolve()] = None
    for record in lock.get("files", []):
        if record.get("source") in allowed_names:
            files[Path(record["path"]).resolve()] = str(record["sha256"])
            evidence = record.get("policy", {}).get("evidence_path")
            if evidence and Path(evidence).is_file():
                files[Path(evidence).resolve()] = None
    tokenizer_dir = Path(mutable["tokenizer"]["save_dir"])
    for name in (
        "tokenizer.model",
        "tokenizer.vocab",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "corpus_manifest.json",
    ):
        path = tokenizer_dir / name
        if path.is_file():
            files[path.resolve()] = None

    records = []
    for path, known_hash in sorted(files.items(), key=lambda item: str(item[0])):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise DataPolicyError(f"Transfer file is outside the project root: {path}") from exc
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": known_hash or _sha256_file(path)})

    payload = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_digest": config.digest,
        "source_lock_digest": lock["lock_digest"],
        "audit_digest": audit.get("audit_digest"),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "training_command_single_gpu": "hana run --config config.yaml --mode auto --continue",
        "training_command_multi_gpu": (
            "torchrun --standalone --nproc_per_node=<GPU_COUNT> -m llm_pipeline "
            "run --config config.yaml --mode pretrain"
        ),
        "files": records,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "GPU_TRANSFER_MANIFEST.json", payload)
    atomic_write_text(
        output / "GPU_TRANSFER_SHA256SUMS.txt",
        "".join(f"{record['sha256']} *{record['path']}\n" for record in records),
    )
    atomic_write_text(output / "GPU_TRANSFER_FILES.txt", "".join(f"{record['path']}\n" for record in records))
    return payload


def _zip_member_sha256(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member, "r") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_transfer_archive(
    config: PipelineConfig,
    output_dir: str | Path,
    archive_name: str,
) -> dict[str, Any]:
    """Create and verify one upload-ready ZIP with Git history and policy-allowed data."""

    if Path(archive_name).name != archive_name or Path(archive_name).suffix.lower() != ".zip":
        raise ValueError("Transfer archive name must be a plain .zip filename.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_transfer_manifest(config, output)
    root = config.base_dir.resolve()
    setup_text = """# Hana GPU server setup

This archive contains the exact Git worktree, Git metadata, policy-allowed locked data,
the finalized tokenizer, and the workstation audit evidence. The included source
lock is intentionally rebuilt on Linux because lock paths are host-specific.

```bash
unzip hana_gpu_ready_*.zip
cd hana
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
sha256sum --check gpu_transfer/GPU_TRANSFER_SHA256SUMS.txt
git status --short
hana verify
hana data lock --config config.yaml
hana data audit --config config.yaml
hana doctor --config config.yaml --verify-hashes
```

The last command must report CUDA availability and `status: ok`. Start training
only after that result. Example:

```bash
hana run --config config.yaml --mode auto --continue
```

That single-GPU command runs tokenizer refresh, analysis, full-corpus pretraining,
SFT, rejected-response generation, DPO, evaluation, inference, full-precision
export, and int8 export in order.
"""
    setup_path = output / "GPU_SERVER_SETUP.md"
    atomic_write_text(setup_path, setup_text)
    transfer_metadata = [
        output / "GPU_TRANSFER_MANIFEST.json",
        output / "GPU_TRANSFER_SHA256SUMS.txt",
        output / "GPU_TRANSFER_FILES.txt",
        setup_path,
    ]
    archive_path = output / archive_name
    temporary_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    added: set[str] = set()

    def add_file(archive: zipfile.ZipFile, source: Path, member: str) -> None:
        normalized = member.replace("\\", "/")
        if normalized in added:
            return
        archive.write(source, normalized)
        added.add(normalized)

    try:
        with zipfile.ZipFile(
            temporary_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for record in manifest["files"]:
                relative = str(record["path"])
                add_file(archive, root / relative, f"hana/{relative}")
            git_root = root / ".git"
            if not git_root.is_dir():
                raise DataPolicyError("A Git worktree is required to create a reproducible GPU archive.")
            for path in sorted(git_root.rglob("*")):
                if path.is_file() and path.name != "index.lock":
                    add_file(archive, path, f"hana/{path.relative_to(root).as_posix()}")
            for path in transfer_metadata:
                add_file(archive, path, f"hana/gpu_transfer/{path.name}")
        temporary_path.replace(archive_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    with zipfile.ZipFile(archive_path, "r") as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise DataPolicyError(f"Transfer archive CRC validation failed: {corrupt}")
        names = set(archive.namelist())
        for record in manifest["files"]:
            member = f"hana/{record['path']}"
            if member not in names or _zip_member_sha256(archive, member) != record["sha256"]:
                raise DataPolicyError(f"Transfer archive content verification failed: {record['path']}")
        if "hana/.git/HEAD" not in names or "hana/gpu_transfer/GPU_SERVER_SETUP.md" not in names:
            raise DataPolicyError("Transfer archive is missing Git metadata or GPU setup instructions.")

    archive_digest = _sha256_file(archive_path)
    checksum_path = output / f"{archive_name}.sha256"
    atomic_write_text(checksum_path, f"{archive_digest} *{archive_name}\n")
    return {
        "archive": str(archive_path),
        "sha256_file": str(checksum_path),
        "sha256": archive_digest,
        "bytes": archive_path.stat().st_size,
        "files": len(added),
        "source_lock_digest": manifest["source_lock_digest"],
        "audit_digest": manifest["audit_digest"],
    }
