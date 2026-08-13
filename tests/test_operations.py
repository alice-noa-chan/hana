from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from llm_pipeline.config import PipelineConfig, load_config
from llm_pipeline.errors import PreflightError
from llm_pipeline.operations import (
    assert_runtime_preconditions,
    build_doctor_report,
    build_transfer_archive,
    write_run_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def local_runtime_config(tmp_path: Path) -> PipelineConfig:
    mutable = load_config(ROOT / "configs/smoke.yaml").mutable_copy()
    mutable["data_policy"]["enforce"] = False
    mutable["run"]["output_dir"] = str(tmp_path / "checkpoints")
    mutable["run"]["require_clean_git"] = False
    mutable["data"]["processed_dir"] = str(tmp_path / "processed")
    mutable["data"]["token_cache_dir"] = str(tmp_path / "token_cache")
    mutable["dpo"]["train_file"] = str(tmp_path / "processed/dpo_rejected.jsonl")
    mutable["experiments"]["output_dir"] = str(tmp_path / "experiments")
    mutable["export"]["export_non_it_dir"] = str(tmp_path / "exports/non_it")
    mutable["export"]["export_it_dir"] = str(tmp_path / "exports/it")
    mutable["export"]["export_quantized_dir"] = str(tmp_path / "exports/quantized")
    mutable["logging"]["log_dir"] = str(tmp_path / "logs")
    mutable["cognitive_architecture"]["memory"]["path"] = str(tmp_path / "memory.json")
    mutable["inference"]["token_trace_file"] = str(tmp_path / "token_trace.jsonl")
    return PipelineConfig.from_dict(mutable)


def test_doctor_reports_model_runtime_and_disk_without_cuda(tmp_path: Path, monkeypatch) -> None:
    config = local_runtime_config(tmp_path)
    expected_vocab_size = int(config["tokenizer"]["vocab_size"])
    monkeypatch.setattr(
        "llm_pipeline.tokenizer.load_tokenizer",
        lambda _config: SimpleNamespace(vocab_size=expected_vocab_size),
    )
    monkeypatch.setattr("llm_pipeline.model.build_model", lambda _config: torch.nn.Linear(4, 4))

    report = build_doctor_report(config, allow_cpu=True)

    assert report["status"] == "ok"
    assert report["model"]["parameters"] == 20
    assert report["model"]["tokenizer_vocab_size"] == expected_vocab_size
    assert report["runtime"]["free_disk_gib"] > 0


def test_runtime_preconditions_block_cloud_synced_write_paths(tmp_path: Path, monkeypatch) -> None:
    cloud = tmp_path / "OneDrive"
    cloud.mkdir()
    monkeypatch.setenv("ONEDRIVE", str(cloud))
    mutable = local_runtime_config(tmp_path).mutable_copy()
    mutable["run"]["output_dir"] = str(cloud / "checkpoints")

    with pytest.raises(PreflightError, match="High-write path"):
        assert_runtime_preconditions(PipelineConfig.from_dict(mutable))


def test_run_manifest_records_config_git_environment_and_backends(tmp_path: Path) -> None:
    config = local_runtime_config(tmp_path)

    path = write_run_manifest(config, tmp_path / "logs")

    payload = path.read_text(encoding="utf-8")
    assert config.digest in payload
    assert '"attention"' in payload
    assert '"packages"' in payload


def test_transfer_archive_contains_verified_worktree_git_and_setup(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "payload.txt"
    source.write_text("approved payload", encoding="utf-8")
    git_head = tmp_path / ".git" / "HEAD"
    git_head.parent.mkdir()
    git_head.write_text("ref: refs/heads/main\n", encoding="utf-8")
    config = PipelineConfig.from_dict({"__base_dir__": str(tmp_path), "__config_path__": str(tmp_path / "config.yaml")})

    def fake_manifest(_config, output_dir):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "GPU_TRANSFER_MANIFEST.json").write_text("{}", encoding="utf-8")
        (output / "GPU_TRANSFER_SHA256SUMS.txt").write_text("", encoding="utf-8")
        (output / "GPU_TRANSFER_FILES.txt").write_text("payload.txt\n", encoding="utf-8")
        return {
            "files": [
                {
                    "path": "payload.txt",
                    "bytes": source.stat().st_size,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            ],
            "source_lock_digest": "lock",
            "audit_digest": "audit",
        }

    monkeypatch.setattr("llm_pipeline.operations.build_transfer_manifest", fake_manifest)
    result = build_transfer_archive(config, tmp_path / "transfer", "ready.zip")

    assert Path(result["archive"]).is_file()
    assert Path(result["sha256_file"]).read_text(encoding="utf-8").startswith(result["sha256"])
    with zipfile.ZipFile(result["archive"]) as archive:
        assert archive.read("hana/payload.txt") == b"approved payload"
        assert "hana/.git/HEAD" in archive.namelist()
        assert "hana/gpu_transfer/GPU_SERVER_SETUP.md" in archive.namelist()
        setup = archive.read("hana/gpu_transfer/GPU_SERVER_SETUP.md").decode("utf-8")
        assert "hana run --config config.yaml --mode auto --continue" in setup
        assert "full-precision" in setup
        assert "DPO" in setup


def test_transfer_archive_rejects_nested_or_non_zip_names(tmp_path: Path) -> None:
    config = PipelineConfig.from_dict({"__base_dir__": str(tmp_path), "__config_path__": str(tmp_path / "config.yaml")})
    with pytest.raises(ValueError, match=r"plain \.zip"):
        build_transfer_archive(config, tmp_path, "nested/ready.zip")
    with pytest.raises(ValueError, match=r"plain \.zip"):
        build_transfer_archive(config, tmp_path, "ready.tar")
