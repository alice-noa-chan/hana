from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.check_public_tree import find_history_violations, find_publication_violations
from scripts.prepare_synthetic_smoke import write_smoke_fixture

ROOT = Path(__file__).resolve().parents[1]


def test_repository_contains_no_tracked_dataset_or_private_profile_assets() -> None:
    assert find_publication_violations(ROOT) == []


def test_publication_check_rejects_common_dataset_artifacts(tmp_path: Path) -> None:
    private_row = tmp_path / "private_data" / "sample.jsonl"
    private_row.parent.mkdir()
    private_row.write_text('{"text":"must stay private"}\n', encoding="utf-8")
    model = tmp_path / "weights.safetensors"
    model.write_bytes(b"synthetic")
    (tmp_path / "config.yaml").write_text(
        "data:\n  sources: []\n  pack:\n    languages: []\ninference: {}\ndpo: {}\n",
        encoding="utf-8",
    )

    violations = find_publication_violations(tmp_path, ["private_data/sample.jsonl", "weights.safetensors"])

    assert any("private or dataset directory" in violation for violation in violations)
    assert any("dataset or model artifact" in violation for violation in violations)


def test_publication_check_rejects_missing_tracked_path_and_forced_local_config(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "data:\n  sources: []\n  pack:\n    languages: []\ninference: {}\ndpo: {}\n",
        encoding="utf-8",
    )

    violations = find_publication_violations(
        tmp_path,
        ["nested/data/deleted.jsonl", "config.local.yaml", "artifacts/private-run.zip"],
    )

    assert any("nested/data/deleted.jsonl" in violation for violation in violations)
    assert any("config.local.yaml" in violation for violation in violations)
    assert any("private-run.zip" in violation for violation in violations)


def test_publication_check_rejects_embedded_private_evidence(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "data:\n  sources: []\n  pack:\n    languages: []\ninference: {}\ndpo: {}\n",
        encoding="utf-8",
    )
    public_yaml = tmp_path / "example.yaml"
    public_yaml.write_text(
        f'source_url: "https://private-source.invalid/corpus"\nevidence_sha256: "{"a" * 64}"\n',
        encoding="utf-8",
    )

    violations = find_publication_violations(tmp_path, ["example.yaml"])

    assert any("non-placeholder source_url" in violation for violation in violations)
    assert any("literal SHA-256" in violation for violation in violations)


def test_publication_check_rejects_private_pilot_configuration(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "data:\n  sources: []\n  pack:\n    languages: []\n"
        "inference: {}\ndpo: {}\n"
        "eval:\n  knowledge_pilot:\n    enabled: true\n    file: private-pilot.jsonl\n",
        encoding="utf-8",
    )

    violations = find_publication_violations(tmp_path, [])

    assert any("private knowledge pilot" in violation for violation in violations)


def test_publication_check_rejects_private_pilot_in_alternate_public_config(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "data:\n  sources: []\ninference: {}\ndpo: {}\n",
        encoding="utf-8",
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "alternate.yaml").write_text(
        "data:\n  sources: []\ninference: {}\ndpo: {}\n"
        "eval:\n  knowledge_pilot:\n    enabled: true\n    file: private-pilot.jsonl\n",
        encoding="utf-8",
    )

    violations = find_publication_violations(tmp_path, [])

    assert any("configs/alternate.yaml" in violation for violation in violations)


def test_publication_check_rejects_dataset_deleted_from_current_tree(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Synthetic Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "synthetic@example.invalid"], cwd=tmp_path, check=True)
    private_row = tmp_path / "private_data" / "sample.jsonl"
    private_row.parent.mkdir()
    private_row.write_text('{"text":"synthetic"}\n', encoding="utf-8")
    subprocess.run(["git", "add", "private_data/sample.jsonl"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "synthetic fixture"], cwd=tmp_path, check=True)
    private_row.unlink()
    subprocess.run(["git", "add", "-u"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "remove synthetic fixture"], cwd=tmp_path, check=True)

    violations = find_history_violations(tmp_path)

    assert any("private_data/sample.jsonl" in violation for violation in violations)


def test_publication_check_scans_nonignored_untracked_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "config.yaml").write_text(
        "data:\n  sources: []\n  pack:\n    languages: []\ninference: {}\ndpo: {}\n",
        encoding="utf-8",
    )
    (tmp_path / "private_data").mkdir()
    (tmp_path / "private_data" / "untracked.jsonl").write_text('{"text":"synthetic"}\n', encoding="utf-8")

    violations = find_publication_violations(tmp_path)

    assert any("private_data/untracked.jsonl" in violation for violation in violations)


def test_synthetic_smoke_writer_is_safe_for_concurrent_test_sessions(tmp_path: Path) -> None:
    output = tmp_path / "synthetic.jsonl"

    with ThreadPoolExecutor(max_workers=4) as executor:
        written = list(executor.map(lambda _: write_smoke_fixture(output, 32), range(8)))

    assert written == [output] * 8
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 32
    assert all(set(record) == {"text"} for record in records)
