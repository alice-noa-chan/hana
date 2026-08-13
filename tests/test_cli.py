from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_pipeline import cli
from llm_pipeline.errors import DataPolicyError, PreflightError

ROOT = Path(__file__).resolve().parents[1]


def test_run_command_forwards_only_runtime_overrides(monkeypatch) -> None:
    received = {}

    def fake_run(**kwargs) -> None:
        received.update(kwargs)

    monkeypatch.setattr("llm_pipeline.main.main", fake_run)
    result = cli.main(["run", "--config", "custom.yaml", "--mode", "pretrain", "--continue", "--force"])

    assert result == 0
    assert received == {
        "config_path": "custom.yaml",
        "run_mode": "pretrain",
        "auto_continue": True,
        "force_run": True,
    }


def test_expected_error_categories_have_stable_exit_codes(monkeypatch, capsys) -> None:
    def fail(**_kwargs) -> None:
        raise FileNotFoundError("missing config")

    monkeypatch.setattr("llm_pipeline.main.main", fail)

    assert cli.main(["run"]) == cli.EXIT_CONFIG
    assert "missing config" in capsys.readouterr().err


def test_data_commands_are_part_of_the_public_parser() -> None:
    parser = cli.build_parser()

    lock = parser.parse_args(["data", "lock", "--config", "x.yaml"])
    audit = parser.parse_args(["data", "audit"])
    quarantine = parser.parse_args(["data", "quarantine-eval", "--config", "private.yaml"])

    assert (lock.command, lock.data_command, lock.config) == ("data", "lock", "x.yaml")
    assert (audit.command, audit.data_command) == ("data", "audit")
    assert (quarantine.data_command, quarantine.config) == ("quarantine-eval", "private.yaml")


def test_experiment_registry_validation_is_public_and_runnable(capsys) -> None:
    registry = ROOT / "configs/experiments/registry.example.yaml"

    assert cli.main(["experiment", "validate", "--registry", str(registry)]) == 0
    output = capsys.readouterr().out
    assert "studies=1" in output
    assert "arms=2" in output
    assert "config_diffs_checked=0" in output


def test_experiment_validation_can_check_declared_config_differences(monkeypatch, capsys) -> None:
    registry = ROOT / "configs/experiments/registry.example.yaml"
    loaded_paths = []
    checked = []

    class Loaded:
        def __init__(self, path: Path) -> None:
            self.path = path

        def mutable_copy(self):
            return {"source_config": self.path.name}

    def fake_load(path: Path) -> Loaded:
        resolved = Path(path)
        loaded_paths.append(resolved)
        return Loaded(resolved)

    def fake_verify(study, arm_id, baseline, candidate, *, ignored_keys) -> None:
        checked.append((study.id, arm_id, baseline, candidate, list(ignored_keys)))

    monkeypatch.setattr("llm_pipeline.config.load_config", fake_load)
    monkeypatch.setattr("llm_pipeline.experiment_contract.verify_arm_config_diff", fake_verify)

    result = cli.main(
        [
            "experiment",
            "validate",
            "--registry",
            str(registry),
            "--check-configs",
            "--ignore-key",
            "run.output_dir",
        ]
    )

    assert result == 0
    assert [path.name for path in loaded_paths] == ["workspace_control.yaml", "workspace_on.yaml"]
    assert checked == [
        (
            "workspace_v1",
            "workspace_on",
            {"source_config": "workspace_control.yaml"},
            {"source_config": "workspace_on.yaml"},
            ["run.output_dir"],
        )
    ]
    assert "config_diffs_checked=1" in capsys.readouterr().out


def test_data_audit_dispatches_the_auditor(monkeypatch, capsys) -> None:
    class Loaded:
        def mutable_copy(self):
            return {"data_policy": {"audit_path": "audit.json"}}

    called = []
    monkeypatch.setattr("llm_pipeline.config.load_config", lambda _path: Loaded())
    monkeypatch.setattr(
        "llm_pipeline.data_governance.audit_allowed_sources",
        lambda config: (
            called.append(config)
            or SimpleNamespace(
                sources={"approved": {"scanned": 3, "rejected": {"pii_email": 1}}},
                digest="audit-digest",
            )
        ),
    )

    assert cli.main(["data", "audit", "--config", "custom.yaml"]) == 0
    assert len(called) == 1
    assert "Audited 3 samples" in capsys.readouterr().out


def test_data_quarantine_dispatches_without_printing_items(monkeypatch, capsys) -> None:
    class Loaded:
        def mutable_copy(self):
            return {"eval": {"knowledge_pilot": {}}, "data_policy": {}}

    monkeypatch.setattr("llm_pipeline.config.load_config", lambda _path: Loaded())
    monkeypatch.setattr(
        "llm_pipeline.multiple_choice.quarantine_knowledge_pilot",
        lambda _config: {
            "item_count": 10,
            "added_hashes": 20,
            "total_hashes": 20,
            "path": "private-denylist.txt",
        },
    )

    assert cli.main(["data", "quarantine-eval"]) == 0
    output = capsys.readouterr().out
    assert "Quarantined 10 private evaluation items" in output
    assert "question" not in output


def test_doctor_and_transfer_commands_are_part_of_the_public_parser() -> None:
    parser = cli.build_parser()

    doctor = parser.parse_args(["doctor", "--allow-cpu", "--verify-hashes"])
    transfer = parser.parse_args(["transfer", "manifest", "--output", "bundle"])
    bundle = parser.parse_args(["transfer", "bundle", "--name", "ready.zip"])

    assert doctor.allow_cpu and doctor.verify_hashes
    assert (transfer.command, transfer.transfer_command, transfer.output) == ("transfer", "manifest", "bundle")
    assert (bundle.transfer_command, bundle.name) == ("bundle", "ready.zip")


def test_verify_dispatches_quality_gates(monkeypatch) -> None:
    called = []
    monkeypatch.setattr("llm_pipeline.verification.run_quality_gates", lambda: called.append(True))

    assert cli.main(["verify"]) == 0
    assert called == [True]


def test_data_lock_dispatches_and_prints_digest(monkeypatch, capsys) -> None:
    class Loaded:
        def mutable_copy(self):
            return {"data_policy": {"source_lock_path": "lock.json"}}

    monkeypatch.setattr("llm_pipeline.config.load_config", lambda _path: Loaded())
    monkeypatch.setattr(
        "llm_pipeline.data_governance.build_source_lock",
        lambda _config: {"files": [{"path": "one"}], "lock_digest": "locked"},
    )

    assert cli.main(["data", "lock"]) == 0
    assert "Locked 1 files; digest=locked" in capsys.readouterr().out


def test_doctor_and_transfer_dispatch(monkeypatch, capsys) -> None:
    loaded = object()
    doctor_calls = []
    transfer_calls = []
    monkeypatch.setattr("llm_pipeline.config.load_config", lambda _path: loaded)
    monkeypatch.setattr(
        "llm_pipeline.operations.run_doctor",
        lambda config, **kwargs: doctor_calls.append((config, kwargs)),
    )
    monkeypatch.setattr(
        "llm_pipeline.operations.build_transfer_manifest",
        lambda config, output: transfer_calls.append((config, output)) or {"file_count": 2, "total_bytes": 1024**3},
    )

    assert cli.main(["doctor", "--allow-cpu", "--verify-hashes"]) == 0
    assert cli.main(["transfer", "manifest", "--output", "bundle"]) == 0
    assert doctor_calls == [(loaded, {"allow_cpu": True, "verify_hashes": True})]
    assert transfer_calls == [(loaded, "bundle")]
    assert "Transfer manifest ready: 2 files, 1.00 GiB" in capsys.readouterr().out


def test_transfer_bundle_dispatch(monkeypatch, capsys) -> None:
    loaded = object()
    calls = []
    monkeypatch.setattr("llm_pipeline.config.load_config", lambda _path: loaded)
    monkeypatch.setattr(
        "llm_pipeline.operations.build_transfer_archive",
        lambda config, output, name: (
            calls.append((config, output, name))
            or {
                "archive": "gpu_transfer/ready.zip",
                "bytes": 2 * 1024**2,
                "sha256": "abc",
                "sha256_file": "gpu_transfer/ready.zip.sha256",
            }
        ),
    )

    assert cli.main(["transfer", "bundle", "--output", "upload", "--name", "ready.zip"]) == 0
    assert calls == [(loaded, "upload", "ready.zip")]
    assert "Upload-ready archive" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("error", "exit_code", "prefix"),
    [
        (DataPolicyError("denied"), cli.EXIT_DATA_POLICY, "data policy error"),
        (PreflightError("no gpu"), cli.EXIT_PREFLIGHT, "preflight error"),
        (RuntimeError("boom"), cli.EXIT_RUNTIME, "runtime error"),
    ],
)
def test_error_categories_map_to_documented_exit_codes(monkeypatch, capsys, error, exit_code, prefix) -> None:
    def fail() -> None:
        raise error

    monkeypatch.setattr("llm_pipeline.verification.run_quality_gates", fail)

    assert cli.main(["verify"]) == exit_code
    assert prefix in capsys.readouterr().err


def test_entrypoint_exits_with_main_result(monkeypatch) -> None:
    monkeypatch.setattr(cli, "main", lambda: cli.EXIT_CONFIG)
    with pytest.raises(SystemExit) as raised:
        cli.entrypoint()
    assert raised.value.code == cli.EXIT_CONFIG
