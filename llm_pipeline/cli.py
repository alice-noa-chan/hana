"""Command-line interface for the reusable decoder-only LLM research pipeline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .errors import ConfigError, DataPolicyError, PreflightError

EXIT_RUNTIME = 1
EXIT_CONFIG = 2
EXIT_DATA_POLICY = 3
EXIT_PREFLIGHT = 4


def build_parser() -> argparse.ArgumentParser:
    """Create the public command parser without importing Torch-heavy modules."""

    parser = argparse.ArgumentParser(prog="hana", description="Reusable decoder-only LLM research pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one or more pipeline stages")
    run_parser.add_argument("--config", default="config.yaml", help="Path to the schema-v2 YAML configuration")
    run_parser.add_argument("--mode", default="auto", help="Concrete stage name or 'auto'")
    run_parser.add_argument(
        "--continue",
        dest="auto_continue",
        action="store_true",
        help="Continue through later runnable stages until completion or failure",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run a completed stage; never bypasses data policy or preflight checks",
    )

    subparsers.add_parser("verify", help="Run compilation, lint, format, tests, and SFT validation")

    experiment_parser = subparsers.add_parser("experiment", help="Validate fair research comparison contracts")
    experiment_subparsers = experiment_parser.add_subparsers(dest="experiment_command", required=True)
    experiment_validate = experiment_subparsers.add_parser(
        "validate", help="Validate a registry and optionally verify every declared config change"
    )
    experiment_validate.add_argument("--registry", required=True, help="Path to a schema-v1 experiment registry")
    experiment_validate.add_argument(
        "--check-configs",
        action="store_true",
        help="Load every arm config and require its actual diff to match changed_keys",
    )
    experiment_validate.add_argument(
        "--ignore-key",
        action="append",
        default=[],
        help="Config path ignored during diff checking; repeat for each operational path",
    )

    data_parser = subparsers.add_parser("data", help="Lock and audit configured data sources")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    lock_parser = data_subparsers.add_parser("lock", help="Resolve and SHA-256 lock every configured source file")
    lock_parser.add_argument("--config", default="config.yaml")
    audit_parser = data_subparsers.add_parser("audit", help="Audit approved sources without retaining rejected text")
    audit_parser.add_argument("--config", default="config.yaml")
    quarantine_parser = data_subparsers.add_parser(
        "quarantine-eval",
        help="Add private knowledge-pilot hashes to the benchmark denylist",
    )
    quarantine_parser.add_argument("--config", default="config.yaml")

    doctor_parser = subparsers.add_parser("doctor", help="Check policy, hardware, storage, and reproducibility")
    doctor_parser.add_argument("--config", default="config.yaml")
    doctor_parser.add_argument("--allow-cpu", action="store_true")
    doctor_parser.add_argument("--verify-hashes", action="store_true")

    transfer_parser = subparsers.add_parser("transfer", help="Build a transfer manifest from approved locked data")
    transfer_subparsers = transfer_parser.add_subparsers(dest="transfer_command", required=True)
    manifest_parser = transfer_subparsers.add_parser("manifest", help="Write checksums and an rsync-ready file list")
    manifest_parser.add_argument("--config", default="config.yaml")
    manifest_parser.add_argument("--output", default="gpu_transfer")
    bundle_parser = transfer_subparsers.add_parser("bundle", help="Build a verified upload-ready ZIP archive")
    bundle_parser.add_argument("--config", default="config.yaml")
    bundle_parser.add_argument("--output", default="gpu_transfer")
    bundle_parser.add_argument("--name", required=True, help="Plain .zip filename")
    return parser


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "run":
        from .main import main as run_pipeline

        run_pipeline(
            config_path=args.config,
            run_mode=args.mode,
            auto_continue=args.auto_continue,
            force_run=args.force,
        )
        return
    if args.command == "verify":
        from .verification import run_quality_gates

        run_quality_gates()
        return
    if args.command == "experiment" and args.experiment_command == "validate":
        from .config import load_config
        from .experiment_contract import load_registry, verify_arm_config_diff

        registry_path = Path(args.registry).resolve()
        registry = load_registry(registry_path)
        checked_candidates = 0
        if args.check_configs:
            for study in registry.studies:
                baseline_arm = study.arm(study.baseline_arm)
                baseline_path = (registry_path.parent / baseline_arm.config).resolve()
                baseline_config = load_config(baseline_path).mutable_copy()
                for arm in study.arms:
                    if arm.id == study.baseline_arm:
                        continue
                    candidate_path = (registry_path.parent / arm.config).resolve()
                    candidate_config = load_config(candidate_path).mutable_copy()
                    verify_arm_config_diff(
                        study,
                        arm.id,
                        baseline_config,
                        candidate_config,
                        ignored_keys=args.ignore_key,
                    )
                    checked_candidates += 1
        arm_count = sum(len(study.arms) for study in registry.studies)
        print(
            f"Experiment registry is valid: studies={len(registry.studies)}, arms={arm_count}, "
            f"config_diffs_checked={checked_candidates}; registry={registry_path}"
        )
        return
    if args.command == "data":
        from .config import load_config
        from .data_governance import audit_allowed_sources, build_source_lock

        config = load_config(args.config).mutable_copy()
        if args.data_command == "lock":
            payload = build_source_lock(config)
            print(
                f"Locked {len(payload['files']):,} files; digest={payload['lock_digest']}; "
                f"output={config['data_policy']['source_lock_path']}"
            )
            return
        if args.data_command == "audit":
            report = audit_allowed_sources(config)
            scanned = sum(int(stats["scanned"]) for stats in report.sources.values())
            rejected = sum(sum(int(count) for count in stats["rejected"].values()) for stats in report.sources.values())
            print(
                f"Audited {scanned:,} samples; category_hits={rejected:,}; digest={report.digest}; "
                f"output={config['data_policy']['audit_path']}"
            )
            return
        if args.data_command == "quarantine-eval":
            from .multiple_choice import quarantine_knowledge_pilot

            result = quarantine_knowledge_pilot(config)
            print(
                f"Quarantined {result['item_count']} private evaluation items; "
                f"added_hashes={result['added_hashes']}; total_hashes={result['total_hashes']}; "
                f"output={result['path']}"
            )
            return
    if args.command == "doctor":
        from .config import load_config
        from .operations import run_doctor

        run_doctor(load_config(args.config), allow_cpu=args.allow_cpu, verify_hashes=args.verify_hashes)
        return
    if args.command == "transfer" and args.transfer_command == "manifest":
        from .config import load_config
        from .operations import build_transfer_manifest

        payload = build_transfer_manifest(load_config(args.config), args.output)
        print(
            f"Transfer manifest ready: {payload['file_count']:,} files, "
            f"{payload['total_bytes'] / 1024**3:.2f} GiB; output={args.output}"
        )
        return
    if args.command == "transfer" and args.transfer_command == "bundle":
        from .config import load_config
        from .operations import build_transfer_archive

        payload = build_transfer_archive(load_config(args.config), args.output, args.name)
        print(
            f"Upload-ready archive: {payload['archive']}; {payload['bytes'] / 1024**2:.2f} MiB; "
            f"sha256={payload['sha256']}; checksum={payload['sha256_file']}"
        )
        return
    raise ConfigError(f"Unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a command and return its documented process exit code."""

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        _dispatch(args)
        return 0
    except DataPolicyError as exc:
        print(f"data policy error: {exc}", file=sys.stderr)
        return EXIT_DATA_POLICY
    except PreflightError as exc:
        print(f"preflight error: {exc}", file=sys.stderr)
        return EXIT_PREFLIGHT
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except Exception as exc:
        print(f"runtime error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME


def entrypoint() -> None:
    """Console-script entrypoint."""

    raise SystemExit(main())
