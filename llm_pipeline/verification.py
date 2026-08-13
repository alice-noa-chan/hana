"""Repository quality gates used by the CLI and local development."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TARGETS = (
    "llm_pipeline",
    "scripts",
    "tests",
)


def _run(command: list[str], output: TextIO) -> None:
    print(f"\n> {' '.join(command)}", file=output, flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_quality_gates(output: TextIO | None = None) -> None:
    """Check publication boundaries, compile, lint, format-check, and test."""

    stream = output or sys.stdout
    python = sys.executable
    targets = list(SOURCE_TARGETS)
    _run([python, "scripts/check_public_tree.py"], stream)
    _run([python, "-m", "compileall", "-q", *targets], stream)
    _run([python, "-m", "ruff", "check", *targets], stream)
    _run([python, "-m", "ruff", "format", "--check", *targets], stream)
    _run([python, "-m", "pytest", "-q"], stream)
    coverage_suites = (
        ("llm_pipeline.config", "tests/test_config.py", "tests/test_config_validation_matrix.py"),
        ("llm_pipeline.cli", "tests/test_cli.py"),
        ("llm_pipeline.data_governance", "tests/test_data_governance.py"),
        ("llm_pipeline.data_reader", "tests/test_data_reader.py"),
    )
    for module, *test_files in coverage_suites:
        _run(
            [
                python,
                "-m",
                "pytest",
                "-q",
                *test_files,
                f"--cov={module}",
                "--cov-branch",
                "--cov-report=term",
                "--cov-fail-under=90",
            ],
            stream,
        )
    print("\nAll quality gates passed.", file=stream, flush=True)
