"""Logging, hardware inspection, and reproducibility helpers."""

from __future__ import annotations

import contextlib
import datetime as _dt
import os
import platform
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .artifacts import strict_json_dumps


def _force_utf8_console() -> None:
    """Make stdout/stderr UTF-8 so Korean/Japanese logs do not crash on Windows.

    The default Windows console codec (e.g. cp949) raises UnicodeEncodeError on
    CJK text.  Reconfiguring to UTF-8 with replacement keeps logging and tqdm
    progress bars working regardless of the active code page.
    """

    import sys

    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]


class RunLogger:
    """Write human-readable logs and machine-readable JSONL metrics together."""

    def __init__(
        self,
        log_dir: str | os.PathLike[str],
        jsonl_enabled: bool = True,
        tensorboard_enabled: bool = False,
    ) -> None:
        _force_utf8_console()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path = self.log_dir / "run.log"
        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self.jsonl_enabled = jsonl_enabled
        self.tensorboard = None
        if tensorboard_enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError("logging.tensorboard=true requires the tensorboard package.") from exc
            self.tensorboard = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

    def info(self, message: str) -> None:
        stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            import sys

            sys.stdout.buffer.write((line + "\n").encode("utf-8", "replace"))
            sys.stdout.flush()
        with self.run_log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def metric(self, payload: dict[str, Any]) -> None:
        payload = {"time": _dt.datetime.now().isoformat(timespec="seconds"), **payload}
        if self.jsonl_enabled:
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(strict_json_dumps(payload) + "\n")
        if self.tensorboard is not None:
            step = int(payload.get("step", 0))
            stage = str(payload.get("stage", "pipeline"))
            for key, value in payload.items():
                if (
                    key not in {"step", "stage", "time"}
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    self.tensorboard.add_scalar(f"{stage}/{key}", value, step)
            self.tensorboard.flush()

    def close(self) -> None:
        if self.tensorboard is not None:
            self.tensorboard.close()
            self.tensorboard = None


def make_experiment_dir(config: dict[str, Any]) -> Path:
    """Create a stable experiment folder under run.output_dir."""

    name = config["run"].get("experiment_name") or _dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    root = Path(config["run"]["output_dir"]) / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def setup_logger(config: dict[str, Any], experiment_dir: Path | None = None) -> RunLogger:
    """Create the logger under the experiment folder when available."""

    log_dir = Path(config["logging"]["log_dir"])
    if experiment_dir is not None and not log_dir.is_absolute():
        log_dir = experiment_dir / log_dir
    return RunLogger(
        log_dir,
        bool(config["logging"].get("jsonl_log", True)),
        bool(config["logging"].get("tensorboard", False)),
    )


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch if available."""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def get_git_commit() -> str:
    """Return the current git commit if this folder is inside a repository."""

    git = shutil.which("git")
    if git is None:
        bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe"
        git = str(bundled) if bundled.exists() else None
    if git is None:
        return "git-not-found"
    try:
        result = subprocess.run(
            [git, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "not-a-git-worktree"
    except Exception as exc:
        return f"git-error:{exc}"


def hardware_summary() -> dict[str, Any]:
    """Collect CPU/RAM/GPU information without making CUDA mandatory."""

    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "ram_total_gb": None,
        "cuda_available": False,
        "gpus": [],
    }

    try:
        import psutil

        summary["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        summary["ram_total_gb"] = "psutil-not-installed"

    try:
        import torch

        summary["torch"] = torch.__version__
        summary["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                free, total = torch.cuda.mem_get_info(idx)
                summary["gpus"].append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_vram_gb": round(total / (1024**3), 2),
                        "free_vram_gb": round(free / (1024**3), 2),
                    }
                )
    except Exception as exc:
        summary["torch"] = f"not-available:{exc}"

    return summary


def require_torch():
    """Import torch with an actionable error message."""

    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for this mode. Install dependencies with "
            "`python -m pip install -r requirements.txt`. "
            "For Python 3.14, check PyTorch wheel compatibility; Python 3.10-3.12 "
            "is currently the safest production choice for CUDA stacks."
        ) from exc
