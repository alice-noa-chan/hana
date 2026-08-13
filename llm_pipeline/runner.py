"""Small public orchestration API over the immutable configuration model."""

from __future__ import annotations

from dataclasses import dataclass

from .config import PipelineConfig


@dataclass(frozen=True)
class PipelineRunner:
    """Execute a validated configuration without exposing mutable internals."""

    config: PipelineConfig

    def run(self, mode: str | None = None, *, auto_continue: bool = False, force: bool = False) -> None:
        from .main import run_pipeline_config

        loaded = self.config
        if mode is not None:
            mutable = loaded.mutable_copy()
            mutable["run"]["mode"] = mode
            loaded = PipelineConfig.from_dict(mutable)
        run_pipeline_config(loaded, auto_continue=auto_continue, force_run=force)
