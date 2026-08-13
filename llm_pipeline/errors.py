"""Stable exception categories used at the command boundary."""

from __future__ import annotations


class HanaError(RuntimeError):
    """Base class for expected, user-actionable pipeline failures."""


class ConfigError(ValueError):
    """Configuration or command input is invalid."""


class DataPolicyError(HanaError):
    """A data source, lock, or audit violates the configured policy."""


class PreflightError(HanaError):
    """The local hardware, storage, or runtime environment is not ready."""
