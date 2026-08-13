"""Shared pytest setup for generated synthetic smoke input."""

from scripts.prepare_synthetic_smoke import write_smoke_fixture


def pytest_sessionstart() -> None:
    """Create ignored test input before any test loads the smoke config."""

    write_smoke_fixture()
