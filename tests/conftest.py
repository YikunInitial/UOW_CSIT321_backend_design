"""Shared backend-test setup for a clean repository checkout."""

from pathlib import Path


def pytest_sessionstart(session) -> None:
    # These unit tests use a stable local parent to keep Windows paths short.
    # It is runtime-only and excluded from version control.
    (Path("runtime") / "test_tmp").mkdir(parents=True, exist_ok=True)

