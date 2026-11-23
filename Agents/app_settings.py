"""Typed, fail-fast application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false.")


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


@dataclass(frozen=True)
class APISettings:
    environment: str
    run_root: Path
    database_path: Path
    api_key: str
    embedded_worker: bool
    max_products: int
    max_body_bytes: int
    max_upload_bytes: int
    rate_limit_per_minute: int
    job_max_attempts: int
    max_active_jobs: int
    require_idempotency_key: bool
    minimum_free_disk_bytes: int
    allow_sample_data: bool
    allow_simulation_events: bool

    @classmethod
    def from_environment(cls, app_root: Path) -> "APISettings":
        run_root = Path(
            os.getenv("STEEL_API_RUN_ROOT", str(app_root / "runtime" / "api"))
        ).resolve()
        settings = cls(
            environment=os.getenv("STEEL_ENV", "development").strip().lower(),
            run_root=run_root,
            database_path=Path(
                os.getenv("STEEL_API_STATE_DATABASE", str(run_root / "state.db"))
            ).resolve(),
            api_key=os.getenv("STEEL_API_KEY", ""),
            embedded_worker=_boolean("STEEL_API_EMBEDDED_WORKER", True),
            max_products=_positive_int("STEEL_API_MAX_PRODUCTS", 2000),
            max_body_bytes=_positive_int("STEEL_API_MAX_BODY_BYTES", 5_242_880),
            max_upload_bytes=_positive_int("STEEL_API_MAX_UPLOAD_BYTES", 52_428_800),
            rate_limit_per_minute=_positive_int("STEEL_API_RATE_LIMIT_PER_MINUTE", 120),
            job_max_attempts=_positive_int("STEEL_JOB_MAX_ATTEMPTS", 3),
            max_active_jobs=_positive_int("STEEL_MAX_ACTIVE_JOBS", 1000),
            require_idempotency_key=_boolean(
                "STEEL_REQUIRE_IDEMPOTENCY_KEY",
                os.getenv("STEEL_ENV", "development").strip().lower() == "production",
            ),
            minimum_free_disk_bytes=_positive_int(
                "STEEL_MINIMUM_FREE_DISK_BYTES", 2_147_483_648
            ),
            allow_sample_data=_boolean(
                "STEEL_ALLOW_SAMPLE_DATA",
                os.getenv("STEEL_ENV", "development").strip().lower() != "production",
            ),
            allow_simulation_events=_boolean(
                "STEEL_ALLOW_SIMULATION_EVENTS",
                os.getenv("STEEL_ENV", "development").strip().lower() != "production",
            ),
        )
        settings.validate()
        return settings

    @property
    def production(self) -> bool:
        return self.environment == "production"

    def validate(self) -> None:
        if self.environment not in {"development", "test", "production"}:
            raise ValueError("STEEL_ENV must be development, test, or production.")
        if self.production and len(self.api_key) < 32:
            raise ValueError(
                "Production requires STEEL_API_KEY with at least 32 characters."
            )
        if self.production and self.embedded_worker:
            raise ValueError(
                "Production requires STEEL_API_EMBEDDED_WORKER=false and a "
                "separately supervised worker process."
            )
        if self.production and (self.allow_sample_data or self.allow_simulation_events):
            raise ValueError(
                "Production forbids sample data and simulated dynamic events. "
                "Use a separate development or UAT environment for demonstrations."
            )
        if self.database_path.parent != self.run_root:
            raise ValueError(
                "STEEL_API_STATE_DATABASE must be directly inside STEEL_API_RUN_ROOT."
            )
