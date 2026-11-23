"""Transactional persistence for plan, event, and replanning state.

The repository deliberately uses the Python standard-library SQLite driver so
the application service, Streamlit UI, and HTTP API share one durable source of
truth without introducing a deployment-specific database dependency.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
from uuid import uuid4

from Agents.planning_orchestrator import PlanningOrchestrator
from Agents.run_manifest import sha256_file

CURRENT_SCHEMA_VERSION = 7
_INITIALISATION_LOCK_GUARD = threading.Lock()
_INITIALISATION_LOCKS: dict[Path, threading.Lock] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateConflictError(RuntimeError):
    """Raised when an idempotency or optimistic-version rule is violated."""


class QueueCapacityError(RuntimeError):
    """Raised when bounded queue capacity has been reached."""


class IdempotencyConflictError(StateConflictError):
    """Raised when an idempotency key is reused with a different request."""


class SQLiteStateRepository:
    """Small transactional repository suitable for a single-site service."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with _INITIALISATION_LOCK_GUARD:
            initialisation_lock = _INITIALISATION_LOCKS.setdefault(
                self.database_path, threading.Lock()
            )
        with initialisation_lock:
            self._initialise()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA synchronous = FULL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._lock, self._connection() as connection:
            for attempt in range(8):
                try:
                    current_mode = connection.execute("PRAGMA journal_mode").fetchone()[
                        0
                    ]
                    if str(current_mode).lower() != "wal":
                        connection.execute("PRAGMA journal_mode = WAL")
                    break
                except sqlite3.OperationalError as error:
                    if "locked" not in str(error).lower() or attempt == 7:
                        raise
                    time.sleep(0.025 * (2**attempt))
            connection.execute("BEGIN EXCLUSIVE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            version_row = connection.execute(
                "SELECT value FROM schema_metadata WHERE key='schema_version'"
            ).fetchone()
            if version_row is None:
                has_legacy_tables = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='jobs'"
                ).fetchone()
                schema_version = 1 if has_legacy_tables else 0
            else:
                schema_version = int(version_row["value"])
            if schema_version > CURRENT_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema {schema_version} is newer than supported "
                    f"schema {CURRENT_SCHEMA_VERSION}."
                )
            if schema_version == 0:
                self._create_schema_v1(connection)
                schema_version = 1
            if schema_version == 1:
                self._migrate_v1_to_v2(connection)
                schema_version = 2
            if schema_version == 2:
                self._migrate_v2_to_v3(connection)
                schema_version = 3
            if schema_version == 3:
                self._migrate_v3_to_v4(connection)
                schema_version = 4
            if schema_version == 4:
                self._migrate_v4_to_v5(connection)
                schema_version = 5
            if schema_version == 5:
                self._migrate_v5_to_v6(connection)
                schema_version = 6
            if schema_version == 6:
                self._migrate_v6_to_v7(connection)
                schema_version = 7
            connection.execute(
                """
                INSERT INTO schema_metadata(key, value) VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (str(schema_version),),
            )

    @staticmethod
    def _create_schema_v1(connection: sqlite3.Connection) -> None:
        statements = [
            """
                CREATE TABLE IF NOT EXISTS plans (
                    plan_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    run_dir TEXT NOT NULL,
                    parent_plan_id TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """,
            """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    plan_id TEXT,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """,
            """
                CREATE TABLE IF NOT EXISTS environments (
                    environment_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """,
            """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """,
        ]
        for statement in statements:
            connection.execute(statement)

    @staticmethod
    def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
        existing_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(jobs)")
        }
        additions = {
            "worker_id": "TEXT",
            "lease_expires_at": "TEXT",
            "heartbeat_at": "TEXT",
            "attempt_count": "INTEGER NOT NULL DEFAULT 0",
            "max_attempts": "INTEGER NOT NULL DEFAULT 3",
        }
        for column, declaration in additions.items():
            if column not in existing_columns:
                connection.execute(
                    f"ALTER TABLE jobs ADD COLUMN {column} {declaration}"
                )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_claim
                ON jobs(status, created_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_lease
                ON jobs(status, lease_expires_at)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS operations (
                operation_id TEXT PRIMARY KEY,
                operation_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                error_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_operations_recovery
                ON operations(status, operation_type)
            """
        )

    @staticmethod
    def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workers (
                worker_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                current_job_id TEXT,
                started_at TEXT NOT NULL,
                heartbeat_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )

    @staticmethod
    def _migrate_v3_to_v4(connection: sqlite3.Connection) -> None:
        existing_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(jobs)")
        }
        additions = {
            "available_at": "TEXT",
            "completed_at": "TEXT",
            "failure_class": "TEXT",
        }
        for column, declaration in additions.items():
            if column not in existing_columns:
                connection.execute(
                    f"ALTER TABLE jobs ADD COLUMN {column} {declaration}"
                )
        connection.execute(
            "UPDATE jobs SET available_at=created_at WHERE available_at IS NULL"
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_available
                ON jobs(status, available_at, created_at)
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS job_attempts (
                job_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                worker_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                outcome TEXT,
                error_json TEXT,
                PRIMARY KEY(job_id, attempt_number),
                FOREIGN KEY(job_id) REFERENCES jobs(job_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                idempotency_key TEXT PRIMARY KEY,
                endpoint TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                job_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(job_id)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_workers_heartbeat
                ON workers(status, heartbeat_at)
            """
        )

    @staticmethod
    def _migrate_v4_to_v5(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS planning_requests (
                request_id TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                status TEXT NOT NULL,
                intake_dir TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                accepted_repair_ids_json TEXT NOT NULL,
                snapshot_path TEXT,
                snapshot_sha256 TEXT,
                confirmed_by TEXT,
                confirmed_at TEXT,
                job_id TEXT,
                plan_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(job_id),
                FOREIGN KEY(plan_id) REFERENCES plans(plan_id)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_planning_requests_status
                ON planning_requests(status, updated_at)
            """
        )

    @staticmethod
    def _migrate_v5_to_v6(connection: sqlite3.Connection) -> None:
        job_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(jobs)")
        }
        if "priority" not in job_columns:
            connection.execute(
                "ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 2"
            )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_priority_claim
                ON jobs(status, priority, available_at, created_at)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scheduling_windows (
                window_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                expected_plan_version INTEGER NOT NULL,
                status TEXT NOT NULL,
                priority TEXT NOT NULL,
                priority_rank INTEGER NOT NULL,
                window_type TEXT NOT NULL,
                opens_at TEXT NOT NULL,
                closes_at TEXT NOT NULL,
                dispatched_at TEXT,
                job_id TEXT,
                event_count INTEGER NOT NULL DEFAULT 0,
                payload_json TEXT NOT NULL,
                result_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(plan_id) REFERENCES plans(plan_id),
                FOREIGN KEY(job_id) REFERENCES jobs(job_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scheduling_window_events (
                event_id TEXT PRIMARY KEY,
                window_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(window_id) REFERENCES scheduling_windows(window_id)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_windows_due
                ON scheduling_windows(status, priority_rank, closes_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_window_events_window
                ON scheduling_window_events(window_id, created_at)
            """
        )

    @staticmethod
    def _migrate_v6_to_v7(connection: sqlite3.Connection) -> None:
        """Add the durable controller for the development/UAT order stream."""
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS continuous_streams (
                stream_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                plan_version INTEGER NOT NULL,
                min_interval_seconds INTEGER NOT NULL,
                max_interval_seconds INTEGER NOT NULL,
                max_products INTEGER NOT NULL,
                cycle_count INTEGER NOT NULL DEFAULT 0,
                next_event_at TEXT NOT NULL,
                current_window_id TEXT,
                last_error_json TEXT,
                recent_events_json TEXT NOT NULL DEFAULT '[]',
                lease_owner TEXT,
                lease_expires_at TEXT,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                stopped_at TEXT,
                FOREIGN KEY(plan_id) REFERENCES plans(plan_id),
                FOREIGN KEY(current_window_id) REFERENCES scheduling_windows(window_id)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_continuous_streams_work
                ON continuous_streams(status, next_event_at, lease_expires_at)
            """
        )
    def schema_version(self) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM schema_metadata WHERE key='schema_version'"
            ).fetchone()
        return int(row["value"])

    def create_planning_request(
        self,
        request_id: str,
        *,
        intake_dir: str | Path,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        now = _utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO planning_requests(
                    request_id, version, status, intake_dir, payload_json,
                    accepted_repair_ids_json, created_at, updated_at
                ) VALUES (?, 1, ?, ?, ?, '[]', ?, ?)
                """,
                (
                    request_id,
                    str(payload.get("status") or "ANALYSING"),
                    str(Path(intake_dir).resolve()),
                    self._dumps(payload),
                    now,
                    now,
                ),
            )
        return self.get_planning_request(request_id)

    def get_planning_request(self, request_id: str) -> Dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM planning_requests WHERE request_id=?", (request_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown planning request: {request_id}")
        value = dict(row)
        value["payload"] = self._loads(value.pop("payload_json"))
        value["accepted_repair_ids"] = self._loads(
            value.pop("accepted_repair_ids_json")
        ) or []
        return value

    def review_planning_request(
        self,
        request_id: str,
        *,
        payload: Dict[str, Any],
        accepted_repair_ids: list[str],
    ) -> Dict[str, Any]:
        now = _utc_now()
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM planning_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown planning request: {request_id}")
            if row["status"] in {
                "CONFIRMED",
                "QUEUED",
                "RUNNING",
                "COMPLETED",
                "FAILED",
            }:
                raise StateConflictError(
                    "Confirmed planning request data is immutable; create a new request to edit it."
                )
            connection.execute(
                """
                UPDATE planning_requests
                SET version=version+1, status=?, payload_json=?,
                    accepted_repair_ids_json=?, updated_at=?
                WHERE request_id=?
                """,
                (
                    str(payload.get("status") or "NEEDS_REVIEW"),
                    self._dumps(payload),
                    self._dumps(list(accepted_repair_ids)),
                    now,
                    request_id,
                ),
            )
        return self.get_planning_request(request_id)

    def confirm_planning_request(
        self,
        request_id: str,
        *,
        accepted_repair_ids: list[str],
        snapshot_path: str | Path,
        snapshot_sha256: str,
        confirmed_by: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        now = _utc_now()
        already_confirmed = False
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, accepted_repair_ids_json FROM planning_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown planning request: {request_id}")
            if row["status"] in {"CONFIRMED", "QUEUED", "RUNNING", "COMPLETED"}:
                existing_ids = self._loads(row["accepted_repair_ids_json"]) or []
                if existing_ids == list(accepted_repair_ids):
                    already_confirmed = True
                else:
                    raise StateConflictError(
                        "Confirmed planning request data cannot be changed."
                    )
            if already_confirmed:
                connection.rollback()
            else:
                connection.execute(
                    """
                    UPDATE planning_requests
                    SET version=version+1, status='CONFIRMED', payload_json=?,
                        accepted_repair_ids_json=?, snapshot_path=?, snapshot_sha256=?,
                        confirmed_by=?, confirmed_at=?, updated_at=?
                    WHERE request_id=?
                    """,
                    (
                        self._dumps(payload),
                        self._dumps(list(accepted_repair_ids)),
                        str(Path(snapshot_path).resolve()),
                        snapshot_sha256,
                        confirmed_by,
                        now,
                        now,
                        request_id,
                    ),
                )
        return self.get_planning_request(request_id)

    def link_planning_request_job(self, request_id: str, job_id: str) -> Dict[str, Any]:
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, job_id FROM planning_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown planning request: {request_id}")
            if row["job_id"] and row["job_id"] != job_id:
                raise StateConflictError("Planning request is already linked to another job.")
            if row["status"] not in {"CONFIRMED", "QUEUED"}:
                raise StateConflictError(
                    f"Planning request cannot be queued from status {row['status']}."
                )
            connection.execute(
                """
                UPDATE planning_requests SET status='QUEUED', job_id=?, updated_at=?
                WHERE request_id=?
                """,
                (job_id, _utc_now(), request_id),
            )
        return self.get_planning_request(request_id)

    @staticmethod
    def _dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _loads(value: Optional[str]) -> Any:
        return json.loads(value) if value else None

    def save_plan(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return self.save_plan_record(
            plan=dict(result["plan"]),
            run_dir=Path(result["solution_path"]).parent,
            solution_path=result["solution_path"],
            manifest_path=result["manifest_path"],
            strategy_path=result["strategy_path"],
        )

    def save_plan_record(
        self,
        *,
        plan: Dict[str, Any],
        run_dir: str | Path,
        solution_path: str | Path,
        manifest_path: str | Path,
        strategy_path: str | Path,
    ) -> Dict[str, Any]:
        now = _utc_now()
        payload = {
            "solution_path": str(solution_path),
            "manifest_path": str(manifest_path),
            "strategy_path": str(strategy_path),
        }
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO plans (
                    plan_id, version, status, run_dir, parent_plan_id,
                    payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO UPDATE SET
                    version=excluded.version,
                    status=excluded.status,
                    run_dir=excluded.run_dir,
                    parent_plan_id=excluded.parent_plan_id,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    plan["plan_id"],
                    int(plan["version"]),
                    plan["status"],
                    str(Path(run_dir)),
                    plan.get("parent_plan_id"),
                    self._dumps(payload),
                    plan.get("created_at") or now,
                    now,
                ),
            )
        return self.get_plan(plan["plan_id"])

    def get_plan(self, plan_id: str) -> Dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM plans WHERE plan_id = ?", (plan_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown plan_id: {plan_id}")
        value = dict(row)
        value["payload"] = self._loads(value.pop("payload_json"))
        return value

    def list_plans_by_status(self, status: str) -> list[Dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM plans WHERE status=? ORDER BY updated_at", (status,)
            ).fetchall()
        values: list[Dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            value["payload"] = self._loads(value.pop("payload_json"))
            values.append(value)
        return values

    def list_recent_plans(
        self, *, limit: int = 20, status: str | None = None
    ) -> list[Dict[str, Any]]:
        """Return newest plans for browser history without exposing SQL details."""
        safe_limit = max(1, min(int(limit), 200))
        query = "SELECT * FROM plans"
        parameters: tuple[Any, ...]
        if status:
            query += " WHERE status=?"
            parameters = (status, safe_limit)
        else:
            parameters = (safe_limit,)
        query += " ORDER BY updated_at DESC LIMIT ?"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        values: list[Dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            value["payload"] = self._loads(value.pop("payload_json"))
            values.append(value)
        return values

    def find_child_plan(
        self, parent_plan_id: str, version: int
    ) -> Optional[Dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM plans
                WHERE parent_plan_id=? AND version=?
                ORDER BY created_at DESC LIMIT 1
                """,
                (parent_plan_id, version),
            ).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["payload"] = self._loads(value.pop("payload_json"))
        return value

    def _record_operation(
        self,
        operation_id: str,
        *,
        operation_type: str,
        entity_id: str,
        status: str,
        payload: Dict[str, Any],
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = _utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO operations(
                    operation_id, operation_type, entity_id, status,
                    payload_json, error_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(operation_id) DO UPDATE SET
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    error_json=excluded.error_json,
                    updated_at=excluded.updated_at
                """,
                (
                    operation_id,
                    operation_type,
                    entity_id,
                    status,
                    self._dumps(payload),
                    self._dumps(error) if error is not None else None,
                    now,
                    now,
                ),
            )

    def approve_plan(
        self,
        plan_id: str,
        expected_version: int,
        *,
        approved_by: str = "business-operator",
    ) -> Dict[str, Any]:
        """Approve a plan with a durable operation journal.

        File artifacts and SQLite cannot share one native transaction. The
        journal makes that boundary explicit: if the process exits between
        writing artifacts and committing plan state, startup recovery can
        safely replay this idempotent operation.
        """
        current = self.get_plan(plan_id)
        if int(current["version"]) != int(expected_version):
            raise StateConflictError("Plan version is stale.")
        if current["status"] == "active":
            return current
        operation_id = f"approve:{plan_id}:v{int(expected_version)}"
        operation_payload = {
            "plan_id": plan_id,
            "expected_version": int(expected_version),
            "approved_by": approved_by,
        }
        self._record_operation(
            operation_id,
            operation_type="approve_plan",
            entity_id=plan_id,
            status="preparing",
            payload=operation_payload,
        )
        try:
            result = self._activate_plan(
                plan_id,
                expected_version,
                approved_by,
                verify_integrity=True,
            )
        except Exception as error:
            self._record_operation(
                operation_id,
                operation_type="approve_plan",
                entity_id=plan_id,
                status="failed",
                payload=operation_payload,
                error={"type": type(error).__name__, "message": str(error)},
            )
            raise
        manifest_path = Path(result["payload"]["manifest_path"])
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            approved_by = str(
                manifest.get("governance", {})
                .get("approval", {})
                .get("approved_by", approved_by)
            )
            operation_payload["approved_by"] = approved_by
        self._record_operation(
            operation_id,
            operation_type="approve_plan",
            entity_id=plan_id,
            status="completed",
            payload=operation_payload,
        )
        return result

    def reject_plan(
        self,
        plan_id: str,
        expected_version: int,
        *,
        rejected_by: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Reject a successor through a recoverable file/database operation."""
        current = self.get_plan(plan_id)
        if int(current["version"]) != int(expected_version):
            raise StateConflictError("Plan version is stale.")
        if current["status"] == "rejected":
            return current
        if current["status"] not in {
            "draft",
            "generated",
            "validated",
            "pending_approval",
        }:
            raise StateConflictError(
                f"Plan cannot be rejected from status {current['status']}."
            )
        operation_id = f"reject:{plan_id}:v{int(expected_version)}"
        operation_payload = {
            "plan_id": plan_id,
            "expected_version": int(expected_version),
            "rejected_by": rejected_by,
            "reason": reason,
            "run_dir": current["run_dir"],
        }
        self._record_operation(
            operation_id,
            operation_type="reject_plan",
            entity_id=plan_id,
            status="preparing",
            payload=operation_payload,
        )
        try:
            self._complete_rejection(operation_payload)
        except Exception as error:
            self._record_operation(
                operation_id,
                operation_type="reject_plan",
                entity_id=plan_id,
                status="failed",
                payload=operation_payload,
                error={"type": type(error).__name__, "message": str(error)},
            )
            raise
        self._record_operation(
            operation_id,
            operation_type="reject_plan",
            entity_id=plan_id,
            status="completed",
            payload=operation_payload,
        )
        return self.get_plan(plan_id)

    def _complete_rejection(self, payload: Dict[str, Any]) -> None:
        plan_id = str(payload["plan_id"])
        expected_version = int(payload["expected_version"])
        disposition_path = Path(payload["run_dir"]) / "plan_disposition.json"
        PlanningOrchestrator.write_json(
            disposition_path,
            {
                "plan_id": plan_id,
                "version": expected_version,
                "status": "rejected",
                "rejected_by": str(payload["rejected_by"]),
                "reason": str(payload["reason"]),
            },
        )
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT version, status FROM plans WHERE plan_id=?", (plan_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown plan_id: {plan_id}")
            if int(row["version"]) != expected_version:
                raise StateConflictError("Plan version is stale.")
            if row["status"] != "rejected":
                if row["status"] not in {
                    "draft",
                    "generated",
                    "validated",
                    "pending_approval",
                }:
                    raise StateConflictError(
                        f"Plan cannot be rejected from status {row['status']}."
                    )
                connection.execute(
                    "UPDATE plans SET status='rejected', updated_at=? WHERE plan_id=?",
                    (_utc_now(), plan_id),
                )

    def _activate_plan(
        self,
        plan_id: str,
        expected_version: int,
        approved_by: str,
        *,
        verify_integrity: bool,
    ) -> Dict[str, Any]:
        """Idempotently synchronise approval artifacts and database state."""
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM plans WHERE plan_id = ?", (plan_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown plan_id: {plan_id}")
            if int(row["version"]) != int(expected_version):
                raise StateConflictError("Plan version is stale.")
            if row["status"] not in {"pending_approval", "approved", "active"}:
                raise StateConflictError(
                    f"Plan cannot be approved from status {row['status']}."
                )
            if row["status"] == "active":
                value = dict(row)
                value["payload"] = self._loads(value.pop("payload_json"))
                return value
            payload = self._loads(row["payload_json"])
            if verify_integrity:
                from Agents.integrity_service import verify_plan_artifacts

                plan_for_verification = dict(row)
                plan_for_verification["payload"] = payload
                plan_for_verification.pop("payload_json", None)
                verify_plan_artifacts(plan_for_verification)
            solution_path = Path(payload["solution_path"])
            solution = json.loads(solution_path.read_text(encoding="utf-8"))
            solution.setdefault("research_system", {}).setdefault("plan", {})[
                "status"
            ] = "active"
            PlanningOrchestrator.write_json(solution_path, solution)

            manifest_path = Path(payload["manifest_path"])
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for artifact in manifest.get("artifacts", []):
                    if artifact.get("name") == "loading_solution":
                        artifact["sha256"] = sha256_file(solution_path)
                manifest.setdefault("governance", {})["approval"] = {
                    "status": "approved",
                    "approved_at": _utc_now(),
                    "approved_by": approved_by,
                    "expected_version": int(expected_version),
                }
                PlanningOrchestrator.write_json(manifest_path, manifest)
            audit_path = Path(row["run_dir"]) / "audit_report.json"
            if audit_path.exists():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                audit.setdefault("governance", {})["plan_status"] = "active"
                audit["governance"]["business_approval_required"] = False
                audit["governance"]["approved_by"] = approved_by
                audit.setdefault("final_sign_off", {})["release_authorised"] = True
                audit["final_sign_off"]["sign_off_reason"] = (
                    "All technical and governance release checks passed."
                )
                PlanningOrchestrator.write_json(audit_path, audit)
                from Agents.audit_service import audit_report_to_markdown

                markdown_path = audit_path.with_suffix(".md")
                temporary_markdown = markdown_path.with_suffix(".md.tmp")
                temporary_markdown.write_text(
                    audit_report_to_markdown(audit), encoding="utf-8"
                )
                temporary_markdown.replace(markdown_path)
            connection.execute(
                "UPDATE plans SET status='active', updated_at=? WHERE plan_id=?",
                (_utc_now(), plan_id),
            )
            if row["parent_plan_id"]:
                connection.execute(
                    """
                    UPDATE plans SET status='superseded', updated_at=?
                    WHERE plan_id=? AND status='active'
                    """,
                    (_utc_now(), row["parent_plan_id"]),
                )
        return self.get_plan(plan_id)

    def recover_incomplete_operations(self) -> Dict[str, int]:
        """Replay operations interrupted while crossing storage boundaries."""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM operations
                WHERE operation_type IN ('approve_plan', 'reject_plan')
                    AND status='preparing'
                ORDER BY created_at
                """
            ).fetchall()
        recovered = 0
        failed = 0
        for row in rows:
            payload = self._loads(row["payload_json"])
            try:
                if row["operation_type"] == "approve_plan":
                    self._activate_plan(
                        str(payload["plan_id"]),
                        int(payload["expected_version"]),
                        str(payload.get("approved_by", "recovery-operator")),
                        verify_integrity=False,
                    )
                else:
                    self._complete_rejection(payload)
            except Exception as error:
                failed += 1
                self._record_operation(
                    row["operation_id"],
                    operation_type=str(row["operation_type"]),
                    entity_id=row["entity_id"],
                    status="failed",
                    payload=payload,
                    error={"type": type(error).__name__, "message": str(error)},
                )
            else:
                recovered += 1
                self._record_operation(
                    row["operation_id"],
                    operation_type=str(row["operation_type"]),
                    entity_id=row["entity_id"],
                    status="completed",
                    payload=payload,
                )
        return {"recovered": recovered, "failed": failed}

    def get_operation(self, operation_id: str) -> Dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown operation_id: {operation_id}")
        value = dict(row)
        value["payload"] = self._loads(value.pop("payload_json"))
        value["error"] = self._loads(value.pop("error_json"))
        return value

    def save_environment(self, environment_id: str, state: Dict[str, Any]) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO environments(environment_id, state_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(environment_id) DO UPDATE SET
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (environment_id, self._dumps(state), _utc_now()),
            )

    def load_environment(self, environment_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT state_json FROM environments WHERE environment_id=?",
                (environment_id,),
            ).fetchone()
        return self._loads(row["state_json"]) if row else None

    def reserve_event(self, event: Dict[str, Any]) -> None:
        now = _utc_now()
        try:
            with self._lock, self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO events(
                        event_id, plan_id, status, payload_json,
                        result_json, created_at, updated_at
                    ) VALUES (?, ?, 'processing', ?, NULL, ?, ?)
                    """,
                    (
                        event["event_id"],
                        event.get("expected_plan_id"),
                        self._dumps(event),
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise StateConflictError(
                f"Event already exists: {event['event_id']}"
            ) from error

    def complete_event(self, event_id: str, result: Dict[str, Any]) -> None:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE events SET status='completed', result_json=?, updated_at=?
                WHERE event_id=? AND status='processing'
                """,
                (self._dumps(result), _utc_now(), event_id),
            )
            if cursor.rowcount != 1:
                raise StateConflictError(
                    f"Event is missing or no longer processing: {event_id}"
                )

    def fail_event(self, event_id: str, error: Dict[str, Any]) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE events SET status='failed', result_json=?, updated_at=? WHERE event_id=?",
                (self._dumps({"error": error}), _utc_now(), event_id),
            )

    def get_event(self, event_id: str) -> Dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown event_id: {event_id}")
        value = dict(row)
        value["payload"] = self._loads(value.pop("payload_json"))
        value["result"] = self._loads(value.pop("result_json"))
        return value

    def reopen_failed_event(self, event: Dict[str, Any]) -> None:
        with self._lock, self._connection() as connection:
            existing = connection.execute(
                "SELECT payload_json FROM events WHERE event_id=? AND status='failed'",
                (event["event_id"],),
            ).fetchone()
            if existing is None:
                raise StateConflictError("Event is not in a retryable failed state.")
            previous = self._loads(existing["payload_json"])
            if previous.get("expected_plan_id") != event.get(
                "expected_plan_id"
            ) or previous.get("event_type") != event.get("event_type"):
                raise StateConflictError(
                    "Event id was already used for a different request."
                )
            connection.execute(
                """
                UPDATE events SET status='processing', payload_json=?,
                    result_json=NULL, updated_at=? WHERE event_id=?
                """,
                (self._dumps(event), _utc_now(), event["event_id"]),
            )

    @staticmethod
    def _continuous_stream_value(row: sqlite3.Row) -> Dict[str, Any]:
        value = dict(row)
        value["last_error"] = SQLiteStateRepository._loads(
            value.pop("last_error_json")
        )
        value["recent_events"] = SQLiteStateRepository._loads(
            value.pop("recent_events_json")
        ) or []
        return value

    def start_continuous_stream(
        self,
        *,
        plan_id: str,
        expected_version: int,
        min_interval_seconds: int,
        max_interval_seconds: int,
        max_products: int,
        created_by: str,
    ) -> Dict[str, Any]:
        """Start one durable simulated order stream for this deployment.

        The controller is intentionally single-instance. Repeated browser
        submissions return the existing stream instead of creating duplicate
        order producers.
        """
        if min_interval_seconds < 1 or max_interval_seconds < min_interval_seconds:
            raise ValueError("Invalid continuous stream interval range.")
        now = _utc_now()
        stream_id = f"stream_{uuid4().hex}"
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM continuous_streams
                WHERE status IN ('running','stopping')
                ORDER BY created_at DESC LIMIT 1
                """
            ).fetchone()
            if existing is not None:
                return self._continuous_stream_value(existing)
            plan = connection.execute(
                "SELECT version, status FROM plans WHERE plan_id=?", (plan_id,)
            ).fetchone()
            if plan is None:
                raise KeyError(f"Unknown plan_id: {plan_id}")
            if plan["status"] != "active":
                raise StateConflictError("Continuous operations require an active plan.")
            if int(plan["version"]) != int(expected_version):
                raise StateConflictError("Plan version is stale.")
            connection.execute(
                """
                INSERT INTO continuous_streams(
                    stream_id, status, plan_id, plan_version,
                    min_interval_seconds, max_interval_seconds, max_products,
                    cycle_count, next_event_at, recent_events_json, created_by,
                    created_at, updated_at
                ) VALUES (?, 'running', ?, ?, ?, ?, ?, 0, ?, '[]', ?, ?, ?)
                """,
                (
                    stream_id,
                    plan_id,
                    int(expected_version),
                    int(min_interval_seconds),
                    int(max_interval_seconds),
                    int(max_products),
                    now,
                    created_by,
                    now,
                    now,
                ),
            )
        return self.get_continuous_stream(stream_id)

    def get_continuous_stream(self, stream_id: str) -> Dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM continuous_streams WHERE stream_id=?", (stream_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown continuous stream: {stream_id}")
        return self._continuous_stream_value(row)

    def get_active_continuous_stream(self) -> Optional[Dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM continuous_streams
                WHERE status IN ('running','stopping')
                ORDER BY created_at DESC LIMIT 1
                """
            ).fetchone()
        return self._continuous_stream_value(row) if row is not None else None

    def request_continuous_stream_stop(self, stream_id: str) -> Dict[str, Any]:
        now = _utc_now()
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, current_window_id FROM continuous_streams WHERE stream_id=?",
                (stream_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown continuous stream: {stream_id}")
            if row["status"] not in {"stopped", "failed"}:
                status = "stopping" if row["current_window_id"] else "stopped"
                connection.execute(
                    """
                    UPDATE continuous_streams
                    SET status=?, stopped_at=CASE WHEN ?='stopped' THEN ? ELSE stopped_at END,
                        lease_owner=NULL, lease_expires_at=NULL, updated_at=?
                    WHERE stream_id=?
                    """,
                    (status, status, now, now, stream_id),
                )
        return self.get_continuous_stream(stream_id)

    def claim_continuous_stream(
        self, worker_id: str, *, lease_seconds: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        now = _utc_now()
        lease_until = (
            datetime.now(timezone.utc) + timedelta(seconds=max(1.0, lease_seconds))
        ).isoformat()
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM continuous_streams
                WHERE status IN ('running','stopping')
                  AND (lease_expires_at IS NULL OR lease_expires_at <= ? OR lease_owner=?)
                ORDER BY created_at LIMIT 1
                """,
                (now, worker_id),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE continuous_streams SET lease_owner=?, lease_expires_at=?, updated_at=?
                WHERE stream_id=?
                """,
                (worker_id, lease_until, now, row["stream_id"]),
            )
        return self.get_continuous_stream(str(row["stream_id"]))

    def update_continuous_stream(
        self,
        stream_id: str,
        worker_id: str,
        *,
        status: Optional[str] = None,
        plan_id: Optional[str] = None,
        plan_version: Optional[int] = None,
        cycle_count: Optional[int] = None,
        next_event_at: Optional[str] = None,
        current_window_id: Any = ...,
        last_error: Any = ...,
        recent_events: Optional[list[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Commit one controller transition and release its worker lease."""
        allowed_statuses = {"running", "stopping", "stopped", "failed"}
        if status is not None and status not in allowed_statuses:
            raise ValueError("Invalid continuous stream status.")
        assignments = ["lease_owner=NULL", "lease_expires_at=NULL", "updated_at=?"]
        values: list[Any] = [_utc_now()]
        fields = {
            "status": status,
            "plan_id": plan_id,
            "plan_version": plan_version,
            "cycle_count": cycle_count,
            "next_event_at": next_event_at,
        }
        for name, value in fields.items():
            if value is not None:
                assignments.append(f"{name}=?")
                values.append(value)
        if current_window_id is not ...:
            assignments.append("current_window_id=?")
            values.append(current_window_id)
        if last_error is not ...:
            assignments.append("last_error_json=?")
            values.append(self._dumps(last_error) if last_error is not None else None)
        if recent_events is not None:
            assignments.append("recent_events_json=?")
            values.append(self._dumps(recent_events[-40:]))
        if status in {"stopped", "failed"}:
            assignments.append("stopped_at=?")
            values.append(_utc_now())
        values.extend([stream_id, worker_id])
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE continuous_streams SET {', '.join(assignments)} "
                "WHERE stream_id=? AND lease_owner=?",
                tuple(values),
            )
            if cursor.rowcount != 1:
                raise StateConflictError("Continuous stream worker lease was lost.")
        return self.get_continuous_stream(stream_id)

    def add_scheduling_event(
        self,
        event: Dict[str, Any],
        *,
        priority: str,
        priority_rank: int,
        window_type: str,
        closes_at: str,
        window_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist an event and atomically place it in a compatible open window."""
        if priority not in {"P0", "P1", "P2", "P3"}:
            raise ValueError("priority must be P0, P1, P2, or P3")
        if priority_rank not in {0, 1, 2, 3}:
            raise ValueError("priority_rank must be between 0 and 3")
        event_id = str(event.get("event_id") or "").strip()
        plan_id = str(event.get("expected_plan_id") or "").strip()
        expected_version = int(event.get("expected_plan_version") or 0)
        if not event_id or not plan_id or expected_version < 1:
            raise ValueError("event identity and expected plan version are required")
        with self._connection() as connection:
            replay = connection.execute(
                "SELECT window_id, payload_json FROM scheduling_window_events WHERE event_id=?",
                (event_id,),
            ).fetchone()
        if replay is not None:
            previous = self._loads(replay["payload_json"]) or {}
            if (
                previous.get("expected_plan_id") != plan_id
                or int(previous.get("expected_plan_version") or 0) != expected_version
                or previous.get("event_type") != event.get("event_type")
            ):
                raise StateConflictError(
                    "Event id was already used for a different scheduled request."
                )
            return self.get_scheduling_window(str(replay["window_id"]))
        now = _utc_now()
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM events WHERE event_id=?", (event_id,)
            ).fetchone() or connection.execute(
                "SELECT 1 FROM scheduling_window_events WHERE event_id=?", (event_id,)
            ).fetchone():
                raise StateConflictError("Event id was already used.")
            plan = connection.execute(
                "SELECT version, status FROM plans WHERE plan_id=?", (plan_id,)
            ).fetchone()
            if plan is None:
                raise KeyError(f"Unknown plan_id: {plan_id}")
            if plan["status"] != "active":
                raise StateConflictError("Scheduled events require an active plan.")
            if int(plan["version"]) != expected_version:
                raise StateConflictError("Plan version is stale.")
            in_flight = connection.execute(
                """
                SELECT 1 FROM scheduling_windows
                WHERE plan_id=? AND expected_plan_version=? AND status='dispatched'
                LIMIT 1
                """,
                (plan_id, expected_version),
            ).fetchone()
            if in_flight is not None:
                raise StateConflictError(
                    "A scheduling window is already replanning this plan version."
                )
            pending_successor = connection.execute(
                """
                SELECT 1 FROM plans
                WHERE parent_plan_id=? AND version=? AND status='pending_approval'
                LIMIT 1
                """,
                (plan_id, expected_version + 1),
            ).fetchone()
            if pending_successor is not None:
                raise StateConflictError(
                    "Approve or reject the pending successor before scheduling more events."
                )
            row = connection.execute(
                """
                SELECT window_id, priority_rank, window_type, closes_at, payload_json
                FROM scheduling_windows
                WHERE plan_id=? AND expected_plan_version=?
                    AND status='collecting'
                ORDER BY priority_rank, closes_at, created_at LIMIT 1
                """,
                (plan_id, expected_version),
            ).fetchone()
            if row is None:
                window_id = f"window_{uuid4().hex}"
                connection.execute(
                    """
                    INSERT INTO scheduling_windows(
                        window_id, plan_id, expected_plan_version, status,
                        priority, priority_rank, window_type, opens_at, closes_at,
                        event_count, payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, 'collecting', ?, ?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (
                        window_id,
                        plan_id,
                        expected_version,
                        priority,
                        priority_rank,
                        window_type,
                        now,
                        closes_at,
                        self._dumps(window_payload or {}),
                        now,
                        now,
                    ),
                )
            else:
                window_id = str(row["window_id"])
                merged_rank = min(int(row["priority_rank"]), priority_rank)
                merged_priority = f"P{merged_rank}"
                merged_closes_at = min(str(row["closes_at"]), closes_at)
                merged_type = (
                    window_type
                    if closes_at < str(row["closes_at"])
                    else str(row["window_type"])
                )
                existing_payload = self._loads(row["payload_json"]) or {}
                history = list(existing_payload.get("priority_decision_history") or [])
                history.append(window_payload or {})
                existing_payload["priority_decision_history"] = history[-50:]
                connection.execute(
                    """
                    UPDATE scheduling_windows SET priority=?, priority_rank=?,
                        window_type=?, closes_at=?, payload_json=?, updated_at=?
                    WHERE window_id=? AND status='collecting'
                    """,
                    (
                        merged_priority,
                        merged_rank,
                        merged_type,
                        merged_closes_at,
                        self._dumps(existing_payload),
                        now,
                        window_id,
                    ),
                )
            connection.execute(
                """
                INSERT INTO scheduling_window_events(
                    event_id, window_id, status, payload_json, created_at, updated_at
                ) VALUES (?, ?, 'scheduled', ?, ?, ?)
                """,
                (event_id, window_id, self._dumps(event), now, now),
            )
            connection.execute(
                """
                UPDATE scheduling_windows
                SET event_count=event_count + 1, updated_at=? WHERE window_id=?
                """,
                (now, window_id),
            )
        return self.get_scheduling_window(window_id)

    def get_scheduling_window(self, window_id: str) -> Dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM scheduling_windows WHERE window_id=?", (window_id,)
            ).fetchone()
            event_rows = connection.execute(
                """
                SELECT event_id, status, payload_json, created_at, updated_at
                FROM scheduling_window_events WHERE window_id=?
                ORDER BY created_at, event_id
                """,
                (window_id,),
            ).fetchall()
        if row is None:
            raise KeyError(f"Unknown scheduling window: {window_id}")
        value = dict(row)
        value["payload"] = self._loads(value.pop("payload_json"))
        value["result"] = self._loads(value.pop("result_json"))
        value["events"] = [
            {
                **{key: item[key] for key in ("event_id", "status", "created_at", "updated_at")},
                "payload": self._loads(item["payload_json"]),
            }
            for item in event_rows
        ]
        return value

    def list_scheduling_windows(
        self, *, status: Optional[str] = None, limit: int = 100
    ) -> list[Dict[str, Any]]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        query = "SELECT window_id FROM scheduling_windows"
        parameters: tuple[Any, ...] = ()
        if status:
            query += " WHERE status=?"
            parameters = (status,)
        query += " ORDER BY created_at DESC, priority_rank LIMIT ?"
        parameters += (limit,)
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self.get_scheduling_window(str(row["window_id"])) for row in rows]

    def dispatch_due_windows(
        self,
        *,
        max_attempts: int = 3,
        max_active_jobs: int = 1000,
        limit: int = 20,
    ) -> list[str]:
        """Convert due durable windows to priority queue jobs exactly once."""
        now = _utc_now()
        dispatched: list[str] = []
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active_jobs = int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM jobs WHERE status IN ('queued','running')"
                ).fetchone()["count"]
            )
            capacity = max(0, min(limit, max_active_jobs - active_jobs))
            if capacity == 0:
                return []
            windows = connection.execute(
                """
                SELECT * FROM scheduling_windows
                WHERE status='collecting' AND closes_at <= ?
                ORDER BY priority_rank, closes_at, created_at LIMIT ?
                """,
                (now, capacity),
            ).fetchall()
            for window in windows:
                event_rows = connection.execute(
                    """
                    SELECT payload_json FROM scheduling_window_events
                    WHERE window_id=? AND status='scheduled'
                    ORDER BY created_at, event_id
                    """,
                    (window["window_id"],),
                ).fetchall()
                events = [self._loads(item["payload_json"]) for item in event_rows]
                if not events:
                    connection.execute(
                        "UPDATE scheduling_windows SET status='failed', updated_at=? WHERE window_id=?",
                        (now, window["window_id"]),
                    )
                    continue
                job_id = f"job_{uuid4().hex}"
                request = {
                    "job_type": "window_replanning",
                    "window_id": window["window_id"],
                    "plan_id": window["plan_id"],
                    "expected_version": int(window["expected_plan_version"]),
                    "priority": window["priority"],
                    "priority_rank": int(window["priority_rank"]),
                    "window_type": window["window_type"],
                    "window_payload": self._loads(window["payload_json"]),
                    "events": events,
                    "time_budget_seconds": max(
                        float(item.get("time_budget_seconds") or 5.0)
                        for item in events
                    ),
                    "generate_visuals": any(
                        bool(item.get("generate_visuals", False)) for item in events
                    ),
                }
                connection.execute(
                    """
                    INSERT INTO jobs(
                        job_id, status, request_json, max_attempts, priority,
                        available_at, created_at, updated_at
                    ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        self._dumps(request),
                        max_attempts,
                        int(window["priority_rank"]),
                        now,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    UPDATE scheduling_windows
                    SET status='dispatched', dispatched_at=?, job_id=?, updated_at=?
                    WHERE window_id=? AND status='collecting'
                    """,
                    (now, job_id, now, window["window_id"]),
                )
                connection.execute(
                    """
                    UPDATE scheduling_window_events SET status='dispatched', updated_at=?
                    WHERE window_id=? AND status='scheduled'
                    """,
                    (now, window["window_id"]),
                )
                dispatched.append(job_id)
        return dispatched

    def create_job(
        self, job_id: str, request: Dict[str, Any], *, max_attempts: int = 3
    ) -> None:
        self.enqueue_job(
            job_id,
            request,
            max_attempts=max_attempts,
            max_active_jobs=2_147_483_647,
        )

    def enqueue_job(
        self,
        job_id: str,
        request: Dict[str, Any],
        *,
        max_attempts: int = 3,
        max_active_jobs: int = 1000,
        idempotency_key: Optional[str] = None,
        endpoint: str = "",
        request_hash: str = "",
        planning_request_id: Optional[str] = None,
        priority: int = 2,
    ) -> tuple[str, bool]:
        """Atomically apply idempotency, backpressure, and job creation."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if max_active_jobs < 1:
            raise ValueError("max_active_jobs must be at least 1")
        if priority not in {0, 1, 2, 3}:
            raise ValueError("priority must be an integer from 0 (highest) to 3")
        if idempotency_key and (not endpoint or not request_hash):
            raise ValueError("endpoint and request_hash are required for idempotency")
        now = _utc_now()
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                existing = connection.execute(
                    "SELECT * FROM idempotency_keys WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["endpoint"] != endpoint
                        or existing["request_hash"] != request_hash
                    ):
                        raise IdempotencyConflictError(
                            "Idempotency key was already used for a different request."
                        )
                    if planning_request_id:
                        planning_row = connection.execute(
                            "SELECT status, job_id FROM planning_requests WHERE request_id=?",
                            (planning_request_id,),
                        ).fetchone()
                        if planning_row is None:
                            raise KeyError(
                                f"Unknown planning request: {planning_request_id}"
                            )
                        if planning_row["job_id"] not in {
                            None,
                            str(existing["job_id"]),
                        }:
                            raise StateConflictError(
                                "Planning request is linked to a different job."
                            )
                        if planning_row["job_id"] is None:
                            self._link_planning_request_job_in_transaction(
                                connection,
                                planning_request_id,
                                str(existing["job_id"]),
                                now,
                            )
                    return str(existing["job_id"]), False
            if planning_request_id:
                planning_row = connection.execute(
                    "SELECT status, job_id FROM planning_requests WHERE request_id=?",
                    (planning_request_id,),
                ).fetchone()
                if planning_row is None:
                    raise KeyError(f"Unknown planning request: {planning_request_id}")
                if planning_row["job_id"] and planning_row["job_id"] != job_id:
                    prior_job = connection.execute(
                        "SELECT status FROM jobs WHERE job_id=?",
                        (planning_row["job_id"],),
                    ).fetchone()
                    can_replace_failed = (
                        planning_row["status"] == "FAILED"
                        and prior_job is not None
                        and prior_job["status"] in {"failed", "dead_letter"}
                    )
                    if not can_replace_failed:
                        raise StateConflictError(
                            "Planning request is already linked to another job."
                        )
                if planning_row["status"] not in {"CONFIRMED", "QUEUED", "FAILED"}:
                    raise StateConflictError(
                        f"Planning request cannot be queued from status {planning_row['status']}."
                    )
            active_jobs = connection.execute(
                """
                SELECT COUNT(*) AS count FROM jobs
                WHERE status IN ('queued', 'running')
                """
            ).fetchone()["count"]
            if int(active_jobs) >= max_active_jobs:
                raise QueueCapacityError(
                    f"Queue capacity of {max_active_jobs} active jobs was reached."
                )
            connection.execute(
                """
                INSERT INTO jobs(
                    job_id, status, request_json, max_attempts, priority,
                    available_at, created_at, updated_at
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    self._dumps(request),
                    max_attempts,
                    priority,
                    now,
                    now,
                    now,
                ),
            )
            if idempotency_key:
                connection.execute(
                    """
                    INSERT INTO idempotency_keys(
                        idempotency_key, endpoint, request_hash, job_id, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (idempotency_key, endpoint, request_hash, job_id, now),
                )
            if planning_request_id:
                self._link_planning_request_job_in_transaction(
                    connection, planning_request_id, job_id, now
                )
        return job_id, True

    @staticmethod
    def _link_planning_request_job_in_transaction(
        connection: sqlite3.Connection,
        request_id: str,
        job_id: str,
        now: str,
    ) -> None:
        row = connection.execute(
            "SELECT status, job_id FROM planning_requests WHERE request_id=?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown planning request: {request_id}")
        if (
            row["job_id"]
            and row["job_id"] != job_id
            and row["status"] != "FAILED"
        ):
            raise StateConflictError("Planning request is already linked to another job.")
        if row["status"] not in {"CONFIRMED", "QUEUED", "FAILED"}:
            raise StateConflictError(
                f"Planning request cannot be queued from status {row['status']}."
            )
        connection.execute(
            """
            UPDATE planning_requests SET status='QUEUED', job_id=?, updated_at=?
            WHERE request_id=?
            """,
            (job_id, now, request_id),
        )

    def claim_next_job(
        self, worker_id: str, *, lease_seconds: float = 60.0
    ) -> Optional[Dict[str, Any]]:
        """Claim one queued job with an expiring lease.

        ``BEGIN IMMEDIATE`` serialises competing claimers, so a job can be
        owned by only one worker even when several worker processes poll at
        the same time.
        """
        if not worker_id.strip():
            raise ValueError("worker_id is required")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = _utc_now()
        lease_expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
        ).isoformat()
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._recover_expired_jobs(connection, now)
            row = connection.execute(
                """
                SELECT job_id FROM jobs
                WHERE status='queued' AND attempt_count < max_attempts
                    AND (available_at IS NULL OR available_at <= ?)
                ORDER BY priority, available_at, created_at, job_id
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                return None
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status='running', worker_id=?, lease_expires_at=?,
                    heartbeat_at=?, attempt_count=attempt_count + 1,
                    updated_at=?
                WHERE job_id=? AND status='queued'
                """,
                (worker_id, lease_expires_at, now, now, row["job_id"]),
            )
            if cursor.rowcount != 1:
                return None
            attempt_number = connection.execute(
                "SELECT attempt_count FROM jobs WHERE job_id=?", (row["job_id"],)
            ).fetchone()["attempt_count"]
            connection.execute(
                """
                INSERT INTO job_attempts(
                    job_id, attempt_number, worker_id, started_at
                ) VALUES (?, ?, ?, ?)
                """,
                (row["job_id"], attempt_number, worker_id, now),
            )
            claimed_request = self._loads(
                connection.execute(
                    "SELECT request_json FROM jobs WHERE job_id=?", (row["job_id"],)
                ).fetchone()["request_json"]
            )
            planning_request_id = claimed_request.get("planning_request_id")
            if planning_request_id:
                connection.execute(
                    """
                    UPDATE planning_requests SET status='RUNNING', updated_at=?
                    WHERE request_id=? AND job_id=? AND status IN ('QUEUED', 'RUNNING')
                    """,
                    (now, str(planning_request_id), row["job_id"]),
                )
        return self.get_job(row["job_id"])

    def heartbeat_job(
        self, job_id: str, worker_id: str, *, lease_seconds: float = 60.0
    ) -> bool:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = _utc_now()
        lease_expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
        ).isoformat()
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs SET heartbeat_at=?, lease_expires_at=?, updated_at=?
                WHERE job_id=? AND worker_id=? AND status='running'
                """,
                (now, lease_expires_at, now, job_id, worker_id),
            )
        return cursor.rowcount == 1

    @staticmethod
    def _recover_expired_jobs(
        connection: sqlite3.Connection, now: str
    ) -> Dict[str, int]:
        rows = connection.execute(
            """
            SELECT job_id, attempt_count, max_attempts, request_json FROM jobs
            WHERE status='running'
                AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
            """,
            (now,),
        ).fetchall()
        requeued = 0
        dead_lettered = 0
        for row in rows:
            exhausted = int(row["attempt_count"]) >= int(row["max_attempts"])
            status = "dead_letter" if exhausted else "queued"
            error = {
                "code": "JOB_LEASE_EXHAUSTED" if exhausted else "JOB_LEASE_EXPIRED",
                "message": (
                    "Worker lease expired and retry limit was reached."
                    if exhausted
                    else "Worker lease expired; job was returned to the queue."
                ),
            }
            connection.execute(
                """
                UPDATE jobs SET status=?, worker_id=NULL,
                    lease_expires_at=NULL, heartbeat_at=NULL, available_at=?,
                    failure_class='transient', error_json=?,
                    completed_at=?, updated_at=?
                WHERE job_id=? AND status='running'
                """,
                (
                    status,
                    now,
                    json.dumps(error),
                    now if exhausted else None,
                    now,
                    row["job_id"],
                ),
            )
            request = json.loads(row["request_json"])
            planning_request_id = request.get("planning_request_id")
            if planning_request_id:
                connection.execute(
                    """
                    UPDATE planning_requests SET status=?, updated_at=?
                    WHERE request_id=? AND job_id=?
                    """,
                    (
                        "FAILED" if exhausted else "QUEUED",
                        now,
                        str(planning_request_id),
                        row["job_id"],
                    ),
                )
            event_id = request.get("event_id")
            if request.get("job_type") == "replanning" and event_id:
                connection.execute(
                    """
                    UPDATE events SET status='failed', result_json=?, updated_at=?
                    WHERE event_id=? AND status='processing'
                    """,
                    (json.dumps({"error": error}), now, event_id),
                )
            if exhausted and request.get("job_type") == "window_replanning":
                window_id = str(request.get("window_id") or "")
                connection.execute(
                    """
                    UPDATE scheduling_windows SET status='failed', result_json=?,
                        updated_at=? WHERE window_id=? AND job_id=?
                    """,
                    (json.dumps({"error": error}), now, window_id, row["job_id"]),
                )
                connection.execute(
                    """
                    UPDATE scheduling_window_events SET status='failed', updated_at=?
                    WHERE window_id=?
                    """,
                    (now, window_id),
                )
            connection.execute(
                """
                UPDATE job_attempts SET finished_at=?, outcome=?, error_json=?
                WHERE job_id=? AND attempt_number=? AND finished_at IS NULL
                """,
                (
                    now,
                    "dead_letter" if exhausted else "lease_expired",
                    json.dumps(error),
                    row["job_id"],
                    row["attempt_count"],
                ),
            )
            if exhausted:
                dead_lettered += 1
            else:
                requeued += 1
        return {"requeued": requeued, "dead_lettered": dead_lettered}

    def recover_expired_jobs(self) -> Dict[str, int]:
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            return self._recover_expired_jobs(connection, _utc_now())

    def recover_orphaned_events(self) -> Dict[str, int]:
        """Fail processing events that no queued/running job can resume."""
        now = _utc_now()
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active_event_ids: set[str] = set()
            for row in connection.execute(
                "SELECT request_json FROM jobs WHERE status IN ('queued', 'running')"
            ):
                request = self._loads(row["request_json"])
                if request.get("job_type") == "replanning" and request.get("event_id"):
                    active_event_ids.add(str(request["event_id"]))
            rows = connection.execute(
                "SELECT event_id FROM events WHERE status='processing'"
            ).fetchall()
            recovered = 0
            for row in rows:
                if row["event_id"] in active_event_ids:
                    continue
                error = {
                    "code": "ORPHANED_EVENT_RECOVERED",
                    "message": "No active job owned this processing event.",
                }
                connection.execute(
                    """
                    UPDATE events SET status='failed', result_json=?, updated_at=?
                    WHERE event_id=? AND status='processing'
                    """,
                    (self._dumps({"error": error}), now, row["event_id"]),
                )
                recovered += 1
        return {"failed_orphans": recovered}

    def complete_job(self, job_id: str, worker_id: str, result: Dict[str, Any]) -> None:
        now = _utc_now()
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT attempt_count, request_json FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown job_id: {job_id}")
            cursor = connection.execute(
                """
                UPDATE jobs SET status='completed', result_json=?, error_json=NULL,
                    failure_class=NULL, worker_id=NULL, lease_expires_at=NULL,
                    heartbeat_at=NULL, completed_at=?, updated_at=?
                WHERE job_id=? AND status='running' AND worker_id=?
                """,
                (self._dumps(result), now, now, job_id, worker_id),
            )
            if cursor.rowcount != 1:
                raise StateConflictError("Job is not owned by this worker.")
            connection.execute(
                """
                UPDATE job_attempts SET finished_at=?, outcome='completed'
                WHERE job_id=? AND attempt_number=? AND finished_at IS NULL
                """,
                (now, job_id, row["attempt_count"]),
            )
            completed_request = self._loads(row["request_json"])
            planning_request_id = completed_request.get("planning_request_id")
            if planning_request_id:
                plan_id = str((result.get("plan") or {}).get("plan_id") or "") or None
                connection.execute(
                    """
                    UPDATE planning_requests
                    SET status='COMPLETED', plan_id=?, updated_at=?
                    WHERE request_id=? AND job_id=?
                    """,
                    (plan_id, now, str(planning_request_id), job_id),
                )
            if completed_request.get("job_type") == "window_replanning":
                window_id = str(completed_request.get("window_id") or "")
                connection.execute(
                    """
                    UPDATE scheduling_windows SET status='completed', result_json=?,
                        updated_at=? WHERE window_id=? AND job_id=?
                    """,
                    (self._dumps(result), now, window_id, job_id),
                )
                connection.execute(
                    """
                    UPDATE scheduling_window_events SET status='completed', updated_at=?
                    WHERE window_id=?
                    """,
                    (now, window_id),
                )

    def fail_job(
        self,
        job_id: str,
        worker_id: str,
        error: Dict[str, Any],
        *,
        failure_class: str = "permanent",
    ) -> None:
        now = _utc_now()
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT attempt_count, request_json FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown job_id: {job_id}")
            cursor = connection.execute(
                """
                UPDATE jobs SET status='failed', error_json=?, failure_class=?,
                    worker_id=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
                    completed_at=?, updated_at=?
                WHERE job_id=? AND status='running' AND worker_id=?
                """,
                (
                    self._dumps(error),
                    failure_class,
                    now,
                    now,
                    job_id,
                    worker_id,
                ),
            )
            if cursor.rowcount != 1:
                raise StateConflictError("Job is not owned by this worker.")
            connection.execute(
                """
                UPDATE job_attempts SET finished_at=?, outcome='failed', error_json=?
                WHERE job_id=? AND attempt_number=? AND finished_at IS NULL
                """,
                (now, self._dumps(error), job_id, row["attempt_count"]),
            )
            failed_request = self._loads(row["request_json"])
            planning_request_id = failed_request.get("planning_request_id")
            if planning_request_id:
                connection.execute(
                    """
                    UPDATE planning_requests SET status='FAILED', updated_at=?
                    WHERE request_id=? AND job_id=?
                    """,
                    (now, str(planning_request_id), job_id),
                )
            if failed_request.get("job_type") == "window_replanning":
                window_id = str(failed_request.get("window_id") or "")
                connection.execute(
                    """
                    UPDATE scheduling_windows SET status='failed', result_json=?,
                        updated_at=? WHERE window_id=? AND job_id=?
                    """,
                    (self._dumps({"error": error}), now, window_id, job_id),
                )
                connection.execute(
                    """
                    UPDATE scheduling_window_events SET status='failed', updated_at=?
                    WHERE window_id=?
                    """,
                    (now, window_id),
                )

    def retry_job(
        self,
        job_id: str,
        worker_id: str,
        error: Dict[str, Any],
        *,
        delay_seconds: float,
    ) -> str:
        """Schedule a delayed retry or dead-letter an exhausted job."""
        if delay_seconds < 0:
            raise ValueError("delay_seconds must not be negative")
        now = _utc_now()
        available_at = (
            datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
        ).isoformat()
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT attempt_count, max_attempts, request_json FROM jobs
                WHERE job_id=? AND status='running' AND worker_id=?
                """,
                (job_id, worker_id),
            ).fetchone()
            if row is None:
                raise StateConflictError("Job is not owned by this worker.")
            exhausted = int(row["attempt_count"]) >= int(row["max_attempts"])
            status = "dead_letter" if exhausted else "queued"
            connection.execute(
                """
                UPDATE jobs SET status=?, error_json=?, failure_class='transient',
                    worker_id=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
                    available_at=?, completed_at=?, updated_at=?
                WHERE job_id=?
                """,
                (
                    status,
                    self._dumps(error),
                    available_at,
                    now if exhausted else None,
                    now,
                    job_id,
                ),
            )
            retry_request = self._loads(row["request_json"])
            planning_request_id = retry_request.get("planning_request_id")
            if planning_request_id:
                connection.execute(
                    """
                    UPDATE planning_requests SET status=?, updated_at=?
                    WHERE request_id=? AND job_id=?
                    """,
                    (
                        "FAILED" if exhausted else "QUEUED",
                        now,
                        str(planning_request_id),
                        job_id,
                    ),
                )
            if exhausted and retry_request.get("job_type") == "window_replanning":
                window_id = str(retry_request.get("window_id") or "")
                connection.execute(
                    """
                    UPDATE scheduling_windows SET status='failed', result_json=?,
                        updated_at=? WHERE window_id=? AND job_id=?
                    """,
                    (self._dumps({"error": error}), now, window_id, job_id),
                )
                connection.execute(
                    """
                    UPDATE scheduling_window_events SET status='failed', updated_at=?
                    WHERE window_id=?
                    """,
                    (now, window_id),
                )
            connection.execute(
                """
                UPDATE job_attempts SET finished_at=?, outcome=?, error_json=?
                WHERE job_id=? AND attempt_number=? AND finished_at IS NULL
                """,
                (
                    now,
                    "dead_letter" if exhausted else "retry_scheduled",
                    self._dumps(error),
                    job_id,
                    row["attempt_count"],
                ),
            )
        return status

    def list_job_attempts(self, job_id: str) -> list[Dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM job_attempts WHERE job_id=? ORDER BY attempt_number
                """,
                (job_id,),
            ).fetchall()
        values = []
        for row in rows:
            value = dict(row)
            value["error"] = self._loads(value.pop("error_json"))
            values.append(value)
        return values

    def update_job(
        self,
        job_id: str,
        status: str,
        *,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                UPDATE jobs SET status=?, result_json=?, error_json=?,
                    worker_id=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
                    completed_at=?, updated_at=?
                WHERE job_id=?
                """,
                (
                    status,
                    self._dumps(result) if result is not None else None,
                    self._dumps(error) if error is not None else None,
                    _utc_now()
                    if status in {"completed", "failed", "dead_letter"}
                    else None,
                    _utc_now(),
                    job_id,
                ),
            )

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown job_id: {job_id}")
        value = dict(row)
        for key in ("request_json", "result_json", "error_json"):
            value[key.removesuffix("_json")] = self._loads(value.pop(key))
        return value

    def job_counts(self) -> Dict[str, int]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            ).fetchall()
        counts = {
            status: 0
            for status in ("queued", "running", "completed", "failed", "dead_letter")
        }
        counts.update({str(row["status"]): int(row["count"]) for row in rows})
        return counts

    def heartbeat_worker(
        self,
        worker_id: str,
        *,
        status: str,
        current_job_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = _utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO workers(
                    worker_id, status, current_job_id, started_at,
                    heartbeat_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    status=excluded.status,
                    current_job_id=excluded.current_job_id,
                    heartbeat_at=excluded.heartbeat_at,
                    metadata_json=excluded.metadata_json
                """,
                (
                    worker_id,
                    status,
                    current_job_id,
                    now,
                    now,
                    self._dumps(metadata or {}),
                ),
            )

    def worker_status(self, *, stale_after_seconds: float = 90.0) -> Dict[str, int]:
        stale_before = (
            datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)
        ).isoformat()
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT status, heartbeat_at FROM workers"
            ).fetchall()
        active = sum(
            1
            for row in rows
            if row["status"] != "stopped" and row["heartbeat_at"] > stale_before
        )
        return {"registered": len(rows), "active": active, "stale": len(rows) - active}

    def operational_metrics(self) -> Dict[str, Any]:
        with self._connection() as connection:
            plan_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM plans GROUP BY status"
            ).fetchall()
            event_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM events GROUP BY status"
            ).fetchall()
            window_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM scheduling_windows GROUP BY status"
            ).fetchall()
            stream_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM continuous_streams GROUP BY status"
            ).fetchall()
            incomplete_operations = connection.execute(
                "SELECT COUNT(*) AS count FROM operations WHERE status='preparing'"
            ).fetchone()["count"]
        return {
            "schema_version": self.schema_version(),
            "jobs": self.job_counts(),
            "workers": self.worker_status(),
            "plans": {str(row["status"]): int(row["count"]) for row in plan_rows},
            "events": {str(row["status"]): int(row["count"]) for row in event_rows},
            "scheduling_windows": {
                str(row["status"]): int(row["count"]) for row in window_rows
            },
            "continuous_streams": {
                str(row["status"]): int(row["count"]) for row in stream_rows
            },
            "incomplete_operations": int(incomplete_operations),
        }
