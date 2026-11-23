import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from starlette.testclient import TestClient

import backend_api
from Agents.app_settings import APISettings
from Agents.state_repository import CURRENT_SCHEMA_VERSION, SQLiteStateRepository
from worker import JobWorker


def _expire_lease(database_path: Path, job_id: str) -> None:
    expired = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at=? WHERE job_id=?", (expired, job_id)
        )


def test_legacy_job_schema_is_migrated_without_losing_jobs(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT,
                error_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO jobs(job_id, status, request_json, created_at, updated_at)
            VALUES ('legacy-job', 'queued', '{}', '2026-01-01', '2026-01-01')
            """
        )

    repository = SQLiteStateRepository(database_path)

    assert repository.schema_version() == CURRENT_SCHEMA_VERSION
    job = repository.get_job("legacy-job")
    assert job["status"] == "queued"
    assert job["attempt_count"] == 0
    assert job["max_attempts"] == 3


def test_job_lease_is_exclusive_and_expired_work_is_requeued(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    repository = SQLiteStateRepository(database_path)
    repository.create_job("job-1", {"job_type": "planning"}, max_attempts=2)

    claimed = repository.claim_next_job("worker-a", lease_seconds=30)

    assert claimed is not None
    assert claimed["job_id"] == "job-1"
    assert claimed["attempt_count"] == 1
    assert repository.claim_next_job("worker-b", lease_seconds=30) is None
    assert repository.heartbeat_job("job-1", "worker-a", lease_seconds=30)
    assert not repository.heartbeat_job("job-1", "worker-b", lease_seconds=30)

    _expire_lease(database_path, "job-1")
    assert repository.recover_expired_jobs() == {"requeued": 1, "dead_lettered": 0}
    reclaimed = repository.claim_next_job("worker-b", lease_seconds=30)
    assert reclaimed is not None
    assert reclaimed["attempt_count"] == 2
    assert reclaimed["worker_id"] == "worker-b"


def test_expired_job_fails_after_retry_limit(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    repository = SQLiteStateRepository(database_path)
    repository.create_job("job-1", {"job_type": "planning"}, max_attempts=1)
    assert repository.claim_next_job("worker-a", lease_seconds=1) is not None

    _expire_lease(database_path, "job-1")
    assert repository.recover_expired_jobs() == {"requeued": 0, "dead_lettered": 1}

    job = repository.get_job("job-1")
    assert job["status"] == "dead_letter"
    assert job["error"]["code"] == "JOB_LEASE_EXHAUSTED"


def test_terminal_job_update_releases_worker_lease(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    repository.create_job("job-1", {"job_type": "planning"})
    assert repository.claim_next_job("worker-a") is not None

    repository.update_job("job-1", "completed", result={"plan_id": "plan-1"})

    job = repository.get_job("job-1")
    assert job["status"] == "completed"
    assert job["worker_id"] is None
    assert job["lease_expires_at"] is None
    assert job["result"] == {"plan_id": "plan-1"}


def test_planning_request_snapshot_is_immutable_and_tracks_job_lifecycle(
    tmp_path: Path,
) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    intake_dir = tmp_path / "intake"
    intake_dir.mkdir()
    snapshot = intake_dir / "confirmed_products.json"
    snapshot.write_text('[{"material_no":"A"}]', encoding="utf-8")

    created = repository.create_planning_request(
        "intake_" + "a" * 32,
        intake_dir=intake_dir,
        payload={"status": "NEEDS_REVIEW", "message": "review"},
    )
    assert created["version"] == 1
    reviewed = repository.review_planning_request(
        created["request_id"],
        payload={"status": "READY_TO_PLAN", "message": "ready"},
        accepted_repair_ids=["repair-1"],
    )
    assert reviewed["version"] == 2

    confirmed = repository.confirm_planning_request(
        created["request_id"],
        accepted_repair_ids=["repair-1"],
        snapshot_path=snapshot,
        snapshot_sha256="a" * 64,
        confirmed_by="operator-a",
        payload={"status": "CONFIRMED", "message": "confirmed"},
    )
    assert confirmed["status"] == "CONFIRMED"
    assert confirmed["version"] == 3

    replayed = repository.confirm_planning_request(
        created["request_id"],
        accepted_repair_ids=["repair-1"],
        snapshot_path=snapshot,
        snapshot_sha256="a" * 64,
        confirmed_by="operator-a",
        payload={"status": "CONFIRMED"},
    )
    assert replayed["version"] == 3

    request_payload = {
        "job_type": "planning",
        "planning_request_id": created["request_id"],
    }
    stored_job_id, was_created = repository.enqueue_job(
        "job-1",
        request_payload,
        planning_request_id=created["request_id"],
        idempotency_key="planning-operation-1",
        endpoint="/v2/planning-requests/{request_id}/submit",
        request_hash="hash-1",
    )
    assert was_created
    assert stored_job_id == "job-1"
    assert repository.get_planning_request(created["request_id"])["status"] == "QUEUED"

    replay_job_id, was_created = repository.enqueue_job(
        "ignored-job-id",
        request_payload,
        planning_request_id=created["request_id"],
        idempotency_key="planning-operation-1",
        endpoint="/v2/planning-requests/{request_id}/submit",
        request_hash="hash-1",
    )
    assert replay_job_id == "job-1"
    assert not was_created

    assert repository.claim_next_job("worker-a") is not None
    assert repository.get_planning_request(created["request_id"])["status"] == "RUNNING"
    repository.complete_job("job-1", "worker-a", {"ok": True})
    final = repository.get_planning_request(created["request_id"])
    assert final["status"] == "COMPLETED"
    assert final["job_id"] == "job-1"


def test_confirmed_planning_request_cannot_be_repaired_again(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    intake_dir = tmp_path / "intake"
    intake_dir.mkdir()
    snapshot = intake_dir / "confirmed_products.json"
    snapshot.write_text("[]", encoding="utf-8")
    request_id = "intake_" + "b" * 32
    repository.create_planning_request(
        request_id, intake_dir=intake_dir, payload={"status": "READY_TO_PLAN"}
    )
    repository.confirm_planning_request(
        request_id,
        accepted_repair_ids=[],
        snapshot_path=snapshot,
        snapshot_sha256="b" * 64,
        confirmed_by="operator-a",
        payload={"status": "CONFIRMED"},
    )

    try:
        repository.review_planning_request(
            request_id,
            payload={"status": "READY_TO_PLAN"},
            accepted_repair_ids=["different"],
        )
    except Exception as error:
        assert isinstance(error, backend_api.StateConflictError)
    else:
        raise AssertionError("Confirmed request unexpectedly accepted another review")


def test_failed_planning_request_can_queue_one_explicit_new_attempt(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    intake_dir = tmp_path / "intake"
    intake_dir.mkdir()
    snapshot = intake_dir / "products.json"
    snapshot.write_text("[]", encoding="utf-8")
    request_id = "intake_" + "c" * 32
    repository.create_planning_request(
        request_id, intake_dir=intake_dir, payload={"status": "READY_TO_PLAN"}
    )
    repository.confirm_planning_request(
        request_id,
        accepted_repair_ids=[],
        snapshot_path=snapshot,
        snapshot_sha256="c" * 64,
        confirmed_by="operator-a",
        payload={"status": "CONFIRMED"},
    )
    request = {"job_type": "planning", "planning_request_id": request_id}
    repository.enqueue_job("job-old", request, planning_request_id=request_id)
    repository.claim_next_job("worker-a")
    repository.fail_job("job-old", "worker-a", {"message": "invalid candidate"})

    job_id, created = repository.enqueue_job(
        "job-new",
        request,
        planning_request_id=request_id,
        idempotency_key="planning-retry-operation",
        endpoint="/v2/planning-requests/{request_id}/submit",
        request_hash="retry-hash",
    )

    assert created
    assert job_id == "job-new"
    durable = repository.get_planning_request(request_id)
    assert durable["status"] == "QUEUED"
    assert durable["job_id"] == "job-new"


def test_incomplete_approval_operation_is_safely_replayed(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    solution_path = run_dir / "loading_solution.json"
    manifest_path = run_dir / "run_manifest.json"
    strategy_path = run_dir / "strategy_plan.json"
    solution_path.write_text(
        json.dumps({"research_system": {"plan": {"status": "pending_approval"}}}),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {"artifacts": [{"name": "loading_solution", "sha256": "before-recovery"}]}
        ),
        encoding="utf-8",
    )
    strategy_path.write_text("{}", encoding="utf-8")
    repository = SQLiteStateRepository(database_path)
    repository.save_plan_record(
        plan={"plan_id": "plan-1", "version": 1, "status": "pending_approval"},
        run_dir=run_dir,
        solution_path=solution_path,
        manifest_path=manifest_path,
        strategy_path=strategy_path,
    )
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO operations(
                operation_id, operation_type, entity_id, status,
                payload_json, created_at, updated_at
            ) VALUES (?, 'approve_plan', ?, 'preparing', ?, ?, ?)
            """,
            (
                "approve:plan-1:v1",
                "plan-1",
                json.dumps({"plan_id": "plan-1", "expected_version": 1}),
                now,
                now,
            ),
        )

    assert repository.recover_incomplete_operations() == {
        "recovered": 1,
        "failed": 0,
    }
    assert repository.get_plan("plan-1")["status"] == "active"
    solution = json.loads(solution_path.read_text(encoding="utf-8"))
    assert solution["research_system"]["plan"]["status"] == "active"
    with sqlite3.connect(database_path) as connection:
        status = connection.execute(
            "SELECT status FROM operations WHERE operation_id='approve:plan-1:v1'"
        ).fetchone()[0]
    assert status == "completed"


def test_pending_successor_can_be_rejected_without_deleting_evidence(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    solution_path = run_dir / "loading_solution.json"
    manifest_path = run_dir / "run_manifest.json"
    strategy_path = run_dir / "strategy_plan.json"
    solution_path.write_text("{}", encoding="utf-8")
    manifest_path.write_text("{}", encoding="utf-8")
    strategy_path.write_text("{}", encoding="utf-8")
    repository = SQLiteStateRepository(database_path)
    repository.save_plan_record(
        plan={
            "plan_id": "plan-successor",
            "version": 2,
            "status": "pending_approval",
            "parent_plan_id": "plan-parent",
        },
        run_dir=run_dir,
        solution_path=solution_path,
        manifest_path=manifest_path,
        strategy_path=strategy_path,
    )

    rejected = repository.reject_plan(
        "plan-successor",
        2,
        rejected_by="operator-a",
        reason="Operational review rejected the change",
    )

    assert rejected["status"] == "rejected"
    assert rejected["parent_plan_id"] == "plan-parent"
    assert solution_path.exists()
    assert manifest_path.exists()
    with sqlite3.connect(database_path) as connection:
        operation = connection.execute(
            "SELECT status, payload_json FROM operations WHERE operation_id=?",
            ("reject:plan-successor:v2",),
        ).fetchone()
    assert operation[0] == "completed"
    assert json.loads(operation[1])["reason"] == "Operational review rejected the change"
    disposition = json.loads((run_dir / "plan_disposition.json").read_text(encoding="utf-8"))
    assert disposition["status"] == "rejected"


def test_interrupted_rejection_is_recovered_with_matching_evidence(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in ("loading_solution.json", "run_manifest.json", "strategy_plan.json"):
        (run_dir / name).write_text("{}", encoding="utf-8")
    repository = SQLiteStateRepository(database_path)
    repository.save_plan_record(
        plan={"plan_id": "plan-recovery", "version": 2, "status": "pending_approval"},
        run_dir=run_dir,
        solution_path=run_dir / "loading_solution.json",
        manifest_path=run_dir / "run_manifest.json",
        strategy_path=run_dir / "strategy_plan.json",
    )
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "plan_id": "plan-recovery",
        "expected_version": 2,
        "rejected_by": "operator-a",
        "reason": "Unsafe operational change",
        "run_dir": str(run_dir),
    }
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO operations(
                operation_id, operation_type, entity_id, status,
                payload_json, created_at, updated_at
            ) VALUES (?, 'reject_plan', ?, 'preparing', ?, ?, ?)
            """,
            ("reject:plan-recovery:v2", "plan-recovery", json.dumps(payload), now, now),
        )

    assert repository.recover_incomplete_operations() == {"recovered": 1, "failed": 0}
    assert repository.get_plan("plan-recovery")["status"] == "rejected"
    disposition = json.loads((run_dir / "plan_disposition.json").read_text(encoding="utf-8"))
    assert disposition["reason"] == "Unsafe operational change"
    assert repository.get_operation("reject:plan-recovery:v2")["status"] == "completed"


def test_worker_claims_executes_and_completes_durable_job(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    repository.create_job("job-1", {"job_type": "test"})
    worker = JobWorker(repository, worker_id="test-worker", lease_seconds=30)

    class FakeExecutor:
        @staticmethod
        def execute(request: dict) -> dict:
            return {"processed": request["job_type"]}

    worker.executor = FakeExecutor()

    assert worker.run_once()
    job = repository.get_job("job-1")
    assert job["status"] == "completed"
    assert job["result"] == {"processed": "test"}
    assert repository.worker_status()["active"] == 1


def test_worker_records_execution_failure_and_releases_lease(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    repository.create_job("job-1", {"job_type": "test"})
    worker = JobWorker(repository, worker_id="test-worker", lease_seconds=30)

    class FailingExecutor:
        @staticmethod
        def execute(request: dict) -> dict:
            raise RuntimeError(f"cannot process {request['job_type']}")

    worker.executor = FailingExecutor()

    assert worker.run_once()
    job = repository.get_job("job-1")
    assert job["status"] == "failed"
    assert job["worker_id"] is None
    assert job["error"]["type"] == "RuntimeError"


def test_api_enqueues_durable_work_without_executing_in_request(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = tmp_path / "api"
    database_path = run_root / "state.db"
    monkeypatch.setattr(backend_api, "API_RUN_ROOT", run_root)
    monkeypatch.setattr(backend_api, "API_DATABASE", database_path)
    monkeypatch.setattr(backend_api, "EMBEDDED_WORKER_ENABLED", False)

    with TestClient(backend_api.app) as client:
        response = client.post(
            "/v1/planning-runs",
            json={
                "products": [
                    {
                        "material_no": "A",
                        "destination": "Sydney",
                        "width": 1000,
                        "outer_diameter": 1200,
                        "gross_weight_tonne": 10,
                    }
                ]
            },
        )

    assert response.status_code == 202
    job = SQLiteStateRepository(database_path).get_job(response.json()["job_id"])
    assert job["status"] == "queued"
    assert job["request"]["job_type"] == "planning"
    assert Path(job["request"]["run_dir"]).parent == run_root


def test_production_configuration_fails_closed_without_security_controls(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = tmp_path / "data"
    monkeypatch.setenv("STEEL_ENV", "production")
    monkeypatch.setenv("STEEL_API_RUN_ROOT", str(run_root))
    monkeypatch.setenv("STEEL_API_STATE_DATABASE", str(run_root / "state.db"))
    monkeypatch.setenv("STEEL_API_KEY", "short")
    monkeypatch.setenv("STEEL_API_EMBEDDED_WORKER", "false")
    try:
        APISettings.from_environment(tmp_path)
    except ValueError as error:
        assert "at least 32" in str(error)
    else:
        raise AssertionError("insecure production configuration was accepted")


def test_production_configuration_disables_demo_data_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = tmp_path / "data"
    monkeypatch.setenv("STEEL_ENV", "production")
    monkeypatch.setenv("STEEL_API_RUN_ROOT", str(run_root))
    monkeypatch.setenv("STEEL_API_STATE_DATABASE", str(run_root / "state.db"))
    monkeypatch.setenv("STEEL_API_KEY", "p" * 32)
    monkeypatch.setenv("STEEL_API_EMBEDDED_WORKER", "false")
    monkeypatch.delenv("STEEL_ALLOW_SAMPLE_DATA", raising=False)
    monkeypatch.delenv("STEEL_ALLOW_SIMULATION_EVENTS", raising=False)

    settings = APISettings.from_environment(tmp_path)

    assert settings.allow_sample_data is False
    assert settings.allow_simulation_events is False


def test_production_configuration_rejects_demo_data_override(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = tmp_path / "data"
    monkeypatch.setenv("STEEL_ENV", "production")
    monkeypatch.setenv("STEEL_API_RUN_ROOT", str(run_root))
    monkeypatch.setenv("STEEL_API_STATE_DATABASE", str(run_root / "state.db"))
    monkeypatch.setenv("STEEL_API_KEY", "p" * 32)
    monkeypatch.setenv("STEEL_API_EMBEDDED_WORKER", "false")
    monkeypatch.setenv("STEEL_ALLOW_SIMULATION_EVENTS", "true")

    try:
        APISettings.from_environment(tmp_path)
    except ValueError as error:
        assert "forbids sample data" in str(error)
    else:
        raise AssertionError("production accepted simulated dynamic events")


def test_api_enforces_authentication_body_limit_and_request_id(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = tmp_path / "api"
    monkeypatch.setattr(backend_api, "API_RUN_ROOT", run_root)
    monkeypatch.setattr(backend_api, "API_DATABASE", run_root / "state.db")
    monkeypatch.setattr(backend_api, "EMBEDDED_WORKER_ENABLED", False)
    monkeypatch.setattr(backend_api, "API_KEY", "test-secret")
    monkeypatch.setattr(
        backend_api, "SETTINGS", replace(backend_api.SETTINGS, max_body_bytes=64)
    )

    with TestClient(backend_api.app) as client:
        unauthorised = client.get("/metrics")
        oversized = client.post(
            "/v1/planning-runs",
            headers={"x-api-key": "test-secret", "x-request-id": "qa-request-1"},
            content=b"{" + (b"x" * 100) + b"}",
        )

    assert unauthorised.status_code == 401
    assert oversized.status_code == 413
    assert oversized.headers["x-request-id"] == "qa-request-1"


def test_api_contract_rejects_unknown_fields(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "api"
    monkeypatch.setattr(backend_api, "API_RUN_ROOT", run_root)
    monkeypatch.setattr(backend_api, "API_DATABASE", run_root / "state.db")
    monkeypatch.setattr(backend_api, "EMBEDDED_WORKER_ENABLED", False)
    monkeypatch.setattr(backend_api, "API_KEY", "")

    with TestClient(backend_api.app) as client:
        response = client.post(
            "/v1/planning-runs",
            json={"products": [{}], "unexpected_control": True},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_api_rate_limit_is_enforced(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "api"
    monkeypatch.setattr(backend_api, "API_RUN_ROOT", run_root)
    monkeypatch.setattr(backend_api, "API_DATABASE", run_root / "state.db")
    monkeypatch.setattr(backend_api, "EMBEDDED_WORKER_ENABLED", False)
    monkeypatch.setattr(backend_api, "API_KEY", "")
    monkeypatch.setattr(
        backend_api, "SETTINGS", replace(backend_api.SETTINGS, rate_limit_per_minute=1)
    )
    backend_api._RATE_LIMIT_WINDOWS.clear()

    with TestClient(backend_api.app) as client:
        first = client.get("/openapi.json")
        second = client.get("/openapi.json")
        third = client.get("/openapi.json")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert third.json()["error"]["code"] == "RATE_LIMITED"
