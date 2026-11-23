import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from Agents.planning_service import PlanningService
from Agents.replanning_service import ReplanningCommand, ReplanningService
from Agents.state_repository import SQLiteStateRepository


def _save_plan(
    repository: SQLiteStateRepository,
    run_dir: Path,
    *,
    plan_id: str,
    version: int,
    status: str,
    parent_plan_id: str | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: run_dir / filename
        for name, filename in {
            "solution": "loading_solution.json",
            "manifest": "run_manifest.json",
            "strategy": "strategy_plan.json",
        }.items()
    }
    for path in paths.values():
        path.write_text("{}", encoding="utf-8")
    repository.save_plan_record(
        plan={
            "plan_id": plan_id,
            "version": version,
            "status": status,
            "parent_plan_id": parent_plan_id,
        },
        run_dir=run_dir,
        solution_path=paths["solution"],
        manifest_path=paths["manifest"],
        strategy_path=paths["strategy"],
    )


def _event() -> dict:
    return {
        "event_id": "event-1",
        "event_type": "Add products",
        "product_count": 1,
        "target_destination": None,
        "disabled_wagon_id": None,
        "expected_plan_id": "parent-plan",
        "expected_plan_version": 1,
    }


def test_retry_recovers_already_persisted_successor_without_duplicate_plan(
    tmp_path: Path,
) -> None:
    # Keep the nested recovery artifact path below the Windows legacy limit.
    tmp_path = Path(tempfile.mkdtemp(dir=Path("runtime") / "test_tmp"))
    repository = SQLiteStateRepository(tmp_path / "state.db")
    _save_plan(
        repository,
        tmp_path / "parent",
        plan_id="parent-plan",
        version=1,
        status="active",
    )
    _save_plan(
        repository,
        tmp_path / "parent" / "event_event-1",
        plan_id="child-plan",
        version=2,
        status="pending_approval",
        parent_plan_id="parent-plan",
    )
    repository.reserve_event(_event())
    repository.fail_event("event-1", {"type": "ProcessExit"})

    result = ReplanningService(repository, PlanningService(repository)).replan(
        ReplanningCommand(
            plan_id="parent-plan",
            expected_version=1,
            event_type="Add products",
            event_id="event-1",
        )
    )

    assert result["plan"]["plan_id"] == "child-plan"
    assert result["recovery_mode"] == "recovered_completed_successor"
    assert repository.get_event("event-1")["status"] == "completed"
    assert len(repository.list_plans_by_status("pending_approval")) == 1


def test_replanning_event_directories_are_flat_and_sanitised(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    service = ReplanningService(repository, PlanningService(repository))

    first = service._successor_run_dir("event_1234")
    second = service._successor_run_dir("event_../../unsafe child")

    assert first.parent == tmp_path / "events"
    assert first.name.startswith("e_1234_")
    assert second.parent == tmp_path / "events"
    assert second.name.startswith("e_unsafe_chi_")
    assert len(first.name) <= 23
    assert len(second.name) <= 23
    assert "event_event" not in str(first)


def test_expired_replanning_lease_marks_processing_event_retryable(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.db"
    repository = SQLiteStateRepository(database_path)
    repository.reserve_event(_event())
    repository.create_job(
        "job-1",
        {"job_type": "replanning", "event_id": "event-1"},
        max_attempts=2,
    )
    assert repository.claim_next_job("worker-1", lease_seconds=1) is not None
    expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at=? WHERE job_id='job-1'", (expired,)
        )

    recovery = repository.recover_expired_jobs()

    assert recovery == {"requeued": 1, "dead_lettered": 0}
    assert repository.get_event("event-1")["status"] == "failed"
    assert repository.get_job("job-1")["status"] == "queued"


def test_startup_recovery_repairs_running_job_with_missing_lease(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.db"
    repository = SQLiteStateRepository(database_path)
    repository.create_job("job-1", {"job_type": "planning"}, max_attempts=2)
    assert repository.claim_next_job("worker-1") is not None
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE jobs SET lease_expires_at=NULL WHERE job_id='job-1'")

    recovery = repository.recover_expired_jobs()

    assert recovery == {"requeued": 1, "dead_lettered": 0}
    assert repository.get_job("job-1")["status"] == "queued"


def test_startup_recovery_marks_unowned_processing_event_failed(
    tmp_path: Path,
) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    repository.reserve_event(_event())

    recovery = repository.recover_orphaned_events()

    assert recovery == {"failed_orphans": 1}
    event = repository.get_event("event-1")
    assert event["status"] == "failed"
    assert event["result"]["error"]["code"] == "ORPHANED_EVENT_RECOVERED"
