import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from Agents.run_manifest import sha256_file
from Agents.state_repository import SQLiteStateRepository, StateConflictError


def _create_pending_plan(data_root: Path) -> tuple[Path, Path]:
    database_path = data_root / "state.db"
    run_dir = data_root / "runs" / "plan-1"
    run_dir.mkdir(parents=True)
    solution = run_dir / "loading_solution.json"
    manifest = run_dir / "run_manifest.json"
    strategy = run_dir / "strategy_plan.json"
    solution.write_text(
        json.dumps({"research_system": {"plan": {"status": "pending_approval"}}}),
        encoding="utf-8",
    )
    strategy.write_text("{}", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "name": "loading_solution",
                        "path": str(solution),
                        "sha256": sha256_file(solution),
                    },
                    {
                        "name": "strategy_plan",
                        "path": str(strategy),
                        "sha256": sha256_file(strategy),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    SQLiteStateRepository(database_path).save_plan_record(
        plan={"plan_id": "plan-1", "version": 1, "status": "pending_approval"},
        run_dir=run_dir,
        solution_path=solution,
        manifest_path=manifest,
        strategy_path=strategy,
    )
    return database_path, manifest


def test_competing_repository_instances_claim_job_exactly_once(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    SQLiteStateRepository(database_path).create_job("job-1", {"job_type": "test"})

    def claim(worker_number: int):
        return SQLiteStateRepository(database_path).claim_next_job(
            f"worker-{worker_number}", lease_seconds=30
        )

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(claim, range(64)))

    claims = [result for result in results if result is not None]
    assert len(claims) == 1
    assert claims[0]["job_id"] == "job-1"
    assert claims[0]["attempt_count"] == 1


def test_concurrent_approval_is_idempotent_and_preserves_one_identity(
    tmp_path: Path,
) -> None:
    database_path, manifest_path = _create_pending_plan(tmp_path)

    def approve(operator: str):
        return SQLiteStateRepository(database_path).approve_plan(
            "plan-1", 1, approved_by=operator
        )

    operators = [f"operator-{index}" for index in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(approve, operators))

    assert all(result["status"] == "active" for result in results)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_operator = manifest["governance"]["approval"]["approved_by"]
    assert recorded_operator in operators
    operation = SQLiteStateRepository(database_path).get_operation("approve:plan-1:v1")
    assert operation["status"] == "completed"
    assert operation["payload"]["approved_by"] == recorded_operator


def test_duplicate_event_is_accepted_only_once_under_concurrency(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.db"
    event = {"event_id": "EVENT-1", "expected_plan_id": "PLAN-1"}

    def reserve(_: int) -> bool:
        try:
            SQLiteStateRepository(database_path).reserve_event(event)
        except StateConflictError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=12) as pool:
        accepted = list(pool.map(reserve, range(24)))

    assert sum(accepted) == 1


def test_burst_job_ingestion_preserves_every_unique_request(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"

    def enqueue(index: int) -> None:
        SQLiteStateRepository(database_path).create_job(
            f"job-{index:03d}", {"job_type": "test", "index": index}
        )

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(enqueue, range(500)))

    counts = SQLiteStateRepository(database_path).job_counts()
    assert counts["queued"] == 500
    assert sum(counts.values()) == 500
