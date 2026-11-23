import json
from pathlib import Path

from Agents.integrity_service import ArtifactIntegrityError, verify_plan_artifacts
from Agents.run_manifest import sha256_file
from Agents.state_repository import SQLiteStateRepository
from Agents.storage_health import check_storage_health


def _valid_pending_plan(tmp_path: Path) -> tuple[SQLiteStateRepository, dict, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    solution = run_dir / "loading_solution.json"
    strategy = run_dir / "strategy_plan.json"
    manifest = run_dir / "run_manifest.json"
    solution.write_text(
        json.dumps({"research_system": {"plan": {"status": "pending_approval"}}}),
        encoding="utf-8",
    )
    strategy.write_text(json.dumps({"selected_strategy": "greedy"}), encoding="utf-8")
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
    repository = SQLiteStateRepository(tmp_path / "state.db")
    plan = repository.save_plan_record(
        plan={"plan_id": "plan-1", "version": 1, "status": "pending_approval"},
        run_dir=run_dir,
        solution_path=solution,
        manifest_path=manifest,
        strategy_path=strategy,
    )
    return repository, plan, solution


def test_valid_plan_artifacts_are_verified(tmp_path: Path) -> None:
    _, plan, _ = _valid_pending_plan(tmp_path)

    result = verify_plan_artifacts(plan)

    assert result["valid"]
    assert set(result["verified_artifacts"]) == {
        "loading_solution",
        "strategy_plan",
    }


def test_tampered_solution_cannot_be_approved(tmp_path: Path) -> None:
    repository, _, solution = _valid_pending_plan(tmp_path)
    solution.write_text('{"tampered":true}', encoding="utf-8")

    try:
        repository.approve_plan("plan-1", 1, approved_by="operator")
    except ArtifactIntegrityError as error:
        assert "SHA-256 mismatch" in str(error)
    else:
        raise AssertionError("tampered plan was approved")

    assert repository.get_plan("plan-1")["status"] == "pending_approval"


def test_storage_health_detects_low_space_floor_and_database_corruption(
    tmp_path: Path,
) -> None:
    healthy_root = tmp_path / "healthy"
    database = healthy_root / "state.db"
    SQLiteStateRepository(database)

    low_space = check_storage_health(healthy_root, database, minimum_free_bytes=10**30)

    assert not low_space["healthy"]
    assert low_space["writable"]
    assert low_space["database_integrity"]

    corrupt_root = tmp_path / "corrupt"
    corrupt_root.mkdir()
    corrupt_database = corrupt_root / "state.db"
    corrupt_database.write_bytes(b"not sqlite")
    corrupt = check_storage_health(corrupt_root, corrupt_database, minimum_free_bytes=1)
    assert not corrupt["healthy"]
    assert not corrupt["database_integrity"]
