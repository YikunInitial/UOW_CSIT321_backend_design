import json
import sqlite3
import zipfile
from pathlib import Path

from Agents.operations_service import OperationsService
from Agents.state_repository import CURRENT_SCHEMA_VERSION, SQLiteStateRepository


def test_status_includes_storage_and_repository_health(tmp_path: Path) -> None:
    data_root = tmp_path / "live"
    service = OperationsService(data_root, data_root / "state.db")

    status = service.status()

    assert status["storage"]["healthy"]
    assert status["schema_version"] == CURRENT_SCHEMA_VERSION


def test_online_backup_is_verified_and_restores_state_and_artifacts(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "live"
    repository = SQLiteStateRepository(data_root / "state.db")
    repository.create_job("job-1", {"job_type": "planning"})
    artifact = data_root / "runs" / "plan-1" / "loading_solution.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"status":"valid"}', encoding="utf-8")
    service = OperationsService(data_root, data_root / "state.db")
    archive = tmp_path / "backups" / "steel-backup.zip"

    created = service.create_backup(archive)

    assert created["valid"]
    restored_root = tmp_path / "restored"
    restored = OperationsService.restore_backup(archive, restored_root)
    assert restored["restored"]
    assert (
        SQLiteStateRepository(restored_root / "state.db").get_job("job-1")["status"]
        == "queued"
    )
    assert (restored_root / "runs" / "plan-1" / "loading_solution.json").read_text(
        encoding="utf-8"
    ) == '{"status":"valid"}'


def test_backup_without_manifest_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "invalid.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("state/state.db", b"not-a-database")

    result = OperationsService.verify_backup(archive)

    assert not result["valid"]
    assert "backup_manifest.json is missing" in result["errors"]


def test_backup_hash_verification_detects_tampering(tmp_path: Path) -> None:
    data_root = tmp_path / "live"
    SQLiteStateRepository(data_root / "state.db").create_job(
        "job-1", {"job_type": "test"}
    )
    archive = tmp_path / "backup.zip"
    OperationsService(data_root, data_root / "state.db").create_backup(archive)
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr("state/state.db", b"tampered")

    result = OperationsService.verify_backup(archive)

    assert not result["valid"]
    assert any("state/state.db" in error for error in result["errors"])


def test_retention_is_dry_run_by_default_and_only_prunes_safe_superseded_runs(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "live"
    run_dir = data_root / "runs" / "old-plan"
    run_dir.mkdir(parents=True)
    solution = run_dir / "loading_solution.json"
    manifest = run_dir / "run_manifest.json"
    strategy = run_dir / "strategy_plan.json"
    for path in (solution, manifest, strategy):
        path.write_text("{}", encoding="utf-8")
    repository = SQLiteStateRepository(data_root / "state.db")
    repository.save_plan_record(
        plan={"plan_id": "old-plan", "version": 1, "status": "superseded"},
        run_dir=run_dir,
        solution_path=solution,
        manifest_path=manifest,
        strategy_path=strategy,
    )
    with sqlite3.connect(data_root / "state.db") as connection:
        connection.execute(
            "UPDATE plans SET updated_at='2020-01-01T00:00:00+00:00' "
            "WHERE plan_id='old-plan'"
        )
    service = OperationsService(data_root, data_root / "state.db")

    preview = service.prune_superseded_artifacts(retention_days=90)

    assert preview["candidate_count"] == 1
    assert preview["deleted_count"] == 0
    assert run_dir.exists()
    applied = service.prune_superseded_artifacts(retention_days=90, apply=True)
    assert applied["deleted_count"] == 1
    assert not run_dir.exists()
    audit_records = [
        json.loads(line)
        for line in (data_root / "operations_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert audit_records[-1]["action"] == "retention_prune"
