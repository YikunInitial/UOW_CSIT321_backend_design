"""Backup, restore, retention, and status services for single-site delivery."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

from Agents.state_repository import SQLiteStateRepository
from Agents.storage_health import check_storage_health


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OperationsService:
    def __init__(self, data_root: str | Path, database_path: str | Path) -> None:
        self.data_root = Path(data_root).resolve()
        self.database_path = Path(database_path).resolve()
        if self.database_path.parent != self.data_root:
            raise ValueError("database_path must be directly inside data_root")
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.repository = SQLiteStateRepository(self.database_path)

    def status(self) -> Dict[str, Any]:
        return {
            "checked_at": _utc_now(),
            "data_root": str(self.data_root),
            "database_path": str(self.database_path),
            "storage": check_storage_health(
                self.data_root,
                self.database_path,
                minimum_free_bytes=0,
            ),
            **self.repository.operational_metrics(),
        }

    def _audit(self, action: str, details: Dict[str, Any]) -> None:
        record = {"timestamp": _utc_now(), "action": action, **details}
        audit_path = self.data_root / "operations_audit.jsonl"
        with audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def create_backup(self, destination: str | Path) -> Dict[str, Any]:
        destination_path = Path(destination).resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_archive = destination_path.with_suffix(
            destination_path.suffix + ".tmp"
        )
        if temporary_archive.exists():
            temporary_archive.unlink()
        with tempfile.TemporaryDirectory(prefix="steel-backup-") as temporary:
            staging = Path(temporary)
            database_snapshot = staging / "state.db"
            source = sqlite3.connect(self.database_path)
            target = sqlite3.connect(database_snapshot)
            try:
                source.backup(target)
            finally:
                target.close()
                source.close()
            files: list[Dict[str, Any]] = []
            with zipfile.ZipFile(
                temporary_archive, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.write(database_snapshot, "state/state.db")
                files.append(
                    {
                        "path": "state/state.db",
                        "size": database_snapshot.stat().st_size,
                        "sha256": _sha256(database_snapshot),
                    }
                )
                excluded = {
                    self.database_path,
                    Path(str(self.database_path) + "-wal"),
                    Path(str(self.database_path) + "-shm"),
                    destination_path,
                    temporary_archive,
                }
                for path in sorted(self.data_root.rglob("*")):
                    resolved = path.resolve()
                    if not path.is_file() or resolved in excluded:
                        continue
                    relative = path.relative_to(self.data_root)
                    archive_name = (Path("artifacts") / relative).as_posix()
                    archive.write(path, archive_name)
                    files.append(
                        {
                            "path": archive_name,
                            "size": path.stat().st_size,
                            "sha256": _sha256(path),
                        }
                    )
                manifest = {
                    "format_version": 1,
                    "created_at": _utc_now(),
                    "schema_version": self.repository.schema_version(),
                    "files": files,
                }
                archive.writestr(
                    "backup_manifest.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                )
        temporary_archive.replace(destination_path)
        verification = self.verify_backup(destination_path)
        self._audit(
            "backup_created",
            {"backup_path": str(destination_path), "file_count": len(files)},
        )
        return verification

    @staticmethod
    def verify_backup(archive_path: str | Path) -> Dict[str, Any]:
        path = Path(archive_path).resolve()
        errors: list[str] = []
        with zipfile.ZipFile(path, "r") as archive:
            try:
                manifest = json.loads(archive.read("backup_manifest.json"))
            except KeyError:
                return {"valid": False, "errors": ["backup_manifest.json is missing"]}
            names = set(archive.namelist())
            for item in manifest.get("files", []):
                name = str(item["path"])
                if name not in names:
                    errors.append(f"missing file: {name}")
                    continue
                content = archive.read(name)
                digest = hashlib.sha256(content).hexdigest()
                if digest != item["sha256"]:
                    errors.append(f"hash mismatch: {name}")
                if len(content) != int(item["size"]):
                    errors.append(f"size mismatch: {name}")
            if "state/state.db" in names:
                with tempfile.TemporaryDirectory(prefix="steel-verify-") as temporary:
                    database = Path(temporary) / "state.db"
                    database.write_bytes(archive.read("state/state.db"))
                    connection = sqlite3.connect(database)
                    try:
                        try:
                            integrity = connection.execute(
                                "PRAGMA integrity_check"
                            ).fetchone()[0]
                        except sqlite3.Error as error:
                            integrity = f"unreadable database: {type(error).__name__}"
                    finally:
                        connection.close()
                    if integrity != "ok":
                        errors.append(f"database integrity check: {integrity}")
            else:
                errors.append("state/state.db is missing")
        return {
            "valid": not errors,
            "archive_path": str(path),
            "errors": errors,
            "manifest": manifest,
        }

    @staticmethod
    def restore_backup(
        archive_path: str | Path, target_root: str | Path
    ) -> Dict[str, Any]:
        verification = OperationsService.verify_backup(archive_path)
        if not verification["valid"]:
            raise ValueError(f"Backup verification failed: {verification['errors']}")
        target = Path(target_root).resolve()
        if target.exists() and any(target.iterdir()):
            raise ValueError("Restore target must be empty.")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(Path(archive_path).resolve(), "r") as archive:
            for item in verification["manifest"]["files"]:
                archive_name = str(item["path"])
                if archive_name == "state/state.db":
                    relative = Path("state.db")
                elif archive_name.startswith("artifacts/"):
                    relative = Path(archive_name.removeprefix("artifacts/"))
                else:
                    raise ValueError(f"Unsafe backup entry: {archive_name}")
                destination = (target / relative).resolve()
                if target not in destination.parents:
                    raise ValueError(
                        f"Backup entry escapes restore target: {archive_name}"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(archive_name))
        restored_repository = SQLiteStateRepository(target / "state.db")
        return {
            "restored": True,
            "target_root": str(target),
            "schema_version": restored_repository.schema_version(),
            "file_count": len(verification["manifest"]["files"]),
        }

    def prune_superseded_artifacts(
        self, *, retention_days: int, apply: bool = False
    ) -> Dict[str, Any]:
        if retention_days < 1:
            raise ValueError("retention_days must be at least 1")
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        rows = self.repository.list_plans_by_status("superseded")
        candidates: list[Dict[str, Any]] = []
        for row in rows:
            updated_at = datetime.fromisoformat(row["updated_at"])
            run_dir = Path(row["run_dir"]).resolve()
            safe = run_dir != self.data_root and self.data_root in run_dir.parents
            if updated_at <= cutoff and safe and run_dir.exists():
                candidates.append(
                    {
                        "plan_id": row["plan_id"],
                        "run_dir": str(run_dir),
                        "updated_at": row["updated_at"],
                    }
                )
        deleted = 0
        if apply:
            for candidate in candidates:
                run_dir = Path(candidate["run_dir"]).resolve()
                if run_dir != self.data_root and self.data_root in run_dir.parents:
                    shutil.rmtree(run_dir)
                    deleted += 1
            self._audit(
                "retention_prune",
                {"retention_days": retention_days, "deleted": deleted},
            )
        return {
            "dry_run": not apply,
            "retention_days": retention_days,
            "candidate_count": len(candidates),
            "deleted_count": deleted,
            "candidates": candidates,
        }
