"""Local durable-storage readiness checks."""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict


def check_storage_health(
    data_root: str | Path,
    database_path: str | Path,
    *,
    minimum_free_bytes: int,
) -> Dict[str, Any]:
    root = Path(data_root).resolve()
    database = Path(database_path).resolve()
    report: Dict[str, Any] = {
        "healthy": False,
        "writable": False,
        "database_integrity": False,
        "minimum_free_bytes": int(minimum_free_bytes),
        "free_bytes": 0,
        "errors": [],
    }
    try:
        root.mkdir(parents=True, exist_ok=True)
        report["free_bytes"] = shutil.disk_usage(root).free
        if report["free_bytes"] < minimum_free_bytes:
            report["errors"].append(
                "available disk space is below the configured floor"
            )
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".storage-probe-", dir=root, delete=True
        ) as probe:
            probe.write(b"steel-loading-storage-probe")
            probe.flush()
            os.fsync(probe.fileno())
        report["writable"] = True
    except OSError:
        report["errors"].append("data root is not durably writable")

    if database.is_file():
        try:
            connection = sqlite3.connect(
                f"file:{database.as_posix()}?mode=ro", uri=True, timeout=5
            )
            try:
                integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
            finally:
                connection.close()
            report["database_integrity"] = integrity == "ok"
            if integrity != "ok":
                report["errors"].append("database quick-check did not return ok")
        except sqlite3.Error:
            report["errors"].append("database quick-check could not run")
    else:
        report["errors"].append("state database is missing")

    report["healthy"] = (
        report["writable"]
        and report["database_integrity"]
        and report["free_bytes"] >= minimum_free_bytes
    )
    return report
