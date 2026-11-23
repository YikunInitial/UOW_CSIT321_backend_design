"""Reproducible run-manifest support for research-system-v2.0.0."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from Agents.core.research_contracts import VersionInfo, utc_now_iso

MANIFEST_SCHEMA_VERSION = "1.0"


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(repo_dir: Path) -> Dict[str, Any]:
    def command(*args: str) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            return completed.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(command("status", "--porcelain")),
    }


class RunManifest:
    """Mutable run recorder with deterministic JSON and atomic persistence."""

    def __init__(
        self,
        run_id: Optional[str] = None,
        scenario_id: str = "unspecified",
        system_variant: str = "full_hybrid",
        random_seed: Optional[int] = None,
        repo_dir: str | Path = ".",
    ) -> None:
        self.repo_dir = Path(repo_dir).resolve()
        self.data: Dict[str, Any] = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": run_id or f"run_{uuid4().hex}",
            "scenario_id": scenario_id,
            "system_variant": system_variant,
            "status": "running",
            "timestamp_start": utc_now_iso(),
            "timestamp_end": None,
            "versions": VersionInfo().to_dict(),
            "git": git_metadata(self.repo_dir),
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "random_seed": random_seed,
            "dataset": {},
            "configuration": {},
            "stages": [],
            "events": [],
            "agent_decisions": [],
            "tool_calls": [],
            "validation_outcomes": [],
            "fallbacks": [],
            "artifacts": [],
            "metrics": {},
            "errors": [],
        }

    def set_dataset(
        self,
        path: str | Path,
        dataset_id: Optional[str] = None,
        dataset_version: Optional[str] = None,
        record_count: Optional[int] = None,
    ) -> None:
        dataset_path = Path(path).resolve()
        self.data["dataset"] = {
            "dataset_id": dataset_id or dataset_path.stem,
            "dataset_version": dataset_version,
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "record_count": record_count,
        }

    def set_configuration(self, **configuration: Any) -> None:
        self.data["configuration"].update(deepcopy(configuration))

    def add_stage(
        self,
        name: str,
        status: str,
        runtime_seconds: Optional[float] = None,
        **details: Any,
    ) -> None:
        self.data["stages"].append(
            {
                "name": name,
                "status": status,
                "runtime_seconds": runtime_seconds,
                "details": deepcopy(details),
            }
        )

    def add_artifact(self, name: str, path: str | Path) -> None:
        artifact_path = Path(path).resolve()
        self.data["artifacts"].append(
            {
                "name": name,
                "path": str(artifact_path),
                "sha256": sha256_file(artifact_path),
            }
        )

    def add_record(self, category: str, record: Dict[str, Any]) -> None:
        if category not in self.data or not isinstance(self.data[category], list):
            raise ValueError(f"Unsupported manifest record category: {category}")
        self.data[category].append(deepcopy(record))

    def set_metrics(self, **metrics: Any) -> None:
        self.data["metrics"].update(deepcopy(metrics))

    def finalise(
        self,
        status: str,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data["status"] = status
        self.data["timestamp_end"] = utc_now_iso()
        if error is not None:
            self.data["errors"].append(deepcopy(error))

    def to_dict(self) -> Dict[str, Any]:
        return deepcopy(self.data)

    def write(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(
            output_path.suffix + f".{uuid4().hex[:10]}.tmp"
        )
        temporary_path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
        return output_path
