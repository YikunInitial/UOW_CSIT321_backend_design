"""Plan-artifact integrity checks used before approval and replanning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from Agents.run_manifest import sha256_file


class ArtifactIntegrityError(RuntimeError):
    """Raised when persisted plan evidence is missing, unsafe, or corrupt."""


def verify_plan_artifacts(plan: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = Path(plan["run_dir"]).resolve()
    payload = plan["payload"]
    required = {
        "loading_solution": Path(payload["solution_path"]).resolve(),
        "strategy_plan": Path(payload["strategy_path"]).resolve(),
        "run_manifest": Path(payload["manifest_path"]).resolve(),
    }
    errors: list[str] = []
    for name, path in required.items():
        if path != run_dir and run_dir not in path.parents:
            errors.append(f"{name} path is outside the plan run directory")
        elif not path.is_file():
            errors.append(f"{name} is missing")
    if errors:
        raise ArtifactIntegrityError("; ".join(errors))
    try:
        manifest = json.loads(required["run_manifest"].read_text(encoding="utf-8"))
        json.loads(required["loading_solution"].read_text(encoding="utf-8"))
        json.loads(required["strategy_plan"].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ArtifactIntegrityError(
            "A required plan artifact is not valid JSON."
        ) from error

    declared = {
        str(item.get("name")): item
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict)
    }
    verified = []
    for name in ("loading_solution", "strategy_plan"):
        item = declared.get(name)
        if item is None:
            errors.append(f"manifest does not declare {name}")
            continue
        artifact_path = Path(str(item.get("path", ""))).resolve()
        if artifact_path != required[name]:
            errors.append(f"manifest path mismatch for {name}")
            continue
        actual_hash = sha256_file(artifact_path)
        if item.get("sha256") != actual_hash:
            errors.append(f"SHA-256 mismatch for {name}")
        else:
            verified.append(name)

    dataset = manifest.get("dataset", {})
    dataset_path_value = dataset.get("path") if isinstance(dataset, dict) else None
    if dataset_path_value:
        dataset_path = Path(str(dataset_path_value)).resolve()
        if dataset_path != run_dir and run_dir not in dataset_path.parents:
            errors.append("dataset path is outside the plan run directory")
        elif not dataset_path.is_file():
            errors.append("dataset artifact is missing")
        elif dataset.get("sha256") != sha256_file(dataset_path):
            errors.append("SHA-256 mismatch for dataset")
        else:
            verified.append("dataset")

    if errors:
        raise ArtifactIntegrityError("; ".join(errors))
    return {
        "valid": True,
        "plan_id": plan["plan_id"],
        "verified_artifacts": verified,
    }
