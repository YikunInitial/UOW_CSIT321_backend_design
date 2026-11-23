"""Submit the synthetic planning request and wait for its durable result."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4


BASE_URL = os.getenv("STEEL_API_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
API_KEY = os.getenv("STEEL_API_KEY", "").strip()
REQUEST_PATH = Path(__file__).with_name("planning_request.json")


def call(method: str, path: str, payload: dict | None = None, **headers: str) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Accept": "application/json", **headers}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    if API_KEY:
        request_headers["X-API-Key"] = API_KEY
    request = urllib.request.Request(
        f"{BASE_URL}{path}", data=body, headers=request_headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {path}: {detail}") from error


def main() -> int:
    live = call("GET", "/health/live")
    print(f"API live: {live}")

    payload = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    queued = call(
        "POST",
        "/v1/planning-runs",
        payload,
        **{"Idempotency-Key": f"student-quickstart-{uuid4().hex}"},
    )
    job_id = queued["job_id"]
    print(f"Queued job: {job_id}")

    deadline = time.monotonic() + 120
    last_status = None
    while time.monotonic() < deadline:
        job = call("GET", f"/v1/jobs/{job_id}")
        status = job["status"]
        if status != last_status:
            print(f"Job status: {status}")
            last_status = status
        if status == "completed":
            result = job.get("result") or {}
            plan = result.get("plan") or {}
            print(f"Plan ID: {plan.get('plan_id')}")
            print(f"Plan version: {plan.get('version')}")
            print(f"Plan status: {plan.get('status')}")
            print("Quick start completed successfully.")
            return 0
        if status in {"failed", "dead_letter"}:
            print(json.dumps(job.get("error"), ensure_ascii=False, indent=2))
            return 1
        time.sleep(1)

    print("Timed out waiting for the job. Check /health/ready and worker logs.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Quick start failed: {error}", file=sys.stderr)
        raise SystemExit(1)
