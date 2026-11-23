"""Small provider-neutral client for the production steel-loading HTTP API."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from uuid import uuid4


class BackendAPIError(RuntimeError):
    """Sanitised transport or API failure suitable for a client UI."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        code: str = "BACKEND_UNAVAILABLE",
        request_id: str = "",
        retry_after: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.retry_after = retry_after


@dataclass(frozen=True)
class BackendResponse:
    data: Dict[str, Any]
    request_id: str
    response_time_ms: Optional[float]


class SteelLoadingAPIClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        user_id: str = "business-operator",
        user_roles: str = "approver,operator",
        timeout_seconds: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("Backend API URL must use http:// or https://.")
        self.api_key = api_key
        self.user_id = user_id
        self.user_roles = user_roles
        self.timeout_seconds = max(0.5, timeout_seconds)

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Optional[Dict[str, Any]] = None,
        idempotency_key: str = "",
        timeout_seconds: Optional[float] = None,
    ) -> BackendResponse:
        request_id = uuid4().hex
        headers = {
            "accept": "application/json",
            "x-request-id": request_id,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        if self.user_id:
            headers["x-user-id"] = self.user_id
        if self.user_roles:
            headers["x-user-roles"] = self.user_roles
        if idempotency_key:
            headers["idempotency-key"] = idempotency_key
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["content-type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout_seconds or self.timeout_seconds
            ) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw) if raw else {}
                if not isinstance(data, dict):
                    raise BackendAPIError(
                        "Backend returned an invalid response.",
                        status_code=response.status,
                        code="INVALID_BACKEND_RESPONSE",
                        request_id=response.headers.get("x-request-id", request_id),
                    )
                timing = response.headers.get("x-response-time-ms")
                return BackendResponse(
                    data=data,
                    request_id=response.headers.get("x-request-id", request_id),
                    response_time_ms=float(timing) if timing else None,
                )
        except urllib.error.HTTPError as error:
            try:
                parsed = json.loads(error.read().decode("utf-8"))
                details = parsed.get("error", {}) if isinstance(parsed, dict) else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                details = {}
            raise BackendAPIError(
                str(details.get("message", "Backend rejected the request.")),
                status_code=error.code,
                code=str(details.get("code", "BACKEND_REQUEST_FAILED")),
                request_id=error.headers.get("x-request-id", request_id),
                retry_after=error.headers.get("retry-after", ""),
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise BackendAPIError(
                "Backend service is unavailable or did not respond in time.",
                code="BACKEND_UNAVAILABLE",
                request_id=request_id,
            ) from error

    def liveness(self) -> BackendResponse:
        return self._request("/health/live", timeout_seconds=2.0)

    def readiness(self) -> BackendResponse:
        return self._request("/health/ready", timeout_seconds=3.0)

    def submit_plan(
        self, payload: Dict[str, Any], *, idempotency_key: str
    ) -> BackendResponse:
        return self._request(
            "/v1/planning-runs",
            method="POST",
            payload=payload,
            idempotency_key=idempotency_key,
        )

    def get_job(self, job_id: str) -> BackendResponse:
        return self._request(f"/v1/jobs/{job_id}")

    def wait_for_job(
        self,
        job_id: str,
        *,
        timeout_seconds: float,
        poll_seconds: float = 0.5,
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> BackendResponse:
        deadline = time.monotonic() + timeout_seconds
        last: BackendResponse | None = None
        previous_status = ""
        while time.monotonic() < deadline:
            last = self.get_job(job_id)
            current_status = str(last.data.get("status", "unknown"))
            if on_status is not None and current_status != previous_status:
                on_status(last.data)
                previous_status = current_status
            if last.data.get("status") in {
                "completed",
                "failed",
                "dead_letter",
            }:
                return last
            time.sleep(max(0.05, poll_seconds))
        status = last.data.get("status", "unknown") if last else "unknown"
        raise BackendAPIError(
            f"Backend job did not finish before the UI timeout (status: {status}).",
            code="JOB_WAIT_TIMEOUT",
            request_id=last.request_id if last else "",
        )

    def get_plan(self, plan_id: str) -> BackendResponse:
        return self._request(f"/v1/plans/{plan_id}")

    def approve_plan(
        self, plan_id: str, expected_version: int, *, approved_by: str
    ) -> BackendResponse:
        return self._request(
            f"/v1/plans/{plan_id}/approve",
            method="POST",
            payload={
                "expected_version": expected_version,
                "approved_by": approved_by,
            },
        )

    def confirm_enrichment(
        self,
        plan_id: str,
        expected_version: int,
        suggestion_ids: list[str],
        *,
        confirmed_by: str,
    ) -> BackendResponse:
        return self._request(
            f"/v1/plans/{plan_id}/enrichment-confirmations",
            method="POST",
            payload={
                "expected_version": expected_version,
                "suggestion_ids": suggestion_ids,
                "confirmed_by": confirmed_by,
            },
        )

    def submit_event(
        self,
        plan_id: str,
        payload: Dict[str, Any],
        *,
        idempotency_key: str,
    ) -> BackendResponse:
        return self._request(
            f"/v1/plans/{plan_id}/events",
            method="POST",
            payload=payload,
            idempotency_key=idempotency_key,
        )
