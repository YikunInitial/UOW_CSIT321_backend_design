"""Durable controller for the development/UAT continuous-order sandbox.

The browser is a control surface only.  A worker owns each short transition,
while SQLite records the stream, its in-flight window and bounded history so a
browser refresh, network outage or worker restart cannot duplicate orders.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from Agents.dynamic_replanning_agent import load_product_records
from Agents.planning_service import PlanningService
from Agents.scheduling_service import ScheduledEventService
from Agents.state_repository import SQLiteStateRepository, StateConflictError


DEFAULT_DESTINATIONS = (
    "Shanghai",
    "Beijing",
    "Tianjin",
    "Jiangsu",
    "Zhejiang",
    "Shandong",
    "Hebei",
    "Guangdong",
    "Hubei",
    "Sichuan",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat()


class ContinuousStreamController:
    """Advance at most one idempotent stream transition per worker tick."""

    def __init__(
        self,
        repository: SQLiteStateRepository,
        *,
        worker_id: str,
        lease_seconds: float = 30.0,
    ) -> None:
        self.repository = repository
        self.worker_id = worker_id
        self.lease_seconds = max(5.0, lease_seconds)

    @staticmethod
    def _interval(stream: Dict[str, Any], next_cycle: int) -> int:
        low = int(stream["min_interval_seconds"])
        high = int(stream["max_interval_seconds"])
        span = max(1, high - low + 1)
        digest = hashlib.sha256(
            f"{stream['stream_id']}:{next_cycle}".encode("utf-8")
        ).digest()
        return low + int.from_bytes(digest[:2], "big") % span

    @staticmethod
    def _scenario(stream: Dict[str, Any], records: list[Dict[str, Any]]) -> Dict[str, Any]:
        next_cycle = int(stream["cycle_count"]) + 1
        digest = hashlib.sha256(
            f"{stream['stream_id']}:{next_cycle}:scenario".encode("utf-8")
        ).digest()
        remaining = int(stream["max_products"]) - len(records)
        count = min(remaining, 1 + digest[0] % 3)
        destinations = sorted(
            {
                str(item.get("destination") or "").strip()
                for item in records
                if str(item.get("destination") or "").strip()
            }
            | set(DEFAULT_DESTINATIONS)
        )
        target = destinations[digest[1] % len(destinations)]
        return {
            "event_type": "Add products",
            "product_count": count,
            "target_destination": target,
            "add_product_mode": "template_simulation",
            "priority": "auto",
            "window_seconds": 8,
            "business_instruction": "",
            "enable_llm_priority_advice": False,
            # Repeated cycles do not generate wagon PNG/PDF bundles. The final
            # approved plan remains fully auditable through JSON and manifests.
            "generate_visuals": False,
            "time_budget_seconds": 5.0,
        }

    @staticmethod
    def _recent(
        stream: Dict[str, Any], *, event: Optional[Dict[str, Any]] = None
    ) -> list[Dict[str, Any]]:
        values = list(stream.get("recent_events") or [])
        if event is not None:
            values.append(event)
        return values[-40:]

    def tick(self) -> bool:
        stream = self.repository.claim_continuous_stream(
            self.worker_id, lease_seconds=self.lease_seconds
        )
        if stream is None:
            return False
        try:
            return self._advance(stream)
        except Exception as error:
            # A controller failure is persisted but does not kill the job
            # worker. The stream backs off and may be stopped by an operator.
            retry_at = _iso(_utc_now() + timedelta(seconds=30))
            self.repository.update_continuous_stream(
                stream["stream_id"],
                self.worker_id,
                status=(
                    "failed"
                    if isinstance(error, (KeyError, ValueError, StateConflictError))
                    else None
                ),
                next_event_at=retry_at,
                last_error={"type": type(error).__name__, "message": str(error)[:500]},
            )
            return True

    def _advance(self, stream: Dict[str, Any]) -> bool:
        stream_id = str(stream["stream_id"])
        current_window_id = stream.get("current_window_id")
        if current_window_id:
            window = self.repository.get_scheduling_window(str(current_window_id))
            if window["status"] in {"collecting", "dispatched"}:
                self.repository.update_continuous_stream(stream_id, self.worker_id)
                return False
            if window["status"] == "failed":
                recent = self._recent(
                    stream,
                    event={
                        "event_id": (window.get("events") or [{}])[0].get("event_id"),
                        "window_id": window["window_id"],
                        "status": "failed",
                        "detail": str((window.get("result") or {}).get("message") or "Replanning failed."),
                        "completed_at": _iso(_utc_now()),
                    },
                )
                self.repository.update_continuous_stream(
                    stream_id,
                    self.worker_id,
                    status="stopped" if stream["status"] == "stopping" else "failed",
                    current_window_id=None,
                    last_error=window.get("result") or {"message": "Replanning failed."},
                    recent_events=recent,
                )
                return True
            if window["status"] != "completed":
                raise ValueError(f"Unsupported scheduling window status: {window['status']}")
            result = window.get("result") or {}
            plan_value = result.get("plan") or {}
            successor_id = str(plan_value.get("plan_id") or result.get("plan_id") or "")
            if not successor_id:
                raise ValueError("Completed stream window has no successor plan.")
            successor = self.repository.get_plan(successor_id)
            PlanningService(self.repository).approve_plan(
                successor_id,
                int(successor["version"]),
                approved_by=f"continuous-sandbox:{stream_id}",
            )
            cycle_count = int(stream["cycle_count"]) + 1
            recent = self._recent(
                stream,
                event={
                    "event_id": (window.get("events") or [{}])[0].get("event_id"),
                    "window_id": window["window_id"],
                    "status": "completed",
                    "plan_id": successor_id,
                    "plan_version": int(successor["version"]),
                    "recovery_mode": result.get("recovery_mode"),
                    "scope_decision": result.get("scope_decision") or {},
                    "product_count": int(
                        ((window.get("events") or [{}])[0].get("payload") or {}).get("product_count") or 0
                    ),
                    "destination": str(
                        ((window.get("events") or [{}])[0].get("payload") or {}).get("target_destination") or ""
                    ),
                    "completed_at": _iso(_utc_now()),
                },
            )
            stopping = stream["status"] == "stopping"
            self.repository.update_continuous_stream(
                stream_id,
                self.worker_id,
                status="stopped" if stopping else "running",
                plan_id=successor_id,
                plan_version=int(successor["version"]),
                cycle_count=cycle_count,
                current_window_id=None,
                next_event_at=_iso(
                    _utc_now()
                    + timedelta(seconds=self._interval(stream, cycle_count + 1))
                ),
                last_error=None,
                recent_events=recent,
            )
            return True

        if stream["status"] == "stopping":
            self.repository.update_continuous_stream(
                stream_id, self.worker_id, status="stopped"
            )
            return True
        due_at = datetime.fromisoformat(str(stream["next_event_at"]))
        if due_at > _utc_now():
            self.repository.update_continuous_stream(stream_id, self.worker_id)
            return False
        plan = self.repository.get_plan(str(stream["plan_id"]))
        if plan["status"] != "active" or int(plan["version"]) != int(stream["plan_version"]):
            raise ValueError("Continuous stream baseline is no longer the active version.")
        records = load_product_records(Path(plan["run_dir"]) / "products.json")
        if len(records) >= int(stream["max_products"]):
            self.repository.update_continuous_stream(
                stream_id,
                self.worker_id,
                status="stopped",
                last_error={
                    "code": "STREAM_PRODUCT_LIMIT_REACHED",
                    "message": "The configured continuous-stream product limit was reached.",
                },
            )
            return True
        payload = self._scenario(stream, records)
        next_cycle = int(stream["cycle_count"]) + 1
        event_id = f"stream-{stream_id}-cycle-{next_cycle}"
        payload["event_id"] = event_id
        scheduled = ScheduledEventService(self.repository).schedule(
            plan_id=str(stream["plan_id"]),
            expected_version=int(stream["plan_version"]),
            payload=payload,
        )
        window_id = str(scheduled["window"]["window_id"])
        recent = self._recent(
            stream,
            event={
                "event_id": event_id,
                "window_id": window_id,
                "status": "scheduled",
                "product_count": payload["product_count"],
                "destination": payload["target_destination"],
                "created_at": _iso(_utc_now()),
            },
        )
        self.repository.update_continuous_stream(
            stream_id,
            self.worker_id,
            current_window_id=window_id,
            last_error=None,
            recent_events=recent,
        )
        return True

