"""Idempotent event and optimistic plan-version controls."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict
from uuid import uuid4

from Agents.core.research_contracts import (
    AgenticSystemError,
    ErrorCode,
    EventStatus,
    StructuredError,
    utc_now_iso,
)


def new_event_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    event = deepcopy(payload)
    event.setdefault("event_id", f"event_{uuid4().hex}")
    event.setdefault("event_status", EventStatus.RECEIVED.value)
    event.setdefault("received_at", utc_now_iso())
    event.setdefault("expected_plan_id", None)
    event.setdefault("expected_plan_version", None)
    return event


def bind_and_validate_event(
    event_request: Dict[str, Any],
    environment_state: Dict[str, Any],
) -> Dict[str, Any]:
    event = new_event_request(event_request)
    processed = set(environment_state.get("processed_event_ids", []))
    if event["event_id"] in processed:
        raise AgenticSystemError(
            StructuredError(
                code=ErrorCode.DUPLICATE_EVENT,
                message=f"Event already processed: {event['event_id']}",
                details={"event_id": event["event_id"]},
            )
        )
    current_plan_id = environment_state.get("active_plan_id")
    current_version = int(environment_state.get("active_plan_version", 1))
    expected_plan_id = event.get("expected_plan_id")
    expected_version = event.get("expected_plan_version")
    if expected_plan_id is not None and expected_plan_id != current_plan_id:
        raise AgenticSystemError(
            StructuredError(
                code=ErrorCode.STALE_PLAN_VERSION,
                message="Event targets a plan that is no longer active.",
                retryable=True,
                details={
                    "event_id": event["event_id"],
                    "expected_plan_id": expected_plan_id,
                    "active_plan_id": current_plan_id,
                },
            )
        )
    if expected_version is not None and int(expected_version) != current_version:
        raise AgenticSystemError(
            StructuredError(
                code=ErrorCode.STALE_PLAN_VERSION,
                message="Event targets a stale plan version.",
                retryable=True,
                details={
                    "event_id": event["event_id"],
                    "expected_plan_version": expected_version,
                    "active_plan_version": current_version,
                },
            )
        )
    event["expected_plan_id"] = current_plan_id
    event["expected_plan_version"] = current_version
    event["event_status"] = EventStatus.PROCESSING.value
    return event


def complete_event(
    environment_state: Dict[str, Any],
    event_request: Dict[str, Any],
    new_plan_id: str,
) -> None:
    processed = list(environment_state.get("processed_event_ids", []))
    if event_request["event_id"] not in processed:
        processed.append(event_request["event_id"])
    environment_state["processed_event_ids"] = processed
    environment_state["previous_plan_id"] = environment_state.get(
        "active_plan_id"
    )
    environment_state["active_plan_id"] = new_plan_id
    environment_state["active_plan_version"] = int(
        environment_state.get("active_plan_version", 1)
    ) + 1
