"""Durable event-window policy and deterministic replanning impact assessment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional

from Agents.event_state import new_event_request
from Agents.llm_provider import create_llm_provider
from Agents.state_repository import SQLiteStateRepository

PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
DEFAULT_WINDOW_SECONDS = {"P0": 0, "P1": 30, "P2": 300, "P3": 900}
EVENT_PRIORITY_FLOOR = {
    "Disable wagon": "P0",
    "Cancel products": "P1",
    "Change destination": "P1",
    "Add products": "P2",
}


def _more_urgent(first: str, second: str) -> str:
    return first if PRIORITY_RANK[first] <= PRIORITY_RANK[second] else second


@dataclass(frozen=True)
class PriorityDecision:
    priority: str
    priority_rank: int
    window_seconds: int
    window_type: str
    evidence: Dict[str, Any]


class EventPriorityPolicy:
    """Safety-owned priority policy with optional, escalation-only LLM advice."""

    def classify(
        self,
        event: Dict[str, Any],
        *,
        requested_priority: str = "auto",
        requested_window_seconds: Optional[int] = None,
        business_instruction: str = "",
        enable_llm_advice: bool = True,
    ) -> PriorityDecision:
        event_type = str(event.get("event_type") or "")
        floor = EVENT_PRIORITY_FLOOR.get(event_type, "P2")
        selected = floor
        evidence: Dict[str, Any] = {
            "policy": "deterministic_safety_floor_v1",
            "event_type": event_type,
            "safety_floor": floor,
            "requested_priority": requested_priority,
            "llm_role": "advisory_escalation_only",
        }
        if requested_priority != "auto":
            if requested_priority not in PRIORITY_RANK:
                raise ValueError("priority must be auto, P0, P1, P2, or P3")
            selected = _more_urgent(selected, requested_priority)
            evidence["request_applied"] = selected == requested_priority
        if business_instruction.strip() and enable_llm_advice:
            advisory = self._llm_advisory(event, business_instruction)
            evidence["llm_advisory"] = advisory
            suggested = str(advisory.get("priority") or "")
            if suggested in PRIORITY_RANK:
                safety_terms = {
                    "safety",
                    "unsafe",
                    "emergency",
                    "failure",
                    "breakdown",
                    "安全",
                    "危险",
                    "故障",
                    "停运",
                    "紧急事故",
                }
                permits_llm_p0 = event_type == "Disable wagon" or any(
                    term in business_instruction.lower() for term in safety_terms
                )
                if suggested == "P0" and not permits_llm_p0:
                    evidence["llm_advisory_applied"] = False
                    evidence["llm_advisory_guard"] = (
                        "P0 is reserved for resource or safety emergencies; "
                        "ordinary business urgency cannot be promoted to P0 by the LLM."
                    )
                else:
                    selected = _more_urgent(selected, suggested)
                    evidence["llm_advisory_applied"] = selected == suggested
        seconds = (
            int(requested_window_seconds)
            if requested_window_seconds is not None
            else DEFAULT_WINDOW_SECONDS[selected]
        )
        if seconds < 0 or seconds > 86_400:
            raise ValueError("window_seconds must be between 0 and 86400")
        if selected == "P0":
            seconds = 0
        window_type = (
            "immediate"
            if seconds == 0
            else "micro"
            if seconds <= 60
            else "standard"
            if seconds <= 600
            else "planning"
        )
        evidence.update(
            {
                "selected_priority": selected,
                "window_seconds": seconds,
                "window_type": window_type,
            }
        )
        return PriorityDecision(
            priority=selected,
            priority_rank=PRIORITY_RANK[selected],
            window_seconds=seconds,
            window_type=window_type,
            evidence=evidence,
        )

    @staticmethod
    def _llm_advisory(event: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        try:
            response = create_llm_provider(
                prefix="CLIENT_LLM", timeout_seconds=4.0
            ).invoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "You advise event urgency for steel loading. Return JSON with "
                            "priority (P0-P3) and reason. You may recommend escalation only; "
                            "deterministic safety policy remains authoritative."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "event": {
                                    "event_type": event.get("event_type"),
                                    "product_count": event.get("product_count"),
                                    "source_destination": event.get(
                                        "source_destination"
                                    ),
                                    "target_destination": event.get(
                                        "target_destination"
                                    ),
                                    "disabled_wagon_id": event.get("disabled_wagon_id"),
                                },
                                "business_instruction": instruction,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                max_tokens=120,
                num_ctx=2048,
                json_mode=True,
            )
            parsed = json.loads(response.content)
            return {
                "status": "available",
                "priority": parsed.get("priority"),
                "reason": str(parsed.get("reason") or "")[:500],
                "provider": response.response_metadata.get("provider"),
                "model": response.response_metadata.get("model"),
            }
        except Exception as error:
            return {
                "status": "deterministic_fallback",
                "reason": type(error).__name__,
            }


class ScheduledEventService:
    def __init__(self, repository: SQLiteStateRepository) -> None:
        self.repository = repository
        self.priority_policy = EventPriorityPolicy()

    def schedule(
        self,
        *,
        plan_id: str,
        expected_version: int,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        clean_payload = dict(payload)
        if not clean_payload.get("event_id"):
            clean_payload.pop("event_id", None)
        event = new_event_request(
            {
                **clean_payload,
                "expected_plan_id": plan_id,
                "expected_plan_version": expected_version,
            }
        )
        decision = self.priority_policy.classify(
            event,
            requested_priority=str(clean_payload.get("priority") or "auto"),
            requested_window_seconds=clean_payload.get("window_seconds"),
            business_instruction=str(clean_payload.get("business_instruction") or ""),
            enable_llm_advice=bool(
                clean_payload.get("enable_llm_priority_advice", True)
            ),
        )
        closes_at = (
            datetime.now(timezone.utc) + timedelta(seconds=decision.window_seconds)
        ).isoformat()
        window = self.repository.add_scheduling_event(
            event,
            priority=decision.priority,
            priority_rank=decision.priority_rank,
            window_type=decision.window_type,
            closes_at=closes_at,
            window_payload={"priority_decision": decision.evidence},
        )
        return {
            "event_id": event["event_id"],
            "window": window,
            "priority_decision": decision.evidence,
        }


def assess_replanning_scope(
    events: Iterable[Dict[str, Any]],
    *,
    total_product_count: int,
    total_destination_count: int,
) -> Dict[str, Any]:
    """Select local or global recovery from measurable event impact."""
    event_list = list(events)
    changed_products = 0
    destinations: set[str] = set()
    affected_wagons: set[str] = set()
    disabled_wagons: set[str] = set()
    for event in event_list:
        direct_count = max(
            int(event.get("product_count") or 0),
            len(event.get("affected_product_ids") or []),
            len(event.get("new_products") or []),
        )
        changed_products += max(1, direct_count)
        for key in ("source_destination", "target_destination"):
            if str(event.get(key) or "").strip():
                destinations.add(str(event[key]).strip())
        destinations.update(
            str(item).strip()
            for item in event.get("affected_destinations", [])
            if str(item).strip()
        )
        if str(event.get("disabled_wagon_id") or "").strip():
            disabled_wagons.add(str(event["disabled_wagon_id"]).strip())
        event_affected_wagons = {
            str(item).strip()
            for item in event.get("affected_wagons", [])
            if str(item).strip()
        }
        affected_wagons.update(event_affected_wagons)
        normalised_event_type = (
            str(event.get("event_type") or "").strip().lower().replace(" ", "_")
        )
        if normalised_event_type == "disable_wagon":
            disabled_wagons.update(event_affected_wagons)
    product_ratio = changed_products / max(1, total_product_count)
    destination_ratio = len(destinations) / max(1, total_destination_count)
    global_reasons: list[str] = []
    if product_ratio >= 0.25:
        global_reasons.append("affected_product_ratio_at_least_25_percent")
    if len(destinations) >= 3 or (len(destinations) >= 2 and destination_ratio >= 0.5):
        global_reasons.append("cross_destination_structural_impact")
    if len(disabled_wagons) >= 2:
        global_reasons.append("multiple_wagons_unavailable")
    if len(event_list) >= 8:
        global_reasons.append("large_event_batch")
    mode = "global" if global_reasons else "local"
    return {
        "policy": "impact_assessor_v1",
        "selected_scope": mode,
        "affected_product_count": changed_products,
        "affected_product_ratio": round(product_ratio, 6),
        "affected_destination_count": len(destinations),
        "affected_destination_ratio": round(destination_ratio, 6),
        "affected_wagon_count": len(affected_wagons),
        "disabled_wagon_count": len(disabled_wagons),
        "event_count": len(event_list),
        "global_reasons": global_reasons,
        "local_reason": None if global_reasons else "bounded_local_impact",
    }
