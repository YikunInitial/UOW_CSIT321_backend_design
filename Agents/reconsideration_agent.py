"""
Validation-grounded reconsideration policy for dynamic replanning.

The agent does not replace the optimiser. It observes validation evidence,
revises the decision state, and proposes a safer fallback intention when the
current intention fails hard validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class ReconsiderationAttempt:
    attempt: int
    intention: str
    execution_source: str
    validation_passed: bool
    violation_count: int | None
    failure_reason: str | None
    belief_revision: Dict[str, Any] | None
    next_action: str


class ReconsiderationAgent:
    """
    Deterministic guarded retry policy for safety-critical dynamic replanning.

    Escalation chain:
    partial_replan_only -> partial_replan_with_full_replan_benchmark
    partial_replan_with_full_replan_benchmark -> full_replan_only
    full_replan_only -> manual_review
    """

    FALLBACK_CHAIN = {
        "partial_replan_only": "partial_replan_with_full_replan_benchmark",
        "partial_replan_with_full_replan_benchmark": "full_replan_only",
        "full_replan_only": "manual_review",
    }

    def __init__(self, max_attempts: int = 3) -> None:
        self.max_attempts = max(1, max_attempts)

    def initialise_loop(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "mode": "validation_grounded_guarded_retry",
            "max_attempts": self.max_attempts,
            "attempts": [],
            "final_status": "not_started",
            "final_intention": None,
            "summary": "",
        }

    def record_attempt(
        self,
        loop_state: Dict[str, Any],
        intention: str,
        solution: Dict[str, Any] | None,
        attempt_number: int | None = None,
        execution_source: str = "legacy_policy",
        belief_revision: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        updated = dict(loop_state)
        attempts: List[Dict[str, Any]] = list(updated.get("attempts", []))
        attempt_number = attempt_number or len(attempts) + 1
        validation = (solution or {}).get("validation_summary", {})
        passed = bool(validation.get("is_valid"))
        violation_count = validation.get("violation_count")
        failure_reason = None if passed else self.failure_reason(validation)
        next_action = "accept" if passed else self.next_intention(intention)

        attempts.append(
            asdict(
                ReconsiderationAttempt(
                    attempt=attempt_number,
                    intention=intention,
                    execution_source=execution_source,
                    validation_passed=passed,
                    violation_count=violation_count,
                    failure_reason=failure_reason,
                    belief_revision=belief_revision,
                    next_action=next_action,
                )
            )
        )

        updated["attempts"] = attempts
        updated["final_status"] = "accepted" if passed else "needs_reconsideration"
        updated["final_intention"] = intention if passed else None
        updated["summary"] = self.build_summary(updated)
        return updated

    def should_reconsider(
        self,
        loop_state: Dict[str, Any],
        current_intention: str,
    ) -> bool:
        attempts = loop_state.get("attempts", [])
        if not attempts:
            return False
        last_attempt = attempts[-1]
        if last_attempt.get("validation_passed"):
            return False
        if len(attempts) >= self.max_attempts:
            return False
        return self.next_intention(current_intention) != "manual_review"

    def next_intention(self, current_intention: str) -> str:
        return self.FALLBACK_CHAIN.get(current_intention, "full_replan_only")

    def revise_beliefs(
        self,
        failed_intention: str,
        failed_solution: Dict[str, Any] | None,
        next_intention: str,
    ) -> Dict[str, Any]:
        validation = (failed_solution or {}).get("validation_summary", {})
        evidence = {
            "validation_passed": validation.get("is_valid"),
            "violation_count": validation.get("violation_count"),
            "overlap_violation_count": validation.get(
                "overlap_violation_count"
            ),
            "boundary_violation_count": validation.get(
                "boundary_violation_count"
            ),
            "overweight_violation_count": validation.get(
                "overweight_violation_count"
            ),
            "coverage_violation_count": validation.get(
                "coverage_violation_count"
            ),
        }
        return {
            "previous_belief": (
                f"Intention '{failed_intention}' could produce an acceptable "
                "dynamic plan."
            ),
            "new_belief": (
                "Validation evidence indicates higher execution risk, so the "
                f"agent should try safer intention '{next_intention}'."
            ),
            "evidence": evidence,
        }

    def mark_manual_review(
        self,
        loop_state: Dict[str, Any],
        failed_intention: str,
    ) -> Dict[str, Any]:
        updated = dict(loop_state)
        updated["final_status"] = "manual_review_required"
        updated["final_intention"] = failed_intention
        updated["summary"] = self.build_summary(updated)
        return updated

    def build_summary(self, loop_state: Dict[str, Any]) -> str:
        attempts = loop_state.get("attempts", [])
        if not attempts:
            return "No reconsideration attempts have run yet."

        last = attempts[-1]
        if last.get("validation_passed"):
            return (
                f"Attempt {last.get('attempt')} accepted intention "
                f"'{last.get('intention')}' after validation passed."
            )

        next_action = last.get("next_action")
        if next_action == "manual_review":
            return (
                "Validation failed and no safer automatic fallback remains; "
                "manual review is required."
            )

        return (
            f"Attempt {last.get('attempt')} failed validation; the agent will "
            f"reconsider and try '{next_action}'."
        )

    def failure_reason(self, validation: Dict[str, Any]) -> str:
        if validation.get("coverage_violation_count"):
            return "coverage violation"
        if validation.get("overlap_violation_count"):
            return "overlap violation"
        if validation.get("boundary_violation_count"):
            return "boundary violation"
        if validation.get("overweight_violation_count"):
            return "overweight violation"
        return "hard validation failed"
