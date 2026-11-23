"""
Unified decision state for agentic dynamic replanning.

The state object keeps decision data in one structured place instead of
spreading it across event dictionaries, strategy plans, BDI traces, and reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentDecisionState:
    decision_id: str
    phase: str
    event: Dict[str, Any]
    input_state: Dict[str, Any]
    belief_state: Dict[str, Any] = field(default_factory=dict)
    desire_state: List[Dict[str, Any]] = field(default_factory=list)
    intention_options: List[Dict[str, Any]] = field(default_factory=list)
    legacy_policy: Dict[str, Any] = field(default_factory=dict)
    shadow_decision: Dict[str, Any] = field(default_factory=dict)
    execution_decision: Dict[str, Any] = field(default_factory=dict)
    validation_state: Dict[str, Any] = field(default_factory=dict)
    audit_state: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_dynamic_decision_state(
    decision_id: str,
    event: Dict[str, Any],
    records: List[Dict[str, Any]],
    baseline_solution: Dict[str, Any],
    partial_scope: Dict[str, Any],
    strategy_decision: Dict[str, Any],
    bdi_trace: Dict[str, Any],
    time_budget_seconds: float | None,
) -> AgentDecisionState:
    validation = baseline_solution.get("validation_summary", {})
    wagon_summary = baseline_solution.get("wagon_summary", {})
    selected_intention = bdi_trace.get("selected_intention", {})

    return AgentDecisionState(
        decision_id=decision_id,
        phase="dynamic_replanning",
        event=event,
        input_state={
            "product_count": len(records),
            "baseline_wagon_count": len(baseline_solution.get("wagons", [])),
            "baseline_valid": validation.get("is_valid"),
            "baseline_violation_count": validation.get("violation_count"),
            "baseline_average_area_utilisation": wagon_summary.get(
                "average_area_utilisation"
            ),
            "time_budget_seconds": time_budget_seconds,
            "partial_scope": partial_scope,
        },
        belief_state=bdi_trace.get("beliefs", {}),
        desire_state=bdi_trace.get("desires", []),
        intention_options=bdi_trace.get("candidate_intentions", []),
        legacy_policy={
            "selected_strategy": strategy_decision.get("selected_strategy"),
            "run_partial_replan": strategy_decision.get(
                "run_partial_replan"
            ),
            "run_full_replan": strategy_decision.get("run_full_replan"),
            "reason": strategy_decision.get("reason"),
            "complexity_flags": strategy_decision.get(
                "complexity_flags",
                [],
            ),
        },
        shadow_decision=bdi_trace.get("shadow_decision", {}),
        execution_decision={
            "executed_strategy": strategy_decision.get("selected_strategy"),
            "executed_intention": selected_intention.get("name"),
            "execution_source": selected_intention.get("selection_source"),
            "reason": selected_intention.get("commitment_reason"),
            "reconsideration_policy": bdi_trace.get(
                "reconsideration_policy",
                {},
            ),
        },
        validation_state={
            "pre_execution_validation": {
                "is_valid": validation.get("is_valid"),
                "violation_count": validation.get("violation_count"),
            },
            "post_execution_validation": {},
        },
        audit_state={
            "decision_model": bdi_trace.get("mode"),
            "bdi_version": bdi_trace.get("version"),
            "agreement": bdi_trace.get("shadow_decision", {}).get(
                "agreement_with_legacy"
            ),
        },
        notes=[
            "Unified state is used for audit and reporting.",
            "Shadow BDI recommendation does not change execution in this mode.",
        ],
    )


def attach_post_execution_validation(
    decision_state: Dict[str, Any],
    final_solution: Dict[str, Any],
) -> Dict[str, Any]:
    updated = dict(decision_state)
    validation_state = dict(updated.get("validation_state", {}))
    validation = final_solution.get("validation_summary", {})
    validation_state["post_execution_validation"] = {
        "is_valid": validation.get("is_valid"),
        "violation_count": validation.get("violation_count"),
        "overlap_violation_count": validation.get("overlap_violation_count"),
        "boundary_violation_count": validation.get("boundary_violation_count"),
        "overweight_violation_count": validation.get(
            "overweight_violation_count"
        ),
        "coverage_violation_count": validation.get("coverage_violation_count"),
    }
    updated["validation_state"] = validation_state
    return updated
