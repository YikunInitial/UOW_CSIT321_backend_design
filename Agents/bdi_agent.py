"""
BDI shadow decision layer for dynamic replanning.

This module intentionally does not execute optimisation. It turns the current
dynamic event context into explicit Beliefs, Desires, and Intentions, then
compares the BDI recommendation with the deterministic legacy policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class Desire:
    name: str
    priority: float
    hard: bool
    success_condition: str
    reason: str


@dataclass
class IntentionOption:
    name: str
    expected_runtime: str
    disruption: str
    quality_confidence: float
    risk: str
    utility: float
    reason: str


class BDIAgent:
    """
    Builds an auditable BDI trace around the existing dynamic replanning policy.

    The default mode is shadow. BDI independently recommends an intention, but
    the legacy policy remains the executed decision. This keeps current demo
    behaviour stable while making the disagreement surface explicit.
    """

    def decide_dynamic_replan(
        self,
        records: List[Dict[str, Any]],
        baseline_solution: Dict[str, Any],
        event: Dict[str, Any],
        partial_scope: Dict[str, Any],
        time_budget_seconds: float | None,
        baseline_strategy_decision: Dict[str, Any],
        compatibility_mode: bool = True,
        mode: str = "shadow",
    ) -> Dict[str, Any]:
        beliefs = self.update_beliefs(
            records=records,
            baseline_solution=baseline_solution,
            event=event,
            partial_scope=partial_scope,
            time_budget_seconds=time_budget_seconds,
            baseline_strategy_decision=baseline_strategy_decision,
        )
        desires = self.generate_desires(beliefs)
        options = self.generate_intention_options(beliefs, desires)
        selected = self.select_intention(
            options=options,
            baseline_strategy_decision=baseline_strategy_decision,
            compatibility_mode=compatibility_mode,
        )
        shadow_decision = self.build_shadow_decision(
            options=options,
            executed=selected,
            baseline_strategy_decision=baseline_strategy_decision,
        )

        return {
            "model": "BDI",
            "version": "2.0_shadow_decision_state",
            "mode": mode,
            "compatibility_mode": compatibility_mode,
            "beliefs": beliefs,
            "desires": [asdict(desire) for desire in desires],
            "candidate_intentions": [
                asdict(option) for option in options
            ],
            "selected_intention": selected,
            "shadow_decision": shadow_decision,
            "reconsideration_policy": self.reconsideration_policy(selected),
            "rationality_summary": self.rationality_summary(
                beliefs=beliefs,
                desires=desires,
                selected=selected,
                shadow_decision=shadow_decision,
            ),
        }

    def update_beliefs(
        self,
        records: List[Dict[str, Any]],
        baseline_solution: Dict[str, Any],
        event: Dict[str, Any],
        partial_scope: Dict[str, Any],
        time_budget_seconds: float | None,
        baseline_strategy_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        total_products = max(1, len(records))
        affected_products = len(event.get("affected_product_ids", []))
        baseline_wagons = baseline_solution.get("wagons", [])
        validation = baseline_solution.get("validation_summary", {})
        wagon_summary = baseline_solution.get("wagon_summary", {})

        return {
            "environment": {
                "event_type": event.get("event_type"),
                "description": event.get("description"),
                "affected_product_count": affected_products,
                "affected_ratio": round(affected_products / total_products, 4),
                "affected_destinations": event.get(
                    "affected_destinations",
                    [],
                ),
                "affected_wagons": event.get("affected_wagons", []),
            },
            "current_plan": {
                "product_count": total_products,
                "baseline_wagon_count": len(baseline_wagons),
                "baseline_valid": validation.get("is_valid"),
                "baseline_violation_count": validation.get(
                    "violation_count",
                ),
                "average_area_utilisation": wagon_summary.get(
                    "average_area_utilisation"
                ),
            },
            "constraints": {
                "hard": [
                    "weight_limit",
                    "boundary_check",
                    "overlap_check",
                    "coverage_check",
                ],
                "soft": [
                    "minimise_wagon_count",
                    "preserve_unaffected_wagons",
                    "respond_within_time_budget",
                    "maintain_balance_quality",
                ],
                "time_budget_seconds": time_budget_seconds,
            },
            "capabilities": {
                "can_partial_replan": partial_scope.get(
                    "partial_product_count",
                    0,
                )
                > 0,
                "can_lock_unaffected_wagons": partial_scope.get(
                    "locked_wagon_count",
                    0,
                )
                > 0,
                "can_full_replan": True,
                "can_benchmark": True,
            },
            "impact_scope": partial_scope,
            "legacy_policy_signal": {
                "selected_strategy": baseline_strategy_decision.get(
                    "selected_strategy"
                ),
                "complexity_flags": baseline_strategy_decision.get(
                    "complexity_flags",
                    [],
                ),
                "reason": baseline_strategy_decision.get("reason"),
            },
        }

    def generate_desires(self, beliefs: Dict[str, Any]) -> List[Desire]:
        time_budget = beliefs["constraints"].get("time_budget_seconds")
        affected_ratio = beliefs["environment"].get("affected_ratio", 0.0)
        can_lock = beliefs["capabilities"].get(
            "can_lock_unaffected_wagons",
            False,
        )

        desires = [
            Desire(
                name="maintain_safety",
                priority=1.0,
                hard=True,
                success_condition="final solution passes all hard validation checks",
                reason="Loading safety constraints are mandatory.",
            ),
            Desire(
                name="recover_feasibility_after_event",
                priority=0.95,
                hard=True,
                success_condition="all products after the event are covered by a valid plan",
                reason="The online plan must remain executable after a dynamic event.",
            ),
        ]

        disruption_priority = 0.8 if can_lock else 0.45
        desires.append(
            Desire(
                name="minimise_operational_disruption",
                priority=disruption_priority,
                hard=False,
                success_condition="preserve unaffected wagons when possible",
                reason=(
                    "Locked unaffected wagons are available."
                    if can_lock
                    else "Few unaffected wagons can be safely preserved."
                ),
            )
        )

        speed_priority = 0.85 if time_budget is not None else 0.55
        desires.append(
            Desire(
                name="respond_fast",
                priority=speed_priority,
                hard=False,
                success_condition="finish within the dynamic time budget",
                reason=(
                    "A runtime budget is active."
                    if time_budget is not None
                    else "No explicit runtime budget is active."
                ),
            )
        )

        quality_priority = 0.75 if affected_ratio >= 0.25 else 0.55
        desires.append(
            Desire(
                name="maintain_global_quality",
                priority=quality_priority,
                hard=False,
                success_condition="avoid avoidable wagon-count or objective regression",
                reason=(
                    "The event has broad enough impact to justify quality checks."
                    if affected_ratio >= 0.25
                    else "The event impact is localized."
                ),
            )
        )

        return desires

    def generate_intention_options(
        self,
        beliefs: Dict[str, Any],
        desires: List[Desire],
    ) -> List[IntentionOption]:
        affected_ratio = beliefs["environment"].get("affected_ratio", 0.0)
        can_lock = beliefs["capabilities"].get(
            "can_lock_unaffected_wagons",
            False,
        )
        time_budget = beliefs["constraints"].get("time_budget_seconds")
        tight_budget = time_budget is not None and time_budget <= 3.0

        desire_priority = {
            desire.name: desire.priority for desire in desires
        }

        partial_utility = (
            0.45 * desire_priority["maintain_safety"]
            + 0.25 * desire_priority["respond_fast"]
            + (0.2 if can_lock else -0.15)
            + (0.1 if affected_ratio <= 0.25 else -0.15)
        )
        benchmark_utility = (
            0.45 * desire_priority["maintain_safety"]
            + 0.2 * desire_priority["maintain_global_quality"]
            + (0.15 if can_lock else -0.05)
            + (0.1 if not tight_budget else -0.05)
        )
        full_utility = (
            0.5 * desire_priority["maintain_safety"]
            + 0.25 * desire_priority["maintain_global_quality"]
            + (0.15 if affected_ratio >= 0.5 else -0.05)
            - (0.1 if tight_budget else 0.0)
        )

        return [
            IntentionOption(
                name="partial_replan_only",
                expected_runtime="fast",
                disruption="low" if can_lock else "medium",
                quality_confidence=0.7 if affected_ratio <= 0.25 else 0.55,
                risk="medium",
                utility=round(partial_utility, 4),
                reason=(
                    "Best when the event is localized and response time is the "
                    "dominant soft goal."
                ),
            ),
            IntentionOption(
                name="partial_replan_with_full_replan_benchmark",
                expected_runtime="medium",
                disruption="low" if can_lock else "medium",
                quality_confidence=0.85,
                risk="low",
                utility=round(benchmark_utility, 4),
                reason=(
                    "Balances fast local recovery with a global quality "
                    "benchmark."
                ),
            ),
            IntentionOption(
                name="full_replan_only",
                expected_runtime="slow",
                disruption="high",
                quality_confidence=0.9,
                risk="low",
                utility=round(full_utility, 4),
                reason=(
                    "Best when the event is broad or preserving old wagons "
                    "would create merge risk."
                ),
            ),
        ]

    def select_intention(
        self,
        options: List[IntentionOption],
        baseline_strategy_decision: Dict[str, Any],
        compatibility_mode: bool,
    ) -> Dict[str, Any]:
        utility_choice = max(options, key=lambda option: option.utility)
        legacy_strategy = baseline_strategy_decision.get("selected_strategy")
        selected = utility_choice

        if compatibility_mode and legacy_strategy:
            for option in options:
                if option.name == legacy_strategy:
                    selected = option
                    break

        return {
            **asdict(selected),
            "selection_source": (
                "legacy_policy_executed_shadow_mode"
                if compatibility_mode and legacy_strategy
                else "bdi_utility"
            ),
            "utility_best_intention": utility_choice.name,
            "legacy_policy_intention": legacy_strategy,
            "commitment_reason": (
                baseline_strategy_decision.get("reason")
                if compatibility_mode and legacy_strategy
                else selected.reason
            ),
        }

    def build_shadow_decision(
        self,
        options: List[IntentionOption],
        executed: Dict[str, Any],
        baseline_strategy_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        recommended = max(options, key=lambda option: option.utility)
        legacy_strategy = baseline_strategy_decision.get("selected_strategy")
        agreement = recommended.name == legacy_strategy

        if agreement:
            comparison_reason = (
                "The BDI utility recommendation agrees with the executed "
                "legacy policy."
            )
        else:
            comparison_reason = (
                "BDI recommends a different intention, but shadow mode keeps "
                "the deterministic legacy policy in control."
            )

        return {
            "mode": "shadow",
            "legacy_selected_strategy": legacy_strategy,
            "bdi_recommended_intention": asdict(recommended),
            "executed_intention": executed.get("name"),
            "agreement_with_legacy": agreement,
            "comparison_reason": comparison_reason,
            "control_policy": (
                "Legacy dynamic strategy remains authoritative until BDI is "
                "moved from shadow mode to guarded control."
            ),
        }

    def reconsideration_policy(
        self,
        selected: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "trigger": [
                "partial replan validation fails",
                "coverage violation is detected",
                "hard safety constraint is violated",
            ],
            "next_intention": (
                "full_replan_only"
                if selected.get("name") != "full_replan_only"
                else "manual_review"
            ),
            "reason": (
                "BDI commitment is not permanent; the agent should revise its "
                "intention when validation contradicts current beliefs."
            ),
        }

    def rationality_summary(
        self,
        beliefs: Dict[str, Any],
        desires: List[Desire],
        selected: Dict[str, Any],
        shadow_decision: Dict[str, Any],
    ) -> str:
        top_desires = sorted(
            desires,
            key=lambda desire: desire.priority,
            reverse=True,
        )[:3]
        desire_names = ", ".join(desire.name for desire in top_desires)
        event_type = beliefs["environment"].get("event_type")
        recommended = shadow_decision.get(
            "bdi_recommended_intention",
            {},
        ).get("name")
        agreement = shadow_decision.get("agreement_with_legacy")
        agreement_text = "agrees with" if agreement else "differs from"

        return (
            f"For event '{event_type}', the agent believes the current impact "
            f"scope and constraints make these goals dominant: {desire_names}. "
            f"In shadow mode, BDI recommends '{recommended}', which "
            f"{agreement_text} the executed legacy intention "
            f"'{selected.get('name')}'. Validation-based reconsideration "
            "remains available."
        )
