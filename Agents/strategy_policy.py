"""Bounded strategy schema and deterministic fallback policy."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

from Agents.core.research_contracts import (
    ErrorCode,
    StrategyDecisionTrace,
    StructuredError,
)


ALLOWED_STRATEGIES = {
    "simple_greedy",
    "destination_grouping_greedy",
    "balance_aware_greedy",
    "destination_grouping_plus_balance_aware_greedy",
    "greedy_plus_simulated_annealing",
    "rule_based_loading",
}

REQUIRED_CONSTRAINTS = {
    "weight_limit",
    "boundary_check",
    "overlap_check",
}

BOOLEAN_CONFIG_FIELDS = {
    "group_by_destination",
    "sort_by_weight_desc",
    "use_balance_aware_scoring",
    "use_simulated_annealing",
    "use_rule_based_constraints",
}


def validate_strategy_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("Strategy plan must be a JSON object.")
    strategy = str(plan.get("selected_strategy", "")).strip()
    if strategy not in ALLOWED_STRATEGIES:
        raise ValueError(f"Unauthorised selected_strategy: {strategy!r}")
    config = plan.get("optimisation_config")
    if not isinstance(config, dict):
        raise ValueError("optimisation_config must be an object.")
    invalid_boolean_fields = [
        field
        for field in BOOLEAN_CONFIG_FIELDS
        if field in config and not isinstance(config[field], bool)
    ]
    if invalid_boolean_fields:
        raise ValueError(
            "Optimisation configuration fields must be boolean: "
            + ", ".join(sorted(invalid_boolean_fields))
        )
    constraints = plan.get("required_constraints")
    if not isinstance(constraints, list):
        raise ValueError("required_constraints must be a list.")
    missing_constraints = REQUIRED_CONSTRAINTS - {
        str(item) for item in constraints
    }
    if missing_constraints:
        raise ValueError(
            "Strategy plan omitted mandatory hard constraints: "
            + ", ".join(sorted(missing_constraints))
        )
    next_agent = plan.get("next_agent")
    if next_agent not in (None, "optimisation_agent"):
        raise ValueError(f"Unauthorised next_agent: {next_agent!r}")
    validated = deepcopy(plan)
    validated.setdefault("next_agent", "optimisation_agent")
    return validated


def deterministic_strategy_plan(
    products: Iterable[Dict[str, Any]],
    time_budget_seconds: Optional[float] = None,
    reason: str = "Deterministic fallback policy selected the strategy.",
) -> Dict[str, Any]:
    product_list = list(products)
    destinations = {
        str(product.get("destination", "")).strip()
        for product in product_list
        if str(product.get("destination", "")).strip()
    }
    multiple_destinations = len(destinations) > 1
    use_sa = (
        len(product_list) > 12
        and (time_budget_seconds is None or time_budget_seconds >= 8.0)
    )
    if use_sa:
        selected_strategy = "greedy_plus_simulated_annealing"
    elif multiple_destinations:
        selected_strategy = "destination_grouping_plus_balance_aware_greedy"
    else:
        selected_strategy = "balance_aware_greedy"
    return {
        "scenario_type": (
            "multi_destination_batch"
            if multiple_destinations
            else "single_destination_batch"
        ),
        "selected_strategy": selected_strategy,
        "strategy_reason": reason,
        "rag_evidence": [],
        "optimisation_config": {
            "group_by_destination": multiple_destinations,
            "sort_by_weight_desc": True,
            "use_balance_aware_scoring": True,
            "use_simulated_annealing": use_sa,
            "use_rule_based_constraints": True,
            **(
                {"time_budget_seconds": float(time_budget_seconds)}
                if time_budget_seconds is not None
                else {}
            ),
        },
        "required_constraints": [
            "weight_limit",
            "boundary_check",
            "overlap_check",
            "coverage_check",
        ],
        "risk_notes": [],
        "next_agent": "optimisation_agent",
        "decision_source": "deterministic_fallback_policy",
    }


def fallback_decision(
    products: Iterable[Dict[str, Any]],
    failure_code: ErrorCode,
    failure_message: str,
    time_budget_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    plan = deterministic_strategy_plan(
        products,
        time_budget_seconds=time_budget_seconds,
        reason=(
            "Deterministic fallback used because the semantic planning stage "
            f"failed: {failure_message}"
        ),
    )
    trace = StrategyDecisionTrace(
        recommended_strategy=None,
        authorised_strategy=plan["selected_strategy"],
        executed_strategy=None,
        fallback_strategy=plan["selected_strategy"],
        recommendation_source="llm_rag_unavailable",
        authorisation_source="deterministic_fallback_policy",
        fallback_reasons=[failure_message],
    )
    error = StructuredError(
        code=failure_code,
        message=failure_message,
        retryable=True,
        details={"fallback_strategy": plan["selected_strategy"]},
    )
    return {
        "strategy_plan": plan,
        "strategy_decision_trace": trace.to_dict(),
        "semantic_failure": error.to_dict(),
    }
