"""
Dynamic Replanning Agent
========================

Pure dynamic replanning utilities for the Streamlit demo and future
event-driven agents.

The module does not own the UI. It handles:
- dynamic event simulation
- impact scope selection
- locked wagon preservation
- affected-scope partial solution composition
- partial-vs-full replan comparison
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from Agents.optimisation_agent import evaluate_loading_solution


def load_product_records(products_path: Path) -> List[Dict[str, Any]]:
    with products_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("products.json must contain a list.")

    return records


def write_product_records(
    records: List[Dict[str, Any]],
    products_path: Path,
) -> Path:
    products_path.parent.mkdir(parents=True, exist_ok=True)

    with products_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    return products_path


def product_key(product: Dict[str, Any]) -> str:
    return str(
        product.get("material_no")
        or product.get("product_id")
        or product.get("id")
        or ""
    )


def solution_product_ids(solution: Dict[str, Any]) -> Dict[str, str]:
    product_to_wagon = {}

    for wagon in solution.get("wagons", []):
        wagon_id = wagon.get("wagon_id")
        for product in wagon.get("products", []):
            key = product_key(product)
            if key:
                product_to_wagon[key] = wagon_id

    return product_to_wagon


def list_destinations(records: List[Dict[str, Any]]) -> List[str]:
    return sorted(
        {
            str(record.get("destination"))
            for record in records
            if record.get("destination")
        }
    )


def build_dynamic_event_products(
    records: List[Dict[str, Any]],
    baseline_solution: Dict[str, Any],
    event_type: str,
    product_count: int,
    target_destination: str | None,
    disabled_wagon_id: str | None,
    source_destination: str | None = None,
    affected_product_ids: List[str] | None = None,
    new_products: List[Dict[str, Any]] | None = None,
    add_product_mode: str = "template_simulation",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    updated_records = [dict(record) for record in records]
    if not updated_records:
        raise ValueError("Dynamic replanning requires at least one current product.")
    product_count = max(1, int(product_count))
    destinations = list_destinations(updated_records)
    target_destination = target_destination or (
        destinations[0] if destinations else "DYNAMIC_DESTINATION"
    )

    if event_type == "Add products":
        added_products = []
        if add_product_mode == "provided":
            if not new_products:
                raise ValueError("Provided-product mode requires new product records.")
            existing_ids = {product_key(item) for item in updated_records}
            for item in new_products:
                product = dict(item)
                key = product_key(product)
                if not key:
                    raise ValueError("Every added product requires a material number.")
                if key in existing_ids:
                    raise ValueError(f"Added product already exists: {key}")
                existing_ids.add(key)
                product["dynamic_event"] = "added_product"
                added_products.append(product)
            product_count = len(added_products)
        else:
            source_records = list(updated_records)
            for index in range(1, product_count + 1):
                source = source_records[(index - 1) % len(source_records)]
                clone = dict(source)
                old_key = product_key(clone) or f"UNKNOWN_{index}"
                new_key = f"DYN_{uuid4().hex[:8]}_{index}_{old_key}"
                clone["material_no"] = new_key
                clone["product_id"] = new_key
                clone["destination"] = target_destination
                clone["source_row"] = None
                clone["dynamic_event"] = "added_product"
                clone["dynamic_source_material_no"] = old_key
                added_products.append(clone)

        updated_records.extend(added_products)

        added_destinations = sorted(
            {
                str(item.get("destination"))
                for item in added_products
                if item.get("destination")
            }
        )
        event = {
            "event_type": "add_products",
            "description": (
                f"Added {len(added_products)} simulated products to "
                f"{', '.join(added_destinations)}."
            ),
            "affected_product_ids": [
                product_key(product) for product in added_products
            ],
            "affected_destinations": added_destinations,
            "affected_wagons": [],
            "add_product_mode": add_product_mode,
        }
        return updated_records, event

    if event_type == "Cancel products":
        selected_ids = [
            str(item).strip()
            for item in affected_product_ids or []
            if str(item).strip()
        ]
        if not selected_ids:
            raise ValueError(
                "Cancelling products requires explicit material identifiers."
            )
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("Cancelled material identifiers must be unique.")
        records_by_id = {product_key(item): item for item in updated_records}
        missing = sorted(set(selected_ids) - set(records_by_id))
        if missing:
            raise ValueError(
                "Cancelled products are not present in the active plan: "
                + ", ".join(missing)
            )
        if len(selected_ids) >= len(updated_records):
            raise ValueError(
                "A dynamic cancellation cannot remove every active product."
            )
        cancelled = [records_by_id[item] for item in selected_ids]
        selected_set = set(selected_ids)
        remaining = [
            item for item in updated_records if product_key(item) not in selected_set
        ]
        product_to_wagon = solution_product_ids(baseline_solution)
        affected_destinations = sorted(
            {
                str(item.get("destination"))
                for item in cancelled
                if item.get("destination")
            }
        )
        event = {
            "event_type": "cancel_products",
            "description": f"Cancelled {len(cancelled)} product order line(s).",
            "affected_product_ids": selected_ids,
            "affected_destinations": affected_destinations,
            "affected_wagons": sorted(
                {
                    product_to_wagon[item]
                    for item in selected_ids
                    if item in product_to_wagon
                }
            ),
            "cancelled_products": [
                {
                    "product_id": product_key(item),
                    "destination": item.get("destination"),
                }
                for item in cancelled
            ],
        }
        return remaining, event

    if event_type == "Change destination":
        if source_destination and source_destination == target_destination:
            raise ValueError("Current and target destinations must be different.")
        selected_ids = {str(item) for item in affected_product_ids or [] if str(item)}
        eligible_records = [
            record
            for record in updated_records
            if source_destination is None
            or str(record.get("destination", "")).strip() == source_destination
        ]
        if selected_ids:
            by_id = {product_key(item): item for item in eligible_records}
            missing = sorted(selected_ids - set(by_id))
            if missing:
                raise ValueError(
                    "Selected products are not available at the current destination: "
                    + ", ".join(missing)
                )
            eligible_records = [by_id[item] for item in affected_product_ids or []]
            product_count = len(eligible_records)
        if source_destination and len(eligible_records) < product_count:
            raise ValueError(
                f"Only {len(eligible_records)} product(s) are available at "
                f"{source_destination}; {product_count} were requested."
            )
        changed_products = []

        for record in eligible_records:
            if len(changed_products) >= product_count:
                break
            if (
                record.get("destination") == target_destination
                and len(destinations) > 1
            ):
                continue
            old_destination = record.get("destination")
            record["destination"] = target_destination
            record["dynamic_event"] = "destination_changed"
            changed_products.append(
                {
                    "product_id": product_key(record),
                    "old_destination": old_destination,
                    "new_destination": target_destination,
                }
            )

        affected_ids = [item["product_id"] for item in changed_products]
        product_to_wagon = solution_product_ids(baseline_solution)
        affected_wagons = sorted(
            {
                product_to_wagon[product_id]
                for product_id in affected_ids
                if product_id in product_to_wagon
            }
        )
        affected_destinations = sorted(
            {
                destination
                for item in changed_products
                for destination in [
                    item.get("old_destination"),
                    item.get("new_destination"),
                ]
                if destination
            }
        )

        event = {
            "event_type": "change_destination",
            "description": (
                f"Changed {len(changed_products)} product destination(s) "
                f"to {target_destination}."
            ),
            "changed_products": changed_products,
            "source_destination": source_destination,
            "target_destination": target_destination,
            "selection_mode": "exact_products" if selected_ids else "automatic_count",
            "affected_product_ids": affected_ids,
            "affected_destinations": affected_destinations,
            "affected_wagons": affected_wagons,
        }
        return updated_records, event

    if event_type == "Disable wagon":
        target_wagon = None
        for wagon in baseline_solution.get("wagons", []):
            if wagon.get("wagon_id") == disabled_wagon_id:
                target_wagon = wagon
                break

        if target_wagon is None:
            raise ValueError("Please select a wagon to disable.")

        affected_ids = [
            product_key(product)
            for product in target_wagon.get("products", [])
            if product_key(product)
        ]

        event = {
            "event_type": "disable_wagon",
            "description": (
                f"Disabled current wagon {disabled_wagon_id}. Products on "
                "that wagon are marked as requiring reallocation."
            ),
            "affected_product_ids": affected_ids,
            "affected_destinations": sorted(
                {
                    product.get("destination")
                    for product in target_wagon.get("products", [])
                    if product.get("destination")
                }
            ),
            "affected_wagons": [disabled_wagon_id],
            "note": (
                "The partial replanner treats the disabled physical wagon as "
                "locked out and replans only its products."
            ),
        }
        return updated_records, event

    raise ValueError(f"Unsupported dynamic event type: {event_type}")


def select_partial_replan_scope(
    records: List[Dict[str, Any]],
    baseline_solution: Dict[str, Any],
    event: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    event_type = event.get("event_type")
    affected_destinations = set(event.get("affected_destinations", []))
    affected_product_ids = set(event.get("affected_product_ids", []))

    if event_type in {"add_products", "change_destination"}:
        partial_records = [
            record
            for record in records
            if record.get("destination") in affected_destinations
        ]
        locked_wagons = [
            dict(wagon)
            for wagon in baseline_solution.get("wagons", [])
            if wagon.get("destination") not in affected_destinations
        ]
        scope_reason = (
            "Only destinations touched by the event are replanned. Wagons "
            "serving unaffected destinations are locked and copied forward."
        )
    elif event_type == "disable_wagon":
        partial_records = [
            record for record in records if product_key(record) in affected_product_ids
        ]
        disabled_wagons = set(event.get("affected_wagons", []))
        locked_wagons = [
            dict(wagon)
            for wagon in baseline_solution.get("wagons", [])
            if wagon.get("wagon_id") not in disabled_wagons
        ]
        scope_reason = (
            "Only products from the disabled wagon are replanned. All other "
            "existing wagons are locked and preserved."
        )
    else:
        partial_records = list(records)
        locked_wagons = []
        scope_reason = "Unknown event type; partial scope falls back to all products."

    scope = {
        "event_type": event_type,
        "partial_product_count": len(partial_records),
        "locked_wagon_count": len(locked_wagons),
        "locked_wagon_ids": [wagon.get("wagon_id") for wagon in locked_wagons],
        "affected_destinations": sorted(affected_destinations),
        "affected_product_ids": sorted(affected_product_ids),
        "reason": scope_reason,
    }

    return partial_records, locked_wagons, scope


def select_dynamic_replan_strategy(
    records: List[Dict[str, Any]],
    baseline_solution: Dict[str, Any],
    event: Dict[str, Any],
    partial_scope: Dict[str, Any],
    time_budget_seconds: float | None,
) -> Dict[str, Any]:
    total_products = max(1, len(records))
    affected_product_count = len(event.get("affected_product_ids", []))
    affected_ratio = affected_product_count / total_products
    affected_destination_count = len(event.get("affected_destinations", []))
    total_destination_count = len(list_destinations(records))
    locked_wagon_count = partial_scope.get("locked_wagon_count", 0)
    baseline_wagon_count = len(baseline_solution.get("wagons", []))
    event_type = event.get("event_type")

    complexity_flags: List[str] = []
    if affected_ratio >= 0.5:
        complexity_flags.append("high_affected_ratio")
    if affected_destination_count >= 2:
        complexity_flags.append("multi_destination_impact")
    if locked_wagon_count == 0:
        complexity_flags.append("no_lockable_wagons")
    if time_budget_seconds is not None and time_budget_seconds <= 2.0:
        complexity_flags.append("tight_time_budget")

    if (
        event_type == "disable_wagon"
        and affected_ratio <= 0.25
        and time_budget_seconds is not None
        and time_budget_seconds <= 3.0
        and locked_wagon_count >= 1
    ):
        selected_strategy = "partial_replan_only"
        reason = (
            "The event is tightly localized, unaffected wagons can be locked, "
            "and the time budget is tight, so the agent accepts a fast "
            "partial-only replan."
        )
    elif (
        affected_ratio <= 0.2
        and affected_destination_count <= 1
        and locked_wagon_count >= 1
        and time_budget_seconds is not None
        and time_budget_seconds <= 2.0
    ):
        selected_strategy = "partial_replan_only"
        reason = (
            "The affected scope is very small and the time budget is tight, "
            "so the agent chooses a partial-only replan to respond quickly."
        )
    elif event_type == "disable_wagon" and affected_ratio <= 0.5:
        selected_strategy = "partial_replan_with_full_replan_benchmark"
        reason = (
            "The event is localized to one disabled wagon, so the agent keeps "
            "unaffected wagons locked, runs partial replanning for speed, and "
            "keeps a full replan benchmark for comparison."
        )
    elif (
        affected_ratio >= 0.5
        or affected_destination_count >= max(2, total_destination_count)
        or locked_wagon_count == 0
    ):
        selected_strategy = "full_replan_only"
        reason = (
            "The event affects a large or structurally broad portion of the "
            "plan, so the agent selects full replan to reduce merge risk and "
            "optimise the entire solution globally."
        )
    else:
        selected_strategy = "partial_replan_with_full_replan_benchmark"
        reason = (
            "The event impact is limited enough to preserve unaffected wagons. "
            "The agent therefore prefers partial replanning, while still "
            "running a full replan benchmark to compare quality and runtime."
        )

    return {
        "selected_strategy": selected_strategy,
        "run_partial_replan": selected_strategy
        in {
            "partial_replan_only",
            "partial_replan_with_full_replan_benchmark",
        },
        "run_full_replan": selected_strategy
        in {
            "partial_replan_with_full_replan_benchmark",
            "full_replan_only",
        },
        "reason": reason,
        "event_type": event_type,
        "affected_product_count": affected_product_count,
        "affected_ratio": round(affected_ratio, 4),
        "affected_destination_count": affected_destination_count,
        "total_destination_count": total_destination_count,
        "locked_wagon_count": locked_wagon_count,
        "baseline_wagon_count": baseline_wagon_count,
        "time_budget_seconds": time_budget_seconds,
        "complexity_flags": complexity_flags,
    }


def rename_replanned_wagons(
    wagons: List[Dict[str, Any]],
    prefix: str = "R",
) -> List[Dict[str, Any]]:
    renamed = []

    for index, wagon in enumerate(wagons, start=1):
        copied = dict(wagon)
        new_wagon_id = f"{prefix}{index:03d}"
        copied["original_wagon_id"] = wagon.get("wagon_id")
        copied["wagon_id"] = new_wagon_id
        copied["dynamic_lock_status"] = "replanned"

        products = []
        for product in copied.get("products", []):
            product_copy = dict(product)
            product_copy["wagon_id"] = new_wagon_id
            products.append(product_copy)
        copied["products"] = products
        renamed.append(copied)

    return renamed


def mark_locked_wagons(wagons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    locked = []

    for wagon in wagons:
        copied = dict(wagon)
        copied["dynamic_lock_status"] = "locked"
        products = []
        for product in copied.get("products", []):
            product_copy = dict(product)
            product_copy["dynamic_lock_status"] = "locked"
            product_copy["wagon_id"] = copied.get("wagon_id")
            products.append(product_copy)
        copied["products"] = products
        locked.append(copied)

    return locked


def build_partial_solution_json(
    baseline_solution: Dict[str, Any],
    partial_solution: Dict[str, Any],
    locked_wagons: List[Dict[str, Any]],
    all_records: List[Dict[str, Any]],
    event: Dict[str, Any],
    partial_scope: Dict[str, Any],
    runtime_seconds: float,
) -> Dict[str, Any]:
    locked = mark_locked_wagons(locked_wagons)
    replanned = rename_replanned_wagons(partial_solution.get("wagons", []))
    combined_wagons = locked + replanned
    expected_ids = [product_key(record) for record in all_records]

    score = evaluate_loading_solution(
        wagons=combined_wagons,
        expected_material_numbers=expected_ids,
        enforce_hard_constraints=False,
    )
    validation_summary = score["validation_summary"]
    total_products = sum(len(wagon.get("products", [])) for wagon in combined_wagons)
    total_weight = sum(
        float(wagon.get("current_weight", 0) or 0) for wagon in combined_wagons
    )
    destinations = sorted(
        {
            record.get("destination")
            for record in all_records
            if record.get("destination")
        }
    )
    objective_score = score["objective_score"]

    optimisation_trace = [
        {
            "stage": "dynamic_partial_scope",
            "destination": "AFFECTED_SCOPE",
            "wagon_count": len(locked_wagons),
            "product_count": partial_scope.get("partial_product_count"),
            "objective_score": None,
            "is_valid": True,
            "reason": partial_scope.get("reason"),
        },
        {
            "stage": "dynamic_partial_replan",
            "destination": "AFFECTED_SCOPE",
            "wagon_count": len(replanned),
            "product_count": partial_scope.get("partial_product_count"),
            "objective_score": partial_solution.get(
                "final_solution_score",
                {},
            ).get("objective_score"),
            "is_valid": partial_solution.get("validation_summary", {}).get("is_valid"),
            "reason": "Affected scope was replanned with OptimisationAgent.",
        },
        {
            "stage": "final_solution",
            "destination": "ALL",
            "wagon_count": len(combined_wagons),
            "product_count": total_products,
            "objective_score": objective_score,
            "average_area_utilisation": score["average_area_utilisation"],
            "is_valid": validation_summary["is_valid"],
            "violation_count": validation_summary["violation_count"],
            "overlap_violation_count": validation_summary["overlap_violation_count"],
            "boundary_violation_count": validation_summary["boundary_violation_count"],
            "overweight_violation_count": validation_summary[
                "overweight_violation_count"
            ],
            "coverage_violation_count": validation_summary["coverage_violation_count"],
        },
    ]

    performance_summary = {
        "baseline_seconds": 0.0,
        "candidate_seconds": round(runtime_seconds, 4),
        "total_seconds": round(runtime_seconds, 4),
        "baseline_objective": objective_score,
        "final_objective": objective_score,
        "accepted_improvement": 0.0,
        "accepted_improvement_percent": 0.0,
        "baseline_wagon_count": len(combined_wagons),
        "final_wagon_count": len(combined_wagons),
        "accepted_candidate_count": 1,
        "rejected_candidate_count": 0,
    }

    return {
        "solution_metadata": {
            "generated_by": "DynamicReplanningAgent",
            "version": "1.0_destination_or_wagon_scope_partial_replan",
            "runtime_seconds": runtime_seconds,
            "random_seed": baseline_solution.get("solution_metadata", {}).get(
                "random_seed"
            ),
            "note": (
                "Dynamic partial replan composed locked baseline wagons with "
                "replanned affected-scope wagons, then revalidated the full "
                "combined solution."
            ),
        },
        "strategy_execution": {
            "requested_strategy": "dynamic_partial_replan",
            "requested_scenario_type": "dynamic_online_demo",
            "applied_strategy": "partial_replan_with_locked_wagons",
            "actual_batch_scenario": "dynamic_event",
            "execution_config": {
                "locked_wagons": partial_scope.get("locked_wagon_ids", []),
                "affected_destinations": partial_scope.get(
                    "affected_destinations",
                    [],
                ),
                "affected_product_ids": partial_scope.get(
                    "affected_product_ids",
                    [],
                ),
            },
            "resolution_notes": [
                partial_scope.get("reason"),
                "Full combined solution was revalidated after merging.",
            ],
            "fallback_reasons": [],
            "algorithm_policy": {
                "mode": "dynamic_partial_replan",
                "selected_pipeline": [
                    "lock_unaffected_wagons",
                    "replan_affected_scope",
                    "merge_and_validate",
                ],
                "complexity_class": "dynamic",
            },
        },
        "llm_strategy_plan_summary": {},
        "input_summary": {
            "product_count": len(all_records),
            "dropped_duplicate_count": 0,
            "dropped_duplicates": [],
            "destinations": destinations,
        },
        "dynamic_event": event,
        "dynamic_partial_scope": partial_scope,
        "initial_solution_score": score,
        "final_solution_score": score,
        "validation_summary": validation_summary,
        "anytime_summary": {
            "enabled": True,
            "stage_count": len(optimisation_trace),
            "accepted_stage_count": 1,
            "candidate_stage_count": 1,
            "final_accepted_stage": "dynamic_partial_replan",
            "guarantee": (
                "The combined partial solution is accepted only after full "
                "hard-constraint and coverage validation."
            ),
        },
        "performance_summary": performance_summary,
        "algorithm_decision_trace": [
            {
                "step": "dynamic_event_detection",
                "decision": event.get("event_type"),
                "reason": event.get("description"),
                "details": event,
            },
            {
                "step": "impact_scope_selection",
                "decision": "partial_scope",
                "reason": partial_scope.get("reason"),
                "details": partial_scope,
            },
            {
                "step": "final_solution",
                "decision": (
                    "accepted" if validation_summary["is_valid"] else "rejected"
                ),
                "reason": (
                    "Partial solution was merged with locked wagons and "
                    "validated globally."
                ),
                "details": {
                    "objective_score": objective_score,
                    "is_valid": validation_summary["is_valid"],
                    "violation_count": validation_summary["violation_count"],
                    "wagon_count": len(combined_wagons),
                    "total_products": total_products,
                },
            },
        ],
        "optimisation_trace": optimisation_trace,
        "simulated_annealing_log": {
            "enabled": False,
            "executed": False,
            "note": "Dynamic partial MVP uses fast affected-scope replanning.",
        },
        "wagon_summary": {
            "wagon_count": len(combined_wagons),
            "total_products": total_products,
            "total_weight": total_weight,
            "average_area_utilisation": score["average_area_utilisation"],
            "wagon_ids": [wagon.get("wagon_id") for wagon in combined_wagons],
        },
        "wagons": combined_wagons,
    }


def compare_solutions(
    before_solution: Dict[str, Any],
    after_solution: Dict[str, Any],
) -> Dict[str, Any]:
    before_perf = before_solution.get("performance_summary", {})
    after_perf = after_solution.get("performance_summary", {})
    before_validation = before_solution.get("validation_summary", {})
    after_validation = after_solution.get("validation_summary", {})
    before_summary = before_solution.get("wagon_summary", {})
    after_summary = after_solution.get("wagon_summary", {})

    before_objective = before_perf.get(
        "final_objective",
        before_solution.get("final_solution_score", {}).get("objective_score"),
    )
    after_objective = after_perf.get(
        "final_objective",
        after_solution.get("final_solution_score", {}).get("objective_score"),
    )

    return {
        "before_wagon_count": before_summary.get("wagon_count"),
        "after_wagon_count": after_summary.get("wagon_count"),
        "wagon_count_delta": (
            after_summary.get("wagon_count", 0) - before_summary.get("wagon_count", 0)
        ),
        "before_product_count": before_summary.get("total_products"),
        "after_product_count": after_summary.get("total_products"),
        "product_count_delta": (
            after_summary.get("total_products", 0)
            - before_summary.get("total_products", 0)
        ),
        "before_objective": before_objective,
        "after_objective": after_objective,
        "objective_delta": (
            after_objective - before_objective
            if isinstance(before_objective, (int, float))
            and isinstance(after_objective, (int, float))
            else None
        ),
        "before_violation_count": before_validation.get("violation_count"),
        "after_violation_count": after_validation.get("violation_count"),
        "after_is_valid": after_validation.get("is_valid"),
    }


def build_dynamic_replan_report(
    event: Dict[str, Any],
    before_solution: Dict[str, Any],
    after_solution: Dict[str, Any],
    before_products_path: Path,
    after_products_path: Path,
    dynamic_solution_path: Path,
    dynamic_audit_report: Dict[str, Any],
    full_replan_solution: Dict[str, Any] | None = None,
    partial_solution: Dict[str, Any] | None = None,
    partial_solution_path: Path | None = None,
    partial_audit_report: Dict[str, Any] | None = None,
    partial_scope: Dict[str, Any] | None = None,
    strategy_decision: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    all_destinations = set(
        before_solution.get("input_summary", {}).get("destinations", [])
    )
    affected_destinations = set(event.get("affected_destinations", []))

    affected_wagons = event.get("affected_wagons", [])
    affected_products = event.get("affected_product_ids", [])
    full_replan_comparison = (
        compare_solutions(before_solution, full_replan_solution)
        if full_replan_solution is not None
        else None
    )

    replan_strategy = (
        strategy_decision.get("selected_strategy")
        if strategy_decision
        else (
            "partial_replan_with_full_replan_benchmark"
            if partial_solution is not None
            else "full_replan_safe_fallback"
        )
    )
    if event.get("event_type") in {"add_products", "change_destination"}:
        future_strategy = "destination_level_partial_replan"
    else:
        future_strategy = "locked_wagon_partial_replan"

    report = {
        "event": event,
        "impact_analysis": {
            "affected_product_count": len(affected_products),
            "affected_product_ids": affected_products,
            "affected_destinations": sorted(affected_destinations),
            "unaffected_destinations": sorted(all_destinations - affected_destinations),
            "affected_wagons": affected_wagons,
        },
        "replanning_decision": {
            "selected_strategy": replan_strategy,
            "initial_selected_strategy": (
                strategy_decision.get("initial_selected_strategy")
                if strategy_decision
                else None
            ),
            "reason": (
                strategy_decision.get("reason")
                if strategy_decision
                else (
                    "Agent locks unaffected wagons, replans only the affected "
                    "scope, and also runs full replan as a benchmark for quality "
                    "and runtime comparison."
                    if partial_solution is not None
                    else "MVP selects full replan as the safest fallback after an "
                    "environment change. It preserves correctness while exposing "
                    "the event, impact, replan, validation, and comparison loop."
                )
            ),
            "next_upgrade_path": future_strategy,
            "strategy_decision": strategy_decision or {},
        },
        "reconsideration_loop": (
            strategy_decision.get("reconsideration_state") if strategy_decision else {}
        ),
        "agent_decision_state": (
            strategy_decision.get("agent_decision_state") if strategy_decision else {}
        ),
        "full_replan_comparison": full_replan_comparison,
        "validation": after_solution.get("validation_summary", {}),
        "paths": {
            "before_products_path": str(before_products_path),
            "after_products_path": str(after_products_path),
            "dynamic_solution_path": str(dynamic_solution_path),
        },
        "audit_sign_off": dynamic_audit_report.get("final_sign_off", {}),
    }

    if partial_solution is not None:
        partial_comparison = compare_solutions(
            before_solution,
            partial_solution,
        )
        benchmark_solution = full_replan_solution or after_solution
        full_perf = benchmark_solution.get("performance_summary", {})
        partial_perf = partial_solution.get("performance_summary", {})
        partial_objective = partial_comparison.get("after_objective")
        full_objective = (full_replan_comparison or {}).get("after_objective")
        partial_seconds = partial_perf.get("total_seconds")
        full_seconds = full_perf.get("total_seconds")
        report["partial_replan_comparison"] = partial_comparison
        report["partial_scope"] = partial_scope or {}
        if full_replan_comparison is not None:
            report["partial_vs_full"] = {
                "partial_wagon_count": partial_comparison.get("after_wagon_count"),
                "full_wagon_count": full_replan_comparison.get("after_wagon_count"),
                "partial_objective": partial_objective,
                "full_objective": full_objective,
                "partial_minus_full_objective": (
                    partial_objective - full_objective
                    if isinstance(partial_objective, (int, float))
                    and isinstance(full_objective, (int, float))
                    else None
                ),
                "partial_seconds": partial_seconds,
                "full_seconds": full_seconds,
                "partial_seconds_saved": (
                    full_seconds - partial_seconds
                    if isinstance(full_seconds, (int, float))
                    and isinstance(partial_seconds, (int, float))
                    else None
                ),
                "partial_valid": partial_solution.get(
                    "validation_summary",
                    {},
                ).get("is_valid"),
                "full_valid": benchmark_solution.get(
                    "validation_summary",
                    {},
                ).get("is_valid"),
            }
        report["paths"]["partial_solution_path"] = (
            str(partial_solution_path) if partial_solution_path else None
        )
        report["partial_audit_sign_off"] = (partial_audit_report or {}).get(
            "final_sign_off", {}
        )

    return report
