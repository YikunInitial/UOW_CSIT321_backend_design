"""Durable, API-safe dynamic recovery application service."""

from __future__ import annotations

import hashlib
import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from Agents.dynamic_replanning_agent import (
    build_dynamic_event_products,
    build_partial_solution_json,
    load_product_records,
    product_key,
    write_product_records,
)
from Agents.event_state import new_event_request
from Agents.integrity_service import verify_plan_artifacts
from Agents.planning_orchestrator import PlanningOrchestrator
from Agents.planning_service import PlanCommand, PlanningService, build_strategy_plan
from Agents.scheduling_service import assess_replanning_scope
from Agents.state_repository import SQLiteStateRepository, StateConflictError


@dataclass(frozen=True)
class ReplanningCommand:
    plan_id: str
    expected_version: int
    event_type: str
    product_count: int = 1
    source_destination: Optional[str] = None
    target_destination: Optional[str] = None
    affected_product_ids: tuple[str, ...] = ()
    new_products: tuple[Dict[str, Any], ...] = ()
    add_product_mode: str = "template_simulation"
    disabled_wagon_id: Optional[str] = None
    event_id: Optional[str] = None
    time_budget_seconds: float = 5.0
    generate_visuals: bool = False


class ReplanningService:
    """Creates a validated successor plan from a durable active plan.

    This service uses safe full replanning for API requests. The richer UI can
    still evaluate partial and BDI-guided alternatives, but both paths persist
    the same event identity and plan-version contracts.
    """

    def __init__(
        self,
        repository: SQLiteStateRepository,
        planning_service: PlanningService,
        *,
        max_products: int = 2000,
    ) -> None:
        self.repository = repository
        self.planning_service = planning_service
        self.max_products = max(1, int(max_products))

    def replan(self, command: ReplanningCommand) -> Dict[str, Any]:
        event = new_event_request(
            {
                "event_type": command.event_type,
                "product_count": int(command.product_count),
                "source_destination": command.source_destination,
                "target_destination": command.target_destination,
                "affected_product_ids": list(command.affected_product_ids),
                "new_products": [dict(item) for item in command.new_products],
                "add_product_mode": command.add_product_mode,
                "disabled_wagon_id": command.disabled_wagon_id,
                **({"event_id": command.event_id} if command.event_id else {}),
                "expected_plan_id": command.plan_id,
                "expected_plan_version": int(command.expected_version),
            }
        )
        existing_event = None
        if command.event_id:
            try:
                existing_event = self.repository.get_event(command.event_id)
            except KeyError:
                pass
            else:
                if existing_event["status"] == "completed":
                    return existing_event["result"]
                if existing_event["status"] == "processing":
                    raise StateConflictError("Dynamic event is already processing.")
                if existing_event["status"] != "failed":
                    raise StateConflictError(
                        f"Dynamic event cannot resume from {existing_event['status']}."
                    )
        plan = self.repository.get_plan(command.plan_id)
        if plan["status"] != "active":
            raise StateConflictError("Dynamic events require an active plan.")
        if int(plan["version"]) != int(command.expected_version):
            raise StateConflictError("Plan version is stale.")
        run_dir = Path(plan["run_dir"])
        if existing_event is None:
            self.repository.reserve_event(event)
        else:
            self.repository.reopen_failed_event(event)
            child_plan = self.repository.find_child_plan(
                command.plan_id, int(command.expected_version) + 1
            )
            if child_plan is not None:
                compact = {
                    "event_id": event["event_id"],
                    "plan": {
                        "plan_id": child_plan["plan_id"],
                        "version": child_plan["version"],
                        "status": child_plan["status"],
                        "parent_plan_id": child_plan["parent_plan_id"],
                    },
                    "run_dir": child_plan["run_dir"],
                    "manifest_path": child_plan["payload"]["manifest_path"],
                    "recovery_mode": "recovered_completed_successor",
                }
                self.repository.complete_event(event["event_id"], compact)
                return compact
        verify_plan_artifacts(plan)
        payload = plan["payload"]
        solution_path = Path(payload["solution_path"])
        products_path = run_dir / "products.json"
        solution = json.loads(solution_path.read_text(encoding="utf-8"))
        records = load_product_records(products_path)
        try:
            updated_records, dynamic_event = build_dynamic_event_products(
                records=records,
                baseline_solution=solution,
                event_type=event["event_type"],
                product_count=int(event["product_count"]),
                target_destination=event.get("target_destination"),
                disabled_wagon_id=event.get("disabled_wagon_id"),
                source_destination=event.get("source_destination"),
                affected_product_ids=event.get("affected_product_ids"),
                new_products=event.get("new_products"),
                add_product_mode=str(event.get("add_product_mode", "template_simulation")),
            )
            if len(updated_records) > self.max_products:
                raise ValueError(
                    f"products exceeds the {self.max_products} record limit."
                )
            dynamic_event.update(
                {
                    "event_id": event["event_id"],
                    "expected_plan_id": command.plan_id,
                    "expected_plan_version": int(command.expected_version),
                }
            )
            successor_dir = self._successor_run_dir(event["event_id"])
            successor_dir.mkdir(parents=True, exist_ok=True)
            successor_products = write_product_records(
                updated_records, successor_dir / "products.json"
            )
            strategy_path = run_dir / "strategy_plan.json"
            strategy = (
                json.loads(strategy_path.read_text(encoding="utf-8"))
                if strategy_path.exists()
                else build_strategy_plan("Agent decides", command.time_budget_seconds)
            )
            strategy = deepcopy(strategy)
            strategy["dynamic_event"] = dynamic_event
            strategy.setdefault("optimisation_config", {})["time_budget_seconds"] = (
                command.time_budget_seconds
            )
            if event["event_type"] == "Disable wagon":
                disabled = set(
                    strategy["optimisation_config"].get("disabled_wagon_ids", [])
                )
                disabled.add(str(event["disabled_wagon_id"]))
                strategy["optimisation_config"]["disabled_wagon_ids"] = sorted(disabled)
            result = self.planning_service.create_plan(
                PlanCommand(
                    products_path=successor_products,
                    run_dir=successor_dir,
                    strategy_plan=strategy,
                    data_intake_audit={
                        "source": "durable_dynamic_event",
                        "event_id": event["event_id"],
                    },
                    execution_mode="multi_agent_council",
                    scenario_id="dynamic_replanning",
                    system_variant="durable_safe_full_replan",
                    time_budget_seconds=command.time_budget_seconds,
                    generate_visuals=command.generate_visuals,
                    plan_version=int(command.expected_version) + 1,
                    parent_plan_id=command.plan_id,
                )
            )
            compact = {
                "event_id": event["event_id"],
                "plan": result["plan"],
                "run_dir": result["run_dir"],
                "manifest_path": result["manifest_path"],
                "recovery_mode": "safe_full_replan",
            }
            self.repository.complete_event(event["event_id"], compact)
            return compact
        except Exception as error:
            self.repository.fail_event(
                event["event_id"],
                {"type": type(error).__name__, "message": str(error)},
            )
            raise

    def replan_window(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a durable event batch using local recovery or safe global fallback."""
        plan_id = str(payload["plan_id"])
        expected_version = int(payload["expected_version"])
        events = [dict(item) for item in payload.get("events") or []]
        if not events:
            raise ValueError("A replanning window must contain at least one event.")
        plan = self.repository.get_plan(plan_id)
        if plan["status"] != "active":
            raise StateConflictError("Window replanning requires an active plan.")
        if int(plan["version"]) != expected_version:
            raise StateConflictError("Window targets a stale plan version.")
        verify_plan_artifacts(plan)
        run_dir = Path(plan["run_dir"])
        baseline_solution = json.loads(
            Path(plan["payload"]["solution_path"]).read_text(encoding="utf-8")
        )
        records = load_product_records(run_dir / "products.json")
        dynamic_events: list[Dict[str, Any]] = []
        updated_records = records
        disabled_wagons: set[str] = set()
        for event in events:
            updated_records, dynamic_event = build_dynamic_event_products(
                records=updated_records,
                baseline_solution=baseline_solution,
                event_type=str(event["event_type"]),
                product_count=int(event.get("product_count") or 1),
                target_destination=event.get("target_destination"),
                disabled_wagon_id=event.get("disabled_wagon_id"),
                source_destination=event.get("source_destination"),
                affected_product_ids=event.get("affected_product_ids"),
                new_products=event.get("new_products"),
                add_product_mode=str(event.get("add_product_mode", "template_simulation")),
            )
            if len(updated_records) > self.max_products:
                raise ValueError(
                    f"products exceeds the {self.max_products} record limit."
                )
            dynamic_event["event_id"] = event.get("event_id")
            dynamic_events.append(dynamic_event)
            # ``affected_wagons`` describes impact, not availability.  Only a
            # Disable wagon event may remove capacity from the optimiser.
            if str(dynamic_event.get("event_type") or "") == "disable_wagon":
                disabled_wagons.update(
                    str(item)
                    for item in dynamic_event.get("affected_wagons", [])
                    if item
                )
        destination_count = len(
            {str(item.get("destination") or "") for item in updated_records}
        )
        scope_decision = assess_replanning_scope(
            dynamic_events,
            total_product_count=len(updated_records),
            total_destination_count=destination_count,
        )
        scope_decision["window_id"] = payload.get("window_id")
        scope_decision["priority"] = payload.get("priority")
        successor_dir = self._successor_run_dir(str(payload["window_id"]))
        successor_dir.mkdir(parents=True, exist_ok=True)
        products_path = write_product_records(
            updated_records, successor_dir / "products.json"
        )
        strategy_path = run_dir / "strategy_plan.json"
        strategy = (
            json.loads(strategy_path.read_text(encoding="utf-8"))
            if strategy_path.exists()
            else build_strategy_plan("Agent decides", 5.0)
        )
        strategy = deepcopy(strategy)
        strategy["dynamic_event_window"] = {
            "window_id": payload.get("window_id"),
            "events": dynamic_events,
            "scope_decision": scope_decision,
        }
        strategy.setdefault("optimisation_config", {})["disabled_wagon_ids"] = sorted(
            disabled_wagons
        )
        local_failure: Optional[Dict[str, str]] = None
        if scope_decision["selected_scope"] == "local":
            try:
                result = self._execute_local_window_replan(
                    baseline_solution=baseline_solution,
                    all_records=updated_records,
                    events=dynamic_events,
                    products_path=products_path,
                    successor_dir=successor_dir,
                    strategy=strategy,
                    parent_plan_id=plan_id,
                    plan_version=expected_version + 1,
                    scope_decision=scope_decision,
                    disabled_wagons=tuple(sorted(disabled_wagons)),
                    generate_visuals=bool(payload.get("generate_visuals", False)),
                )
                return {
                    "window_id": payload.get("window_id"),
                    "event_ids": [item.get("event_id") for item in events],
                    "plan": result["plan"],
                    "run_dir": result["run_dir"],
                    "manifest_path": result["manifest_path"],
                    "recovery_mode": "local_replan_with_locked_wagons",
                    "scope_decision": scope_decision,
                }
            except Exception as error:
                local_failure = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                scope_decision["automatic_escalation"] = {
                    "from": "local",
                    "to": "global",
                    "reason": local_failure,
                }
        strategy["dynamic_event_window"]["scope_decision"] = scope_decision
        result = self.planning_service.create_plan(
            PlanCommand(
                products_path=products_path,
                run_dir=successor_dir,
                strategy_plan=strategy,
                data_intake_audit={
                    "source": "durable_event_window",
                    "window_id": payload.get("window_id"),
                    "event_ids": [item.get("event_id") for item in events],
                },
                execution_mode="multi_agent_council",
                scenario_id="window_dynamic_replanning",
                system_variant="impact_selected_safe_global_replan",
                time_budget_seconds=float(payload.get("time_budget_seconds", 5.0)),
                generate_visuals=bool(payload.get("generate_visuals", False)),
                plan_version=expected_version + 1,
                parent_plan_id=plan_id,
            )
        )
        return {
            "window_id": payload.get("window_id"),
            "event_ids": [item.get("event_id") for item in events],
            "plan": result["plan"],
            "run_dir": result["run_dir"],
            "manifest_path": result["manifest_path"],
            "recovery_mode": (
                "global_replan_after_local_validation_failure"
                if local_failure
                else "impact_selected_global_replan"
            ),
            "scope_decision": scope_decision,
        }

    def _execute_local_window_replan(
        self,
        *,
        baseline_solution: Dict[str, Any],
        all_records: list[Dict[str, Any]],
        events: list[Dict[str, Any]],
        products_path: Path,
        successor_dir: Path,
        strategy: Dict[str, Any],
        parent_plan_id: str,
        plan_version: int,
        scope_decision: Dict[str, Any],
        disabled_wagons: tuple[str, ...],
        generate_visuals: bool,
    ) -> Dict[str, Any]:
        affected_destinations = {
            str(destination)
            for event in events
            if event.get("event_type") != "disable_wagon"
            for destination in event.get("affected_destinations", [])
            if destination
        }
        affected_ids = {
            str(identifier)
            for event in events
            for identifier in event.get("affected_product_ids", [])
            if identifier
        }
        partial_records = [
            item
            for item in all_records
            if str(item.get("destination") or "") in affected_destinations
            or product_key(item) in affected_ids
        ]
        locked_wagons = []
        for wagon in baseline_solution.get("wagons", []):
            wagon_ids = {product_key(item) for item in wagon.get("products", [])}
            if str(wagon.get("wagon_id")) in set(disabled_wagons):
                continue
            if str(wagon.get("destination") or "") in affected_destinations:
                continue
            if wagon_ids.intersection(affected_ids):
                continue
            locked_wagons.append(dict(wagon))
        if not partial_records or not locked_wagons:
            raise ValueError("No safe bounded local scope could be isolated.")
        partial_dir = successor_dir / "local_scope"
        partial_products = write_product_records(
            partial_records, partial_dir / "products.json"
        )
        local_strategy = deepcopy(strategy)
        local_strategy["dynamic_partial_scope"] = {
            "partial_product_count": len(partial_records),
            "locked_wagon_count": len(locked_wagons),
            "locked_wagon_ids": [item.get("wagon_id") for item in locked_wagons],
            "affected_destinations": sorted(affected_destinations),
            "affected_product_ids": sorted(affected_ids),
            "reason": "Unaffected wagons are immutable during local recovery.",
        }
        started = time.perf_counter()
        partial_solution, _ = PlanningOrchestrator.run_candidate(
            partial_products, local_strategy, partial_dir, 42
        )
        aggregate_event = {
            "event_type": "window_batch",
            "affected_destinations": sorted(affected_destinations),
            "affected_product_ids": sorted(affected_ids),
            "event_ids": [item.get("event_id") for item in events],
        }
        combined = build_partial_solution_json(
            baseline_solution=baseline_solution,
            partial_solution=partial_solution,
            locked_wagons=locked_wagons,
            all_records=all_records,
            event=aggregate_event,
            partial_scope=local_strategy["dynamic_partial_scope"],
            runtime_seconds=time.perf_counter() - started,
        )
        combined_wagon_ids = [
            str(item.get("wagon_id") or "") for item in combined.get("wagons", [])
        ]
        if len(combined_wagon_ids) != len(set(combined_wagon_ids)):
            raise ValueError("Local merge produced duplicate wagon identifiers.")
        scope_decision.update(
            {
                "partial_product_count": len(partial_records),
                "locked_wagon_count": len(locked_wagons),
                "locked_wagon_ids": [item.get("wagon_id") for item in locked_wagons],
            }
        )
        return self.planning_service.persist_precomputed_plan(
            solution=combined,
            products_path=products_path,
            run_dir=successor_dir,
            strategy_plan=local_strategy,
            data_intake_audit={
                "source": "durable_event_window_local_replan",
                "window_id": scope_decision.get("window_id"),
            },
            parent_plan_id=parent_plan_id,
            plan_version=plan_version,
            system_variant="impact_selected_local_replan",
            scope_decision=scope_decision,
            unavailable_wagon_ids=disabled_wagons,
            generate_visuals=generate_visuals,
        )

    def _successor_run_dir(self, event_id: str) -> Path:
        """Allocate event runs beside plans instead of nesting plan histories.

        A successor plan can itself receive another event. Building the next
        directory below the current plan would therefore grow the path on every
        version and eventually exceed Windows path limits. Event identifiers are
        globally unique in the repository, so a flat, sanitised event directory
        under the durable run root preserves traceability without nesting.
        """
        event_text = str(event_id)
        token = re.sub(r"[^A-Za-z0-9_-]+", "_", event_text).strip("_-")
        if token.lower().startswith("event_"):
            token = token[6:]
        token = token.strip("_-") or "unknown"
        # The full event identifier remains in the repository and manifests.
        # A compact readable prefix plus digest prevents Windows MAX_PATH
        # failures when candidate and atomic-write paths are added below it.
        digest = hashlib.sha256(event_text.encode("utf-8")).hexdigest()[:10]
        compact = f"{token[:10]}_{digest}"
        return self.repository.database_path.parent / "events" / f"e_{compact}"


def build_dynamic_event_request(
    event_type: str,
    product_count: int,
    target_destination: Optional[str],
    disabled_wagon_id: Optional[str],
    source_destination: Optional[str] = None,
) -> Dict[str, Any]:
    return new_event_request(
        {
            "event_type": event_type,
            "product_count": int(product_count),
            "source_destination": source_destination,
            "target_destination": target_destination,
            "disabled_wagon_id": disabled_wagon_id,
        }
    )


def build_dynamic_environment_state(
    baseline_state: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_run_dir = str(baseline_state["run_dir"])
    plan = baseline_state["solution"].get("research_system", {}).get("plan", {})
    return {
        "baseline_run_dir": baseline_run_dir,
        "current_run_dir": baseline_run_dir,
        "current_products_path": str(baseline_state["products_path"]),
        "current_solution": baseline_state["solution"],
        "strategy_plan": baseline_state["strategy_plan"],
        "data_intake_audit": baseline_state["data_intake_audit"],
        "event_counter": 0,
        "active_plan_id": plan.get(
            "plan_id", f"baseline_{Path(baseline_run_dir).name}"
        ),
        "active_plan_version": plan.get("version", 1),
        "processed_event_ids": [],
    }


def build_dynamic_environment_summary(
    environment_state: Dict[str, Any],
) -> Dict[str, Any]:
    records = load_product_records(Path(environment_state["current_products_path"]))
    solution = environment_state["current_solution"]
    destinations = sorted(
        {
            str(record.get("destination", "")).strip()
            for record in records
            if str(record.get("destination", "")).strip()
        }
    )
    wagons = solution.get("wagon_summary", {})
    validation = solution.get("validation_summary", {})
    return {
        "event_counter": environment_state.get("event_counter", 0),
        "product_count": len(records),
        "destination_count": len(destinations),
        "destinations": destinations,
        "wagon_count": wagons.get("wagon_count", 0),
        "total_products_in_solution": wagons.get("total_products", 0),
        "average_area_utilisation": wagons.get("average_area_utilisation"),
        "violation_count": validation.get("violation_count", 0),
        "is_valid": validation.get("is_valid"),
        "current_run_dir": environment_state.get("current_run_dir"),
        "active_plan_id": environment_state.get("active_plan_id"),
        "active_plan_version": environment_state.get("active_plan_version"),
    }
