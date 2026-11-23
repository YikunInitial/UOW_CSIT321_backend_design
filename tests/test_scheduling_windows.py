from __future__ import annotations

import json
from pathlib import Path

import pytest

from Agents.api_contracts import ScheduledDynamicEventRequest
from Agents.dynamic_replanning_agent import build_dynamic_event_products
from Agents.job_service import JobExecutor
from Agents.planning_service import PlanCommand, PlanningService, build_strategy_plan
from Agents.scheduling_service import (
    EventPriorityPolicy,
    ScheduledEventService,
    assess_replanning_scope,
)
from Agents.state_repository import SQLiteStateRepository


def _active_plan(repository: SQLiteStateRepository, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in ("loading_solution.json", "run_manifest.json", "strategy_plan.json"):
        (root / name).write_text("{}", encoding="utf-8")
    repository.save_plan_record(
        plan={"plan_id": "plan-1", "version": 1, "status": "active"},
        run_dir=root,
        solution_path=root / "loading_solution.json",
        manifest_path=root / "run_manifest.json",
        strategy_path=root / "strategy_plan.json",
    )


def test_priority_policy_enforces_safety_floor_without_llm() -> None:
    decision = EventPriorityPolicy().classify(
        {"event_type": "Disable wagon"},
        requested_priority="P3",
        enable_llm_advice=False,
    )
    assert decision.priority == "P0"
    assert decision.window_seconds == 0
    assert decision.evidence["safety_floor"] == "P0"


def test_llm_priority_outage_uses_deterministic_fallback(monkeypatch) -> None:
    def unavailable(**_kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr("Agents.scheduling_service.create_llm_provider", unavailable)
    decision = EventPriorityPolicy().classify(
        {"event_type": "Change destination", "product_count": 1},
        business_instruction="urgent customer request",
        enable_llm_advice=True,
    )
    assert decision.priority == "P1"
    assert decision.evidence["llm_advisory"]["status"] == "deterministic_fallback"


def test_llm_cannot_promote_ordinary_business_urgency_to_p0(monkeypatch) -> None:
    monkeypatch.setattr(
        EventPriorityPolicy,
        "_llm_advisory",
        staticmethod(
            lambda _event, _instruction: {"status": "available", "priority": "P0"}
        ),
    )
    decision = EventPriorityPolicy().classify(
        {"event_type": "Change destination", "product_count": 1},
        business_instruction="Customer requests dispatch this shift",
    )
    assert decision.priority == "P1"
    assert decision.evidence["llm_advisory_applied"] is False


def test_events_persist_merge_escalate_and_dispatch_after_restart(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.db"
    repository = SQLiteStateRepository(database)
    _active_plan(repository, tmp_path / "run")
    service = ScheduledEventService(repository)
    first = service.schedule(
        plan_id="plan-1",
        expected_version=1,
        payload={
            "event_id": "event-add",
            "event_type": "Add products",
            "product_count": 1,
            "target_destination": "Sydney",
            "add_product_mode": "template_simulation",
            "priority": "auto",
            "window_seconds": 300,
            "enable_llm_priority_advice": False,
        },
    )
    replay = service.schedule(
        plan_id="plan-1",
        expected_version=1,
        payload={
            "event_id": "event-add",
            "event_type": "Add products",
            "product_count": 1,
            "target_destination": "Sydney",
            "add_product_mode": "template_simulation",
            "priority": "auto",
            "window_seconds": 300,
            "enable_llm_priority_advice": False,
        },
    )
    assert replay["window"]["window_id"] == first["window"]["window_id"]
    assert replay["window"]["event_count"] == 1
    second = service.schedule(
        plan_id="plan-1",
        expected_version=1,
        payload={
            "event_id": "event-disable",
            "event_type": "Disable wagon",
            "disabled_wagon_id": "W001",
            "priority": "auto",
            "enable_llm_priority_advice": False,
        },
    )
    assert first["window"]["window_id"] == second["window"]["window_id"]
    assert second["window"]["priority"] == "P0"
    assert second["window"]["event_count"] == 2

    restarted = SQLiteStateRepository(database)
    jobs = restarted.dispatch_due_windows()
    assert len(jobs) == 1
    window = restarted.get_scheduling_window(first["window"]["window_id"])
    assert window["status"] == "dispatched"
    assert window["job_id"] == jobs[0]
    job = restarted.get_job(jobs[0])
    assert job["priority"] == 0
    assert [item["event_id"] for item in job["request"]["events"]] == [
        "event-add",
        "event-disable",
    ]
    claimed = restarted.claim_next_job("worker-1")
    assert claimed and claimed["job_id"] == jobs[0]
    restarted.complete_job(jobs[0], "worker-1", {"recovery_mode": "test"})
    completed = restarted.get_scheduling_window(first["window"]["window_id"])
    assert completed["status"] == "completed"
    assert {item["status"] for item in completed["events"]} == {"completed"}


def test_queue_claims_higher_priority_first(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    repository.enqueue_job("low", {"job_type": "test"}, priority=3)
    repository.enqueue_job("urgent", {"job_type": "test"}, priority=0)
    assert repository.claim_next_job("worker")["job_id"] == "urgent"


def test_impact_assessor_selects_local_and_global() -> None:
    local = assess_replanning_scope(
        [{"event_type": "Add products", "product_count": 2, "target_destination": "A"}],
        total_product_count=100,
        total_destination_count=10,
    )
    global_scope = assess_replanning_scope(
        [
            {
                "event_type": "Add products",
                "product_count": 30,
                "target_destination": "A",
            }
        ],
        total_product_count=100,
        total_destination_count=10,
    )
    assert local["selected_scope"] == "local"
    assert global_scope["selected_scope"] == "global"
    assert (
        "affected_product_ratio_at_least_25_percent" in global_scope["global_reasons"]
    )


def test_affected_wagons_are_not_misclassified_as_unavailable() -> None:
    scope = assess_replanning_scope(
        [
            {
                "event_type": "change_destination",
                "product_count": 1,
                "source_destination": "Sydney",
                "target_destination": "Melbourne",
                "affected_wagons": ["W001", "W002"],
            }
        ],
        total_product_count=100,
        total_destination_count=10,
    )
    assert scope["affected_wagon_count"] == 2
    assert scope["disabled_wagon_count"] == 0
    assert "multiple_wagons_unavailable" not in scope["global_reasons"]
    assert scope["selected_scope"] == "local"


def test_multiple_explicitly_disabled_wagons_require_global_replanning() -> None:
    scope = assess_replanning_scope(
        [
            {
                "event_type": "disable_wagon",
                "affected_wagons": ["W001", "W002"],
            }
        ],
        total_product_count=100,
        total_destination_count=10,
    )
    assert scope["affected_wagon_count"] == 2
    assert scope["disabled_wagon_count"] == 2
    assert "multiple_wagons_unavailable" in scope["global_reasons"]
    assert scope["selected_scope"] == "global"


def test_cancel_products_removes_only_explicit_materials() -> None:
    records = [
        {"material_no": "A", "destination": "Sydney"},
        {"material_no": "B", "destination": "Sydney"},
        {"material_no": "C", "destination": "Melbourne"},
    ]
    solution = {
        "wagons": [
            {"wagon_id": "W1", "products": records[:2]},
            {"wagon_id": "W2", "products": records[2:]},
        ]
    }
    updated, event = build_dynamic_event_products(
        records=records,
        baseline_solution=solution,
        event_type="Cancel products",
        product_count=1,
        target_destination=None,
        disabled_wagon_id=None,
        affected_product_ids=["B"],
    )
    assert [item["material_no"] for item in updated] == ["A", "C"]
    assert event["event_type"] == "cancel_products"
    assert event["affected_product_ids"] == ["B"]
    assert event["affected_wagons"] == ["W1"]


def test_scheduled_cancel_contract_requires_exact_materials() -> None:
    request = ScheduledDynamicEventRequest(
        expected_version=1,
        event_type="Cancel products",
        product_count=2,
        affected_product_ids=["A", "B"],
        window_seconds=30,
    )
    assert request.priority == "auto"
    assert request.affected_product_ids == ["A", "B"]


def test_window_replanning_reapplies_total_product_limit(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    baseline_dir = tmp_path / "baseline-limit"
    baseline_dir.mkdir()
    products = [
        {
            "material_no": "LIMIT-1",
            "destination": "Sydney",
            "outer_diameter": 900.0,
            "width": 900.0,
            "gross_weight_tonne": 5.0,
        }
    ]
    products_path = baseline_dir / "products.json"
    products_path.write_text(json.dumps(products), encoding="utf-8")
    planning = PlanningService(repository)
    baseline = planning.create_plan(
        PlanCommand(
            products_path=products_path,
            run_dir=baseline_dir,
            strategy_plan=build_strategy_plan("Destination grouping", 1.0),
            data_intake_audit={"source": "test"},
            execution_mode="single_strategy",
            time_budget_seconds=1.0,
            generate_visuals=False,
        )
    )
    planning.approve_plan(baseline["plan"]["plan_id"], 1, approved_by="test")
    ScheduledEventService(repository).schedule(
        plan_id=baseline["plan"]["plan_id"],
        expected_version=1,
        payload={
            "event_id": "event-over-limit",
            "event_type": "Add products",
            "product_count": 1,
            "target_destination": "Sydney",
            "add_product_mode": "template_simulation",
            "window_seconds": 0,
            "enable_llm_priority_advice": False,
        },
    )
    job_id = repository.dispatch_due_windows()[0]
    with pytest.raises(ValueError, match="record limit"):
        JobExecutor(repository, max_products=1).execute(
            repository.get_job(job_id)["request"]
        )


def test_window_job_executes_real_local_replan_and_preserves_locked_wagons(
    tmp_path: Path,
) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    products = []
    for destination in ("Sydney", "Melbourne"):
        for index in range(5):
            products.append(
                {
                    "material_no": f"{destination[0]}-{index}",
                    "destination": destination,
                    "outer_diameter": 900.0,
                    "width": 900.0,
                    "gross_weight_tonne": 5.0,
                    "net_weight_tonne": 4.9,
                }
            )
    products_path = baseline_dir / "products.json"
    products_path.write_text(json.dumps(products), encoding="utf-8")
    planning = PlanningService(repository)
    baseline = planning.create_plan(
        PlanCommand(
            products_path=products_path,
            run_dir=baseline_dir,
            strategy_plan=build_strategy_plan("Destination grouping", 1.0),
            data_intake_audit={"source": "test"},
            execution_mode="single_strategy",
            time_budget_seconds=1.0,
            generate_visuals=False,
        )
    )
    planning.approve_plan(baseline["plan"]["plan_id"], 1, approved_by="test")
    original = json.loads(Path(baseline["solution_path"]).read_text(encoding="utf-8"))
    melbourne_ids = {
        wagon["wagon_id"]
        for wagon in original["wagons"]
        if wagon.get("destination") == "Melbourne"
    }
    ScheduledEventService(repository).schedule(
        plan_id=baseline["plan"]["plan_id"],
        expected_version=1,
        payload={
            "event_id": "event-local",
            "event_type": "Add products",
            "product_count": 1,
            "target_destination": "Sydney",
            "add_product_mode": "template_simulation",
            "priority": "P2",
            "window_seconds": 0,
            "enable_llm_priority_advice": False,
        },
    )
    merged = ScheduledEventService(repository).schedule(
        plan_id=baseline["plan"]["plan_id"],
        expected_version=1,
        payload={
            "event_id": "event-cancel",
            "event_type": "Cancel products",
            "product_count": 1,
            "affected_product_ids": ["S-0"],
            "priority": "P1",
            "window_seconds": 0,
            "enable_llm_priority_advice": False,
        },
    )
    assert merged["window"]["event_count"] == 2
    job_id = repository.dispatch_due_windows()[0]
    result = JobExecutor(repository).execute(repository.get_job(job_id)["request"])
    assert result["recovery_mode"] == "local_replan_with_locked_wagons"
    successor = json.loads(
        (Path(result["run_dir"]) / "loading_solution.json").read_text(encoding="utf-8")
    )
    locked_ids = {
        wagon["wagon_id"]
        for wagon in successor["wagons"]
        if wagon.get("dynamic_lock_status") == "locked"
    }
    assert melbourne_ids.issubset(locked_ids)
    assert successor["research_system"]["independent_validation"]["is_valid"]
    successor_ids = {
        str(product.get("material_no"))
        for wagon in successor["wagons"]
        for product in wagon.get("products", [])
    }
    assert "S-0" not in successor_ids
    assert any(item.startswith("DYN_") for item in successor_ids)
    assert len(successor_ids) == 10
    successor_strategy = json.loads(
        (Path(result["run_dir"]) / "strategy_plan.json").read_text(encoding="utf-8")
    )
    assert successor_strategy["optimisation_config"].get("disabled_wagon_ids", []) == []
