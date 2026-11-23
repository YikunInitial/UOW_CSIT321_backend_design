from __future__ import annotations

import json
from pathlib import Path

from Agents.continuous_stream_service import ContinuousStreamController
from Agents.scheduling_service import ScheduledEventService
from Agents.state_repository import SQLiteStateRepository
from worker import JobWorker


def _active_plan(repository: SQLiteStateRepository, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    products = [
        {
            "material_no": "M-001",
            "destination": "Sydney",
            "outer_diameter": 900.0,
            "width": 900.0,
            "gross_weight_tonne": 5.0,
        }
    ]
    (root / "products.json").write_text(json.dumps(products), encoding="utf-8")
    for name in ("loading_solution.json", "run_manifest.json", "strategy_plan.json"):
        (root / name).write_text("{}", encoding="utf-8")
    repository.save_plan_record(
        plan={"plan_id": "plan-1", "version": 1, "status": "active"},
        run_dir=root,
        solution_path=root / "loading_solution.json",
        manifest_path=root / "run_manifest.json",
        strategy_path=root / "strategy_plan.json",
    )


def test_continuous_stream_is_single_instance_and_stop_is_idempotent(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    _active_plan(repository, tmp_path / "plan")
    first = repository.start_continuous_stream(
        plan_id="plan-1",
        expected_version=1,
        min_interval_seconds=35,
        max_interval_seconds=65,
        max_products=250,
        created_by="test",
    )
    replay = repository.start_continuous_stream(
        plan_id="plan-1",
        expected_version=1,
        min_interval_seconds=35,
        max_interval_seconds=65,
        max_products=250,
        created_by="test",
    )
    assert replay["stream_id"] == first["stream_id"]
    assert repository.get_active_continuous_stream()["stream_id"] == first["stream_id"]
    stopped = repository.request_continuous_stream_stop(first["stream_id"])
    assert stopped["status"] == "stopped"
    assert repository.request_continuous_stream_stop(first["stream_id"])["status"] == "stopped"


def test_controller_persists_one_bounded_nonvisual_event(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    _active_plan(repository, tmp_path / "plan")
    stream = repository.start_continuous_stream(
        plan_id="plan-1",
        expected_version=1,
        min_interval_seconds=1,
        max_interval_seconds=1,
        max_products=10,
        created_by="test",
    )
    controller = ContinuousStreamController(repository, worker_id="worker-1")
    assert controller.tick() is True
    current = repository.get_continuous_stream(stream["stream_id"])
    assert current["current_window_id"]
    window = repository.get_scheduling_window(current["current_window_id"])
    event = window["events"][0]["payload"]
    assert event["event_id"] == f"stream-{stream['stream_id']}-cycle-1"
    assert event["generate_visuals"] is False
    assert 1 <= int(event["product_count"]) <= 3
    assert len(current["recent_events"]) == 1
    stopping = repository.request_continuous_stream_stop(stream["stream_id"])
    assert stopping["status"] == "stopping"


def test_controller_stops_before_exceeding_product_limit(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    _active_plan(repository, tmp_path / "plan")
    stream = repository.start_continuous_stream(
        plan_id="plan-1",
        expected_version=1,
        min_interval_seconds=1,
        max_interval_seconds=1,
        max_products=1,
        created_by="test",
    )
    controller = ContinuousStreamController(repository, worker_id="worker-1")
    assert controller.tick() is True
    stopped = repository.get_continuous_stream(stream["stream_id"])
    assert stopped["status"] == "stopped"
    assert stopped["last_error"]["code"] == "STREAM_PRODUCT_LIMIT_REACHED"
    assert repository.list_scheduling_windows(limit=10) == []


def test_lost_response_reuses_deterministic_event_instead_of_duplicating(
    tmp_path: Path,
) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    _active_plan(repository, tmp_path / "plan")
    stream = repository.start_continuous_stream(
        plan_id="plan-1",
        expected_version=1,
        min_interval_seconds=1,
        max_interval_seconds=1,
        max_products=10,
        created_by="test",
    )
    claimed = repository.claim_continuous_stream("crashed-worker")
    scenario = ContinuousStreamController._scenario(claimed, json.loads(
        (tmp_path / "plan" / "products.json").read_text(encoding="utf-8")
    ))
    scenario["event_id"] = f"stream-{stream['stream_id']}-cycle-1"
    first = ScheduledEventService(repository).schedule(
        plan_id="plan-1", expected_version=1, payload=scenario
    )
    # Simulate losing the response before current_window_id was recorded.
    repository.update_continuous_stream(stream["stream_id"], "crashed-worker")

    restarted = ContinuousStreamController(repository, worker_id="new-worker")
    assert restarted.tick() is True
    recovered = repository.get_continuous_stream(stream["stream_id"])
    assert recovered["current_window_id"] == first["window"]["window_id"]
    assert len(repository.list_scheduling_windows(limit=10)) == 1
    assert repository.get_scheduling_window(first["window"]["window_id"])["event_count"] == 1


def test_production_worker_never_advances_simulation_stream(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("STEEL_ENV", "production")
    monkeypatch.setenv("STEEL_ALLOW_SIMULATION_EVENTS", "false")
    worker = JobWorker(SQLiteStateRepository(tmp_path / "state.db"))
    assert worker.stream_controller is None

