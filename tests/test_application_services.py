import asyncio
import json
from pathlib import Path

from Agents.job_service import JobExecutor
from Agents.planning_service import PlanCommand, PlanningService, build_strategy_plan
from Agents.run_manifest import sha256_file
from Agents.state_repository import SQLiteStateRepository, StateConflictError
from backend_api import _validate_products, health


def products() -> list[dict]:
    return [
        {
            "material_no": "A",
            "destination": "Sydney",
            "thickness": 1.0,
            "width": 1000.0,
            "outer_diameter": 1200.0,
            "gross_weight_tonne": 10.0,
            "net_weight_tonne": 9.8,
        }
    ]


def test_plan_requires_approval_and_repository_updates_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    products_path = run_dir / "products.json"
    run_dir.mkdir()
    products_path.write_text(json.dumps(products()), encoding="utf-8")
    repository = SQLiteStateRepository(tmp_path / "state.db")
    service = PlanningService(repository)

    result = service.create_plan(
        PlanCommand(
            products_path=products_path,
            run_dir=run_dir,
            strategy_plan=build_strategy_plan("Fast greedy", 1.0),
            data_intake_audit={"source": "test"},
            execution_mode="single_strategy",
            time_budget_seconds=1.0,
            generate_visuals=False,
        )
    )

    assert result["plan"]["status"] == "pending_approval"
    plan_id = result["plan"]["plan_id"]
    approved = service.approve_plan(plan_id, 1, approved_by="qa-operator")
    assert approved["status"] == "active"
    solution = json.loads(Path(result["solution_path"]).read_text(encoding="utf-8"))
    assert solution["research_system"]["plan"]["status"] == "active"
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    artifact = next(
        item for item in manifest["artifacts"] if item["name"] == "loading_solution"
    )
    assert artifact["sha256"] == sha256_file(Path(result["solution_path"]))
    assert manifest["governance"]["approval"]["approved_by"] == "qa-operator"


def test_event_ids_are_transactionally_idempotent(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    event = {"event_id": "EVENT-1", "expected_plan_id": "PLAN-1"}
    repository.reserve_event(event)
    try:
        repository.reserve_event(event)
    except StateConflictError:
        pass
    else:
        raise AssertionError("duplicate event id was accepted")


def test_api_health_and_product_contract() -> None:
    response = asyncio.run(health(None))
    assert response.status_code == 200
    assert json.loads(response.body)["status"] == "ok"
    assert _validate_products(products())[0]["material_no"] == "A"


def test_dynamic_event_requires_business_target_for_product_changes() -> None:
    from pydantic import ValidationError

    from Agents.api_contracts import DynamicEventRequest

    try:
        DynamicEventRequest(expected_version=1, event_type="Add products")
    except ValidationError as error:
        assert "必须选择目标目的地" in str(error)
    else:
        raise AssertionError("Add products without a destination was accepted")

    valid = DynamicEventRequest(
        expected_version=1,
        event_type="Add products",
        product_count=3,
        target_destination="  Newcastle  ",
    )
    assert valid.target_destination == "Newcastle"

    try:
        DynamicEventRequest(
            expected_version=1,
            event_type="Change destination",
            product_count=1,
            target_destination="Sydney",
        )
    except ValidationError as error:
        assert "必须选择当前目的地" in str(error)
    else:
        raise AssertionError("Destination change without a source was accepted")

    try:
        DynamicEventRequest(
            expected_version=1,
            event_type="Change destination",
            product_count=1,
            source_destination="Sydney",
            target_destination="Sydney",
        )
    except ValidationError as error:
        assert "不能相同" in str(error)
    else:
        raise AssertionError("No-op destination change was accepted")


def test_job_executor_persists_optional_enrichment_without_mutating_plan_input(
    tmp_path: Path, monkeypatch
) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    run_dir = tmp_path / "run"
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    original_input = staging_dir / "uploaded.xlsx"
    repaired_source = staging_dir / "repaired_source.xlsx"
    repair_audit = staging_dir / "data_repair_audit.json"
    original_input.write_bytes(b"immutable-original")
    repaired_source.write_bytes(b"confirmed-working-copy")
    repair_audit.write_text(
        json.dumps({"confirmed_repairs": [], "original_input_mutated": False}),
        encoding="utf-8",
    )

    def fake_enrichment(_self, supplied_products, output_path, **_kwargs):
        report = {
            "status": "completed",
            "suggestions": [
                {
                    "suggestion_id": "enr-1",
                    "material_no": "A",
                    "field": "product_summary",
                    "value": "A: 10 tonne product for Sydney.",
                }
            ],
            "rejected_suggestions": [],
            "loading_input_mutated": False,
        }
        Path(output_path).write_text(json.dumps(report), encoding="utf-8")
        assert supplied_products == products()
        return report

    monkeypatch.setattr("Agents.job_service.EnrichmentAgent.run", fake_enrichment)
    result = JobExecutor(repository).execute(
        {
            "job_type": "planning",
            "products": products(),
            "run_dir": str(run_dir),
            "execution_mode": "single_strategy",
            "time_budget_seconds": 1.0,
            "generate_visuals": False,
            "optional_enrichment_enabled": True,
            "optional_enrichment_fields": ["product_summary"],
            "optional_enrichment_max_products": 1,
            "source_input_path": str(original_input),
            "source_input_name": "uploaded.xlsx",
            "source_input_sha256": sha256_file(original_input),
            "repaired_source_path": str(repaired_source),
            "repair_audit_path": str(repair_audit),
        }
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    artifact = next(
        item for item in manifest["artifacts"] if item["name"] == "enrichment_report"
    )
    assert artifact["sha256"] == sha256_file(Path(result["enrichment_report_path"]))
    artifact_names = {item["name"] for item in manifest["artifacts"]}
    assert {
        "original_input",
        "repaired_source",
        "repaired_products_workbook",
        "data_repair_audit",
    }.issubset(artifact_names)
    assert Path(result["original_input_path"]).read_bytes() == b"immutable-original"
    assert Path(result["repaired_source_path"]).read_bytes() == b"confirmed-working-copy"
    stored_products = json.loads((run_dir / "products.json").read_text(encoding="utf-8"))
    assert stored_products == products()
