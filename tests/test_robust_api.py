import io
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
from starlette.requests import Request
from starlette.testclient import TestClient

import backend_api
from Agents.state_repository import SQLiteStateRepository


def test_openapi_covers_every_business_route() -> None:
    documented = set(backend_api.openapi_document()["paths"])
    excluded = {"/health", "/health/live", "/health/ready", "/metrics", "/openapi.json"}
    routed = {
        route.path
        for route in backend_api.app.routes
        if getattr(route, "path", None) not in excluded
    }
    assert routed <= documented


def test_continuous_stream_api_is_durable_single_instance_and_stoppable(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = _configure_isolated_api(tmp_path, monkeypatch)
    repo = SQLiteStateRepository(run_root / "state.db")
    plan_dir = run_root / "active-plan"
    plan_dir.mkdir(parents=True)
    (plan_dir / "products.json").write_text(
        json.dumps([_product()]), encoding="utf-8"
    )
    for name in ("loading_solution.json", "run_manifest.json", "strategy_plan.json"):
        (plan_dir / name).write_text("{}", encoding="utf-8")
    repo.save_plan_record(
        plan={"plan_id": "plan-active", "version": 1, "status": "active"},
        run_dir=plan_dir,
        solution_path=plan_dir / "loading_solution.json",
        manifest_path=plan_dir / "run_manifest.json",
        strategy_path=plan_dir / "strategy_plan.json",
    )
    payload = {
        "expected_version": 1,
        "min_interval_seconds": 35,
        "max_interval_seconds": 65,
        "max_products": 20,
    }
    with TestClient(backend_api.app) as client:
        first = client.post("/v1/plans/plan-active/continuous-stream", json=payload)
        replay = client.post("/v1/plans/plan-active/continuous-stream", json=payload)
        active = client.get("/v1/continuous-streams/active")
        stream_id = first.json()["stream"]["stream_id"]
        stopped = client.post(f"/v1/continuous-streams/{stream_id}/stop", json={})

    assert first.status_code == 202
    assert replay.status_code == 202
    assert replay.json()["stream"]["stream_id"] == stream_id
    assert active.json()["stream"]["stream_id"] == stream_id
    assert stopped.status_code == 202
    assert stopped.json()["stream"]["status"] == "stopped"


def _product(material_no: str = "A") -> dict:
    return {
        "material_no": material_no,
        "destination": "Sydney",
        "width": 1000,
        "outer_diameter": 1200,
        "gross_weight_tonne": 10,
    }


def _configure_isolated_api(tmp_path: Path, monkeypatch, *, max_jobs: int = 1000):
    run_root = tmp_path / "api"
    monkeypatch.setattr(backend_api, "API_RUN_ROOT", run_root)
    monkeypatch.setattr(backend_api, "API_DATABASE", run_root / "state.db")
    monkeypatch.setattr(backend_api, "API_KEY", "")
    monkeypatch.setattr(backend_api, "EMBEDDED_WORKER_ENABLED", False)
    monkeypatch.setattr(
        backend_api,
        "SETTINGS",
        replace(
            backend_api.SETTINGS,
            max_active_jobs=max_jobs,
            require_idempotency_key=False,
        ),
    )
    backend_api._RATE_LIMIT_WINDOWS.clear()
    return run_root


def test_api_idempotency_replay_returns_original_job(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = _configure_isolated_api(tmp_path, monkeypatch)
    headers = {"idempotency-key": "planning-request-001"}
    payload = {"products": [_product()]}

    with TestClient(backend_api.app) as client:
        first = client.post("/v1/planning-runs", headers=headers, json=payload)
        second = client.post("/v1/planning-runs", headers=headers, json=payload)

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["job_id"] == second.json()["job_id"]
    assert not first.json()["idempotent_replay"]
    assert second.json()["idempotent_replay"]
    repository = SQLiteStateRepository(run_root / "state.db")
    assert repository.job_counts()["queued"] == 1
    run_dir = Path(repository.get_job(first.json()["job_id"])["request"]["run_dir"])
    assert not run_dir.exists()


def test_api_rejects_idempotency_key_reuse_for_different_body(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_isolated_api(tmp_path, monkeypatch)
    headers = {"idempotency-key": "planning-request-002"}

    with TestClient(backend_api.app) as client:
        first = client.post(
            "/v1/planning-runs", headers=headers, json={"products": [_product("A")]}
        )
        conflict = client.post(
            "/v1/planning-runs", headers=headers, json={"products": [_product("B")]}
        )

    assert first.status_code == 202
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"


def test_api_applies_queue_backpressure_and_remains_live(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_isolated_api(tmp_path, monkeypatch, max_jobs=1)

    with TestClient(backend_api.app) as client:
        accepted = client.post("/v1/planning-runs", json={"products": [_product("A")]})
        saturated = client.post("/v1/planning-runs", json={"products": [_product("B")]})
        live = client.get("/health/live")

    assert accepted.status_code == 202
    assert saturated.status_code == 503
    assert saturated.headers["retry-after"] == "30"
    assert live.status_code == 200


def test_unhandled_dependency_error_is_structured_and_process_stays_live(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_isolated_api(tmp_path, monkeypatch)

    def broken_repository():
        raise OSError("sensitive internal storage detail")

    monkeypatch.setattr(backend_api, "repository", broken_repository)
    with TestClient(backend_api.app, raise_server_exceptions=False) as client:
        failed = client.get("/metrics", headers={"x-request-id": "fault-test-001"})
        live = client.get("/health/live")

    assert failed.status_code == 500
    assert failed.json()["error"] == {
        "code": "INTERNAL_SERVICE_ERROR",
        "message": "服务未能完成该请求，请记录支持编号后联系系统管理员。",
    }
    assert failed.headers["x-request-id"] == "fault-test-001"
    assert "sensitive" not in failed.text
    assert live.status_code == 200


def test_low_disk_floor_rejects_new_work_but_keeps_api_live(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_isolated_api(tmp_path, monkeypatch)
    monkeypatch.setattr(
        backend_api,
        "SETTINGS",
        replace(backend_api.SETTINGS, minimum_free_disk_bytes=10**30),
    )

    with TestClient(backend_api.app) as client:
        rejected = client.post("/v1/planning-runs", json={"products": [_product()]})
        live = client.get("/health/live")

    assert rejected.status_code == 507
    assert rejected.json()["error"]["code"] == "INSUFFICIENT_STORAGE"
    assert live.status_code == 200


def test_browser_intake_uses_server_side_validation_and_confirmation(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_isolated_api(tmp_path, monkeypatch)
    csv_payload = (
        "材料号,厚度,宽度,外径,毛重,净重,目的地,备注\n"
        "C001,1.2,1000,1200,10.0,9.8,Sydney, normal delivery \n"
    ).encode("utf-8")

    with TestClient(backend_api.app) as client:
        inspected = client.post(
            "/v1/data-intakes",
            files={"file": ("products.csv", csv_payload, "text/csv")},
        )
        assert inspected.status_code == 201
        report = inspected.json()
        assert report["record_count"] == 1
        assert report["planning_ready"] is True
        assert report["repair_analysis"]["suggestions"]

        not_reviewed = client.post(
            f"/v1/data-intakes/{report['intake_id']}/confirm",
            json={"accepted_repair_ids": []},
        )
        assert not_reviewed.status_code == 422

        suggestion_ids = [
            item["repair_id"] for item in report["repair_analysis"]["suggestions"]
        ]
        confirmed = client.post(
            f"/v1/data-intakes/{report['intake_id']}/confirm",
            json={
                "accepted_repair_ids": suggestion_ids,
                "review_completed": True,
                "reviewed_by": "test-operator",
            },
        )

    assert confirmed.status_code == 200
    assert len(confirmed.json()["products"]) == 1
    audit = confirmed.json()["data_intake_audit"]
    assert audit["data_repair"]["confirmed_repair_count"] == 1
    assert audit["data_repair"]["original_input_mutated"] is False


def test_plan_history_is_compact_and_sorted(tmp_path: Path, monkeypatch) -> None:
    run_root = _configure_isolated_api(tmp_path, monkeypatch)
    repo = SQLiteStateRepository(run_root / "state.db")
    for index in range(2):
        run_dir = run_root / f"run-{index}"
        run_dir.mkdir(parents=True)
        repo.save_plan_record(
            plan={"plan_id": f"plan-{index}", "version": 1, "status": "draft"},
            run_dir=run_dir,
            solution_path=run_dir / "loading_solution.json",
            manifest_path=run_dir / "run_manifest.json",
            strategy_path=run_dir / "strategy_plan.json",
        )

    with TestClient(backend_api.app) as client:
        response = client.get("/v1/plans?limit=1")

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert {
        "plan_id", "version", "status", "parent_plan_id", "created_at", "updated_at"
    }.issubset(response.json()["items"][0])


def test_browser_intake_accepts_supported_loading_json(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_isolated_api(tmp_path, monkeypatch)
    payload = b'[{"product_id":"P1","length":1200,"width":1000,"weight":10,"destination":"Sydney"}]'

    with TestClient(backend_api.app) as client:
        inspected = client.post(
            "/v1/data-intakes",
            files={"file": ("products.json", payload, "application/json")},
        )
        assert inspected.status_code == 201
        report = inspected.json()
        assert report["planning_ready"] is True
        confirmed = client.post(
            f"/v1/data-intakes/{report['intake_id']}/confirm",
            json={"accepted_repair_ids": [], "review_completed": True},
        )

    assert confirmed.status_code == 200
    assert confirmed.json()["products"][0]["product_id"] == "P1"


def test_repair_selection_preview_matches_streamlit_rerun_logic(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_isolated_api(tmp_path, monkeypatch)
    source = (
        "material_no,thickness,width,outer_diameter,gross_weight_tonne,net_weight_tonne,destination\n"
        "C001,,1000,1200,10.0,9.8,Sydney\n"
    ).encode("utf-8")
    master = "material_no,thickness\nC001,1.2\n".encode("utf-8")

    with TestClient(backend_api.app) as client:
        inspected = client.post(
            "/v1/data-intakes",
            files={
                "file": ("products.csv", source, "text/csv"),
                "master_file": ("master.csv", master, "text/csv"),
            },
        ).json()
        assert inspected["planning_ready"] is False
        repair_id = next(
            item["repair_id"]
            for item in inspected["repair_analysis"]["suggestions"]
            if item["field"] == "thickness"
        )
        preview = client.post(
            f"/v1/data-intakes/{inspected['intake_id']}/preview",
            json={"accepted_repair_ids": [repair_id]},
        )

    assert preview.status_code == 200
    assert preview.json()["planning_ready"] is True
    assert preview.json()["invalid_product_count"] == 0
    assert preview.json()["selected_repair_ids"] == [repair_id]


def test_browser_intake_classifies_embedded_wagon_sections_without_losing_products(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_isolated_api(tmp_path, monkeypatch)
    source = (
        "材料号,厚度,宽度,外径,毛重,净重,到站\n"
        "P001,1.2,1000,1200,10.0,9.8,Sydney\n"
        "车皮号,钢支架,数量,钢支架号,,,\n"
        "1660216,C1,3,13102,5479,1300,\n"
        "材料号,厚度,宽度,外径,毛重,净重,到站\n"
        "P002,1.5,1100,1300,11.0,10.8,Melbourne\n"
    ).encode("utf-8")

    with TestClient(backend_api.app) as client:
        inspected = client.post(
            "/v1/data-intakes",
            files={"file": ("mixed_report.csv", source, "text/csv")},
        )
        assert inspected.status_code == 201
        report = inspected.json()
        assert report["planning_ready"] is True
        assert report["record_count"] == 2
        assert report["candidate_count"] == 2
        assert report["invalid_product_count"] == 0
        assert report["row_classification"]["reason_counts"] == {
            "auxiliary_section_header": 1,
            "auxiliary_section_record": 1,
            "repeated_product_header": 1,
        }
        confirmed = client.post(
            f"/v1/data-intakes/{report['intake_id']}/confirm",
            json={"accepted_repair_ids": [], "review_completed": True},
        )

    assert confirmed.status_code == 200
    assert [item["material_no"] for item in confirmed.json()["products"]] == [
        "P001",
        "P002",
    ]


def test_browser_intake_still_blocks_an_incomplete_product_row(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_isolated_api(tmp_path, monkeypatch)
    source = (
        "材料号,厚度,宽度,外径,毛重,净重,到站\n"
        "P001,1.2,1000,1200,10.0,9.8,Sydney\n"
        "P002,1.5,1100,1300,11.0,10.8,\n"
    ).encode("utf-8")

    with TestClient(backend_api.app) as client:
        inspected = client.post(
            "/v1/data-intakes",
            files={"file": ("incomplete_product.csv", source, "text/csv")},
        )

    assert inspected.status_code == 201
    report = inspected.json()
    assert report["planning_ready"] is False
    assert report["record_count"] == 1
    assert report["candidate_count"] == 2
    assert report["invalid_product_count"] == 1
    assert report["row_classification"]["structural_row_count"] == 0


def _excel_bytes(sheets: dict[str, pd.DataFrame], *, startrows: dict[str, int] | None = None) -> bytes:
    payload = io.BytesIO()
    with pd.ExcelWriter(payload, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False, startrow=(startrows or {}).get(name, 0))
    return payload.getvalue()


def test_excel_structure_discovery_finds_title_row_and_product_sheet(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_isolated_api(tmp_path, monkeypatch)
    products = pd.DataFrame([
        {"材料号": "S001", "厚度": 1.2, "宽度": 1000, "外径": 1200, "毛重": 10, "净重": 9.8, "到站": "Sydney"}
    ])
    payload = _excel_bytes(
        {"说明": pd.DataFrame({"使用说明": ["请勿修改模板"]}), "产品清单": products},
        startrows={"产品清单": 3},
    )

    with TestClient(backend_api.app) as client:
        response = client.post(
            "/v1/data-intakes",
            files={"file": ("多工作表产品清单.xlsx", payload, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    assert response.status_code == 201
    report = response.json()
    assert report["planning_ready"] is True
    assert report["input_structure"]["selected_sheet"] == "产品清单"
    assert report["input_structure"]["header_row"] == 4
    assert report["record_count"] == 1


def test_duplicate_material_number_is_blocked_during_precheck(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_isolated_api(tmp_path, monkeypatch)
    rows = pd.DataFrame([
        {"材料号": "D001", "厚度": 1.2, "宽度": 1000, "外径": 1200, "毛重": 10, "净重": 9.8, "到站": "Sydney"},
        {"材料号": "D001", "厚度": 1.2, "宽度": 1000, "外径": 1200, "毛重": 10, "净重": 9.8, "到站": "Melbourne"},
    ])
    with TestClient(backend_api.app) as client:
        response = client.post(
            "/v1/data-intakes",
            files={"file": ("重复材料号.xlsx", _excel_bytes({"产品": rows}), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
        report = response.json()
        confirmation = client.post(
            f"/v1/data-intakes/{report['intake_id']}/confirm",
            json={"accepted_repair_ids": [], "review_completed": True},
        )

    assert response.status_code == 201
    assert report["planning_ready"] is False
    assert report["duplicate_product_count"] == 1
    assert report["blocking_issues"][0]["material_no"] == "D001"
    assert report["blocking_issues"][0]["source_rows"] == [2, 3]
    assert confirmation.status_code == 422
    assert "重复材料号" in confirmation.json()["error"]["message"]


def test_llm_assistance_is_a_blocking_intermediate_state(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_isolated_api(tmp_path, monkeypatch)
    unknown_columns = pd.DataFrame([
        {"编号列": "L001", "规格一": 1.2, "规格二": 1000, "规格三": 1200, "重量一": 10, "重量二": 9.8, "地点列": "Sydney"}
    ])

    def fake_llm_recovery(uploaded_file, run_dir, **kwargs):
        products = [{
            "material_no": "L001", "thickness": 1.2, "width": 1000,
            "outer_diameter": 1200, "gross_weight_tonne": 10,
            "net_weight_tonne": 9.8, "destination": "Sydney",
        }]
        output = run_dir / "products.json"
        output.write_text(json.dumps(products, ensure_ascii=False), encoding="utf-8")
        return output, {"llm_called": True, "field_mapping": {"material_no": "编号列"}}

    monkeypatch.setattr(
        "Agents.business_intake_service.save_uploaded_products", fake_llm_recovery
    )
    with TestClient(backend_api.app) as client:
        inspected = client.post(
            "/v1/data-intakes",
            files={"file": ("非标准字段.xlsx", _excel_bytes({"业务清单": unknown_columns}), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        ).json()
        blocked = client.post(
            f"/v1/data-intakes/{inspected['intake_id']}/confirm",
            json={"accepted_repair_ids": [], "review_completed": True},
        )
        assisted = client.post(f"/v1/data-intakes/{inspected['intake_id']}/assist")
        confirmed = client.post(
            f"/v1/data-intakes/{inspected['intake_id']}/confirm",
            json={"accepted_repair_ids": [], "review_completed": True},
        )

    assert inspected["planning_ready"] is False
    assert inspected["requires_assistance"] is True
    assert inspected["assistance_status"] == "pending"
    assert blocked.status_code == 422
    assert assisted.status_code == 200
    assert assisted.json()["planning_ready"] is True
    assert assisted.json()["assistance_status"] == "completed"
    assert confirmed.status_code == 200


def test_simplified_planning_request_exposes_one_next_action(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = _configure_isolated_api(tmp_path, monkeypatch)
    source = (
        "材料号,厚度,宽度,外径,毛重,净重,到站\n"
        "P001,1.2,1000,1200,10.0,9.8,Sydney\n"
    ).encode("utf-8")
    with TestClient(backend_api.app) as client:
        created = client.post(
            "/v2/planning-requests",
            files={"file": ("products.csv", source, "text/csv")},
        )
        payload = created.json()
        confirmed = client.post(
            f"/v2/planning-requests/{payload['request_id']}/confirm",
            json={"accepted_repair_ids": []},
        )
        submitted = client.post(
            f"/v2/planning-requests/{payload['request_id']}/submit",
            headers={"idempotency-key": "planning-browser-operation-001"},
            json={
                "time_budget_seconds": 3,
                "destination_grouping_required": True,
            },
        )
        replayed = client.post(
            f"/v2/planning-requests/{payload['request_id']}/submit",
            headers={"idempotency-key": "planning-browser-operation-001"},
            json={
                "time_budget_seconds": 3,
                "destination_grouping_required": True,
            },
        )

    assert created.status_code == 201
    assert payload["status"] == "READY_TO_PLAN"
    assert payload["next_action"] == "CONFIRM_AND_PLAN"
    assert payload["planning_ready"] is True
    assert confirmed.status_code == 200
    assert confirmed.json()["status"] == "CONFIRMED"
    assert confirmed.json()["next_action"] == "SUBMIT"
    assert "products" not in confirmed.json()
    assert submitted.status_code == 202
    assert replayed.status_code == 202
    assert submitted.json()["job_id"] == replayed.json()["job_id"]
    assert replayed.json()["idempotent_replay"] is True
    queued = SQLiteStateRepository(run_root / "state.db").get_job(
        submitted.json()["job_id"]
    )["request"]
    assert queued["strategy_plan"]["operational_requirements"] == {
        "destination_grouping_required": True,
        "authorised_by": "confirmed_web_request",
    }


def test_production_planning_actions_require_gateway_identity(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_isolated_api(tmp_path, monkeypatch)
    monkeypatch.setattr(
        backend_api,
        "SETTINGS",
        replace(backend_api.SETTINGS, environment="production"),
    )
    source = (
        "材料号,厚度,宽度,外径,毛重,净重,到站\n"
        "P001,1.2,1000,1200,10.0,9.8,Sydney\n"
    ).encode("utf-8")
    trusted_headers = {
        "x-user-id": "operator-a",
        "x-user-roles": "operator",
    }
    with TestClient(backend_api.app) as client:
        denied_session = client.get("/v1/session")
        trusted_session = client.get("/v1/session", headers=trusted_headers)
        denied_create = client.post(
            "/v2/planning-requests",
            files={"file": ("products.csv", source, "text/csv")},
        )
        created = client.post(
            "/v2/planning-requests",
            headers=trusted_headers,
            files={"file": ("products.csv", source, "text/csv")},
        ).json()
        denied = client.post(
            f"/v2/planning-requests/{created['request_id']}/confirm",
            json={"accepted_repair_ids": []},
        )
        allowed = client.post(
            f"/v2/planning-requests/{created['request_id']}/confirm",
            headers=trusted_headers,
            json={"accepted_repair_ids": []},
        )

    assert denied_session.status_code == 403
    assert denied_session.json()["error"]["code"] == "TRUSTED_IDENTITY_REQUIRED"
    assert trusted_session.status_code == 200
    assert trusted_session.json() == {
        "user_id": "operator-a",
        "roles": ["operator"],
        "identity_source": "trusted_gateway",
    }
    assert denied_create.status_code == 403
    assert denied_create.json()["error"]["code"] == "TRUSTED_IDENTITY_REQUIRED"
    assert denied.status_code == 403
    assert denied.json()["error"]["code"] == "TRUSTED_IDENTITY_REQUIRED"
    assert allowed.status_code == 200
    assert allowed.json()["status"] == "CONFIRMED"


def test_production_rejects_template_simulation_events(monkeypatch) -> None:
    monkeypatch.setattr(
        backend_api,
        "SETTINGS",
        replace(
            backend_api.SETTINGS,
            environment="production",
            allow_simulation_events=False,
        ),
    )

    blocked = backend_api._simulation_event_guard(
        {
            "event_type": "Add products",
            "add_product_mode": "template_simulation",
        }
    )
    real_upload = backend_api._simulation_event_guard(
        {
            "event_type": "Add products",
            "add_product_mode": "confirmed_upload",
        }
    )

    assert blocked is not None
    assert blocked.status_code == 403
    assert json.loads(blocked.body)["error"]["code"] == "SIMULATION_EVENTS_DISABLED"
    assert real_upload is None


def test_production_request_control_requires_trusted_identity(monkeypatch) -> None:
    monkeypatch.setattr(
        backend_api,
        "SETTINGS",
        replace(backend_api.SETTINGS, environment="production"),
    )
    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "https",
            "path": "/v1/plans",
            "raw_path": b"/v1/plans",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("internal", 443),
            "root_path": "",
        }
    )

    async def should_not_call(_request):
        raise AssertionError("business handler was reached without identity")

    coroutine = backend_api.RequestControlMiddleware(backend_api.app).dispatch(
        request, should_not_call
    )
    try:
        coroutine.send(None)
    except StopIteration as completed:
        response = completed.value
    else:
        raise AssertionError("identity rejection unexpectedly awaited downstream work")

    assert response.status_code == 403
    assert json.loads(response.body)["error"]["code"] == "TRUSTED_IDENTITY_REQUIRED"


def test_simplified_request_runs_llm_assistance_automatically(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_isolated_api(tmp_path, monkeypatch)
    unknown_columns = pd.DataFrame([
        {"编号列": "L001", "规格一": 1.2, "规格二": 1000, "规格三": 1200, "重量一": 10, "重量二": 9.8, "地点列": "Sydney"}
    ])

    def fake_llm_recovery(uploaded_file, run_dir, **kwargs):
        products = [{
            "material_no": "L001", "thickness": 1.2, "width": 1000,
            "outer_diameter": 1200, "gross_weight_tonne": 10,
            "net_weight_tonne": 9.8, "destination": "Sydney",
        }]
        output = run_dir / "products.json"
        output.write_text(json.dumps(products, ensure_ascii=False), encoding="utf-8")
        return output, {"llm_called": True, "field_mapping": {"material_no": "编号列"}}

    monkeypatch.setattr(
        "Agents.business_intake_service.save_uploaded_products", fake_llm_recovery
    )
    with TestClient(backend_api.app) as client:
        response = client.post(
            "/v2/planning-requests",
            files={"file": ("非标准字段.xlsx", _excel_bytes({"业务清单": unknown_columns}), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "READY_TO_PLAN"
    assert payload["next_action"] == "CONFIRM_AND_PLAN"
    assert payload["assistance_status"] == "completed"


def test_simplified_request_blocks_without_repairable_core_data(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_isolated_api(tmp_path, monkeypatch)
    source = (
        "材料号,厚度,宽度,外径,毛重,净重,到站\n"
        "P001,1.2,1000,1200,10.0,9.8,\n"
    ).encode("utf-8")
    with TestClient(backend_api.app) as client:
        response = client.post(
            "/v2/planning-requests",
            files={"file": ("missing_destination.csv", source, "text/csv")},
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "BLOCKED"
    assert payload["next_action"] == "REPLACE_FILE"
    assert payload["blocking_issues"][0]["field"] == "destination"


def test_simplified_request_blocks_kilogram_scale_weight_in_tonne_column(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_isolated_api(tmp_path, monkeypatch)
    source = (
        "材料号,厚度,宽度,外径,毛重,净重,到站\n"
        "P001,1.2,1000,1200,12400,12000,Sydney\n"
    ).encode("utf-8")

    with TestClient(backend_api.app) as client:
        response = client.post(
            "/v2/planning-requests",
            files={"file": ("wrong_weight_unit.csv", source, "text/csv")},
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "BLOCKED"
    assert payload["next_action"] == "REPLACE_FILE"
    assert payload["planning_ready"] is False
    assert "核对重量单位" in payload["message"]
    assert payload["blocking_issues"] == [
        {
            "field": "gross_weight_tonne",
            "count": 1,
            "source_rows": [2],
            "issue_type": "gross_weight_exceeds_deployment_limit",
            "maximum_tonne": 60.0,
        }
    ]
