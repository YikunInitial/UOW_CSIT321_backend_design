"""Versioned HTTP API for planning, approval, and dynamic events."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from Agents.analytics_assistant_service import answer_analytics_question
from Agents.api_contracts import (
    AnalyticsAssistantRequest,
    ApprovalRequest,
    ContinuousStreamStartRequest,
    DynamicEventRequest,
    EnrichmentConfirmationRequest,
    PlanDispositionRequest,
    PlanningRequestReviewRequest,
    PlanningRequestSubmitRequest,
    PlanningRunRequest,
    ScheduledDynamicEventRequest,
    openapi_document,
)
from Agents.app_settings import APISettings
from Agents.business_intake_service import (
    assist_intake,
    finalise_intake,
    inspect_intake,
    safe_filename,
)
from Agents.dispatch_analytics_service import build_dispatch_dashboard
from Agents.enrichment_agent import EnrichmentConfirmationService
from Agents.integrity_service import ArtifactIntegrityError, verify_plan_artifacts
from Agents.job_service import validate_products
from Agents.observability import configure_logging
from Agents.planning_request_service import planning_request_view
from Agents.planning_service import PlanningService
from Agents.scheduling_service import ScheduledEventService
from Agents.state_repository import (
    IdempotencyConflictError,
    QueueCapacityError,
    SQLiteStateRepository,
    StateConflictError,
)
from Agents.storage_health import check_storage_health
from Agents.strategy_policy import deterministic_strategy_plan
from worker import JobWorker

APP_ROOT = Path(__file__).resolve().parent
SETTINGS = APISettings.from_environment(APP_ROOT)
API_RUN_ROOT = SETTINGS.run_root
API_DATABASE = SETTINGS.database_path
API_MAX_PRODUCTS = SETTINGS.max_products
API_KEY = SETTINGS.api_key
EMBEDDED_WORKER_ENABLED = SETTINGS.embedded_worker
_EMBEDDED_WORKER: JobWorker | None = None
_EMBEDDED_WORKER_THREAD: threading.Thread | None = None
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_WINDOWS: dict[str, tuple[int, int]] = {}
_REPOSITORY_LOCK = threading.Lock()
_ANALYTICS_AUDIT_LOCK = threading.Lock()
_REPOSITORY_INSTANCE: SQLiteStateRepository | None = None
LOGGER = logging.getLogger("steel.api")
_STORAGE_HEALTH_LOCK = threading.Lock()
_STORAGE_HEALTH_CACHE: tuple[float, Path, int, Dict[str, Any]] | None = None


def repository() -> SQLiteStateRepository:
    global _REPOSITORY_INSTANCE
    with _REPOSITORY_LOCK:
        if (
            _REPOSITORY_INSTANCE is None
            or _REPOSITORY_INSTANCE.database_path != API_DATABASE.resolve()
        ):
            _REPOSITORY_INSTANCE = SQLiteStateRepository(API_DATABASE)
        return _REPOSITORY_INSTANCE


def planning_service() -> PlanningService:
    repo = repository()
    return PlanningService(repo)


def _verified_dispatch_dashboard(plans: list[Dict[str, Any]]) -> Dict[str, Any]:
    eligible: list[Dict[str, Any]] = []
    integrity_failures: list[Dict[str, str]] = []
    for plan in plans:
        try:
            verify_plan_artifacts(plan)
            eligible.append(plan)
        except (ArtifactIntegrityError, OSError, ValueError) as error:
            integrity_failures.append(
                {"plan_id": str(plan.get("plan_id") or ""), "reason": str(error)}
            )
    result = build_dispatch_dashboard(eligible)
    result["skipped_plans"] = integrity_failures + result["skipped_plans"]
    result["summary"]["skipped_plan_count"] = len(result["skipped_plans"])
    return result


def _verified_plan_evidence(plans: list[Dict[str, Any]], limit: int = 10) -> list[Dict[str, Any]]:
    """Load compact, verified evidence for explanation and comparison only."""
    result: list[Dict[str, Any]] = []
    for plan in plans:
        if len(result) >= limit:
            break
        try:
            verify_plan_artifacts(plan)
            solution_path = Path(str((plan.get("payload") or {}).get("solution_path") or ""))
            solution = json.loads(solution_path.read_text(encoding="utf-8"))
            wagons = solution.get("wagons") or []
            weight_utilisation = []
            for wagon in wagons:
                weight = float(wagon.get("current_weight_tonne") or 0)
                capacity = float(wagon.get("max_weight_tonne") or 0)
                if not weight and wagon.get("current_weight") is not None:
                    weight = float(wagon.get("current_weight") or 0) / 1000.0
                if not capacity and wagon.get("max_weight") is not None:
                    capacity = float(wagon.get("max_weight") or 0) / 1000.0
                weight_utilisation.append(weight / capacity if capacity else 0.0)
            score = dict(solution.get("final_solution_score") or {})
            score["average_weight_utilisation"] = (
                sum(weight_utilisation) / len(weight_utilisation)
                if weight_utilisation else 0.0
            )
            result.append(
                {
                    "plan_id": plan.get("plan_id"),
                    "version": plan.get("version"),
                    "status": plan.get("status"),
                    "updated_at": plan.get("updated_at"),
                    "summary": solution.get("wagon_summary") or {},
                    "score": score,
                    "validation": solution.get("validation_summary") or {},
                    "strategy": solution.get("strategy_execution") or {},
                    "council": (solution.get("multi_agent_council") or {}).get("coordinator_decision") or {},
                }
            )
        except (ArtifactIntegrityError, OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return result


def _error(message: str, status_code: int, code: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message}},
        status_code=status_code,
    )


async def _payload(request: Request) -> Dict[str, Any]:
    try:
        value = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("请求内容必须是有效的 JSON。") from error
    if not isinstance(value, dict):
        raise ValueError("请求内容必须是 JSON 对象。")
    return value


async def _validated_payload(
    request: Request,
    contract: type[
        PlanningRunRequest
        | ApprovalRequest
        | DynamicEventRequest
        | EnrichmentConfirmationRequest
        | PlanningRequestReviewRequest
        | PlanningRequestSubmitRequest
    ],
) -> Dict[str, Any]:
    model = contract.model_validate(await _payload(request))
    return model.model_dump(exclude_none=True)


def _validate_products(products: Any) -> list[Dict[str, Any]]:
    return validate_products(products, max_products=API_MAX_PRODUCTS)


def _idempotency_key(request: Request) -> str | None:
    value = request.headers.get("idempotency-key", "").strip()
    if not value:
        if SETTINGS.require_idempotency_key:
            raise ValueError("该请求缺少防重复提交标识。")
        return None
    if not re.fullmatch(r"[A-Za-z0-9._:-]{8,128}", value):
        raise ValueError("防重复提交标识必须为 8–128 位安全字符。")
    return value


def _request_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _authorised_actor(
    request: Request,
    *,
    requested_actor: str = "",
    allowed_roles: set[str],
    default_actor: str = "api-operator",
) -> tuple[str, JSONResponse | None]:
    """Resolve identity supplied by a trusted production reverse proxy."""
    trusted_user = request.headers.get("x-user-id", "").strip()
    roles = {
        role.strip().lower()
        for role in request.headers.get("x-user-roles", "").split(",")
        if role.strip()
    }
    if SETTINGS.production and (
        not trusted_user or not roles.intersection(allowed_roles)
    ):
        return "", _error(
            "该操作需要可信的业务用户身份。", 403, "TRUSTED_ROLE_REQUIRED"
        )
    requested = requested_actor.strip()
    if trusted_user and requested and trusted_user != requested:
        return "", _error(
            "操作人信息与当前可信用户身份不一致。",
            403,
            "TRUSTED_IDENTITY_MISMATCH",
        )
    return trusted_user or requested or default_actor, None


def _simulation_event_guard(payload: Dict[str, Any]) -> JSONResponse | None:
    """Keep generated demo products outside production decision records."""
    if (
        payload.get("event_type") == "Add products"
        and payload.get("add_product_mode") == "template_simulation"
        and not SETTINGS.allow_simulation_events
    ):
        return _error(
            "生产环境禁止使用模拟新增产品；请上传并确认真实业务数据。",
            403,
            "SIMULATION_EVENTS_DISABLED",
        )
    return None


def _storage_health(*, force: bool = False) -> Dict[str, Any]:
    global _STORAGE_HEALTH_CACHE
    now = time.monotonic()
    with _STORAGE_HEALTH_LOCK:
        if (
            not force
            and _STORAGE_HEALTH_CACHE is not None
            and now - _STORAGE_HEALTH_CACHE[0] < 5.0
            and _STORAGE_HEALTH_CACHE[1] == API_DATABASE.resolve()
            and _STORAGE_HEALTH_CACHE[2] == SETTINGS.minimum_free_disk_bytes
        ):
            return dict(_STORAGE_HEALTH_CACHE[3])
        repository()
        report = check_storage_health(
            API_RUN_ROOT,
            API_DATABASE,
            minimum_free_bytes=SETTINGS.minimum_free_disk_bytes,
        )
        _STORAGE_HEALTH_CACHE = (
            now,
            API_DATABASE.resolve(),
            SETTINGS.minimum_free_disk_bytes,
            report,
        )
        return dict(report)


def _storage_unavailable_response(report: Dict[str, Any]) -> JSONResponse:
    low_disk = report["free_bytes"] < report["minimum_free_bytes"]
    return _error(
            "持久化存储尚未就绪，暂时不能接收新任务。",
        507 if low_disk else 503,
        "INSUFFICIENT_STORAGE" if low_disk else "STORAGE_UNAVAILABLE",
    )


def _start_embedded_worker() -> None:
    global _EMBEDDED_WORKER, _EMBEDDED_WORKER_THREAD
    configure_logging()
    if not EMBEDDED_WORKER_ENABLED or (
        _EMBEDDED_WORKER_THREAD is not None and _EMBEDDED_WORKER_THREAD.is_alive()
    ):
        return
    _EMBEDDED_WORKER = JobWorker(
        repository(),
        worker_id=f"embedded-api-{os.getpid()}",
        poll_seconds=0.25,
        max_products=SETTINGS.max_products,
        minimum_free_disk_bytes=SETTINGS.minimum_free_disk_bytes,
    )
    _EMBEDDED_WORKER_THREAD = threading.Thread(
        target=_EMBEDDED_WORKER.run_forever,
        name="embedded-steel-worker",
        daemon=True,
    )
    _EMBEDDED_WORKER_THREAD.start()


def _stop_embedded_worker() -> None:
    if _EMBEDDED_WORKER is not None:
        _EMBEDDED_WORKER.stop()


@asynccontextmanager
async def lifespan(_: Starlette):
    _start_embedded_worker()
    try:
        yield
    finally:
        _stop_embedded_worker()


async def health(_: Request) -> JSONResponse:
    repo = repository()
    metrics = repo.operational_metrics()
    return JSONResponse(
        {
            "status": "ok",
            "service": "steel-loading-api",
            "version": "v1",
            "schema_version": metrics["schema_version"],
            "jobs": metrics["jobs"],
            "workers": metrics["workers"],
        }
    )


async def liveness(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "steel-loading-api"})


async def readiness(_: Request) -> JSONResponse:
    try:
        metrics = repository().operational_metrics()
    except Exception:
        LOGGER.exception("readiness_check_failed")
        return _error("状态存储当前不可用。", 503, "STATE_STORE_UNAVAILABLE")
    storage = _storage_health()
    queued = metrics["jobs"]["queued"]
    ready = storage["healthy"] and (queued == 0 or metrics["workers"]["active"] > 0)
    return JSONResponse(
        {
            "status": "ready" if ready else "degraded",
            **metrics,
            "storage": storage,
        },
        status_code=200 if ready else 503,
    )


async def metrics(_: Request) -> JSONResponse:
    return JSONResponse(repository().operational_metrics())


def _intake_directory(intake_id: str) -> Path:
    if not re.fullmatch(r"intake_[0-9a-f]{32}", intake_id):
        raise KeyError("找不到该数据接入记录，请重新上传文件。")
    path = (API_RUN_ROOT / "intakes" / intake_id).resolve()
    root = (API_RUN_ROOT / "intakes").resolve()
    if root not in path.parents:
        raise KeyError("找不到该数据接入记录，请重新上传文件。")
    return path


async def create_data_intake(request: Request) -> JSONResponse:
    """Persist and inspect browser uploads without trusting client-side parsing."""
    try:
        payload = await _stage_and_inspect_intake(request)
        return JSONResponse(payload, status_code=201)
    except (ValueError, ValidationError) as error:
        return _error(_localise_intake_error(error), 422, "INVALID_DATA_INTAKE")
    except Exception:
        LOGGER.exception("data_intake_failed")
        return _error("无法检查上传文件，请确认文件未损坏且格式正确。", 422, "DATA_INTAKE_FAILED")


async def _stage_and_inspect_intake(request: Request) -> Dict[str, Any]:
    storage = _storage_health()
    if not storage["healthy"]:
        raise RuntimeError("持久化存储尚未就绪，暂时不能接收新任务。")
    form = await request.form()
    source = form.get("file")
    if source is None or not hasattr(source, "read"):
        raise ValueError("请选择需要上传的产品数据文件。")
    source_name = safe_filename(str(getattr(source, "filename", "upload.xlsx")))
    source_bytes = await source.read()
    if not source_bytes:
        raise ValueError("上传文件为空，请检查后重新选择。")
    if len(source_bytes) > SETTINGS.max_upload_bytes:
        raise ValueError("上传文件超过当前部署大小限制，请拆分批次。")
    intake_id = f"intake_{uuid4().hex}"
    intake_dir = _intake_directory(intake_id)
    intake_dir.mkdir(parents=True, exist_ok=False)
    source_path = intake_dir / f"source{Path(source_name).suffix.lower()}"
    source_path.write_bytes(source_bytes)
    master = form.get("master_file")
    master_path: Path | None = None
    master_name: str | None = None
    if master is not None and hasattr(master, "read"):
        master_name = safe_filename(str(getattr(master, "filename", "master.xlsx")))
        master_bytes = await master.read()
        if len(master_bytes) > SETTINGS.max_upload_bytes:
            raise ValueError("权威主数据文件超过当前部署大小限制。")
        if master_bytes:
            master_path = intake_dir / f"master{Path(master_name).suffix.lower()}"
            master_path.write_bytes(master_bytes)
    metadata = {
        "intake_id": intake_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_original_name": source_name,
        "source_stored_name": source_path.name,
        "master_original_name": master_name,
        "master_stored_name": master_path.name if master_path else None,
    }
    (intake_dir / "intake.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = await asyncio.to_thread(
        inspect_intake,
        source_path,
        master_path=master_path,
        max_products=API_MAX_PRODUCTS,
    )
    return {**metadata, **report}


def _localise_intake_error(error: Exception) -> str:
    message = str(error)
    lowered = message.casefold()
    if "excel file format cannot be determined" in lowered or "file is not a zip" in lowered:
        return "无法读取该 Excel 文件。文件可能已损坏、扩展名不正确或不是有效工作簿。"
    if "llm selected unknown columns" in lowered:
        return "智能字段识别未能建立有效字段映射，请使用标准模板或联系数据管理员。"
    if "llm did not return a json object" in lowered:
        return "智能字段识别返回格式无效，请稍后重试或使用标准模板。"
    if "could not map required fields" in lowered or "could not infer all required fields" in lowered:
        return "智能字段识别无法确认全部核心字段，请使用标准模板或补充清晰的列名。"
    if "only excel and json product files are supported" in lowered:
        return "智能识别仅支持 Excel 或 JSON 产品文件。"
    if "input file is empty" in lowered or "products json is empty" in lowered:
        return "上传文件中没有可识别的产品记录。"
    if "products json must be a list" in lowered:
        return "JSON 顶层必须是产品记录列表。"
    if "product records do not contain a supported" in lowered:
        return "产品记录缺少可支持的核心业务字段。"
    duplicate_match = re.search(r"row\s+(\d+)\s+duplicates product identifier\s+(.+?)\.?$", message, re.I)
    if duplicate_match:
        return f"第 {duplicate_match.group(1)} 条产品与材料号 {duplicate_match.group(2)} 重复。"
    missing_match = re.search(r"row\s+(\d+)\s+is missing\s+(.+?)\.?$", message, re.I)
    if missing_match:
        field = missing_match.group(2).replace("a product identifier", "material_no")
        return f"第 {missing_match.group(1)} 条产品缺少{_business_field_name(field)}。"
    invalid_match = re.search(r"row\s+(\d+)\s+has invalid\s+(.+?)\.?$", message, re.I)
    if invalid_match:
        return f"第 {invalid_match.group(1)} 条产品的{_business_field_name(invalid_match.group(2))}无效。"
    replacements = {
        "material_no": "材料号",
        "destination": "目的地",
        "thickness": "厚度",
        "width": "宽度",
        "outer_diameter": "外径",
        "gross_weight_tonne": "毛重",
        "net_weight_tonne": "净重",
        "receiver": "收货单位",
        "contract_no": "合同号",
        "storage_location": "库位",
        "remarks": "备注",
    }
    for source, target in replacements.items():
        message = message.replace(source, target)
    return message


def _business_field_name(field: str) -> str:
    return {
        "material_no": "材料号", "destination": "目的地", "thickness": "厚度",
        "width": "宽度", "outer_diameter": "外径", "gross_weight_tonne": "毛重",
        "net_weight_tonne": "净重", "receiver": "收货单位", "contract_no": "合同号",
        "storage_location": "库位", "remarks": "备注",
    }.get(str(field).strip().casefold(), str(field))


async def assist_data_intake(request: Request) -> JSONResponse:
    """Run bounded LLM field mapping before operator review and planning."""
    try:
        intake_dir = _intake_directory(request.path_params["intake_id"])
        metadata_path = intake_dir / "intake.json"
        if not metadata_path.is_file():
            raise KeyError("找不到该数据接入记录，请重新上传文件。")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        report = assist_intake(intake_dir, max_products=API_MAX_PRODUCTS)
        return JSONResponse({**metadata, **report})
    except KeyError:
        return _error("找不到该数据接入记录，请重新上传文件。", 404, "INTAKE_NOT_FOUND")
    except (ValueError, ValidationError) as error:
        return _error(_localise_intake_error(error), 422, "INTAKE_ASSISTANCE_FAILED")
    except Exception:
        LOGGER.exception("data_intake_assistance_failed")
        return _error(
            "智能字段识别服务暂时不可用，请稍后重试或使用标准数据模板。",
            503,
            "INTAKE_ASSISTANCE_UNAVAILABLE",
        )


async def confirm_data_intake(request: Request) -> JSONResponse:
    try:
        intake_dir = _intake_directory(request.path_params["intake_id"])
        if not (intake_dir / "intake.json").is_file():
            raise KeyError("找不到该数据接入记录，请重新上传文件。")
        payload = await _payload(request)
        repair_ids = payload.get("accepted_repair_ids", [])
        if not isinstance(repair_ids, list):
            raise ValueError("修复确认内容格式无效，请重新检查。")
        if payload.get("review_completed") is not True:
            raise ValueError("必须由操作员明确完成数据核对后才能规划。")
        trusted_user = request.headers.get("x-user-id", "").strip()
        result = finalise_intake(
            intake_dir,
            accepted_repair_ids=[str(item) for item in repair_ids],
            reviewed_by=trusted_user or str(payload.get("reviewed_by") or "web-operator"),
            max_products=API_MAX_PRODUCTS,
        )
        return JSONResponse({"intake_id": request.path_params["intake_id"], **result})
    except KeyError as error:
        return _error(str(error), 404, "INTAKE_NOT_FOUND")
    except (ValueError, ValidationError) as error:
        return _error(_localise_intake_error(error), 422, "INTAKE_CONFIRMATION_FAILED")


async def preview_data_intake(request: Request) -> JSONResponse:
    """Re-run the same deterministic inspection after repair selections change."""
    try:
        intake_dir = _intake_directory(request.path_params["intake_id"])
        metadata_path = intake_dir / "intake.json"
        if not metadata_path.is_file():
            raise KeyError("找不到该数据接入记录，请重新上传文件。")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload = await _payload(request)
        repair_ids = payload.get("accepted_repair_ids", [])
        if not isinstance(repair_ids, list):
            raise ValueError("修复确认内容格式无效，请重新检查。")
        source_path = intake_dir / metadata["source_stored_name"]
        master_path = (
            intake_dir / metadata["master_stored_name"]
            if metadata.get("master_stored_name")
            else None
        )
        report = inspect_intake(
            source_path,
            master_path=master_path,
            accepted_repair_ids=[str(item) for item in repair_ids],
            max_products=API_MAX_PRODUCTS,
        )
        return JSONResponse({**metadata, **report})
    except KeyError as error:
        return _error(str(error), 404, "INTAKE_NOT_FOUND")
    except ValueError as error:
        return _error(_localise_intake_error(error), 422, "INVALID_DATA_INTAKE")


async def create_planning_request(request: Request) -> JSONResponse:
    """Create the simplified web request and automatically run safe fallbacks."""
    try:
        payload = await _stage_and_inspect_intake(request)
        if payload.get("requires_assistance"):
            intake_dir = _intake_directory(str(payload["intake_id"]))
            try:
                assisted = await asyncio.to_thread(
                    assist_intake, intake_dir, max_products=API_MAX_PRODUCTS
                )
                payload = {**payload, **assisted}
            except Exception as error:
                LOGGER.warning(
                    "automatic_intake_assistance_failed",
                    extra={"event": "automatic_intake_assistance_failed", "intake_id": payload["intake_id"]},
                )
                view = planning_request_view(
                    payload,
                    analysis_error=_localise_intake_error(error),
                )
                repository().create_planning_request(
                    str(payload["intake_id"]),
                    intake_dir=_intake_directory(str(payload["intake_id"])),
                    payload=view,
                )
                return JSONResponse(view, status_code=201)
        view = planning_request_view(payload)
        repository().create_planning_request(
            str(payload["intake_id"]),
            intake_dir=_intake_directory(str(payload["intake_id"])),
            payload=view,
        )
        return JSONResponse(view, status_code=201)
    except (ValueError, ValidationError) as error:
        return _error(_localise_intake_error(error), 422, "INVALID_PLANNING_REQUEST")
    except Exception:
        LOGGER.exception("planning_request_creation_failed")
        return _error("无法分析上传文件，请确认文件格式后重试。", 422, "PLANNING_REQUEST_FAILED")


def _planning_request_paths(request_id: str) -> tuple[Path, Dict[str, Any], Path, Path | None]:
    intake_dir = _intake_directory(request_id)
    metadata_path = intake_dir / "intake.json"
    if not metadata_path.is_file():
        raise KeyError("找不到该规划请求，请重新上传文件。")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_path = intake_dir / metadata["source_stored_name"]
    master_path = (
        intake_dir / metadata["master_stored_name"]
        if metadata.get("master_stored_name")
        else None
    )
    return intake_dir, metadata, source_path, master_path


def _planning_request_response(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the durable workflow state without exposing the frozen products."""
    payload = dict(record["payload"])
    lifecycle = str(record["status"])
    if lifecycle == "CONFIRMED":
        payload.update(
            status="CONFIRMED",
            next_action="SUBMIT",
            progress=100,
            stage="DATA_CONFIRMED",
        )
    elif lifecycle in {"QUEUED", "RUNNING"}:
        payload.update(
            status="PLANNING",
            next_action="VIEW_PROGRESS",
            progress=100,
            stage=lifecycle,
        )
    elif lifecycle == "COMPLETED":
        payload.update(
            status="COMPLETED",
            next_action="VIEW_RESULT",
            progress=100,
            stage="COMPLETED",
        )
    elif lifecycle == "FAILED":
        payload.update(
            status="FAILED",
            next_action="RETRY",
            progress=100,
            stage="FAILED",
        )
    payload.update(
        request_id=record["request_id"],
        intake_id=record["request_id"],
        request_version=record["version"],
        job_id=record.get("job_id"),
        plan_id=record.get("plan_id"),
        confirmed_at=record.get("confirmed_at"),
    )
    payload.pop("products", None)
    payload.pop("data_intake_audit", None)
    return payload


async def get_planning_request(request: Request) -> JSONResponse:
    try:
        record = repository().get_planning_request(request.path_params["request_id"])
        return JSONResponse(_planning_request_response(record))
    except KeyError as error:
        return _error(str(error), 404, "PLANNING_REQUEST_NOT_FOUND")
    except (ValueError, ValidationError) as error:
        return _error(_localise_intake_error(error), 422, "INVALID_PLANNING_REQUEST")


async def review_planning_request(request: Request) -> JSONResponse:
    try:
        _, metadata, source_path, master_path = _planning_request_paths(
            request.path_params["request_id"]
        )
        body = await _validated_payload(request, PlanningRequestReviewRequest)
        repair_ids = body["accepted_repair_ids"]
        report = await asyncio.to_thread(
            inspect_intake,
            source_path,
            master_path=master_path,
            accepted_repair_ids=[str(item) for item in repair_ids],
            max_products=API_MAX_PRODUCTS,
        )
        view = planning_request_view({**metadata, **report})
        record = repository().review_planning_request(
            request.path_params["request_id"],
            payload=view,
            accepted_repair_ids=[str(item) for item in repair_ids],
        )
        return JSONResponse(_planning_request_response(record))
    except KeyError as error:
        return _error(str(error), 404, "PLANNING_REQUEST_NOT_FOUND")
    except (ValueError, ValidationError) as error:
        return _error(_localise_intake_error(error), 422, "INVALID_PLANNING_REQUEST")


async def confirm_planning_request(request: Request) -> JSONResponse:
    """Treat the explicit confirm action as the operator's review decision."""
    try:
        request_id = request.path_params["request_id"]
        intake_dir, _, _, _ = _planning_request_paths(request_id)
        body = await _validated_payload(request, PlanningRequestReviewRequest)
        repair_ids = body["accepted_repair_ids"]
        reviewed_by, auth_error = _authorised_actor(
            request, allowed_roles={"operator", "approver"}, default_actor="web-operator"
        )
        if auth_error:
            return auth_error
        existing = repository().get_planning_request(request_id)
        if existing["status"] in {
            "CONFIRMED",
            "QUEUED",
            "RUNNING",
            "COMPLETED",
            "FAILED",
        }:
            if existing["accepted_repair_ids"] != [str(item) for item in repair_ids]:
                raise StateConflictError("已确认的数据快照不能再修改。")
            return JSONResponse(_planning_request_response(existing))
        result = await asyncio.to_thread(
            finalise_intake,
            intake_dir,
            accepted_repair_ids=[str(item) for item in repair_ids],
            reviewed_by=reviewed_by,
            max_products=API_MAX_PRODUCTS,
        )
        snapshot_path = intake_dir / "products.json"
        snapshot_sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        confirmed_payload = {
            **existing["payload"],
            "status": "CONFIRMED",
            "message": "数据快照已确认，可以提交正式规划。",
            "record_count": len(result["products"]),
            "data_intake_audit": result["data_intake_audit"],
        }
        record = repository().confirm_planning_request(
            request_id,
            accepted_repair_ids=[str(item) for item in repair_ids],
            snapshot_path=snapshot_path,
            snapshot_sha256=snapshot_sha256,
            confirmed_by=reviewed_by,
            payload=confirmed_payload,
        )
        return JSONResponse(_planning_request_response(record))
    except KeyError as error:
        return _error(str(error), 404, "PLANNING_REQUEST_NOT_FOUND")
    except (ValueError, ValidationError) as error:
        return _error(_localise_intake_error(error), 422, "PLANNING_REQUEST_CONFIRMATION_FAILED")
    except StateConflictError as error:
        return _error(str(error), 409, "PLANNING_REQUEST_STATE_CONFLICT")


async def submit_planning_request(request: Request) -> JSONResponse:
    """Queue planning from the immutable server-side confirmed snapshot."""
    try:
        storage = _storage_health()
        if not storage["healthy"]:
            return _storage_unavailable_response(storage)
        _, auth_error = _authorised_actor(
            request, allowed_roles={"operator", "approver"}
        )
        if auth_error:
            return auth_error
        request_id = request.path_params["request_id"]
        record = repository().get_planning_request(request_id)
        if record["status"] not in {
            "CONFIRMED",
            "QUEUED",
            "RUNNING",
            "COMPLETED",
            "FAILED",
        }:
            raise StateConflictError(
                "规划请求尚未完成数据确认，不能提交正式计算。"
            )
        options = await _validated_payload(request, PlanningRequestSubmitRequest)
        snapshot_path = Path(str(record.get("snapshot_path") or ""))
        if not snapshot_path.is_file():
            raise StateConflictError("已确认的数据快照不存在，请重新上传文件。")
        actual_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        if actual_hash != record.get("snapshot_sha256"):
            raise StateConflictError("已确认的数据快照完整性检查失败。")
        products = json.loads(snapshot_path.read_text(encoding="utf-8"))
        _validate_products(products)
        audit = dict(record["payload"].get("data_intake_audit") or {})
        strategy_plan = None
        if options["destination_grouping_required"]:
            strategy_plan = deterministic_strategy_plan(
                products,
                time_budget_seconds=options["time_budget_seconds"],
                reason=(
                    "Operator required destination-separated wagon planning for "
                    "operational unloading clarity and province attribution."
                ),
            )
            strategy_plan["operational_requirements"] = {
                "destination_grouping_required": True,
                "authorised_by": "confirmed_web_request",
            }
        payload = {
            "products": products,
            "execution_mode": "multi_agent_council",
            "scenario_id": "web_request",
            "system_variant": "full_hybrid",
            "time_budget_seconds": options["time_budget_seconds"],
            "random_seed": 42,
            "generate_visuals": options["generate_visuals"],
            "strategy_plan": strategy_plan,
            "data_intake_audit": audit,
            "semantic_planning_enabled": options["semantic_planning_enabled"],
            "rag_top_k": options["rag_top_k"],
            "optional_enrichment_enabled": options["optional_enrichment_enabled"],
            "optional_enrichment_fields": options["optional_enrichment_fields"],
            "optional_enrichment_max_products": options[
                "optional_enrichment_max_products"
            ],
            "source_input_path": audit.get("source_input_path"),
            "source_input_name": audit.get("input_file"),
            "source_input_sha256": audit.get("source_input_sha256"),
            "repaired_source_path": audit.get("repaired_source_path"),
            "repair_audit_path": audit.get("repair_audit_path"),
            "authoritative_master_path": audit.get("authoritative_master_path"),
        }
        idempotency_key = _idempotency_key(request)
        job_id = f"job_{uuid4().hex}"
        run_dir = API_RUN_ROOT / (
            time.strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        )
        queued_request = {
            **payload,
            "job_type": "planning",
            "run_dir": str(run_dir),
            "planning_request_id": request_id,
        }
        endpoint = "/v2/planning-requests/{request_id}/submit"
        stored_job_id, created = repository().enqueue_job(
            job_id,
            queued_request,
            max_attempts=SETTINGS.job_max_attempts,
            max_active_jobs=SETTINGS.max_active_jobs,
            idempotency_key=idempotency_key,
            endpoint=endpoint,
            request_hash=_request_hash({"request_id": request_id, **options}),
            planning_request_id=request_id,
        )
        return JSONResponse(
            {
                "request_id": request_id,
                "job_id": stored_job_id,
                "status": repository().get_job(stored_job_id)["status"],
                "idempotent_replay": not created,
            },
            status_code=202,
        )
    except QueueCapacityError as error:
        response = _error(str(error), 503, "QUEUE_CAPACITY_REACHED")
        response.headers["retry-after"] = "30"
        return response
    except IdempotencyConflictError as error:
        return _error(str(error), 409, "IDEMPOTENCY_CONFLICT")
    except KeyError as error:
        return _error(str(error), 404, "PLANNING_REQUEST_NOT_FOUND")
    except StateConflictError as error:
        return _error(str(error), 409, "PLANNING_REQUEST_STATE_CONFLICT")
    except (ValueError, ValidationError, json.JSONDecodeError) as error:
        return _error(str(error), 422, "INVALID_PLANNING_REQUEST")


async def create_sample_data_intake(_: Request) -> JSONResponse:
    """Stage the repository training dataset through the same server-side path."""
    try:
        if not SETTINGS.allow_sample_data:
            return _error("当前部署已禁用系统训练数据。", 403, "SAMPLE_DATA_DISABLED")
        source = APP_ROOT / "products_clean.xlsx"
        if not source.is_file():
            raise FileNotFoundError("系统训练数据当前不可用。")
        intake_id = f"intake_{uuid4().hex}"
        intake_dir = _intake_directory(intake_id)
        intake_dir.mkdir(parents=True, exist_ok=False)
        source_path = intake_dir / "source.xlsx"
        source_path.write_bytes(source.read_bytes())
        metadata = {
            "intake_id": intake_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_original_name": "products_clean.xlsx [TRAINING]",
            "source_stored_name": source_path.name,
            "master_original_name": None,
            "master_stored_name": None,
            "training_data": True,
        }
        (intake_dir / "intake.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report = inspect_intake(source_path, max_products=API_MAX_PRODUCTS)
        report["message"] = "系统训练数据已就绪；仅用于演示，不得作为现场输入。"
        report["repair_analysis"] = {
            "schema_version": "1.0",
            "suggestions": [],
            "unresolved": [],
            "policy": {"mode": "trusted_training_fixture", "original_input_mutated": False},
        }
        payload = {**metadata, **report}
        view = planning_request_view(payload)
        repository().create_planning_request(
            intake_id, intake_dir=intake_dir, payload=view
        )
        return JSONResponse(view, status_code=201)
    except (FileNotFoundError, ValueError) as error:
        return _error(str(error), 422, "SAMPLE_DATA_UNAVAILABLE")


async def list_plans(request: Request) -> JSONResponse:
    try:
        limit = int(request.query_params.get("limit", "20"))
        status = request.query_params.get("status") or None
        plans = repository().list_recent_plans(limit=limit, status=status)
        compact = []
        for item in plans:
            summary: Dict[str, Any] = {
                "plan_id": item["plan_id"], "version": item["version"],
                "status": item["status"], "parent_plan_id": item.get("parent_plan_id"),
                "created_at": item["created_at"], "updated_at": item["updated_at"],
                "restorable": False,
            }
            try:
                summary_path = Path(item["run_dir"]) / "plan_summary.json"
                if summary_path.is_file():
                    persisted_summary = json.loads(
                        summary_path.read_text(encoding="utf-8")
                    )
                    summary.update(
                        {
                            "restorable": True,
                            "integrity_status": "verified_on_open",
                            **{
                                key: persisted_summary.get(key)
                                for key in (
                                    "product_count",
                                    "wagon_count",
                                    "destination_count",
                                    "is_valid",
                                )
                            },
                        }
                    )
                else:
                    verify_plan_artifacts(item)
                    solution = json.loads(
                        Path(item["payload"]["solution_path"]).read_text(
                            encoding="utf-8"
                        )
                    )
                    wagon_summary = solution.get("wagon_summary", {})
                    summary.update(
                        {
                            "restorable": True,
                            "product_count": wagon_summary.get("total_products", 0),
                            "wagon_count": wagon_summary.get("wagon_count", 0),
                            "destination_count": len(
                                solution.get("input_summary", {}).get(
                                    "destinations", []
                                )
                            ),
                            "is_valid": bool(
                                solution.get("validation_summary", {}).get(
                                    "is_valid"
                                )
                            ),
                        }
                    )
            except (ArtifactIntegrityError, OSError, ValueError, json.JSONDecodeError):
                summary["integrity_status"] = "failed"
            compact.append(summary)
        return JSONResponse({"items": compact, "count": len(compact)})
    except ValueError as error:
        return _error(str(error), 422, "INVALID_QUERY")


async def dispatch_dashboard(request: Request) -> JSONResponse:
    """Return a read-only operational view derived from persisted plan evidence."""
    try:
        limit = max(1, min(int(request.query_params.get("limit", "200")), 200))
        status = (request.query_params.get("status") or "active").strip()
        allowed = {"active", "pending_approval", "superseded", "rejected", "all"}
        if status not in allowed:
            raise ValueError("不支持的方案状态筛选。")
        plans = repository().list_recent_plans(
            limit=limit,
            status=None if status == "all" else status,
        )
        result = await asyncio.to_thread(_verified_dispatch_dashboard, plans)
        result.pop("_query_dispatches", None)
        result["filters"] = {"status": status, "limit": limit}
        return JSONResponse(result)
    except ValueError as error:
        return _error(str(error), 422, "INVALID_ANALYTICS_QUERY")


async def current_session(request: Request) -> JSONResponse:
    """Expose only the identity already authenticated by the trusted gateway."""
    actor, auth_error = _authorised_actor(
        request,
        allowed_roles={"operator", "approver", "viewer", "analyst", "admin"},
        default_actor="web-operator",
    )
    if auth_error:
        return auth_error
    roles = sorted(
        {
            role.strip().lower()
            for role in request.headers.get("x-user-roles", "").split(",")
            if role.strip()
        }
    )
    if not roles and not SETTINGS.production:
        roles = ["operator", "approver"]
    return JSONResponse(
        {
            "user_id": actor,
            "roles": roles,
            "identity_source": (
                "trusted_gateway"
                if request.headers.get("x-user-id", "").strip()
                else "development_default"
            ),
        }
    )


async def analytics_assistant(request: Request) -> JSONResponse:
    """Interpret a question, validate its query plan, and aggregate read-only evidence."""
    try:
        actor, auth_error = _authorised_actor(
            request,
            allowed_roles={"operator", "approver", "viewer", "analyst"},
            default_actor="web-operator",
        )
        if auth_error:
            return auth_error
        payload = await _validated_payload(request, AnalyticsAssistantRequest)
        status = payload["status"]
        plans = repository().list_recent_plans(
            limit=200,
            status=None if status == "all" else status,
        )
        dashboard = await asyncio.to_thread(_verified_dispatch_dashboard, plans)
        historical = repository().list_recent_plans(limit=20, status=None)
        seen = {str(item.get("plan_id")) for item in plans}
        evidence_input = plans + [
            item for item in historical if str(item.get("plan_id")) not in seen
        ]
        plan_evidence = await asyncio.to_thread(_verified_plan_evidence, evidence_input)
        result = await asyncio.to_thread(
            answer_analytics_question,
            payload["question"],
            dashboard,
            plan_evidence=plan_evidence,
            previous_query_plan=payload.get("previous_query_plan"),
        )
        result["question"] = payload["question"]
        result["filters"] = {"status": status, "limit": 200}
        audit_record = {
            "timestamp": time.time(),
            "action": "analytics_assistant_query",
            "actor": actor,
            "question": payload["question"],
            "status_scope": status,
            "interpretation_source": result["interpretation_source"],
            "query_plan": result["query_plan"],
            "response_kind": result["response_kind"],
            "result_row_count": len(result["rows"]),
        }
        audit_path = API_RUN_ROOT / "analytics_assistant_audit.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with _ANALYTICS_AUDIT_LOCK, audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(audit_record, ensure_ascii=False) + "\n")
        return JSONResponse(result)
    except ValueError as error:
        return _error(str(error), 422, "INVALID_ANALYTICS_ASSISTANT_QUERY")
    except Exception:
        LOGGER.exception("analytics_assistant_failed")
        return _error(
            "智能分析暂时不可用，运营 Dashboard 和正式规划不受影响。",
            503,
            "ANALYTICS_ASSISTANT_UNAVAILABLE",
        )


def _plan_files(plan: Dict[str, Any]) -> list[Dict[str, Any]]:
    run_dir = Path(plan["run_dir"]).resolve()
    allowed_names = {
        "loading_solution.json", "strategy_plan.json", "run_manifest.json",
        "audit_report.json", "audit_report.md", "products.json",
        "visualization_summary.json", "enrichment_report.json",
        "enrichment_confirmation.json", "data_repair_audit.json",
        "repaired_source.xlsx", "repaired_products.xlsx", "plan_disposition.json",
        "plan_summary.json",
    }
    paths = [path for path in run_dir.iterdir() if path.is_file() and path.name in allowed_names]
    output_dir = run_dir / "outputs"
    if output_dir.is_dir():
        paths.extend(path for path in output_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")
    return [
        {
            "name": path.name,
            "kind": "layout" if path.suffix.lower() == ".png" else "artifact",
            "size_bytes": path.stat().st_size,
            "url": f"/v1/plans/{plan['plan_id']}/artifacts/{path.name}",
        }
        for path in sorted(paths, key=lambda item: item.name)
    ]


async def get_plan_result(request: Request) -> JSONResponse:
    try:
        plan = repository().get_plan(request.path_params["plan_id"])
        integrity = verify_plan_artifacts(plan)
        run_dir = Path(plan["run_dir"])
        audit_path = run_dir / "audit_report.json"
        strategy_path = run_dir / "strategy_plan.json"
        enrichment_path = run_dir / "enrichment_report.json"
        enrichment_confirmation_path = run_dir / "enrichment_confirmation.json"
        manifest_path = run_dir / "run_manifest.json"
        return JSONResponse(
            {
                "plan": {key: plan.get(key) for key in ("plan_id", "version", "status", "parent_plan_id", "created_at", "updated_at")},
                "integrity": integrity,
                "solution": json.loads((run_dir / "loading_solution.json").read_text(encoding="utf-8")),
                "audit_report": json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {},
                "strategy_plan": json.loads(strategy_path.read_text(encoding="utf-8")),
                "run_manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
                "enrichment_report": (
                    json.loads(enrichment_path.read_text(encoding="utf-8"))
                    if enrichment_path.is_file()
                    else None
                ),
                "enrichment_confirmation": (
                    json.loads(enrichment_confirmation_path.read_text(encoding="utf-8"))
                    if enrichment_confirmation_path.is_file()
                    else None
                ),
                "artifacts": _plan_files(plan),
            }
        )
    except KeyError as error:
        return _error(str(error), 404, "PLAN_NOT_FOUND")
    except ArtifactIntegrityError as error:
        return _error(str(error), 409, "ARTIFACT_INTEGRITY_FAILED")


async def download_plan_artifact(request: Request):
    try:
        plan = repository().get_plan(request.path_params["plan_id"])
        requested = Path(request.path_params["artifact_name"]).name
        if requested not in {item["name"] for item in _plan_files(plan)}:
            raise KeyError("找不到指定的方案产物。")
        run_dir = Path(plan["run_dir"]).resolve()
        candidate = run_dir / requested
        if not candidate.is_file():
            candidate = run_dir / "outputs" / requested
        resolved = candidate.resolve()
        if run_dir not in resolved.parents or not resolved.is_file():
            raise KeyError("找不到指定的方案产物。")
        return FileResponse(resolved, filename=resolved.name)
    except KeyError as error:
        return _error(str(error), 404, "ARTIFACT_NOT_FOUND")


async def create_planning_run(request: Request) -> JSONResponse:
    try:
        storage = _storage_health()
        if not storage["healthy"]:
            return _storage_unavailable_response(storage)
        payload = await _validated_payload(request, PlanningRunRequest)
        _validate_products(payload.get("products"))
        idempotency_key = _idempotency_key(request)
        job_id = f"job_{uuid4().hex}"
        run_dir = API_RUN_ROOT / (
            time.strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        )
        queued_request = dict(payload)
        queued_request.update({"job_type": "planning", "run_dir": str(run_dir)})
        stored_job_id, created = repository().enqueue_job(
            job_id,
            queued_request,
            max_attempts=SETTINGS.job_max_attempts,
            max_active_jobs=SETTINGS.max_active_jobs,
            idempotency_key=idempotency_key,
            endpoint="/v1/planning-runs",
            request_hash=_request_hash(payload),
        )
        return JSONResponse(
            {
                "job_id": stored_job_id,
                "status": repository().get_job(stored_job_id)["status"],
                "idempotent_replay": not created,
            },
            status_code=202,
        )
    except QueueCapacityError as error:
        response = _error(str(error), 503, "QUEUE_CAPACITY_REACHED")
        response.headers["retry-after"] = "30"
        return response
    except IdempotencyConflictError as error:
        return _error(str(error), 409, "IDEMPOTENCY_CONFLICT")
    except (ValueError, ValidationError) as error:
        return _error(str(error), 422, "INVALID_REQUEST")


async def get_job(request: Request) -> JSONResponse:
    try:
        job = repository().get_job(request.path_params["job_id"])
        return JSONResponse(
            {
                key: job.get(key)
                for key in (
                    "job_id",
                    "status",
                    "result",
                    "error",
                    "attempt_count",
                    "max_attempts",
                    "created_at",
                    "updated_at",
                    "completed_at",
                )
            }
        )
    except KeyError as error:
        return _error(str(error), 404, "JOB_NOT_FOUND")


async def get_plan(request: Request) -> JSONResponse:
    try:
        return JSONResponse(repository().get_plan(request.path_params["plan_id"]))
    except KeyError as error:
        return _error(str(error), 404, "PLAN_NOT_FOUND")


async def approve_plan(request: Request) -> JSONResponse:
    try:
        payload = await _validated_payload(request, ApprovalRequest)
        requested_approver = str(payload.get("approved_by", "")).strip()
        approved_by, auth_error = _authorised_actor(
            request,
            requested_actor=requested_approver,
            allowed_roles={"approver"},
        )
        if auth_error:
            return auth_error
        result = planning_service().approve_plan(
            request.path_params["plan_id"],
            int(payload["expected_version"]),
            approved_by=approved_by,
        )
        return JSONResponse(result)
    except KeyError as error:
        return _error(str(error), 404, "PLAN_NOT_FOUND")
    except (ValueError, StateConflictError) as error:
        return _error(str(error), 409, "PLAN_STATE_CONFLICT")


async def reject_plan(request: Request) -> JSONResponse:
    try:
        payload = await _validated_payload(request, PlanDispositionRequest)
        rejected_by, auth_error = _authorised_actor(
            request,
            requested_actor=str(payload.get("rejected_by") or ""),
            allowed_roles={"operator", "approver"},
        )
        if auth_error:
            return auth_error
        result = planning_service().reject_plan(
            request.path_params["plan_id"],
            int(payload["expected_version"]),
            rejected_by=rejected_by,
            reason=str(payload["reason"]),
        )
        return JSONResponse(result)
    except KeyError as error:
        return _error(str(error), 404, "PLAN_NOT_FOUND")
    except (ValueError, StateConflictError) as error:
        return _error(str(error), 409, "PLAN_STATE_CONFLICT")


async def confirm_enrichment(request: Request) -> JSONResponse:
    try:
        payload = await _validated_payload(request, EnrichmentConfirmationRequest)
        requested_user = str(payload.get("confirmed_by", "")).strip()
        confirmed_by, auth_error = _authorised_actor(
            request,
            requested_actor=requested_user,
            allowed_roles={"operator", "approver"},
        )
        if auth_error:
            return auth_error
        result = EnrichmentConfirmationService(repository()).confirm(
            plan_id=request.path_params["plan_id"],
            expected_version=int(payload["expected_version"]),
            suggestion_ids=list(payload["suggestion_ids"]),
            confirmed_by=confirmed_by,
        )
        return JSONResponse(result)
    except KeyError as error:
        return _error(str(error), 404, "PLAN_NOT_FOUND")
    except FileNotFoundError as error:
        return _error(str(error), 404, "ENRICHMENT_NOT_FOUND")
    except ValueError as error:
        return _error(str(error), 409, "ENRICHMENT_CONFIRMATION_CONFLICT")


async def create_event(request: Request) -> JSONResponse:
    try:
        storage = _storage_health()
        if not storage["healthy"]:
            return _storage_unavailable_response(storage)
        _, auth_error = _authorised_actor(
            request, allowed_roles={"operator", "approver"}
        )
        if auth_error:
            return auth_error
        payload = await _validated_payload(request, DynamicEventRequest)
        simulation_error = _simulation_event_guard(payload)
        if simulation_error:
            return simulation_error
        snapshot_request_id = payload.pop("new_products_request_id", None)
        if snapshot_request_id:
            snapshot_record = repository().get_planning_request(snapshot_request_id)
            if snapshot_record["status"] != "CONFIRMED":
                raise StateConflictError("新增产品数据尚未确认，不能提交动态调整。")
            snapshot_path = Path(str(snapshot_record.get("snapshot_path") or ""))
            if not snapshot_path.is_file():
                raise StateConflictError("新增产品的确认快照不存在。")
            actual_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
            if actual_hash != snapshot_record.get("snapshot_sha256"):
                raise StateConflictError("新增产品确认快照完整性检查失败。")
            payload["new_products"] = json.loads(
                snapshot_path.read_text(encoding="utf-8")
            )
            _validate_products(payload["new_products"])
            payload["product_count"] = len(payload["new_products"])
            payload["new_products_request_id"] = snapshot_request_id
        payload["plan_id"] = request.path_params["plan_id"]
        if not payload.get("event_id"):
            payload["event_id"] = f"event_{uuid4().hex}"
        idempotency_key = _idempotency_key(request)
        for required in ("expected_version", "event_type"):
            if required not in payload:
                raise ValueError(f"{required} is required.")
        job_id = f"job_{uuid4().hex}"
        payload["job_type"] = "replanning"
        stored_job_id, created = repository().enqueue_job(
            job_id,
            payload,
            max_attempts=SETTINGS.job_max_attempts,
            max_active_jobs=SETTINGS.max_active_jobs,
            idempotency_key=idempotency_key,
            endpoint=f"/v1/plans/{request.path_params['plan_id']}/events",
            request_hash=_request_hash(payload),
        )
        return JSONResponse(
            {
                "job_id": stored_job_id,
                "status": repository().get_job(stored_job_id)["status"],
                "idempotent_replay": not created,
            },
            status_code=202,
        )
    except QueueCapacityError as error:
        response = _error(str(error), 503, "QUEUE_CAPACITY_REACHED")
        response.headers["retry-after"] = "30"
        return response
    except IdempotencyConflictError as error:
        return _error(str(error), 409, "IDEMPOTENCY_CONFLICT")
    except KeyError as error:
        return _error(str(error), 404, "PLANNING_REQUEST_NOT_FOUND")
    except StateConflictError as error:
        return _error(str(error), 409, "PLANNING_REQUEST_STATE_CONFLICT")
    except (ValueError, ValidationError) as error:
        return _error(str(error), 422, "INVALID_REQUEST")


async def create_scheduled_event(request: Request) -> JSONResponse:
    """Persist first, then collect the event in a priority scheduling window."""
    try:
        storage = _storage_health()
        if not storage["healthy"]:
            return _storage_unavailable_response(storage)
        _, auth_error = _authorised_actor(
            request, allowed_roles={"operator", "approver"}
        )
        if auth_error:
            return auth_error
        payload = await _validated_payload(request, ScheduledDynamicEventRequest)
        simulation_error = _simulation_event_guard(payload)
        if simulation_error:
            return simulation_error
        snapshot_request_id = payload.pop("new_products_request_id", None)
        if snapshot_request_id:
            snapshot_record = repository().get_planning_request(snapshot_request_id)
            if snapshot_record["status"] != "CONFIRMED":
                raise StateConflictError("新增产品数据尚未确认，不能提交动态调整。")
            snapshot_path = Path(str(snapshot_record.get("snapshot_path") or ""))
            if not snapshot_path.is_file():
                raise StateConflictError("新增产品的确认快照不存在。")
            actual_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
            if actual_hash != snapshot_record.get("snapshot_sha256"):
                raise StateConflictError("新增产品确认快照完整性检查失败。")
            payload["new_products"] = json.loads(
                snapshot_path.read_text(encoding="utf-8")
            )
            _validate_products(payload["new_products"])
            payload["product_count"] = len(payload["new_products"])
        if not payload.get("event_id"):
            payload["event_id"] = f"event_{uuid4().hex}"
        result = ScheduledEventService(repository()).schedule(
            plan_id=request.path_params["plan_id"],
            expected_version=int(payload["expected_version"]),
            payload=payload,
        )
        repository().dispatch_due_windows(
            max_attempts=SETTINGS.job_max_attempts,
            max_active_jobs=SETTINGS.max_active_jobs,
        )
        window = repository().get_scheduling_window(
            result["window"]["window_id"]
        )
        return JSONResponse(
            {
                **result,
                "window": window,
                "status": window["status"],
                "job_id": window.get("job_id"),
            },
            status_code=202,
        )
    except QueueCapacityError as error:
        response = _error(str(error), 503, "QUEUE_CAPACITY_REACHED")
        response.headers["retry-after"] = "30"
        return response
    except KeyError as error:
        return _error(str(error), 404, "PLAN_NOT_FOUND")
    except StateConflictError as error:
        return _error(str(error), 409, "PLAN_STATE_CONFLICT")
    except (ValueError, ValidationError) as error:
        return _error(str(error), 422, "INVALID_REQUEST")


async def list_scheduling_windows(request: Request) -> JSONResponse:
    try:
        _, auth_error = _authorised_actor(
            request, allowed_roles={"operator", "approver", "viewer"}
        )
        if auth_error:
            return auth_error
        status = request.query_params.get("status") or None
        limit = int(request.query_params.get("limit", "100"))
        return JSONResponse(
            {"windows": repository().list_scheduling_windows(status=status, limit=limit)}
        )
    except ValueError as error:
        return _error(str(error), 422, "INVALID_REQUEST")


async def get_scheduling_window(request: Request) -> JSONResponse:
    try:
        _, auth_error = _authorised_actor(
            request, allowed_roles={"operator", "approver", "viewer"}
        )
        if auth_error:
            return auth_error
        return JSONResponse(
            repository().get_scheduling_window(request.path_params["window_id"])
        )
    except KeyError as error:
        return _error(str(error), 404, "SCHEDULING_WINDOW_NOT_FOUND")


def _continuous_stream_response(stream: Dict[str, Any] | None) -> Dict[str, Any]:
    if stream is None:
        return {"stream": None, "window": None}
    current_window = None
    if stream.get("current_window_id"):
        try:
            current_window = repository().get_scheduling_window(
                str(stream["current_window_id"])
            )
        except KeyError:
            current_window = None
    return {"stream": stream, "window": current_window}


async def start_continuous_stream(request: Request) -> JSONResponse:
    """Start the durable UAT order stream; production simulation stays disabled."""
    try:
        if not SETTINGS.allow_simulation_events:
            return _error(
                "生产环境禁止启动模拟订单流。",
                403,
                "SIMULATION_EVENTS_DISABLED",
            )
        storage = _storage_health()
        if not storage["healthy"]:
            return _storage_unavailable_response(storage)
        actor, auth_error = _authorised_actor(
            request, allowed_roles={"operator", "approver"}, default_actor="web-operator"
        )
        if auth_error:
            return auth_error
        payload = await _validated_payload(request, ContinuousStreamStartRequest)
        plan_id = request.path_params["plan_id"]
        plan = repository().get_plan(plan_id)
        product_count = len(
            json.loads((Path(plan["run_dir"]) / "products.json").read_text(encoding="utf-8"))
        )
        requested_limit = min(int(payload["max_products"]), API_MAX_PRODUCTS)
        if product_count >= requested_limit:
            raise ValueError(
                f"当前方案已有 {product_count} 件产品，必须将持续流上限设置得更高。"
            )
        stream = repository().start_continuous_stream(
            plan_id=plan_id,
            expected_version=int(payload["expected_version"]),
            min_interval_seconds=int(payload["min_interval_seconds"]),
            max_interval_seconds=int(payload["max_interval_seconds"]),
            max_products=requested_limit,
            created_by=actor,
        )
        return JSONResponse(_continuous_stream_response(stream), status_code=202)
    except KeyError as error:
        return _error(str(error), 404, "PLAN_NOT_FOUND")
    except StateConflictError as error:
        return _error(str(error), 409, "CONTINUOUS_STREAM_CONFLICT")
    except (ValueError, ValidationError, json.JSONDecodeError) as error:
        return _error(str(error), 422, "INVALID_CONTINUOUS_STREAM")


async def get_active_continuous_stream(request: Request) -> JSONResponse:
    _, auth_error = _authorised_actor(
        request, allowed_roles={"operator", "approver", "viewer"}
    )
    if auth_error:
        return auth_error
    return JSONResponse(
        _continuous_stream_response(repository().get_active_continuous_stream())
    )


async def get_continuous_stream(request: Request) -> JSONResponse:
    try:
        _, auth_error = _authorised_actor(
            request, allowed_roles={"operator", "approver", "viewer"}
        )
        if auth_error:
            return auth_error
        return JSONResponse(
            _continuous_stream_response(
                repository().get_continuous_stream(request.path_params["stream_id"])
            )
        )
    except KeyError as error:
        return _error(str(error), 404, "CONTINUOUS_STREAM_NOT_FOUND")


async def stop_continuous_stream(request: Request) -> JSONResponse:
    try:
        _, auth_error = _authorised_actor(
            request, allowed_roles={"operator", "approver"}
        )
        if auth_error:
            return auth_error
        stream = repository().request_continuous_stream_stop(
            request.path_params["stream_id"]
        )
        return JSONResponse(_continuous_stream_response(stream), status_code=202)
    except KeyError as error:
        return _error(str(error), 404, "CONTINUOUS_STREAM_NOT_FOUND")


async def openapi(_: Request) -> JSONResponse:
    return JSONResponse(openapi_document())


async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", uuid4().hex)
    LOGGER.exception(
        "unhandled_request_error",
        extra={
            "event": "unhandled_request_error",
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
        },
    )
    response = _error(
        "服务未能完成该请求，请记录支持编号后联系系统管理员。",
        500,
        "INTERNAL_SERVICE_ERROR",
    )
    response.headers["x-request-id"] = request_id
    return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        public_path = (
            request.url.path.startswith("/health")
            or request.url.path == "/openapi.json"
        )
        if API_KEY and not public_path:
            if request.headers.get("x-api-key") != API_KEY:
                return _error("访问凭证无效或缺失。", 401, "UNAUTHORISED")
        return await call_next(request)


class RequestControlMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        supplied_request_id = request.headers.get("x-request-id", "")
        request_id = (
            supplied_request_id
            if re.fullmatch(r"[A-Za-z0-9._:-]{1,64}", supplied_request_id)
            else uuid4().hex
        )
        request.state.request_id = request_id
        if SETTINGS.production and request.url.path.startswith(("/v1/", "/v2/")):
            trusted_user = request.headers.get("x-user-id", "").strip()
            trusted_roles = request.headers.get("x-user-roles", "").strip()
            if not trusted_user or not trusted_roles:
                response = _error(
                    "生产业务接口需要可信的企业用户身份。",
                    403,
                    "TRUSTED_IDENTITY_REQUIRED",
                )
                response.headers["x-request-id"] = request_id
                return response
        if request.method in {"POST", "PUT", "PATCH"}:
            content_length = request.headers.get("content-length")
            if SETTINGS.production and content_length is None:
                response = _error(
                    "请求缺少内容长度信息。", 411, "CONTENT_LENGTH_REQUIRED"
                )
                response.headers["x-request-id"] = request_id
                return response
            if content_length is not None:
                body_limit = (
                    SETTINGS.max_upload_bytes
                    if request.url.path in {"/v1/data-intakes", "/v2/planning-requests"}
                    else SETTINGS.max_body_bytes
                )
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = body_limit + 1
                if declared_size > body_limit:
                    response = _error(
                        "上传或请求内容超过系统允许的大小。", 413, "REQUEST_TOO_LARGE"
                    )
                    response.headers["x-request-id"] = request_id
                    return response
            upload_path = request.url.path in {
                "/v1/data-intakes",
                "/v2/planning-requests",
            }
            if not upload_path:
                body = await request.body()
                if len(body) > SETTINGS.max_body_bytes:
                    response = _error(
                        "上传或请求内容超过系统允许的大小。",
                        413,
                        "REQUEST_TOO_LARGE",
                    )
                    response.headers["x-request-id"] = request_id
                    return response

        if not request.url.path.startswith("/health"):
            identifier = request.headers.get("x-user-id", "").strip() or (
                request.client.host if request.client else "unknown"
            )
            if request.method == "GET" and request.url.path.startswith("/v1/jobs/"):
                endpoint_class = "job_poll"
                limit = SETTINGS.rate_limit_per_minute * 5
            elif request.method == "GET":
                endpoint_class = "read"
                limit = SETTINGS.rate_limit_per_minute * 2
            else:
                endpoint_class = "write"
                limit = SETTINGS.rate_limit_per_minute
            rate_key = f"{identifier}:{endpoint_class}"
            window = int(time.time() // 60)
            with _RATE_LIMIT_LOCK:
                if len(_RATE_LIMIT_WINDOWS) > 10_000:
                    stale_keys = [
                        key
                        for key, (stored, _) in _RATE_LIMIT_WINDOWS.items()
                        if stored < window
                    ]
                    for key in stale_keys:
                        _RATE_LIMIT_WINDOWS.pop(key, None)
                stored_window, count = _RATE_LIMIT_WINDOWS.get(rate_key, (window, 0))
                if stored_window != window:
                    stored_window, count = window, 0
                count += 1
                _RATE_LIMIT_WINDOWS[rate_key] = (stored_window, count)
            if count > limit:
                response = _error("操作过于频繁，请稍后重试。", 429, "RATE_LIMITED")
                response.headers["retry-after"] = "60"
                response.headers["x-request-id"] = request_id
                return response

        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = (
            f"{(time.perf_counter() - started) * 1000:.1f}"
        )
        response.headers["x-content-type-options"] = "nosniff"
        LOGGER.info(
            "request_completed",
            extra={
                "event": "request_completed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "response_time_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return response


app = Starlette(
    debug=False,
    lifespan=lifespan,
    exception_handlers={Exception: unexpected_error},
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/health/live", liveness, methods=["GET"]),
        Route("/health/ready", readiness, methods=["GET"]),
        Route("/metrics", metrics, methods=["GET"]),
        Route("/openapi.json", openapi, methods=["GET"]),
        Route("/v1/data-intakes", create_data_intake, methods=["POST"]),
        Route("/v1/data-intakes/sample", create_sample_data_intake, methods=["POST"]),
        Route("/v2/planning-requests", create_planning_request, methods=["POST"]),
        Route(
            "/v2/planning-requests/sample",
            create_sample_data_intake,
            methods=["POST"],
        ),
        Route(
            "/v2/planning-requests/{request_id}",
            get_planning_request,
            methods=["GET"],
        ),
        Route(
            "/v2/planning-requests/{request_id}/review",
            review_planning_request,
            methods=["POST"],
        ),
        Route(
            "/v2/planning-requests/{request_id}/confirm",
            confirm_planning_request,
            methods=["POST"],
        ),
        Route(
            "/v2/planning-requests/{request_id}/submit",
            submit_planning_request,
            methods=["POST"],
        ),
        Route(
            "/v1/data-intakes/{intake_id}/assist",
            assist_data_intake,
            methods=["POST"],
        ),
        Route(
            "/v1/data-intakes/{intake_id}/confirm",
            confirm_data_intake,
            methods=["POST"],
        ),
        Route(
            "/v1/data-intakes/{intake_id}/preview",
            preview_data_intake,
            methods=["POST"],
        ),
        Route("/v1/planning-runs", create_planning_run, methods=["POST"]),
        Route("/v1/jobs/{job_id}", get_job, methods=["GET"]),
        Route("/v1/plans", list_plans, methods=["GET"]),
        Route("/v1/session", current_session, methods=["GET"]),
        Route("/v1/analytics/dispatch-dashboard", dispatch_dashboard, methods=["GET"]),
        Route("/v1/analytics/assistant/query", analytics_assistant, methods=["POST"]),
        Route("/v1/plans/{plan_id}", get_plan, methods=["GET"]),
        Route("/v1/plans/{plan_id}/result", get_plan_result, methods=["GET"]),
        Route(
            "/v1/plans/{plan_id}/artifacts/{artifact_name}",
            download_plan_artifact,
            methods=["GET"],
        ),
        Route("/v1/plans/{plan_id}/approve", approve_plan, methods=["POST"]),
        Route("/v1/plans/{plan_id}/reject", reject_plan, methods=["POST"]),
        Route(
            "/v1/plans/{plan_id}/enrichment-confirmations",
            confirm_enrichment,
            methods=["POST"],
        ),
        Route("/v1/plans/{plan_id}/events", create_event, methods=["POST"]),
        Route(
            "/v1/plans/{plan_id}/scheduled-events",
            create_scheduled_event,
            methods=["POST"],
        ),
        Route(
            "/v1/scheduling/windows", list_scheduling_windows, methods=["GET"]
        ),
        Route(
            "/v1/scheduling/windows/{window_id}",
            get_scheduling_window,
            methods=["GET"],
        ),
        Route(
            "/v1/plans/{plan_id}/continuous-stream",
            start_continuous_stream,
            methods=["POST"],
        ),
        Route(
            "/v1/continuous-streams/active",
            get_active_continuous_stream,
            methods=["GET"],
        ),
        Route(
            "/v1/continuous-streams/{stream_id}",
            get_continuous_stream,
            methods=["GET"],
        ),
        Route(
            "/v1/continuous-streams/{stream_id}/stop",
            stop_continuous_stream,
            methods=["POST"],
        ),
    ],
)
app.add_middleware(RequestControlMiddleware)
app.add_middleware(APIKeyMiddleware)
