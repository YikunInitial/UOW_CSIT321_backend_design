"""Durable background-job execution for planning and replanning requests."""

from __future__ import annotations

import errno
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from Agents.enrichment_agent import EnrichmentAgent
from Agents.llm_provider import LLMProviderError
from Agents.planning_orchestrator import PlanningOrchestrator
from Agents.planning_service import PlanCommand, PlanningService, build_strategy_plan
from Agents.provider_semantic_planner import ProviderSemanticPlanner
from Agents.replanning_service import ReplanningCommand, ReplanningService
from Agents.run_manifest import sha256_file
from Agents.runtime_config import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
)
from Agents.state_repository import SQLiteStateRepository


class RetryableJobError(RuntimeError):
    """Failure produced outside the executor that is safe to retry."""


class PermanentJobError(RuntimeError):
    """Failure produced outside the executor that must not be retried."""


def _exception_chain(error: BaseException) -> list[BaseException]:
    chain = []
    current: BaseException | None = error
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def is_retryable_job_error(error: BaseException) -> bool:
    """Classify only well-known transient failures as retryable."""
    for candidate in _exception_chain(error):
        if isinstance(candidate, RetryableJobError):
            return True
        if isinstance(candidate, PermanentJobError):
            return False
        if isinstance(candidate, LLMProviderError):
            return candidate.retryable
        if isinstance(candidate, (TimeoutError, ConnectionError)):
            return True
        if isinstance(candidate, sqlite3.OperationalError):
            message = str(candidate).lower()
            return "locked" in message or "busy" in message
        if isinstance(candidate, OSError):
            return candidate.errno in {
                errno.EAGAIN,
                errno.EBUSY,
                errno.ECONNABORTED,
                errno.ECONNREFUSED,
                errno.ECONNRESET,
                errno.ETIMEDOUT,
            }
    return False


def job_error_payload(error: BaseException) -> Dict[str, Any]:
    message = " ".join(str(error).split())[:500]
    payload: Dict[str, Any] = {
        "type": type(error).__name__,
        "message": message or "Job execution failed.",
    }
    if isinstance(error, LLMProviderError):
        payload["code"] = error.code
        payload["status_code"] = error.status_code
    return payload


def validate_products(
    products: Any, *, max_products: int = 2000
) -> list[Dict[str, Any]]:
    if not isinstance(products, list) or not products:
        raise ValueError("products must be a non-empty JSON array.")
    if len(products) > max_products:
        raise ValueError(f"products exceeds the {max_products} record limit.")
    required = ("destination", "width")
    for index, product in enumerate(products, start=1):
        if not isinstance(product, dict):
            raise ValueError(f"products[{index}] must be an object.")
        if not (product.get("material_no") or product.get("product_id")):
            raise ValueError(f"products[{index}] is missing a product identifier.")
        for field in required:
            if product.get(field) in (None, ""):
                raise ValueError(f"products[{index}] is missing {field}.")
        if not (product.get("outer_diameter") or product.get("length")):
            raise ValueError(f"products[{index}] is missing a loading length.")
        if not (product.get("gross_weight_tonne") or product.get("weight")):
            raise ValueError(f"products[{index}] is missing weight.")
    return products


class JobExecutor:
    """Execute one already-claimed job without owning queue lifecycle state."""

    def __init__(
        self, repository: SQLiteStateRepository, *, max_products: int = 2000
    ) -> None:
        self.repository = repository
        self.max_products = max_products

    def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        job_type = request.get("job_type")
        if job_type == "planning":
            return self._execute_planning(request)
        if job_type == "replanning":
            return self._execute_replanning(request)
        if job_type == "window_replanning":
            return ReplanningService(
                self.repository,
                PlanningService(self.repository),
                max_products=self.max_products,
            ).replan_window(request)
        raise ValueError(f"Unsupported job_type: {job_type!r}")

    def _execute_planning(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        products = validate_products(
            payload.get("products"), max_products=self.max_products
        )
        run_dir = Path(payload["run_dir"]).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        products_path = run_dir / "products.json"
        PlanningOrchestrator.write_json(products_path, products)
        delivery_artifacts: list[Dict[str, Any]] = []

        def copy_shared_artifact(
            payload_field: str,
            artifact_name: str,
            target_name: str,
            *,
            expected_hash: str | None = None,
        ) -> Path | None:
            path_value = payload.get(payload_field)
            if not path_value:
                return None
            source = Path(str(path_value)).resolve()
            shared_root = run_dir.parent.resolve()
            if shared_root not in source.parents or not source.is_file():
                raise ValueError(f"{payload_field} is outside the shared run root.")
            if expected_hash and sha256_file(source) != expected_hash:
                raise ValueError(f"{payload_field} failed SHA-256 verification.")
            target = run_dir / target_name
            shutil.copy2(source, target)
            delivery_artifacts.append(
                {
                    "name": artifact_name,
                    "path": str(target),
                    "sha256": sha256_file(target),
                }
            )
            return target

        source_path_value = payload.get("source_input_path")
        source_suffix = (
            Path(str(source_path_value)).suffix.lower() if source_path_value else ".bin"
        )
        original_input_path = copy_shared_artifact(
            "source_input_path",
            "original_input",
            f"original_input{source_suffix}",
            expected_hash=payload.get("source_input_sha256"),
        )
        repaired_source_path = copy_shared_artifact(
            "repaired_source_path",
            "repaired_source",
            "repaired_source.xlsx",
        )
        repair_audit_path = copy_shared_artifact(
            "repair_audit_path",
            "data_repair_audit",
            "data_repair_audit.json",
        )
        authoritative_master_path = None
        master_path_value = payload.get("authoritative_master_path")
        if master_path_value:
            master_suffix = Path(str(master_path_value)).suffix.lower() or ".bin"
            authoritative_master_path = copy_shared_artifact(
                "authoritative_master_path",
                "authoritative_master",
                f"authoritative_master{master_suffix}",
            )
        repaired_products_path = None
        if repair_audit_path is not None:
            repaired_products_path = run_dir / "repaired_products.xlsx"
            pd.DataFrame(products).to_excel(repaired_products_path, index=False)
            delivery_artifacts.append(
                {
                    "name": "repaired_products_workbook",
                    "path": str(repaired_products_path),
                    "sha256": sha256_file(repaired_products_path),
                }
            )
        time_budget = float(payload.get("time_budget_seconds", 12.0))
        strategy = payload.get("strategy_plan") or build_strategy_plan(
            "Agent decides", time_budget
        )
        intake_audit = dict(payload.get("data_intake_audit") or {})
        intake_audit.update(
            {
                "source": "steel_loading_http_api",
                "record_count": len(products),
            }
        )
        enrichment_report_path: Path | None = None
        if bool(payload.get("optional_enrichment_enabled", False)):
            enrichment_report_path = run_dir / "enrichment_report.json"
            try:
                enrichment_report = EnrichmentAgent(
                    provider_prefix="CLIENT_LLM",
                    provider_name=str(
                        payload.get("llm_provider_name", DEFAULT_LLM_PROVIDER)
                    ),
                    model_name=str(payload.get("llm_model_name", DEFAULT_LLM_MODEL)),
                    base_url=str(payload.get("llm_base_url", DEFAULT_LLM_BASE_URL)),
                    timeout_seconds=float(
                        os.getenv("CLIENT_LLM_TIMEOUT_SECONDS", "60")
                    ),
                    seed=int(payload.get("random_seed", 42)),
                    max_products=int(
                        payload.get("optional_enrichment_max_products", 25)
                    ),
                ).run(
                    products,
                    enrichment_report_path,
                    requested_fields=payload.get("optional_enrichment_fields"),
                )
            except Exception as error:
                enrichment_report = {
                    "schema_version": "1.0",
                    "status": "safe_fallback",
                    "policy": {
                        "mode": "evidence_driven_repair_suggestions",
                        "loading_input_mutated": False,
                    },
                    "suggestions": [],
                    "errors": [
                        {
                            "type": type(error).__name__,
                            "message": "Optional enrichment was unavailable.",
                        }
                    ],
                }
                PlanningOrchestrator.write_json(
                    enrichment_report_path, enrichment_report
                )
            intake_audit["optional_enrichment"] = {
                "enabled": True,
                "status": enrichment_report.get("status"),
                "suggestion_count": len(enrichment_report.get("suggestions", [])),
                "rejected_suggestion_count": len(
                    enrichment_report.get("rejected_suggestions", [])
                ),
                "loading_input_mutated": False,
                "operator_confirmation_required": True,
            }
        semantic_enabled = bool(payload.get("semantic_planning_enabled", False))
        if semantic_enabled:
            try:
                strategy = ProviderSemanticPlanner(
                    knowledge_base_dir=Path("KnowledgeBase").resolve(),
                    vector_store_dir=Path("VectorDB/chroma_db").resolve(),
                    provider_prefix="CLIENT_LLM",
                    provider_name=str(
                        payload.get("llm_provider_name", DEFAULT_LLM_PROVIDER)
                    ),
                    model_name=str(payload.get("llm_model_name", DEFAULT_LLM_MODEL)),
                    base_url=str(payload.get("llm_base_url", DEFAULT_LLM_BASE_URL)),
                    top_k=int(payload.get("rag_top_k", 6)),
                    request_timeout_seconds=float(
                        os.getenv("CLIENT_LLM_TIMEOUT_SECONDS", "60")
                    ),
                    seed=int(payload.get("random_seed", 42)),
                ).recommend(products_path, run_dir)
                intake_audit["semantic_llm"] = {
                    "enabled": True,
                    "status": "completed",
                    "provider": payload.get(
                        "llm_provider_name", DEFAULT_LLM_PROVIDER
                    ),
                    "model": payload.get("llm_model_name", DEFAULT_LLM_MODEL),
                    "rag_top_k": int(payload.get("rag_top_k", 6)),
                }
            except Exception as error:
                intake_audit["semantic_llm"] = {
                    "enabled": True,
                    "status": "deterministic_fallback",
                    "reason": type(error).__name__,
                }
        result = PlanningService(self.repository).create_plan(
            PlanCommand(
                products_path=products_path,
                run_dir=run_dir,
                strategy_plan=strategy,
                data_intake_audit=intake_audit,
                execution_mode=str(
                    payload.get("execution_mode", "multi_agent_council")
                ),
                scenario_id=str(payload.get("scenario_id", "api_request")),
                system_variant=str(payload.get("system_variant", "full_hybrid")),
                time_budget_seconds=time_budget,
                random_seed=int(payload.get("random_seed", 42)),
                generate_visuals=bool(payload.get("generate_visuals", False)),
                llm_provider_name=str(
                    payload.get("llm_provider_name", DEFAULT_LLM_PROVIDER)
                ),
                llm_model_name=str(payload.get("llm_model_name", DEFAULT_LLM_MODEL)),
                llm_base_url=str(payload.get("llm_base_url", DEFAULT_LLM_BASE_URL)),
                semantic_planning_enabled=semantic_enabled,
            )
        )
        if (
            enrichment_report_path
            and enrichment_report_path.exists()
            or delivery_artifacts
        ):
            manifest_path = Path(result["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            replacement_names = {item["name"] for item in delivery_artifacts}
            if enrichment_report_path and enrichment_report_path.exists():
                replacement_names.add("enrichment_report")
            artifacts = [
                item
                for item in manifest.get("artifacts", [])
                if item.get("name") not in replacement_names
            ]
            if enrichment_report_path and enrichment_report_path.exists():
                artifacts.append(
                    {
                        "name": "enrichment_report",
                        "path": str(enrichment_report_path),
                        "sha256": sha256_file(enrichment_report_path),
                    }
                )
                manifest.setdefault("configuration", {})[
                    "optional_enrichment_enabled"
                ] = True
            artifacts.extend(delivery_artifacts)
            manifest["artifacts"] = artifacts
            if delivery_artifacts:
                manifest.setdefault("governance", {})["data_repair"] = {
                    "status": "operator_confirmed_before_planning",
                    "original_input_mutated": False,
                    "audit_artifact": "data_repair_audit",
                }
            PlanningOrchestrator.write_json(manifest_path, manifest)
        return {
            "plan": result["plan"],
            "run_dir": result["run_dir"],
            "manifest_path": result["manifest_path"],
            "audit_json_path": result["audit_json_path"],
            "enrichment_report_path": (
                str(enrichment_report_path)
                if enrichment_report_path and enrichment_report_path.exists()
                else None
            ),
            "original_input_path": (
                str(original_input_path) if original_input_path else None
            ),
            "repaired_source_path": (
                str(repaired_source_path) if repaired_source_path else None
            ),
            "repaired_products_path": (
                str(repaired_products_path) if repaired_products_path else None
            ),
            "repair_audit_path": (
                str(repair_audit_path) if repair_audit_path else None
            ),
            "authoritative_master_path": (
                str(authoritative_master_path)
                if authoritative_master_path
                else None
            ),
        }

    def _execute_replanning(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        new_products = payload.get("new_products") or []
        if new_products:
            new_products = validate_products(new_products, max_products=self.max_products)
        return ReplanningService(
            self.repository,
            PlanningService(self.repository),
            max_products=self.max_products,
        ).replan(
            ReplanningCommand(
                plan_id=str(payload["plan_id"]),
                expected_version=int(payload["expected_version"]),
                event_type=str(payload["event_type"]),
                product_count=int(payload.get("product_count", 1)),
                source_destination=payload.get("source_destination"),
                target_destination=payload.get("target_destination"),
                affected_product_ids=tuple(payload.get("affected_product_ids") or []),
                new_products=tuple(new_products),
                add_product_mode=str(payload.get("add_product_mode", "template_simulation")),
                disabled_wagon_id=payload.get("disabled_wagon_id"),
                event_id=payload.get("event_id"),
                time_budget_seconds=float(payload.get("time_budget_seconds", 5.0)),
                generate_visuals=bool(payload.get("generate_visuals", False)),
            )
        )
