"""Unified planning entry point for research-system-v2.0.0."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from Agents.core.research_contracts import (
    AgenticSystemError,
    ErrorCode,
    PlanStatus,
    PlanVersion,
    StrategyDecisionTrace,
    StructuredError,
    VersionInfo,
)
from Agents.multi_agent_council import MultiAgentCouncil
from Agents.optimisation_agent import OptimisationAgent
from Agents.run_manifest import RunManifest
from Agents.runtime_config import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
)
from Agents.semantic_planner import SemanticPlanner
from Agents.strategy_policy import (
    deterministic_strategy_plan,
    validate_strategy_plan,
)
from Agents.validation_agent import ValidationAgent


@dataclass(frozen=True)
class PlanningRequest:
    products_path: Path
    run_dir: Path
    strategy_plan: Optional[Dict[str, Any]] = None
    execution_mode: str = "single_strategy"
    scenario_id: str = "normal"
    system_variant: str = "full_hybrid"
    time_budget_seconds: Optional[float] = None
    random_seed: int = 42
    plan_id: Optional[str] = None
    plan_version: int = 1
    parent_plan_id: Optional[str] = None
    semantic_planning_enabled: bool = False
    semantic_strategy_provider: Optional[
        Callable[[Path, Path], Dict[str, Any]]
    ] = None
    llm_provider_name: str = DEFAULT_LLM_PROVIDER
    llm_model_name: str = DEFAULT_LLM_MODEL
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_timeout_seconds: float = 60.0
    rag_top_k: int = 6
    safety_veto_enabled: bool = True
    validator_enabled: bool = True
    auto_activate: bool = False


class PlanningOrchestrator:
    """Coordinates strategy authorisation, execution, validation, and audit."""

    def __init__(self, validation_agent: Optional[ValidationAgent] = None) -> None:
        self.validation_agent = validation_agent or ValidationAgent()

    def run(self, request: PlanningRequest) -> Dict[str, Any]:
        started = time.perf_counter()
        request.run_dir.mkdir(parents=True, exist_ok=True)
        products = self.load_products(request.products_path)
        material_numbers = self.material_numbers(products)
        manifest = RunManifest(
            run_id=request.run_dir.name,
            scenario_id=request.scenario_id,
            system_variant=request.system_variant,
            random_seed=request.random_seed,
        )
        manifest.set_dataset(
            request.products_path,
            record_count=len(products),
        )
        manifest.set_configuration(
            execution_mode=request.execution_mode,
            time_budget_seconds=request.time_budget_seconds,
            semantic_planning_enabled=request.semantic_planning_enabled,
            llm_invocation_expected=request.semantic_planning_enabled,
            llm_provider_name=request.llm_provider_name,
            llm_model_name=request.llm_model_name,
            llm_base_url=request.llm_base_url,
            llm_timeout_seconds=request.llm_timeout_seconds,
            rag_top_k=request.rag_top_k,
            safety_veto_enabled=request.safety_veto_enabled,
            validator_enabled=request.validator_enabled,
            activation_policy=("automatic" if request.auto_activate else "manual"),
        )
        plan_version = PlanVersion(
            plan_id=request.plan_id or f"plan_{uuid4().hex}",
            version=request.plan_version,
            parent_plan_id=request.parent_plan_id,
        )
        try:
            strategy_plan, strategy_trace = self.authorise_strategy(
                request,
                products,
            )
            strategy_path = request.run_dir / "strategy_plan.json"
            self.write_json(strategy_path, strategy_plan)
            manifest.add_artifact("strategy_plan", strategy_path)
            execution_started = time.perf_counter()
            if request.execution_mode == "multi_agent_council":
                solution = self.run_council(
                    request,
                    strategy_plan,
                )
            elif request.execution_mode == "single_strategy":
                solution, _ = self.run_candidate(
                    request.products_path,
                    strategy_plan,
                    request.run_dir,
                    request.random_seed,
                )
            else:
                raise ValueError(
                    f"Unsupported execution_mode: {request.execution_mode}"
                )
            execution_seconds = time.perf_counter() - execution_started
            plan_version.transition(PlanStatus.GENERATED)

            validation = (
                self.validation_agent.validate(
                    solution,
                    expected_material_numbers=material_numbers,
                    unavailable_wagon_ids=(
                        strategy_plan.get("optimisation_config", {}).get(
                            "disabled_wagon_ids", []
                        )
                    ),
                )
                if request.validator_enabled
                else {
                    "is_valid": True,
                    "validation_skipped": True,
                    "warning": "Independent validation disabled for isolated ablation.",
                    "veto_reasons": [],
                }
            )
            manifest.add_record("validation_outcomes", validation)
            if not validation["is_valid"]:
                plan_version.transition(PlanStatus.REJECTED)
                raise AgenticSystemError(
                    StructuredError(
                        code=ErrorCode.VALIDATION_FAILED,
                        message="Final solution failed independent validation.",
                        requires_manual_review=True,
                        details={
                            "veto_reasons": validation.get("veto_reasons", [])
                        },
                    )
                )
            plan_version.transition(PlanStatus.VALIDATED)
            if request.auto_activate:
                plan_version.transition(PlanStatus.APPROVED)
                plan_version.transition(PlanStatus.ACTIVE)
            else:
                plan_version.transition(PlanStatus.PENDING_APPROVAL)

            executed_strategy = solution.get("strategy_execution", {}).get(
                "applied_strategy"
            )
            strategy_trace.executed_strategy = executed_strategy
            strategy_trace.execution_source = "optimisation_agent"
            strategy_trace.override_reasons.extend(
                solution.get("strategy_execution", {}).get(
                    "resolution_notes", []
                )
            )
            # Record the trace only after execution has populated the actual
            # strategy. RunManifest.add_record deep-copies its input, so an
            # earlier write would permanently retain executed_strategy=None.
            manifest.add_record("agent_decisions", strategy_trace.to_dict())
            solution["research_system"] = {
                "versions": VersionInfo().to_dict(),
                "plan": plan_version.to_dict(),
                "strategy_decision_trace": strategy_trace.to_dict(),
                "independent_validation": validation,
            }
            solution_path = request.run_dir / "loading_solution.json"
            self.write_json(solution_path, solution)
            manifest.add_artifact("loading_solution", solution_path)
            council = solution.get("multi_agent_council")
            if isinstance(council, dict):
                manifest.add_record(
                    "agent_decisions",
                    {
                        "decision_type": "multi_agent_council",
                        "coordinator_decision": council.get(
                            "coordinator_decision", {}
                        ),
                        "diversity_summary": council.get(
                            "diversity_summary", {}
                        ),
                        "deadline": council.get("deadline", {}),
                    },
                )
                for artifact_name, artifact_path in (
                    council.get("artifacts") or {}
                ).items():
                    manifest.add_artifact(artifact_name, artifact_path)
            manifest.add_stage(
                "planning_execution",
                "completed",
                execution_seconds,
                execution_mode=request.execution_mode,
            )
            score = solution.get("final_solution_score", {})
            manifest.set_metrics(
                objective_score=score.get("objective_score"),
                wagon_count=solution.get("wagon_summary", {}).get("wagon_count"),
                total_runtime_seconds=time.perf_counter() - started,
                validation_passed=True,
            )
            manifest.finalise("completed")
            manifest_path = manifest.write(request.run_dir / "run_manifest.json")
            return {
                "solution": solution,
                "solution_path": solution_path,
                "strategy_plan": strategy_plan,
                "strategy_path": strategy_path,
                "manifest_path": manifest_path,
                "plan": plan_version.to_dict(),
            }
        except Exception as error:
            structured = (
                error.to_dict()
                if isinstance(error, AgenticSystemError)
                else StructuredError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=str(error),
                ).to_dict()
            )
            manifest.finalise("failed", error=structured)
            manifest.write(request.run_dir / "run_manifest.json")
            raise

    def authorise_strategy(
        self,
        request: PlanningRequest,
        products: List[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], StrategyDecisionTrace]:
        requested = request.strategy_plan
        if requested is None:
            if request.semantic_planning_enabled:
                try:
                    provider = request.semantic_strategy_provider
                    semantic_plan = (
                        provider(request.products_path, request.run_dir)
                        if provider is not None
                        else SemanticPlanner(
                            model_name=request.llm_model_name,
                            base_url=request.llm_base_url,
                            top_k=request.rag_top_k,
                            request_timeout_seconds=request.llm_timeout_seconds,
                            seed=request.random_seed,
                        ).recommend(request.products_path, request.run_dir)
                    )
                    authorised = validate_strategy_plan(semantic_plan)
                    return authorised, StrategyDecisionTrace(
                        requested_strategy=None,
                        recommended_strategy=authorised["selected_strategy"],
                        authorised_strategy=authorised["selected_strategy"],
                        recommendation_source="rag_grounded_llm",
                        authorisation_source="bounded_strategy_schema",
                    )
                except Exception as error:
                    authorised = deterministic_strategy_plan(
                        products,
                        time_budget_seconds=request.time_budget_seconds,
                        reason=(
                            "Semantic planner failed; deterministic fallback: "
                            f"{error}"
                        ),
                    )
                    authorised["semantic_failure"] = StructuredError(
                        code=ErrorCode.LLM_UNAVAILABLE,
                        message=str(error),
                        retryable=True,
                    ).to_dict()
                    return validate_strategy_plan(authorised), StrategyDecisionTrace(
                        requested_strategy=None,
                        recommended_strategy=None,
                        authorised_strategy=authorised["selected_strategy"],
                        fallback_strategy=authorised["selected_strategy"],
                        recommendation_source="semantic_planner_failed",
                        authorisation_source="deterministic_fallback_policy",
                        fallback_reasons=[str(error)],
                    )
            authorised = deterministic_strategy_plan(
                products,
                time_budget_seconds=request.time_budget_seconds,
            )
            trace = StrategyDecisionTrace(
                requested_strategy=None,
                recommended_strategy=authorised["selected_strategy"],
                authorised_strategy=authorised["selected_strategy"],
                recommendation_source="deterministic_policy",
                authorisation_source="deterministic_policy",
            )
            return validate_strategy_plan(authorised), trace
        try:
            authorised = validate_strategy_plan(requested)
            trace = StrategyDecisionTrace(
                requested_strategy=requested.get("selected_strategy"),
                recommended_strategy=requested.get("selected_strategy"),
                authorised_strategy=authorised["selected_strategy"],
                recommendation_source=str(
                    requested.get("decision_source", "external_request")
                ),
                authorisation_source="bounded_strategy_schema",
            )
            return authorised, trace
        except ValueError as error:
            fallback = deterministic_strategy_plan(
                products,
                time_budget_seconds=request.time_budget_seconds,
                reason=f"Invalid requested strategy; deterministic fallback: {error}",
            )
            trace = StrategyDecisionTrace(
                requested_strategy=requested.get("selected_strategy"),
                authorised_strategy=fallback["selected_strategy"],
                fallback_strategy=fallback["selected_strategy"],
                recommendation_source="rejected_external_request",
                authorisation_source="deterministic_fallback_policy",
                fallback_reasons=[str(error)],
            )
            return validate_strategy_plan(fallback), trace

    def run_council(
        self,
        request: PlanningRequest,
        strategy_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        def callback(
            products_path: Path,
            candidate_plan: Dict[str, Any],
            candidate_dir: Path,
        ) -> tuple[Dict[str, Any], Path]:
            return self.run_candidate(
                products_path,
                candidate_plan,
                candidate_dir,
                request.random_seed,
            )

        council = MultiAgentCouncil(
            request.time_budget_seconds,
            safety_veto_enabled=request.safety_veto_enabled,
        )
        result = council.run(
            products_path=request.products_path,
            base_strategy_plan=strategy_plan,
            run_dir=request.run_dir,
            run_candidate=callback,
        )
        solution = result["selected_solution"]
        solution["multi_agent_council"] = {
            key: value
            for key, value in result.items()
            if key not in {"selected_solution", "selected_strategy_plan"}
        }
        solution["multi_agent_council"]["selected_strategy_plan"] = result[
            "selected_strategy_plan"
        ]
        return solution

    @staticmethod
    def run_candidate(
        products_path: Path,
        strategy_plan: Dict[str, Any],
        run_dir: Path,
        seed: int,
    ) -> tuple[Dict[str, Any], Path]:
        run_dir.mkdir(parents=True, exist_ok=True)
        strategy_path = run_dir / "strategy_plan.json"
        solution_path = run_dir / "loading_solution.json"
        PlanningOrchestrator.write_json(strategy_path, strategy_plan)
        solution = OptimisationAgent(
            products_json_path=products_path,
            strategy_plan_path=strategy_path,
            output_path=solution_path,
            seed=seed,
        ).run()
        return solution, solution_path

    @staticmethod
    def load_products(path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as file:
            products = json.load(file)
        if not isinstance(products, list) or not all(
            isinstance(product, dict) for product in products
        ):
            raise ValueError("products_path must contain a JSON list of objects.")
        return products

    @staticmethod
    def material_numbers(products: List[Dict[str, Any]]) -> List[str]:
        return [
            str(
                product.get("material_no")
                or product.get("product_id")
                or product.get("id")
            )
            for product in products
        ]

    @staticmethod
    def write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{uuid4().hex[:10]}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
