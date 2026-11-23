"""Application service shared by UIs and external APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from Agents.audit_service import build_audit_report, write_audit_report_files
from Agents.core.research_contracts import PlanStatus, PlanVersion, VersionInfo
from Agents.planning_orchestrator import PlanningOrchestrator, PlanningRequest
from Agents.run_manifest import RunManifest
from Agents.runtime_config import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_RANDOM_SEED,
)
from Agents.state_repository import SQLiteStateRepository
from Agents.validation_agent import ValidationAgent
from Agents.visualization_agent import VisualizationAgent

STRATEGY_OPTIONS = {
    "Agent decides": "destination_grouping_plus_balance_aware_greedy",
    "Fast greedy": "balance_aware_greedy",
    "Destination grouping": "destination_grouping_greedy",
    "Balance-aware grouping": "destination_grouping_plus_balance_aware_greedy",
    "Greedy + simulated annealing": "greedy_plus_simulated_annealing",
    "Rule-based loading": "rule_based_loading",
}


def build_strategy_plan(
    strategy_label: str,
    time_budget_seconds: Optional[float],
    force_sa: bool = False,
    disable_sa: bool = False,
) -> Dict[str, Any]:
    selected = STRATEGY_OPTIONS[strategy_label]
    config = {
        "group_by_destination": selected
        in {
            "destination_grouping_greedy",
            "destination_grouping_plus_balance_aware_greedy",
            "greedy_plus_simulated_annealing",
            "rule_based_loading",
        },
        "sort_by_weight_desc": True,
        "use_balance_aware_scoring": selected != "simple_greedy",
        "use_simulated_annealing": selected == "greedy_plus_simulated_annealing",
        "use_rule_based_constraints": True,
    }
    if strategy_label == "Agent decides":
        config["use_simulated_annealing"] = False
    if force_sa:
        selected = "greedy_plus_simulated_annealing"
        config["use_simulated_annealing"] = True
    if disable_sa:
        config["use_simulated_annealing"] = False
    if time_budget_seconds is not None:
        config["time_budget_seconds"] = float(time_budget_seconds)
    return {
        "scenario_type": "agent_demo",
        "selected_strategy": selected,
        "strategy_reason": (
            "Selected at the application boundary; the OptimisationAgent may "
            "normalise it through the bounded algorithm policy."
        ),
        "optimisation_config": config,
        "required_constraints": [
            "weight_limit",
            "boundary_check",
            "overlap_check",
            "centerline_balance",
            "axle_balance",
            "diagonal_balance",
        ],
    }


@dataclass(frozen=True)
class PlanCommand:
    products_path: Path
    run_dir: Path
    strategy_plan: Dict[str, Any]
    data_intake_audit: Dict[str, Any]
    execution_mode: str = "multi_agent_council"
    scenario_id: str = "normal"
    system_variant: str = "multi_agent_council"
    time_budget_seconds: Optional[float] = None
    random_seed: int = DEFAULT_RANDOM_SEED
    generate_visuals: bool = True
    llm_provider_name: str = DEFAULT_LLM_PROVIDER
    llm_model_name: str = DEFAULT_LLM_MODEL
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    semantic_planning_enabled: bool = False
    auto_activate: bool = False
    plan_id: Optional[str] = None
    plan_version: int = 1
    parent_plan_id: Optional[str] = None


class PlanningService:
    """Coordinates a complete plan without depending on Streamlit."""

    def __init__(
        self,
        repository: SQLiteStateRepository,
        orchestrator: Optional[PlanningOrchestrator] = None,
    ) -> None:
        self.repository = repository
        self.orchestrator = orchestrator or PlanningOrchestrator()

    def create_plan(self, command: PlanCommand) -> Dict[str, Any]:
        orchestration = self.orchestrator.run(
            PlanningRequest(
                products_path=command.products_path,
                run_dir=command.run_dir,
                strategy_plan=command.strategy_plan,
                execution_mode=command.execution_mode,
                scenario_id=command.scenario_id,
                system_variant=command.system_variant,
                time_budget_seconds=command.time_budget_seconds,
                random_seed=command.random_seed,
                llm_provider_name=command.llm_provider_name,
                llm_model_name=command.llm_model_name,
                llm_base_url=command.llm_base_url,
                semantic_planning_enabled=command.semantic_planning_enabled,
                auto_activate=command.auto_activate,
                plan_id=command.plan_id,
                plan_version=command.plan_version,
                parent_plan_id=command.parent_plan_id,
            )
        )
        solution = orchestration["solution"]
        visualization_summary = None
        output_dir = None
        if command.generate_visuals:
            output_dir = command.run_dir / "outputs"
            visualization_summary = VisualizationAgent(
                loading_solution_path=orchestration["solution_path"],
                output_dir=output_dir,
                summary_output_path=command.run_dir / "visualization_summary.json",
                figure_dpi=160,
            ).run()
        selected_strategy_plan = solution.get("multi_agent_council", {}).get(
            "selected_strategy_plan", orchestration["strategy_plan"]
        )
        audit_report = build_audit_report(
            solution=solution,
            run_dir=command.run_dir,
            products_path=command.products_path,
            data_intake_audit=command.data_intake_audit,
            strategy_plan=selected_strategy_plan,
            solution_path=orchestration["solution_path"],
            output_dir=output_dir,
            visualization_summary=visualization_summary,
        )
        audit_json_path, audit_markdown_path = write_audit_report_files(
            command.run_dir, audit_report
        )
        plan_record = self.repository.save_plan(orchestration)
        wagon_summary = solution.get("wagon_summary", {})
        PlanningOrchestrator.write_json(
            command.run_dir / "plan_summary.json",
            {
                "schema_version": "1.0",
                "product_count": int(wagon_summary.get("total_products", 0)),
                "wagon_count": int(wagon_summary.get("wagon_count", 0)),
                "destination_count": len(
                    solution.get("input_summary", {}).get("destinations", [])
                ),
                "is_valid": bool(
                    solution.get("validation_summary", {}).get("is_valid")
                ),
            },
        )
        return {
            **orchestration,
            "run_dir": str(command.run_dir),
            "products_path": str(command.products_path),
            "solution_path": str(orchestration["solution_path"]),
            "strategy_path": str(orchestration["strategy_path"]),
            "manifest_path": str(orchestration["manifest_path"]),
            "audit_report": audit_report,
            "audit_json_path": str(audit_json_path),
            "audit_markdown_path": str(audit_markdown_path),
            "visualization_summary": visualization_summary,
            "output_dir": str(output_dir) if output_dir else None,
            "plan_record": plan_record,
            "data_intake_audit": command.data_intake_audit,
            "strategy_plan": selected_strategy_plan,
        }

    def persist_precomputed_plan(
        self,
        *,
        solution: Dict[str, Any],
        products_path: Path,
        run_dir: Path,
        strategy_plan: Dict[str, Any],
        data_intake_audit: Dict[str, Any],
        parent_plan_id: str,
        plan_version: int,
        system_variant: str,
        scope_decision: Dict[str, Any],
        unavailable_wagon_ids: tuple[str, ...] = (),
        generate_visuals: bool = False,
    ) -> Dict[str, Any]:
        """Persist a merged local-replan solution through the normal safety boundary."""
        run_dir.mkdir(parents=True, exist_ok=True)
        products = PlanningOrchestrator.load_products(products_path)
        expected = PlanningOrchestrator.material_numbers(products)
        validation = ValidationAgent().validate(
            solution,
            expected_material_numbers=expected,
            unavailable_wagon_ids=unavailable_wagon_ids,
        )
        if not validation["is_valid"]:
            raise ValueError(
                "Merged local replan failed independent validation: "
                f"{validation.get('veto_reasons', [])}"
            )
        plan = PlanVersion(
            plan_id=f"plan_{uuid4().hex}",
            version=plan_version,
            parent_plan_id=parent_plan_id,
        )
        plan.transition(PlanStatus.GENERATED)
        plan.transition(PlanStatus.VALIDATED)
        plan.transition(PlanStatus.PENDING_APPROVAL)
        plan_payload = plan.to_dict()
        solution["research_system"] = {
            "versions": VersionInfo().to_dict(),
            "plan": plan_payload,
            "independent_validation": validation,
            "replanning_scope_decision": scope_decision,
        }
        strategy_path = run_dir / "strategy_plan.json"
        solution_path = run_dir / "loading_solution.json"
        PlanningOrchestrator.write_json(strategy_path, strategy_plan)
        PlanningOrchestrator.write_json(solution_path, solution)
        manifest = RunManifest(
            run_id=run_dir.name,
            scenario_id="window_dynamic_replanning",
            system_variant=system_variant,
            random_seed=42,
        )
        manifest.set_dataset(products_path, record_count=len(products))
        manifest.set_configuration(
            execution_mode="local_replan_with_locked_wagons",
            activation_policy="manual",
            replanning_scope=scope_decision,
            unavailable_wagon_ids=list(unavailable_wagon_ids),
        )
        manifest.add_record("agent_decisions", scope_decision)
        manifest.add_record("validation_outcomes", validation)
        manifest.add_artifact("strategy_plan", strategy_path)
        manifest.add_artifact("loading_solution", solution_path)
        manifest.set_metrics(
            objective_score=solution.get("final_solution_score", {}).get(
                "objective_score"
            ),
            wagon_count=solution.get("wagon_summary", {}).get("wagon_count"),
            validation_passed=True,
        )
        manifest.finalise("completed")
        manifest_path = manifest.write(run_dir / "run_manifest.json")
        output_dir = None
        visualization_summary = None
        if generate_visuals:
            output_dir = run_dir / "outputs"
            visualization_summary = VisualizationAgent(
                loading_solution_path=solution_path,
                output_dir=output_dir,
                summary_output_path=run_dir / "visualization_summary.json",
                figure_dpi=160,
            ).run()
        audit_report = build_audit_report(
            solution=solution,
            run_dir=run_dir,
            products_path=products_path,
            data_intake_audit=data_intake_audit,
            strategy_plan=strategy_plan,
            solution_path=solution_path,
            output_dir=output_dir,
            visualization_summary=visualization_summary,
        )
        audit_json_path, audit_markdown_path = write_audit_report_files(
            run_dir, audit_report
        )
        plan_record = self.repository.save_plan_record(
            plan=plan_payload,
            run_dir=run_dir,
            solution_path=solution_path,
            manifest_path=manifest_path,
            strategy_path=strategy_path,
        )
        return {
            "solution": solution,
            "plan": plan_payload,
            "plan_record": plan_record,
            "run_dir": str(run_dir),
            "products_path": str(products_path),
            "solution_path": str(solution_path),
            "strategy_path": str(strategy_path),
            "manifest_path": str(manifest_path),
            "audit_json_path": str(audit_json_path),
            "audit_markdown_path": str(audit_markdown_path),
        }

    def approve_plan(
        self,
        plan_id: str,
        expected_version: int,
        *,
        approved_by: str = "business-operator",
    ) -> Dict[str, Any]:
        return self.repository.approve_plan(
            plan_id, expected_version, approved_by=approved_by
        )

    def reject_plan(
        self,
        plan_id: str,
        expected_version: int,
        *,
        rejected_by: str,
        reason: str,
    ) -> Dict[str, Any]:
        return self.repository.reject_plan(
            plan_id,
            expected_version,
            rejected_by=rejected_by,
            reason=reason,
        )
