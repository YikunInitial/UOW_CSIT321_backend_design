"""Presentation-neutral construction and persistence of planning audit reports."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from Agents.planning_orchestrator import PlanningOrchestrator


def build_audit_report(
    solution: Dict[str, Any],
    run_dir: Path,
    products_path: Path,
    data_intake_audit: Dict[str, Any],
    strategy_plan: Dict[str, Any],
    solution_path: Path,
    output_dir: Optional[Path],
    visualization_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    validation = solution.get("validation_summary", {})
    independent = solution.get("research_system", {}).get("independent_validation", {})
    input_summary = solution.get("input_summary", {})
    strategy_execution = solution.get("strategy_execution", {})
    wagon_summary = solution.get("wagon_summary", {})
    plan = solution.get("research_system", {}).get("plan", {})
    coverage_pass = not any(
        independent.get(key, validation.get(key, []))
        for key in (
            "duplicate_product_ids",
            "missing_product_ids",
            "unexpected_product_ids",
        )
    )
    hard_constraint_pass = bool(independent.get("is_valid", validation.get("is_valid")))
    technical_pass = hard_constraint_pass and coverage_pass
    pdf_path = output_dir / "loading_visualisation_report.pdf" if output_dir else None
    image_count = (
        len(list(output_dir.glob("*_layout.png")))
        if output_dir and output_dir.exists()
        else 0
    )
    return {
        "run_metadata": {
            "run_id": run_dir.name,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": str(run_dir),
            "products_path": str(products_path),
            "solution_path": str(solution_path),
            "random_seed": solution.get("solution_metadata", {}).get("random_seed"),
            "runtime_seconds": solution.get("solution_metadata", {}).get(
                "runtime_seconds"
            ),
        },
        "input_audit": {
            "data_intake": data_intake_audit,
            "product_count": input_summary.get("product_count"),
            "destination_count": len(input_summary.get("destinations", [])),
            "destinations": input_summary.get("destinations", []),
            "dropped_duplicate_count": input_summary.get("dropped_duplicate_count", 0),
            "dropped_duplicates": input_summary.get("dropped_duplicates", []),
        },
        "strategy_audit": {
            "requested_strategy": strategy_execution.get("requested_strategy"),
            "applied_strategy": strategy_execution.get("applied_strategy"),
            "strategy_plan": strategy_plan,
            "execution_config": strategy_execution.get("execution_config"),
            "algorithm_policy": strategy_execution.get("algorithm_policy"),
            "resolution_notes": strategy_execution.get("resolution_notes", []),
            "fallback_reasons": strategy_execution.get("fallback_reasons", []),
        },
        "algorithm_audit": {
            "performance_summary": solution.get("performance_summary", {}),
            "anytime_summary": solution.get("anytime_summary", {}),
            "optimisation_trace": solution.get("optimisation_trace", []),
            "simulated_annealing": solution.get("simulated_annealing_log", {}),
        },
        "multi_agent_council_audit": solution.get("multi_agent_council", {}),
        "hard_constraint_audit": {
            **validation,
            "independent_validation": independent,
        },
        "output_audit": {
            "wagon_count": wagon_summary.get("wagon_count"),
            "total_products": wagon_summary.get("total_products"),
            "average_area_utilisation": wagon_summary.get("average_area_utilisation"),
            "solution_json_exists": solution_path.exists(),
            "layout_image_count": image_count,
            "pdf_report_exists": bool(pdf_path and pdf_path.exists()),
            "pdf_report_path": str(pdf_path) if pdf_path else None,
            "visualization_summary": visualization_summary,
        },
        "governance": {
            "plan_id": plan.get("plan_id"),
            "plan_version": plan.get("version"),
            "plan_status": plan.get("status"),
            "technical_validation_passed": technical_pass,
            "business_approval_required": plan.get("status") == "pending_approval",
        },
        "final_sign_off": {
            "status": "PASS" if technical_pass else "FAIL",
            "hard_constraints_pass": hard_constraint_pass,
            "coverage_pass": coverage_pass,
            "release_authorised": plan.get("status") == "active",
            "failure_reasons": []
            if technical_pass
            else ["Independent hard-constraint or coverage validation failed."],
            "sign_off_reason": (
                "Technical validation passed; business approval is required "
                "before activation."
                if technical_pass and plan.get("status") == "pending_approval"
                else "All technical and governance release checks passed."
                if technical_pass
                else "The loading plan must not be released."
            ),
        },
    }


def audit_report_to_markdown(report: Dict[str, Any]) -> str:
    metadata = report["run_metadata"]
    inputs = report["input_audit"]
    strategy = report["strategy_audit"]
    constraints = report["hard_constraint_audit"]
    governance = report["governance"]
    sign_off = report["final_sign_off"]
    return "\n".join(
        [
            "# Loading Optimisation Audit Report",
            "",
            f"Technical status: **{sign_off['status']}**",
            f"Plan status: **{governance.get('plan_status')}**",
            f"Run id: `{metadata.get('run_id')}`",
            f"Generated at: `{metadata.get('generated_at')}`",
            "",
            "## Input",
            f"- Products: {inputs.get('product_count')}",
            f"- Destinations: {inputs.get('destination_count')}",
            "",
            "## Strategy",
            f"- Requested: `{strategy.get('requested_strategy')}`",
            f"- Executed: `{strategy.get('applied_strategy')}`",
            "",
            "## Independent validation",
            f"- Valid: {constraints.get('is_valid')}",
            f"- Violations: {constraints.get('violation_count', 0)}",
            "",
            "## Release governance",
            f"- Business approval required: {governance.get('business_approval_required')}",
            f"- Release authorised: {sign_off.get('release_authorised')}",
            f"- Reason: {sign_off.get('sign_off_reason')}",
            "",
        ]
    )


def write_audit_report_files(
    run_dir: Path, audit_report: Dict[str, Any]
) -> Tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "audit_report.json"
    markdown_path = run_dir / "audit_report.md"
    PlanningOrchestrator.write_json(json_path, audit_report)
    temporary = markdown_path.with_suffix(".md.tmp")
    temporary.write_text(audit_report_to_markdown(audit_report), encoding="utf-8")
    temporary.replace(markdown_path)
    return json_path, markdown_path
