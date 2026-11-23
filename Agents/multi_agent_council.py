"""
Multi-agent optimisation council.

Several strategy-specialised candidate agents generate loading plans. Role
specialised evaluator agents score the candidates, and a coordinator selects
the best safety-valid option.
"""

from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

from Agents.core.research_contracts import ErrorCode, StructuredError
from Agents.validation_agent import ValidationAgent

RunCandidate = Callable[
    [Path, Dict[str, Any], Path],
    tuple[Dict[str, Any], Path],
]


@dataclass
class CandidateAgentSpec:
    agent_name: str
    selected_strategy: str
    role: str
    optimisation_config: Dict[str, Any]
    budget_weight: float


class MultiAgentCouncil:
    """
    Coordinates candidate generation, evaluator scoring, and final selection.

    The council is intentionally deterministic and validator-gated: any
    candidate that fails hard constraints receives a SafetyAgent veto and cannot
    be selected by the coordinator.
    """

    COORDINATION_RESERVE_RATIO = 0.10

    def __init__(
        self,
        time_budget_seconds: float | None = None,
        safety_veto_enabled: bool = True,
    ) -> None:
        self.time_budget_seconds = time_budget_seconds
        self.safety_veto_enabled = safety_veto_enabled
        self.validation_agent = ValidationAgent()

    def candidate_specs(self) -> List[CandidateAgentSpec]:
        budget = self.time_budget_seconds
        specs = [
            CandidateAgentSpec(
                agent_name="FastGreedyAgent",
                selected_strategy="balance_aware_greedy",
                role="Prioritises fast feasible loading with lightweight scoring.",
                optimisation_config={
                    "group_by_destination": False,
                    "sort_by_weight_desc": True,
                    "use_balance_aware_scoring": True,
                    "use_simulated_annealing": False,
                    "use_rule_based_constraints": True,
                    "time_budget_seconds": min(budget or 2.0, 2.0),
                    "allow_policy_strategy_override": False,
                },
                budget_weight=0.10,
            ),
            CandidateAgentSpec(
                agent_name="DestinationGroupingAgent",
                selected_strategy="destination_grouping_greedy",
                role="Prioritises unloading clarity by grouping destinations.",
                optimisation_config={
                    "group_by_destination": True,
                    "sort_by_weight_desc": True,
                    "use_balance_aware_scoring": False,
                    "use_simulated_annealing": False,
                    "use_rule_based_constraints": True,
                    "time_budget_seconds": budget,
                    "allow_policy_strategy_override": False,
                },
                budget_weight=0.20,
            ),
            CandidateAgentSpec(
                agent_name="BalanceAwareAgent",
                selected_strategy="destination_grouping_plus_balance_aware_greedy",
                role="Balances destination grouping, utilisation, and weight distribution.",
                optimisation_config={
                    "group_by_destination": True,
                    "sort_by_weight_desc": True,
                    "use_balance_aware_scoring": True,
                    "use_simulated_annealing": False,
                    "use_rule_based_constraints": True,
                    "time_budget_seconds": budget,
                    "allow_policy_strategy_override": False,
                },
                budget_weight=0.20,
            ),
        ]
        if budget is not None and budget >= 6.0:
            specs.append(
                CandidateAgentSpec(
                    agent_name="AnnealingAgent",
                    selected_strategy="greedy_plus_simulated_annealing",
                    role="Explores a higher-effort candidate under the time budget.",
                    optimisation_config={
                        "group_by_destination": True,
                        "sort_by_weight_desc": True,
                        "use_balance_aware_scoring": True,
                        "use_simulated_annealing": True,
                        "use_rule_based_constraints": True,
                        "time_budget_seconds": min(budget, 8.0),
                        "allow_policy_strategy_override": False,
                    },
                    budget_weight=0.50,
                )
            )
        return specs

    def run(
        self,
        products_path: Path,
        base_strategy_plan: Dict[str, Any],
        run_dir: Path,
        run_candidate: RunCandidate,
    ) -> Dict[str, Any]:
        # Keep physical paths deliberately short for Windows deployments.  Full
        # agent names remain in the manifest and audit evidence; directory names
        # are internal implementation details only.
        council_dir = run_dir / "council"
        council_dir.mkdir(parents=True, exist_ok=True)

        candidates = []
        council_started = time.perf_counter()
        council_deadline = (
            council_started + self.time_budget_seconds
            if self.time_budget_seconds is not None
            else None
        )
        coordination_reserve_seconds = (
            self.time_budget_seconds * self.COORDINATION_RESERVE_RATIO
            if self.time_budget_seconds is not None
            else None
        )
        candidate_deadline = (
            council_deadline - coordination_reserve_seconds
            if council_deadline is not None
            and coordination_reserve_seconds is not None
            else None
        )
        specs = self.candidate_specs()
        for index, spec in enumerate(specs, start=1):
            remaining_seconds = (
                candidate_deadline - time.perf_counter()
                if candidate_deadline is not None
                else None
            )
            if remaining_seconds is not None and remaining_seconds <= 0:
                candidates.append(
                    {
                        "agent_name": spec.agent_name,
                        "role": spec.role,
                        "selected_strategy": spec.selected_strategy,
                        "status": "skipped_insufficient_budget",
                        "runtime_seconds": 0.0,
                        "strategy_plan": self.build_candidate_strategy_plan(
                            base_strategy_plan,
                            spec,
                        ),
                        "solution": None,
                        "solution_path": None,
                        "error": StructuredError(
                            code=ErrorCode.COUNCIL_DEADLINE_EXCEEDED,
                            message=(
                                "Council candidate budget expired before candidate "
                                "start; coordination reserve was preserved."
                            ),
                            retryable=True,
                        ).to_dict(),
                    }
                )
                continue
            candidate_dir = council_dir / f"c{index:02d}"
            strategy_plan = self.build_candidate_strategy_plan(
                base_strategy_plan,
                spec,
            )
            if remaining_seconds is not None:
                remaining_weight = sum(
                    item.budget_weight for item in specs[index - 1:]
                )
                fair_budget = (
                    remaining_seconds * spec.budget_weight / remaining_weight
                )
                strategy_plan["optimisation_config"]["time_budget_seconds"] = (
                    max(0.0, min(fair_budget, remaining_seconds))
                )
            allocated_budget = strategy_plan["optimisation_config"].get(
                "time_budget_seconds"
            )
            started = time.perf_counter()
            try:
                solution, solution_path = run_candidate(
                    products_path,
                    strategy_plan,
                    candidate_dir,
                )
                runtime_seconds = time.perf_counter() - started
                candidates.append(
                    {
                        "agent_name": spec.agent_name,
                        "role": spec.role,
                        "selected_strategy": spec.selected_strategy,
                        "status": "completed",
                        "runtime_seconds": round(runtime_seconds, 4),
                        "allocated_budget_seconds": allocated_budget,
                        "strategy_plan": strategy_plan,
                        "solution": solution,
                        "solution_path": str(solution_path),
                        "error": None,
                    }
                )
            except Exception as error:
                runtime_seconds = time.perf_counter() - started
                candidates.append(
                    {
                        "agent_name": spec.agent_name,
                        "role": spec.role,
                        "selected_strategy": spec.selected_strategy,
                        "status": "failed",
                        "runtime_seconds": round(runtime_seconds, 4),
                        "allocated_budget_seconds": allocated_budget,
                        "strategy_plan": strategy_plan,
                        "solution": None,
                        "solution_path": None,
                        "error": str(error),
                    }
                )

        expected_material_numbers = self.load_expected_material_numbers(
            products_path
        )
        evaluated = [
            self.evaluate_candidate(candidate, expected_material_numbers)
            for candidate in candidates
        ]
        selected = self.select_candidate(evaluated)
        selected_candidate = next(
            item
            for item in evaluated
            if item["agent_name"] == selected["selected_candidate"]
        )
        diversity_summary = self.build_diversity_summary(evaluated)

        elapsed_seconds = time.perf_counter() - council_started
        overrun_seconds = (
            max(0.0, elapsed_seconds - self.time_budget_seconds)
            if self.time_budget_seconds is not None
            else 0.0
        )
        deadline_met = (
            elapsed_seconds <= self.time_budget_seconds
            if self.time_budget_seconds is not None
            else None
        )
        result = {
            "enabled": True,
            "version": "2.0_fair_budget_candidate_competition",
            "deadline": {
                "deadline_type": "cooperative",
                "time_budget_seconds": self.time_budget_seconds,
                "candidate_budget_seconds": (
                    self.time_budget_seconds - coordination_reserve_seconds
                    if self.time_budget_seconds is not None
                    and coordination_reserve_seconds is not None
                    else None
                ),
                "coordination_reserve_ratio": self.COORDINATION_RESERVE_RATIO,
                "coordination_reserve_seconds": coordination_reserve_seconds,
                "elapsed_seconds": round(elapsed_seconds, 4),
                "overrun_seconds": round(overrun_seconds, 4),
                "overrun_ratio": round(
                    overrun_seconds / self.time_budget_seconds,
                    4,
                ) if self.time_budget_seconds else 0.0,
                "deadline_met": deadline_met,
                "deadline_exhausted": (
                    council_deadline is not None
                    and elapsed_seconds >= self.time_budget_seconds
                ),
                "anytime_solution_returned": True,
            },
            "candidate_agents": [
                {
                    "agent_name": item["agent_name"],
                    "role": item["role"],
                    "selected_strategy": item["selected_strategy"],
                    "status": item["status"],
                }
                for item in evaluated
            ],
            "evaluation_matrix": [
                self.public_candidate_row(item) for item in evaluated
            ],
            "diversity_summary": diversity_summary,
            "coordinator_decision": selected,
            "selected_solution": selected_candidate["solution"],
            "selected_solution_path": selected_candidate["solution_path"],
            "selected_strategy_plan": selected_candidate["strategy_plan"],
        }
        result["artifacts"] = self.write_council_artifacts(
            council_dir=council_dir,
            result=result,
        )
        return result

    def build_candidate_strategy_plan(
        self,
        base_strategy_plan: Dict[str, Any],
        spec: CandidateAgentSpec,
    ) -> Dict[str, Any]:
        strategy_plan = copy.deepcopy(base_strategy_plan)
        operational_requirements = base_strategy_plan.get(
            "operational_requirements", {}
        )
        destination_grouping_required = bool(
            isinstance(operational_requirements, dict)
            and operational_requirements.get("destination_grouping_required")
        )
        strategy_plan["scenario_type"] = "multi_agent_council_candidate"
        selected_strategy = spec.selected_strategy
        if destination_grouping_required and selected_strategy == "balance_aware_greedy":
            selected_strategy = "destination_grouping_plus_balance_aware_greedy"
        strategy_plan["selected_strategy"] = selected_strategy
        strategy_plan["strategy_reason"] = (
            f"{spec.agent_name} candidate: {spec.role}"
        )
        strategy_plan["optimisation_config"] = {
            key: value
            for key, value in spec.optimisation_config.items()
            if value is not None
        }
        if destination_grouping_required:
            strategy_plan["optimisation_config"]["group_by_destination"] = True
            strategy_plan["strategy_reason"] += (
                " Operator-confirmed destination grouping was enforced."
            )
        base_config = base_strategy_plan.get("optimisation_config", {})
        if isinstance(base_config, dict) and "disabled_wagon_ids" in base_config:
            strategy_plan["optimisation_config"]["disabled_wagon_ids"] = list(
                base_config.get("disabled_wagon_ids") or []
            )
        candidate_agent = asdict(spec)
        candidate_agent["selected_strategy"] = selected_strategy
        if destination_grouping_required:
            candidate_agent["optimisation_config"]["group_by_destination"] = True
        strategy_plan["candidate_agent"] = candidate_agent
        return strategy_plan

    def evaluate_candidate(
        self,
        candidate: Dict[str, Any],
        expected_material_numbers: List[str] | None = None,
    ) -> Dict[str, Any]:
        solution = candidate.get("solution") or {}
        validation = self.validation_agent.validate(
            solution,
            expected_material_numbers=expected_material_numbers,
            unavailable_wagon_ids=(
                candidate.get("strategy_plan", {})
                .get("optimisation_config", {})
                .get("disabled_wagon_ids", [])
            ),
        )
        score = solution.get("final_solution_score", {})
        summary = solution.get("wagon_summary", {})

        safety_pass = bool(validation.get("is_valid"))
        safety_score = 1.0 if safety_pass else 0.0
        wagon_count = summary.get("wagon_count") or score.get("wagon_count")
        utilisation = float(summary.get("average_area_utilisation") or 0.0)
        runtime = float(candidate.get("runtime_seconds") or 0.0)
        allocated_budget = candidate.get("allocated_budget_seconds")

        efficiency_score = self.efficiency_score(wagon_count, utilisation)
        balance_score = self.balance_score(score)
        operations_score = self.operations_score(solution)
        time_score = self.time_score(runtime, allocated_budget)
        final_score = 0.0
        execution_failed = candidate.get("status") != "completed"
        veto = execution_failed or (
            self.safety_veto_enabled and not safety_pass
        )
        if not veto:
            final_score = round(
                0.30 * efficiency_score
                + 0.25 * balance_score
                + 0.20 * operations_score
                + 0.15 * time_score
                + 0.10 * safety_score,
                4,
            )

        candidate = dict(candidate)
        candidate["evaluation"] = {
            "SafetyAgent": {
                "enabled": self.safety_veto_enabled,
                "pass": safety_pass,
                "score": safety_score,
                "veto": veto,
                "violation_count": validation.get("violation_count"),
                "veto_reasons": validation.get("veto_reasons", []),
                "validator": validation.get("validator", {}),
            },
            "EfficiencyAgent": {
                "score": efficiency_score,
                "wagon_count": wagon_count,
                "average_area_utilisation": utilisation,
            },
            "BalanceAgent": {
                "score": balance_score,
                "front_rear_balance_penalty": score.get(
                    "front_rear_balance_penalty"
                ),
                "left_right_balance_penalty": score.get(
                    "left_right_balance_penalty"
                ),
                "diagonal_balance_penalty": score.get(
                    "diagonal_balance_penalty"
                ),
            },
            "OperationsAgent": {
                "score": operations_score,
                "destination_count": len(
                    solution.get("input_summary", {}).get("destinations", [])
                ),
            },
            "TimeAgent": {
                "score": time_score,
                "runtime_seconds": runtime,
                "allocated_budget_seconds": allocated_budget,
            },
            "CoordinatorAgent": {
                "final_score": final_score,
                "vetoed": veto,
            },
        }
        return candidate

    def select_candidate(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_candidates = [
            candidate
            for candidate in candidates
            if not candidate["evaluation"]["CoordinatorAgent"]["vetoed"]
        ]
        if not valid_candidates:
            raise RuntimeError(
                StructuredError(
                    code=ErrorCode.ALL_CANDIDATES_VETOED,
                    message="All multi-agent council candidates were vetoed.",
                    requires_manual_review=True,
                    details={
                        "candidates": [
                            {
                                "agent_name": item.get("agent_name"),
                                "status": item.get("status"),
                                "error": item.get("error"),
                                "safety_veto_reasons": item.get("evaluation", {})
                                .get("SafetyAgent", {})
                                .get("veto_reasons", []),
                            }
                            for item in candidates
                        ]
                    },
                ).to_dict()
            )

        ranked = sorted(
            valid_candidates,
            key=self.coordinator_rank_key,
            reverse=True,
        )
        selected = ranked[0]
        runner_up = None
        if len(valid_candidates) > 1:
            runner_up = ranked[1]["agent_name"]

        return {
            "selected_candidate": selected["agent_name"],
            "selected_strategy": selected["selected_strategy"],
            "runner_up": runner_up,
            "reason": (
                f"{selected['agent_name']} passed SafetyAgent veto and achieved "
                "the best weighted evaluator score across efficiency, balance, "
                "operations, and time."
            ),
            "tie_break_policy": [
                "higher weighted evaluator score",
                "lower objective score",
                "lower runtime",
                "stable candidate name",
            ],
            "vetoed_candidates": [
                item["agent_name"]
                for item in candidates
                if item["evaluation"]["CoordinatorAgent"]["vetoed"]
            ],
        }

    def public_candidate_row(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        evaluation = candidate["evaluation"]
        return {
            "candidate": candidate["agent_name"],
            "strategy": candidate["selected_strategy"],
            "applied_strategy": (
                (candidate.get("solution") or {})
                .get("strategy_execution", {})
                .get("applied_strategy")
            ),
            "status": candidate["status"],
            "safety_pass": evaluation["SafetyAgent"]["pass"],
            "safety_veto": evaluation["SafetyAgent"]["veto"],
            "efficiency_score": evaluation["EfficiencyAgent"]["score"],
            "balance_score": evaluation["BalanceAgent"]["score"],
            "operations_score": evaluation["OperationsAgent"]["score"],
            "time_score": evaluation["TimeAgent"]["score"],
            "final_score": evaluation["CoordinatorAgent"]["final_score"],
            "wagon_count": evaluation["EfficiencyAgent"]["wagon_count"],
            "utilisation": round(
                evaluation["EfficiencyAgent"]["average_area_utilisation"],
                4,
            ),
            "runtime_seconds": evaluation["TimeAgent"]["runtime_seconds"],
            "allocated_budget_seconds": evaluation["TimeAgent"][
                "allocated_budget_seconds"
            ],
            "safety_veto_reasons": evaluation["SafetyAgent"]["veto_reasons"],
        }

    @staticmethod
    def load_expected_material_numbers(products_path: Path) -> List[str]:
        with products_path.open("r", encoding="utf-8") as file:
            products = json.load(file)
        if not isinstance(products, list):
            raise ValueError("Council products input must contain a JSON list.")
        material_numbers = []
        for product in products:
            if not isinstance(product, dict):
                continue
            value = (
                product.get("material_no")
                or product.get("product_id")
                or product.get("id")
            )
            if value not in (None, ""):
                material_numbers.append(str(value))
        return material_numbers

    def efficiency_score(
        self,
        wagon_count: int | float | None,
        utilisation: float,
    ) -> float:
        if not wagon_count:
            return 0.0
        wagon_component = max(0.0, min(1.0, 20.0 / float(wagon_count) / 2.0))
        utilisation_component = max(0.0, min(1.0, utilisation * 4.0))
        return round(0.55 * wagon_component + 0.45 * utilisation_component, 4)

    def balance_score(self, score: Dict[str, Any]) -> float:
        # OptimisationAgent reports fleet-level sums for the three balance
        # penalties.  Council scores are defined on a 0..1 scale, so compare
        # candidates by their mean per-wagon penalty.  Without this
        # normalisation, any realistic multi-wagon run saturates every
        # candidate at zero and the coordinator effectively ignores balance.
        wagon_count = max(1.0, float(score.get("wagon_count") or 1.0))
        penalties = [
            float(score.get("front_rear_balance_penalty") or 0.0) / wagon_count,
            float(score.get("left_right_balance_penalty") or 0.0) / wagon_count,
            float(score.get("diagonal_balance_penalty") or 0.0) / wagon_count,
        ]
        penalty = sum(penalties) / max(1, len(penalties))
        return round(max(0.0, min(1.0, 1.0 - penalty)), 4)

    def operations_score(self, solution: Dict[str, Any]) -> float:
        wagons = solution.get("wagons", [])
        if not wagons:
            return 0.0
        mixed_wagons = 0
        for wagon in wagons:
            destinations = {
                str(product.get("destination", "")).strip()
                for product in wagon.get("products", [])
                if str(product.get("destination", "")).strip()
            }
            if len(destinations) > 1:
                mixed_wagons += 1
        return round(max(0.0, 1.0 - mixed_wagons / len(wagons)), 4)

    def time_score(
        self,
        runtime_seconds: float,
        allocated_budget_seconds: float | None = None,
    ) -> float:
        budget = allocated_budget_seconds or self.time_budget_seconds or 8.0
        if runtime_seconds <= 0:
            return 1.0
        return round(max(0.0, min(1.0, 1.0 - runtime_seconds / budget)), 4)

    @staticmethod
    def coordinator_rank_key(candidate: Dict[str, Any]) -> tuple:
        evaluation = candidate["evaluation"]
        final_score = float(evaluation["CoordinatorAgent"]["final_score"])
        objective = float(
            (candidate.get("solution") or {})
            .get("final_solution_score", {})
            .get("objective_score")
            or float("inf")
        )
        runtime = float(candidate.get("runtime_seconds") or float("inf"))
        return (final_score, -objective, -runtime, candidate["agent_name"])

    @staticmethod
    def build_diversity_summary(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        completed = [item for item in candidates if item.get("status") == "completed"]
        applied_strategies = [
            (item.get("solution") or {})
            .get("strategy_execution", {})
            .get("applied_strategy")
            for item in completed
        ]
        fingerprints = []
        for item in completed:
            solution = item.get("solution") or {}
            score = solution.get("final_solution_score", {})
            fingerprints.append(
                (
                    solution.get("wagon_summary", {}).get("wagon_count"),
                    round(float(score.get("objective_score") or 0.0), 6),
                )
            )
        return {
            "completed_candidate_count": len(completed),
            "distinct_applied_strategy_count": len(set(applied_strategies)),
            "distinct_outcome_count": len(set(fingerprints)),
            "duplicate_outcome_count": len(fingerprints) - len(set(fingerprints)),
            "applied_strategies": applied_strategies,
        }

    @staticmethod
    def write_council_artifacts(
        council_dir: Path,
        result: Dict[str, Any],
    ) -> Dict[str, str]:
        decision_path = council_dir / "council_decision.json"
        matrix_json_path = council_dir / "council_evaluation_matrix.json"
        matrix_csv_path = council_dir / "council_evaluation_matrix.csv"
        public_decision = {
            key: value
            for key, value in result.items()
            if key not in {
                "selected_solution",
                "selected_strategy_plan",
                "artifacts",
            }
        }
        decision_path.write_text(
            json.dumps(public_decision, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        matrix = result["evaluation_matrix"]
        matrix_json_path.write_text(
            json.dumps(matrix, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if matrix:
            fieldnames = list(matrix[0].keys())
            with matrix_csv_path.open(
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for row in matrix:
                    serialised = dict(row)
                    serialised["safety_veto_reasons"] = json.dumps(
                        row.get("safety_veto_reasons", []),
                        ensure_ascii=False,
                    )
                    writer.writerow(serialised)
        return {
            "decision_json": str(decision_path),
            "evaluation_matrix_json": str(matrix_json_path),
            "evaluation_matrix_csv": str(matrix_csv_path),
        }
