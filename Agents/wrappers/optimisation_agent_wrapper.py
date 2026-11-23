from pathlib import Path
from typing import Any, Dict, Optional

from Agents.core.base_agent import BaseAgent
from Agents.core.task import AgentTask
from Agents.optimisation_agent import OptimisationAgent


class OptimisationAgentWrapper(BaseAgent):
    """
    Managed wrapper for Agents/optimisation_agent.py.

    Current role:
    products.json + strategy_plan.json -> loading_solution.json

    In addition to normal optimisation outputs, this wrapper returns
    the strategy execution audit trail:

    - requested_strategy: strategy proposed by the LLM
    - applied_strategy: strategy actually executed by OptimisationAgent
    - actual_batch_scenario: scenario derived from real batch data
    - strategy_resolution_notes: normalisation / safety notes
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="optimisation",
            name="Optimisation Agent Wrapper",
            config=config,
        )

    def act(self, task: AgentTask) -> Dict[str, Any]:
        self.update_status("running")
        self.set_current_task(task.task_id)

        try:
            products_json_path = self._resolve_products_json_path(task)
            strategy_plan_path = self._resolve_strategy_plan_path(task)
            output_path = self._resolve_output_path(task)

            car_length = self._resolve_float(
                task,
                "car_length",
                13000.0,
            )
            car_width = self._resolve_float(
                task,
                "car_width",
                3000.0,
            )
            max_weight_tonne = self._resolve_float(
                task,
                "max_weight_tonne",
                60.0,
            )
            parallel_gap = self._resolve_float(
                task,
                "parallel_gap",
                50.0,
            )
            bogie_tol_tons = self._resolve_float(
                task,
                "bogie_tol_tons",
                2.0,
            )
            max_weight_diff_for_pair = self._resolve_float(
                task,
                "max_weight_diff_for_pair",
                1.0,
            )
            max_outer_diameter_diff_for_pair = self._resolve_float(
                task,
                "max_outer_diameter_diff_for_pair",
                100.0,
            )
            force_same_outer_diameter_for_pair = self._resolve_bool(
                task,
                "force_same_outer_diameter_for_pair",
                True,
            )
            enable_low_utilisation_merge = self._resolve_bool(
                task,
                "enable_low_utilisation_merge",
                True,
            )
            low_utilisation_threshold = self._resolve_float(
                task,
                "low_utilisation_threshold",
                0.35,
            )
            enable_wagon_reassignment = self._resolve_bool(
                task,
                "enable_wagon_reassignment",
                True,
            )
            max_wagon_reassignment_passes = self._resolve_int(
                task,
                "max_wagon_reassignment_passes",
                2,
            )
            seed = self._resolve_int(task, "seed", 42)

            self.update_belief("task_id", task.task_id)
            self.update_belief(
                "products_json_path",
                str(products_json_path),
            )
            self.update_belief(
                "strategy_plan_path",
                str(strategy_plan_path),
            )
            self.update_belief(
                "loading_solution_output_path",
                str(output_path),
            )
            self.update_belief("objective", task.objective)
            self.update_belief("constraints", task.constraints)
            self.update_belief("car_length", car_length)
            self.update_belief("car_width", car_width)
            self.update_belief("max_weight_tonne", max_weight_tonne)
            self.update_belief("parallel_gap", parallel_gap)
            self.update_belief("bogie_tol_tons", bogie_tol_tons)
            self.update_belief(
                "max_weight_diff_for_pair",
                max_weight_diff_for_pair,
            )
            self.update_belief(
                "max_outer_diameter_diff_for_pair",
                max_outer_diameter_diff_for_pair,
            )
            self.update_belief(
                "force_same_outer_diameter_for_pair",
                force_same_outer_diameter_for_pair,
            )
            self.update_belief(
                "enable_low_utilisation_merge",
                enable_low_utilisation_merge,
            )
            self.update_belief(
                "low_utilisation_threshold",
                low_utilisation_threshold,
            )
            self.update_belief(
                "enable_wagon_reassignment",
                enable_wagon_reassignment,
            )
            self.update_belief(
                "max_wagon_reassignment_passes",
                max_wagon_reassignment_passes,
            )
            self.update_belief("seed", seed)

            self.add_desire("generate_feasible_loading_solution")
            self.add_intention(
                "call_strategy_driven_OptimisationAgent_run"
            )

            optimisation_agent = OptimisationAgent(
                products_json_path=products_json_path,
                strategy_plan_path=strategy_plan_path,
                output_path=output_path,
                car_length=car_length,
                car_width=car_width,
                max_weight_tonne=max_weight_tonne,
                parallel_gap=parallel_gap,
                bogie_tol_tons=bogie_tol_tons,
                max_weight_diff_for_pair=max_weight_diff_for_pair,
                max_outer_diameter_diff_for_pair=(
                    max_outer_diameter_diff_for_pair
                ),
                force_same_outer_diameter_for_pair=(
                    force_same_outer_diameter_for_pair
                ),
                enable_low_utilisation_merge=(
                    enable_low_utilisation_merge
                ),
                low_utilisation_threshold=(
                    low_utilisation_threshold
                ),
                enable_wagon_reassignment=(
                    enable_wagon_reassignment
                ),
                max_wagon_reassignment_passes=(
                    max_wagon_reassignment_passes
                ),
                seed=seed,
            )

            loading_solution = optimisation_agent.run()

            wagon_summary = loading_solution.get("wagon_summary", {})
            final_score = loading_solution.get(
                "final_solution_score",
                {},
            )
            validation_summary = loading_solution.get(
                "validation_summary",
                {},
            )
            optimisation_trace = loading_solution.get(
                "optimisation_trace",
                [],
            )
            anytime_summary = loading_solution.get(
                "anytime_summary",
                {},
            )
            performance_summary = loading_solution.get(
                "performance_summary",
                {},
            )
            algorithm_decision_trace = loading_solution.get(
                "algorithm_decision_trace",
                [],
            )
            strategy_execution = loading_solution.get(
                "strategy_execution",
                {},
            )
            simulated_annealing_log = loading_solution.get(
                "simulated_annealing_log",
                {},
            )
            algorithm_policy = strategy_execution.get(
                "algorithm_policy",
                {},
            )
            solution_metadata = loading_solution.get(
                "solution_metadata",
                {},
            )

            requested_strategy = strategy_execution.get(
                "requested_strategy"
            )
            applied_strategy = strategy_execution.get(
                "applied_strategy"
            )
            actual_batch_scenario = strategy_execution.get(
                "actual_batch_scenario"
            )

            print(
                f"[{self.agent_id}] Strategy execution | "
                f"requested={requested_strategy} | "
                f"applied={applied_strategy} | "
                f"scenario={actual_batch_scenario}"
            )

            result = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "status": "completed",
                "message": (
                    "OptimisationAgentWrapper completed successfully."
                ),
                "products_json_path": str(products_json_path),
                "strategy_plan_path": str(strategy_plan_path),
                "loading_solution_output_path": str(output_path),

                # Actual strategy execution audit trail
                "requested_strategy": requested_strategy,
                "requested_scenario_type": strategy_execution.get(
                    "requested_scenario_type"
                ),
                "applied_strategy": applied_strategy,
                "actual_batch_scenario": actual_batch_scenario,
                "strategy_execution_config": strategy_execution.get(
                    "execution_config",
                    {},
                ),
                "strategy_resolution_notes": strategy_execution.get(
                    "resolution_notes",
                    [],
                ),
                "strategy_fallback_reasons": strategy_execution.get(
                    "fallback_reasons",
                    [],
                ),
                "algorithm_policy": algorithm_policy,

                # Optimisation outcome
                "wagon_count": wagon_summary.get("wagon_count"),
                "total_products": wagon_summary.get("total_products"),
                "total_weight": wagon_summary.get("total_weight"),
                "average_area_utilisation": wagon_summary.get(
                    "average_area_utilisation"
                ),
                "objective_score": final_score.get("objective_score"),
                "validation_summary": validation_summary,
                "optimisation_trace": optimisation_trace,
                "anytime_summary": anytime_summary,
                "performance_summary": performance_summary,
                "algorithm_decision_trace": algorithm_decision_trace,
                "balance_penalties": {
                    "front_rear_balance_penalty": final_score.get(
                        "front_rear_balance_penalty"
                    ),
                    "left_right_balance_penalty": final_score.get(
                        "left_right_balance_penalty"
                    ),
                    "diagonal_balance_penalty": final_score.get(
                        "diagonal_balance_penalty"
                    ),
                },

                # Runtime / strategy-execution metadata
                "optimisation_runtime_seconds": solution_metadata.get(
                    "runtime_seconds"
                ),
                "simulated_annealing": {
                    "enabled": simulated_annealing_log.get("enabled"),
                    "executed": simulated_annealing_log.get("executed"),
                    "iterations_requested": (
                        simulated_annealing_log.get(
                            "iterations_requested"
                        )
                    ),
                    "iterations_completed": (
                        simulated_annealing_log.get(
                            "iterations_completed"
                        )
                    ),
                    "accepted_moves": simulated_annealing_log.get(
                        "accepted_moves"
                    ),
                },
            }

            self.update_belief(
                "requested_strategy",
                requested_strategy,
            )
            self.update_belief(
                "applied_strategy",
                applied_strategy,
            )
            self.update_belief(
                "actual_batch_scenario",
                actual_batch_scenario,
            )
            self.update_belief(
                "strategy_resolution_notes",
                result["strategy_resolution_notes"],
            )
            self.update_belief(
                "wagon_count",
                wagon_summary.get("wagon_count"),
            )
            self.update_belief(
                "total_products",
                wagon_summary.get("total_products"),
            )
            self.update_belief(
                "objective_score",
                final_score.get("objective_score"),
            )

            self.update_status("completed")
            self.clear_current_task()

            return result

        except Exception as error:
            self.update_belief("error", str(error))
            self.update_status("failed")
            self.clear_current_task()

            return {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "status": "failed",
                "message": "OptimisationAgentWrapper failed.",
                "error": str(error),
            }

    def _resolve_products_json_path(self, task: AgentTask) -> Path:
        """
        Priority:
        1. task.metadata["products_json_path"]
        2. task.metadata["json_output_path"]
        3. self.config["products_json_path"]
        4. Agents/products.json
        """

        if task.metadata.get("products_json_path"):
            return Path(task.metadata["products_json_path"])

        if task.metadata.get("json_output_path"):
            return Path(task.metadata["json_output_path"])

        if self.config.get("products_json_path"):
            return Path(self.config["products_json_path"])

        return Path("Agents/products.json")

    def _resolve_strategy_plan_path(self, task: AgentTask) -> Path:
        """
        Priority:
        1. task.metadata["strategy_plan_output_path"]
        2. task.metadata["strategy_plan_path"]
        3. self.config["strategy_plan_path"]
        4. Agents/strategy_plan.json
        """

        if task.metadata.get("strategy_plan_output_path"):
            return Path(task.metadata["strategy_plan_output_path"])

        if task.metadata.get("strategy_plan_path"):
            return Path(task.metadata["strategy_plan_path"])

        if self.config.get("strategy_plan_path"):
            return Path(self.config["strategy_plan_path"])

        return Path("Agents/strategy_plan.json")

    def _resolve_output_path(self, task: AgentTask) -> Path:
        """
        Priority:
        1. task.metadata["loading_solution_output_path"]
        2. task.metadata["loading_solution_path"]
        3. self.config["loading_solution_output_path"]
        4. Agents/loading_solution.json
        """

        if task.metadata.get("loading_solution_output_path"):
            return Path(
                task.metadata["loading_solution_output_path"]
            )

        if task.metadata.get("loading_solution_path"):
            return Path(task.metadata["loading_solution_path"])

        if self.config.get("loading_solution_output_path"):
            return Path(
                self.config["loading_solution_output_path"]
            )

        return Path("Agents/loading_solution.json")

    def _resolve_float(
        self,
        task: AgentTask,
        key: str,
        default: float,
    ) -> float:
        if key in task.metadata:
            return float(task.metadata[key])

        if key in self.config:
            return float(self.config[key])

        return default

    def _resolve_int(
        self,
        task: AgentTask,
        key: str,
        default: int,
    ) -> int:
        if key in task.metadata:
            return int(task.metadata[key])

        if key in self.config:
            return int(self.config[key])

        return default

    def _resolve_bool(
        self,
        task: AgentTask,
        key: str,
        default: bool,
    ) -> bool:
        if key in task.metadata:
            return self._parse_bool(task.metadata[key], default)

        if key in self.config:
            return self._parse_bool(self.config[key], default)

        return default

    @staticmethod
    def _parse_bool(value: Any, default: bool) -> bool:
        """
        Parse bool safely.

        bool("false") is True in Python, so strings must be handled
        explicitly for future JSON/web configuration inputs.
        """

        if isinstance(value, bool):
            return value

        if value is None:
            return default

        if isinstance(value, (int, float)):
            return bool(value)

        text = str(value).strip().lower()

        if text in {"true", "1", "yes", "y", "on"}:
            return True

        if text in {"false", "0", "no", "n", "off", ""}:
            return False

        return default
