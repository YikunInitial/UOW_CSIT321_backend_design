from pathlib import Path
from typing import Any, Dict, Optional

from Agents.core.base_agent import BaseAgent
from Agents.core.task import AgentTask
from Agents.llm_agent import LLMAgent


class LLMAgentWrapper(BaseAgent):
    """
    Managed wrapper for the existing Agents/llm_agent.py.

    This wrapper does not replace the original LLMAgent.
    It only adds:
    - agent_id
    - status tracking
    - task tracking
    - BDI fields
    - standard act(task) interface

    Current role:
    products.json + retrieved_context.json -> strategy_plan.json
    """

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=agent_id,
            agent_type="llm",
            name="LLM Agent Wrapper",
            config=config,
        )

    def act(self, task: AgentTask) -> Dict[str, Any]:
        self.update_status("running")
        self.set_current_task(task.task_id)

        try:
            products_json_path = self._resolve_products_json_path(task)
            retrieved_context_path = self._resolve_retrieved_context_path(task)
            output_path = self._resolve_output_path(task)

            model_name = self._resolve_model_name(task)
            temperature = self._resolve_temperature(task)
            num_ctx = self._resolve_num_ctx(task)

            self.update_belief("task_id", task.task_id)
            self.update_belief("products_json_path", str(products_json_path))
            self.update_belief("retrieved_context_path", str(retrieved_context_path))
            self.update_belief("strategy_plan_output_path", str(output_path))
            self.update_belief("model_name", model_name)
            self.update_belief("temperature", temperature)
            self.update_belief("num_ctx", num_ctx)
            self.update_belief("objective", task.objective)
            self.update_belief("constraints", task.constraints)

            self.add_desire("generate_strategy_plan_for_optimisation")
            self.add_intention("call_existing_LLMAgent_run")

            llm_agent = LLMAgent(
                products_json_path=products_json_path,
                retrieved_context_path=retrieved_context_path,
                output_path=output_path,
                model_name=model_name,
                temperature=temperature,
                num_ctx=num_ctx,
            )

            strategy_plan = llm_agent.run()

            result = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "status": "completed",
                "message": "LLMAgentWrapper completed successfully.",
                "products_json_path": str(products_json_path),
                "retrieved_context_path": str(retrieved_context_path),
                "strategy_plan_output_path": str(output_path),
                "model_name": model_name,
                "selected_strategy": strategy_plan.get("selected_strategy"),
                "scenario_type": strategy_plan.get("scenario_type"),
                "next_agent": strategy_plan.get("next_agent"),
                "strategy_plan_preview": {
                    "scenario_type": strategy_plan.get("scenario_type"),
                    "selected_strategy": strategy_plan.get("selected_strategy"),
                    "strategy_reason": strategy_plan.get("strategy_reason"),
                    "optimisation_config": strategy_plan.get("optimisation_config"),
                    "required_constraints": strategy_plan.get("required_constraints"),
                    "risk_notes": strategy_plan.get("risk_notes"),
                    "next_agent": strategy_plan.get("next_agent"),
                },
            }

            self.update_belief("selected_strategy", strategy_plan.get("selected_strategy"))
            self.update_belief("scenario_type", strategy_plan.get("scenario_type"))
            self.update_belief("next_agent", strategy_plan.get("next_agent"))

            self.update_status("completed")
            self.clear_current_task()

            return result

        except Exception as e:
            self.update_belief("error", str(e))
            self.update_status("failed")
            self.clear_current_task()

            return {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "status": "failed",
                "message": "LLMAgentWrapper failed.",
                "error": str(e),
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

    def _resolve_retrieved_context_path(self, task: AgentTask) -> Path:
        """
        Priority:
        1. task.metadata["retrieved_context_output_path"]
        2. task.metadata["retrieved_context_path"]
        3. self.config["retrieved_context_path"]
        4. Agents/retrieved_context.json
        """

        if task.metadata.get("retrieved_context_output_path"):
            return Path(task.metadata["retrieved_context_output_path"])

        if task.metadata.get("retrieved_context_path"):
            return Path(task.metadata["retrieved_context_path"])

        if self.config.get("retrieved_context_path"):
            return Path(self.config["retrieved_context_path"])

        return Path("Agents/retrieved_context.json")

    def _resolve_output_path(self, task: AgentTask) -> Path:
        """
        Priority:
        1. task.metadata["strategy_plan_output_path"]
        2. self.config["strategy_plan_output_path"]
        3. Agents/strategy_plan.json
        """

        if task.metadata.get("strategy_plan_output_path"):
            return Path(task.metadata["strategy_plan_output_path"])

        if self.config.get("strategy_plan_output_path"):
            return Path(self.config["strategy_plan_output_path"])

        return Path("Agents/strategy_plan.json")

    def _resolve_model_name(self, task: AgentTask) -> str:
        if task.metadata.get("model_name"):
            return str(task.metadata["model_name"])

        if self.config.get("model_name"):
            return str(self.config["model_name"])

        return "qwen2.5:7b"

    def _resolve_temperature(self, task: AgentTask) -> float:
        if "temperature" in task.metadata:
            return float(task.metadata["temperature"])

        if "temperature" in self.config:
            return float(self.config["temperature"])

        return 0.1

    def _resolve_num_ctx(self, task: AgentTask) -> int:
        if task.metadata.get("num_ctx"):
            return int(task.metadata["num_ctx"])

        if self.config.get("num_ctx"):
            return int(self.config["num_ctx"])

        return 8192