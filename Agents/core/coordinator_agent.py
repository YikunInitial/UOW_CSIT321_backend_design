from typing import Any, Dict

from Agents.core.base_agent import BaseAgent
from Agents.core.agent_registry import AgentRegistry
from Agents.core.task import AgentTask


class CoordinatorAgent(BaseAgent):
    """
    CoordinatorAgent controls the workflow between different managed agents.

    Current full workflow:
    1. DataAgentWrapper:
       cleaned Excel -> products.json

    2. RAGAgentWrapper:
       products.json + knowledge base -> retrieved_context.json

    3. LLMAgentWrapper:
       products.json + retrieved_context.json -> strategy_plan.json

    4. OptimisationAgentWrapper:
       products.json + strategy_plan.json -> loading_solution.json

    5. VisualizationAgentWrapper:
       loading_solution.json -> layout images + PDF report + visualization summary
    """

    def __init__(
        self,
        agent_id: str,
        registry: AgentRegistry,
        config: Dict[str, Any] | None = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="coordinator",
            name="Coordinator Agent",
            config=config,
        )

        self.registry = registry

    def act(self, task: AgentTask) -> Dict[str, Any]:
        self.update_status("running")
        self.set_current_task(task.task_id)

        try:
            task.update_status("running")

            self.update_belief("task_id", task.task_id)
            self.update_belief("task_type", task.task_type)
            self.update_belief("objective", task.objective)
            self.update_belief("constraints", task.constraints)

            self.add_desire("complete_full_agent_workflow")
            self.add_intention("coordinate_data_agent")
            self.add_intention("coordinate_rag_agent")
            self.add_intention("coordinate_llm_agent")
            self.add_intention("coordinate_optimisation_agent")
            self.add_intention("coordinate_visualization_agent")

            # Step 1: Data Agent
            data_agent = self._get_available_agent(agent_type="data")
            data_result = data_agent.act(task)

            if data_result.get("status") != "completed":
                raise RuntimeError(
                    f"Data agent failed: {data_result.get('error', data_result)}"
                )

            if data_result.get("json_output_path"):
                task.update_metadata("products_json_path", data_result["json_output_path"])
                task.update_metadata("json_output_path", data_result["json_output_path"])

            # Step 2: RAG Agent
            rag_agent = self._get_available_agent(agent_type="rag")
            rag_result = rag_agent.act(task)

            if rag_result.get("status") != "completed":
                raise RuntimeError(
                    f"RAG agent failed: {rag_result.get('error', rag_result)}"
                )

            if rag_result.get("retrieved_context_output_path"):
                task.update_metadata(
                    "retrieved_context_path",
                    rag_result["retrieved_context_output_path"],
                )
                task.update_metadata(
                    "retrieved_context_output_path",
                    rag_result["retrieved_context_output_path"],
                )

            # Step 3: LLM Agent
            llm_agent = self._get_available_agent(agent_type="llm")
            llm_result = llm_agent.act(task)

            if llm_result.get("status") != "completed":
                raise RuntimeError(
                    f"LLM agent failed: {llm_result.get('error', llm_result)}"
                )

            if llm_result.get("strategy_plan_output_path"):
                task.update_metadata(
                    "strategy_plan_path",
                    llm_result["strategy_plan_output_path"],
                )
                task.update_metadata(
                    "strategy_plan_output_path",
                    llm_result["strategy_plan_output_path"],
                )

            # Step 4: Optimisation Agent
            optimisation_agent = self._get_available_agent(agent_type="optimisation")
            optimisation_result = optimisation_agent.act(task)

            if optimisation_result.get("status") != "completed":
                raise RuntimeError(
                    "Optimisation agent failed: "
                    f"{optimisation_result.get('error', optimisation_result)}"
                )

            if optimisation_result.get("loading_solution_output_path"):
                task.update_metadata(
                    "loading_solution_path",
                    optimisation_result["loading_solution_output_path"],
                )
                task.update_metadata(
                    "loading_solution_output_path",
                    optimisation_result["loading_solution_output_path"],
                )

            # Step 5: Visualization Agent
            visualization_agent = self._get_available_agent(agent_type="visualization")
            visualization_result = visualization_agent.act(task)

            if visualization_result.get("status") != "completed":
                raise RuntimeError(
                    "Visualization agent failed: "
                    f"{visualization_result.get('error', visualization_result)}"
                )

            if visualization_result.get("visualization_summary_output_path"):
                task.update_metadata(
                    "visualization_summary_path",
                    visualization_result["visualization_summary_output_path"],
                )

            if visualization_result.get("pdf_report"):
                task.update_metadata(
                    "pdf_report",
                    visualization_result["pdf_report"],
                )

            workflow_result = {
                "task_id": task.task_id,
                "coordinator_id": self.agent_id,
                "status": "completed",
                "message": (
                    "CoordinatorAgent completed Data -> RAG -> LLM -> "
                    "Optimisation -> Visualization workflow."
                ),
                "steps": {
                    "data_agent": data_result,
                    "rag_agent": rag_result,
                    "llm_agent": llm_result,
                    "optimisation_agent": optimisation_result,
                    "visualization_agent": visualization_result,
                },
                "outputs": {
                    "products_json_path": task.metadata.get("products_json_path"),
                    "retrieved_context_path": task.metadata.get("retrieved_context_path"),
                    "strategy_plan_path": task.metadata.get("strategy_plan_path"),
                    "loading_solution_path": task.metadata.get("loading_solution_path"),
                    "visualization_summary_path": task.metadata.get(
                        "visualization_summary_path"
                    ),
                    "pdf_report": task.metadata.get("pdf_report"),
                },
                "summary": {
                    "selected_strategy": llm_result.get("selected_strategy"),
                    "scenario_type": llm_result.get("scenario_type"),
                    "wagon_count": optimisation_result.get("wagon_count"),
                    "total_products": optimisation_result.get("total_products"),
                    "average_area_utilisation": optimisation_result.get(
                        "average_area_utilisation"
                    ),
                    "objective_score": optimisation_result.get("objective_score"),
                    "layout_image_count": visualization_result.get("layout_image_count"),
                    "pdf_report": visualization_result.get("pdf_report"),
                },
            }

            task.set_result(workflow_result)

            self.update_status("completed")
            self.clear_current_task()

            return workflow_result

        except Exception as e:
            task.set_error(str(e))

            self.update_belief("error", str(e))
            self.update_status("failed")
            self.clear_current_task()

            return {
                "task_id": task.task_id,
                "coordinator_id": self.agent_id,
                "status": "failed",
                "message": "CoordinatorAgent failed.",
                "error": str(e),
            }

    def _get_available_agent(self, agent_type: str) -> BaseAgent:
        """
        Find an available agent by type.

        Priority:
        1. idle agent
        2. completed agent, reuse it
        3. any existing agent of that type
        """

        idle_agents = self.registry.get_idle_by_type(agent_type)
        if idle_agents:
            return idle_agents[0]

        reusable_agents = [
            agent
            for agent in self.registry.get_by_type(agent_type)
            if agent.status in {"completed", "idle"}
        ]
        if reusable_agents:
            return reusable_agents[0]

        all_agents = self.registry.get_by_type(agent_type)
        if all_agents:
            return all_agents[0]

        raise RuntimeError(f"No agent available for type: {agent_type}")