import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from Agents.core.base_agent import BaseAgent
from Agents.core.event import Event
from Agents.core.event_bus import EventBus
from Agents.core.task import AgentTask

from Agents.wrappers.llm_agent_wrapper import LLMAgentWrapper
from Agents.wrappers.optimisation_agent_wrapper import OptimisationAgentWrapper
from Agents.wrappers.rag_agent_wrapper import RAGAgentWrapper
from Agents.wrappers.visualization_agent_wrapper import VisualizationAgentWrapper


class BatchCoordinatorAgentWrapper(BaseAgent):
    """
    Event-driven batch coordinator.

    Workflow:
        batch.ready
        -> products.json
        -> RAG
        -> LLM strategy recommendation
        -> strategy-driven optimisation
        -> visualisation + PDF
        -> batch.completed

    The final batch.completed payload reports both:
    - requested_strategy: LLM recommendation
    - applied_strategy: strategy actually executed by OptimisationAgent
    """

    def __init__(
        self,
        agent_id: str,
        event_bus: EventBus,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="batch_coordinator",
            name="Batch Coordinator Agent Wrapper",
            config=config,
        )

        self.event_bus = event_bus

    def start(self) -> None:
        self.event_bus.subscribe(
            topic="batch.ready",
            handler=self.handle_batch_ready,
        )

        self.update_status("idle")
        self.update_belief("subscribed_topic", "batch.ready")
        self.add_desire("process_ready_batches")
        self.add_intention("listen_to_batch_ready_events")

        print(f"[{self.agent_id}] Subscribed to batch.ready")

    def handle_batch_ready(self, event: Event) -> None:
        self.update_status("running")

        try:
            batch = event.payload.get("batch")

            if not batch:
                raise ValueError(
                    "batch.ready event missing payload['batch']."
                )

            batch_id = batch.get("batch_id")
            destination = batch.get("destination")
            products = batch.get("products", [])
            source_task_ids = batch.get("source_task_ids", [])
            customer_ids = batch.get("customer_ids", [])

            if not batch_id:
                raise ValueError("Batch missing batch_id.")

            if not products:
                raise ValueError(
                    f"Batch {batch_id} has no products."
                )

            self.set_current_task(batch_id)

            self.update_belief("batch_id", batch_id)
            self.update_belief("destination", destination)
            self.update_belief("source_task_ids", source_task_ids)
            self.update_belief("customer_ids", customer_ids)
            self.update_belief("product_count", len(products))

            self.add_desire("complete_batch_level_workflow")
            self.add_intention("create_batch_workspace")
            self.add_intention(
                "run_rag_llm_optimisation_visualization"
            )

            print(
                f"[{self.agent_id}] Received batch.ready | "
                f"batch={batch_id} | destination={destination} | "
                f"products={len(products)}"
            )

            workflow_started = time.perf_counter()

            setup_started = time.perf_counter()
            task = self._build_agent_task_from_batch(batch)
            setup_seconds = time.perf_counter() - setup_started

            rag_started = time.perf_counter()
            rag_result = self._run_rag(task)
            rag_seconds = time.perf_counter() - rag_started

            llm_started = time.perf_counter()
            llm_result = self._run_llm(task)
            llm_seconds = time.perf_counter() - llm_started

            optimisation_started = time.perf_counter()
            optimisation_result = self._run_optimisation(task)
            optimisation_seconds = (
                time.perf_counter() - optimisation_started
            )

            visualization_started = time.perf_counter()
            visualization_result = self._run_visualization(task)
            visualization_seconds = (
                time.perf_counter() - visualization_started
            )

            workflow_total_seconds = (
                time.perf_counter() - workflow_started
            )

            timing_seconds = {
                "task_setup": round(setup_seconds, 4),
                "rag": round(rag_seconds, 4),
                "llm": round(llm_seconds, 4),
                "optimisation": round(optimisation_seconds, 4),
                "visualization": round(visualization_seconds, 4),
                "workflow_total": round(workflow_total_seconds, 4),
            }

            self.update_belief(
                "last_batch_timing_seconds",
                timing_seconds,
            )

            requested_strategy = (
                optimisation_result.get("requested_strategy")
                or llm_result.get("selected_strategy")
            )

            requested_scenario_type = (
                optimisation_result.get("requested_scenario_type")
                or llm_result.get("scenario_type")
            )

            applied_strategy = (
                optimisation_result.get("applied_strategy")
                or requested_strategy
            )

            actual_batch_scenario = (
                optimisation_result.get("actual_batch_scenario")
                or requested_scenario_type
            )

            strategy_resolution_notes = optimisation_result.get(
                "strategy_resolution_notes",
                [],
            )

            strategy_fallback_reasons = optimisation_result.get(
                "strategy_fallback_reasons",
                [],
            )
            validation_summary = optimisation_result.get(
                "validation_summary",
                {},
            )
            optimisation_trace = optimisation_result.get(
                "optimisation_trace",
                [],
            )
            algorithm_policy = optimisation_result.get(
                "algorithm_policy",
                {},
            )
            anytime_summary = optimisation_result.get(
                "anytime_summary",
                {},
            )
            performance_summary = optimisation_result.get(
                "performance_summary",
                {},
            )
            algorithm_decision_trace = optimisation_result.get(
                "algorithm_decision_trace",
                [],
            )

            print(
                f"[{self.agent_id}] Final strategy | "
                f"requested={requested_strategy} | "
                f"applied={applied_strategy} | "
                f"scenario={actual_batch_scenario}"
            )

            print(
                f"[Performance] batch={batch_id} | "
                f"setup={setup_seconds:.2f}s | "
                f"rag={rag_seconds:.2f}s | "
                f"llm={llm_seconds:.2f}s | "
                f"optimisation={optimisation_seconds:.2f}s | "
                f"visualization={visualization_seconds:.2f}s | "
                f"total={workflow_total_seconds:.2f}s"
            )

            result = {
                "batch_id": batch_id,
                "destination": destination,
                "source_task_ids": source_task_ids,
                "customer_ids": customer_ids,
                "status": "completed",
                "message": (
                    "BatchCoordinatorAgentWrapper completed "
                    "RAG -> LLM -> Optimisation -> Visualization workflow."
                ),
                "outputs": {
                    "workspace": task.metadata.get("workspace"),
                    "products_json_path": task.metadata.get(
                        "products_json_path"
                    ),
                    "retrieved_context_path": task.metadata.get(
                        "retrieved_context_path"
                    ),
                    "strategy_plan_path": task.metadata.get(
                        "strategy_plan_path"
                    ),
                    "loading_solution_path": task.metadata.get(
                        "loading_solution_path"
                    ),
                    "visualization_summary_path": task.metadata.get(
                        "visualization_summary_output_path"
                    ),
                    "visualization_output_dir": task.metadata.get(
                        "visualization_output_dir"
                    ),
                    "pdf_report": visualization_result.get(
                        "pdf_report"
                    ),
                },

                # Monitor-friendly final outcome.
                # selected_strategy now means the strategy actually applied.
                "summary": {
                    "selected_strategy": applied_strategy,
                    "scenario_type": actual_batch_scenario,
                    "requested_strategy": requested_strategy,
                    "requested_scenario_type": (
                        requested_scenario_type
                    ),
                    "wagon_count": optimisation_result.get(
                        "wagon_count"
                    ),
                    "total_products": optimisation_result.get(
                        "total_products"
                    ),
                    "average_area_utilisation": (
                        optimisation_result.get(
                            "average_area_utilisation"
                        )
                    ),
                    "objective_score": optimisation_result.get(
                        "objective_score"
                    ),
                    "validation_summary": validation_summary,
                    "anytime_summary": anytime_summary,
                    "performance_summary": performance_summary,
                    "layout_image_count": visualization_result.get(
                        "layout_image_count"
                    ),
                },

                # Full audit trail for UI, API, logs, or future dashboard.
                "strategy_execution": {
                    "requested_strategy": requested_strategy,
                    "requested_scenario_type": (
                        requested_scenario_type
                    ),
                    "applied_strategy": applied_strategy,
                    "actual_batch_scenario": (
                        actual_batch_scenario
                    ),
                    "execution_config": optimisation_result.get(
                        "strategy_execution_config",
                        {},
                    ),
                    "resolution_notes": strategy_resolution_notes,
                    "fallback_reasons": strategy_fallback_reasons,
                    "simulated_annealing": optimisation_result.get(
                        "simulated_annealing",
                        {},
                    ),
                    "algorithm_policy": algorithm_policy,
                    "anytime_summary": anytime_summary,
                    "performance_summary": performance_summary,
                    "algorithm_decision_trace": (
                        algorithm_decision_trace
                    ),
                    "optimisation_trace": optimisation_trace,
                    "validation_summary": validation_summary,
                },

                "timing_seconds": timing_seconds,

                "steps": {
                    "rag_agent": rag_result,
                    "llm_agent": llm_result,
                    "optimisation_agent": optimisation_result,
                    "visualization_agent": visualization_result,
                },
            }

            self._publish_batch_completed(
                batch_id=batch_id,
                result=result,
                source_event=event,
            )

            self.update_belief(
                "last_completed_batch_id",
                batch_id,
            )
            self.update_belief(
                "last_applied_strategy",
                applied_strategy,
            )

            self.update_status("idle")
            self.clear_current_task()

        except Exception as error:
            self.update_belief("error", str(error))
            self.update_status("failed")
            self.clear_current_task()

            print(
                f"[{self.agent_id}] Failed to process "
                f"batch.ready: {error}"
            )

            self._publish_batch_failed(
                source_event=event,
                error=str(error),
            )

    def _build_agent_task_from_batch(
        self,
        batch: Dict[str, Any],
    ) -> AgentTask:
        batch_id = batch["batch_id"]
        destination = batch.get(
            "destination",
            "UNKNOWN_DESTINATION",
        )
        products = batch.get("products", [])

        workspace = Path("runtime") / "batches" / batch_id
        outputs_dir = workspace / "outputs"

        workspace.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        products_json_path = workspace / "products.json"
        retrieved_context_path = workspace / "retrieved_context.json"
        strategy_plan_path = workspace / "strategy_plan.json"
        loading_solution_path = workspace / "loading_solution.json"
        visualization_summary_path = (
            workspace / "visualization_summary.json"
        )

        with products_json_path.open("w", encoding="utf-8") as file:
            json.dump(products, file, ensure_ascii=False, indent=2)

        task = AgentTask(
            task_id=batch_id,
            customer_id=",".join(
                batch.get("customer_ids", [])
            ),
            task_type="batch_loading_optimisation",
            input_file=None,
            objective=batch.get(
                "objective",
                "minimise_vehicle_number",
            ),
            constraints=batch.get("constraints", []),
            metadata={
                "batch_id": batch_id,
                "destination": destination,
                "source_task_ids": batch.get(
                    "source_task_ids",
                    [],
                ),
                "customer_ids": batch.get("customer_ids", []),
                "workspace": str(workspace),

                "products_json_path": str(products_json_path),
                "json_output_path": str(products_json_path),
                "retrieved_context_path": str(
                    retrieved_context_path
                ),
                "retrieved_context_output_path": str(
                    retrieved_context_path
                ),
                "strategy_plan_path": str(strategy_plan_path),
                "strategy_plan_output_path": str(
                    strategy_plan_path
                ),
                "loading_solution_path": str(loading_solution_path),
                "loading_solution_output_path": str(
                    loading_solution_path
                ),

                "visualization_output_dir": str(outputs_dir),
                "visualization_summary_output_path": str(
                    visualization_summary_path
                ),

                "knowledge_base_dir": self.config.get(
                    "knowledge_base_dir",
                    "KnowledgeBase",
                ),
                "persist_directory": self.config.get(
                    "persist_directory",
                    "VectorDB/chroma_db",
                ),
                "top_k": self.config.get("top_k", 6),
                "rebuild_vector_db": self.config.get(
                    "rebuild_vector_db",
                    False,
                ),

                "model_name": self.config.get(
                    "model_name",
                    "qwen2.5:7b",
                ),
                "temperature": self.config.get(
                    "temperature",
                    0.1,
                ),
                "num_ctx": self.config.get("num_ctx", 8192),

                "car_length": self.config.get(
                    "car_length",
                    13000.0,
                ),
                "car_width": self.config.get(
                    "car_width",
                    3000.0,
                ),
                "max_weight_tonne": self.config.get(
                    "max_weight_tonne",
                    60.0,
                ),
                "parallel_gap": self.config.get(
                    "parallel_gap",
                    50.0,
                ),
                "bogie_tol_tons": self.config.get(
                    "bogie_tol_tons",
                    2.0,
                ),
                "max_weight_diff_for_pair": self.config.get(
                    "max_weight_diff_for_pair",
                    1.0,
                ),
                "max_outer_diameter_diff_for_pair": self.config.get(
                    "max_outer_diameter_diff_for_pair",
                    100.0,
                ),
                "force_same_outer_diameter_for_pair": (
                    self.config.get(
                        "force_same_outer_diameter_for_pair",
                        True,
                    )
                ),
                "enable_low_utilisation_merge": self.config.get(
                    "enable_low_utilisation_merge",
                    True,
                ),
                "low_utilisation_threshold": self.config.get(
                    "low_utilisation_threshold",
                    0.35,
                ),
                "enable_wagon_reassignment": self.config.get(
                    "enable_wagon_reassignment",
                    True,
                ),
                "max_wagon_reassignment_passes": self.config.get(
                    "max_wagon_reassignment_passes",
                    2,
                ),
                "seed": self.config.get("seed", 42),

                "figure_dpi": self.config.get("figure_dpi", 200),
            },
        )

        print(
            f"[{self.agent_id}] Batch AgentTask created | "
            f"batch={batch_id} | workspace={workspace}"
        )

        return task

    def _run_rag(self, task: AgentTask) -> Dict[str, Any]:
        rag_agent = RAGAgentWrapper(
            agent_id=f"RAGAgent_{task.task_id}",
            config={
                "products_json_path": task.metadata[
                    "products_json_path"
                ],
                "retrieved_context_output_path": task.metadata[
                    "retrieved_context_output_path"
                ],
                "knowledge_base_dir": task.metadata[
                    "knowledge_base_dir"
                ],
                "persist_directory": task.metadata[
                    "persist_directory"
                ],
                "top_k": task.metadata["top_k"],
                "rebuild_vector_db": task.metadata[
                    "rebuild_vector_db"
                ],
                "device": self.config.get("device", "cpu"),
            },
        )

        result = rag_agent.act(task)

        if result.get("status") != "completed":
            raise RuntimeError(f"RAG failed: {result}")

        return result

    def _run_llm(self, task: AgentTask) -> Dict[str, Any]:
        llm_agent = LLMAgentWrapper(
            agent_id=f"LLMAgent_{task.task_id}",
            config={
                "products_json_path": task.metadata[
                    "products_json_path"
                ],
                "retrieved_context_path": task.metadata[
                    "retrieved_context_path"
                ],
                "strategy_plan_output_path": task.metadata[
                    "strategy_plan_output_path"
                ],
                "model_name": task.metadata["model_name"],
                "temperature": task.metadata["temperature"],
                "num_ctx": task.metadata["num_ctx"],
            },
        )

        result = llm_agent.act(task)

        if result.get("status") != "completed":
            raise RuntimeError(f"LLM failed: {result}")

        return result

    def _run_optimisation(
        self,
        task: AgentTask,
    ) -> Dict[str, Any]:
        optimisation_agent = OptimisationAgentWrapper(
            agent_id=f"OptimisationAgent_{task.task_id}",
            config={
                "products_json_path": task.metadata[
                    "products_json_path"
                ],
                "strategy_plan_path": task.metadata[
                    "strategy_plan_path"
                ],
                "loading_solution_output_path": task.metadata[
                    "loading_solution_output_path"
                ],
                "car_length": task.metadata["car_length"],
                "car_width": task.metadata["car_width"],
                "max_weight_tonne": task.metadata[
                    "max_weight_tonne"
                ],
                "parallel_gap": task.metadata["parallel_gap"],
                "bogie_tol_tons": task.metadata[
                    "bogie_tol_tons"
                ],
                "max_weight_diff_for_pair": task.metadata[
                    "max_weight_diff_for_pair"
                ],
                "max_outer_diameter_diff_for_pair": task.metadata[
                    "max_outer_diameter_diff_for_pair"
                ],
                "force_same_outer_diameter_for_pair": task.metadata[
                    "force_same_outer_diameter_for_pair"
                ],
                "enable_low_utilisation_merge": task.metadata[
                    "enable_low_utilisation_merge"
                ],
                "low_utilisation_threshold": task.metadata[
                    "low_utilisation_threshold"
                ],
                "enable_wagon_reassignment": task.metadata[
                    "enable_wagon_reassignment"
                ],
                "max_wagon_reassignment_passes": task.metadata[
                    "max_wagon_reassignment_passes"
                ],
                "seed": task.metadata["seed"],
            },
        )

        result = optimisation_agent.act(task)

        if result.get("status") != "completed":
            raise RuntimeError(f"Optimisation failed: {result}")

        return result

    def _run_visualization(
        self,
        task: AgentTask,
    ) -> Dict[str, Any]:
        visualization_agent = VisualizationAgentWrapper(
            agent_id=f"VisualizationAgent_{task.task_id}",
            config={
                "loading_solution_path": task.metadata[
                    "loading_solution_path"
                ],
                "visualization_output_dir": task.metadata[
                    "visualization_output_dir"
                ],
                "visualization_summary_output_path": task.metadata[
                    "visualization_summary_output_path"
                ],
                "figure_dpi": task.metadata["figure_dpi"],
            },
        )

        result = visualization_agent.act(task)

        if result.get("status") != "completed":
            raise RuntimeError(f"Visualization failed: {result}")

        return result

    def _publish_batch_completed(
        self,
        batch_id: str,
        result: Dict[str, Any],
        source_event: Event,
    ) -> None:
        event = Event(
            topic="batch.completed",
            source_agent_id=self.agent_id,
            correlation_id=batch_id,
            payload={
                "batch_id": batch_id,
                "result": result,
                "source_event_id": source_event.event_id,
            },
        )

        print(
            f"[{self.agent_id}] Publishing batch.completed | "
            f"batch={batch_id}"
        )
        self.event_bus.publish(event)

    def _publish_batch_failed(
        self,
        source_event: Event,
        error: str,
    ) -> None:
        batch_payload = source_event.payload.get("batch", {})
        batch_id = batch_payload.get(
            "batch_id",
            source_event.correlation_id,
        )

        event = Event(
            topic="batch.failed",
            source_agent_id=self.agent_id,
            correlation_id=batch_id,
            payload={
                "batch_id": batch_id,
                "error": error,
                "source_event_id": source_event.event_id,
            },
        )

        print(
            f"[{self.agent_id}] Publishing batch.failed | "
            f"batch={batch_id}"
        )
        self.event_bus.publish(event)

    def act(self, task: Any) -> Any:
        raise NotImplementedError(
            "BatchCoordinatorAgentWrapper is event-driven. "
            "Use start() and handle_batch_ready(event)."
        )
