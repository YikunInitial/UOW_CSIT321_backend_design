from pathlib import Path
from typing import Any, Dict, Optional

from Agents.core.base_agent import BaseAgent
from Agents.core.task import AgentTask
from Agents.visualization_agent import VisualizationAgent


class VisualizationAgentWrapper(BaseAgent):
    """
    Managed wrapper for the existing Agents/visualization_agent.py.

    This wrapper does not replace the original VisualizationAgent.
    It only adds:
    - agent_id
    - status tracking
    - task tracking
    - BDI fields
    - standard act(task) interface

    Current role:
    loading_solution.json -> layout images + PDF report + visualization_summary.json
    """

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=agent_id,
            agent_type="visualization",
            name="Visualization Agent Wrapper",
            config=config,
        )

    def act(self, task: AgentTask) -> Dict[str, Any]:
        self.update_status("running")
        self.set_current_task(task.task_id)

        try:
            loading_solution_path = self._resolve_loading_solution_path(task)
            output_dir = self._resolve_output_dir(task)
            summary_output_path = self._resolve_summary_output_path(task)
            figure_dpi = self._resolve_figure_dpi(task)

            self.update_belief("task_id", task.task_id)
            self.update_belief("loading_solution_path", str(loading_solution_path))
            self.update_belief("output_dir", str(output_dir))
            self.update_belief("visualization_summary_output_path", str(summary_output_path))
            self.update_belief("figure_dpi", figure_dpi)
            self.update_belief("objective", task.objective)
            self.update_belief("constraints", task.constraints)

            self.add_desire("generate_visual_loading_report")
            self.add_intention("call_existing_VisualizationAgent_run")

            visualization_agent = VisualizationAgent(
                loading_solution_path=loading_solution_path,
                output_dir=output_dir,
                summary_output_path=summary_output_path,
                figure_dpi=figure_dpi,
            )

            visualization_summary = visualization_agent.run()

            layout_images = visualization_summary.get("layout_images", [])
            pdf_report = visualization_summary.get("pdf_report")
            wagon_count = visualization_summary.get("wagon_count")
            total_products = visualization_summary.get("total_products")
            average_area_utilisation = visualization_summary.get("average_area_utilisation")

            result = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "status": "completed",
                "message": "VisualizationAgentWrapper completed successfully.",
                "loading_solution_path": str(loading_solution_path),
                "output_dir": str(output_dir),
                "visualization_summary_output_path": str(summary_output_path),
                "layout_image_count": len(layout_images),
                "layout_images": layout_images,
                "pdf_report": pdf_report,
                "wagon_count": wagon_count,
                "total_products": total_products,
                "average_area_utilisation": average_area_utilisation,
                "strategy_used": visualization_summary.get("strategy_used"),
                "scenario_type": visualization_summary.get("scenario_type"),
            }

            self.update_belief("layout_image_count", len(layout_images))
            self.update_belief("pdf_report", pdf_report)
            self.update_belief("wagon_count", wagon_count)
            self.update_belief("total_products", total_products)

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
                "message": "VisualizationAgentWrapper failed.",
                "error": str(e),
            }

    def _resolve_loading_solution_path(self, task: AgentTask) -> Path:
        """
        Priority:
        1. task.metadata["loading_solution_output_path"]
        2. task.metadata["loading_solution_path"]
        3. self.config["loading_solution_path"]
        4. Agents/loading_solution.json
        """

        if task.metadata.get("loading_solution_output_path"):
            return Path(task.metadata["loading_solution_output_path"])

        if task.metadata.get("loading_solution_path"):
            return Path(task.metadata["loading_solution_path"])

        if self.config.get("loading_solution_path"):
            return Path(self.config["loading_solution_path"])

        return Path("Agents/loading_solution.json")

    def _resolve_output_dir(self, task: AgentTask) -> Path:
        """
        Priority:
        1. task.metadata["visualization_output_dir"]
        2. task.metadata["output_dir"]
        3. self.config["visualization_output_dir"]
        4. self.config["output_dir"]
        5. demo_outputs
        """

        if task.metadata.get("visualization_output_dir"):
            return Path(task.metadata["visualization_output_dir"])

        if task.metadata.get("output_dir"):
            return Path(task.metadata["output_dir"])

        if self.config.get("visualization_output_dir"):
            return Path(self.config["visualization_output_dir"])

        if self.config.get("output_dir"):
            return Path(self.config["output_dir"])

        return Path("demo_outputs")

    def _resolve_summary_output_path(self, task: AgentTask) -> Path:
        """
        Priority:
        1. task.metadata["visualization_summary_output_path"]
        2. task.metadata["summary_output_path"]
        3. self.config["visualization_summary_output_path"]
        4. self.config["summary_output_path"]
        5. Agents/visualization_summary.json
        """

        if task.metadata.get("visualization_summary_output_path"):
            return Path(task.metadata["visualization_summary_output_path"])

        if task.metadata.get("summary_output_path"):
            return Path(task.metadata["summary_output_path"])

        if self.config.get("visualization_summary_output_path"):
            return Path(self.config["visualization_summary_output_path"])

        if self.config.get("summary_output_path"):
            return Path(self.config["summary_output_path"])

        return Path("Agents/visualization_summary.json")

    def _resolve_figure_dpi(self, task: AgentTask) -> int:
        if task.metadata.get("figure_dpi"):
            return int(task.metadata["figure_dpi"])

        if self.config.get("figure_dpi"):
            return int(self.config["figure_dpi"])

        return 200