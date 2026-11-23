from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class AgentTask:
    """
    AgentTask represents one user/customer request in the agent system.

    It is used as the common input object passed between:
    - CustomerAgent
    - CoordinatorAgent
    - DataAgentWrapper
    - RAGAgentWrapper
    - LLMAgentWrapper
    - OptimisationAgentWrapper
    - VisualizationAgentWrapper
    """

    task_id: str
    customer_id: Optional[str] = None
    task_type: str = "loading_optimisation"

    input_file: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None

    objective: str = "minimise_vehicle_number"
    constraints: List[str] = field(default_factory=list)

    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_status(self, status: str) -> None:
        valid_statuses = {"pending", "running", "completed", "failed"}

        if status not in valid_statuses:
            raise ValueError(
                f"Invalid task status '{status}'. "
                f"Valid statuses are: {valid_statuses}"
            )

        self.status = status
        self.updated_at = datetime.now()

    def set_result(self, result: Dict[str, Any]) -> None:
        self.result = result
        self.error = None
        self.status = "completed"
        self.updated_at = datetime.now()

    def set_error(self, error: str) -> None:
        self.error = error
        self.status = "failed"
        self.updated_at = datetime.now()

    def update_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "customer_id": self.customer_id,
            "task_type": self.task_type,
            "input_file": self.input_file,
            "input_data": self.input_data,
            "objective": self.objective,
            "constraints": self.constraints,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"AgentTask("
            f"task_id='{self.task_id}', "
            f"task_type='{self.task_type}', "
            f"status='{self.status}'"
            f")"
        )