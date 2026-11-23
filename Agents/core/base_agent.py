from datetime import datetime
from typing import Any, Dict, List, Optional


class BaseAgent:
    """
    BaseAgent is the parent class for all managed agent wrappers.

    Important:
    - This class does NOT replace your existing functional agents.
    - Your current data_agent.py, rag_agent.py, llm_agent.py, etc. can remain unchanged.
    - New wrapper agents will inherit from BaseAgent and call your existing functions internally.
    """

    VALID_STATUSES = {
        "idle",
        "running",
        "completed",
        "failed",
        "removed",
    }

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if not agent_id:
            raise ValueError("agent_id cannot be empty.")

        if not agent_type:
            raise ValueError("agent_type cannot be empty.")

        self.agent_id = agent_id
        self.agent_type = agent_type
        self.name = name or agent_id
        self.config = config or {}

        self.status = "idle"
        self.current_task_id: Optional[str] = None

        # BDI-related fields
        self.beliefs: Dict[str, Any] = {}
        self.desires: List[str] = []
        self.intentions: List[str] = []

        self.created_at = datetime.now()
        self.last_active = datetime.now()

    # -----------------------------
    # Basic state management
    # -----------------------------

    def update_status(self, status: str) -> None:
        if status not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. "
                f"Valid statuses are: {self.VALID_STATUSES}"
            )

        self.status = status
        self.touch()

    def touch(self) -> None:
        self.last_active = datetime.now()

    def set_current_task(self, task_id: str) -> None:
        self.current_task_id = task_id
        self.touch()

    def clear_current_task(self) -> None:
        self.current_task_id = None
        self.touch()

    # -----------------------------
    # BDI helper methods
    # -----------------------------

    def update_belief(self, key: str, value: Any) -> None:
        self.beliefs[key] = value
        self.touch()

    def remove_belief(self, key: str) -> None:
        if key in self.beliefs:
            del self.beliefs[key]
        self.touch()

    def add_desire(self, desire: str) -> None:
        if desire not in self.desires:
            self.desires.append(desire)
        self.touch()

    def clear_desires(self) -> None:
        self.desires.clear()
        self.touch()

    def add_intention(self, intention: str) -> None:
        if intention not in self.intentions:
            self.intentions.append(intention)
        self.touch()

    def clear_intentions(self) -> None:
        self.intentions.clear()
        self.touch()

    # -----------------------------
    # Main execution interface
    # -----------------------------

    def act(self, task: Any) -> Any:
        """
        Each wrapper agent must implement this method.

        Example:
            DataAgentWrapper.act(task)
            OptimisationAgentWrapper.act(task)
            VisualizationAgentWrapper.act(task)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement act()."
        )

    # -----------------------------
    # Export / debug helper
    # -----------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "name": self.name,
            "status": self.status,
            "current_task_id": self.current_task_id,
            "beliefs": self.beliefs,
            "desires": self.desires,
            "intentions": self.intentions,
            "config": self.config,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"agent_id='{self.agent_id}', "
            f"agent_type='{self.agent_type}', "
            f"status='{self.status}'"
            f")"
        )