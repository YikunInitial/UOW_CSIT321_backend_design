from typing import Any, Dict, Optional

from Agents.core.base_agent import BaseAgent

from Agents.wrappers.data_agent_wrapper import DataAgentWrapper
from Agents.wrappers.rag_agent_wrapper import RAGAgentWrapper
from Agents.wrappers.llm_agent_wrapper import LLMAgentWrapper
from Agents.wrappers.optimisation_agent_wrapper import OptimisationAgentWrapper
from Agents.wrappers.visualization_agent_wrapper import VisualizationAgentWrapper


class AgentFactory:
    """
    AgentFactory is responsible for creating managed agent wrappers.

    It hides the concrete wrapper class names from the rest of the system.

    Example:
        factory.create_agent("data", "DataAgent_001")
        factory.create_agent("rag", "RAGAgent_001")
        factory.create_agent("llm", "LLMAgent_001")
    """

    def __init__(self):
        self.agent_classes = {
            "data": DataAgentWrapper,
            "rag": RAGAgentWrapper,
            "llm": LLMAgentWrapper,
            "optimisation": OptimisationAgentWrapper,
            "visualization": VisualizationAgentWrapper,
        }

        self.default_id_prefix = {
            "data": "DataAgent",
            "rag": "RAGAgent",
            "llm": "LLMAgent",
            "optimisation": "OptimisationAgent",
            "visualization": "VisualizationAgent",
        }

        self._counters = {
            "data": 0,
            "rag": 0,
            "llm": 0,
            "optimisation": 0,
            "visualization": 0,
        }

    def create_agent(
        self,
        agent_type: str,
        agent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseAgent:
        """
        Create an agent wrapper by type.
        """

        if agent_type not in self.agent_classes:
            raise ValueError(
                f"Unsupported agent type: {agent_type}. "
                f"Supported types are: {list(self.agent_classes.keys())}"
            )

        if agent_id is None:
            agent_id = self._generate_agent_id(agent_type)

        agent_class = self.agent_classes[agent_type]

        return agent_class(
            agent_id=agent_id,
            config=config or {},
        )

    def register_agent_type(
        self,
        agent_type: str,
        agent_class,
        id_prefix: Optional[str] = None,
    ) -> None:
        """
        Register a new agent type in the factory.

        This is useful if you add new wrappers later, such as:
        - CustomerAgentWrapper
        - VerificationAgentWrapper
        - ReportAgentWrapper
        """

        if agent_type in self.agent_classes:
            raise ValueError(f"Agent type already registered: {agent_type}")

        self.agent_classes[agent_type] = agent_class
        self.default_id_prefix[agent_type] = id_prefix or agent_type.capitalize() + "Agent"
        self._counters[agent_type] = 0

    def _generate_agent_id(self, agent_type: str) -> str:
        self._counters[agent_type] += 1
        prefix = self.default_id_prefix[agent_type]
        return f"{prefix}_{self._counters[agent_type]:03d}"

    def supported_agent_types(self):
        return list(self.agent_classes.keys())