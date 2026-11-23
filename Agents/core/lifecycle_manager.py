from typing import Any, Dict, List, Optional

from Agents.core.agent_factory import AgentFactory
from Agents.core.agent_registry import AgentRegistry
from Agents.core.base_agent import BaseAgent


class LifecycleManager:
    """
    LifecycleManager controls the lifecycle of managed agents.

    It is responsible for:
    - creating agents
    - registering agents
    - reusing idle/completed agents
    - removing agents
    - removing all agents of a specific type
    - checking active agent status
    """

    def __init__(
        self,
        registry: AgentRegistry,
        factory: AgentFactory,
    ):
        self.registry = registry
        self.factory = factory

    # -----------------------------
    # Create / register
    # -----------------------------

    def create_agent(
        self,
        agent_type: str,
        agent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseAgent:
        """
        Create a new agent and register it into the registry.
        """

        agent = self.factory.create_agent(
            agent_type=agent_type,
            agent_id=agent_id,
            config=config or {},
        )

        self.registry.register(agent)

        return agent

    def create_agents_from_config(
        self,
        agent_configs: Dict[str, Dict[str, Any]],
    ) -> List[BaseAgent]:
        """
        Create multiple agents from a config dictionary.

        Example:
            {
                "data": {"agent_id": "DataAgent_001", "config": {...}},
                "rag": {"agent_id": "RAGAgent_001", "config": {...}}
            }
        """

        created_agents = []

        for agent_type, settings in agent_configs.items():
            agent_id = settings.get("agent_id")
            config = settings.get("config", {})

            agent = self.create_agent(
                agent_type=agent_type,
                agent_id=agent_id,
                config=config,
            )

            created_agents.append(agent)

        return created_agents

    # -----------------------------
    # Get or create
    # -----------------------------

    def get_or_create_agent(
        self,
        agent_type: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseAgent:
        """
        Try to reuse an idle or completed agent.
        If none exists, create a new one.
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
            reusable_agents[0].update_status("idle")
            return reusable_agents[0]

        return self.create_agent(
            agent_type=agent_type,
            config=config or {},
        )

    # -----------------------------
    # Remove
    # -----------------------------

    def remove_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Remove one agent from the registry.
        """

        return self.registry.remove(agent_id)

    def remove_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """
        Remove all agents of a specific type.
        """

        agents = list(self.registry.get_by_type(agent_type))
        removed_agents = []

        for agent in agents:
            removed_agent = self.registry.remove(agent.agent_id)
            if removed_agent:
                removed_agents.append(removed_agent)

        return removed_agents

    def remove_completed_agents(self) -> List[BaseAgent]:
        """
        Remove all agents whose status is completed.
        Useful after a workflow has finished.
        """

        agents = [
            agent
            for agent in self.registry.get_all()
            if agent.status == "completed"
        ]

        removed_agents = []

        for agent in agents:
            removed_agent = self.registry.remove(agent.agent_id)
            if removed_agent:
                removed_agents.append(removed_agent)

        return removed_agents

    def remove_failed_agents(self) -> List[BaseAgent]:
        """
        Remove all failed agents.
        """

        agents = self.registry.get_failed_agents()
        removed_agents = []

        for agent in agents:
            removed_agent = self.registry.remove(agent.agent_id)
            if removed_agent:
                removed_agents.append(removed_agent)

        return removed_agents

    # -----------------------------
    # Status helpers
    # -----------------------------

    def list_agents(self) -> List[BaseAgent]:
        return self.registry.get_all()

    def count_agents(self) -> int:
        return self.registry.count()

    def summary(self) -> Dict[str, Dict[str, Any]]:
        return self.registry.summary()

    def print_summary(self) -> None:
        self.registry.print_summary()