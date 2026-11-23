from typing import Dict, List, Optional

from Agents.core.base_agent import BaseAgent


class AgentRegistry:
    """
    AgentRegistry manages all active agents in the system.

    It is responsible for:
    - registering agents
    - removing agents
    - finding agents by ID
    - finding agents by type
    - finding idle agents
    - checking current system agent status
    """

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}

    # -----------------------------
    # Register / remove agents
    # -----------------------------

    def register(self, agent: BaseAgent) -> BaseAgent:
        if agent.agent_id in self.agents:
            raise ValueError(f"Agent already exists: {agent.agent_id}")

        self.agents[agent.agent_id] = agent
        return agent

    def remove(self, agent_id: str) -> Optional[BaseAgent]:
        agent = self.agents.get(agent_id)

        if agent is None:
            return None

        agent.update_status("removed")
        removed_agent = self.agents.pop(agent_id)
        return removed_agent

    # -----------------------------
    # Get agents
    # -----------------------------

    def get(self, agent_id: str) -> Optional[BaseAgent]:
        return self.agents.get(agent_id)

    def get_all(self) -> List[BaseAgent]:
        return list(self.agents.values())

    def get_by_type(self, agent_type: str) -> List[BaseAgent]:
        return [
            agent
            for agent in self.agents.values()
            if agent.agent_type == agent_type
        ]

    def get_idle_by_type(self, agent_type: str) -> List[BaseAgent]:
        return [
            agent
            for agent in self.agents.values()
            if agent.agent_type == agent_type and agent.status == "idle"
        ]

    def get_running_agents(self) -> List[BaseAgent]:
        return [
            agent
            for agent in self.agents.values()
            if agent.status == "running"
        ]

    def get_failed_agents(self) -> List[BaseAgent]:
        return [
            agent
            for agent in self.agents.values()
            if agent.status == "failed"
        ]

    # -----------------------------
    # Update agent status
    # -----------------------------

    def update_status(self, agent_id: str, status: str) -> None:
        agent = self.get(agent_id)

        if agent is None:
            raise ValueError(f"Agent not found: {agent_id}")

        agent.update_status(status)

    # -----------------------------
    # Summary / debug
    # -----------------------------

    def count(self) -> int:
        return len(self.agents)

    def exists(self, agent_id: str) -> bool:
        return agent_id in self.agents

    def summary(self) -> Dict[str, Dict]:
        return {
            agent_id: agent.to_dict()
            for agent_id, agent in self.agents.items()
        }

    def print_summary(self) -> None:
        print("Agent Registry Summary")
        print("----------------------")

        if not self.agents:
            print("No agents registered.")
            return

        for agent_id, agent in self.agents.items():
            print(
                f"{agent_id} | "
                f"type={agent.agent_type} | "
                f"status={agent.status} | "
                f"task={agent.current_task_id}"
            )