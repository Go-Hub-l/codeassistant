from __future__ import annotations

from typing import Any, cast

from coding_assistant.agents.architect_agent import ArchitectAgent
from coding_assistant.agents.base import Agent
from coding_assistant.agents.dev_agent import DevAgent
from coding_assistant.agents.pm_agent import PMAgent
from coding_assistant.agents.qa_agent import QAAgent
from coding_assistant.agents.reviewer_agent import ReviewerAgent
from coding_assistant.core.types import AgentRole


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[AgentRole, Agent] = {}

    def register(self, agent: Agent) -> None:
        if agent.role in self._agents:
            raise ValueError(f"Agent with role '{agent.role.value}' is already registered")
        self._agents[agent.role] = agent

    def get(self, role: AgentRole) -> Agent:
        if role not in self._agents:
            raise KeyError(f"No agent registered for role '{role.value}'")
        return self._agents[role]

    def list_roles(self) -> list[AgentRole]:
        return list(self._agents.keys())

    def has(self, role: AgentRole) -> bool:
        return role in self._agents

    def all_agents(self) -> dict[AgentRole, Agent]:
        return dict(self._agents)


def create_default_registry(**kwargs: Any) -> AgentRegistry:
    registry = AgentRegistry()

    registry.register(
        PMAgent(
            llm_client=kwargs.get("llm_client"),
            model=kwargs.get("model"),
        )
    )

    registry.register(
        ArchitectAgent(
            llm_client=kwargs.get("llm_client"),
            model=kwargs.get("model"),
        )
    )

    registry.register(
        DevAgent(
            llm_client=kwargs.get("llm_client"),
            model=kwargs.get("model"),
            project_dir=kwargs.get("project_dir"),
            fs_tool=kwargs.get("fs_tool"),
            shell_tool=kwargs.get("shell_tool"),
        )
    )

    registry.register(
        ReviewerAgent(
            llm_client=kwargs.get("llm_client"),
            model=kwargs.get("model"),
            project_dir=kwargs.get("project_dir"),
            fs_tool=kwargs.get("fs_tool"),
            shell_tool=kwargs.get("shell_tool"),
        )
    )

    registry.register(
        QAAgent(
            llm_client=kwargs.get("llm_client"),
            model=kwargs.get("model"),
            project_dir=kwargs.get("project_dir"),
            fs_tool=kwargs.get("fs_tool"),
            shell_tool=kwargs.get("shell_tool"),
        )
    )

    pmgr_config = {
        "role": AgentRole.PMGR,
        "system_prompt": (
            "You are a Project Manager agent (the Host). Orchestrate agent scheduling, "
            "manage handoffs, enforce checkpoints, and track progress. "
            "Decide which agent speaks next based on the current task state. "
            "Present artifacts to the user at checkpoints for confirmation. "
            "When done, call the handoff tool."
        ),
    }

    for config in [pmgr_config]:
        role: AgentRole = cast(AgentRole, config["role"])
        agent_kwargs = {
            "llm_client": kwargs.get("llm_client"),
            "model": kwargs.get("model"),
            "role": role,
        }
        agent_kwargs.update({k: v for k, v in config.items() if k != "role"})
        registry.register(Agent(**agent_kwargs))

    return registry
