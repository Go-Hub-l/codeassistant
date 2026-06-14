from __future__ import annotations

from typing import Any, cast

from coding_assistant.agents.architect_agent import ArchitectAgent
from coding_assistant.agents.base import Agent
from coding_assistant.agents.dev_agent import DevAgent
from coding_assistant.agents.pm_agent import PMAgent
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

    reviewer_config = {
        "role": AgentRole.REVIEWER,
        "system_prompt": (
            "You are a Code Reviewer agent. Audit code for quality, security, "
            "and convention compliance. Run static analysis tools (ruff, bandit, mypy). "
            "Classify issues by severity: minor, major, or critical. "
            "Write your review report to the Workspace Review partition. "
            "When done, call the handoff tool."
        ),
    }
    qa_config = {
        "role": AgentRole.QA,
        "system_prompt": (
            "You are a QA agent. Generate and execute tests for the implemented code. "
            "Create unit tests and integration tests using pytest. "
            "Run tests in Docker, capture results and coverage. "
            "Classify failures by severity. "
            "Write your test report to the Workspace Test partition. "
            "When done, call the handoff tool."
        ),
    }
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

    for config in [reviewer_config, qa_config, pmgr_config]:
        role: AgentRole = cast(AgentRole, config["role"])
        agent_kwargs = {
            "llm_client": kwargs.get("llm_client"),
            "model": kwargs.get("model"),
            "role": role,
        }
        agent_kwargs.update({k: v for k, v in config.items() if k != "role"})
        registry.register(Agent(**agent_kwargs))

    return registry
