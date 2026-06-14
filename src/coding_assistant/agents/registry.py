from __future__ import annotations

from typing import Any

from coding_assistant.agents.base import Agent, AgentRole


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

    role_configs: list[dict[str, Any]] = [
        {
            "role": AgentRole.PM,
            "system_prompt": (
                "You are a Product Manager agent. Analyze natural language requirements, "
                "produce structured requirement documents including PRD, user stories, "
                "feature list with priorities, and acceptance criteria. "
                "Write your output to the Workspace Requirements partition. "
                "When done, call the handoff tool."
            ),
        },
        {
            "role": AgentRole.ARCHITECT,
            "system_prompt": (
                "You are an Architect agent. Design technical architecture based on requirements. "
                "Select the best Python framework and libraries. Define API contracts, "
                "database schema, project structure, and security considerations. "
                "Write your output to the Workspace Architecture partition. "
                "When done, call the handoff tool."
            ),
        },
        {
            "role": AgentRole.DEV,
            "system_prompt": (
                "You are a Developer agent. Implement code based on the architecture design. "
                "Write all sub-features in a single phase. Generate source code, "
                "configuration files, and database scripts. "
                "Update the Workspace Code partition with file references. "
                "When done, call the handoff tool."
            ),
        },
        {
            "role": AgentRole.REVIEWER,
            "system_prompt": (
                "You are a Code Reviewer agent. Audit code for quality, security, "
                "and convention compliance. Run static analysis tools (ruff, bandit, mypy). "
                "Classify issues by severity: minor, major, or critical. "
                "Write your review report to the Workspace Review partition. "
                "When done, call the handoff tool."
            ),
        },
        {
            "role": AgentRole.QA,
            "system_prompt": (
                "You are a QA agent. Generate and execute tests for the implemented code. "
                "Create unit tests and integration tests using pytest. "
                "Run tests in Docker, capture results and coverage. "
                "Classify failures by severity. "
                "Write your test report to the Workspace Test partition. "
                "When done, call the handoff tool."
            ),
        },
        {
            "role": AgentRole.PMGR,
            "system_prompt": (
                "You are a Project Manager agent (the Host). Orchestrate agent scheduling, "
                "manage handoffs, enforce checkpoints, and track progress. "
                "Decide which agent speaks next based on the current task state. "
                "Present artifacts to the user at checkpoints for confirmation. "
                "When done, call the handoff tool."
            ),
        },
    ]

    for config in role_configs:
        agent_kwargs = {k: v for k, v in kwargs.items()}
        agent_kwargs.update(config)
        registry.register(Agent(**agent_kwargs))

    return registry
