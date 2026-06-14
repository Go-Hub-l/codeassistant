import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from coding_assistant.agents.architect_agent import ArchitectAgent
from coding_assistant.agents.base import Agent
from coding_assistant.agents.dev_agent import DevAgent
from coding_assistant.agents.pm_agent import PMAgent
from coding_assistant.agents.registry import AgentRegistry, create_default_registry
from coding_assistant.core.types import AgentRole, HandoffStatus, Severity


class TestAgentBase:
    def test_create_agent_with_defaults(self):
        agent = Agent(role=AgentRole.PM)
        assert agent.role == AgentRole.PM
        assert agent.system_prompt  # not empty
        assert agent.tools == []
        assert agent.llm_client is None
        assert agent.model is None

    def test_create_agent_with_custom_prompt(self):
        agent = Agent(role=AgentRole.DEV, system_prompt="Custom prompt")
        assert agent.system_prompt == "Custom prompt"

    def test_conversation_history(self):
        agent = Agent(role=AgentRole.PM)
        agent.add_message("user", "Hello")
        agent.add_message("assistant", "Hi there")

        history = agent.get_conversation_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"

    def test_clear_conversation_history(self):
        agent = Agent(role=AgentRole.PM)
        agent.add_message("user", "Hello")
        agent.clear_conversation_history()
        assert len(agent.get_conversation_history()) == 0

    def test_run_without_llm_client_returns_failed(self):
        import asyncio

        agent = Agent(role=AgentRole.PM)
        result = asyncio.run(agent.run("test input"))
        assert result.status == HandoffStatus.FAILED
        assert "No LLM client" in result.summary

    def test_run_with_llm_error_returns_failed(self):
        mock_client = MagicMock()
        mock_response = {
            "content": "LLM API call failed after 3 retries: Connection timeout",
            "tool_calls": [],
            "error": True,
        }
        mock_client.chat = AsyncMock(return_value=mock_response)

        agent = Agent(role=AgentRole.PM, llm_client=mock_client)
        result = asyncio.run(agent.run("test input"))
        assert result.status == HandoffStatus.FAILED
        assert "LLM" in result.summary

    def test_extract_handoff_from_tool_calls(self):
        agent = Agent(role=AgentRole.PM)
        args = '{"status": "completed", "summary": "Done", "suggested_next": "architect"}'
        tool_calls = [{"function": {"name": "handoff", "arguments": args}}]
        result = agent._extract_handoff(tool_calls)
        assert result is not None
        assert result.status == HandoffStatus.COMPLETED
        assert result.summary == "Done"
        assert result.suggested_next == "architect"

    def test_extract_handoff_with_severity(self):
        agent = Agent(role=AgentRole.REVIEWER)
        args = '{"status": "completed", "summary": "Issues found", "severity": "critical"}'
        tool_calls = [{"function": {"name": "handoff", "arguments": args}}]
        result = agent._extract_handoff(tool_calls)
        assert result is not None
        assert result.severity == Severity.CRITICAL

    def test_extract_handoff_no_handoff_call(self):
        agent = Agent(role=AgentRole.PM)
        tool_calls = [
            {
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "test.py", "content": "hello"}',
                }
            }
        ]
        result = agent._extract_handoff(tool_calls)
        assert result is None

    def test_extract_handoff_invalid_json(self):
        agent = Agent(role=AgentRole.PM)
        tool_calls = [
            {
                "function": {
                    "name": "handoff",
                    "arguments": "invalid json{{{",
                }
            }
        ]
        result = agent._extract_handoff(tool_calls)
        assert result is None

    def test_build_handoff_tool_schema(self):
        agent = Agent(role=AgentRole.PM)
        schema = agent.build_handoff_tool_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "handoff"
        assert "status" in schema["function"]["parameters"]["properties"]
        assert "summary" in schema["function"]["parameters"]["properties"]
        assert "suggested_next" in schema["function"]["parameters"]["properties"]
        assert "severity" in schema["function"]["parameters"]["properties"]


class TestAgentRegistry:
    def test_register_and_get(self):
        registry = AgentRegistry()
        agent = Agent(role=AgentRole.PM, system_prompt="PM prompt")
        registry.register(agent)

        retrieved = registry.get(AgentRole.PM)
        assert retrieved.role == AgentRole.PM
        assert retrieved.system_prompt == "PM prompt"

    def test_register_duplicate_raises(self):
        registry = AgentRegistry()
        registry.register(Agent(role=AgentRole.PM))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(Agent(role=AgentRole.PM))

    def test_get_unregistered_raises(self):
        registry = AgentRegistry()
        with pytest.raises(KeyError, match="No agent registered"):
            registry.get(AgentRole.ARCHITECT)

    def test_has(self):
        registry = AgentRegistry()
        assert not registry.has(AgentRole.PM)
        registry.register(Agent(role=AgentRole.PM))
        assert registry.has(AgentRole.PM)

    def test_list_roles(self):
        registry = AgentRegistry()
        registry.register(Agent(role=AgentRole.PM))
        registry.register(Agent(role=AgentRole.DEV))
        roles = registry.list_roles()
        assert AgentRole.PM in roles
        assert AgentRole.DEV in roles

    def test_all_agents(self):
        registry = AgentRegistry()
        registry.register(Agent(role=AgentRole.PM))
        registry.register(Agent(role=AgentRole.DEV))
        agents = registry.all_agents()
        assert len(agents) == 2


class TestCreateDefaultRegistry:
    def test_creates_all_six_agents(self):
        registry = create_default_registry()
        assert registry.has(AgentRole.PM)
        assert registry.has(AgentRole.ARCHITECT)
        assert registry.has(AgentRole.DEV)
        assert registry.has(AgentRole.REVIEWER)
        assert registry.has(AgentRole.QA)
        assert registry.has(AgentRole.PMGR)

    def test_each_agent_has_system_prompt(self):
        registry = create_default_registry()
        for role in AgentRole:
            agent = registry.get(role)
            assert len(agent.system_prompt) > 50

    def test_pass_llm_client(self):
        registry = create_default_registry(llm_client="mock_client")
        for role in AgentRole:
            agent = registry.get(role)
            assert agent.llm_client == "mock_client"

    def test_registry_uses_specialized_agent_classes(self):
        registry = create_default_registry()
        pm_agent = registry.get(AgentRole.PM)
        architect_agent = registry.get(AgentRole.ARCHITECT)
        dev_agent = registry.get(AgentRole.DEV)

        assert isinstance(pm_agent, PMAgent)
        assert isinstance(architect_agent, ArchitectAgent)
        assert isinstance(dev_agent, DevAgent)
        assert not isinstance(dev_agent, PMAgent)
