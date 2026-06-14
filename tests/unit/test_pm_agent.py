import asyncio
from unittest.mock import AsyncMock, MagicMock

from coding_assistant.agents.pm_agent import PMAgent
from coding_assistant.core.types import HandoffStatus
from coding_assistant.core.workspace import Workspace


class TestPMAgent:
    def test_create_pm_agent(self):
        agent = PMAgent()
        assert agent.role.value == "pm"
        assert "Product Manager" in agent.system_prompt

    def test_build_context_empty(self):
        agent = PMAgent()
        ws = Workspace(project_name="test")
        context = agent._build_context(ws)
        assert "new project" in context.lower()

    def test_build_context_with_existing(self):
        agent = PMAgent()
        ws = Workspace(project_name="test")
        ws.requirements.prd = "Existing PRD content"
        context = agent._build_context(ws)
        assert "Existing PRD content" in context

    def test_update_workspace(self):
        agent = PMAgent()
        ws = Workspace(project_name="test")
        agent._update_workspace("Generated PRD content", ws)
        assert ws.requirements.prd == "Generated PRD content"
        assert ws.progress.current_phase == "requirements"

    def test_analyze_requirements_no_llm(self):
        agent = PMAgent()
        ws = Workspace(project_name="test")
        result = asyncio.run(agent.analyze_requirements("Build a TODO app", ws))
        assert result.status == HandoffStatus.FAILED
        assert "No LLM client" in result.summary

    def test_analyze_requirements_with_mock_llm(self):
        mock_client = MagicMock()

        mock_response = {
            "content": "PRD for TODO app",
            "tool_calls": [
                {
                    "function": {
                        "name": "handoff",
                        "arguments": '{"status": "completed", "summary": "PRD generated"}',
                    }
                }
            ],
            "error": False,
        }
        mock_client.chat = AsyncMock(return_value=mock_response)
        mock_client.get_model = MagicMock(return_value="gpt-4o")

        agent = PMAgent(llm_client=mock_client)
        ws = Workspace(project_name="test")
        result = asyncio.run(agent.analyze_requirements("Build a TODO app", ws))

        assert result.status == HandoffStatus.COMPLETED
        assert result.summary == "PRD generated"
        assert ws.requirements.prd == "PRD for TODO app"

    def test_analyze_requirements_iteration_mode(self):
        agent = PMAgent()
        ws = Workspace(project_name="test")
        ws.requirements.prd = "Existing PRD"
        ws.requirements.feature_list = [{"name": "auth", "priority": "high"}]
        context = agent._build_context(ws)
        assert "Existing PRD" in context
        assert "auth" in context
