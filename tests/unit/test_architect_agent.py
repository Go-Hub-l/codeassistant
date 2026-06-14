import asyncio
from unittest.mock import AsyncMock, MagicMock

from coding_assistant.agents.architect_agent import ArchitectAgent
from coding_assistant.core.types import HandoffStatus
from coding_assistant.core.workspace import Workspace


class TestArchitectAgent:
    def test_create_architect_agent(self):
        agent = ArchitectAgent()
        assert agent.role.value == "architect"
        assert "Architect" in agent.system_prompt

    def test_no_requirements_fails(self):
        agent = ArchitectAgent()
        ws = Workspace(project_name="test")
        result = asyncio.run(agent.design_architecture(ws))
        assert result.status == HandoffStatus.FAILED
        assert "No requirements" in result.summary

    def test_no_llm_client_fails(self):
        agent = ArchitectAgent()
        ws = Workspace(project_name="test")
        ws.requirements.prd = "Build a TODO app"
        result = asyncio.run(agent.design_architecture(ws))
        assert result.status == HandoffStatus.FAILED
        assert "No LLM client" in result.summary

    def test_design_architecture_with_mock_llm(self):
        mock_client = MagicMock()

        arch_response = (
            "## Technology Stack\n"
            "- FastAPI: async web framework\n"
            "- PostgreSQL: relational database\n"
            "- SQLAlchemy: ORM\n"
            "- Pydantic: data validation\n\n"
            "## Project Directory Structure\n"
            "```\n"
            "src/\n"
            "  api/\n"
            "  models/\n"
            "  services/\n"
            "  schemas/\n"
            "tests/\n"
            "```\n\n"
            "## API Contracts\n"
            "GET /api/tasks - List tasks\n"
            "POST /api/tasks - Create task\n"
            "PUT /api/tasks/{id} - Update task\n"
            "DELETE /api/tasks/{id} - Delete task\n\n"
            "## Database Schema\n"
            "CREATE TABLE tasks (id UUID PRIMARY KEY, title TEXT, done BOOLEAN)\n"
            "CREATE TABLE users (id UUID PRIMARY KEY, email TEXT UNIQUE)\n\n"
            "## Security Considerations\n"
            "Use JWT auth, validate all inputs with Pydantic, rate limiting"
        )

        mock_response = {
            "content": arch_response,
            "tool_calls": [
                {
                    "function": {
                        "name": "handoff",
                        "arguments": (
                            '{"status": "completed",'
                            ' "summary": "Architecture designed for TODO app",'
                            ' "suggested_next": "dev"}'
                        ),
                    }
                }
            ],
            "error": False,
        }
        mock_client.chat = AsyncMock(return_value=mock_response)
        mock_client.get_model = MagicMock(return_value="gpt-4o")

        agent = ArchitectAgent(llm_client=mock_client)
        ws = Workspace(project_name="test")
        ws.requirements.prd = "Build a TODO app with user auth"

        result = asyncio.run(agent.design_architecture(ws))

        assert result.status == HandoffStatus.COMPLETED
        assert result.summary == "Architecture designed for TODO app"
        assert result.suggested_next == "dev"

        assert "fastapi" in ws.architecture.tech_stack
        assert ws.architecture.tech_stack["fastapi"] == "FastAPI"
        assert "postgresql" in ws.architecture.tech_stack
        assert "sqlalchemy" in ws.architecture.tech_stack
        assert "pydantic" in ws.architecture.tech_stack
        assert "src/" in ws.architecture.project_structure

        assert len(ws.architecture.api_contracts) >= 1
        methods = [c["method"] for c in ws.architecture.api_contracts]
        assert "GET" in methods or "GET" in str(ws.architecture.api_contracts)

        assert "tasks" in ws.architecture.database_schema.lower()
        assert "JWT" in ws.architecture.security_considerations
        assert ws.progress.current_phase == "architecture"

    def test_extract_tech_stack_detects_frameworks(self):
        agent = ArchitectAgent()
        result = agent._extract_tech_stack(
            "We will use FastAPI with SQLAlchemy and PostgreSQL. Pytest for tests."
        )
        assert "fastapi" in result
        assert result["fastapi"] == "FastAPI"
        assert "sqlalchemy" in result
        assert "postgresql" in result
        assert "pytest" in result

    def test_extract_tech_stack_returns_none_for_empty(self):
        agent = ArchitectAgent()
        result = agent._extract_tech_stack("Just use whatever framework")
        assert result is None

    def test_extract_api_contracts_from_text(self):
        agent = ArchitectAgent()
        text = (
            "API Endpoints:\n"
            "GET /api/users - List all users\n"
            "POST /api/users - Create a new user\n"
            "GET /api/users/{id} - Get user by ID\n"
            "DELETE /api/users/{id} - Delete user\n"
        )
        contracts = agent._extract_api_contracts(text)
        assert len(contracts) == 4
        assert contracts[0]["method"] == "GET"
        assert contracts[0]["path"] == "/api/users"
        assert contracts[1]["method"] == "POST"

    def test_format_existing_architecture(self):
        agent = ArchitectAgent()
        ws = Workspace(project_name="test")
        ws.architecture.tech_stack = {"fastapi": "FastAPI"}
        ws.architecture.project_structure = "src/\n  app.py"
        ws.architecture.api_contracts = [
            {"method": "GET", "path": "/api/items", "description": "List items"}
        ]

        result = agent._format_existing_architecture(ws)
        assert "FastAPI" in result
        assert "app.py" in result
        assert "api/items" in result

    def test_format_existing_architecture_empty(self):
        agent = ArchitectAgent()
        ws = Workspace(project_name="test")
        result = agent._format_existing_architecture(ws)
        assert result == ""

    def test_update_workspace_with_tech_stack(self):
        agent = ArchitectAgent()
        ws = Workspace(project_name="test")
        response = (
            "## Technology Stack\n"
            "Using Flask and SQLite\n\n"
            "## Project Directory Structure\n"
            "```\napp/\n  __init__.py\n  routes.py\n  models.py\n```\n\n"
            "## API Contracts\n"
            "GET / - Home page\n"
            "POST /submit - Submit form\n\n"
            "## Database Schema\n"
            "CREATE TABLE items (id INT, name TEXT)\n\n"
            "## Security\n"
            "CSRF protection, input sanitization\n"
        )
        agent._update_workspace(response, ws)

        assert "flask" in ws.architecture.tech_stack
        assert "sqlite" in ws.architecture.tech_stack
        assert ws.architecture.project_structure != ""
        assert "routes.py" in ws.architecture.project_structure
        assert len(ws.architecture.api_contracts) >= 1
        assert "items" in ws.architecture.database_schema.lower()
        assert "CSRF" in ws.architecture.security_considerations
        assert ws.progress.current_phase == "architecture"

    def test_design_architecture_handles_error_response(self):
        mock_client = MagicMock()

        mock_response = {
            "content": "LLM API call failed after 3 retries",
            "tool_calls": [],
            "error": True,
        }
        mock_client.chat = AsyncMock(return_value=mock_response)
        mock_client.get_model = MagicMock(return_value="gpt-4o")

        agent = ArchitectAgent(llm_client=mock_client)
        ws = Workspace(project_name="test")
        ws.requirements.prd = "Build a TODO app"

        result = asyncio.run(agent.design_architecture(ws))

        assert result.status == HandoffStatus.FAILED
        assert "LLM" in result.summary

    def test_design_architecture_with_user_feedback(self):
        mock_client = MagicMock()

        arch_response = "## Technology Stack\nUsing FastAPI and PostgreSQL"
        mock_response = {
            "content": arch_response,
            "tool_calls": [
                {
                    "function": {
                        "name": "handoff",
                        "arguments": '{"status": "completed", "summary": "Revised architecture"}',
                    }
                }
            ],
            "error": False,
        }
        mock_client.chat = AsyncMock(return_value=mock_response)
        mock_client.get_model = MagicMock(return_value="gpt-4o")

        agent = ArchitectAgent(llm_client=mock_client)
        ws = Workspace(project_name="test")
        ws.requirements.prd = "Build a TODO app"

        result = asyncio.run(
            agent.design_architecture(ws, user_feedback="Please use PostgreSQL instead of SQLite")
        )

        assert result.status == HandoffStatus.COMPLETED

        call_args = mock_client.chat.call_args
        messages = call_args.kwargs["messages"]
        full_prompt = messages[1]["content"]
        assert "PostgreSQL instead of SQLite" in full_prompt
