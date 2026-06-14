import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from coding_assistant.agents.dev_agent import DevAgent
from coding_assistant.core.types import AgentRole, HandoffStatus
from coding_assistant.core.workspace import Workspace
from coding_assistant.tools.file_system import FileSystemTool


class TestDevAgent:
    def test_create_dev_agent(self):
        agent = DevAgent()
        assert agent.role == AgentRole.DEV
        assert "Developer" in agent.system_prompt
        assert len(agent.tools) >= 1

    def test_has_handoff_and_fs_tools(self):
        agent = DevAgent()
        tool_names = [t["function"]["name"] for t in agent.tools if t.get("type") == "function"]
        assert "handoff" in tool_names
        assert "write_file" in tool_names
        assert "read_file" in tool_names
        assert "list_dir" in tool_names
        assert "run_shell" in tool_names

    def test_no_architecture_fails(self):
        agent = DevAgent()
        ws = Workspace(project_name="test")
        result = asyncio.run(agent.implement_code(ws))
        assert result.status == HandoffStatus.FAILED
        assert "No architecture" in result.summary

    def test_no_llm_client_fails(self):
        agent = DevAgent()
        ws = Workspace(project_name="test")
        ws.architecture.tech_stack = {"fastapi": "FastAPI"}
        result = asyncio.run(agent.implement_code(ws))
        assert result.status == HandoffStatus.FAILED
        assert "No LLM client" in result.summary

    def test_implement_code_with_mock_llm(self, tmp_path: Path):
        mock_client = MagicMock()
        fs_tool = FileSystemTool(tmp_path)

        code_response = (
            "```python\n"
            "# File: app/main.py\n"
            "from fastapi import FastAPI\n\n"
            "app = FastAPI()\n\n"
            "@app.get('/')\n"
            "def root():\n"
            '    return {"status": "ok"}\n'
            "```\n\n"
            "```python\n"
            "# File: app/models.py\n"
            "from pydantic import BaseModel\n\n"
            "class Item(BaseModel):\n"
            "    name: str\n"
            "    price: float\n"
            "```\n"
        )

        mock_response = {
            "content": code_response,
            "tool_calls": [
                {
                    "function": {
                        "name": "handoff",
                        "arguments": (
                            '{"status": "completed",'
                            ' "summary": "Implemented main.py and models.py",'
                            ' "suggested_next": "reviewer"}'
                        ),
                    }
                }
            ],
            "error": False,
        }
        mock_client.chat = AsyncMock(return_value=mock_response)
        mock_client.get_model = MagicMock(return_value="gpt-4o")

        agent = DevAgent(llm_client=mock_client, project_dir=tmp_path, fs_tool=fs_tool)
        ws = Workspace(project_name="test")
        ws.architecture.tech_stack = {"fastapi": "FastAPI", "pydantic": "Pydantic"}
        ws.architecture.project_structure = "app/\n  main.py\n  models.py"

        result = asyncio.run(agent.implement_code(ws))

        assert result.status == HandoffStatus.COMPLETED
        assert result.summary == "Implemented main.py and models.py"
        assert result.suggested_next == "reviewer"

        assert (tmp_path / "app" / "main.py").exists()
        assert (tmp_path / "app" / "models.py").exists()

        main_content = (tmp_path / "app" / "main.py").read_text()
        assert "FastAPI" in main_content
        assert "app.get" in main_content

        models_content = (tmp_path / "app" / "models.py").read_text()
        assert "BaseModel" in models_content
        assert "Item" in models_content

        assert len(ws.code.files) == 2
        paths = {f.path for f in ws.code.files}
        assert "app/main.py" in paths
        assert "app/models.py" in paths
        assert ws.progress.current_phase == "development"

    def test_implement_code_with_feedback(self, tmp_path: Path):
        mock_client = MagicMock()
        fs_tool = FileSystemTool(tmp_path)

        code_response = (
            "```python\n# File: app/main.py\nfrom fastapi import FastAPI\n\napp = FastAPI()\n```\n"
        )

        mock_response = {
            "content": code_response,
            "tool_calls": [
                {
                    "function": {
                        "name": "handoff",
                        "arguments": '{"status": "completed", "summary": "Fixed auth bug"}',
                    }
                }
            ],
            "error": False,
        }
        mock_client.chat = AsyncMock(return_value=mock_response)
        mock_client.get_model = MagicMock(return_value="gpt-4o")

        agent = DevAgent(llm_client=mock_client, project_dir=tmp_path, fs_tool=fs_tool)
        ws = Workspace(project_name="test")
        ws.architecture.tech_stack = {"fastapi": "FastAPI"}

        feedback = "Fix the authentication bug - JWT token not validated properly"
        result = asyncio.run(agent.implement_code(ws, feedback=feedback))

        assert result.status == HandoffStatus.COMPLETED

        call_args = mock_client.chat.call_args
        messages = call_args.kwargs["messages"]
        full_prompt = messages[1]["content"]
        assert "Reviewer feedback" in full_prompt
        assert "JWT token" in full_prompt

    def test_implement_code_with_existing_files(self, tmp_path: Path):
        mock_client = MagicMock()
        fs_tool = FileSystemTool(tmp_path)

        code_response = "```python\n# File: app/main.py\napp = FastAPI()\n```\n"

        mock_response = {
            "content": code_response,
            "tool_calls": [
                {
                    "function": {
                        "name": "handoff",
                        "arguments": '{"status": "completed", "summary": "Done"}',
                    }
                }
            ],
            "error": False,
        }
        mock_client.chat = AsyncMock(return_value=mock_response)
        mock_client.get_model = MagicMock(return_value="gpt-4o")

        agent = DevAgent(llm_client=mock_client, project_dir=tmp_path, fs_tool=fs_tool)
        ws = Workspace(project_name="test")
        ws.architecture.tech_stack = {"fastapi": "FastAPI"}
        ws.add_file_reference("app/main.py", "Main entry point", AgentRole.DEV)
        ws.add_file_reference("tests/test_app.py", "Tests", AgentRole.QA)

        asyncio.run(agent.implement_code(ws))

        call_args = mock_client.chat.call_args
        messages = call_args.kwargs["messages"]
        full_prompt = messages[1]["content"]
        assert "Existing code files" in full_prompt
        assert "app/main.py" in full_prompt

    def test_generate_documentation(self, tmp_path: Path):
        mock_client = MagicMock()
        fs_tool = FileSystemTool(tmp_path)

        docs_response = (
            "```markdown\n"
            "# File: README.md\n"
            "# My Project\n\nA sample project.\n"
            "```\n\n"
            "```markdown\n"
            "# File: API.md\n"
            "# API Reference\n\n## GET /\nReturns status.\n"
            "```\n"
        )

        mock_response = {
            "content": docs_response,
            "tool_calls": [
                {
                    "function": {
                        "name": "handoff",
                        "arguments": (
                            '{"status": "completed", "summary": "Generated README and API docs"}'
                        ),
                    }
                }
            ],
            "error": False,
        }
        mock_client.chat = AsyncMock(return_value=mock_response)
        mock_client.get_model = MagicMock(return_value="gpt-4o")

        agent = DevAgent(llm_client=mock_client, project_dir=tmp_path, fs_tool=fs_tool)
        ws = Workspace(project_name="test")
        ws.architecture.tech_stack = {"fastapi": "FastAPI"}
        ws.add_file_reference("app/main.py", "Main", AgentRole.DEV)

        result = asyncio.run(agent.generate_documentation(ws))

        assert result.status == HandoffStatus.COMPLETED
        assert (tmp_path / "README.md").exists()
        assert (tmp_path / "API.md").exists()

        readme = (tmp_path / "README.md").read_text()
        assert "My Project" in readme

        assert ws.progress.current_phase == "documentation"

    def test_extract_and_write_files_no_file_markers(self, tmp_path: Path):
        fs_tool = FileSystemTool(tmp_path)
        agent = DevAgent(project_dir=tmp_path, fs_tool=fs_tool)
        ws = Workspace(project_name="test")

        response = "Here is some code without file markers."
        agent._extract_and_write_files(response, ws)

        assert len(ws.code.files) == 0

    def test_format_architecture(self):
        agent = DevAgent()
        ws = Workspace(project_name="test")
        ws.architecture.tech_stack = {"flask": "Flask", "sqlite": "SQLite"}
        ws.architecture.database_schema = "CREATE TABLE items (...)"

        result = agent._format_architecture(ws)
        assert "Flask" in result
        assert "items" in result

    def test_format_architecture_empty(self):
        agent = DevAgent()
        ws = Workspace(project_name="test")
        result = agent._format_architecture(ws)
        assert result == ""

    def test_format_existing_code(self):
        agent = DevAgent()
        ws = Workspace(project_name="test")
        result = agent._format_existing_code(ws)
        assert result == ""

    def test_format_existing_code_with_files(self):
        agent = DevAgent()
        ws = Workspace(project_name="test")
        ws.add_file_reference("app/main.py", "Main app", AgentRole.DEV)
        ws.add_file_reference("app/models.py", "Models", AgentRole.DEV)

        result = agent._format_existing_code(ws)
        assert "app/main.py" in result
        assert "app/models.py" in result
        assert "Existing code files" in result

    def test_implement_code_handles_error_response(self, tmp_path: Path):
        mock_client = MagicMock()

        mock_response = {
            "content": "LLM API call failed after 3 retries: Timeout",
            "tool_calls": [],
            "error": True,
        }
        mock_client.chat = AsyncMock(return_value=mock_response)
        mock_client.get_model = MagicMock(return_value="gpt-4o")

        agent = DevAgent(llm_client=mock_client, project_dir=tmp_path)
        ws = Workspace(project_name="test")
        ws.architecture.tech_stack = {"fastapi": "FastAPI"}

        result = asyncio.run(agent.implement_code(ws))

        assert result.status == HandoffStatus.FAILED
        assert "LLM" in result.summary

    def test_generate_documentation_no_llm(self, tmp_path: Path):
        agent = DevAgent(project_dir=tmp_path)
        ws = Workspace(project_name="test")
        ws.add_file_reference("app/main.py", "Main", AgentRole.DEV)

        result = asyncio.run(agent.generate_documentation(ws))
        assert result.status == HandoffStatus.FAILED
        assert "No LLM client" in result.summary

    def test_get_generated_files(self, tmp_path: Path):
        fs_tool = FileSystemTool(tmp_path)
        agent = DevAgent(project_dir=tmp_path, fs_tool=fs_tool)
        ws = Workspace(project_name="test")

        response = "```python\n# File: app/main.py\napp = FastAPI()\n```\n"
        agent._extract_and_write_files(response, ws)

        files = agent.get_generated_files()
        assert "app/main.py" in files

    def test_find_file_path_with_marker(self):
        agent = DevAgent()
        path = agent._find_file_path("# File: src/app.py\ncontent here")
        assert path == "src/app.py"

    def test_find_file_path_no_marker(self):
        agent = DevAgent()
        path = agent._find_file_path("just code content")
        assert path is None

    def test_write_file_invalid_path(self):
        agent = DevAgent()
        result = agent._write_file("not/a/file", "content")
        assert result is None

    def test_write_file_empty_content(self):
        agent = DevAgent()
        result = agent._write_file("empty.py", "")
        assert result is None
