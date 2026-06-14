import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from coding_assistant.agents.qa_agent import QAAgent
from coding_assistant.core.types import AgentRole, HandoffStatus
from coding_assistant.core.workspace import Workspace
from coding_assistant.tools.code_executor import ShellTool
from coding_assistant.tools.file_system import FileSystemTool


class TestQAAgent:
    def test_create_qa_agent(self):
        agent = QAAgent()
        assert agent.role == AgentRole.QA
        assert "QA" in agent.system_prompt

    def test_has_handoff_fs_and_shell_tools(self):
        agent = QAAgent()
        tool_names = [t["function"]["name"] for t in agent.tools if t.get("type") == "function"]
        assert "handoff" in tool_names
        assert "read_file" in tool_names
        assert "run_shell" in tool_names

    def test_no_code_files_fails(self):
        agent = QAAgent()
        ws = Workspace(project_name="test")
        result = asyncio.run(agent.test_code(ws))
        assert result.status == HandoffStatus.FAILED
        assert "No code files" in result.summary

    def test_no_llm_client_fails(self, tmp_path: Path):
        fs_tool = FileSystemTool(tmp_path)
        (tmp_path / "app").mkdir(exist_ok=True)
        (tmp_path / "app" / "main.py").write_text("print('hello')")
        agent = QAAgent(project_dir=tmp_path, fs_tool=fs_tool)
        ws = Workspace(project_name="test")
        ws.add_file_reference("app/main.py", "Main", AgentRole.DEV)
        result = asyncio.run(agent.test_code(ws))
        assert result.status == HandoffStatus.FAILED
        assert "No LLM client" in result.summary

    def test_test_code_with_mock_llm(self, tmp_path: Path):
        mock_client = MagicMock()
        fs_tool = FileSystemTool(tmp_path)
        shell_tool = ShellTool(tmp_path)

        (tmp_path / "app").mkdir(exist_ok=True)
        (tmp_path / "app" / "main.py").write_text("def add(a, b):\n    return a + b\n")

        test_response = (
            "```python\n"
            "# File: tests/test_add.py\n"
            "import pytest\n"
            "from app.main import add\n\n"
            "def test_add_positive():\n"
            "    assert add(1, 2) == 3\n\n"
            "def test_add_negative():\n"
            "    assert add(-1, -2) == -3\n"
            "```\n"
        )

        mock_response = {
            "content": test_response,
            "tool_calls": [
                {
                    "function": {
                        "name": "handoff",
                        "arguments": (
                            '{"status": "completed",'
                            ' "summary": "Generated tests for add function",'
                            ' "suggested_next": "documentation"}'
                        ),
                    }
                }
            ],
            "error": False,
        }
        mock_client.chat = AsyncMock(return_value=mock_response)
        mock_client.get_model = MagicMock(return_value="gpt-4o-mini")

        agent = QAAgent(
            llm_client=mock_client, project_dir=tmp_path, fs_tool=fs_tool, shell_tool=shell_tool
        )
        ws = Workspace(project_name="test")
        ws.add_file_reference("app/main.py", "Main module", AgentRole.DEV)

        result = asyncio.run(agent.test_code(ws))

        assert result.status == HandoffStatus.COMPLETED
        assert result.summary == "Generated tests for add function"
        assert result.suggested_next == "documentation"

        assert (tmp_path / "tests" / "test_add.py").exists()
        content = (tmp_path / "tests" / "test_add.py").read_text()
        assert "test_add_positive" in content
        assert "def add" not in content

    def test_parse_pytest_output_passed(self):
        agent = QAAgent()
        output = (
            "tests/test_app.py::test_one PASSED\ntests/test_app.py::test_two PASSED\n2 passed\n"
        )
        result = agent._parse_pytest_output(output)
        assert result["status"] == "passed"
        assert result["passed"] == 2
        assert result["failed"] == 0

    def test_parse_pytest_output_with_failures(self):
        agent = QAAgent()
        output = (
            "tests/test_app.py::test_one PASSED\n"
            "tests/test_app.py::test_two FAILED\n"
            "FAILED tests/test_app.py::test_two - assert False\n"
            "1 passed, 1 failed\n"
        )
        result = agent._parse_pytest_output(output)
        assert result["status"] == "failed"
        assert result["passed"] == 1
        assert result["failed"] == 1

    def test_parse_pytest_output_empty(self):
        agent = QAAgent()
        result = agent._parse_pytest_output("")
        assert result["status"] == "unknown"
        assert result["total"] == 0

    def test_classify_severity_critical(self):
        agent = QAAgent()
        assert agent._classify_severity({"error": 1, "failed": 0, "passed": 5}) == "critical"
        assert agent._classify_severity({"error": 3, "failed": 0, "passed": 5}) == "critical"

    def test_classify_severity_major(self):
        agent = QAAgent()
        assert agent._classify_severity({"error": 0, "failed": 3, "passed": 4}) == "major"

    def test_classify_severity_minor(self):
        agent = QAAgent()
        assert agent._classify_severity({"error": 0, "failed": 1, "passed": 10}) == "minor"

    def test_classify_severity_none(self):
        agent = QAAgent()
        assert agent._classify_severity({"error": 0, "failed": 0, "passed": 5}) is None

    def test_update_workspace(self):
        agent = QAAgent()
        ws = Workspace(project_name="test")
        test_results = {
            "status": "passed",
            "passed": 5,
            "failed": 0,
            "error": 0,
            "skipped": 0,
            "total": 5,
            "summary": "5 passed",
            "failures": [],
        }
        agent._update_workspace("Test report summary", test_results, ws)
        assert ws.test.summary == "Test report summary"
        assert ws.test.results["passed"] == 5

    def test_update_workspace_with_failures(self):
        agent = QAAgent()
        ws = Workspace(project_name="test")
        test_results = {
            "status": "failed",
            "passed": 10,
            "failed": 1,
            "error": 0,
            "skipped": 0,
            "total": 11,
            "summary": "10 passed, 1 failed",
            "failures": [],
        }
        agent._update_workspace("Tests failed", test_results, ws)
        assert ws.test.severity == "minor"

    def test_read_code_for_testing(self, tmp_path: Path):
        fs_tool = FileSystemTool(tmp_path)
        (tmp_path / "app").mkdir(exist_ok=True)
        (tmp_path / "app" / "main.py").write_text("def add(a, b):\n    return a + b\n")
        agent = QAAgent(project_dir=tmp_path, fs_tool=fs_tool)
        ws = Workspace(project_name="test")
        ws.add_file_reference("app/main.py", "Main", AgentRole.DEV)

        content = agent._read_code_for_testing(ws.code.files)
        assert "app/main.py" in content
        assert "def add" in content

    def test_find_test_path(self):
        agent = QAAgent()
        assert agent._find_test_path("# File: tests/test_app.py\ncode") == "tests/test_app.py"
        assert agent._find_test_path("just code") is None

    def test_clean_test_content(self):
        agent = QAAgent()
        content = "python\n# File: tests/test_app.py\nimport pytest\n\ndef test_one():\n    pass\n"
        cleaned = agent._clean_test_content(content)
        assert "# File:" not in cleaned
        assert "import pytest" in cleaned
        assert "python" not in cleaned.split("\n")[0].strip()
