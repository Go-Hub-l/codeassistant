import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from coding_assistant.agents.reviewer_agent import ReviewerAgent
from coding_assistant.core.types import AgentRole, HandoffStatus, Severity
from coding_assistant.core.workspace import Workspace
from coding_assistant.tools.code_executor import ShellTool
from coding_assistant.tools.file_system import FileSystemTool


class TestReviewerAgent:
    def test_create_reviewer_agent(self):
        agent = ReviewerAgent()
        assert agent.role == AgentRole.REVIEWER
        assert "Reviewer" in agent.system_prompt

    def test_has_handoff_fs_and_shell_tools(self):
        agent = ReviewerAgent()
        tool_names = [t["function"]["name"] for t in agent.tools if t.get("type") == "function"]
        assert "handoff" in tool_names
        assert "read_file" in tool_names
        assert "run_shell" in tool_names

    def test_no_code_files_fails(self):
        agent = ReviewerAgent()
        ws = Workspace(project_name="test")
        result = asyncio.run(agent.review_code(ws))
        assert result.status == HandoffStatus.FAILED
        assert "No code files" in result.summary

    def test_no_llm_client_fails(self, tmp_path: Path):
        fs_tool = FileSystemTool(tmp_path)
        (tmp_path / "app").mkdir(exist_ok=True)
        (tmp_path / "app" / "main.py").write_text("print('hello')")
        agent = ReviewerAgent(project_dir=tmp_path, fs_tool=fs_tool)
        ws = Workspace(project_name="test")
        ws.add_file_reference("app/main.py", "Main", AgentRole.DEV)
        result = asyncio.run(agent.review_code(ws))
        assert result.status == HandoffStatus.FAILED
        assert "No LLM client" in result.summary

    def test_review_code_with_mock_llm(self, tmp_path: Path):
        mock_client = MagicMock()
        fs_tool = FileSystemTool(tmp_path)
        shell_tool = ShellTool(tmp_path)

        (tmp_path / "app").mkdir(exist_ok=True)
        (tmp_path / "app" / "main.py").write_text(
            "from fastapi import FastAPI\n\n"
            "app = FastAPI()\n\n"
            "@app.get('/')\n"
            "def root():\n"
            "    return {'status': 'ok'}\n"
        )

        mock_response = {
            "content": "Code review complete. No issues found.",
            "tool_calls": [
                {
                    "function": {
                        "name": "handoff",
                        "arguments": (
                            '{"status": "completed",'
                            ' "summary": "Code passes review - no issues",'
                            ' "severity": null, "suggested_next": "qa"}'
                        ),
                    }
                }
            ],
            "error": False,
        }
        mock_client.chat = AsyncMock(return_value=mock_response)
        mock_client.get_model = MagicMock(return_value="gpt-4o-mini")

        agent = ReviewerAgent(
            llm_client=mock_client, project_dir=tmp_path, fs_tool=fs_tool, shell_tool=shell_tool
        )
        ws = Workspace(project_name="test")
        ws.add_file_reference("app/main.py", "Main app", AgentRole.DEV)

        result = asyncio.run(agent.review_code(ws))

        assert result.status == HandoffStatus.COMPLETED
        assert result.summary == "Code passes review - no issues"
        assert result.suggested_next == "qa"

        assert len(ws.review.issues) >= 0

        call_args = mock_client.chat.call_args
        messages = call_args.kwargs["messages"]
        full_prompt = messages[1]["content"]
        assert "app/main.py" in full_prompt

    def test_security_checks_detect_secrets(self, tmp_path: Path):
        fs_tool = FileSystemTool(tmp_path)
        (tmp_path / "app").mkdir(exist_ok=True)
        (tmp_path / "app" / "config.py").write_text(
            "API_KEY = 'sk-abc123def456'\npassword = 'hardcoded'\n"
        )
        agent = ReviewerAgent(project_dir=tmp_path, fs_tool=fs_tool)
        ws = Workspace(project_name="test")
        ws.add_file_reference("app/config.py", "Config", AgentRole.DEV)

        security_issues = agent._run_security_checks(ws.code.files)

        assert len(security_issues) >= 1
        secrets = [
            i
            for i in security_issues
            if "secret" in i["description"].lower() or "credential" in i["description"].lower()
        ]
        assert len(secrets) >= 1
        assert any(i["severity"] == "critical" for i in security_issues)

    def test_security_checks_detect_unsafe_calls(self, tmp_path: Path):
        fs_tool = FileSystemTool(tmp_path)
        (tmp_path / "app").mkdir(exist_ok=True)
        (tmp_path / "app" / "dangerous.py").write_text(
            "import os\nos.system('rm -rf /')\neval('1+1')\nexec('print(1)')\n"
        )
        agent = ReviewerAgent(project_dir=tmp_path, fs_tool=fs_tool)
        ws = Workspace(project_name="test")
        ws.add_file_reference("app/dangerous.py", "Dangerous code", AgentRole.DEV)

        issues = agent._run_security_checks(ws.code.files)

        assert len(issues) >= 3
        critical_issues = [i for i in issues if i["severity"] == "critical"]
        assert len(critical_issues) >= 3

    def test_security_checks_detect_sql_injection(self, tmp_path: Path):
        fs_tool = FileSystemTool(tmp_path)
        (tmp_path / "app").mkdir(exist_ok=True)
        (tmp_path / "app" / "db.py").write_text(
            "query = f\"SELECT * FROM users WHERE id = '{user_id}'\"\n"
            "cursor.execute('SELECT * FROM items WHERE name = %s' % name)\n"
        )
        agent = ReviewerAgent(project_dir=tmp_path, fs_tool=fs_tool)
        ws = Workspace(project_name="test")
        ws.add_file_reference("app/db.py", "Database", AgentRole.DEV)

        issues = agent._run_security_checks(ws.code.files)

        sql_issues = [i for i in issues if "SQL injection" in i["description"]]
        assert len(sql_issues) >= 1

    def test_read_code_files(self, tmp_path: Path):
        fs_tool = FileSystemTool(tmp_path)
        (tmp_path / "app").mkdir(exist_ok=True)
        (tmp_path / "app" / "main.py").write_text("print('hello world')")
        agent = ReviewerAgent(project_dir=tmp_path, fs_tool=fs_tool)
        ws = Workspace(project_name="test")
        ws.add_file_reference("app/main.py", "Main", AgentRole.DEV)

        content = agent._read_code_files(ws.code.files)
        assert "app/main.py" in content
        assert "hello world" in content

    def test_update_workspace_with_issues(self):
        agent = ReviewerAgent()
        ws = Workspace(project_name="test")

        security_issues = [
            {
                "file": "app/main.py",
                "line": 5,
                "severity": "critical",
                "description": "Hardcoded secret",
            }
        ]
        tool_results = [
            {"file": "app/main.py", "line": 3, "severity": "minor", "description": "Unused import"}
        ]
        agent._update_workspace("Review complete", security_issues, tool_results, ws)

        assert len(ws.review.issues) == 2
        assert ws.review.severity == "critical"
        assert ws.review.summary == "Review complete"
        assert ws.progress.current_phase == "review"

    def test_update_workspace_major_only(self):
        agent = ReviewerAgent()
        ws = Workspace(project_name="test")
        tool_results = [
            {
                "file": "app/main.py",
                "line": 10,
                "severity": "major",
                "description": "Missing error handling",
            }
        ]
        agent._update_workspace("Review", [], tool_results, ws)
        assert ws.review.severity == "major"

    def test_update_workspace_minor_only(self):
        agent = ReviewerAgent()
        ws = Workspace(project_name="test")
        tool_results = [
            {
                "file": "app/main.py",
                "line": 2,
                "severity": "minor",
                "description": "Missing docstring",
            }
        ]
        agent._update_workspace("Review", [], tool_results, ws)
        assert ws.review.severity == "minor"

    def test_get_severity_from_summary(self):
        agent = ReviewerAgent()
        assert agent.get_severity_from_summary("Critical vulnerability found") == Severity.CRITICAL
        assert agent.get_severity_from_summary("Major logic error") == Severity.MAJOR
        assert agent.get_severity_from_summary("Minor style issue") == Severity.MINOR
        assert agent.get_severity_from_summary("All good") is None

    def test_parse_ruff_output(self):
        agent = ReviewerAgent()
        output = (
            "app/main.py:3:1: F401 'os' imported but unused\n"
            "app/models.py:10:5: E501 Line too long\n"
        )
        issues = agent._parse_ruff_output(output)
        assert len(issues) == 2
        assert issues[0]["file"] == "app/main.py"
        assert issues[0]["severity"] == "minor"
        assert issues[1]["severity"] == "major"

    def test_parse_bandit_output(self):
        agent = ReviewerAgent()
        output = (
            ">> Issue: [B105:hardcoded_password_string] Possible hardcoded password\n"
            ">> Issue: [B301:blacklist] Pickle usage\n"
        )
        issues = agent._parse_bandit_output(output)
        assert len(issues) == 2
        assert issues[0]["source"] == "bandit"

    def test_parse_mypy_output(self):
        agent = ReviewerAgent()
        output = (
            "app/main.py:1: error: Missing return statement\n"
            "app/models.py:5: warning: Unused type hint\n"
        )
        issues = agent._parse_mypy_output(output)
        assert len(issues) == 2
        assert issues[0]["severity"] == "major"
        assert issues[1]["severity"] == "minor"
