import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coding_assistant.cli.main import CodingAssistantSession


@pytest.mark.e2e
@pytest.mark.skip(reason="E2E requires Host checkpoint loop fix for infinite re-check")
class TestE2EPipeline:
    """End-to-end tests exercising the full pipeline with mocked LLM."""

    def _make_mock_llm(self):
        mock = MagicMock()
        mock.chat = AsyncMock()
        mock.get_model = MagicMock(return_value="gpt-4o-mini")
        return mock

    def _handoff(
        self, status="completed", summary="Done", suggested_next="architect", severity=None
    ):
        sev = f', "severity": "{severity}"' if severity else ""
        args = (
            f'{{"status": "{status}",'
            f' "summary": "{summary}"{sev},'
            f' "suggested_next": "{suggested_next}"}}'
        )
        return {"function": {"name": "handoff", "arguments": args}}

    def _pm_response(self):
        return {
            "content": "## PRD\nBuild a simple TODO API with CRUD operations.",
            "tool_calls": [self._handoff("completed", "PRD done", "architect")],
            "error": False,
        }

    def _architect_response(self):
        return {
            "content": "## Technology Stack\n- FastAPI\n- SQLite\n## API\nGET /tasks, POST /tasks",
            "tool_calls": [self._handoff("completed", "Architecture done", "dev")],
            "error": False,
        }

    def _dev_response(self):
        return {
            "content": (
                "```python\n"
                "# File: app/main.py\n"
                "from fastapi import FastAPI\n\napp = FastAPI()\n"
                "@app.get('/tasks')\ndef list_tasks():\n    return []\n"
                "```\n"
            ),
            "tool_calls": [self._handoff("completed", "Code done", "reviewer")],
            "error": False,
        }

    def _reviewer_response(self):
        return {
            "content": "Code review complete.",
            "tool_calls": [self._handoff("completed", "review ok", "qa")],
            "error": False,
        }

    def _qa_response(self):
        content = (
            "```python\n"
            "# File: tests/test_app.py\n"
            "import pytest\n"
            "def test_pass():\n"
            "    assert True\n"
            "```\n"
        )
        return {
            "content": content,
            "tool_calls": [self._handoff("completed", "Tests done", "documentation")],
            "error": False,
        }

    def _docs_response(self):
        return {
            "content": "```markdown\n# File: README.md\n# TODO API\n```\n",
            "tool_calls": [self._handoff("completed", "Docs done", "reviewer")],
            "error": False,
        }

    def _create_session(self, tmp_path, llm_client, monkeypatch, is_iteration=False):
        monkeypatch.setattr("builtins.input", lambda prompt="": "ok")
        monkeypatch.setattr("coding_assistant.cli.main.resolve_api_key", lambda: "mock-key")
        monkeypatch.setattr("coding_assistant.cli.main.signal.signal", lambda *a: None)

        with (
            patch("coding_assistant.cli.main.LLMClient", return_value=llm_client),
            patch("coding_assistant.cli.main.GitTool", autospec=True),
        ):
            return CodingAssistantSession(
                project_dir=tmp_path,
                project_name="e2e-test",
                llm_client=llm_client,
                is_iteration=is_iteration,
            )

    def test_full_pipeline_new_project(self, tmp_path: Path, monkeypatch):
        mock_llm = self._make_mock_llm()
        mock_llm.chat.side_effect = [
            self._pm_response(),
            self._architect_response(),
            self._dev_response(),
            self._reviewer_response(),
            self._qa_response(),
            self._docs_response(),
        ]
        session = self._create_session(tmp_path, mock_llm, monkeypatch)
        asyncio.run(session.run("Build a simple TODO API"))
        ws = session.workspace_manager.workspace
        assert ws.requirements.prd != ""
        assert ws.architecture.tech_stack != {}
        assert len(ws.code.files) >= 1

    def test_pipeline_handles_retry(self, tmp_path: Path, monkeypatch):
        mock_llm = self._make_mock_llm()
        mock_llm.chat.side_effect = [
            self._pm_response(),
            self._architect_response(),
            {
                "content": "Code done",
                "tool_calls": [
                    self._handoff("completed", "Code with minor issue", "reviewer", "minor")
                ],
                "error": False,
            },
            self._dev_response(),
            self._reviewer_response(),
            self._qa_response(),
            self._docs_response(),
        ]
        session = self._create_session(tmp_path, mock_llm, monkeypatch)
        asyncio.run(session.run("Build a simple TODO API"))
        assert len(session.workspace_manager.workspace.code.files) >= 1

    def test_pipeline_handles_llm_error(self, tmp_path: Path, monkeypatch):
        mock_llm = self._make_mock_llm()
        mock_llm.chat.side_effect = [
            self._pm_response(),
            self._architect_response(),
            {"content": "LLM API failed: Timeout", "tool_calls": [], "error": True},
            self._dev_response(),
            self._reviewer_response(),
            self._qa_response(),
            self._docs_response(),
        ]
        session = self._create_session(tmp_path, mock_llm, monkeypatch)
        asyncio.run(session.run("Build a simple TODO API"))
        assert len(session.workspace_manager.workspace.code.files) >= 1

    def test_pipeline_generates_docs(self, tmp_path: Path, monkeypatch):
        mock_llm = self._make_mock_llm()
        mock_llm.chat.side_effect = [
            self._pm_response(),
            self._architect_response(),
            self._dev_response(),
            self._reviewer_response(),
            self._qa_response(),
            self._docs_response(),
        ]
        session = self._create_session(tmp_path, mock_llm, monkeypatch)
        asyncio.run(session.run("Build a simple TODO API"))
        assert (tmp_path / "README.md").exists()
