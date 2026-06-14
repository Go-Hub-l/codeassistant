from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from coding_assistant.agents.base import Agent
from coding_assistant.core.types import AgentRole, HandoffResult, HandoffStatus
from coding_assistant.core.workspace import Workspace
from coding_assistant.llm.client import LLMClient
from coding_assistant.llm.templates import PromptTemplateManager
from coding_assistant.tools.code_executor import ShellTool
from coding_assistant.tools.file_system import FileSystemTool

logger = logging.getLogger(__name__)

PYTEST_RESULT_PATTERN = re.compile(r"(\d+)\s+(passed|failed|error|skipped)")
PYTEST_SUMMARY_PATTERN = re.compile(r"(failed|FAILED|error|ERROR).*$", re.MULTILINE)
COVERAGE_LINE = re.compile(r"TOTAL\s+\d+\s+\d+\s+(\d+)%")


class QAAgent(Agent):
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str | None = None,
        project_dir: Path | None = None,
        fs_tool: FileSystemTool | None = None,
        shell_tool: ShellTool | None = None,
    ) -> None:
        template_mgr = PromptTemplateManager()
        prompt = template_mgr.render(AgentRole.QA, context="", code="", architecture="")
        self.project_dir = project_dir or Path(".")
        self.fs_tool = fs_tool or FileSystemTool(self.project_dir)
        self.shell_tool = shell_tool or ShellTool(self.project_dir)
        tools: list[dict[str, Any]] = [
            self.build_handoff_tool_schema(),
            *self.fs_tool.get_tool_schemas(),
            *self.shell_tool.get_tool_schemas(),
        ]
        super().__init__(
            role=AgentRole.QA,
            system_prompt=prompt,
            tools=tools,
            llm_client=llm_client,
            model=model,
        )

    async def test_code(self, workspace: Workspace) -> HandoffResult:
        code_files = workspace.code.files
        if not code_files:
            return HandoffResult(
                status=HandoffStatus.FAILED,
                summary="No code files found in workspace",
            )

        arch_text = self._format_qa_context(workspace)
        code_summary = self._read_code_for_testing(code_files)

        full_prompt = (
            "## Architecture:\n"
            f"{arch_text}\n\n"
            "## Source code to test:\n\n"
            f"{code_summary}\n\n"
            "## Instructions:\n"
            "Generate comprehensive pytest test cases for this code.\n"
            "Include unit tests for all modules and integration tests for API endpoints.\n"
            "Use fixtures, parameterized tests, and proper mocking where needed.\n\n"
            "Write each test file using this format:\n"
            "```python\n"
            "# File: tests/test_module.py\n"
            "import pytest\n"
            "<test code>\n"
            "```\n\n"
            "After generating all tests, call the handoff tool "
            "with status, summary, and suggested_next='dev' (if tests need fixing) "
            "or suggested_next='documentation' (if all tests pass)."
        )

        self.add_message("user", full_prompt)

        if self.llm_client is None:
            return HandoffResult(
                status=HandoffStatus.FAILED,
                summary="No LLM client configured",
            )

        model = self.model or (self.llm_client.get_model(self.role) if self.llm_client else None)
        messages = [{"role": "system", "content": self.system_prompt}] + self._conversation_history

        response = await self.llm_client.chat(
            messages=messages,
            tools=self.tools,
            model=model,
        )

        assistant_msg = response.get("content", "")
        tool_calls = response.get("tool_calls", [])
        self.add_message("assistant", assistant_msg, tool_calls=tool_calls)

        if response.get("error"):
            return HandoffResult(
                status=HandoffStatus.FAILED,
                summary=assistant_msg[:500] if assistant_msg else "LLM API call failed",
            )

        self._extract_and_write_tests(assistant_msg, workspace)

        test_results = await self._run_tests()
        workspace.test.results = test_results

        coverage = await self._run_coverage()
        workspace.test.coverage = coverage

        self._update_workspace(assistant_msg, test_results, workspace)

        workspace.progress.current_phase = "testing"

        handoff = self._extract_handoff(tool_calls)
        if handoff:
            return handoff

        return HandoffResult(
            status=HandoffStatus.INCOMPLETE,
            summary=assistant_msg[:500] if assistant_msg else "No response",
        )

    def _format_qa_context(self, workspace: Workspace) -> str:
        arch = workspace.architecture
        parts: list[str] = []
        if arch.tech_stack:
            parts.append("Tech stack:\n" + json.dumps(arch.tech_stack, indent=2))
        if arch.api_contracts:
            parts.append("API contracts:\n" + json.dumps(arch.api_contracts, indent=2))
        if arch.database_schema:
            parts.append("Database schema:\n" + arch.database_schema)
        return "\n\n".join(parts) if parts else "No architecture context"

    def _read_code_for_testing(self, code_files: list) -> str:
        parts: list[str] = []
        for cf in code_files:
            result = self.fs_tool.read(cf.path)
            if result.success:
                lines = result.output.split("\n")
                if len(lines) > 100:
                    truncated = "\n".join(lines[:100])
                    parts.append(
                        f"### {cf.path}\n```python\n{truncated}\n"
                        f"... ({len(lines)} lines total)\n```"
                    )
                else:
                    parts.append(f"### {cf.path}\n```python\n{result.output}\n```")
            else:
                parts.append(f"### {cf.path}\n[Error: {result.error}]")
        return "\n\n".join(parts)

    def _extract_and_write_tests(self, response: str, workspace: Workspace) -> None:
        blocks = re.split(r"(```)", response)
        in_block = False
        current_content: list[str] = []

        i = 0
        while i < len(blocks):
            chunk = blocks[i]
            if chunk == "```":
                if not in_block:
                    in_block = True
                    current_content = []
                else:
                    content = "".join(current_content)
                    if content.strip():
                        file_path = self._find_test_path(content)
                        if file_path:
                            cleaned = self._clean_test_content(content)
                            self.fs_tool.write(file_path, cleaned.strip() + "\n")
                            logger.info("QA Agent wrote test: %s", file_path)
                    in_block = False
                    current_content = []
                i += 1
            elif in_block:
                current_content.append(chunk)
                i += 1
            else:
                i += 1

    def _find_test_path(self, content: str) -> str | None:
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# File: ") or stripped.startswith("// File: "):
                return stripped.split("File:", 1)[1].strip()
        return None

    def _clean_test_content(self, content: str) -> str:
        lines: list[str] = []
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# File:") or stripped.startswith("// File:"):
                continue
            lines.append(line)
        while lines and lines[0].strip() in ("python", "py", "python3", "markdown"):
            lines.pop(0)
        return "\n".join(lines).strip()

    async def _run_tests(self) -> dict[str, Any]:
        test_paths = list(self.project_dir.glob("tests/**/test_*.py"))
        test_paths.extend(self.project_dir.glob("test_*.py"))

        if not test_paths:
            return {
                "status": "no_tests",
                "passed": 0,
                "failed": 0,
                "summary": "No test files found",
            }

        test_files = " ".join(str(p.relative_to(self.project_dir)) for p in test_paths)
        cmd = f"python -m pytest {test_files} -v --tb=short 2>&1"
        try:
            result = await self.shell_tool.execute(cmd, timeout=120)
            return self._parse_pytest_output(result.output or result.error)
        except Exception as e:
            return {"status": "error", "passed": 0, "failed": 0, "summary": str(e)}

    async def _run_coverage(self) -> dict[str, Any]:
        test_paths = list(self.project_dir.glob("tests/**/test_*.py"))
        test_paths.extend(self.project_dir.glob("test_*.py"))
        if not test_paths:
            return {"percent": 0, "status": "skipped"}

        test_files = " ".join(str(p.relative_to(self.project_dir)) for p in test_paths)
        cmd = f"python -m pytest {test_files} --cov-report=term --no-header -q 2>&1"
        try:
            result = await self.shell_tool.execute(cmd, timeout=120)
            output = result.output or result.error or ""
            match = COVERAGE_LINE.search(output)
            if match:
                return {"percent": int(match.group(1)), "status": "complete"}
            return {"percent": 0, "status": "no_coverage_data"}
        except Exception:
            return {"percent": 0, "status": "error"}

    def _parse_pytest_output(self, output: str) -> dict[str, Any]:
        counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
        for match in PYTEST_RESULT_PATTERN.finditer(output):
            count = int(match.group(1))
            category = match.group(2)
            if category in counts:
                counts[category] += count

        total = sum(counts.values())
        if counts["failed"] > 0 or counts["error"] > 0:
            status = "failed" if counts["failed"] > 0 else "error"
        elif total == 0:
            status = "unknown"
        else:
            status = "passed"

        failed_outputs: list[dict[str, Any]] = []
        for match in PYTEST_SUMMARY_PATTERN.finditer(output):
            line = match.group(0).strip()
            failed_outputs.append(
                {
                    "type": "test_failure",
                    "output": line[:200],
                }
            )

        return {
            "status": status,
            "passed": counts["passed"],
            "failed": counts["failed"],
            "error": counts["error"],
            "skipped": counts["skipped"],
            "total": total,
            "summary": f"{total} tests: {counts['passed']} passed, "
            f"{counts['failed']} failed, {counts['error']} errors",
            "failures": failed_outputs,
        }

    def _classify_severity(self, test_results: dict[str, Any]) -> str | None:
        if test_results.get("error", 0) > 0:
            return "critical"
        if test_results.get("failed", 0) > test_results.get("passed", 0) // 2:
            return "major"
        if test_results.get("failed", 0) > 0:
            return "minor"
        return None

    def _update_workspace(
        self,
        response: str,
        test_results: dict[str, Any],
        workspace: Workspace,
    ) -> None:
        test = workspace.test
        test.summary = response[:1000]
        test.test_cases = []

        severity = self._classify_severity(test_results)
        if severity:
            test.severity = severity

        test.results = test_results
        test.coverage = workspace.test.coverage
