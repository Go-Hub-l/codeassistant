from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from coding_assistant.agents.base import Agent
from coding_assistant.core.types import AgentRole, HandoffResult, HandoffStatus, Severity
from coding_assistant.core.workspace import Workspace
from coding_assistant.llm.client import LLMClient
from coding_assistant.llm.templates import PromptTemplateManager
from coding_assistant.tools.code_executor import ShellTool
from coding_assistant.tools.file_system import FileSystemTool

logger = logging.getLogger(__name__)

SECURITY_PATTERNS = [
    (
        r"(?:password|secret|api[_-]?key|token)\s*=\s*['\"][^'\"]+['\"]",
        "critical",
        "Hardcoded secret/credential detected",
    ),
    (r"exec\s*\(.*\)", "critical", "Unsafe exec() call"),
    (r"eval\s*\(.*\)", "critical", "Unsafe eval() call"),
    (r"os\.system\s*\(.*\)", "critical", "Unsafe os.system() call"),
    (
        r"subprocess\.(?:call|Popen)\s*\(.*shell\s*=\s*True",
        "critical",
        "Shell injection risk via subprocess with shell=True",
    ),
    (r"pickle\.(?:load|loads)\s*\(.*\)", "critical", "Unsafe pickle deserialization"),
    (
        r"(?:SELECT|INSERT|UPDATE|DELETE).*%.*\)",
        "major",
        "Potential SQL injection via string formatting",
    ),
    (
        r"(?:SELECT|INSERT|UPDATE|DELETE).*f['\"].*\{",
        "major",
        "Potential SQL injection via f-string",
    ),
    (r"assert\s", "minor", "Use of assert (removed with -O flag)"),
    (r"print\s*\(.*\)", "minor", "Use of print() instead of logging"),
    (r"except\s*:", "minor", "Bare except clause"),
    (r"open\s*\(.*\)\s*(?!.*with)", "major", "File opened without context manager"),
    (r"\.read\s*\(\s*\)", "minor", "Unsafe file read (possible large file)"),
]

RUFF_WARNING_PATTERN = re.compile(
    r"([^:]+):(\d+):(\d+):\s*([A-Z]+\d+)\s+(.+)",
)
BANDIT_FINDING_PATTERN = re.compile(
    r">>\s*Issue:\s*\[([^:]+):([^\]]+)\]\s*(.+)",
)
MYPY_ERROR_PATTERN = re.compile(
    r"([^:]+):(\d+):\s*(error|warning):\s*(.+)",
)


class ReviewerAgent(Agent):
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str | None = None,
        project_dir: Path | None = None,
        fs_tool: FileSystemTool | None = None,
        shell_tool: ShellTool | None = None,
    ) -> None:
        template_mgr = PromptTemplateManager()
        prompt = template_mgr.render(AgentRole.REVIEWER, context="", code="")
        self.project_dir = project_dir or Path(".")
        self.fs_tool = fs_tool or FileSystemTool(self.project_dir)
        self.shell_tool = shell_tool or ShellTool(self.project_dir)
        tools: list[dict[str, Any]] = [
            self.build_handoff_tool_schema(),
            *self.fs_tool.get_tool_schemas(),
            *self.shell_tool.get_tool_schemas(),
        ]
        super().__init__(
            role=AgentRole.REVIEWER,
            system_prompt=prompt,
            tools=tools,
            llm_client=llm_client,
            model=model,
        )

    async def review_code(self, workspace: Workspace) -> HandoffResult:
        code_files = workspace.code.files
        if not code_files:
            return HandoffResult(
                status=HandoffStatus.FAILED,
                summary="No code files found in workspace",
            )

        security_issues = self._run_security_checks(code_files)
        tool_results = await self._run_static_analysis(code_files)
        code_summary = self._read_code_files(code_files)

        full_prompt = f"## Code to review:\n\n{code_summary}\n\n"

        if security_issues:
            full_prompt += (
                "## Security scan results:\n" + json.dumps(security_issues, indent=2) + "\n\n"
            )

        if tool_results:
            full_prompt += (
                "## Static analysis results:\n" + json.dumps(tool_results, indent=2) + "\n\n"
            )

        full_prompt += (
            "## Instructions:\n"
            "Review the code for quality, security, and convention compliance.\n"
            "Classify each issue by severity:\n"
            "- **minor**: code style, formatting, simple improvements\n"
            "- **major**: logic errors, missing error handling, performance issues\n"
            "- **critical**: security vulnerabilities, architectural flaws, data loss risks\n\n"
            "Produce a review report with:\n"
            "1. Overall severity assessment (highest among issues found)\n"
            "2. List of issues with file, line, severity, and description\n"
            "3. Summary of what was checked\n\n"
            "Set suggested_next='dev' if issues found, or 'qa' if no issues.\n"
            "When done, call the handoff tool with your summary and severity."
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

        self._update_workspace(assistant_msg, security_issues, tool_results, workspace)

        handoff = self._try_handoff(tool_calls, assistant_msg)
        if handoff:
            return handoff

        return HandoffResult(
            status=HandoffStatus.INCOMPLETE,
            summary=assistant_msg[:500] if assistant_msg else "No response",
        )

    def _read_code_files(self, code_files: list) -> str:
        parts: list[str] = []
        for cf in code_files:
            result = self.fs_tool.read(cf.path)
            if result.success:
                parts.append(f"### {cf.path}\n```python\n{result.output[:2000]}\n```")
            else:
                parts.append(f"### {cf.path}\n[Error reading: {result.error}]")
        return "\n\n".join(parts)

    def _run_security_checks(self, code_files: list) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        for cf in code_files:
            result = self.fs_tool.read(cf.path)
            if not result.success:
                continue
            content = result.output
            for pattern, severity, description in SECURITY_PATTERNS:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_no = content[: match.start()].count("\n") + 1
                    issues.append(
                        {
                            "file": cf.path,
                            "line": line_no,
                            "severity": severity,
                            "description": description,
                            "match": match.group(0)[:120],
                            "source": "security_scanner",
                        }
                    )
        return issues

    async def _run_static_analysis(self, code_files: list) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        py_files = [cf.path for cf in code_files if cf.path.endswith(".py")]

        if py_files:
            ruff_issues = await self._run_ruff(py_files)
            if ruff_issues:
                results.extend(ruff_issues)

            bandit_issues = await self._run_bandit(py_files)
            if bandit_issues:
                results.extend(bandit_issues)

            mypy_issues = await self._run_mypy(py_files)
            if mypy_issues:
                results.extend(mypy_issues)

        return results

    async def _run_ruff(self, py_files: list[str]) -> list[dict[str, Any]]:
        cmd = f"ruff check {' '.join(py_files)} --output-format text 2>&1"
        try:
            result = await self.shell_tool.execute(cmd, timeout=30)
            if result.success and result.output:
                return self._parse_ruff_output(result.output)
        except Exception:
            pass
        return []

    async def _run_bandit(self, py_files: list[str]) -> list[dict[str, Any]]:
        cmd = f"bandit -r {' '.join(py_files)} -f txt 2>&1"
        try:
            result = await self.shell_tool.execute(cmd, timeout=30)
            if result.output and "Issue:" in result.output:
                return self._parse_bandit_output(result.output)
        except Exception:
            pass
        return []

    async def _run_mypy(self, py_files: list[str]) -> list[dict[str, Any]]:
        cmd = f"mypy {' '.join(py_files)} --no-color-output 2>&1"
        try:
            result = await self.shell_tool.execute(cmd, timeout=30)
            if result.output and "error:" in result.output:
                return self._parse_mypy_output(result.output)
        except Exception:
            pass
        return []

    def _parse_ruff_output(self, output: str) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        ruff_severity_map = {
            "F": "minor",
            "E": "major",
            "W": "minor",
            "I": "minor",
            "N": "minor",
            "D": "minor",
            "PLC": "major",
            "PLE": "major",
            "PLW": "minor",
            "S": "critical",
            "B": "major",
            "SIM": "minor",
            "RUF": "minor",
            "C": "minor",
            "T": "minor",
        }
        for line in output.strip().split("\n"):
            match = RUFF_WARNING_PATTERN.match(line)
            if match:
                file_path, line_no, col, rule, message = match.groups()
                severity = ruff_severity_map.get(rule[0], "minor")
                issues.append(
                    {
                        "file": file_path.strip(),
                        "line": int(line_no),
                        "severity": severity,
                        "description": f"[{rule}] {message.strip()}",
                        "source": "ruff",
                    }
                )
        return issues

    def _parse_bandit_output(self, output: str) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        severity_map = {"HIGH": "critical", "MEDIUM": "major", "LOW": "minor"}
        for match in BANDIT_FINDING_PATTERN.finditer(output):
            severity_key, test_id, description = match.groups()
            issues.append(
                {
                    "file": "",
                    "line": 0,
                    "severity": severity_map.get(severity_key.strip(), "major"),
                    "description": f"[{test_id.strip()}] {description.strip()}",
                    "source": "bandit",
                }
            )
        return issues

    def _parse_mypy_output(self, output: str) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        for match in MYPY_ERROR_PATTERN.finditer(output):
            file_path, line_no, level, message = match.groups()
            severity = "major" if level == "error" else "minor"
            issues.append(
                {
                    "file": file_path.strip(),
                    "line": int(line_no),
                    "severity": severity,
                    "description": f"{level}: {message.strip()}",
                    "source": "mypy",
                }
            )
        return issues

    def _update_workspace(
        self,
        response: str,
        security_issues: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        workspace: Workspace,
    ) -> None:
        review = workspace.review
        review.summary = response[:1000]
        review.issues = []

        for issue in security_issues:
            review.issues.append(issue)

        for result in tool_results:
            review.issues.append(result)

        severities = [i.get("severity") for i in review.issues]
        if "critical" in severities:
            review.severity = "critical"
        elif "major" in severities:
            review.severity = "major"
        elif "minor" in severities:
            review.severity = "minor"

        workspace.progress.current_phase = "review"

    def get_severity_from_summary(self, summary: str) -> Severity | None:
        lower = summary.lower()
        if "critical" in lower or "vulnerability" in lower:
            return Severity.CRITICAL
        if "major" in lower or "logic error" in lower:
            return Severity.MAJOR
        if "minor" in lower or "style" in lower or "formatting" in lower:
            return Severity.MINOR
        return None
