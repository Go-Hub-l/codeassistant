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

FILE_MARKER_PATTERN = re.compile(
    r"(?:^|\n)(?:#\s*File:\s*([^\n]+)|//\s*File:\s*([^\n]+)|<!--\s*File:\s*([^\n]+)\s*-->)",
    re.IGNORECASE | re.MULTILINE,
)

ALTERNATE_FILE_PATTERN = re.compile(
    r"^[#*]{1,4}\s*([\w./-]+\.[\w]+)\s*$",
    re.MULTILINE,
)

DOC_FILES = {
    "readme": "README.md",
    "api": "API.md",
    "database": "DATABASE.md",
    "deployment": "DEPLOYMENT.md",
    "changelog": "CHANGELOG.md",
}


class DevAgent(Agent):
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str | None = None,
        project_dir: Path | None = None,
        fs_tool: FileSystemTool | None = None,
        shell_tool: ShellTool | None = None,
    ) -> None:
        template_mgr = PromptTemplateManager()
        prompt = template_mgr.render(AgentRole.DEV, context="", architecture="")
        self.project_dir = project_dir or Path(".")
        self.fs_tool = fs_tool or FileSystemTool(self.project_dir)
        self.shell_tool = shell_tool or ShellTool(self.project_dir)
        tools: list[dict[str, Any]] = [
            self.build_handoff_tool_schema(),
            *self.fs_tool.get_tool_schemas(),
            *self.shell_tool.get_tool_schemas(),
        ]
        super().__init__(
            role=AgentRole.DEV,
            system_prompt=prompt,
            tools=tools,
            llm_client=llm_client,
            model=model,
        )
        self._generated_files: set[str] = set()

    async def implement_code(
        self, workspace: Workspace, feedback: str | None = None
    ) -> HandoffResult:
        arch_text = self._format_architecture(workspace)
        if not arch_text:
            return HandoffResult(
                status=HandoffStatus.FAILED,
                summary="No architecture found in workspace",
            )

        existing_code = self._format_existing_code(workspace)
        feedback_note = ""
        if feedback:
            feedback_note = (
                f"\n\n## Reviewer feedback - fix these issues:\n{feedback}\n\n"
                "Update the code to address all issues."
            )

        full_prompt = (
            f"## Architecture:\n{arch_text}\n\n"
            f"{existing_code}\n"
            f"{feedback_note}"
            "## Instructions:\n"
            "Implement ALL features according to the architecture. "
            "Generate complete, runnable Python code.\n\n"
            "For each file you create or modify, use this format:\n"
            "```python\n"
            "# File: path/to/file.py\n"
            "<file content>\n"
            "```\n\n"
            "Write complete files, not fragments. "
            "Include all imports, type hints, and docstrings.\n"
            "When you are done implementing ALL code, call the handoff tool."
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

        self._extract_and_write_files(assistant_msg, workspace)

        workspace.progress.current_phase = "development"

        handoff = self._try_handoff(tool_calls, assistant_msg)
        if handoff:
            return handoff

        return HandoffResult(
            status=HandoffStatus.INCOMPLETE,
            summary=assistant_msg[:500] if assistant_msg else "No response",
        )

    async def generate_documentation(self, workspace: Workspace) -> HandoffResult:
        code_files = [f.path for f in workspace.code.files]
        arch_text = self._format_architecture(workspace)

        full_prompt = (
            f"## Architecture:\n{arch_text}\n\n"
            f"## Generated code files:\n{json.dumps(code_files, indent=2)}\n\n"
            "## Instructions:\n"
            "Generate the following documentation files:\n"
            "1. **README.md** — project overview, setup instructions, architecture overview\n"
            "2. **API.md** — endpoint reference with request/response examples\n"
            "3. **DATABASE.md** — schema reference, migration guide\n"
            "4. **DEPLOYMENT.md** — deployment steps, environment variables, requirements\n"
            "5. **CHANGELOG.md** — version history starting with v0.1.0\n\n"
            "Use this format for each file:\n"
            "```markdown\n"
            "# File: README.md\n"
            "<content>\n"
            "```\n\n"
            "When you are done generating all docs, call the handoff tool."
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

        self._extract_and_write_files(assistant_msg, workspace)

        workspace.progress.current_phase = "documentation"

        handoff = self._try_handoff(tool_calls, assistant_msg)
        if handoff:
            return handoff

        return HandoffResult(
            status=HandoffStatus.INCOMPLETE,
            summary=assistant_msg[:500] if assistant_msg else "No response",
        )

    def _format_architecture(self, workspace: Workspace) -> str:
        arch = workspace.architecture
        parts: list[str] = []
        if arch.tech_stack:
            parts.append("Tech stack:\n" + json.dumps(arch.tech_stack, indent=2))
        if arch.project_structure:
            parts.append("Project structure:\n" + arch.project_structure)
        if arch.api_contracts:
            parts.append("API contracts:\n" + json.dumps(arch.api_contracts, indent=2))
        if arch.database_schema:
            parts.append("Database schema:\n" + arch.database_schema)
        if arch.security_considerations:
            parts.append("Security considerations:\n" + arch.security_considerations)
        return "\n\n".join(parts) if parts else ""

    def _format_existing_code(self, workspace: Workspace) -> str:
        if not workspace.code.files:
            return ""
        files_info = []
        for f in workspace.code.files:
            status = "modified" if f.modified else "exists"
            files_info.append(f"- {f.path} ({status}): {f.description}")
        return (
            "## Existing code files:\n" + "\n".join(files_info) + "\n\n"
            "Update these files as needed."
        )

    def _extract_and_write_files(self, response: str, workspace: Workspace) -> None:
        self._generated_files.clear()

        blocks = re.split(r"(```)", response)
        in_block = False
        current_lang = ""
        current_content: list[str] = []

        i = 0
        while i < len(blocks):
            chunk = blocks[i]
            if chunk == "```":
                if not in_block:
                    in_block = True
                    if i + 1 < len(blocks):
                        next_block = blocks[i + 1]
                        first_line = next_block.split("\n", 1)[0].strip()
                        if first_line and not first_line.startswith("#"):
                            current_lang = first_line.lower()
                    else:
                        current_lang = ""
                    current_content = []
                else:
                    content = "".join(current_content)
                    if content.strip():
                        file_path = self._find_file_path(content, current_lang)
                        if file_path:
                            cleaned = self._clean_content(content, current_lang)
                            full_path = self._write_file(file_path, cleaned)
                            if full_path:
                                self._generated_files.add(full_path)
                                workspace.add_file_reference(
                                    full_path,
                                    description="Generated by Dev Agent",
                                    created_by=AgentRole.DEV,
                                )
                    in_block = False
                    current_lang = ""
                    current_content = []
                i += 1
            elif in_block:
                current_content.append(chunk)
                i += 1
            else:
                i += 1

    def _clean_content(self, content: str, lang: str) -> str:
        lines = content.split("\n")

        filtered: list[str] = []
        for line in lines:
            stripped = line.strip()
            if (
                stripped.startswith("# File:")
                or stripped.startswith("// File:")
                or stripped.startswith("<!-- File:")
            ):
                continue
            if lang and not filtered and stripped == lang:
                continue
            if (
                not lang
                and not filtered
                and stripped
                in (
                    "python",
                    "yaml",
                    "json",
                    "toml",
                    "env",
                    "sh",
                    "sql",
                    "markdown",
                    "md",
                    "html",
                    "css",
                    "js",
                    "txt",
                )
            ):
                continue
            filtered.append(line)

        return "\n".join(filtered)

    def _find_file_path(self, content: str, lang: str = "") -> str | None:
        match = FILE_MARKER_PATTERN.search(content)
        if match:
            return (match.group(1) or match.group(2) or match.group(3)).strip()

        lines = content.strip().split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# File: "):
                return stripped.split("File:", 1)[1].strip()
            if stripped.startswith("// File: "):
                return stripped.split("File:", 1)[1].strip()

        return None

    def _write_file(self, file_path: str, content: str) -> str | None:
        path = file_path.strip()
        if path.startswith("/"):
            path = path.lstrip("/")

        if not re.match(r"^[\w./-]+\.[\w]+$", path):
            return None

        if not content.strip():
            return None

        result = self.fs_tool.write(path, content.strip() + "\n")
        if result.success:
            logger.info("Dev Agent wrote file: %s (%d chars)", path, len(content))
            return path
        else:
            logger.warning("Dev Agent failed to write %s: %s", path, result.error)
            return None

    def get_generated_files(self) -> set[str]:
        return self._generated_files.copy()
