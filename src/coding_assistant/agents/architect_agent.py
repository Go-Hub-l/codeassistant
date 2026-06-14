from __future__ import annotations

import json
import logging
import re
from typing import Any

from coding_assistant.agents.base import Agent
from coding_assistant.core.types import AgentRole, HandoffResult, HandoffStatus
from coding_assistant.core.workspace import Workspace
from coding_assistant.llm.client import LLMClient
from coding_assistant.llm.templates import PromptTemplateManager

logger = logging.getLogger(__name__)


class ArchitectAgent(Agent):
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str | None = None,
    ) -> None:
        template_mgr = PromptTemplateManager()
        prompt = template_mgr.render(AgentRole.ARCHITECT, context="", requirements="")
        tools: list[dict[str, Any]] = [
            self.build_handoff_tool_schema(),
        ]
        super().__init__(
            role=AgentRole.ARCHITECT,
            system_prompt=prompt,
            tools=tools,
            llm_client=llm_client,
            model=model,
        )

    async def design_architecture(
        self, workspace: Workspace, user_feedback: str | None = None
    ) -> HandoffResult:
        requirements_text = workspace.requirements.prd
        if not requirements_text:
            return HandoffResult(
                status=HandoffStatus.FAILED,
                summary="No requirements found in workspace",
            )

        feedback_note = ""
        if user_feedback:
            feedback_note = (
                f"\n\n## User feedback on previous architecture:\n{user_feedback}\n\n"
                "Revise the architecture based on this feedback."
            )

        existing_arch = self._format_existing_architecture(workspace)
        existing_note = ""
        if existing_arch:
            existing_note = (
                f"\n\n## Existing architecture design:\n{existing_arch}\n\n"
                "Update or replace this as needed."
            )

        full_prompt = (
            f"## Requirements:\n{requirements_text}\n\n"
            f"{existing_note}"
            f"{feedback_note}"
            "Design the technical architecture for this project. Include:\n"
            "1. **Technology Stack**: Python framework and all libraries with rationale\n"
            "2. **Project Directory Structure**: Full directory tree\n"
            "3. **API Contracts**: REST/GraphQL endpoints with methods, paths, and descriptions\n"
            "4. **Database Schema**: Tables, columns, types, and relationships\n"
            "5. **Security Considerations**: Auth strategy, input validation, threat model\n\n"
            "Write your architecture to the workspace and call the handoff tool when done."
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

        self._update_workspace(assistant_msg, workspace)

        handoff = self._extract_handoff(tool_calls)
        if handoff:
            return handoff

        return HandoffResult(
            status=HandoffStatus.INCOMPLETE,
            summary=assistant_msg[:500] if assistant_msg else "No response",
        )

    def _format_existing_architecture(self, workspace: Workspace) -> str:
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

    def _update_workspace(self, response: str, workspace: Workspace) -> None:
        arch = workspace.architecture
        arch.summary = response[:1000]

        tech_stack = self._extract_tech_stack(response)
        if tech_stack:
            arch.tech_stack = tech_stack

        proj_structure = self._extract_section(
            response, r"project\s*(?:directory\s*)?structure", r"api\s*contract"
        )
        if proj_structure:
            arch.project_structure = proj_structure.strip()

        api_contracts = self._extract_api_contracts(response)
        if api_contracts:
            arch.api_contracts = api_contracts

        db_schema = self._extract_section(
            response, r"database\s*schema", r"(?:security|api\s*contract)"
        )
        if db_schema:
            arch.database_schema = db_schema.strip()

        security = self._extract_section(response, r"security\s*(?:considerations)?", r"$")
        if security:
            arch.security_considerations = security.strip()

        workspace.progress.current_phase = "architecture"

    def _extract_tech_stack(self, text: str) -> dict[str, str] | None:
        frameworks = {
            "fastapi": "FastAPI",
            "flask": "Flask",
            "django": "Django",
            "litestar": "Litestar",
            "starlette": "Starlette",
            "sanic": "Sanic",
            "tornado": "Tornado",
            "aiohttp": "aiohttp",
            "uvicorn": "Uvicorn",
            "gunicorn": "Gunicorn",
            "postgresql": "PostgreSQL",
            "postgres": "PostgreSQL",
            "mysql": "MySQL",
            "sqlite": "SQLite",
            "redis": "Redis",
            "celery": "Celery",
            "rabbitmq": "RabbitMQ",
            "docker": "Docker",
            "pytest": "pytest",
            "ruff": "ruff",
            "mypy": "mypy",
            "sqlalchemy": "SQLAlchemy",
            "pydantic": "Pydantic",
            "alembic": "Alembic",
            "jwt": "JWT",
            "pytest-asyncio": "pytest-asyncio",
        }
        lower_text = text.lower()
        found: dict[str, str] = {}
        for key, label in frameworks.items():
            if key in lower_text:
                found[key] = label
        return found if found else None

    def _extract_section(self, text: str, start_pattern: str, end_pattern: str) -> str | None:
        match = re.search(
            rf"(?i)(?:##\s*|###\s*|(?:\*\*))?\s*{start_pattern}[:\s]*(?:\*\*)?+\n?(.*?)(?=(?:##\s*|###\s*|(?:\*\*))?\s*{end_pattern}|$)",
            text,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return None

    def _extract_api_contracts(self, text: str) -> list[dict[str, Any]]:
        contracts: list[dict[str, Any]] = []
        endpoint_pattern = re.compile(
            r"(?:GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\s+(\S+)",
            re.IGNORECASE,
        )
        lines = text.split("\n")
        current: dict[str, Any] | None = None

        for line in lines:
            ep_match = endpoint_pattern.search(line)
            if ep_match:
                if current:
                    contracts.append(current)
                current = {
                    "method": ep_match.group(0).split()[0].upper(),
                    "path": ep_match.group(1),
                    "description": line[ep_match.end() :].strip(" -:"),
                }
            elif current and line.strip():
                current["description"] += " " + line.strip()

        if current:
            contracts.append(current)

        return contracts
