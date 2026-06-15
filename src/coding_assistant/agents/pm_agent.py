from __future__ import annotations

import json
import logging
from typing import Any

from coding_assistant.agents.base import Agent
from coding_assistant.core.types import AgentRole, HandoffResult, HandoffStatus
from coding_assistant.core.workspace import Workspace
from coding_assistant.llm.client import LLMClient
from coding_assistant.llm.templates import PromptTemplateManager

logger = logging.getLogger(__name__)


class PMAgent(Agent):
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str | None = None,
    ) -> None:
        template_mgr = PromptTemplateManager()
        prompt = template_mgr.render(AgentRole.PM, context="")
        tools: list[dict[str, Any]] = [
            self.build_handoff_tool_schema(),
        ]
        super().__init__(
            role=AgentRole.PM,
            system_prompt=prompt,
            tools=tools,
            llm_client=llm_client,
            model=model,
        )

    async def analyze_requirements(self, user_input: str, workspace: Workspace) -> HandoffResult:
        context = self._build_context(workspace)
        full_prompt = (
            f"## Existing project context:\n{context}\n\n"
            f"## New user requirement:\n{user_input}\n\n"
            "Analyze this requirement and produce a structured PRD. "
            "Include: problem statement, feature list with priorities, "
            "user stories, and acceptance criteria.\n\n"
            "Write your analysis to the workspace and call the handoff tool when done."
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
            temperature=0.5,
        )

        assistant_msg = response.get("content", "")
        tool_calls = response.get("tool_calls", [])
        self.add_message("assistant", assistant_msg, tool_calls=tool_calls)

        if not response.get("error"):
            self._update_workspace(assistant_msg, workspace)

        handoff = self._try_handoff(tool_calls, assistant_msg)
        if handoff:
            return handoff

        return HandoffResult(
            status=HandoffStatus.INCOMPLETE,
            summary=assistant_msg[:500] if assistant_msg else "No response",
        )

    def _build_context(self, workspace: Workspace) -> str:
        parts = []
        if workspace.requirements.prd:
            parts.append(f"Existing PRD:\n{workspace.requirements.prd}")
        if workspace.requirements.feature_list:
            features = json.dumps(workspace.requirements.feature_list, indent=2)
            parts.append(f"Existing features:\n{features}")
        if workspace.progress.phase_summaries:
            summaries = json.dumps(workspace.progress.phase_summaries, indent=2)
            parts.append(f"Phase summaries:\n{summaries}")
        return "\n\n".join(parts) if parts else "No existing context (new project)."

    def _update_workspace(self, response: str, workspace: Workspace) -> None:
        workspace.requirements.prd = response
        workspace.progress.current_phase = "requirements"
