from __future__ import annotations

from typing import TYPE_CHECKING, Any

from coding_assistant.core.types import AgentRole, HandoffResult, HandoffStatus

if TYPE_CHECKING:
    from coding_assistant.llm.client import LLMClient


class Agent:
    role: AgentRole
    system_prompt: str
    tools: list[dict[str, Any]]

    def __init__(
        self,
        role: AgentRole,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        llm_client: LLMClient | None = None,
        model: str | None = None,
    ) -> None:
        self.role = role
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.tools = tools or []
        self.llm_client = llm_client
        self.model = model
        self._conversation_history: list[dict[str, Any]] = []

    def _default_system_prompt(self) -> str:
        return f"You are a {self.role.value} agent. Perform your duties accordingly."

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        message: dict[str, Any] = {"role": role, "content": content}
        message.update(kwargs)
        self._conversation_history.append(message)

    def get_conversation_history(self) -> list[dict[str, Any]]:
        return list(self._conversation_history)

    def clear_conversation_history(self) -> None:
        self._conversation_history = []

    async def run(self, input_text: str, **kwargs: Any) -> HandoffResult:
        self.add_message("user", input_text)

        if self.llm_client is None:
            return HandoffResult(
                status=HandoffStatus.FAILED,
                summary="No LLM client configured",
            )

        messages = [{"role": "system", "content": self.system_prompt}] + self._conversation_history

        response = await self.llm_client.chat(
            messages=messages,
            tools=self.tools,
            model=self.model,
            **kwargs,
        )

        assistant_message = response.get("content", "")
        tool_calls = response.get("tool_calls", [])

        self.add_message("assistant", assistant_message, tool_calls=tool_calls)

        if response.get("error"):
            return HandoffResult(
                status=HandoffStatus.FAILED,
                summary=assistant_message[:500] if assistant_message else "LLM API call failed",
            )

        handoff_result = self._extract_handoff(tool_calls)
        if handoff_result:
            return handoff_result

        return HandoffResult(
            status=HandoffStatus.INCOMPLETE,
            summary=assistant_message[:500] if assistant_message else "No response generated",
        )

    def _extract_handoff(self, tool_calls: list[dict[str, Any]]) -> HandoffResult | None:
        for call in tool_calls:
            if call.get("function", {}).get("name") == "handoff":
                import json

                try:
                    args = json.loads(call["function"]["arguments"])
                    return HandoffResult(**args)
                except (json.JSONDecodeError, TypeError):
                    continue
        return None

    def build_handoff_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "handoff",
                "description": "Signal completion of your phase and hand off to the next agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["completed", "incomplete", "failed"],
                            "description": "Whether this phase completed successfully.",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Summary of what was accomplished or why it failed.",
                        },
                        "suggested_next": {
                            "type": ["string", "null"],
                            "description": "Role name of the suggested next agent, or null.",
                        },
                        "severity": {
                            "type": ["string", "null"],
                            "enum": ["minor", "major", "critical", None],
                            "description": "Severity of issues found, if any.",
                        },
                    },
                    "required": ["status", "summary"],
                },
            },
        }
