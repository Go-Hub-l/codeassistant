from __future__ import annotations

import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI

from coding_assistant.core.types import AgentRole

logger = logging.getLogger(__name__)

DEFAULT_MODELS: dict[AgentRole, str] = {
    AgentRole.PM: "gpt-4o",
    AgentRole.ARCHITECT: "gpt-4o",
    AgentRole.DEV: "gpt-4o",
    AgentRole.REVIEWER: "gpt-4o-mini",
    AgentRole.QA: "gpt-4o-mini",
    AgentRole.PMGR: "gpt-4o-mini",
}

MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_overrides: dict[AgentRole, str] | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model_overrides = model_overrides or {}
        self._token_usage: dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}

    def get_model(self, role: AgentRole) -> str:
        if role in self._model_overrides:
            return self._model_overrides[role]
        return DEFAULT_MODELS[role]

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        backoff = INITIAL_BACKOFF
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                return await self._chat_once(
                    messages=messages,
                    tools=tools,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM API call failed (attempt %d/%d): %s",
                    attempt + 1,
                    MAX_RETRIES,
                    str(e),
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(backoff)
                    backoff *= 2

        return {
            "content": f"LLM API call failed after {MAX_RETRIES} retries: {last_error}",
            "tool_calls": [],
            "error": True,
        }

    async def _chat_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
        }
        if model:
            kwargs["model"] = model
        if tools:
            kwargs["tools"] = tools
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if stream:
            return await self._chat_stream(**kwargs)

        response = await self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        message = choice.message

        if response.usage:
            self._token_usage["prompt"] += response.usage.prompt_tokens
            self._token_usage["completion"] += response.usage.completion_tokens
            self._token_usage["total"] += response.usage.total_tokens

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        return {
            "content": message.content or "",
            "tool_calls": tool_calls,
            "finish_reason": choice.finish_reason,
        }

    async def _chat_stream(self, **kwargs: Any) -> dict[str, Any]:
        kwargs["stream"] = True
        collected_content: list[str] = []
        collected_tool_calls: dict[int, dict[str, Any]] = {}

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    collected_content.append(delta.content)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in collected_tool_calls:
                            collected_tool_calls[idx] = {
                                "id": tc.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc.id:
                            collected_tool_calls[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                collected_tool_calls[idx]["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                collected_tool_calls[idx]["function"][
                                    "arguments"
                                ] += tc.function.arguments

        tool_calls_list = [collected_tool_calls[i] for i in sorted(collected_tool_calls.keys())]

        return {
            "content": "".join(collected_content),
            "tool_calls": tool_calls_list,
            "finish_reason": "stop",
        }

    def get_token_usage(self) -> dict[str, int]:
        return dict(self._token_usage)
