from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

STUCK_SIMILARITY_THRESHOLD = 0.85


class ErrorCategory(str, Enum):
    RECOVERABLE = "recoverable"
    UNRECOVERABLE = "unrecoverable"
    LLM_API = "llm_api"
    TOOL = "tool"
    WORKSPACE_CORRUPTION = "workspace_corruption"
    AGENT_STUCK = "agent_stuck"
    TIMEOUT = "timeout"


class RecoveryStrategy(str, Enum):
    RETRY = "retry"
    CHECKPOINT = "checkpoint"
    FORCE_HANDOFF = "force_handoff"
    ABORT = "abort"
    BACKUP_AND_RECREATE = "backup_and_recreate"


@dataclass
class ErrorContext:
    category: ErrorCategory
    message: str
    agent_role: str = ""
    phase: str = ""
    recoverable: bool = True
    retry_count: int = 0
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorDecision:
    strategy: RecoveryStrategy
    reason: str
    suggested_action: str = ""


class ErrorHandler:
    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries
        self._previous_summaries: dict[str, str] = {}

    def classify_tool_error(
        self, error: str, metadata: dict[str, Any] | None = None
    ) -> ErrorContext:
        metadata = metadata or {}

        if metadata.get("recoverable") is False:
            return ErrorContext(
                category=ErrorCategory.UNRECOVERABLE,
                message=error,
                recoverable=False,
            )

        if metadata.get("recoverable") is not None:
            return ErrorContext(
                category=ErrorCategory.TOOL,
                message=error,
                recoverable=True,
                metadata=metadata,
            )

        unrecoverable_patterns = [
            "permission denied",
            "disk full",
            "no space left",
            "read-only file system",
            "path traversal",
        ]
        for pattern in unrecoverable_patterns:
            if pattern in error.lower():
                return ErrorContext(
                    category=ErrorCategory.UNRECOVERABLE,
                    message=error,
                    recoverable=False,
                )

        return ErrorContext(
            category=ErrorCategory.TOOL,
            message=error,
            recoverable=True,
        )

    def classify_llm_error(self, error: str, attempt: int, max_retries: int) -> ErrorContext:
        if attempt >= max_retries - 1:
            return ErrorContext(
                category=ErrorCategory.LLM_API,
                message=error,
                recoverable=False,
                retry_count=attempt + 1,
                max_retries=max_retries,
            )

        return ErrorContext(
            category=ErrorCategory.LLM_API,
            message=error,
            recoverable=True,
            retry_count=attempt + 1,
            max_retries=max_retries,
        )

    def detect_stuck_agent(self, agent_id: str, current_summary: str) -> ErrorContext | None:
        previous = self._previous_summaries.get(agent_id, "")
        self._previous_summaries[agent_id] = current_summary

        if not previous or not current_summary:
            return None

        ratio = difflib.SequenceMatcher(None, previous.lower(), current_summary.lower()).ratio()

        if ratio >= STUCK_SIMILARITY_THRESHOLD:
            return ErrorContext(
                category=ErrorCategory.AGENT_STUCK,
                message=f"Agent {agent_id} appears stuck (similarity={ratio:.2f})",
                recoverable=False,
            )

        return None

    def classify_workspace_error(self, error: str) -> ErrorContext:
        return ErrorContext(
            category=ErrorCategory.WORKSPACE_CORRUPTION,
            message=error,
            recoverable=False,
        )

    def decide(self, context: ErrorContext) -> ErrorDecision:
        if context.category == ErrorCategory.WORKSPACE_CORRUPTION:
            return ErrorDecision(
                strategy=RecoveryStrategy.BACKUP_AND_RECREATE,
                reason="Workspace file is corrupt",
                suggested_action="Backup the corrupt file and recreate the workspace. "
                "Previous state will be recovered from the backup if possible.",
            )

        if context.category == ErrorCategory.AGENT_STUCK:
            return ErrorDecision(
                strategy=RecoveryStrategy.CHECKPOINT,
                reason="Agent appears stuck in a loop",
                suggested_action="Review the agent output and provide direction.",
            )

        if context.category == ErrorCategory.LLM_API:
            if context.recoverable and context.retry_count < context.max_retries:
                return ErrorDecision(
                    strategy=RecoveryStrategy.RETRY,
                    reason=f"LLM API retry {context.retry_count}/{context.max_retries}",
                )
            return ErrorDecision(
                strategy=RecoveryStrategy.CHECKPOINT,
                reason=f"LLM API failed after {context.max_retries} retries",
                suggested_action="Check API key and network connectivity.",
            )

        if context.category == ErrorCategory.UNRECOVERABLE:
            return ErrorDecision(
                strategy=RecoveryStrategy.CHECKPOINT,
                reason=f"Unrecoverable error: {context.message[:200]}",
                suggested_action="Review the error and decide how to proceed.",
            )

        if context.recoverable and context.retry_count < context.max_retries:
            return ErrorDecision(
                strategy=RecoveryStrategy.RETRY,
                reason=(
                    f"Recoverable error, retrying ({context.retry_count + 1}/{context.max_retries})"
                ),
            )

        return ErrorDecision(
            strategy=RecoveryStrategy.CHECKPOINT,
            reason=f"Max retries ({context.max_retries}) reached for recoverable error",
        )

    def reset_agent_similarity(self, agent_id: str) -> None:
        self._previous_summaries.pop(agent_id, None)

    def clear_all_similarity(self) -> None:
        self._previous_summaries.clear()
