from coding_assistant.core.error_handler import (
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    RecoveryStrategy,
)


class TestErrorHandlerClassification:
    def test_tool_error_recoverable(self):
        handler = ErrorHandler()
        ctx = handler.classify_tool_error("File write failed", {"recoverable": True})
        assert ctx.category == ErrorCategory.TOOL
        assert ctx.recoverable is True

    def test_tool_error_unrecoverable(self):
        handler = ErrorHandler()
        ctx = handler.classify_tool_error("Permission denied", {"recoverable": False})
        assert ctx.category == ErrorCategory.UNRECOVERABLE
        assert ctx.recoverable is False

    def test_tool_error_unrecoverable_by_pattern(self):
        handler = ErrorHandler()
        ctx = handler.classify_tool_error("Permission denied")
        assert ctx.category == ErrorCategory.UNRECOVERABLE

    def test_tool_error_path_traversal(self):
        handler = ErrorHandler()
        ctx = handler.classify_tool_error("Path traversal detected")
        assert ctx.category == ErrorCategory.UNRECOVERABLE

    def test_tool_error_disk_full(self):
        handler = ErrorHandler()
        ctx = handler.classify_tool_error("No space left on device")
        assert ctx.category == ErrorCategory.UNRECOVERABLE

    def test_tool_error_recoverable_default(self):
        handler = ErrorHandler()
        ctx = handler.classify_tool_error("Temporary network error")
        assert ctx.category == ErrorCategory.TOOL
        assert ctx.recoverable is True

    def test_llm_error_recoverable(self):
        handler = ErrorHandler(max_retries=3)
        ctx = handler.classify_llm_error("Rate limit", attempt=1, max_retries=3)
        assert ctx.category == ErrorCategory.LLM_API
        assert ctx.recoverable is True
        assert ctx.retry_count == 2

    def test_llm_error_unrecoverable_at_max(self):
        handler = ErrorHandler(max_retries=3)
        ctx = handler.classify_llm_error("Connection timeout", attempt=2, max_retries=3)
        assert ctx.category == ErrorCategory.LLM_API
        assert ctx.recoverable is False

    def test_stuck_detection_similar_summaries(self):
        handler = ErrorHandler()
        handler._previous_summaries["agent-1"] = (
            "Fixed the bug in the authentication module by updating the token validation logic"
        )

        ctx = handler.detect_stuck_agent(
            "agent-1",
            "Fixed the bug in the authentication module by updating the token validation logic",
        )

        assert ctx is not None
        assert ctx.category == ErrorCategory.AGENT_STUCK
        assert ctx.recoverable is False

    def test_stuck_detection_different_summaries(self):
        handler = ErrorHandler()
        handler._previous_summaries["agent-1"] = "Fixed auth bug"

        ctx = handler.detect_stuck_agent(
            "agent-1",
            "Added new payment module with Stripe integration",
        )

        assert ctx is None

    def test_stuck_detection_empty_previous(self):
        handler = ErrorHandler()
        ctx = handler.detect_stuck_agent("agent-1", "Some summary")
        assert ctx is None

    def test_classify_workspace_error(self):
        handler = ErrorHandler()
        ctx = handler.classify_workspace_error("Invalid JSON in workspace")
        assert ctx.category == ErrorCategory.WORKSPACE_CORRUPTION
        assert ctx.recoverable is False

    def test_reset_agent_similarity(self):
        handler = ErrorHandler()
        handler._previous_summaries["agent-1"] = "Old summary"
        handler.reset_agent_similarity("agent-1")
        assert "agent-1" not in handler._previous_summaries

    def test_clear_all_similarity(self):
        handler = ErrorHandler()
        handler._previous_summaries["agent-1"] = "A"
        handler._previous_summaries["agent-2"] = "B"
        handler.clear_all_similarity()
        assert len(handler._previous_summaries) == 0


class TestErrorHandlerDecisions:
    def test_workspace_corruption_triggers_backup(self):
        handler = ErrorHandler()
        ctx = ErrorContext(
            category=ErrorCategory.WORKSPACE_CORRUPTION,
            message="Corrupt file",
        )
        decision = handler.decide(ctx)
        assert decision.strategy == RecoveryStrategy.BACKUP_AND_RECREATE

    def test_agent_stuck_triggers_checkpoint(self):
        handler = ErrorHandler()
        ctx = ErrorContext(
            category=ErrorCategory.AGENT_STUCK,
            message="Agent stuck",
        )
        decision = handler.decide(ctx)
        assert decision.strategy == RecoveryStrategy.CHECKPOINT

    def test_llm_api_recoverable_triggers_retry(self):
        handler = ErrorHandler(max_retries=3)
        ctx = ErrorContext(
            category=ErrorCategory.LLM_API,
            message="Rate limit",
            recoverable=True,
            retry_count=1,
            max_retries=3,
        )
        decision = handler.decide(ctx)
        assert decision.strategy == RecoveryStrategy.RETRY

    def test_llm_api_unrecoverable_triggers_checkpoint(self):
        handler = ErrorHandler(max_retries=3)
        ctx = ErrorContext(
            category=ErrorCategory.LLM_API,
            message="Max retries",
            recoverable=False,
            retry_count=3,
            max_retries=3,
        )
        decision = handler.decide(ctx)
        assert decision.strategy == RecoveryStrategy.CHECKPOINT

    def test_tool_unrecoverable_triggers_checkpoint(self):
        handler = ErrorHandler()
        ctx = ErrorContext(
            category=ErrorCategory.UNRECOVERABLE,
            message="Permission denied",
            recoverable=False,
        )
        decision = handler.decide(ctx)
        assert decision.strategy == RecoveryStrategy.CHECKPOINT

    def test_tool_recoverable_with_retries_triggers_retry(self):
        handler = ErrorHandler(max_retries=3)
        ctx = ErrorContext(
            category=ErrorCategory.TOOL,
            message="File busy",
            recoverable=True,
            retry_count=0,
        )
        decision = handler.decide(ctx)
        assert decision.strategy == RecoveryStrategy.RETRY

    def test_tool_recoverable_max_retries_triggers_checkpoint(self):
        handler = ErrorHandler(max_retries=3)
        ctx = ErrorContext(
            category=ErrorCategory.TOOL,
            message="File busy",
            recoverable=True,
            retry_count=3,
            max_retries=3,
        )
        decision = handler.decide(ctx)
        assert decision.strategy == RecoveryStrategy.CHECKPOINT
