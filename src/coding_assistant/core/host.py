from __future__ import annotations

import difflib
import enum
import logging
from collections.abc import Awaitable, Callable

from coding_assistant.agents.base import Agent
from coding_assistant.agents.registry import AgentRegistry
from coding_assistant.core.pipeline import (
    CHECKPOINT_PHASES,
    PHASE_AGENT_MAP,
    PHASE_ORDER,
    HostState,
    PipelinePhase,
)
from coding_assistant.core.types import HandoffResult, HandoffStatus, Severity
from coding_assistant.core.workspace import Workspace
from coding_assistant.core.workspace_manager import WorkspaceManager

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.85


class HostAction(str, enum.Enum):
    ADVANCE = "advance"
    CHECKPOINT_THEN_ADVANCE = "checkpoint_then_advance"
    CHECKPOINT = "checkpoint"
    RETRY_DEV = "retry_dev"
    FORCE_HANDOFF = "force_handoff"
    COMPLETE = "complete"


class HostDecision:
    def __init__(
        self,
        action: HostAction,
        reason: str,
        next_phase: PipelinePhase | None = None,
        phase: PipelinePhase | None = None,
    ) -> None:
        self.action = action
        self.reason = reason
        self.next_phase = next_phase
        self.phase = phase or (next_phase if next_phase else None)


class Host:
    def __init__(
        self,
        registry: AgentRegistry,
        workspace_manager: WorkspaceManager,
        max_retries: int = 3,
        checkpoint_callback: Callable[[PipelinePhase, Workspace], Awaitable[str | None]]
        | None = None,
    ) -> None:
        self.registry = registry
        self.workspace_manager = workspace_manager
        self.state = HostState(max_retries=max_retries)
        self.checkpoint_callback = checkpoint_callback
        self._previous_summary: str = ""

    @property
    def workspace(self) -> Workspace:
        return self.workspace_manager.workspace

    def get_current_agent(self) -> Agent:
        role = PHASE_AGENT_MAP.get(self.state.current_phase)
        if role is None:
            raise ValueError(f"No agent for phase: {self.state.current_phase}")
        return self.registry.get(role)

    def get_next_phase(self, current: PipelinePhase) -> PipelinePhase | None:
        try:
            idx = PHASE_ORDER.index(current)
            if idx + 1 < len(PHASE_ORDER):
                return PHASE_ORDER[idx + 1]
        except ValueError:
            pass
        return None

    def decide_next_action(self, handoff: HandoffResult) -> HostDecision:
        if handoff.status == HandoffStatus.INCOMPLETE:
            return HostDecision(
                action=HostAction.FORCE_HANDOFF,
                reason=f"Agent did not complete: {handoff.summary[:200]}",
            )

        if handoff.status == HandoffStatus.FAILED:
            return HostDecision(
                action=HostAction.CHECKPOINT,
                reason=f"Agent failed: {handoff.summary[:200]}",
                phase=self.state.current_phase,
            )

        if self._is_stuck(handoff.summary):
            return HostDecision(
                action=HostAction.CHECKPOINT,
                reason="Agent appears stuck (similar output to previous attempt)",
                phase=self.state.current_phase,
            )

        if handoff.severity == Severity.CRITICAL:
            return HostDecision(
                action=HostAction.CHECKPOINT,
                reason=f"Critical issue found: {handoff.summary[:200]}",
                phase=self.state.current_phase,
            )

        if (
            handoff.severity == Severity.MINOR
            and self.state.retry_count < self.state.max_retries
        ):
            self.state.retry_count += 1
            self.workspace.increment_retry_count()
            return HostDecision(
                action=HostAction.RETRY_DEV,
                reason=f"Minor issue, retry #{self.state.retry_count}: {handoff.summary[:200]}",
            )

        if (
            handoff.severity in (Severity.MAJOR, Severity.MINOR)
            and self.state.retry_count >= self.state.max_retries
        ):
            return HostDecision(
                action=HostAction.CHECKPOINT,
                reason=f"Max retries ({self.state.max_retries}) reached",
                phase=self.state.current_phase,
            )

        next_phase = self.get_next_phase(self.state.current_phase)
        if next_phase is None or next_phase == PipelinePhase.DONE:
            return HostDecision(action=HostAction.COMPLETE, reason="Pipeline complete")

        if next_phase in CHECKPOINT_PHASES and next_phase != PipelinePhase.GIT_COMMIT:
            return HostDecision(
                action=HostAction.CHECKPOINT_THEN_ADVANCE,
                reason=f"Checkpoint required before {next_phase.value}",
                next_phase=next_phase,
            )

        self.state.retry_count = 0
        self.workspace.reset_retry_count()
        return HostDecision(
            action=HostAction.ADVANCE,
            reason=f"Phase complete, advancing to {next_phase.value}",
            next_phase=next_phase,
        )

    def advance_to(self, phase: PipelinePhase) -> None:
        if self.state.current_phase not in self.state.completed_phases:
            self.state.completed_phases.append(self.state.current_phase)
        self.state.current_phase = phase
        self.state.waiting_for_checkpoint = False
        self.state.checkpoint_phase = None

    def _is_stuck(self, current_summary: str) -> bool:
        if not self._previous_summary or not current_summary:
            self._previous_summary = current_summary
            return False
        ratio = difflib.SequenceMatcher(
            None, self._previous_summary.lower(), current_summary.lower()
        ).ratio()
        self._previous_summary = current_summary
        return ratio >= SIMILARITY_THRESHOLD

    async def handle_handoff(self, handoff: HandoffResult) -> HostDecision:
        self.decide_next_action(handoff)
        self.workspace.add_phase_summary(
            self.state.current_phase.value,
            handoff.summary,
        )
        self.workspace_manager.save()
        return self.decide_next_action(handoff)

    async def handle_forced_handoff(self, summary: str) -> HostDecision:
        forced = HandoffResult(
            status=HandoffStatus.INCOMPLETE,
            summary=summary,
        )
        return await self.handle_handoff(forced)

    async def run_checkpoint(self, phase: PipelinePhase) -> str | None:
        self.state.waiting_for_checkpoint = True
        self.state.checkpoint_phase = phase
        if self.checkpoint_callback:
            return await self.checkpoint_callback(phase, self.workspace)
        return None
