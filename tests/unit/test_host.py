import asyncio
from pathlib import Path

from coding_assistant.agents.base import Agent
from coding_assistant.agents.registry import AgentRegistry
from coding_assistant.core.host import Host, HostAction
from coding_assistant.core.pipeline import PipelinePhase
from coding_assistant.core.types import AgentRole, HandoffResult, HandoffStatus, Severity
from coding_assistant.core.workspace_manager import WorkspaceManager


def _make_host(tmp_path: Path) -> Host:
    wm = WorkspaceManager(tmp_path)
    wm.create("test-project")
    registry = AgentRegistry()
    for role in AgentRole:
        registry.register(Agent(role=role))
    return Host(registry=registry, workspace_manager=wm, max_retries=3)


class TestHostScheduling:
    def test_initial_phase_is_requirements(self, tmp_path: Path):
        host = _make_host(tmp_path)
        assert host.state.current_phase == PipelinePhase.REQUIREMENTS

    def test_get_current_agent(self, tmp_path: Path):
        host = _make_host(tmp_path)
        agent = host.get_current_agent()
        assert agent.role == AgentRole.PM

    def test_get_next_phase(self, tmp_path: Path):
        host = _make_host(tmp_path)
        assert host.get_next_phase(PipelinePhase.REQUIREMENTS) == PipelinePhase.ARCHITECTURE
        assert host.get_next_phase(PipelinePhase.ARCHITECTURE) == PipelinePhase.DEVELOPMENT
        assert host.get_next_phase(PipelinePhase.TESTING) == PipelinePhase.DOCUMENTATION
        assert host.get_next_phase(PipelinePhase.GIT_COMMIT) == PipelinePhase.DONE

    def test_advance_to(self, tmp_path: Path):
        host = _make_host(tmp_path)
        host.advance_to(PipelinePhase.ARCHITECTURE)
        assert host.state.current_phase == PipelinePhase.ARCHITECTURE
        assert PipelinePhase.REQUIREMENTS in host.state.completed_phases


class TestHostDecisions:
    def test_completed_handoff_advances(self, tmp_path: Path):
        host = _make_host(tmp_path)
        handoff = HandoffResult(status=HandoffStatus.COMPLETED, summary="Requirements done")
        decision = host.decide_next_action(handoff)
        assert decision.action == HostAction.CHECKPOINT_THEN_ADVANCE
        assert decision.next_phase == PipelinePhase.ARCHITECTURE

    def test_minor_issue_retries(self, tmp_path: Path):
        host = _make_host(tmp_path)
        host.advance_to(PipelinePhase.REVIEW)
        handoff = HandoffResult(
            status=HandoffStatus.COMPLETED,
            summary="Code style issues",
            severity=Severity.MINOR,
        )
        decision = host.decide_next_action(handoff)
        assert decision.action == HostAction.RETRY_DEV
        assert host.state.retry_count == 1

    def test_critical_issue_checkpoint(self, tmp_path: Path):
        host = _make_host(tmp_path)
        host.advance_to(PipelinePhase.REVIEW)
        handoff = HandoffResult(
            status=HandoffStatus.COMPLETED,
            summary="Security vulnerability",
            severity=Severity.CRITICAL,
        )
        decision = host.decide_next_action(handoff)
        assert decision.action == HostAction.CHECKPOINT

    def test_max_retries_triggers_checkpoint(self, tmp_path: Path):
        host = _make_host(tmp_path)
        host.advance_to(PipelinePhase.REVIEW)
        host.state.retry_count = 3
        handoff = HandoffResult(
            status=HandoffStatus.COMPLETED,
            summary="Still minor issues",
            severity=Severity.MINOR,
        )
        decision = host.decide_next_action(handoff)
        assert decision.action == HostAction.CHECKPOINT

    def test_incomplete_handoff_force_handoff(self, tmp_path: Path):
        host = _make_host(tmp_path)
        handoff = HandoffResult(status=HandoffStatus.INCOMPLETE, summary="Agent stopped")
        decision = host.decide_next_action(handoff)
        assert decision.action == HostAction.FORCE_HANDOFF

    def test_failed_handoff_checkpoint(self, tmp_path: Path):
        host = _make_host(tmp_path)
        handoff = HandoffResult(status=HandoffStatus.FAILED, summary="LLM error")
        decision = host.decide_next_action(handoff)
        assert decision.action == HostAction.CHECKPOINT

    def test_stuck_detection(self, tmp_path: Path):
        host = _make_host(tmp_path)
        host._previous_summary = "Fixed the bug in auth module"
        handoff = HandoffResult(
            status=HandoffStatus.COMPLETED,
            summary="Fixed the bug in auth module",
        )
        decision = host.decide_next_action(handoff)
        assert decision.action == HostAction.CHECKPOINT
        assert "stuck" in decision.reason.lower()

    def test_pipeline_complete(self, tmp_path: Path):
        host = _make_host(tmp_path)
        host.advance_to(PipelinePhase.GIT_COMMIT)
        handoff = HandoffResult(status=HandoffStatus.COMPLETED, summary="Committed")
        decision = host.decide_next_action(handoff)
        assert decision.action == HostAction.COMPLETE


class TestHostHandoffHandling:
    def test_handle_handoff_saves_workspace(self, tmp_path: Path):
        host = _make_host(tmp_path)
        handoff = HandoffResult(status=HandoffStatus.COMPLETED, summary="PM analysis done")
        asyncio.run(host.handle_handoff(handoff))
        assert host.workspace.progress.phase_summaries.get("requirements") == "PM analysis done"

    def test_handle_forced_handoff(self, tmp_path: Path):
        host = _make_host(tmp_path)
        decision = asyncio.run(host.handle_forced_handoff("Agent timed out"))
        assert decision.action == HostAction.FORCE_HANDOFF


class TestHostIteration:
    def test_initial_iteration_count_is_zero(self, tmp_path: Path):
        host = _make_host(tmp_path)
        assert host.state.iteration_count == 0

    def test_start_iteration_resets_pipeline(self, tmp_path: Path):
        host = _make_host(tmp_path)
        host.state.iteration_count = 1
        host.advance_to(PipelinePhase.ARCHITECTURE)
        host.state.retry_count = 2
        host.workspace.increment_retry_count()

        result = host.start_iteration()

        assert result is True
        assert host.state.current_phase == PipelinePhase.REQUIREMENTS
        assert host.state.retry_count == 0
        assert host.workspace.progress.retry_count == 0

    def test_get_iteration_context(self, tmp_path: Path):
        host = _make_host(tmp_path)
        host.state.iteration_count = 1
        host.advance_to(PipelinePhase.DEVELOPMENT)
        host.workspace.add_phase_summary("requirements", "PRD done")
        host.workspace.add_phase_summary("architecture", "Architecture done")

        context = host.get_iteration_context()

        assert "Iteration 2" in context
        assert "PRD done" in context
        assert "Architecture done" in context
