from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from coding_assistant.core.types import AgentRole


class PipelinePhase(str, Enum):
    REQUIREMENTS = "requirements"
    ARCHITECTURE = "architecture"
    DEVELOPMENT = "development"
    REVIEW = "review"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    GIT_COMMIT = "git_commit"
    DONE = "done"


PHASE_AGENT_MAP: dict[PipelinePhase, AgentRole] = {
    PipelinePhase.REQUIREMENTS: AgentRole.PM,
    PipelinePhase.ARCHITECTURE: AgentRole.ARCHITECT,
    PipelinePhase.DEVELOPMENT: AgentRole.DEV,
    PipelinePhase.REVIEW: AgentRole.REVIEWER,
    PipelinePhase.TESTING: AgentRole.QA,
    PipelinePhase.DOCUMENTATION: AgentRole.DEV,
    PipelinePhase.GIT_COMMIT: AgentRole.PMGR,
}

PHASE_ORDER: list[PipelinePhase] = [
    PipelinePhase.REQUIREMENTS,
    PipelinePhase.ARCHITECTURE,
    PipelinePhase.DEVELOPMENT,
    PipelinePhase.REVIEW,
    PipelinePhase.TESTING,
    PipelinePhase.DOCUMENTATION,
    PipelinePhase.GIT_COMMIT,
    PipelinePhase.DONE,
]

ITERATION_PHASES: set[PipelinePhase] = {
    PipelinePhase.REQUIREMENTS,
    PipelinePhase.ARCHITECTURE,
    PipelinePhase.DEVELOPMENT,
    PipelinePhase.REVIEW,
    PipelinePhase.TESTING,
    PipelinePhase.DOCUMENTATION,
    PipelinePhase.GIT_COMMIT,
}

CHECKPOINT_PHASES: set[PipelinePhase] = {
    PipelinePhase.REQUIREMENTS,
    PipelinePhase.ARCHITECTURE,
    PipelinePhase.GIT_COMMIT,
}


class HostState(BaseModel):
    current_phase: PipelinePhase = PipelinePhase.REQUIREMENTS
    retry_count: int = 0
    max_retries: int = 3
    last_summary: str = ""
    waiting_for_checkpoint: bool = False
    checkpoint_phase: PipelinePhase | None = None
    completed_phases: list[PipelinePhase] = Field(default_factory=list)
    iteration_count: int = 0
