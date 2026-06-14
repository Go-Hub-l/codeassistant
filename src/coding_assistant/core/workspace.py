from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from coding_assistant.core.types import AgentRole

WORKSPACE_VERSION = 1


class CodeFileEntry(BaseModel):
    path: str
    description: str = ""
    created_by: AgentRole | None = None
    modified: bool = False


class RequirementsPartition(BaseModel):
    prd: str = ""
    user_stories: list[dict[str, Any]] = Field(default_factory=list)
    feature_list: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""


class ArchitecturePartition(BaseModel):
    tech_stack: dict[str, str] = Field(default_factory=dict)
    project_structure: str = ""
    api_contracts: list[dict[str, Any]] = Field(default_factory=list)
    database_schema: str = ""
    security_considerations: str = ""
    summary: str = ""


class CodePartition(BaseModel):
    files: list[CodeFileEntry] = Field(default_factory=list)
    summary: str = ""


class ReviewPartition(BaseModel):
    issues: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
    severity: str | None = None


class TestPartition(BaseModel):
    test_cases: list[dict[str, Any]] = Field(default_factory=list)
    results: dict[str, Any] = Field(default_factory=dict)
    coverage: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    severity: str | None = None


class ProgressPartition(BaseModel):
    milestones: list[dict[str, Any]] = Field(default_factory=list)
    current_phase: str = ""
    phase_summaries: dict[str, str] = Field(default_factory=dict)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    retry_count: int = 0


class Workspace(BaseModel):
    version: int = WORKSPACE_VERSION
    project_name: str = ""
    requirements: RequirementsPartition = Field(default_factory=RequirementsPartition)
    architecture: ArchitecturePartition = Field(default_factory=ArchitecturePartition)
    code: CodePartition = Field(default_factory=CodePartition)
    review: ReviewPartition = Field(default_factory=ReviewPartition)
    test: TestPartition = Field(default_factory=TestPartition)
    progress: ProgressPartition = Field(default_factory=ProgressPartition)

    def get_partition(self, role: AgentRole) -> BaseModel:
        partition_map: dict[AgentRole, str] = {
            AgentRole.PM: "requirements",
            AgentRole.ARCHITECT: "architecture",
            AgentRole.DEV: "code",
            AgentRole.REVIEWER: "review",
            AgentRole.QA: "test",
            AgentRole.PMGR: "progress",
        }
        partition_name = partition_map.get(role)
        if partition_name is None:
            raise ValueError(f"No partition defined for role '{role.value}'")
        return getattr(self, partition_name)

    def add_file_reference(
        self, path: str, description: str = "", created_by: AgentRole | None = None
    ) -> None:
        entry = CodeFileEntry(path=path, description=description, created_by=created_by)
        for existing in self.code.files:
            if existing.path == path:
                existing.description = description
                existing.modified = True
                return
        self.code.files.append(entry)

    def add_phase_summary(self, phase: str, summary: str) -> None:
        self.progress.phase_summaries[phase] = summary

    def record_decision(self, title: str, rationale: str, decided_by: AgentRole) -> None:
        self.progress.decisions.append({
            "title": title,
            "rationale": rationale,
            "decided_by": decided_by.value,
        })

    def increment_retry_count(self) -> int:
        self.progress.retry_count += 1
        return self.progress.retry_count

    def reset_retry_count(self) -> None:
        self.progress.retry_count = 0
