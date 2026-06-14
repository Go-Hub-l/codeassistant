from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class HandoffStatus(str, Enum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


class Severity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class HandoffResult(BaseModel):
    status: HandoffStatus
    summary: str
    suggested_next: str | None = None
    severity: Severity | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRole(str, Enum):
    PM = "pm"
    ARCHITECT = "architect"
    DEV = "dev"
    REVIEWER = "reviewer"
    QA = "qa"
    PMGR = "pmgr"
