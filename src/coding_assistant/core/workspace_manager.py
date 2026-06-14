from __future__ import annotations

import json
import logging
from pathlib import Path

from coding_assistant.core.workspace import WORKSPACE_VERSION, Workspace

logger = logging.getLogger(__name__)

WORKSPACE_DIR = ".coding-assistant"
WORKSPACE_FILE = "workspace.json"


class WorkspaceManager:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.workspace_dir = project_root / WORKSPACE_DIR
        self.workspace_file = self.workspace_dir / WORKSPACE_FILE
        self._workspace: Workspace | None = None

    @property
    def workspace(self) -> Workspace:
        if self._workspace is None:
            raise RuntimeError("Workspace not loaded. Call load() or create() first.")
        return self._workspace

    def create(self, project_name: str) -> Workspace:
        self._workspace = Workspace(project_name=project_name)
        self.save()
        return self._workspace

    def load(self) -> Workspace:
        if not self.workspace_file.exists():
            raise FileNotFoundError(f"Workspace file not found: {self.workspace_file}")
        with open(self.workspace_file) as f:
            data = json.load(f)
        data = self._migrate(data)
        self._workspace = Workspace.model_validate(data)
        return self._workspace

    def save(self) -> None:
        if self._workspace is None:
            raise RuntimeError("No workspace to save.")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        data = self._workspace.model_dump(mode="json")
        with open(self.workspace_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug("Workspace saved to %s", self.workspace_file)

    def exists(self) -> bool:
        return self.workspace_file.exists()

    def _migrate(self, data: dict) -> dict:
        version = data.get("version", 0)
        if version == WORKSPACE_VERSION:
            return data
        if version < WORKSPACE_VERSION:
            logger.info("Migrating workspace from version %d to %d", version, WORKSPACE_VERSION)
        data["version"] = WORKSPACE_VERSION
        return data
