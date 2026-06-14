from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from coding_assistant.core.workspace import WORKSPACE_VERSION, Workspace

logger = logging.getLogger(__name__)

WORKSPACE_DIR = ".coding-assistant"
WORKSPACE_FILE = "workspace.json"


class WorkspaceCorruptionError(Exception):
    def __init__(self, message: str, backup_path: str | None = None) -> None:
        super().__init__(message)
        self.backup_path = backup_path


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
        try:
            with open(self.workspace_file) as f:
                raw = f.read()
                if not raw.strip():
                    raise ValueError("Workspace file is empty")
                data = json.loads(raw)
        except json.JSONDecodeError as e:
            backup = self._backup_corrupt_file()
            raise WorkspaceCorruptionError(
                f"Workspace file contains invalid JSON: {e}. "
                f"Corrupt file backed up to {backup}. "
                "Delete the corrupt file and reinitialize with `coding-assistant --iter`.",
                backup_path=backup,
            ) from e
        except ValueError as e:
            backup = self._backup_corrupt_file()
            raise WorkspaceCorruptionError(
                f"Workspace file is corrupted: {e}. Corrupt file backed up to {backup}.",
                backup_path=backup,
            ) from e

        data = self._migrate(data)
        try:
            self._workspace = Workspace.model_validate(data)
        except Exception as e:
            backup = self._backup_corrupt_file()
            raise WorkspaceCorruptionError(
                f"Workspace validation failed: {e}. "
                f"Corrupt file backed up to {backup}. "
                "The file may be from an incompatible version.",
                backup_path=backup,
            ) from e

        return self._workspace

    def save(self) -> None:
        if self._workspace is None:
            raise RuntimeError("No workspace to save.")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        data = self._workspace.model_dump(mode="json")
        tmp_file = self.workspace_file.with_suffix(".tmp")
        try:
            with open(tmp_file, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp_file.replace(self.workspace_file)
        except Exception:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)
            raise
        logger.debug("Workspace saved to %s", self.workspace_file)

    def exists(self) -> bool:
        return self.workspace_file.exists()

    def try_load(self, project_name: str = "recovered_project") -> Workspace | None:
        try:
            return self.load()
        except (WorkspaceCorruptionError, FileNotFoundError):
            logger.warning("Could not load workspace, creating new one")
            return self.create(project_name)

    def _backup_corrupt_file(self) -> str:
        if not self.workspace_file.exists():
            return ""
        import time

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        micro = str(time.time_ns() % 1_000_000).zfill(6)
        backup_name = f"workspace.corrupt.{timestamp}.{micro}.json"
        backup_path = self.workspace_dir / backup_name
        shutil.copy2(self.workspace_file, backup_path)
        logger.info("Backed up corrupt workspace to %s", backup_path)
        return str(backup_path)

    def _migrate(self, data: dict) -> dict:
        version = data.get("version", 0)
        if version == WORKSPACE_VERSION:
            return data
        if version < WORKSPACE_VERSION:
            logger.info(
                "Migrating workspace from version %d to %d",
                version,
                WORKSPACE_VERSION,
            )
        data["version"] = WORKSPACE_VERSION
        return data
