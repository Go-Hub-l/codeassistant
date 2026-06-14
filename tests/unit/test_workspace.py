import json
from pathlib import Path

import pytest

from coding_assistant.core.types import AgentRole
from coding_assistant.core.workspace import (
    RequirementsPartition,
    Workspace,
)
from coding_assistant.core.workspace_manager import (
    WorkspaceCorruptionError,
    WorkspaceManager,
)


class TestWorkspace:
    def test_create_empty_workspace(self):
        ws = Workspace(project_name="test-project")
        assert ws.project_name == "test-project"
        assert ws.version == 1
        assert ws.requirements.prd == ""
        assert ws.architecture.tech_stack == {}
        assert ws.code.files == []

    def test_get_partition_by_role(self):
        ws = Workspace(project_name="test")
        assert isinstance(ws.get_partition(AgentRole.PM), RequirementsPartition)
        assert isinstance(ws.get_partition(AgentRole.DEV), type(ws.code))

    def test_get_partition_for_all_roles(self):
        ws = Workspace(project_name="test")
        for role in AgentRole:
            ws.get_partition(role)

    def test_add_file_reference(self):
        ws = Workspace(project_name="test")
        ws.add_file_reference("src/main.py", "Main entry point", AgentRole.DEV)
        assert len(ws.code.files) == 1
        assert ws.code.files[0].path == "src/main.py"
        assert ws.code.files[0].created_by == AgentRole.DEV

    def test_add_file_reference_update_existing(self):
        ws = Workspace(project_name="test")
        ws.add_file_reference("src/main.py", "Old desc", AgentRole.DEV)
        ws.add_file_reference("src/main.py", "New desc", AgentRole.DEV)
        assert len(ws.code.files) == 1
        assert ws.code.files[0].description == "New desc"
        assert ws.code.files[0].modified is True

    def test_add_phase_summary(self):
        ws = Workspace(project_name="test")
        ws.add_phase_summary("requirements", "Analyzed user auth system")
        assert ws.progress.phase_summaries["requirements"] == "Analyzed user auth system"

    def test_record_decision(self):
        ws = Workspace(project_name="test")
        ws.record_decision("Use FastAPI", "Lightweight and async-friendly", AgentRole.ARCHITECT)
        assert len(ws.progress.decisions) == 1
        assert ws.progress.decisions[0]["title"] == "Use FastAPI"
        assert ws.progress.decisions[0]["decided_by"] == "architect"

    def test_retry_count(self):
        ws = Workspace(project_name="test")
        assert ws.progress.retry_count == 0
        count = ws.increment_retry_count()
        assert count == 1
        ws.increment_retry_count()
        assert ws.progress.retry_count == 2
        ws.reset_retry_count()
        assert ws.progress.retry_count == 0

    def test_serialization_roundtrip(self):
        ws = Workspace(project_name="test-project")
        ws.requirements.prd = "Build a TODO app"
        ws.add_file_reference("src/main.py", "Main file", AgentRole.DEV)
        ws.add_phase_summary("requirements", "Done")

        data = ws.model_dump(mode="json")
        json_str = json.dumps(data)
        loaded = json.loads(json_str)
        ws2 = Workspace.model_validate(loaded)

        assert ws2.project_name == "test-project"
        assert ws2.requirements.prd == "Build a TODO app"
        assert len(ws2.code.files) == 1
        assert ws2.progress.phase_summaries["requirements"] == "Done"

    def test_version_field(self):
        ws = Workspace(project_name="test")
        assert ws.version == 1


class TestWorkspaceManager:
    def test_create_workspace(self, tmp_path: Path):
        manager = WorkspaceManager(tmp_path)
        ws = manager.create("my-project")
        assert ws.project_name == "my-project"
        assert manager.workspace_file.exists()

    def test_load_workspace(self, tmp_path: Path):
        manager = WorkspaceManager(tmp_path)
        manager.create("my-project")
        manager.workspace.requirements.prd = "Build auth"
        manager.save()

        manager2 = WorkspaceManager(tmp_path)
        ws = manager2.load()
        assert ws.project_name == "my-project"
        assert ws.requirements.prd == "Build auth"

    def test_load_missing_raises(self, tmp_path: Path):
        manager = WorkspaceManager(tmp_path)
        with pytest.raises(FileNotFoundError):
            manager.load()

    def test_save_creates_directory(self, tmp_path: Path):
        manager = WorkspaceManager(tmp_path)
        assert not manager.workspace_dir.exists()
        manager.create("test")
        assert manager.workspace_dir.exists()

    def test_exists(self, tmp_path: Path):
        manager = WorkspaceManager(tmp_path)
        assert not manager.exists()
        manager.create("test")
        assert manager.exists()

    def test_migration_adds_version(self, tmp_path: Path):
        manager = WorkspaceManager(tmp_path)
        manager.create("test")

        with open(manager.workspace_file) as f:
            data = json.load(f)
        del data["version"]
        with open(manager.workspace_file, "w") as f:
            json.dump(data, f)

        manager2 = WorkspaceManager(tmp_path)
        ws = manager2.load()
        assert ws.version == 1

    def test_access_before_load_raises(self, tmp_path: Path):
        manager = WorkspaceManager(tmp_path)
        with pytest.raises(RuntimeError, match="not loaded"):
            _ = manager.workspace


class TestWorkspaceCorruption:
    def test_invalid_json_raises_corruption_error(self, tmp_path: Path):
        manager = WorkspaceManager(tmp_path)
        manager.create("test")
        with open(manager.workspace_file, "w") as f:
            f.write("this is not valid json {{{")

        manager2 = WorkspaceManager(tmp_path)
        with pytest.raises(WorkspaceCorruptionError, match="invalid JSON"):
            manager2.load()

    def test_corruption_creates_backup(self, tmp_path: Path):
        manager = WorkspaceManager(tmp_path)
        manager.create("test")
        with open(manager.workspace_file, "w") as f:
            f.write("corrupted content")

        manager2 = WorkspaceManager(tmp_path)
        try:
            manager2.load()
        except WorkspaceCorruptionError as e:
            assert e.backup_path is not None
            assert ".corrupt." in e.backup_path
            assert Path(e.backup_path).exists()

    def test_empty_file_raises_corruption_error(self, tmp_path: Path):
        manager = WorkspaceManager(tmp_path)
        manager.create("test")
        with open(manager.workspace_file, "w") as f:
            f.write("")

        manager2 = WorkspaceManager(tmp_path)
        with pytest.raises(WorkspaceCorruptionError, match="empty"):
            manager2.load()

    def test_invalid_schema_raises_corruption_error(self, tmp_path: Path):
        manager = WorkspaceManager(tmp_path)
        manager.create("test")
        with open(manager.workspace_file, "w") as f:
            json.dump({"version": 1, "project_name": "test", "requirements": 12345}, f)

        manager2 = WorkspaceManager(tmp_path)
        with pytest.raises(WorkspaceCorruptionError, match="validation"):
            manager2.load()

    def test_try_load_falls_back_on_corruption(self, tmp_path: Path):
        manager = WorkspaceManager(tmp_path)
        manager.create("test")
        with open(manager.workspace_file, "w") as f:
            f.write("corrupted")

        manager2 = WorkspaceManager(tmp_path)
        ws = manager2.try_load(project_name="recovered")
        assert ws is not None
        assert ws.project_name == "recovered"

    def test_try_load_falls_back_on_missing(self, tmp_path: Path):
        manager = WorkspaceManager(tmp_path)
        ws = manager.try_load(project_name="new-project")
        assert ws is not None
        assert ws.project_name == "new-project"

    def test_save_uses_atomic_write(self, tmp_path: Path):
        manager = WorkspaceManager(tmp_path)
        manager.create("test")
        tmp_file = manager.workspace_file.with_suffix(".tmp")
        assert not tmp_file.exists()
        assert manager.workspace_file.exists()

    def test_multiple_corrupt_files_get_unique_backups(self, tmp_path: Path):
        manager = WorkspaceManager(tmp_path)
        manager.create("test")

        for content in ["corrupt1", "corrupt2"]:
            with open(manager.workspace_file, "w") as f:
                f.write(content)

            m = WorkspaceManager(tmp_path)
            try:
                m.load()
            except WorkspaceCorruptionError:
                pass

        backup_files = sorted(manager.workspace_dir.glob("workspace.corrupt.*.json"))
        assert len(backup_files) >= 1
