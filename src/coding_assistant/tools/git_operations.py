from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import git
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GitResult(BaseModel):
    success: bool
    output: str = ""
    error: str = ""


class GitTool:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._repo: git.Repo | None = None

    @property
    def repo(self) -> git.Repo:
        if self._repo is None:
            try:
                self._repo = git.Repo(self.project_root)
            except git.InvalidGitRepositoryError:
                self._repo = git.Repo.init(self.project_root)
        return self._repo

    def init(self) -> GitResult:
        try:
            if (self.project_root / ".git").exists():
                return GitResult(success=True, output="Git repository already exists")
            self._repo = git.Repo.init(self.project_root)
            return GitResult(success=True, output="Git repository initialized")
        except Exception as e:
            return GitResult(success=False, error=str(e))

    def add(self, paths: list[str] | None = None) -> GitResult:
        try:
            if paths:
                self.repo.index.add(paths)
            else:
                self.repo.index.add("*")
            return GitResult(success=True, output=f"Added {len(paths) if paths else 'all'} files")
        except Exception as e:
            return GitResult(success=False, error=str(e))

    def commit(self, message: str) -> GitResult:
        try:
            if not self.repo.is_dirty() and not self.repo.untracked_files:
                return GitResult(success=True, output="Nothing to commit")
            self.repo.index.add("*")
            commit = self.repo.index.commit(message)
            return GitResult(success=True, output=f"Committed: {commit.hexsha[:8]}")
        except Exception as e:
            return GitResult(success=False, error=str(e))

    def status(self) -> GitResult:
        try:
            dirty = self.repo.is_dirty()
            untracked = self.repo.untracked_files
            output_parts = []
            if dirty:
                output_parts.append("Modified files detected")
            if untracked:
                output_parts.append(f"{len(untracked)} untracked files")
            if not output_parts:
                output_parts.append("Working tree clean")
            return GitResult(success=True, output="; ".join(output_parts))
        except Exception as e:
            return GitResult(success=False, error=str(e))

    def log(self, max_count: int = 10) -> GitResult:
        try:
            commits = list(self.repo.iter_commits(max_count=max_count))
            lines = []
            for c in commits:
                lines.append(f"{c.hexsha[:8]} {c.message.split(chr(10))[0]}")
            return GitResult(success=True, output="\n".join(lines))
        except Exception as e:
            return GitResult(success=False, error=str(e))

    def create_branch(self, name: str) -> GitResult:
        try:
            self.repo.create_head(name)
            return GitResult(success=True, output=f"Branch '{name}' created")
        except Exception as e:
            return GitResult(success=False, error=str(e))

    def checkout(self, branch: str) -> GitResult:
        try:
            self.repo.heads[branch].checkout()
            return GitResult(success=True, output=f"Checked out '{branch}'")
        except Exception as e:
            return GitResult(success=False, error=str(e))

    def current_branch(self) -> GitResult:
        try:
            return GitResult(success=True, output=self.repo.active_branch.name)
        except Exception as e:
            return GitResult(success=False, error=str(e))

    def commit_at_checkpoint(
        self, phase_name: str, summary: str
    ) -> GitResult:
        message = f"[{phase_name}] {summary}"
        return self.commit(message)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "git_init",
                    "description": "Initialize a git repository",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_commit",
                    "description": "Stage all changes and commit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Commit message",
                            },
                        },
                        "required": ["message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_status",
                    "description": "Check git status",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_log",
                    "description": "View recent commit history",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "max_count": {
                                "type": "integer",
                                "description": "Max commits to show (default: 10)",
                                "default": 10,
                            },
                        },
                    },
                },
            },
        ]
