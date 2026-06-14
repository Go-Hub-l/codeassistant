from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    success: bool
    output: str = ""
    error: str = ""
    metadata: dict[str, Any] = {}


class FileSystemTool:
    DANGEROUS_PATTERNS = {"..", "~", "/etc", "/usr", "/bin", "/sbin", "/var", "/sys", "/proc"}

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()

    def _resolve_path(self, path: str) -> Path:
        target = (self.project_root / path).resolve()
        if not str(target).startswith(str(self.project_root)):
            raise PermissionError(f"Path traversal detected: {path}")
        return target

    def _is_safe_path(self, path: str) -> bool:
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in Path(path).parts:
                return False
        return True

    def read(self, path: str) -> ToolResult:
        try:
            target = self._resolve_path(path)
            if not target.exists():
                return ToolResult(success=False, error=f"File not found: {path}")
            if not target.is_file():
                return ToolResult(success=False, error=f"Not a file: {path}")
            content = target.read_text(encoding="utf-8")
            return ToolResult(success=True, output=content)
        except PermissionError as e:
            return ToolResult(success=False, error=str(e), metadata={"recoverable": False})
        except Exception as e:
            return ToolResult(success=False, error=str(e), metadata={"recoverable": True})

    def write(self, path: str, content: str) -> ToolResult:
        try:
            target = self._resolve_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult(success=True, output=f"Written {len(content)} chars to {path}")
        except PermissionError as e:
            return ToolResult(success=False, error=str(e), metadata={"recoverable": False})
        except Exception as e:
            return ToolResult(success=False, error=str(e), metadata={"recoverable": True})

    def list_dir(self, path: str = ".") -> ToolResult:
        try:
            target = self._resolve_path(path)
            if not target.exists():
                return ToolResult(success=False, error=f"Directory not found: {path}")
            if not target.is_dir():
                return ToolResult(success=False, error=f"Not a directory: {path}")
            entries = sorted(
                str(p.relative_to(self.project_root)) for p in target.iterdir()
            )
            return ToolResult(
                success=True,
                output="\n".join(entries),
                metadata={"count": len(entries)},
            )
        except PermissionError as e:
            return ToolResult(success=False, error=str(e), metadata={"recoverable": False})
        except Exception as e:
            return ToolResult(success=False, error=str(e), metadata={"recoverable": True})

    def delete(self, path: str) -> ToolResult:
        try:
            target = self._resolve_path(path)
            if not target.exists():
                return ToolResult(success=False, error=f"Not found: {path}")
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            return ToolResult(success=True, output=f"Deleted: {path}")
        except PermissionError as e:
            return ToolResult(success=False, error=str(e), metadata={"recoverable": False})
        except Exception as e:
            return ToolResult(success=False, error=str(e), metadata={"recoverable": True})

    def exists(self, path: str) -> ToolResult:
        target = self._resolve_path(path)
        return ToolResult(
            success=True,
            output=str(target.exists()),
            metadata={"exists": target.exists()},
        )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        root = str(self.project_root)
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": f"Read file content (project dir: {root})",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path within the project",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": f"Write content to file (project dir: {root})",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path within the project",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "List directory contents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path (default: project root)",
                                "default": ".",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Delete a file or directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to delete",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
        ]
