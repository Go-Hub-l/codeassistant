from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ToolResult(BaseModel):
    success: bool
    output: str = ""
    error: str = ""
    metadata: dict[str, Any] = {}


class CodeExecutionTool:
    def __init__(self, project_root: Path, docker_image: str = "python:3.11-slim") -> None:
        self.project_root = project_root.resolve()
        self.docker_image = docker_image
        self._docker_available: bool | None = None

    async def check_docker(self) -> bool:
        if self._docker_available is not None:
            return self._docker_available
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            self._docker_available = proc.returncode == 0
        except FileNotFoundError:
            self._docker_available = False
        return self._docker_available

    async def execute(
        self,
        command: str,
        timeout: int = 120,
        working_dir: str | None = None,
    ) -> ToolResult:
        if not await self.check_docker():
            return ToolResult(
                success=False,
                error="Docker is not installed or not running",
                metadata={"recoverable": False},
            )

        work_dir = working_dir or "/project"
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "run",
                "--rm",
                "-v", f"{self.project_root}:/project",
                "-w", work_dir,
                "--network", "none",
                "--memory", "512m",
                "--cpus", "1",
                self.docker_image,
                "bash", "-c", command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")

            if proc.returncode == 0:
                return ToolResult(
                    success=True,
                    output=output,
                    metadata={"return_code": 0},
                )
            return ToolResult(
                success=False,
                output=output,
                error=error_output,
                metadata={"return_code": proc.returncode, "recoverable": True},
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Execution timed out after {timeout}s",
                metadata={"recoverable": True},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"recoverable": True},
            )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "description": "Execute a command in a Docker container",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Shell command to execute",
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout in seconds (default: 120)",
                                "default": 120,
                            },
                            "working_dir": {
                                "type": "string",
                                "description": (
                                    "Working directory inside container "
                                    "(default: /project)"
                                ),
                                "default": "/project",
                            },
                        },
                        "required": ["command"],
                    },
                },
            },
        ]


BLOCKED_COMMANDS = {
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=", ":(){ :|:& };:",
    "wget", "curl -o /", "chmod -R 777 /",
}

BLOCKED_PREFIXES = {"sudo ", "su ", "apt ", "yum ", "brew ", "pip install --user "}


class ShellTool:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()

    def _is_command_safe(self, command: str) -> tuple[bool, str]:
        stripped = command.strip()
        for blocked in BLOCKED_COMMANDS:
            if blocked in stripped:
                return False, f"Blocked dangerous command pattern: {blocked}"
        for prefix in BLOCKED_PREFIXES:
            if stripped.startswith(prefix):
                return False, f"Blocked command prefix: {prefix}"
        return True, ""

    async def execute(self, command: str, timeout: int = 60) -> ToolResult:
        is_safe, reason = self._is_command_safe(command)
        if not is_safe:
            return ToolResult(
                success=False,
                error=reason,
                metadata={"recoverable": False},
            )

        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")

            if proc.returncode == 0:
                return ToolResult(success=True, output=output, metadata={"return_code": 0})
            return ToolResult(
                success=False,
                output=output,
                error=error_output,
                metadata={"return_code": proc.returncode, "recoverable": True},
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Command timed out after {timeout}s",
                metadata={"recoverable": True},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"recoverable": True},
            )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_shell",
                    "description": "Run a shell command in the project directory. "
                    "Dangerous commands are blocked.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command to run"},
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout in seconds (default: 60)",
                                "default": 60,
                            },
                        },
                        "required": ["command"],
                    },
                },
            },
        ]
