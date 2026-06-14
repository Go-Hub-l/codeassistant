from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from coding_assistant.cli.main import cli


class TestCLI:
    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "coding assistant" in result.output.lower()

    def test_new_command_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["new", "--help"])
        assert result.exit_code == 0
        assert "PROJECT_NAME" in result.output

    def test_iter_command_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["iter", "--help"])
        assert result.exit_code == 0

    @patch("coding_assistant.cli.main.resolve_api_key", return_value=None)
    @patch("coding_assistant.cli.main.prompt_api_key_interactive", return_value=None)
    def test_new_no_api_key_exits(self, mock_prompt, mock_resolve):
        runner = CliRunner()
        result = runner.invoke(cli, ["new", "test-project"])
        assert result.exit_code != 0

    @patch("coding_assistant.cli.main.resolve_api_key", return_value="sk-test")
    def test_new_creates_project_dir(self, mock_resolve):
        runner = CliRunner()
        with runner.isolated_filesystem():
            runner.invoke(
                cli, ["new", "my-project", "--base-dir", "."], input="Build a TODO app\n"
            )
            assert (Path(".") / "my-project").exists()

    @patch("coding_assistant.cli.main.resolve_api_key", return_value="sk-test")
    def test_iter_missing_dir_exits(self, mock_resolve):
        runner = CliRunner()
        result = runner.invoke(cli, ["iter", "nonexistent-project"])
        assert result.exit_code != 0
