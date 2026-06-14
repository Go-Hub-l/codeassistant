from pathlib import Path

from coding_assistant.tools.git_operations import GitTool


class TestGitTool:
    def test_init_creates_repo(self, tmp_path: Path):
        git_tool = GitTool(tmp_path)
        result = git_tool.init()
        assert result.success
        assert (tmp_path / ".git").exists()

    def test_init_already_exists(self, tmp_path: Path):
        git_tool = GitTool(tmp_path)
        git_tool.init()
        result = git_tool.init()
        assert result.success
        assert "already exists" in result.output

    def test_commit(self, tmp_path: Path):
        git_tool = GitTool(tmp_path)
        git_tool.init()
        (tmp_path / "hello.py").write_text("print('hello')")
        result = git_tool.commit("Initial commit")
        assert result.success

    def test_commit_nothing(self, tmp_path: Path):
        git_tool = GitTool(tmp_path)
        git_tool.init()
        result = git_tool.commit("Empty commit")
        assert result.success
        assert "Nothing to commit" in result.output

    def test_status_clean(self, tmp_path: Path):
        git_tool = GitTool(tmp_path)
        git_tool.init()
        result = git_tool.status()
        assert result.success
        assert "clean" in result.output

    def test_status_dirty(self, tmp_path: Path):
        git_tool = GitTool(tmp_path)
        git_tool.init()
        (tmp_path / "new_file.py").write_text("x = 1")
        result = git_tool.status()
        assert result.success
        assert "untracked" in result.output.lower()

    def test_log(self, tmp_path: Path):
        git_tool = GitTool(tmp_path)
        git_tool.init()
        (tmp_path / "hello.py").write_text("print('hello')")
        git_tool.commit("First commit")
        (tmp_path / "world.py").write_text("print('world')")
        git_tool.commit("Second commit")
        result = git_tool.log()
        assert result.success
        assert "Second commit" in result.output
        assert "First commit" in result.output

    def test_create_branch(self, tmp_path: Path):
        git_tool = GitTool(tmp_path)
        git_tool.init()
        (tmp_path / "f.py").write_text("x = 1")
        git_tool.commit("init")
        result = git_tool.create_branch("develop")
        assert result.success

    def test_current_branch(self, tmp_path: Path):
        git_tool = GitTool(tmp_path)
        git_tool.init()
        (tmp_path / "f.py").write_text("x = 1")
        git_tool.commit("init")
        result = git_tool.current_branch()
        assert result.success

    def test_commit_at_checkpoint(self, tmp_path: Path):
        git_tool = GitTool(tmp_path)
        git_tool.init()
        (tmp_path / "prd.md").write_text("# PRD")
        result = git_tool.commit_at_checkpoint("requirements", "PRD generated")
        assert result.success

    def test_get_tool_schemas(self, tmp_path: Path):
        git_tool = GitTool(tmp_path)
        schemas = git_tool.get_tool_schemas()
        assert len(schemas) == 4
        names = [s["function"]["name"] for s in schemas]
        assert "git_init" in names
        assert "git_commit" in names
        assert "git_status" in names
        assert "git_log" in names
