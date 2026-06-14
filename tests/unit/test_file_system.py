from pathlib import Path

from coding_assistant.tools.file_system import FileSystemTool


class TestFileSystemTool:
    def test_write_and_read(self, tmp_path: Path):
        fs = FileSystemTool(tmp_path)
        result = fs.write("hello.txt", "Hello World")
        assert result.success

        result = fs.read("hello.txt")
        assert result.success
        assert result.output == "Hello World"

    def test_read_nonexistent(self, tmp_path: Path):
        fs = FileSystemTool(tmp_path)
        result = fs.read("nonexistent.txt")
        assert not result.success
        assert "not found" in result.error.lower()

    def test_write_creates_parent_dirs(self, tmp_path: Path):
        fs = FileSystemTool(tmp_path)
        result = fs.write("src/app/main.py", "print('hello')")
        assert result.success
        assert (tmp_path / "src" / "app" / "main.py").exists()

    def test_list_dir(self, tmp_path: Path):
        fs = FileSystemTool(tmp_path)
        fs.write("a.txt", "a")
        fs.write("b.txt", "b")
        result = fs.list_dir()
        assert result.success
        assert "a.txt" in result.output
        assert "b.txt" in result.output

    def test_list_nonexistent_dir(self, tmp_path: Path):
        fs = FileSystemTool(tmp_path)
        result = fs.list_dir("nonexistent")
        assert not result.success

    def test_delete_file(self, tmp_path: Path):
        fs = FileSystemTool(tmp_path)
        fs.write("temp.txt", "temp")
        result = fs.delete("temp.txt")
        assert result.success
        assert not (tmp_path / "temp.txt").exists()

    def test_delete_nonexistent(self, tmp_path: Path):
        fs = FileSystemTool(tmp_path)
        result = fs.delete("nonexistent.txt")
        assert not result.success

    def test_exists_check(self, tmp_path: Path):
        fs = FileSystemTool(tmp_path)
        assert not fs.exists("test.txt").metadata["exists"]
        fs.write("test.txt", "content")
        assert fs.exists("test.txt").metadata["exists"]

    def test_path_traversal_blocked(self, tmp_path: Path):
        fs = FileSystemTool(tmp_path)
        result = fs.read("../../../etc/passwd")
        assert not result.success
        assert "traversal" in result.error.lower()

    def test_write_path_traversal_blocked(self, tmp_path: Path):
        fs = FileSystemTool(tmp_path)
        result = fs.write("../../../tmp/evil.txt", "evil")
        assert not result.success

    def test_get_tool_schemas(self, tmp_path: Path):
        fs = FileSystemTool(tmp_path)
        schemas = fs.get_tool_schemas()
        assert len(schemas) == 4
        names = [s["function"]["name"] for s in schemas]
        assert "read_file" in names
        assert "write_file" in names
        assert "list_dir" in names
        assert "delete_file" in names

    def test_overwrite_file(self, tmp_path: Path):
        fs = FileSystemTool(tmp_path)
        fs.write("file.txt", "original")
        fs.write("file.txt", "updated")
        result = fs.read("file.txt")
        assert result.output == "updated"

    def test_delete_directory(self, tmp_path: Path):
        fs = FileSystemTool(tmp_path)
        fs.write("subdir/file.txt", "content")
        result = fs.delete("subdir")
        assert result.success
        assert not (tmp_path / "subdir").exists()
