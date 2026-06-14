import os
from unittest.mock import patch

from coding_assistant.llm.config import (
    load_config,
    resolve_api_key,
    save_config,
)


class TestResolveApiKey:
    def test_from_env_variable(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
            assert resolve_api_key() == "sk-test-key"

    def test_from_config_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("api_key: sk-config-key\n")

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("coding_assistant.llm.config.CONFIG_FILE", config_file),
        ):
            assert resolve_api_key() == "sk-config-key"

    def test_env_takes_priority_over_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("api_key: sk-config-key\n")

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-key"}),
            patch("coding_assistant.llm.config.CONFIG_FILE", config_file),
        ):
            assert resolve_api_key() == "sk-env-key"

    def test_returns_none_when_not_found(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("coding_assistant.llm.config.CONFIG_FILE", "/nonexistent/config.yaml"):
                assert resolve_api_key() is None


class TestLoadConfig:
    def test_load_existing_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("api_key: sk-test\nmodel_overrides:\n  pm: gpt-4o-mini\n")
        with patch("coding_assistant.llm.config.CONFIG_FILE", config_file):
            config = load_config()
            assert config["api_key"] == "sk-test"

    def test_load_missing_config(self, tmp_path):
        with patch("coding_assistant.llm.config.CONFIG_FILE", tmp_path / "missing.yaml"):
            config = load_config()
            assert config == {}


class TestSaveConfig:
    def test_save_creates_directory(self, tmp_path):
        config_dir = tmp_path / "new_dir"
        config_file = config_dir / "config.yaml"
        with patch("coding_assistant.llm.config.CONFIG_DIR", config_dir), \
             patch("coding_assistant.llm.config.CONFIG_FILE", config_file):
            save_config({"api_key": "sk-test"})
            assert config_file.exists()
