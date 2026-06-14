from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path.home() / ".coding-assistant"
CONFIG_FILE = CONFIG_DIR / "config.yaml"


def resolve_api_key() -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key

    config = load_config()
    api_key = config.get("api_key")
    if api_key:
        return api_key

    return None


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or CONFIG_FILE
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def save_config(config: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def prompt_api_key_interactive() -> str | None:
    try:
        api_key = input("Enter your OpenAI API key: ").strip()
        if api_key:
            save_config({"api_key": api_key})
            return api_key
    except (EOFError, KeyboardInterrupt):
        pass
    return None
