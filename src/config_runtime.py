"""
Runtime config path helpers.

This module centralizes how scripts resolve the active config file so pipeline
steps can run against temporary per-step config snapshots without mutating
config/config.yaml on disk.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml


DEFAULT_CONFIG_PATH = "config/config.yaml"
RUNTIME_CONFIG_ENV_VAR = "DS_CONFIG_PATH"


def get_config_path(explicit_path: Optional[str] = None) -> str:
    """
    Resolve the effective config path.

    Priority:
    1) explicit_path (if provided)
    2) DS_CONFIG_PATH environment variable
    3) default repo config path
    """
    runtime_path = os.getenv(RUNTIME_CONFIG_ENV_VAR, "").strip()
    if explicit_path:
        explicit_str = str(explicit_path)
        if runtime_path and explicit_str.strip() == DEFAULT_CONFIG_PATH:
            return runtime_path
        return explicit_str
    if runtime_path:
        return runtime_path
    return DEFAULT_CONFIG_PATH


def load_config(explicit_path: Optional[str] = None) -> dict[str, Any]:
    """Load YAML config from the effective config path."""
    config_path = get_config_path(explicit_path)
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_config(config_data: dict[str, Any], explicit_path: Optional[str] = None) -> str:
    """Write YAML config to the effective config path and return that path."""
    config_path = get_config_path(explicit_path)
    path_obj = Path(config_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as file:
        yaml.safe_dump(config_data, file, sort_keys=False)
    return config_path
