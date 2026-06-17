"""Loads and validates YAML configs for LLM experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_key = cfg.pop("_base", None)
    if base_key:
        base_path = path.parent / base_key
        with open(base_path, encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        cfg = _deep_merge(base, cfg)

    return cfg


def load_model_registry(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_invoke_field(model_entry: dict) -> tuple[str, str]:
    """Returns (field_name, value) to invoke the model via Bedrock Converse."""
    iw = model_entry["invokeWith"]
    return iw["field"], iw["value"]
