"""Carregamento de configs YAML com herança de _base.yaml."""
from __future__ import annotations
import copy
from pathlib import Path
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_config(config_path: str | Path, configs_dir: str | Path | None = None) -> dict:
    config_path = Path(config_path)
    if configs_dir is None:
        configs_dir = config_path.parent
    configs_dir = Path(configs_dir)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = configs_dir / config_path.name
    base_path = configs_dir / "_base.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
    override = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    merged = _deep_merge(base, override)
    merged["__config_name__"] = config_path.stem
    return merged
