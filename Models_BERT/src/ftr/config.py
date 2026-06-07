"""
Carregamento e mesclagem de configurações de experimento (YAML).

Cada experimento é descrito por um arquivo YAML em ``configs/``. Todos herdam de
``configs/_base.yaml`` via mesclagem profunda (deep-merge), de modo que cada
config de modelo só precisa declarar o que difere do padrão.

A função :func:`load_config` retorna um dicionário simples (não um objeto
tipado), para manter o pipeline transparente e fácil de serializar como
snapshot dentro de cada pasta de experimento.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

BASE_CONFIG_NAME = "_base.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Mescla ``override`` sobre ``base`` recursivamente (sem mutar entradas)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Lê um arquivo YAML e retorna um dicionário (vazio se o arquivo for vazio)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(config_path: str | Path, configs_dir: str | Path | None = None) -> dict[str, Any]:
    """Carrega um config de experimento já mesclado com ``_base.yaml``.

    Args:
        config_path: caminho para o YAML do experimento (ou apenas o nome do
            arquivo, resolvido dentro de ``configs_dir``).
        configs_dir: diretório onde ficam os configs e o ``_base.yaml``. Se
            ``None``, usa a pasta do próprio ``config_path``.

    Returns:
        Dicionário de configuração mesclado, com a chave ``__config_name__``
        adicionada (nome do arquivo, sem extensão) para rastreabilidade.
    """
    config_path = Path(config_path)
    if configs_dir is None:
        configs_dir = config_path.parent
    configs_dir = Path(configs_dir)

    if not config_path.is_absolute() and not config_path.exists():
        config_path = configs_dir / config_path.name

    base_path = configs_dir / BASE_CONFIG_NAME
    base = load_yaml(base_path) if base_path.exists() else {}
    override = load_yaml(config_path)

    merged = _deep_merge(base, override)
    merged["__config_name__"] = config_path.stem
    return merged


def save_config_snapshot(config: dict[str, Any], dest: str | Path) -> Path:
    """Salva um snapshot do config (YAML) numa pasta de experimento."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return dest
