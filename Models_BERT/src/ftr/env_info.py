"""
Captura de informações do ambiente de execução (para reprodutibilidade).

Gera um dicionário ``env.json`` com versões de bibliotecas, hardware, CUDA e
seeds — salvo em cada pasta de experimento. Tudo aqui é tolerante a ausência de
bibliotecas (torch/transformers podem não estar instalados localmente).
"""

from __future__ import annotations

import importlib
import platform
import sys
from typing import Any

# Bibliotecas cujas versões importam para reproduzir os experimentos.
_TRACKED_PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "transformers",
    "datasets",
    "evaluate",
    "tokenizers",
    "accelerate",
    "sentencepiece",
    "matplotlib",
    "seaborn",
    "spacy",
    "tqdm",
    "pyyaml",
]

# Mapeia nome do pacote (pip) -> nome do módulo importável quando diferem.
_IMPORT_NAME = {
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
}


def _package_version(pkg: str) -> str | None:
    """Retorna a versão instalada de um pacote, ou ``None`` se ausente."""
    module_name = _IMPORT_NAME.get(pkg, pkg)
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", "desconhecida")
    except Exception:
        try:
            from importlib.metadata import version

            return version(pkg)
        except Exception:
            return None


def collect_package_versions() -> dict[str, str | None]:
    """Coleta as versões das bibliotecas rastreadas."""
    return {pkg: _package_version(pkg) for pkg in _TRACKED_PACKAGES}


def collect_hardware_info() -> dict[str, Any]:
    """Coleta informações de CPU/GPU/CUDA disponíveis."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "cpu_count": None,
        "ram_gb": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_names": [],
        "gpu_memory_gb": [],
    }

    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass

    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass

    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpu_names"].append(props.name)
                info["gpu_memory_gb"].append(round(props.total_memory / 1e9, 1))
    except Exception:
        pass

    return info


def collect_environment(seed: int | None = None) -> dict[str, Any]:
    """Snapshot completo do ambiente para registro num experimento."""
    from .env import describe_environment

    return {
        "seed": seed,
        "packages": collect_package_versions(),
        "hardware": collect_hardware_info(),
        "paths": describe_environment(),
    }
