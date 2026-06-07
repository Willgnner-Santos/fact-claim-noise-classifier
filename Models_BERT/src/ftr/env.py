"""
Resolução de ambiente e caminhos (Local / Google Colab / Drive).

Toda a árvore de artefatos do projeto é relativa a um único ``PROJECT_ROOT``.
Isto torna o pipeline portável: o mesmo código roda localmente (para EDA,
pré-processamento e validação de métricas) e no Google Colab (para o treino em
GPU), bastando que ``PROJECT_ROOT`` aponte para o lugar certo.

Ordem de resolução do ``PROJECT_ROOT``:

1. Variável de ambiente ``FTR_PROJECT_ROOT`` (tem prioridade máxima — útil para
   sobrescrever em qualquer ambiente).
2. Se rodando no **Google Colab**: ``<DRIVE>/MyDrive/Fato_Tese_Ruido`` caso o
   Drive já esteja montado; senão a pasta local de trabalho do Colab.
3. Caso contrário (execução **local**): a raiz do repositório, detectada subindo
   a árvore de diretórios a partir deste arquivo até encontrar marcadores do
   projeto (``configs/``, ``Data/`` ou ``src/ftr``).

Use :func:`get_paths` para obter um objeto :class:`Paths` com todas as pastas já
resolvidas e criadas sob demanda.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Nome padrão da pasta do projeto dentro do Google Drive.
DEFAULT_DRIVE_PROJECT_DIR = "Fato_Tese_Ruido"

# Marcadores que identificam a raiz do projeto em execução local.
_ROOT_MARKERS = ("configs", "src", "Data")


def in_colab() -> bool:
    """Retorna ``True`` se estivermos rodando dentro do Google Colab."""
    return "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG") is not None


def _drive_mount_point() -> Path | None:
    """Retorna o ponto de montagem do Google Drive, se já montado."""
    for candidate in (Path("/content/drive/MyDrive"), Path("/content/drive/My Drive")):
        if candidate.exists():
            return candidate
    return None


def _detect_local_root() -> Path:
    """Sobe a árvore a partir deste arquivo procurando a raiz do projeto."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if any((parent / marker).exists() for marker in _ROOT_MARKERS):
            # Garante que não paramos em src/ftr — queremos a raiz do repo.
            if (parent / "configs").exists() or (parent / "Data").exists():
                return parent
    # Fallback: dois níveis acima de src/ftr/env.py == raiz do repo.
    return here.parents[2]


def resolve_project_root() -> Path:
    """Resolve o ``PROJECT_ROOT`` segundo a ordem documentada no módulo."""
    env_override = os.environ.get("FTR_PROJECT_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()

    if in_colab():
        drive = _drive_mount_point()
        if drive is not None:
            return (drive / DEFAULT_DRIVE_PROJECT_DIR).resolve()
        # Drive ainda não montado: usa pasta de trabalho do Colab.
        return Path("/content") / DEFAULT_DRIVE_PROJECT_DIR

    return _detect_local_root()


@dataclass(frozen=True)
class Paths:
    """Caminhos canônicos do projeto, todos relativos a ``root``."""

    root: Path
    # Entradas
    data_raw: Path = field(init=False)
    data_processed: Path = field(init=False)
    configs: Path = field(init=False)
    # Saídas
    experiments: Path = field(init=False)
    results: Path = field(init=False)
    results_figures: Path = field(init=False)
    reports: Path = field(init=False)
    logs: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_raw", self.root / "Data")
        object.__setattr__(self, "data_processed", self.root / "data" / "processed")
        object.__setattr__(self, "configs", self.root / "configs")
        object.__setattr__(self, "experiments", self.root / "experiments")
        object.__setattr__(self, "results", self.root / "results")
        object.__setattr__(self, "results_figures", self.root / "results" / "figures")
        object.__setattr__(self, "reports", self.root / "reports")
        object.__setattr__(self, "logs", self.root / "logs")

    @property
    def raw_dataset(self) -> Path:
        """Caminho do dataset bruto ``Data/dataset_FINAL.json``."""
        return self.data_raw / "dataset_FINAL.json"

    def ensure(self) -> "Paths":
        """Cria as pastas de saída (idempotente)."""
        for p in (
            self.data_processed,
            self.experiments,
            self.results,
            self.results_figures,
            self.reports,
            self.logs,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


_PATHS_CACHE: Paths | None = None


def get_paths(root: str | os.PathLike | None = None, ensure: bool = True) -> Paths:
    """Retorna (e memoiza) o objeto :class:`Paths` do projeto.

    Args:
        root: força um ``PROJECT_ROOT`` explícito; se ``None``, resolve
            automaticamente.
        ensure: se ``True`` cria as pastas de saída.
    """
    global _PATHS_CACHE
    if root is not None:
        paths = Paths(Path(root).expanduser().resolve())
        return paths.ensure() if ensure else paths
    if _PATHS_CACHE is None:
        _PATHS_CACHE = Paths(resolve_project_root())
    if ensure:
        _PATHS_CACHE.ensure()
    return _PATHS_CACHE


def describe_environment() -> dict:
    """Resumo legível do ambiente atual (para logs e checagem no Colab)."""
    paths = get_paths(ensure=False)
    return {
        "in_colab": in_colab(),
        "drive_mounted": _drive_mount_point() is not None,
        "project_root": str(paths.root),
        "root_exists": paths.root.exists(),
        "raw_dataset": str(paths.raw_dataset),
        "raw_dataset_exists": paths.raw_dataset.exists(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
