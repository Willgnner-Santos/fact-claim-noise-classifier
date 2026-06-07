"""
Bootstrap de import para os scripts CLI.

Garante que ``src/`` esteja no ``sys.path`` para que ``import ftr`` funcione tanto
localmente (``python scripts/01_prepare_data.py``) quanto no Colab, sem exigir
``pip install -e .``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path() -> Path:
    """Adiciona ``<repo>/src`` ao sys.path e retorna a raiz do repo."""
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    src = repo_root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return repo_root
