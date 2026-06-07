"""
Controle de reprodutibilidade — sementes (seeds) e determinismo.

Centraliza a fixação de seeds para Python, NumPy e (quando disponíveis) PyTorch
e Transformers. Importações de torch/transformers são preguiçosas para que o
módulo funcione localmente mesmo sem essas bibliotecas instaladas.
"""

from __future__ import annotations

import os
import random

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> int:
    """Fixa todas as sementes relevantes para reprodutibilidade.

    Args:
        seed: valor da semente.
        deterministic: se ``True``, força algoritmos determinísticos no PyTorch
            e cuDNN (mais lento; usar quando a reprodução exata importa mais que
            a velocidade).

    Returns:
        A semente aplicada (útil para registro).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy sempre presente neste projeto
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Exigido por algumas ops para determinismo total no CUDA.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (AttributeError, RuntimeError):
                pass
    except ImportError:
        pass

    # Transformers usa as seeds acima, mas chamamos set_seed deles se existir
    # para cobrir geradores internos adicionais.
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        pass

    return seed
