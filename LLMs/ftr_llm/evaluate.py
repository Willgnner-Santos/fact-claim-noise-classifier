"""
Avaliação completa: reutiliza src/ftr/metrics.py + análise por faixa de comprimento.

Computa:
  - Todas as métricas da planilha Métricas_Avaliação_Modelos.xlsx
  - F1-Macro por faixa de comprimento (length_bin)
  - Custo por ponto de F1 (eficiência)
  - Consistência entre execuções (se múltiplos runs disponíveis)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_SRC = Path(__file__).parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ftr.metrics import compute_all_metrics, flatten_metrics
from ftr_llm.dataset import LENGTH_BINS


def evaluate_run(
    y_true: list[int],
    y_pred: list[int],
    checkpoint_records: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Avaliação completa de um run.

    Args:
        y_true: labels verdadeiros
        y_pred: labels preditos
        checkpoint_records: lista de registros do JSONL (para análise por bin)
    """
    metrics = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=None,  # LLMs não retornam logits
    )

    result: dict[str, Any] = {
        "overall": metrics,
        "flat": flatten_metrics(metrics),
        "by_length_bin": {},
        "cost_efficiency": None,
    }

    # --- Análise por faixa de comprimento ---
    if checkpoint_records:
        bin_records: dict[str, tuple[list, list]] = {}
        for rec in checkpoint_records:
            if rec.get("pred") is None:
                continue
            b = rec.get("length_bin", "unknown")
            if b not in bin_records:
                bin_records[b] = ([], [])
            bin_records[b][0].append(rec["label"])
            bin_records[b][1].append(rec["pred"])

        for bin_name, (yt, yp) in sorted(bin_records.items()):
            if len(yt) < 10:
                continue
            bin_metrics = compute_all_metrics(yt, yp)
            result["by_length_bin"][bin_name] = {
                "n": len(yt),
                "f1_macro": bin_metrics["f1_macro"],
                "f1_per_class": {
                    cls: bin_metrics["per_class"][cls]["f1"]
                    for cls in bin_metrics["per_class"]
                },
                "accuracy": bin_metrics["accuracy"],
            }

    return result


def compute_cost_efficiency(
    f1_macro: float,
    total_cost_usd: float,
    n_docs: int,
) -> dict[str, float]:
    """Custo por ponto de F1 e custo por 1k documentos."""
    cost_per_1k = (total_cost_usd / max(n_docs, 1)) * 1000
    cost_per_f1_point = total_cost_usd / max(f1_macro, 0.001)
    return {
        "f1_macro": round(f1_macro, 4),
        "total_cost_usd": round(total_cost_usd, 4),
        "cost_per_1k_docs_usd": round(cost_per_1k, 4),
        "cost_per_f1_point_usd": round(cost_per_f1_point, 4),
        "n_docs": n_docs,
    }


def load_checkpoint_records(jsonl_path: str | Path) -> list[dict]:
    import json
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records
