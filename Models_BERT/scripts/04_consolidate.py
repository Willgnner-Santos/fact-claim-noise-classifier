"""
04 — Consolida todos os experimentos numa visão comparativa.

Varre experiments/, gera results/consolidated_results.json e
results/comparison_table.{csv,xlsx}, e produz gráficos comparativos entre modelos
(f1_macro, accuracy, mcc, ...) em PT-BR e EN.

Uso:
    python scripts/04_consolidate.py
"""

from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from ftr.env import get_paths  # noqa: E402
from ftr.registry import consolidate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Consolida resultados dos experimentos.")
    ap.add_argument("--root", default=None)
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["f1_macro", "accuracy", "mcc", "balanced_accuracy"],
        help="métricas para os gráficos de comparação",
    )
    args = ap.parse_args()

    paths = get_paths(root=args.root)
    out = consolidate(paths.experiments, paths.results)
    print(f"Execuções encontradas: {out['n_runs']} | com métricas de teste: {out['n_with_metrics']}")
    print(f"JSON consolidado: {out['consolidated_json']}")
    print(f"Tabela: {out['comparison_table']}")

    # Gráficos comparativos a partir da tabela.
    csv = out["comparison_table"].get("csv")
    if csv:
        import pandas as pd

        from ftr import viz

        rows = pd.read_csv(csv).to_dict(orient="records")
        cmp_dir = paths.results_figures / "comparison"
        cmp_dir.mkdir(parents=True, exist_ok=True)
        for metric in args.metrics:
            if rows and metric in rows[0]:
                for lang in ("pt", "en"):
                    p = viz.plot_model_comparison(rows, metric, cmp_dir, lang=lang)
                    if p:
                        print(f"  gráfico: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
