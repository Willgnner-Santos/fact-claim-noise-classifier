"""
03 — Reavalia um experimento já treinado (recarrega best_model).

Útil para recomputar métricas/gráficos sem re-treinar, ou para avaliar com uma
estratégia diferente de texto longo (truncate vs sliding_window).

Uso:
    python scripts/03_evaluate.py --exp-dir experiments/roberta_lexpt__20260601_1530__abcd1234 \
        --config configs/roberta_lexpt.yaml
"""

from __future__ import annotations

import argparse

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from ftr.config import load_config  # noqa: E402
from ftr.env import get_paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Reavalia um experimento treinado.")
    ap.add_argument("--exp-dir", required=True, help="pasta do experimento (em experiments/)")
    ap.add_argument("--config", required=True, help="config YAML usado no experimento")
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    paths = get_paths(root=args.root)
    cfg = load_config(args.config, configs_dir=paths.configs)

    from ftr.evaluate import evaluate_from_dir

    exp_dir = args.exp_dir
    if not str(exp_dir).startswith(str(paths.experiments)):
        exp_dir = paths.experiments / exp_dir if not __import__("os").path.isabs(exp_dir) else exp_dir

    metrics = evaluate_from_dir(exp_dir, cfg, paths)
    print(f"acc={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f} mcc={metrics['mcc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
