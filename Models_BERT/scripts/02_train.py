"""
02 — Treina um experimento a partir de um config.

Carrega o config (mesclado com _base.yaml), os splits de data/processed/ e treina
o modelo via ftr.train.train_experiment. Cada execução cria uma pasta isolada em
experiments/ com config, env, logs, checkpoints, métricas e gráficos.

Uso:
    python scripts/02_train.py --config configs/roberta_lexpt.yaml
    python scripts/02_train.py --config configs/longformer_pt.yaml --max-steps 5   # smoke-test
    python scripts/02_train.py --config configs/bertimbau_large.yaml --data-subdir frac050

Requer torch + transformers + datasets (use no Colab/GPU).
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from ftr.config import load_config  # noqa: E402
from ftr.data import load_splits  # noqa: E402
from ftr.env import get_paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Treina um experimento.")
    ap.add_argument("--config", required=True, help="config YAML do experimento")
    ap.add_argument("--root", default=None, help="PROJECT_ROOT")
    ap.add_argument("--data-subdir", default=None, help="subpasta de data/processed/")
    ap.add_argument("--max-steps", type=int, default=None, help="limita passos (smoke-test)")
    args = ap.parse_args()

    paths = get_paths(root=args.root)
    cfg = load_config(args.config, configs_dir=paths.configs)

    data_dir = paths.data_processed / args.data_subdir if args.data_subdir else paths.data_processed
    splits = load_splits(data_dir)
    if "train" not in splits:
        print(f"ERRO: splits não encontrados em {data_dir}. Rode 01_prepare_data.py antes.",
              file=sys.stderr)
        return 1

    from ftr.train import train_experiment

    print(f">>> Treinando '{cfg.get('experiment_id')}' ({cfg['model']['name']})")
    if args.max_steps:
        print(f"    [SMOKE-TEST] max_steps={args.max_steps}")
    status = train_experiment(cfg, paths, splits=splits, max_steps=args.max_steps)
    print(f">>> Status: {status['status']} | dir: {status['run_dir']}")
    return 0 if status["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
