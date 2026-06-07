"""
run_all — Treina todos os modelos em sequência com alerta de convergência.

Resiliente a falhas: cada modelo é tratado isoladamente; se um falhar, os
demais continuam. Ao final, consolida os resultados.

ALERTA DE CONVERGÊNCIA (--convergence-alert)
============================================
A partir do 3º modelo, compara o f1_macro do modelo atual com a média dos
anteriores. Se todos os modelos dentro de uma janela (--window) estiverem a
menos de ``--convergence-threshold`` de diferença entre si, o pipeline emite
um alerta e pergunta se deseja continuar ou parar.

Motivação: evitar gastar créditos de Colab treinando modelos que estão
todos convergindo para o mesmo resultado — sinal de que o task pode ser
trivial (ex: data leakage) ou que os modelos restantes dificilmente mudarão
o panorama.

Uso:
    python scripts/run_all.py                          # todos os configs
    python scripts/run_all.py --only roberta_lexpt mbert
    python scripts/run_all.py --reproductions-only
    python scripts/run_all.py --max-steps 5            # smoke-test
    python scripts/run_all.py --convergence-threshold 0.005 --window 3
    python scripts/run_all.py --no-convergence-alert   # desativa alerta
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import traceback
from statistics import mean, stdev

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from ftr.config import load_config   # noqa: E402
from ftr.data import load_splits     # noqa: E402
from ftr.env import get_paths        # noqa: E402


def _discover_configs(configs_dir) -> list[str]:
    files = sorted(glob.glob(os.path.join(str(configs_dir), "*.yaml")))
    return [f for f in files if not os.path.basename(f).startswith("_")]


def _check_convergence(
    completed: list[tuple[str, float]],
    window: int,
    threshold: float,
) -> tuple[bool, str]:
    """Verifica se os últimos `window` modelos estão todos dentro de `threshold`.

    Returns:
        (alert_triggered, mensagem_explicativa)
    """
    if len(completed) < window:
        return False, ""
    recent = [f1 for _, f1 in completed[-window:]]
    span = max(recent) - min(recent)
    if span <= threshold:
        names = [n for n, _ in completed[-window:]]
        msg = (
            f"\n{'='*70}\n"
            f"⚠  ALERTA DE CONVERGÊNCIA\n"
            f"{'='*70}\n"
            f"Os últimos {window} modelos têm f1_macro dentro de {threshold:.4f}:\n"
        )
        for name, f1 in completed[-window:]:
            msg += f"  {name:20s}: {f1:.4f}\n"
        msg += (
            f"  Amplitude: {span:.4f}  (threshold: {threshold:.4f})\n\n"
            "Possíveis causas:\n"
            "  1. Task muito fácil para todos os modelos (labels muito distintos)\n"
            "  2. Data leakage (fragmentos do mesmo doc em treino e teste)\n"
            "  3. Modelos restantes provavelmente darão resultado similar\n\n"
            "Opções:\n"
            "  [c] Continuar treinando os próximos modelos\n"
            "  [s] Parar aqui e consolidar os resultados\n"
            f"{'='*70}"
        )
        return True, msg
    return False, ""


def _ask_continue(msg: str, non_interactive: bool) -> bool:
    """Exibe alerta e retorna True se deve continuar, False se parar."""
    print(msg)
    if non_interactive:
        print("Modo não-interativo: continuando automaticamente.")
        return True
    try:
        resp = input("\nContinuar? [c/s] ").strip().lower()
        return resp != "s"
    except (EOFError, KeyboardInterrupt):
        # No Colab sem terminal interativo: continua por segurança.
        print("Sem terminal interativo: continuando.")
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Treina todos os modelos em sequência.")
    ap.add_argument("--root", default=None)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--reproductions-only", action="store_true")
    ap.add_argument("--new-only", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--data-subdir", default=None)
    ap.add_argument("--no-consolidate", action="store_true")
    # Alerta de convergência
    ap.add_argument(
        "--convergence-threshold", type=float, default=0.005,
        help="Diferença máxima de f1_macro para acionar alerta (padrão: 0.005)",
    )
    ap.add_argument(
        "--window", type=int, default=3,
        help="Nº de modelos consecutivos para verificar convergência (padrão: 3)",
    )
    ap.add_argument(
        "--no-convergence-alert", action="store_true",
        help="Desativa o alerta de convergência",
    )
    ap.add_argument(
        "--non-interactive", action="store_true",
        help="Modo não-interativo: continua automaticamente no alerta (útil no Colab)",
    )
    args = ap.parse_args()

    paths = get_paths(root=args.root)
    all_configs = _discover_configs(paths.configs)

    # Filtragem de configs
    selected = []
    for cfg_path in all_configs:
        cfg = load_config(cfg_path, configs_dir=paths.configs)
        name = cfg.get("experiment_id", cfg["__config_name__"])
        is_repro = cfg["model"].get("is_reproduction", False)
        if args.only and name not in args.only and cfg["__config_name__"] not in args.only:
            continue
        if args.reproductions_only and not is_repro:
            continue
        if args.new_only and is_repro:
            continue
        selected.append((cfg_path, cfg))

    if not selected:
        print("Nenhum config selecionado.", file=sys.stderr)
        return 1

    print(f"Modelos a treinar ({len(selected)}): {[c['experiment_id'] for _, c in selected]}")

    data_dir = (
        paths.data_processed / args.data_subdir
        if args.data_subdir
        else paths.data_processed
    )
    splits = load_splits(data_dir)
    if "train" not in splits:
        print(
            f"ERRO: splits ausentes em {data_dir}. Rode 01_prepare_data.py antes.",
            file=sys.stderr,
        )
        return 1

    from ftr.metrics import flatten_metrics
    from ftr.train import train_experiment

    results: list[tuple[str, str]] = []
    completed_f1: list[tuple[str, float]] = []  # (experiment_id, f1_macro do teste)

    for cfg_path, cfg in selected:
        eid = cfg.get("experiment_id")
        print("\n" + "=" * 70)
        print(f">>> {eid}  ({cfg['model']['name']})")
        print("=" * 70)
        try:
            status = train_experiment(
                cfg, paths, splits=splits, max_steps=args.max_steps
            )
            results.append((eid, status["status"]))

            # Extrai f1_macro do teste para alerta de convergência.
            # Pega do epoch_metrics (val) se test_metrics ainda não disponível.
            test_f1 = None
            test_metrics_path = (
                paths.experiments
                / status.get("run_dir", "")
                / "metrics"
                / "test_metrics.json"
            )
            if test_metrics_path.exists():
                import json
                tm = json.loads(test_metrics_path.read_text(encoding="utf-8"))
                test_f1 = flatten_metrics(tm).get("f1_macro")
            if test_f1 is None:
                # Fallback: melhor f1_macro de validação (última época)
                em = status.get("epoch_metrics") or []
                val_f1s = [
                    e.get("eval_f1_macro") for e in em
                    if e.get("eval_f1_macro") is not None
                ]
                if val_f1s:
                    test_f1 = max(val_f1s)

            if test_f1 is not None:
                completed_f1.append((eid, test_f1))
                print(f"  f1_macro: {test_f1:.4f}")

        except Exception as e:
            print(f"!!! Falha em {eid}: {e}", file=sys.stderr)
            traceback.print_exc()
            results.append((eid, "failed"))
            continue

        # Verifica convergência após cada modelo (a partir do window-ésimo)
        if not args.no_convergence_alert and not args.max_steps:
            triggered, msg = _check_convergence(
                completed_f1,
                window=args.window,
                threshold=args.convergence_threshold,
            )
            if triggered:
                keep_going = _ask_continue(msg, non_interactive=args.non_interactive)
                if not keep_going:
                    print("\nParando por decisão do usuário após alerta de convergência.")
                    break

    print("\n=== RESUMO ===")
    for eid, st in results:
        print(f"  {eid:20s} -> {st}")

    if completed_f1:
        f1s = [f for _, f in completed_f1]
        print(f"\n  f1_macro: min={min(f1s):.4f}  max={max(f1s):.4f}  "
              f"mean={mean(f1s):.4f}"
              + (f"  std={stdev(f1s):.4f}" if len(f1s) > 1 else ""))

    if not args.no_consolidate:
        from ftr.registry import consolidate

        out = consolidate(paths.experiments, paths.results)
        print(f"\nConsolidado: {out['consolidated_json']}")
        print(f"Tabela: {out['comparison_table']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
