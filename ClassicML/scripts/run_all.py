"""Roda todos os modelos clássicos em sequência com alerta de convergência."""
from __future__ import annotations
import argparse, glob, json, sys, traceback
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent

sys.path.insert(0, str(ROOT))

# Localiza src/ftr: tenta Models_BERT/src primeiro, depois src/ direto
for _candidate in [PROJ / "Models_BERT" / "src", PROJ / "src"]:
    if (_candidate / "ftr").exists():
        sys.path.insert(0, str(_candidate))
        break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--convergence-threshold", type=float, default=0.005)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--non-interactive", action="store_true")
    args = ap.parse_args()

    configs = sorted(glob.glob(str(ROOT / "configs" / "*.yaml")))
    configs = [c for c in configs if not Path(c).name.startswith("_")]
    if args.only:
        configs = [c for c in configs if Path(c).stem in args.only]

    from ftr_classic.trainer import run_experiment

    results, completed_f1 = [], []
    exp_root = ROOT / "experiments"

    for cfg_path in configs:
        eid = Path(cfg_path).stem
        print(f"\n{'='*60}\n>>> {eid}\n{'='*60}")
        try:
            status = run_experiment(cfg_path, PROJ, exp_root)
            results.append((eid, status["status"]))
            f1 = status.get("f1_macro_mean")
            if f1:
                completed_f1.append((eid, f1))
                print(f"  f1_macro: {f1:.4f} ± {status.get('f1_macro_std',0):.4f}")
        except Exception as e:
            print(f"!!! Falha em {eid}: {e}", file=sys.stderr)
            traceback.print_exc()
            results.append((eid, "failed"))
            continue

        # Alerta de convergência
        if len(completed_f1) >= args.window:
            recent = [f for _, f in completed_f1[-args.window:]]
            span = max(recent) - min(recent)
            if span <= args.convergence_threshold:
                names = [n for n, _ in completed_f1[-args.window:]]
                print(f"\n{'='*60}")
                print(f"ALERTA: os últimos {args.window} modelos variam apenas {span:.4f} de f1_macro.")
                print(f"  {names}: {[f'{f:.4f}' for _, f in completed_f1[-args.window:]]}")
                if not args.non_interactive:
                    resp = input("Continuar? [c/s] ").strip().lower()
                    if resp == "s":
                        print("Parando por decisão do usuário.")
                        break

    print("\n=== RESUMO ===")
    for eid, st in results:
        print(f"  {eid:25s} -> {st}")

    # Consolida
    from scripts.consolidate import consolidate
    consolidate(ROOT)


if __name__ == "__main__":
    main()
