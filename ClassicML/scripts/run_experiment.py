"""Run a single classical ML experiment."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent

sys.path.insert(0, str(ROOT))

for _candidate in [PROJ / "Models_BERT" / "src", PROJ / "src"]:
    if (_candidate / "ftr").exists():
        sys.path.insert(0, str(_candidate))
        break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--experiments-dir", default=None)
    args = ap.parse_args()

    from ftr_classic.trainer import run_experiment
    status = run_experiment(
        config_path=args.config,
        project_root=PROJ,
        experiments_root=args.experiments_dir or ROOT / "experiments",
    )
    print(f"Status: {status['status']}")
    if status.get("f1_macro_mean"):
        print(f"f1_macro: {status['f1_macro_mean']:.4f} ± {status['f1_macro_std']:.4f}")
    if status.get("test_metrics"):
        print(f"Test — f1_macro: {status['test_metrics'].get('f1_macro', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
