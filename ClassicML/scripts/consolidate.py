"""Consolidate results from all experiments into a comparative table."""
from __future__ import annotations
import json, glob, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent

sys.path.insert(0, str(ROOT))

for _candidate in [PROJ / "Models_BERT" / "src", PROJ / "src"]:
    if (_candidate / "ftr").exists():
        sys.path.insert(0, str(_candidate))
        break


def consolidate(classic_root: Path | None = None):
    classic_root = classic_root or ROOT
    exp_root = classic_root / "experiments"
    results_dir = classic_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for status_path in sorted(exp_root.glob("*/status.json")):
        try:
            s = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if s.get("status") != "completed":
            continue
        row = {
            "experiment_id":  s.get("experiment_id"),
            "model_name":     s.get("model_name"),
            "status":         s.get("status"),
            "f1_macro_mean":  s.get("f1_macro_mean"),
            "f1_macro_std":   s.get("f1_macro_std"),
            "duration_s":     s.get("duration_seconds"),
        }
        tm = s.get("test_metrics") or {}
        for k in ("f1_macro","accuracy","mcc","cohen_kappa","roc_auc_macro_ovr","log_loss","brier_score_macro"):
            row[f"test_{k}"] = tm.get(k)
        rows.append(row)

    if not rows:
        print("No completed experiments found.")
        return

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False, na_position="last")
    df.to_csv(results_dir / "comparison_table.csv", index=False, encoding="utf-8")
    try:
        df.to_excel(results_dir / "comparison_table.xlsx", index=False)
    except Exception:
        pass
    print(f"Tabela salva em {results_dir / 'comparison_table.csv'}")
    print(df[["experiment_id","f1_macro_mean","f1_macro_std","test_f1_macro"]].to_string(index=False))

    # Comparative chart PT and EN
    sys.path.insert(0, str(ROOT))
    from ftr_classic.viz import plot_comparison
    result_list = rows
    fig_dir = results_dir / "figures"
    for lang in ("pt", "en"):
        plot_comparison(result_list, fig_dir, lang=lang)
    print(f"Figures at {fig_dir}")


if __name__ == "__main__":
    consolidate()
