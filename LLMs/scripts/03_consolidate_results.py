"""
03_consolidate_results.py — Consolidates all results into tables and final plots.

Reads all individual result JSONs and generates:
  - results/summary_all.json
  - results/summary_all.csv
  - reports/plots_pt/ + reports/plots_en/
  - Confusion matrix per model x strategy

Usage:
  python scripts/03_consolidate_results.py
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("consolidate")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "Models_BERT" / "src"))

from ftr_llm.config import load_config
from ftr_llm.evaluate import load_checkpoint_records
from ftr_llm.viz import generate_all_plots, plot_confusion_matrix


def main():
    cfg = load_config(_ROOT / "configs" / "_base_llm.yaml")
    results_dir = Path(cfg["output"]["results_dir"])
    reports_dir = Path(cfg["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    result_files = sorted(results_dir.glob("*.json"))
    logger.info("Found %d result files", len(result_files))

    all_summaries = []
    for rf in result_files:
        try:
            with open(rf, encoding="utf-8") as f:
                data = json.load(f)
            summary = data.get("summary", {})
            if summary:
                all_summaries.append(summary)
        except Exception as e:
            logger.warning("Error reading %s: %s", rf.name, e)

    if not all_summaries:
        logger.error("No results found in %s", results_dir)
        return

    # Consolidated JSON
    out_json = results_dir / "summary_all.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    logger.info("Consolidated summary: %s", out_json)

    # Consolidated CSV
    try:
        import csv
        csv_path = results_dir / "summary_all.csv"
        flat_cols = [
            "model", "strategy", "split", "sample_n",
            "f1_macro", "f1_tese", "f1_fato", "f1_ruido",
            "accuracy", "mcc", "n_predicted", "n_failed",
            "parse_failure_rate", "total_cost_usd",
            "cost_per_1k_docs_usd", "cost_per_f1_point_usd",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=flat_cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_summaries)
        logger.info("Consolidated CSV: %s", csv_path)
    except Exception as e:
        logger.warning("Could not generate CSV: %s", e)

    # Consolidated plots
    logger.info("Generating consolidated plots...")
    generate_all_plots(all_summaries, reports_dir, dpi=cfg["output"].get("dpi", 150))

    # Individual confusion matrices
    ckpt_dir = Path(cfg["output"]["ckpt_dir"])
    for ckpt_file in sorted(ckpt_dir.glob("*.jsonl")):
        parts = ckpt_file.stem.split("__")
        if len(parts) < 2:
            continue
        model_alias, strategy = parts[0], parts[1]
        records = load_checkpoint_records(ckpt_file)
        y_true = [r["label"] for r in records if r.get("pred") is not None]
        y_pred = [r["pred"] for r in records if r.get("pred") is not None]
        if len(y_true) < 10:
            continue

        from ftr_llm.evaluate import evaluate_run
        eval_r = evaluate_run(y_true, y_pred)
        cm = eval_r["overall"]["confusion_matrix"]
        cm_labels = eval_r["overall"]["confusion_matrix_labels"]

        for lang in ("pt", "en"):
            out_path = reports_dir / f"plots_{lang}" / f"cm_{model_alias}_{strategy}.png"
            plot_confusion_matrix(
                cm, cm_labels,
                title=f"{model_alias} | {strategy}",
                out_path=out_path,
                lang=lang,
                dpi=cfg["output"].get("dpi", 150),
            )

    logger.info("Consolidation complete. Plots at: %s", reports_dir)


if __name__ == "__main__":
    main()
