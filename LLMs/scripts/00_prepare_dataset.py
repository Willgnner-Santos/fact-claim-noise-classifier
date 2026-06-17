"""
00_prepare_dataset.py — Prepares the dataset for LLM experiments.

What it does:
  1. Loads dataset_FINAL.json
  2. Applies preprocessing + anonymization (reuses src/ftr)
  3. Creates stratified train/val/test split
  4. Saves to LLMs/data/splits/{train,val,test}.json
  5. Saves anonymization audit and statistics

Usage:
  python scripts/00_prepare_dataset.py
  python scripts/00_prepare_dataset.py --force   # regenerates even if splits exist
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_dataset")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "Models_BERT" / "src"))

from ftr_llm.config import load_config
from ftr_llm.dataset import (
    dataset_stats,
    load_raw,
    make_splits,
    preprocess_and_anonymize,
    save_splits,
)
from ftr.anonymize import AnonymizationConfig
from ftr.preprocess import PreprocessConfig


def main():
    parser = argparse.ArgumentParser(description="Prepare LLM dataset")
    parser.add_argument("--config", default=str(_ROOT / "configs" / "_base_llm.yaml"))
    parser.add_argument("--force", action="store_true", help="Regenerate splits even if they exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_dir = Path(cfg["data"]["processed_file"])

    if not args.force and (split_dir / "test.json").exists():
        logger.info("Splits already exist at %s. Use --force to regenerate.", split_dir)
        return

    raw_path = Path(cfg["data"]["raw_file"])
    logger.info("Loading dataset: %s", raw_path)
    records = load_raw(raw_path)
    logger.info("Records loaded: %d", len(records))

    anon_cfg = AnonymizationConfig.from_dict(cfg.get("anonymization"))
    prep_cfg = PreprocessConfig.from_dict(cfg.get("preprocess"))

    logger.info("Preprocessing and anonymizing...")
    processed, audit = preprocess_and_anonymize(records, anon_cfg, prep_cfg)
    logger.info("Masked entities: %s", audit.get("masked_by_type"))

    split_cfg = cfg.get("split", {})
    splits = make_splits(
        processed,
        test_size=split_cfg.get("test_size", 0.10),
        val_size=split_cfg.get("val_size", 0.10),
        seed=split_cfg.get("random_state", 42),
    )

    for name, recs in splits.items():
        logger.info("Split %s: %d records", name, len(recs))
        logger.info("  Stats: %s", dataset_stats(recs))

    save_splits(splits, split_dir)
    logger.info("Splits saved to: %s", split_dir)

    # Save audit and stats
    reports_dir = Path(cfg["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "anonymization_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    stats_all = {name: dataset_stats(recs) for name, recs in splits.items()}
    with open(reports_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_all, f, ensure_ascii=False, indent=2)

    logger.info("Audit and statistics saved to: %s", reports_dir)


if __name__ == "__main__":
    main()
