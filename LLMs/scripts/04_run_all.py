"""
04_run_all.py — Full orchestrator: all models x all strategies.

Runs experiments in sequence, model by model, strategy by strategy.
If it stops mid-run (Ctrl+C, network drop, timeout), just run again:
  - Already-completed models are skipped automatically (--skip-done)
  - Within each model, per-line checkpoints guarantee resumption at the exact index

Execution order (ascending cost):
  1. claude_3_haiku       — light, cheap, validates everything first
  2. gemma_3_27b          — medium, no Claude cost
  3. magistral_small_2509 — medium Mistral
  4. qwen3_32b            — medium Qwen
  5. deepseek_v32         — large, low cost
  6. mistral_large_3      — large Mistral
  7. sabia4               — Maritaca (separate API)
  8. claude_sonnet_46     — elite, most expensive — saved for last

Usage:
  # Run everything on the val set (development)
  python scripts/04_run_all.py --split val

  # Final evaluation on the test set
  python scripts/04_run_all.py --split test

  # Only some models
  python scripts/04_run_all.py --models claude_3_haiku gemma_3_27b --split val

  # Resume from where it stopped (default — automatic checkpoints)
  python scripts/04_run_all.py --split val

  # Skip models whose result JSON already exists
  python scripts/04_run_all.py --split test --skip-done
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_all")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "Models_BERT" / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from ftr_llm.config import load_config

# Order by ascending cost
MODEL_ORDER = [
    "claude_3_haiku",
    "gemma_3_27b",
    "magistral_small_2509",
    "qwen3_32b",
    "deepseek_v32",
    "mistral_large_3",
    "sabia4",
    "claude_sonnet_46",
]

ALL_STRATEGIES = ["zero_shot", "few_shot", "evidence_constrained", "decomposed"]


def result_exists(results_dir: Path, model: str, strategy: str, split: str) -> bool:
    """Checks whether the JSON result for this experiment already exists and has F1-Macro."""
    p = results_dir / f"{model}__{strategy}__{split}.json"
    if not p.exists():
        return False
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("summary", {}).get("f1_macro", 0) > 0
    except Exception:
        return False


def run_one(
    model: str,
    split: str,
    strategies: list[str],
    no_smoke_test: bool,
    config: str,
    delay: float,
) -> bool:
    """Calls 02_run_experiment.py as a subprocess. Returns True if successful."""
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "02_run_experiment.py"),
        "--model", model,
        "--split", split,
        "--strategies", *strategies,
        "--config", config,
        "--delay", str(delay),
    ]
    if no_smoke_test:
        cmd.append("--no-smoke-test")

    logger.info("Starting: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(_ROOT))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all LLM experiments")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="Split to evaluate (val=development, test=final)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model subset (default: all in ascending cost order)")
    parser.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES,
                        choices=ALL_STRATEGIES + ["all"])
    parser.add_argument("--config", default=str(_ROOT / "configs" / "_base_llm.yaml"))
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip experiments whose JSON result already exists with F1 > 0")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between calls in seconds (default 0.5)")
    parser.add_argument("--no-smoke-test", action="store_true",
                        help="Skip smoke test for all models")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(cfg["output"]["results_dir"])
    reports_dir = Path(cfg["output"]["reports_dir"])

    models = args.models or MODEL_ORDER
    strategies = ALL_STRATEGIES if "all" in args.strategies else args.strategies

    # Progress report
    total = len(models) * len(strategies)
    done_count = 0
    skipped_count = 0
    failed = []

    logger.info("=" * 70)
    logger.info("FULL PIPELINE")
    logger.info("  Split:       %s", args.split)
    logger.info("  Models:      %s", models)
    logger.info("  Strategies:  %s", strategies)
    logger.info("  Total:       %d experiments", total)
    logger.info("  Skip-done:   %s", args.skip_done)
    logger.info("=" * 70)

    session_start = time.time()
    first_model = True

    for model in models:
        # Check which strategies are already done for this model
        pending_strategies = []
        for strategy in strategies:
            if args.skip_done and result_exists(results_dir, model, strategy, args.split):
                logger.info("Skipping (already exists): %s x %s", model, strategy)
                skipped_count += 1
                done_count += 1
            else:
                pending_strategies.append(strategy)

        if not pending_strategies:
            logger.info("Model %s: all strategies already completed.", model)
            continue

        # Smoke test only for the first model (others skip)
        no_smoke = args.no_smoke_test or not first_model
        first_model = False

        logger.info("-" * 60)
        logger.info("Model %s — %d pending strategies: %s",
                    model, len(pending_strategies), pending_strategies)

        ok = run_one(
            model=model,
            split=args.split,
            strategies=pending_strategies,
            no_smoke_test=no_smoke,
            config=args.config,
            delay=args.delay,
        )

        if ok:
            done_count += len(pending_strategies)
            elapsed = time.time() - session_start
            logger.info(
                "Progress: %d/%d completed — total time: %.0fs",
                done_count, total, elapsed,
            )
        else:
            failed.append(model)
            logger.error(
                "Model %s failed. Continuing with the next one...", model
            )
            # Pause before the next model in case of failure (rate limit, etc.)
            time.sleep(10)

    # Final consolidation
    logger.info("=" * 70)
    logger.info("All models processed. Generating final consolidation...")

    consolidate_cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "03_consolidate_results.py"),
        "--config", args.config,
    ]
    subprocess.run(consolidate_cmd, cwd=str(_ROOT))

    # Final summary
    elapsed_total = time.time() - session_start
    logger.info("=" * 70)
    logger.info("DONE")
    logger.info("  Total time:     %.0f min", elapsed_total / 60)
    logger.info("  Experiments:    %d/%d", done_count, total)
    logger.info("  Skipped:        %d", skipped_count)
    if failed:
        logger.warning("  Models with failure: %s", failed)
        logger.warning("  To retry: python scripts/04_run_all.py --split %s --models %s",
                       args.split, " ".join(failed))
    logger.info("  Results:      %s", results_dir)
    logger.info("  Plots:        %s", reports_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
