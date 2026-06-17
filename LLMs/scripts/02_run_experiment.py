"""
02_run_experiment.py — Runs an experiment: 1 model x N strategies x 1 split.

Supports:
  --sample-n N    : uses only N examples from the split (fast pipeline validation)
  --strategies    : strategy subset
  --split         : "val" for development, "test" for final evaluation
  --resume        : automatically resumes from checkpoint (default)
  --no-smoke-test : skips smoke test (useful if already tested)

Examples:
  # Fast validation with 30 examples
  python scripts/02_run_experiment.py --model claude_3_haiku --sample-n 30

  # Run all strategies on the test set
  python scripts/02_run_experiment.py --model claude_3_haiku --split test

  # Only zero_shot and few_shot
  python scripts/02_run_experiment.py --model qwen3_32b --strategies zero_shot few_shot
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "Models_BERT" / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from ftr_llm.bedrock_client import BedrockClient
from ftr_llm.maritaca_client import MaritacaClient
from ftr_llm.checkpoint import load_checkpoint, checkpoint_to_lists
from ftr_llm.config import load_config, load_model_registry, get_invoke_field
from ftr_llm.dataset import load_splits
from ftr_llm.evaluate import evaluate_run, compute_cost_efficiency, load_checkpoint_records
from ftr_llm.runner import run_experiment, run_experiment_async
from ftr_llm.viz import generate_all_plots


ALL_STRATEGIES = ["zero_shot", "few_shot", "evidence_constrained", "decomposed"]


def get_client_and_invoke(model_alias: str, registry: dict, cfg: dict):
    if model_alias == "sabia4":
        api_key = os.environ.get("API_KEY_MARITACA", "")
        if not api_key:
            raise ValueError("API_KEY_MARITACA not found in .env")
        client = MaritacaClient(api_key=api_key)
        return client, "invoke_maritaca", "sabia-4"

    models = registry.get("rankingGeneralPurpose", [])
    entry = next((m for m in models if m["alias"] == model_alias), None)
    if entry is None:
        raise ValueError(f"Alias not found: {model_alias!r}. Check bedrock_modelos_refatorado.json")

    field, value = get_invoke_field(entry)
    fallback_alias = cfg.get("fallback_model_alias", "claude_3_haiku")
    fallback_entry = next((m for m in models if m["alias"] == fallback_alias), None)
    fallback_value = get_invoke_field(fallback_entry)[1] if fallback_entry else None

    region = os.environ.get("AWS_DEFAULT_REGION", "sa-east-1")
    client = BedrockClient(region=region, fallback_model_id=fallback_value)
    return client, field, value


def main():
    parser = argparse.ArgumentParser(description="Run LLM experiment")
    parser.add_argument("--model", required=True, help="Model alias (e.g.: claude_3_haiku)")
    parser.add_argument("--config", default=str(_ROOT / "configs" / "_base_llm.yaml"))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--strategies", nargs="+", default=None, choices=ALL_STRATEGIES + ["all"])
    parser.add_argument("--sample-n", type=int, default=None,
                        help="Use only N examples from the split (fast validation). None = all.")
    parser.add_argument("--no-smoke-test", action="store_true")
    parser.add_argument("--delay", type=float, default=None, help="Delay between calls (sec)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Concurrent calls (default: config; 1=sequential)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg_path = _ROOT / "configs" / f"{args.model}.yaml"
    if model_cfg_path.exists():
        model_cfg = load_config(model_cfg_path)
        cfg.update({k: v for k, v in model_cfg.items() if k not in ("_base",)})

    registry = load_model_registry(cfg["data"]["bedrock_registry"])
    splits = load_splits(cfg["data"]["processed_file"])
    records = splits[args.split]

    if args.sample_n:
        # Stratified sample by label to preserve all classes
        import random
        random.seed(cfg.get("seed", 42))
        from collections import defaultdict
        by_label = defaultdict(list)
        for r in records:
            by_label[r["label"]].append(r)
        per_class = max(1, args.sample_n // 3)
        records = []
        for label_recs in by_label.values():
            records.extend(random.sample(label_recs, min(per_class, len(label_recs))))
        random.shuffle(records)
        logger.info("Sample mode: %d records (--sample-n %d)", len(records), args.sample_n)

    strategies = args.strategies or cfg.get("experiment", {}).get("strategies", ALL_STRATEGIES)
    if "all" in strategies:
        strategies = ALL_STRATEGIES

    exp_cfg = cfg.get("experiment", {})
    delay = args.delay if args.delay is not None else exp_cfg.get("delay_between_calls", 0.5)
    concurrency = args.concurrency if args.concurrency is not None else exp_cfg.get("concurrency", 5)
    # For small samples, limit concurrency to avoid firing more slots than records per class
    if args.sample_n:
        concurrency = min(concurrency, max(1, args.sample_n // 3))

    client, field, value = get_client_and_invoke(args.model, registry, cfg)

    ckpt_dir = Path(cfg["output"]["ckpt_dir"])
    results_dir = Path(cfg["output"]["results_dir"])
    reports_dir = Path(cfg["output"]["reports_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for strategy in strategies:
        logger.info("=" * 60)
        logger.info("Model: %s | Strategy: %s | Split: %s", args.model, strategy, args.split)
        if args.sample_n:
            logger.info("  [SAMPLE: %d examples — not a final evaluation]", len(records))
        logger.info("=" * 60)

        run_kwargs = dict(
            records=records,
            model_alias=args.model,
            invoke_field=field,
            invoke_value=value,
            strategy=strategy,
            split=args.split + (f"_sample{len(records)}" if args.sample_n else ""),
            client=client,
            ckpt_dir=ckpt_dir,
            max_tokens=exp_cfg.get("max_tokens", 512),
            temperature=exp_cfg.get("temperature", 0.0),
            delay_between_calls=delay,
            smoke_test=not args.no_smoke_test,
        )
        if concurrency > 1:
            logger.info("Async mode — concurrency: %d", concurrency)
            run_result = asyncio.run(
                run_experiment_async(**run_kwargs, concurrency=concurrency)
            )
        else:
            logger.info("Sequential mode (concurrency=1)")
            run_result = run_experiment(**run_kwargs)

        if not run_result.get("ok"):
            logger.error("Experiment failed: %s", run_result.get("error"))
            continue

        # Evaluation
        ckpt_records = load_checkpoint_records(run_result["checkpoint_file"])
        eval_result = evaluate_run(
            run_result["y_true"],
            run_result["y_pred"],
            checkpoint_records=ckpt_records,
        )

        cost_eff = compute_cost_efficiency(
            f1_macro=eval_result["flat"].get("f1_macro", 0),
            total_cost_usd=run_result["cost_summary"]["total_cost_usd"],
            n_docs=run_result["n_predicted"],
        )

        summary_entry = {
            "model": args.model,
            "strategy": strategy,
            "split": args.split,
            "sample_n": args.sample_n,
            "f1_macro": eval_result["flat"].get("f1_macro", 0),
            "f1_tese": eval_result["flat"].get("f1_tese", 0),
            "f1_fato": eval_result["flat"].get("f1_fato", 0),
            "f1_ruido": eval_result["flat"].get("f1_ruido", 0),
            "accuracy": eval_result["flat"].get("accuracy", 0),
            "mcc": eval_result["flat"].get("mcc", 0),
            "n_predicted": run_result["n_predicted"],
            "n_failed": run_result["n_failed"],
            "parse_failure_rate": run_result["parse_failure_rate"],
            "by_length_bin": {
                b: v["f1_macro"] for b, v in eval_result.get("by_length_bin", {}).items()
            },
            **cost_eff,
            "cost_summary": run_result["cost_summary"],
        }
        all_results.append(summary_entry)

        # Save individual result
        out_file = results_dir / f"{args.model}__{strategy}__{args.split}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary_entry,
                "metrics": eval_result["overall"],
            }, f, ensure_ascii=False, indent=2)

        logger.info(
            "F1-Macro: %.4f | Cost: $%.4f | Parse failures: %.1f%%",
            summary_entry["f1_macro"],
            summary_entry["total_cost_usd"],
            summary_entry["parse_failure_rate"] * 100,
        )
        logger.info("Result saved: %s", out_file)

        # After smoke test / sample, do NOT generate individual plots — only in consolidated
        args.no_smoke_test = True  # Only test once per run

    # Generate plots if there are enough results
    if all_results:
        logger.info("Generating plots...")
        generate_all_plots(all_results, reports_dir, dpi=cfg["output"].get("dpi", 150))
        logger.info("Plots saved to: %s", reports_dir)

    logger.info("Total session cost: $%.4f", client.cost_summary()["total_cost_usd"])


if __name__ == "__main__":
    main()
