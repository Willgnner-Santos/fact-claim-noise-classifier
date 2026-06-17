"""
04_reeval_on_bert_test.py — Re-evaluates LLMs on the SAME test set used by
BERT/ClassicML models, for a fair comparison across model families.

Why:
  The BERT and LLM pipelines generated different splits (dedup + group_by_document
  in BERT; simple stratified split in LLM). The test sets had only ~10% overlap.
  This routine fixes the BERT test as the canonical baseline.

What it does:
  1. Loads the canonical test set (Models_BERT/Data/processed/test.jsonl):
     2,559 excerpts, deduplicated, anonymized, with label_text.
  2. Reuses existing predictions from old LLM pipeline checkpoints
     (data/checkpoints/*.jsonl), matched by normalised TEXT — zero cost.
  3. Calls the API only for texts still without a prediction.
  4. Own checkpoint in data/checkpoints_bertset/ (does not touch the old ones).
  5. Saves metrics per model x strategy in data/results_bertset/.

Metrics computed by comparing STRINGS (pred_text vs label_text), avoiding
any integer-mapping ambiguity.

Usage:
  python scripts/04_reeval_on_bert_test.py --models all --strategies all
  python scripts/04_reeval_on_bert_test.py --models gemma_3_27b --strategies zero_shot
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reeval")

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
from ftr_llm.config import load_config, load_model_registry, get_invoke_field
from ftr_llm.prompts import build_prompt
from ftr_llm.parser import parse_response
from ftr_llm.checkpoint import load_checkpoint, append_record, async_append_record, checkpoint_path

ALL_STRATEGIES = ["zero_shot", "few_shot", "evidence_constrained", "decomposed"]
ALL_MODELS = [
    "claude_sonnet_46", "claude_3_haiku", "qwen3_32b", "mistral_large_3",
    "gemma_3_27b", "deepseek_v32", "magistral_small_2509", "sabia4",
]
CLASS_NAMES = ["tese", "fato", "ruido"]

BERT_TEST = _ROOT.parent / "Models_BERT" / "Data" / "processed" / "test.jsonl"
OLD_CKPT_DIR = _ROOT / "data" / "checkpoints"
NEW_CKPT_DIR = _ROOT / "data" / "checkpoints_bertset"
RESULTS_DIR = _ROOT / "data" / "results_bertset"
OLD_SPLITS = _ROOT / "data" / "splits"


def norm(s: str) -> str:
    return " ".join(s.split()).strip().lower()


def load_bert_test() -> list[dict]:
    recs = []
    with open(BERT_TEST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs.append({
                "text": r["text"],
                "label_text": r["label_text"],
                "key": norm(r["text"]),
            })
    return recs


def build_prior_predictions() -> dict[tuple[str, str, str], str]:
    """
    Scans old checkpoints and returns a map:
       (model, strategy, text_key) -> pred_text
    Matches the checkpoint idx with the text via the old splits.
    """
    # idx -> text_key, per old split
    split_idx_text: dict[str, dict[int, str]] = {}
    for sp in ("test", "val", "train"):
        p = OLD_SPLITS / f"{sp}.json"
        if p.exists():
            arr = json.load(open(p, encoding="utf-8"))
            split_idx_text[sp] = {i: norm(x["text"]) for i, x in enumerate(arr)}

    prior: dict[tuple[str, str, str], str] = {}
    for f in glob.glob(str(OLD_CKPT_DIR / "*.jsonl")):
        name = Path(f).stem  # model__strategy__split
        try:
            model, strategy, split = name.split("__")
        except ValueError:
            continue
        if split not in split_idx_text:
            continue
        idx_text = split_idx_text[split]
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            pred_text = rec.get("pred_text")
            if not pred_text:
                continue
            tkey = idx_text.get(rec["idx"])
            if tkey is None:
                continue
            prior.setdefault((model, strategy, tkey), pred_text)
    logger.info("Reusable old predictions: %d entries", len(prior))
    return prior


def get_client_and_invoke(model_alias: str, registry: dict, cfg: dict):
    if model_alias == "sabia4":
        api_key = os.environ.get("API_KEY_MARITACA", "")
        if not api_key:
            raise ValueError("API_KEY_MARITACA not found in .env")
        return MaritacaClient(api_key=api_key), "invoke_maritaca", "sabia-4"
    models = registry.get("rankingGeneralPurpose", [])
    entry = next((m for m in models if m["alias"] == model_alias), None)
    if entry is None:
        raise ValueError(f"Alias not found: {model_alias!r}")
    field, value = get_invoke_field(entry)
    region = os.environ.get("AWS_DEFAULT_REGION", "sa-east-1")
    fb = next((m for m in models if m["alias"] == "claude_3_haiku"), None)
    fb_val = get_invoke_field(fb)[1] if fb else None
    return BedrockClient(region=region, fallback_model_id=fb_val), field, value


def compute_metrics(pairs: list[tuple[str, str]]) -> dict:
    """pairs = [(true_text, pred_text)] already filtered (non-null pred)."""
    labels = CLASS_NAMES
    n = len(pairs)
    correct = sum(1 for t, p in pairs if t == p)
    acc = correct / n if n else 0.0
    # per-class precision/recall/f1
    per = {}
    f1s = []
    for c in labels:
        tp = sum(1 for t, p in pairs if t == c and p == c)
        fp = sum(1 for t, p in pairs if t != c and p == c)
        fn = sum(1 for t, p in pairs if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                  "f1": round(f1, 4), "support": tp + fn}
        f1s.append(f1)
    f1_macro = sum(f1s) / len(f1s) if f1s else 0.0
    # confusion matrix (rows=true, cols=pred), order = labels
    cm = [[0] * len(labels) for _ in labels]
    li = {c: i for i, c in enumerate(labels)}
    for t, p in pairs:
        cm[li[t]][li[p]] += 1
    return {
        "n_evaluated": n,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_tese": per["tese"]["f1"],
        "f1_fato": per["fato"]["f1"],
        "f1_ruido": per["ruido"]["f1"],
        "per_class": per,
        "confusion_matrix": cm,
        "labels": labels,
    }


async def process_pending(client, field, value, strategy, pending, ckpt_file,
                          max_tokens, temperature, concurrency, delay):
    sem = asyncio.Semaphore(concurrency)
    progress = {"done": 0}
    total = len(pending)

    async def one(key, text, true_text):
        async with sem:
            sys_p, usr = build_prompt(strategy, text)
            t0 = time.perf_counter()
            try:
                res = await client.async_call(field, value, sys_p, usr,
                                              max_tokens=max_tokens, temperature=temperature)
            except Exception as e:
                rec = {"key": key, "label_text": true_text, "pred_text": None,
                       "parse_ok": False, "error": str(e)[:200], "cost_usd": 0.0}
                await async_append_record(ckpt_file, rec)
                progress["done"] += 1
                return
            if res.ok:
                parsed = parse_response(res.text, strategy)
                pred_text = parsed["classe"]
            else:
                pred_text = None
            rec = {"key": key, "label_text": true_text, "pred_text": pred_text,
                   "parse_ok": bool(pred_text), "cost_usd": res.cost_usd,
                   "input_tokens": res.input_tokens, "output_tokens": res.output_tokens}
            await async_append_record(ckpt_file, rec)
            progress["done"] += 1
            if progress["done"] % 100 == 0 or progress["done"] == total:
                logger.info("  [%s] %d/%d — accum. cost: $%.4f",
                            strategy, progress["done"], total,
                            client.cost_summary()["total_cost_usd"])
            el = time.perf_counter() - t0
            if delay - el > 0:
                await asyncio.sleep(delay - el)

    tasks = [asyncio.create_task(one(k, t, tt)) for k, t, tt in pending]
    for c in asyncio.as_completed(tasks):
        await c


def run_model_strategy(model, strategy, bert_recs, prior, registry, cfg, args):
    NEW_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_file = NEW_CKPT_DIR / f"{model}__{strategy}__berttest.jsonl"

    # load already-existing new checkpoint (keyed by 'key')
    done_by_key = {}
    if ckpt_file.exists():
        for line in open(ckpt_file, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done_by_key[r["key"]] = r
            except Exception:
                pass

    # build results: reuse prior; collect pending
    reused = 0
    pending = []
    for rec in bert_recs:
        k = rec["key"]
        if k in done_by_key:
            continue
        pk = (model, strategy, k)
        if pk in prior:
            # record as reused (zero cost)
            append_record(ckpt_file, {
                "key": k, "label_text": rec["label_text"],
                "pred_text": prior[pk], "parse_ok": True,
                "cost_usd": 0.0, "reused": True,
            })
            reused += 1
        else:
            pending.append((k, rec["text"], rec["label_text"]))

    logger.info("[%s | %s] reused=%d  pending=%d  (already in new ckpt=%d)",
                model, strategy, reused, len(pending), len(done_by_key))

    if pending and not args.dry_run:
        client, field, value = get_client_and_invoke(model, registry, cfg)
        if not client.smoke_test(field, value):
            logger.error("Smoke test failed for %s — skipping", model)
            return None
        exp = cfg.get("experiment", {})
        asyncio.run(process_pending(
            client, field, value, strategy, pending, ckpt_file,
            max_tokens=exp.get("max_tokens", 1024),
            temperature=exp.get("temperature", 0.0),
            concurrency=args.concurrency,
            delay=exp.get("delay_between_calls", 0.5),
        ))

    # reload everything and compute metrics
    all_recs = {}
    for line in open(ckpt_file, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            all_recs[r["key"]] = r
        except Exception:
            pass

    pairs, n_failed, cost = [], 0, 0.0
    for rec in bert_recs:
        r = all_recs.get(rec["key"])
        if r is None:
            n_failed += 1
            continue
        cost += float(r.get("cost_usd") or 0.0)
        pt = r.get("pred_text")
        if pt in CLASS_NAMES:
            pairs.append((rec["label_text"], pt))
        else:
            n_failed += 1

    metrics = compute_metrics(pairs)
    out = {
        "model": model, "strategy": strategy, "test_set": "bert_canonical",
        "n_total": len(bert_recs), "n_reused": reused,
        "n_failed_or_unpredicted": n_failed,
        "parse_failure_rate": round(n_failed / len(bert_recs), 4),
        "api_cost_usd_new_calls": round(cost, 4),
        **metrics,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{model}__{strategy}__berttest.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info("[%s | %s] F1-macro=%.4f acc=%.4f failures=%d new_cost=$%.4f",
                model, strategy, metrics["f1_macro"], metrics["accuracy"],
                n_failed, cost)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--strategies", nargs="+", default=["all"])
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true",
                    help="Only shows reuse/pending stats, without calling the API")
    args = ap.parse_args()

    models = ALL_MODELS if "all" in args.models else args.models
    strategies = ALL_STRATEGIES if "all" in args.strategies else args.strategies

    cfg = load_config(str(_ROOT / "configs" / "_base_llm.yaml"))
    registry = load_model_registry(cfg["data"]["bedrock_registry"])

    bert_recs = load_bert_test()
    logger.info("Canonical test set (BERT): %d excerpts | dist: %s",
                len(bert_recs), dict(Counter(r["label_text"] for r in bert_recs)))
    prior = build_prior_predictions()

    summary = []
    for model in models:
        # load model-specific config (if it exists)
        mcfg_path = _ROOT / "configs" / f"{model}.yaml"
        cfg_m = dict(cfg)
        if mcfg_path.exists():
            mc = load_config(mcfg_path)
            cfg_m.update({k: v for k, v in mc.items() if k != "_base"})
        for strategy in strategies:
            try:
                r = run_model_strategy(model, strategy, bert_recs, prior, registry, cfg_m, args)
                if r:
                    summary.append(r)
            except Exception as e:
                logger.error("Failure in %s/%s: %s", model, strategy, str(e)[:200])

    # Consolidated CSV
    if summary:
        import csv
        cols = ["model", "strategy", "n_evaluated", "n_reused",
                "parse_failure_rate", "accuracy", "f1_macro",
                "f1_tese", "f1_fato", "f1_ruido", "api_cost_usd_new_calls"]
        with open(RESULTS_DIR / "summary_berttest.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in sorted(summary, key=lambda x: -x["f1_macro"]):
                w.writerow(r)
        total_new = sum(r["api_cost_usd_new_calls"] for r in summary)
        logger.info("=" * 60)
        logger.info("DONE. %d combinations. TOTAL cost of new calls: $%.4f",
                    len(summary), total_new)
        logger.info("Results: %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
