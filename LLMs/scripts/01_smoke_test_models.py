"""
01_smoke_test_models.py — Tests all 8 models with "hi" before running.

Generates a JSON report with the status of each model.

Usage:
  python scripts/01_smoke_test_models.py
  python scripts/01_smoke_test_models.py --models claude_sonnet_46 sabia4
"""

import argparse
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
logger = logging.getLogger("smoke_test")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "Models_BERT" / "src"))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from ftr_llm.bedrock_client import BedrockClient
from ftr_llm.maritaca_client import MaritacaClient
from ftr_llm.config import load_config, load_model_registry, get_invoke_field

# Alias -> config mapping (the 8 experiment models)
EXPERIMENT_MODELS = [
    "claude_sonnet_46",
    "qwen3_32b",
    "mistral_large_3",
    "gemma_3_27b",
    "deepseek_v32",
    "claude_3_haiku",
    "magistral_small_2509",
    "sabia4",
]


def get_client(model_alias: str, registry: dict, cfg: dict):
    if model_alias == "sabia4":
        api_key = os.environ.get("API_KEY_MARITACA", "")
        return MaritacaClient(api_key=api_key), "invoke_maritaca", "sabia-4"

    models = registry.get("rankingGeneralPurpose", [])
    entry = next((m for m in models if m["alias"] == model_alias), None)
    if entry is None:
        raise ValueError(f"Model not found in registry: {model_alias}")

    field, value = get_invoke_field(entry)
    fallback = next(
        (m["invokeWith"]["value"] for m in models if m["alias"] == cfg.get("fallback_model_alias")),
        None
    )
    region = os.environ.get("AWS_DEFAULT_REGION", "sa-east-1")
    client = BedrockClient(region=region, fallback_model_id=fallback)
    return client, field, value


def main():
    parser = argparse.ArgumentParser(description="Smoke test for models")
    parser.add_argument("--config", default=str(_ROOT / "configs" / "_base_llm.yaml"))
    parser.add_argument("--models", nargs="+", default=EXPERIMENT_MODELS)
    args = parser.parse_args()

    cfg = load_config(args.config)
    registry = load_model_registry(cfg["data"]["bedrock_registry"])
    reports_dir = Path(cfg["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for alias in args.models:
        logger.info("Testing: %s", alias)
        try:
            client, field, value = get_client(alias, registry, cfg)
            ok = client.smoke_test(field, value)
            results.append({"model": alias, "ok": ok, "error": None})
        except Exception as e:
            logger.error("Error testing %s: %s", alias, e)
            results.append({"model": alias, "ok": False, "error": str(e)})

    out_path = reports_dir / "smoke_test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok_count = sum(1 for r in results if r["ok"])
    logger.info("Result: %d/%d models OK — report: %s", ok_count, len(results), out_path)

    failed = [r["model"] for r in results if not r["ok"]]
    if failed:
        logger.warning("Models with failure: %s", failed)


if __name__ == "__main__":
    main()
