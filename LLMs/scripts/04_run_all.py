"""
04_run_all.py — Orquestrador completo: todos os modelos × todas as estratégias.

Roda os experimentos em sequência, modelo a modelo, estratégia a estratégia.
Se cair no meio (Ctrl+C, queda de rede, timeout), basta rodar novamente:
  - Modelos já concluídos são pulados automaticamente (--skip-done)
  - Dentro de cada modelo, checkpoints por linha garantem retomada do índice exato

Ordem de execução (custo crescente):
  1. claude_3_haiku       — leve, barato, valida tudo primeiro
  2. gemma_3_27b          — médio, sem custo Claude
  3. magistral_small_2509 — médio Mistral
  4. qwen3_32b            — médio Qwen
  5. deepseek_v32         — grande, custo baixo
  6. mistral_large_3      — grande Mistral
  7. sabia4               — Maritaca (API separada)
  8. claude_sonnet_46     — elite, mais caro — deixa por último

Uso:
  # Roda tudo no val set (desenvolvimento)
  python scripts/04_run_all.py --split val

  # Avaliação final no test set
  python scripts/04_run_all.py --split test

  # Só alguns modelos
  python scripts/04_run_all.py --models claude_3_haiku gemma_3_27b --split val

  # Retoma de onde parou (padrão — checkpoints automáticos)
  python scripts/04_run_all.py --split val

  # Pula modelos cujo resultado JSON já existe
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

# Ordem por custo crescente
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
    """Verifica se o resultado JSON desse experimento já existe e tem F1-Macro."""
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
    """Chama 02_run_experiment.py como subprocesso. Retorna True se bem-sucedido."""
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

    logger.info("Iniciando: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(_ROOT))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Roda todos os experimentos LLM")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="Split a avaliar (val=desenvolvimento, test=final)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset de modelos (padrão: todos em ordem de custo)")
    parser.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES,
                        choices=ALL_STRATEGIES + ["all"])
    parser.add_argument("--config", default=str(_ROOT / "configs" / "_base_llm.yaml"))
    parser.add_argument("--skip-done", action="store_true",
                        help="Pula experimentos cujo resultado JSON já existe com F1 > 0")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay entre chamadas em segundos (padrão 0.5)")
    parser.add_argument("--no-smoke-test", action="store_true",
                        help="Pula smoke test em todos os modelos")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(cfg["output"]["results_dir"])
    reports_dir = Path(cfg["output"]["reports_dir"])

    models = args.models or MODEL_ORDER
    strategies = ALL_STRATEGIES if "all" in args.strategies else args.strategies

    # Relatório de progresso
    total = len(models) * len(strategies)
    done_count = 0
    skipped_count = 0
    failed = []

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETO")
    logger.info("  Split:      %s", args.split)
    logger.info("  Modelos:    %s", models)
    logger.info("  Estratégias: %s", strategies)
    logger.info("  Total:      %d experimentos", total)
    logger.info("  Skip-done:  %s", args.skip_done)
    logger.info("=" * 70)

    session_start = time.time()
    first_model = True

    for model in models:
        # Verifica quais estratégias já estão prontas para esse modelo
        pending_strategies = []
        for strategy in strategies:
            if args.skip_done and result_exists(results_dir, model, strategy, args.split):
                logger.info("Pulando (já existe): %s × %s", model, strategy)
                skipped_count += 1
                done_count += 1
            else:
                pending_strategies.append(strategy)

        if not pending_strategies:
            logger.info("Modelo %s: todas as estratégias já concluídas.", model)
            continue

        # Smoke test só no primeiro modelo (os demais pulam)
        no_smoke = args.no_smoke_test or not first_model
        first_model = False

        logger.info("-" * 60)
        logger.info("Modelo %s — %d estratégias pendentes: %s",
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
                "Progresso: %d/%d concluídos — tempo total: %.0fs",
                done_count, total, elapsed,
            )
        else:
            failed.append(model)
            logger.error(
                "Modelo %s falhou. Continuando com o próximo...", model
            )
            # Pausa antes do próximo modelo em caso de falha (rate limit, etc.)
            time.sleep(10)

    # Consolidação final
    logger.info("=" * 70)
    logger.info("Todos os modelos processados. Gerando consolidação final...")

    consolidate_cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "03_consolidate_results.py"),
        "--config", args.config,
    ]
    subprocess.run(consolidate_cmd, cwd=str(_ROOT))

    # Resumo final
    elapsed_total = time.time() - session_start
    logger.info("=" * 70)
    logger.info("CONCLUÍDO")
    logger.info("  Tempo total:    %.0f min", elapsed_total / 60)
    logger.info("  Experimentos:   %d/%d", done_count, total)
    logger.info("  Pulados:        %d", skipped_count)
    if failed:
        logger.warning("  Modelos com falha: %s", failed)
        logger.warning("  Para retentar: python scripts/04_run_all.py --split %s --models %s",
                       args.split, " ".join(failed))
    logger.info("  Resultados:   %s", results_dir)
    logger.info("  Gráficos:     %s", reports_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
