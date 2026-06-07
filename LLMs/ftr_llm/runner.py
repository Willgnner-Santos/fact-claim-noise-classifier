"""
Runner principal: itera sobre o dataset, chama o LLM, salva checkpoints e métricas.

Suporta:
  - Retomada de checkpoint (--resume automático)
  - Rate limiting (delay entre chamadas)
  - Logging de custo acumulado
  - Smoke-test antes de iniciar
  - Saída em JSONL de checkpoint + JSON de resultados finais
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from ftr_llm.checkpoint import (
    append_record,
    async_append_record,
    checkpoint_path,
    checkpoint_to_lists,
    load_checkpoint,
)
from ftr_llm.parser import parse_response
from ftr_llm.prompts import build_prompt

logger = logging.getLogger("ftr_llm.runner")


def run_experiment(
    records: list[dict],
    model_alias: str,
    invoke_field: str,
    invoke_value: str,
    strategy: str,
    split: str,
    client,                         # BedrockClient ou MaritacaClient
    ckpt_dir: Path,
    max_tokens: int = 512,
    temperature: float = 0.0,
    delay_between_calls: float = 0.3,
    smoke_test: bool = True,
    max_parse_failures_pct: float = 0.20,  # aborta se >20% de falhas de parse
) -> dict[str, Any]:
    """
    Roda um experimento completo (modelo × estratégia × split).

    Returns:
        dict com y_true, y_pred, métricas brutas, custo e metadados.
    """
    # --- Smoke test ---
    if smoke_test:
        ok = client.smoke_test(invoke_field, invoke_value)
        if not ok:
            logger.error("Smoke test falhou para %s. Abortando.", model_alias)
            return {"ok": False, "error": "smoke_test_failed", "model": model_alias}

    # --- Carrega checkpoint ---
    ckpt_file = checkpoint_path(ckpt_dir, model_alias, strategy, split)
    done = load_checkpoint(ckpt_file)
    n_total = len(records)
    n_done = len(done)
    logger.info(
        "Experimento %s × %s × %s: %d/%d já processados",
        model_alias, strategy, split, n_done, n_total
    )

    # --- Loop principal ---
    parse_failures = 0
    for idx, record in enumerate(records):
        if idx in done:
            continue

        text = record["text"]
        true_label = record["label"]

        try:
            system_prompt, user_message = build_prompt(strategy, text)
            result = client.call(
                invoke_field, invoke_value,
                system_prompt, user_message,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning("Erro na chamada idx=%d: %s", idx, str(e)[:120])
            ckpt_rec = {
                "idx": idx, "label": true_label, "pred": None,
                "parse_ok": False, "error": str(e),
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                "latency_ms": 0.0, "model": model_alias, "strategy": strategy,
                "length_bin": record.get("length_bin", ""),
            }
            append_record(ckpt_file, ckpt_rec)
            parse_failures += 1
            continue

        if not result.ok:
            ckpt_rec = {
                "idx": idx, "label": true_label, "pred": None,
                "parse_ok": False, "error": result.error,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
                "latency_ms": result.latency_ms,
                "model": model_alias, "strategy": strategy,
                "length_bin": record.get("length_bin", ""),
            }
            append_record(ckpt_file, ckpt_rec)
            parse_failures += 1
        else:
            parsed = parse_response(result.text, strategy)
            pred = parsed["label"]
            if pred is None:
                parse_failures += 1

            ckpt_rec = {
                "idx": idx,
                "label": true_label,
                "label_text": record.get("label_text", ""),
                "pred": pred,
                "pred_text": parsed["classe"],
                "parse_ok": parsed["parse_ok"],
                "trecho_evidencia": parsed.get("trecho_evidencia"),
                "justificativa": parsed.get("justificativa"),
                "q1_narrativa": parsed.get("q1_narrativa"),
                "q2_argumentacao": parsed.get("q2_argumentacao"),
                "q3_ruido": parsed.get("q3_ruido"),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
                "latency_ms": result.latency_ms,
                "model": model_alias,
                "strategy": strategy,
                "length_bin": record.get("length_bin", ""),
                "n_words": record.get("n_words", 0),
            }
            append_record(ckpt_file, ckpt_rec)

        # Progress log a cada 100
        processed_now = idx + 1
        if processed_now % 100 == 0 or processed_now == n_total:
            cost_so_far = client.cost_summary()["total_cost_usd"]
            logger.info(
                "[%s|%s|%s] %d/%d — custo acum.: $%.4f — falhas parse: %d",
                model_alias, strategy, split,
                processed_now, n_total,
                cost_so_far, parse_failures,
            )

        # Rate limiting
        if delay_between_calls > 0:
            time.sleep(delay_between_calls)

        # Aborta se muitas falhas
        n_processed = idx + 1 - len([i for i in range(idx + 1) if i in done])
        if n_processed >= 50:
            fail_pct = parse_failures / n_processed
            if fail_pct > max_parse_failures_pct:
                logger.error(
                    "Abortando: %.0f%% de falhas de parse (limite %.0f%%)",
                    fail_pct * 100, max_parse_failures_pct * 100
                )
                return {"ok": False, "error": "too_many_parse_failures", "model": model_alias}

    # --- Monta resultado final ---
    done_final = load_checkpoint(ckpt_file)
    y_true, y_pred, failed_idxs = checkpoint_to_lists(done_final, n_total)

    return {
        "ok": True,
        "model": model_alias,
        "strategy": strategy,
        "split": split,
        "n_total": n_total,
        "n_predicted": len(y_pred),
        "n_failed": len(failed_idxs),
        "parse_failure_rate": round(len(failed_idxs) / n_total, 4) if n_total else 0,
        "y_true": y_true,
        "y_pred": y_pred,
        "cost_summary": client.cost_summary(),
        "checkpoint_file": str(ckpt_file),
    }


# ---------------------------------------------------------------------------
# Runner assíncrono — paralelismo controlado por Semaphore
# ---------------------------------------------------------------------------

def _build_ckpt_record(
    idx: int, record: dict, model_alias: str, strategy: str,
    result=None, parsed=None, error: str | None = None,
) -> dict:
    """Monta o dict de checkpoint — centraliza para não duplicar entre sync/async."""
    base = {
        "idx": idx,
        "label": record["label"],
        "label_text": record.get("label_text", ""),
        "model": model_alias,
        "strategy": strategy,
        "length_bin": record.get("length_bin", ""),
        "n_words": record.get("n_words", 0),
    }
    if error or result is None or not result.ok:
        base.update({
            "pred": None, "pred_text": None, "parse_ok": False,
            "error": error or (result.error if result else "unknown"),
            "raw_text": result.text if result else "",
            "input_tokens": result.input_tokens if result else 0,
            "output_tokens": result.output_tokens if result else 0,
            "cost_usd": result.cost_usd if result else 0.0,
            "latency_ms": result.latency_ms if result else 0.0,
        })
    else:
        base.update({
            "pred": parsed["label"],
            "pred_text": parsed["classe"],
            "parse_ok": parsed["parse_ok"],
            "raw_text": result.text,
            "trecho_evidencia": parsed.get("trecho_evidencia"),
            "justificativa": parsed.get("justificativa"),
            "q1_narrativa": parsed.get("q1_narrativa"),
            "q2_argumentacao": parsed.get("q2_argumentacao"),
            "q3_ruido": parsed.get("q3_ruido"),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
        })
    return base


async def _process_one(
    idx: int,
    record: dict,
    model_alias: str,
    invoke_field: str,
    invoke_value: str,
    strategy: str,
    client,
    ckpt_file: Path,
    semaphore: asyncio.Semaphore,
    progress: dict,           # {"done": int, "failures": int} — compartilhado
    max_tokens: int,
    temperature: float,
    min_delay: float,
) -> None:
    """Processa uma instância com controle de concorrência."""
    async with semaphore:
        system_prompt, user_message = build_prompt(strategy, record["text"])
        t0 = time.perf_counter()
        try:
            result = await client.async_call(
                invoke_field, invoke_value,
                system_prompt, user_message,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            ckpt_rec = _build_ckpt_record(idx, record, model_alias, strategy, error=str(e))
            await async_append_record(ckpt_file, ckpt_rec)
            progress["failures"] += 1
            progress["done"] += 1
            return

        if result.ok:
            parsed = parse_response(result.text, strategy)
            if parsed["label"] is None:
                progress["failures"] += 1
        else:
            parsed = {"label": None, "classe": None, "parse_ok": False,
                      "trecho_evidencia": None, "justificativa": None,
                      "q1_narrativa": None, "q2_argumentacao": None, "q3_ruido": None}
            progress["failures"] += 1

        ckpt_rec = _build_ckpt_record(idx, record, model_alias, strategy, result, parsed)
        await async_append_record(ckpt_file, ckpt_rec)
        progress["done"] += 1

        # Delay mínimo respeitado por slot (não bloqueia os outros slots)
        elapsed = time.perf_counter() - t0
        remaining = min_delay - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining)


async def run_experiment_async(
    records: list[dict],
    model_alias: str,
    invoke_field: str,
    invoke_value: str,
    strategy: str,
    split: str,
    client,
    ckpt_dir: Path,
    max_tokens: int = 512,
    temperature: float = 0.0,
    delay_between_calls: float = 0.3,
    smoke_test: bool = True,
    max_parse_failures_pct: float = 0.20,
    concurrency: int = 5,
) -> dict[str, Any]:
    """
    Runner assíncrono: N chamadas concorrentes controladas por Semaphore.

    concurrency=5 é o padrão conservador — equivale a ~5x a velocidade do runner
    sequencial sem risco de throttling para nenhum dos provedores testados.
    """
    if smoke_test:
        ok = client.smoke_test(invoke_field, invoke_value)
        if not ok:
            logger.error("Smoke test falhou para %s.", model_alias)
            return {"ok": False, "error": "smoke_test_failed", "model": model_alias}

    ckpt_file = checkpoint_path(ckpt_dir, model_alias, strategy, split)
    done = load_checkpoint(ckpt_file)
    n_total = len(records)
    pending = [
        (idx, rec) for idx, rec in enumerate(records) if idx not in done
    ]
    logger.info(
        "Async runner %s × %s × %s: %d/%d pendentes (concurrency=%d)",
        model_alias, strategy, split, len(pending), n_total, concurrency,
    )

    if not pending:
        done_final = load_checkpoint(ckpt_file)
        y_true, y_pred, failed_idxs = checkpoint_to_lists(done_final, n_total)
        return {
            "ok": True, "model": model_alias, "strategy": strategy, "split": split,
            "n_total": n_total, "n_predicted": len(y_pred), "n_failed": len(failed_idxs),
            "parse_failure_rate": 0.0, "y_true": y_true, "y_pred": y_pred,
            "cost_summary": client.cost_summary(), "checkpoint_file": str(ckpt_file),
        }

    semaphore = asyncio.Semaphore(concurrency)
    progress = {"done": 0, "failures": 0}

    # Cria todas as tasks mas o Semaphore limita execução simultânea
    tasks = [
        asyncio.create_task(
            _process_one(
                idx, rec, model_alias, invoke_field, invoke_value,
                strategy, client, ckpt_file, semaphore, progress,
                max_tokens, temperature, delay_between_calls,
            )
        )
        for idx, rec in pending
    ]

    # Aguarda em lotes para poder logar progresso
    log_interval = max(50, len(pending) // 20)
    last_logged = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        if progress["done"] - last_logged >= log_interval or progress["done"] == len(pending):
            cost = client.cost_summary()["total_cost_usd"]
            logger.info(
                "[%s|%s|%s] %d/%d — custo acum.: $%.4f — falhas: %d",
                model_alias, strategy, split,
                len(done) + progress["done"], n_total,
                cost, progress["failures"],
            )
            last_logged = progress["done"]

            # Aborta se muitas falhas — threshold relaxado para amostras pequenas
            # para não interromper experimentos de validação prematuramente.
            if progress["done"] >= 50:
                fail_pct = progress["failures"] / progress["done"]
                effective_limit = (
                    max_parse_failures_pct
                    if len(pending) >= 300
                    else min(0.40, max_parse_failures_pct * 2.0)
                )
                if fail_pct > effective_limit:
                    logger.error(
                        "Abortando: %.0f%% de falhas (limite %.0f%% para n=%d)",
                        fail_pct * 100, effective_limit * 100, len(pending),
                    )
                    for t in tasks:
                        t.cancel()
                    return {"ok": False, "error": "too_many_parse_failures", "model": model_alias}

    done_final = load_checkpoint(ckpt_file)
    y_true, y_pred, failed_idxs = checkpoint_to_lists(done_final, n_total)

    return {
        "ok": True,
        "model": model_alias,
        "strategy": strategy,
        "split": split,
        "n_total": n_total,
        "n_predicted": len(y_pred),
        "n_failed": len(failed_idxs),
        "parse_failure_rate": round(len(failed_idxs) / n_total, 4) if n_total else 0,
        "y_true": y_true,
        "y_pred": y_pred,
        "cost_summary": client.cost_summary(),
        "checkpoint_file": str(ckpt_file),
    }
