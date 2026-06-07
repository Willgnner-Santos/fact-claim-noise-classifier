"""
Cliente para Maritaca AI (Sabiá-4) via API REST direta.

A Maritaca não está no Bedrock — usa endpoint próprio.
Interface idêntica ao BedrockClient para o runner não precisar saber a diferença.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("ftr_llm.maritaca")

MARITACA_API_URL = "https://chat.maritaca.ai/api/chat/completions"
MARITACA_MODEL_ID = "sabia-4"  # nome exato na API Maritaca
MARITACA_PRICE = {"input": 0.50, "output": 1.50}  # USD / 1M tokens (estimativa)


@dataclass
class CallResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    model_used: str
    ok: bool
    error: str | None = None
    raw_response: dict | None = field(default=None, repr=False)


class MaritacaClient:
    """Cliente REST para Sabiá-4 (Maritaca AI)."""

    def __init__(
        self,
        api_key: str,
        model: str = MARITACA_MODEL_ID,
        max_retries: int = 4,
        base_delay: float = 2.0,
        timeout: float = 120.0,
    ):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout
        self.total_cost_usd: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    def _call_once(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> CallResult:
        import urllib.request

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")

        req = urllib.request.Request(
            MARITACA_API_URL,
            data=payload,
            headers={
                "Authorization": f"Key {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        latency_ms = (time.perf_counter() - t0) * 1000

        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)
        cost = (in_tok * MARITACA_PRICE["input"] + out_tok * MARITACA_PRICE["output"]) / 1_000_000

        self.total_cost_usd += cost
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok

        return CallResult(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            cost_usd=cost,
            model_used=self.model,
            ok=True,
            raw_response=body,
        )

    def call(
        self,
        invoke_field: str,  # ignorado — compatibilidade com BedrockClient
        invoke_value: str,  # ignorado
        system_prompt: str,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> CallResult:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._call_once(system_prompt, user_message, max_tokens, temperature)
            except Exception as e:
                last_error = str(e)
                logger.warning("Maritaca tentativa %d/%d falhou: %s", attempt + 1, self.max_retries + 1, last_error[:120])
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)

        return CallResult(
            text="", input_tokens=0, output_tokens=0,
            latency_ms=0.0, cost_usd=0.0, model_used=self.model,
            ok=False, error=last_error,
        )

    def smoke_test(self, invoke_field: str = "", invoke_value: str = "") -> bool:
        result = self.call("", "", "", "Responda apenas com: oi", max_tokens=10)
        ok = result.ok and bool(result.text.strip())
        logger.info("Smoke test Sabiá-4: %s (latency=%.0fms)", "OK" if ok else "FAIL", result.latency_ms)
        return ok

    async def async_call(
        self,
        invoke_field: str,
        invoke_value: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> CallResult:
        """Versão async de call() — executa o HTTP síncrono em thread pool."""
        return await asyncio.to_thread(
            self.call,
            invoke_field, invoke_value,
            system_prompt, user_message,
            max_tokens, temperature,
        )

    def cost_summary(self) -> dict:
        return {
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
