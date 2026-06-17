"""
Bedrock client with retry, fallback and cost logging.

Uses the Converse API (supports all models from bedrock_modelos_refatorado.json).
Includes:
  - Exponential retry with jitter
  - Fallback to backup model if the primary fails
  - Cost estimate per call (input + output tokens)
  - Smoke-test ("hi") to validate model before processing the dataset
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger("ftr_llm.bedrock")

# ---------------------------------------------------------------------------
# Estimated prices (USD / 1M tokens) — update according to Bedrock pricing
# Conservative values for cost estimation.
# ---------------------------------------------------------------------------
_PRICE_TABLE: dict[str, dict[str, float]] = {
    # Claude
    "claude-opus":   {"input": 15.00, "output": 75.00},
    "claude-sonnet": {"input":  3.00, "output": 15.00},
    "claude-haiku":  {"input":  0.80, "output":  4.00},
    "claude-3-haiku": {"input": 0.25, "output":  1.25},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    # Mistral
    "mistral-large":  {"input": 4.00, "output": 12.00},
    "mistral-7b":     {"input": 0.20, "output":  0.20},
    "mixtral-8x7b":   {"input": 0.45, "output":  0.70},
    "ministral":      {"input": 0.10, "output":  0.30},
    "magistral":      {"input": 2.00, "output":  6.00},
    "devstral":       {"input": 0.50, "output":  1.50},
    # Qwen
    "qwen3-vl-235b":  {"input": 2.00, "output":  6.00},
    "qwen3-32b":      {"input": 0.50, "output":  1.50},
    "qwen3-80b":      {"input": 0.80, "output":  2.50},
    # DeepSeek
    "deepseek":       {"input": 0.27, "output":  1.10},
    # GLM
    "glm":            {"input": 0.50, "output":  1.50},
    # MiniMax
    "minimax":        {"input": 0.20, "output":  0.80},
    # NVIDIA Nemotron
    "nemotron":       {"input": 0.20, "output":  0.60},
    # Kimi
    "kimi":           {"input": 1.00, "output":  3.00},
    # Gemma
    "gemma":          {"input": 0.10, "output":  0.30},
    # GPT-OSS
    "gpt-oss":        {"input": 0.80, "output":  2.40},
    # Maritaca Sabiá (direct API, not Bedrock)
    "sabia":          {"input": 0.50, "output":  1.50},
    "default":        {"input": 1.00, "output":  3.00},
}


def _price_for_model(model_id: str) -> dict[str, float]:
    lower = model_id.lower()
    for key, price in _PRICE_TABLE.items():
        if key in lower:
            return price
    return _PRICE_TABLE["default"]


def estimate_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    price = _price_for_model(model_id)
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


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


class BedrockClient:
    """
    Wrapper over boto3 Bedrock Converse with retry, fallback and cost metrics.
    """

    def __init__(
        self,
        region: str = "sa-east-1",
        max_retries: int = 4,
        base_delay: float = 2.0,
        timeout: float = 120.0,
        fallback_model_id: str | None = None,
    ):
        self.region = region
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout
        self.fallback_model_id = fallback_model_id
        from botocore.config import Config as BotocoreConfig
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=BotocoreConfig(
                read_timeout=int(timeout),
                max_pool_connections=50,  # supports concurrency without "connection pool exhausted"
            ),
        )
        self.total_cost_usd: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    def _build_converse_args(
        self,
        invoke_field: str,
        invoke_value: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict:
        args: dict[str, Any] = {
            "messages": [{"role": "user", "content": [{"text": user_message}]}],
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        # Some Claude models do not accept temperature — omit if 0
        if temperature > 0:
            args["inferenceConfig"]["temperature"] = temperature

        if system_prompt:
            args["system"] = [{"text": system_prompt}]

        if invoke_field == "inferenceProfileId":
            args["modelId"] = invoke_value
        else:
            args["modelId"] = invoke_value

        return args

    def _call_once(
        self,
        invoke_field: str,
        invoke_value: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> CallResult:
        args = self._build_converse_args(
            invoke_field, invoke_value, system_prompt, user_message, max_tokens, temperature
        )
        t0 = time.perf_counter()
        response = self._client.converse(**args)
        latency_ms = (time.perf_counter() - t0) * 1000

        text = response["output"]["message"]["content"][0]["text"]
        usage = response.get("usage", {})
        in_tok = usage.get("inputTokens", 0)
        out_tok = usage.get("outputTokens", 0)
        cost = estimate_cost_usd(invoke_value, in_tok, out_tok)

        self.total_cost_usd += cost
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok

        return CallResult(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            cost_usd=cost,
            model_used=invoke_value,
            ok=True,
            raw_response=response,
        )

    def call(
        self,
        invoke_field: str,
        invoke_value: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> CallResult:
        """Calls the model with exponential retry + jitter. Uses fallback if retries are exhausted."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._call_once(
                    invoke_field, invoke_value, system_prompt, user_message, max_tokens, temperature
                )
            except ClientError as e:
                code = e.response["Error"]["Code"]
                last_error = str(e)
                logger.warning(
                    "Attempt %d/%d failed (%s): %s",
                    attempt + 1, self.max_retries + 1, code, last_error[:120]
                )
                if code in ("ThrottlingException", "ServiceUnavailableException", "ModelStreamErrorException"):
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info("Waiting %.1fs before retry...", delay)
                    time.sleep(delay)
                elif code in ("ValidationException", "ModelNotReadyException"):
                    break
                else:
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)
            except Exception as e:
                last_error = str(e)
                logger.warning("Unexpected error attempt %d: %s", attempt + 1, last_error[:120])
                time.sleep(self.base_delay * (2 ** attempt))

        # Try fallback
        if self.fallback_model_id and self.fallback_model_id != invoke_value:
            logger.warning("Using fallback model: %s", self.fallback_model_id)
            try:
                return self._call_once(
                    "modelId", self.fallback_model_id, system_prompt, user_message, max_tokens, temperature
                )
            except Exception as fe:
                last_error = f"fallback also failed: {fe}"

        return CallResult(
            text="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            cost_usd=0.0,
            model_used=invoke_value,
            ok=False,
            error=last_error,
        )

    def smoke_test(self, invoke_field: str, invoke_value: str) -> bool:
        """Tests the model with 'Reply only with: hi'. Returns True if it responds."""
        logger.info("Smoke test: %s", invoke_value)
        result = self.call(
            invoke_field, invoke_value,
            system_prompt="",
            user_message="Reply only with: hi",
            max_tokens=10,
            temperature=0.0,
        )
        ok = result.ok and bool(result.text.strip())
        logger.info("Smoke test %s: %s (latency=%.0fms)", invoke_value, "OK" if ok else "FAIL", result.latency_ms)
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
        """Async version of call() — runs the synchronous boto3 in a thread pool."""
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
