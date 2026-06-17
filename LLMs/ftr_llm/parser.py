"""
Parser for LLM JSON responses.

Robust to:
  - JSON wrapped in markdown (```json ... ```)
  - Extra spaces/newlines
  - Fields with varying casing (Fato, FATO, fato)
  - Partial responses (class only, without evidence)

Always returns a normalized dict with at least {"classe": str, "parse_ok": bool}.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ftr_llm import LABEL2ID

_VALID_CLASSES = set(LABEL2ID.keys())  # {"tese", "fato", "ruido"}

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_INLINE_JSON_RE = re.compile(r"\{[\s\S]*\}")

# Markers that precede the conclusion in CoT responses — the class after these
# patterns is the final decision, not intermediate reasoning text.
_CONCLUSION_RE = re.compile(
    r"(?:classificaç[aã]o(?:\s+final)?|portanto|logo|conclus[aã]o|"
    r"classe(?:\s+final)?|resposta|therefore|classification)\s*[:\-–]\s*",
    re.IGNORECASE,
)


def _extract_json_str(text: str) -> str | None:
    """Tries to extract a JSON string from a possibly wrapped response."""
    # 1. Markdown fence
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # 2. Inline { ... }
    m = _INLINE_JSON_RE.search(text)
    if m:
        return m.group(0)
    return None


_MISTO_ALIASES = {"misto", "mista", "híbrido", "hibrido", "ambíguo", "ambiguo", "mixed"}


def _normalize_class(raw: str) -> str | None:
    c = raw.strip().lower()
    if c in _VALID_CLASSES:
        return c
    # fuzzy prefix match
    for valid in _VALID_CLASSES:
        if c.startswith(valid[:3]):
            return valid
    # "misto"/"mista" — return None to trigger _keyword_fallback on the full raw_text.
    # The model reasoning may mention the predominant class after the JSON.
    # If nothing is found, it stays as a real failure (pred=None, parse_ok=False) —
    # more honest than an arbitrary tiebreak that contaminates the metrics.
    return None


def _keyword_fallback(text: str) -> str | None:
    """
    Extracts the class as a last resort when there is no JSON.

    Three-step strategy:
    1. Looks for the class after conclusion markers ("portanto:", "classificação: ")
       — in CoT the final decision appears after these markers, not in the middle of reasoning.
    2. If no marker is found, uses the LAST occurrence of a class in the text
       — in CoT the model typically concludes at the end.
    3. Only accepts the first occurrence (old behavior) if it is the sole match.

    Never returns the first keyword found without first checking the other options,
    avoiding the classic false positive: "describes a fact, but concludes as thesis" -> tese.
    """
    lower = text.lower()

    # Step 1 — after conclusion marker
    for m in _CONCLUSION_RE.finditer(lower):
        snippet = lower[m.end():m.end() + 20]
        for cls in _VALID_CLASSES:
            if cls in snippet:
                return cls

    # Step 2 — last occurrence in text
    last_pos = -1
    last_cls = None
    for cls in _VALID_CLASSES:
        pos = lower.rfind(cls)
        if pos > last_pos:
            last_pos = pos
            last_cls = cls

    # Only uses the last occurrence if it is unambiguous:
    # ensures the word is isolated (not a substring of another word).
    if last_cls and last_pos >= 0:
        before = lower[last_pos - 1] if last_pos > 0 else " "
        after_pos = last_pos + len(last_cls)
        after = lower[after_pos] if after_pos < len(lower) else " "
        if not before.isalpha() and not after.isalpha():
            return last_cls

    return None


def parse_response(text: str, strategy: str = "zero_shot") -> dict[str, Any]:
    """
    Parses the LLM response and returns a dict with:
      - classe: str | None
      - label: int | None
      - parse_ok: bool
      - trecho_evidencia: str | None (if evidence_constrained)
      - justificativa: str | None (if evidence_constrained)
      - q1/q2/q3: str | None (if decomposed)
      - raw_text: str
    """
    result: dict[str, Any] = {
        "classe": None,
        "label": None,
        "parse_ok": False,
        "trecho_evidencia": None,
        "justificativa": None,
        "q1_narrativa": None,
        "q2_argumentacao": None,
        "q3_ruido": None,
        "raw_text": text,
    }

    if not text:
        return result

    json_str = _extract_json_str(text)
    if json_str is None:
        classe = _keyword_fallback(text)
        if classe:
            result["classe"] = classe
            result["label"] = LABEL2ID[classe]
            result["parse_ok"] = True
        return result

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try to fix single quotes
        try:
            data = json.loads(json_str.replace("'", '"'))
        except Exception:
            return result

    if not isinstance(data, dict):
        # If the model returned a list of objects, try to take the first one
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            data = data[0]
        else:
            return result

    # Intermediate fields of decomposed
    q1_ruido      = str(data.get("q1_ruido", "")).strip().lower()
    q2_argum      = str(data.get("q2_argumentacao", "")).strip().lower()
    q3_narrativa  = str(data.get("q3_narrativa", "")).strip().lower()
    # compat with old format
    q1_narrativa_old = str(data.get("q1_narrativa", "")).strip().lower()
    q3_ruido_old     = str(data.get("q3_ruido", "")).strip().lower()

    is_decomposed = q1_ruido or q2_argum or q3_narrativa or q1_narrativa_old or q3_ruido_old

    if is_decomposed:
        # Apply the priority rule IN CODE — do not trust the model's "classe" field,
        # which has shown to ignore the decision logic even when stated explicitly in the prompt.
        # Priority: ruido > tese > fato
        effective_q1_ruido = q1_ruido or q3_ruido_old
        effective_q2_argum = q2_argum
        effective_q3_narr  = q3_narrativa or q1_narrativa_old

        if effective_q1_ruido == "sim":
            classe = "ruido"
        elif effective_q2_argum == "sim":
            classe = "tese"
        elif effective_q3_narr == "sim":
            classe = "fato"
        else:
            classe = "ruido"  # conservative fallback
        parse_ok = True
    else:
        # Non-decomposed strategies: extract class from "classe" field
        raw_classe = str(data.get("classe") or data.get("class") or data.get("classificacao") or "")
        classe = _normalize_class(raw_classe)
        if classe is None:
            return result
        parse_ok = raw_classe.strip().lower() in _VALID_CLASSES

    result["classe"] = classe
    result["label"] = LABEL2ID[classe]
    result["parse_ok"] = parse_ok

    # Additional fields (optional)
    result["trecho_evidencia"] = data.get("trecho_evidencia") or data.get("evidencia")
    result["justificativa"] = data.get("justificativa") or data.get("reasoning")
    result["q1_narrativa"] = q1_narrativa_old or q3_narrativa
    result["q2_argumentacao"] = q2_argum
    result["q3_ruido"] = q3_ruido_old or q1_ruido

    return result
