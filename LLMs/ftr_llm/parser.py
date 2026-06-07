"""
Parser de respostas JSON dos LLMs.

Robusto a:
  - JSON embrulhado em markdown (```json ... ```)
  - Espaços/newlines extras
  - Campos com casing variado (Fato, FATO, fato)
  - Respostas parciais (só classe, sem evidência)

Retorna sempre um dict normalizado com pelo menos {"classe": str, "parse_ok": bool}.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ftr_llm import LABEL2ID

_VALID_CLASSES = set(LABEL2ID.keys())  # {"tese", "fato", "ruido"}

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_INLINE_JSON_RE = re.compile(r"\{[\s\S]*\}")

# Marcadores que precedem a conclusão em respostas CoT — a classe após esses
# padrões é a decisão final, não texto de raciocínio intermediário.
_CONCLUSION_RE = re.compile(
    r"(?:classificaç[aã]o(?:\s+final)?|portanto|logo|conclus[aã]o|"
    r"classe(?:\s+final)?|resposta|therefore|classification)\s*[:\-–]\s*",
    re.IGNORECASE,
)


def _extract_json_str(text: str) -> str | None:
    """Tenta extrair string JSON de uma resposta possivelmente embrulhada."""
    # 1. Fence markdown
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
    # "misto"/"mista" — retorna None para acionar _keyword_fallback no raw_text completo.
    # O raciocínio do modelo pode mencionar a classe predominante após o JSON.
    # Se não houver nada, fica como falha real (pred=None, parse_ok=False) —
    # mais honesto do que um desempate arbitrário que contamina as métricas.
    return None


def _keyword_fallback(text: str) -> str | None:
    """
    Extrai a classe como último recurso quando não há JSON.

    Estratégia em três passos:
    1. Procura a classe após marcadores de conclusão ("portanto:", "classificação: ")
       — em CoT a decisão final aparece depois desses marcadores, não no meio do raciocínio.
    2. Se não encontrar marcador, usa a ÚLTIMA ocorrência de uma classe no texto
       — em CoT o modelo costuma concluir no final.
    3. Só aceita a primeira ocorrência (comportamento antigo) se for o único match.

    Nunca retorna a primeira keyword encontrada sem antes verificar as outras opções,
    evitando o falso positivo clássico: "relata um fato, mas conclui como tese" → tese.
    """
    lower = text.lower()

    # Passo 1 — após marcador de conclusão
    for m in _CONCLUSION_RE.finditer(lower):
        snippet = lower[m.end():m.end() + 20]
        for cls in _VALID_CLASSES:
            if cls in snippet:
                return cls

    # Passo 2 — última ocorrência no texto
    last_pos = -1
    last_cls = None
    for cls in _VALID_CLASSES:
        pos = lower.rfind(cls)
        if pos > last_pos:
            last_pos = pos
            last_cls = cls

    # Só usa a última ocorrência se ela for inequívoca:
    # garante que a palavra está isolada (não é substring de outra palavra).
    if last_cls and last_pos >= 0:
        before = lower[last_pos - 1] if last_pos > 0 else " "
        after_pos = last_pos + len(last_cls)
        after = lower[after_pos] if after_pos < len(lower) else " "
        if not before.isalpha() and not after.isalpha():
            return last_cls

    return None


def parse_response(text: str, strategy: str = "zero_shot") -> dict[str, Any]:
    """
    Parseia a resposta do LLM e retorna um dict com:
      - classe: str | None
      - label: int | None
      - parse_ok: bool
      - trecho_evidencia: str | None (se evidence_constrained)
      - justificativa: str | None (se evidence_constrained)
      - q1/q2/q3: str | None (se decomposed)
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
        # Tenta corrigir aspas simples
        try:
            data = json.loads(json_str.replace("'", '"'))
        except Exception:
            return result

    if not isinstance(data, dict):
        # Se o modelo retornou uma lista de objetos, tenta pegar o primeiro
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            data = data[0]
        else:
            return result

    # Campos intermediários do decomposed
    q1_ruido      = str(data.get("q1_ruido", "")).strip().lower()
    q2_argum      = str(data.get("q2_argumentacao", "")).strip().lower()
    q3_narrativa  = str(data.get("q3_narrativa", "")).strip().lower()
    # compat com formato antigo
    q1_narrativa_old = str(data.get("q1_narrativa", "")).strip().lower()
    q3_ruido_old     = str(data.get("q3_ruido", "")).strip().lower()

    is_decomposed = q1_ruido or q2_argum or q3_narrativa or q1_narrativa_old or q3_ruido_old

    if is_decomposed:
        # Aplica a regra de prioridade NO CÓDIGO — não confia no campo "classe" do modelo,
        # que demonstrou ignorar a lógica de decisão mesmo quando explicitada no prompt.
        # Prioridade: ruido > tese > fato
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
            classe = "ruido"  # fallback conservador
        parse_ok = True
    else:
        # Estratégias não-decomposed: extrai classe do campo "classe"
        raw_classe = str(data.get("classe") or data.get("class") or data.get("classificacao") or "")
        classe = _normalize_class(raw_classe)
        if classe is None:
            return result
        parse_ok = raw_classe.strip().lower() in _VALID_CLASSES

    result["classe"] = classe
    result["label"] = LABEL2ID[classe]
    result["parse_ok"] = parse_ok

    # Campos adicionais (não-obrigatórios)
    result["trecho_evidencia"] = data.get("trecho_evidencia") or data.get("evidencia")
    result["justificativa"] = data.get("justificativa") or data.get("reasoning")
    result["q1_narrativa"] = q1_narrativa_old or q3_narrativa
    result["q2_argumentacao"] = q2_argum
    result["q3_ruido"] = q3_ruido_old or q1_ruido

    return result
