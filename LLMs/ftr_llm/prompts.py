"""
Prompts for legal classification Fato/Tese/Ruído.

Implemented strategies:
  - zero_shot         : direct instruction without examples
  - few_shot          : instruction + 3 examples per class (fixed)
  - evidence_constrained : few-shot with structured output (class + evidence + justification)
  - decomposed        : intermediate legal questions -> final class

Expected output always in JSON for deterministic parsing.
"""

from __future__ import annotations

from typing import Literal

StrategyName = Literal["zero_shot", "few_shot", "evidence_constrained", "decomposed"]

# ---------------------------------------------------------------------------
# Canonical examples (fixed few-shot — 1 per class)
# ---------------------------------------------------------------------------
_FEW_SHOT_EXAMPLES = [
    {
        "label": "fato",
        "text": (
            "o autor possui com o devedor um contrato no valor total de r$ 500,00. "
            "a dívida foi firmada através de um documento particular assinado por ele e "
            "por 2 testemunhas, datado em 30/10/2019. o devedor pagou somente a primeira "
            "parcela; a partir de janeiro de 2020 não adimpliu com as demais, restando "
            "o valor de r$ 450,00 em aberto."
        ),
    },
    {
        "label": "tese",
        "text": (
            "o código de processo civil prevê expressamente a figura do documento particular "
            "assinado pelo devedor e por 2 testemunhas como título executivo extrajudicial, "
            "nos termos do art. 784, inciso iii. assim, presentes os requisitos legais, "
            "requer-se a execução do débito com penhora de bens suficientes à satisfação "
            "da dívida, acrescida de juros e correção monetária."
        ),
    },
    {
        "label": "ruido",
        "text": (
            "excelentíssimo(a) senhor(a) doutor(a) juiz(a) de direito da vara cível "
            "da comarca de goiânia, estado de goiás. [NOME], [CPF], residente e domiciliado "
            "na rua das flores, nº 100, centro, goiânia/go, cep [CEP], por meio de seu "
            "advogado infra-assinado, [OAB], vem, respeitosamente, à presença de v. exa."
        ),
    },
]

# ---------------------------------------------------------------------------
# Shared system prompt
# ---------------------------------------------------------------------------
_SYSTEM = (
    "Você é um especialista em direito processual civil brasileiro. "
    "Sua tarefa é classificar trechos de peças jurídicas em exatamente uma de três categorias:\n"
    "- FATO: narrativa de eventos, relato do que aconteceu, quem fez o quê e quando.\n"
    "- TESE: argumentação jurídica, citação de lei ou jurisprudência, fundamento legal, pedido fundamentado.\n"
    "- RUIDO: cabeçalho, qualificação das partes, dados processuais, fórmulas protocolares sem conteúdo substantivo.\n\n"
    "Se o trecho misturar elementos, escolha a classe PREDOMINANTE — aquela que ocupa mais espaço ou é mais relevante. "
    "Responda SEMPRE em JSON válido, sem texto fora do JSON."
)


def _examples_block() -> str:
    lines = []
    for ex in _FEW_SHOT_EXAMPLES:
        lines.append(f'Trecho: "{ex["text"][:300]}"')
        lines.append(f'Resposta: {{"classe": "{ex["label"]}"}}\n')
    return "\n".join(lines)


def _examples_block_evidence() -> str:
    lines = []
    justificativas = {
        "fato": "descreve eventos concretos com datas e valores — narrativa factual",
        "tese": "cita artigo de lei e formula pedido de execução — argumentação jurídica",
        "ruido": "endereçamento protocolar, qualificação da parte — sem conteúdo substantivo",
    }
    for ex in _FEW_SHOT_EXAMPLES:
        trecho_ev = ex["text"][:120].replace('"', "'")
        lines.append(f'Trecho: "{ex["text"][:300]}"')
        lines.append(
            f'Resposta: {{"classe": "{ex["label"]}", '
            f'"trecho_evidencia": "{trecho_ev}...", '
            f'"justificativa": "{justificativas[ex["label"]]}"}}\n'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_zero_shot(text: str) -> tuple[str, str]:
    """Returns (system, user)."""
    user = (
        f'Classifique o trecho abaixo em FATO, TESE ou RUIDO.\n\n'
        f'Trecho:\n"""\n{text}\n"""\n\n'
        f'Responda SOMENTE com: {{"classe": "fato"|"tese"|"ruido"}}'
    )
    return _SYSTEM, user


def build_few_shot(text: str) -> tuple[str, str]:
    examples = _examples_block()
    user = (
        f"Exemplos:\n\n{examples}\n"
        f'Agora classifique:\nTrecho: "{text}"\n\n'
        f'Responda SOMENTE com: {{"classe": "fato"|"tese"|"ruido"}}'
    )
    return _SYSTEM, user


def build_evidence_constrained(text: str) -> tuple[str, str]:
    examples = _examples_block_evidence()
    user = (
        f"Exemplos:\n\n{examples}\n"
        f'Classifique o trecho abaixo. Identifique o trecho de evidência e a justificativa.\n\n'
        f'Trecho:\n"""\n{text}\n"""\n\n'
        f'Responda SOMENTE com JSON no formato:\n'
        f'{{"classe": "fato"|"tese"|"ruido", "trecho_evidencia": "...", "justificativa": "..."}}'
    )
    return _SYSTEM, user


def build_decomposed(text: str) -> tuple[str, str]:
    """
    Decomposed chain prompt: 3 intermediate legal questions -> final class.
    Explicit priority rule to guide the model in the final decision.
    """
    user = (
        f'Analise o trecho jurídico abaixo respondendo as três perguntas:\n\n'
        f'1. O trecho é cabeçalho, qualificação de parte ou fórmula protocolar SEM conteúdo substantivo? (sim/não)\n'
        f'2. O trecho CITA lei, jurisprudência ou FORMULA pedido juridicamente fundamentado? (sim/não)\n'
        f'3. O trecho NARRA eventos concretos (o que aconteceu, quem fez o quê, datas, valores)? (sim/não)\n\n'
        f'Regra de decisão — aplique nesta ordem de prioridade:\n'
        f'  - Se q1=sim → classe = "ruido" (independente das outras)\n'
        f'  - Se q1=não e q2=sim → classe = "tese"\n'
        f'  - Se q1=não e q2=não e q3=sim → classe = "fato"\n'
        f'  - Se nenhuma → classe = "ruido"\n\n'
        f'Trecho:\n"""\n{text}\n"""\n\n'
        f'Responda SOMENTE com JSON no formato:\n'
        f'{{"q1_ruido": "sim"|"não", "q2_argumentacao": "sim"|"não", '
        f'"q3_narrativa": "sim"|"não", "classe": "fato"|"tese"|"ruido"}}'
    )
    return _SYSTEM, user


def build_prompt(strategy: StrategyName, text: str) -> tuple[str, str]:
    """Central dispatcher."""
    builders = {
        "zero_shot": build_zero_shot,
        "few_shot": build_few_shot,
        "evidence_constrained": build_evidence_constrained,
        "decomposed": build_decomposed,
    }
    fn = builders.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown strategy: {strategy!r}. Options: {list(builders)}")
    return fn(text)
