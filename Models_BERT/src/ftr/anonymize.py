"""
Anonimização de dados pessoais (PII) em textos jurídicos brasileiros — LGPD.

Mascara identificadores sensíveis substituindo-os por placeholders (``[CPF]``,
``[EMAIL]``…). Funciona em duas camadas:

1. **Regras (regex)** — sempre disponível, sem dependências. Cobre CPF, CNPJ,
   e-mail, telefone, CEP, OAB, valores monetários, URLs e números de processo
   (padrão CNJ).
2. **NER (opcional)** — se o spaCy + modelo PT estiverem instalados e
   ``mask_names`` estiver ativo, mascara nomes de pessoas (``[PESSOA]``).

A função :func:`build_audit_log` produz um relatório **agregado** (apenas
contagens por tipo de entidade) — nunca registra o valor original da PII, para
não recriar o problema que a anonimização resolve.

Determinístico: a mesma entrada sempre produz a mesma saída, garantindo
consistência entre treino/validação/teste.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

# --- Placeholders padrão ----------------------------------------------------
DEFAULT_PLACEHOLDERS: dict[str, str] = {
    "cpf": "[CPF]",
    "cnpj": "[CNPJ]",
    "email": "[EMAIL]",
    "phone": "[TELEFONE]",
    "cep": "[CEP]",
    "oab": "[OAB]",
    "money": "[VALOR]",
    "url": "[URL]",
    "processo": "[PROCESSO]",
    "person": "[PESSOA]",
}

# --- Padrões regex ----------------------------------------------------------
# A ORDEM importa: padrões mais específicos/longos primeiro para evitar que um
# padrão genérico (ex.: telefone) consuma parte de um mais específico (ex.: CNPJ,
# processo). Cada entrada é (nome_da_entidade, regex_compilada).
_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Número de processo CNJ: NNNNNNN-DD.AAAA.J.TR.OOOO
    (
        "processo",
        re.compile(r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b"),
    ),
    # CNPJ: 00.000.000/0000-00 (com ou sem pontuação)
    (
        "cnpj",
        re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"),
    ),
    # CPF: 000.000.000-00 (com ou sem pontuação)
    (
        "cpf",
        re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    ),
    # E-mail
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ),
    # URL
    (
        "url",
        re.compile(r"\bhttps?://[^\s<>\"]+|\bwww\.[^\s<>\"]+", re.IGNORECASE),
    ),
    # OAB: OAB/UF 123456  ou  OAB 123456
    (
        "oab",
        re.compile(r"\bOAB\s*[/\-]?\s*[A-Z]{0,2}\s*\.?\s*\d{2,6}\b", re.IGNORECASE),
    ),
    # CEP: 00000-000
    (
        "cep",
        re.compile(r"\b\d{5}-\d{3}\b"),
    ),
    # Valor monetário: "R$ 1.234,56" OU "1.234,56 reais" (valor numérico).
    # Nota: valores por extenso ("cem mil reais") NÃO são capturados — exigiriam
    # NLP e, além disso, um valor por extenso raramente identifica uma pessoa.
    (
        "money",
        re.compile(
            r"R\$\s?\d{1,3}(?:\.\d{3})*(?:,\d{2})?"
            r"|\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s?(?:reais|real)\b",
            re.IGNORECASE,
        ),
    ),
    # Telefone BR: (00) 00000-0000 / 0000-0000 / +55 ... (último, pois é o mais genérico)
    (
        "phone",
        re.compile(
            r"(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?9?\d{4}[-\s]?\d{4}\b"
        ),
    ),
]

_PATTERN_INDEX = {name: pat for name, pat in _PATTERNS}


@dataclass
class AnonymizationConfig:
    """Configuração de anonimização derivada do YAML."""

    enabled: bool = True
    entities: tuple[str, ...] = (
        "cpf",
        "cnpj",
        "email",
        "phone",
        "cep",
        "oab",
        "money",
        "url",
        "processo",
    )
    mask_names: bool = False
    spacy_model: str = "pt_core_news_lg"
    placeholders: dict[str, str] | None = None

    def placeholder_for(self, entity: str) -> str:
        ph = (self.placeholders or {})
        return ph.get(entity, DEFAULT_PLACEHOLDERS.get(entity, f"[{entity.upper()}]"))

    @classmethod
    def from_dict(cls, d: dict | None) -> "AnonymizationConfig":
        d = d or {}
        return cls(
            enabled=d.get("enabled", True),
            entities=tuple(d.get("entities", cls.entities)),
            mask_names=d.get("mask_names", False),
            spacy_model=d.get("spacy_model", "pt_core_news_lg"),
            placeholders=d.get("placeholders"),
        )


class Anonymizer:
    """Aplica anonimização a textos, acumulando estatísticas de auditoria."""

    def __init__(self, config: AnonymizationConfig | None = None):
        self.config = config or AnonymizationConfig()
        self.counts: Counter = Counter()
        self.docs_with_entity: Counter = Counter()
        self._nlp = None
        self._spacy_attempted = False

    # -- NER opcional --------------------------------------------------------
    def _get_nlp(self, required: bool = False):
        """Carrega o modelo spaCy de NER.

        Args:
            required: se ``True`` e o modelo não puder ser carregado, lança
                ``RuntimeError`` em vez de degradar silenciosamente. Usado quando
                ``mask_names=True`` — não queremos gerar um dataset que o usuário
                *pensa* estar sem nomes mas não está (risco de privacidade/LGPD).
        """
        if self._nlp is None and not self._spacy_attempted:
            self._spacy_attempted = True
            try:
                import spacy

                self._nlp = spacy.load(
                    self.config.spacy_model, disable=["parser", "lemmatizer"]
                )
            except Exception as e:  # noqa: BLE001
                self._nlp = None
                self._spacy_error = str(e)
        if self._nlp is None and required:
            raise RuntimeError(
                "Anonimização de NOMES (mask_names=true) foi solicitada, mas o modelo "
                f"spaCy '{self.config.spacy_model}' não pôde ser carregado "
                f"({getattr(self, '_spacy_error', 'desconhecido')}). "
                "Instale com:\n"
                "  pip install spacy\n"
                f"  python -m spacy download {self.config.spacy_model}\n"
                "Ou desative mask_names no config para anonimizar só via regras (regex)."
            )
        return self._nlp

    def _mask_names(self, text: str) -> str:
        # required=True: falha alto se o NER não estiver disponível.
        nlp = self._get_nlp(required=True)
        if nlp is None:
            return text
        doc = nlp(text)
        # Substitui da direita para a esquerda para não invalidar offsets.
        ents = [e for e in doc.ents if e.label_ in ("PER", "PERSON")]
        if not ents:
            return text
        placeholder = self.config.placeholder_for("person")
        per_doc = 0
        chars = list(text)
        for ent in sorted(ents, key=lambda e: e.start_char, reverse=True):
            chars[ent.start_char : ent.end_char] = list(placeholder)
            per_doc += 1
        if per_doc:
            self.counts["person"] += per_doc
            self.docs_with_entity["person"] += 1
        return "".join(chars)

    # -- API principal -------------------------------------------------------
    def anonymize(self, text: str) -> str:
        """Anonimiza um único texto, atualizando contadores de auditoria."""
        if not self.config.enabled or not text:
            return text

        result = text
        for entity in self.config.entities:
            pattern = _PATTERN_INDEX.get(entity)
            if pattern is None:
                continue
            placeholder = self.config.placeholder_for(entity)
            n = 0

            def _sub(_match, _ph=placeholder):
                nonlocal n
                n += 1
                return _ph

            result = pattern.sub(_sub, result)
            if n:
                self.counts[entity] += n
                self.docs_with_entity[entity] += 1

        if self.config.mask_names:
            result = self._mask_names(result)

        return result

    def ensure_ready(self) -> None:
        """Valida dependências ANTES de processar um lote (fail-fast).

        Se ``mask_names=True``, força o carregamento do spaCy aqui — assim o erro
        aparece no início, não no meio de milhares de documentos.
        """
        if self.config.enabled and self.config.mask_names:
            self._get_nlp(required=True)

    def anonymize_many(self, texts: Iterable[str]) -> list[str]:
        self.ensure_ready()
        return [self.anonymize(t) for t in texts]

    def audit_summary(self, n_docs: int | None = None) -> dict:
        """Resumo agregado de auditoria — NÃO contém PII, só contagens."""
        total = sum(self.counts.values())
        summary = {
            "enabled": self.config.enabled,
            "entities_configured": list(self.config.entities)
            + (["person"] if self.config.mask_names else []),
            "total_entities_masked": total,
            "masked_by_type": dict(sorted(self.counts.items())),
            "docs_affected_by_type": dict(sorted(self.docs_with_entity.items())),
        }
        if n_docs:
            summary["n_documents"] = n_docs
            summary["avg_entities_per_doc"] = round(total / n_docs, 3) if n_docs else 0.0
        return summary


def anonymize_series(texts: Iterable[str], config: AnonymizationConfig) -> tuple[list[str], dict]:
    """Anonimiza uma coleção de textos e retorna (textos, resumo_auditoria)."""
    texts = list(texts)
    anonymizer = Anonymizer(config)
    out = anonymizer.anonymize_many(texts)
    return out, anonymizer.audit_summary(n_docs=len(texts))
