"""
Registry de modelos e construção de modelo/tokenizador.

Centraliza o carregamento do tokenizador e do
``AutoModelForSequenceClassification`` a partir do config, aplicando o
mapeamento canônico de rótulos (id2label/label2id) e ajustes específicos por
família (encoder padrão vs. Longformer).

Importações de ``transformers`` são preguiçosas — o módulo pode ser importado
localmente sem a biblioteca instalada (apenas as funções de fato exigem-na).
"""

from __future__ import annotations

from typing import Any

from . import ID2LABEL, LABEL2ID


def build_tokenizer(config: dict):
    """Instancia o tokenizador conforme o config (AutoTokenizer)."""
    from transformers import AutoTokenizer

    model_cfg = config["model"]
    name = model_cfg.get("tokenizer_name") or model_cfg["name"]
    max_length = config["tokenization"]["max_length"]
    # use_fast=False ajuda alguns tokenizers SentencePiece (mDeBERTa) com versões antigas.
    try:
        return AutoTokenizer.from_pretrained(name, model_max_length=max_length)
    except Exception:
        return AutoTokenizer.from_pretrained(name, model_max_length=max_length, use_fast=False)


def build_model(config: dict):
    """Instancia o modelo de classificação de sequência conforme o config."""
    from transformers import AutoModelForSequenceClassification

    model_cfg = config["model"]
    name = model_cfg["name"]
    num_labels = config["labels"]["num_labels"]

    # id2label precisa de chaves int; label2id de chaves str (rótulo->id).
    id2label = {int(k): v for k, v in ID2LABEL.items()}
    label2id = {v: int(k) for k, v in ID2LABEL.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    if config["train"].get("gradient_checkpointing"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    return model


def model_summary(config: dict) -> dict[str, Any]:
    """Resumo do modelo para registro (sem baixar pesos)."""
    m = config["model"]
    return {
        "name": m["name"],
        "family": m.get("family", "encoder"),
        "is_reproduction": m.get("is_reproduction", False),
        "tokenizer_name": m.get("tokenizer_name") or m["name"],
        "num_labels": config["labels"]["num_labels"],
        "max_length": config["tokenization"]["max_length"],
    }
