"""
Tokenização: truncação simples (fiel) e janela deslizante (long-context).

Duas estratégias controladas pelo config (``tokenization.long_text_strategy``):

* ``"truncate"`` — comportamento dos notebooks originais: ``truncation=True``
  com ``max_length`` (512). Rápido; perde o que excede o limite.
* ``"sliding_window"`` — divide o texto em janelas sobrepostas (stride
  configurável); usado na **inferência** para agregar predições de documentos
  longos (média/máximo/voto sobre as janelas). Para o treino mantemos truncação
  (compatível com o ``Trainer``); a janela deslizante entra na avaliação como
  melhoria documentada.

A função :func:`make_tokenize_function` devolve o ``preprocess_function`` usado
no ``dataset.map(..., batched=True)`` dos originais.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def make_tokenize_function(tokenizer, text_column: str, max_length: int) -> Callable:
    """Cria a função de tokenização para ``Dataset.map(batched=True)``."""

    def _tok(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
        )

    return _tok


def sliding_window_encode(
    tokenizer,
    text: str,
    max_length: int,
    stride: int = 128,
):
    """Codifica um texto em janelas sobrepostas (para inferência long-context).

    Retorna um ``BatchEncoding`` com múltiplas janelas (uma linha por janela),
    usando o ``return_overflowing_tokens`` do tokenizer fast.
    """
    return tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_tensors="pt",
        padding="max_length",
    )


def aggregate_window_logits(window_logits: np.ndarray, method: str = "mean") -> np.ndarray:
    """Agrega logits/probas de várias janelas de um mesmo documento.

    Args:
        window_logits: array ``(n_windows, n_classes)``.
        method: ``"mean"`` (média), ``"max"`` (máximo por classe) ou ``"vote"``
            (voto majoritário das janelas).

    Returns:
        Vetor ``(n_classes,)`` agregado.
    """
    window_logits = np.asarray(window_logits, dtype=float)
    if window_logits.ndim == 1:
        return window_logits
    if method == "mean":
        return window_logits.mean(axis=0)
    if method == "max":
        return window_logits.max(axis=0)
    if method == "vote":
        preds = window_logits.argmax(axis=1)
        n_classes = window_logits.shape[1]
        counts = np.bincount(preds, minlength=n_classes)
        # devolve um "logit" one-hot suave proporcional aos votos
        return counts.astype(float)
    raise ValueError(f"Método de agregação desconhecido: {method}")


def predict_long_text(
    model,
    tokenizer,
    text: str,
    max_length: int,
    stride: int = 128,
    aggregation: str = "mean",
):
    """Prediz a classe de um texto longo via janela deslizante + agregação.

    Usado na avaliação long-context. Requer torch + modelo carregado.
    """
    import torch

    enc = sliding_window_encode(tokenizer, text, max_length=max_length, stride=stride)
    inputs = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits.detach().cpu().numpy()  # (n_windows, n_classes)
    # softmax por janela antes de agregar probabilidades
    z = logits - logits.max(axis=1, keepdims=True)
    proba = np.exp(z)
    proba /= proba.sum(axis=1, keepdims=True)
    agg = aggregate_window_logits(proba, method=aggregation)
    # "vote" retorna contagens — normaliza para distribuição de probabilidade.
    # "mean"/"max" já são distribuições de probabilidade (softmax aplicado acima).
    if aggregation == "vote":
        total = agg.sum()
        agg = agg / total if total > 0 else np.ones_like(agg) / len(agg)
    return int(np.argmax(agg)), agg
