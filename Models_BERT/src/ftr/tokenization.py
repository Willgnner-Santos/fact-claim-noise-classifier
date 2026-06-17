"""
Tokenisation: simple truncation (faithful) and sliding window (long-context).

Two strategies controlled by config (``tokenization.long_text_strategy``):

* ``"truncate"`` — original notebook behaviour: ``truncation=True``
  with ``max_length`` (512). Fast; discards text beyond the limit.
* ``"sliding_window"`` — splits text into overlapping windows (configurable
  stride); used at **inference** to aggregate predictions over long documents
  (mean/max/vote across windows). Training uses truncation
  (compatible with ``Trainer``); sliding window is applied at evaluation as
  a documented improvement.

The :func:`make_tokenize_function` function returns the ``preprocess_function``
used in the original ``dataset.map(..., batched=True)`` calls.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def make_tokenize_function(
    tokenizer,
    text_column: str,
    max_length: int,
    set_global_attention_cls: bool = False,
) -> Callable:
    """Creates the tokenisation function for ``Dataset.map(batched=True)``.

    ``set_global_attention_cls=True`` enables global attention on the ``[CLS]``
    token (position 0) for Longformer — required for sequence classification.
    Without it, ``[CLS]`` only receives local attention and classification
    pooling degrades significantly.
    """

    def _tok(batch):
        enc = tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
        )
        if set_global_attention_cls:
            # global_attention_mask: 1 = global attention, 0 = local attention.
            # [CLS] (pos 0) must see the full document for classification.
            enc["global_attention_mask"] = [
                [1] + [0] * (len(ids) - 1) for ids in enc["input_ids"]
            ]
        return enc

    return _tok


def sliding_window_encode(
    tokenizer,
    text: str,
    max_length: int,
    stride: int = 128,
):
    """Encodes a text into overlapping windows (for long-context inference).

    Returns a ``BatchEncoding`` with multiple windows (one row per window),
    using the fast tokenizer's ``return_overflowing_tokens``.
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
    """Aggregates logits/probas from multiple windows of the same document.

    Args:
        window_logits: array ``(n_windows, n_classes)``.
        method: ``"mean"`` (average), ``"max"`` (per-class maximum), or ``"vote"``
            (majority vote across windows).

    Returns:
        Aggregated vector ``(n_classes,)``.
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
        # returns a soft one-hot "logit" proportional to vote counts
        return counts.astype(float)
    raise ValueError(f"Unknown aggregation method: {method}")


def predict_long_text(
    model,
    tokenizer,
    text: str,
    max_length: int,
    stride: int = 128,
    aggregation: str = "mean",
):
    """Predicts the class of a long text via sliding window + aggregation.

    Used for long-context evaluation. Requires torch and a loaded model.
    """
    import torch

    enc = sliding_window_encode(tokenizer, text, max_length=max_length, stride=stride)
    inputs = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits.detach().cpu().numpy()  # (n_windows, n_classes)
    # softmax per window before aggregating probabilities
    z = logits - logits.max(axis=1, keepdims=True)
    proba = np.exp(z)
    proba /= proba.sum(axis=1, keepdims=True)
    agg = aggregate_window_logits(proba, method=aggregation)
    # "vote" returns counts — normalise to probability distribution.
    # "mean"/"max" are already probability distributions (softmax applied above).
    if aggregation == "vote":
        total = agg.sum()
        agg = agg / total if total > 0 else np.ones_like(agg) / len(agg)
    return int(np.argmax(agg)), agg
