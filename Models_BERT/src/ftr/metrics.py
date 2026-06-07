"""
Métricas de avaliação para classificação multiclasse (Fato/Tese/Ruído).

Implementa **todas** as métricas pedidas na planilha
``Métricas_Avaliação_Modelos.xlsx`` mais as métricas clássicas solicitadas:

Da planilha:
  * Accuracy
  * Precision/Recall/F1 por classe + support (classification_report)
  * Macro-F1 (principal), Weighted-F1, Micro-F1
  * Balanced Accuracy
  * MCC (Matthews)
  * Matriz de confusão (absoluta + normalizada)
  * ROC-AUC macro / OvR
  * PR-AUC / Average Precision (macro/weighted)
  * Log Loss
  * Brier Score (multiclasse, one-vs-rest médio)
  * Cohen's Kappa

Clássicas adicionais:
  * Precision/Recall macro, micro e weighted

Duas funções principais:
  * :func:`compute_all_metrics` — recebe ``y_true``, ``y_pred`` e (opcional)
    ``y_proba``; devolve um dicionário aninhado serializável em JSON.
  * :func:`compute_metrics_for_trainer` — adaptador para o ``compute_metrics`` do
    HuggingFace ``Trainer`` (recebe ``(logits, labels)``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import CLASS_NAMES, ID2LABEL


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compute_all_metrics(
    y_true,
    y_pred,
    y_proba=None,
    labels: list[int] | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Calcula o conjunto completo de métricas.

    Args:
        y_true: rótulos verdadeiros (int).
        y_pred: rótulos preditos (int).
        y_proba: probabilidades por classe ``(n, n_classes)`` (opcional, habilita
            métricas probabilísticas).
        labels: lista de índices de classe na ordem canônica.
        class_names: nomes das classes na mesma ordem de ``labels``.

    Returns:
        Dicionário aninhado com todas as métricas.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if labels is None:
        labels = sorted(ID2LABEL.keys())
    if class_names is None:
        class_names = [ID2LABEL[i] for i in labels]

    results: dict[str, Any] = {}

    # --- Globais ------------------------------------------------------------
    results["accuracy"] = float(accuracy_score(y_true, y_pred))
    results["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    results["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    results["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

    for avg in ("macro", "micro", "weighted"):
        results[f"precision_{avg}"] = float(
            precision_score(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        )
        results[f"recall_{avg}"] = float(
            recall_score(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        )
        results[f"f1_{avg}"] = float(
            f1_score(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        )

    # --- Por classe ---------------------------------------------------------
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    results["per_class"] = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }
        for name in class_names
        if name in report
    }
    results["classification_report"] = report

    # --- Matriz de confusão -------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    results["confusion_matrix"] = cm.tolist()
    results["confusion_matrix_normalized"] = cm_norm.tolist()
    results["confusion_matrix_labels"] = class_names

    # --- Probabilísticas (se houver y_proba) --------------------------------
    if y_proba is not None:
        results["probabilistic"] = _probabilistic_metrics(
            y_true, np.asarray(y_proba, dtype=float), labels
        )
    else:
        results["probabilistic"] = None

    return results


def _probabilistic_metrics(y_true: np.ndarray, y_proba: np.ndarray, labels: list[int]) -> dict:
    """ROC-AUC, PR-AUC/AP, Log Loss, Brier — exigem probabilidades."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize

    out: dict[str, Any] = {}
    n_classes = len(labels)
    y_bin = label_binarize(y_true, classes=labels)
    if n_classes == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    # ROC-AUC (OvR macro e weighted)
    try:
        out["roc_auc_macro_ovr"] = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels)
        )
        out["roc_auc_weighted_ovr"] = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted", labels=labels)
        )
    except Exception as e:
        out["roc_auc_macro_ovr"] = None
        out["roc_auc_error"] = str(e)

    # PR-AUC / Average Precision (macro e weighted)
    try:
        out["pr_auc_macro"] = float(average_precision_score(y_bin, y_proba, average="macro"))
        out["pr_auc_weighted"] = float(average_precision_score(y_bin, y_proba, average="weighted"))
    except Exception as e:
        out["pr_auc_macro"] = None
        out["pr_auc_error"] = str(e)

    # Log Loss
    try:
        out["log_loss"] = float(log_loss(y_true, y_proba, labels=labels))
    except Exception as e:
        out["log_loss"] = None
        out["log_loss_error"] = str(e)

    # Brier Score multiclasse: média do Brier OvR por classe.
    try:
        briers = []
        for j in range(n_classes):
            briers.append(brier_score_loss(y_bin[:, j], y_proba[:, j]))
        out["brier_score_macro"] = float(np.mean(briers))
        out["brier_score_per_class"] = {
            ID2LABEL[labels[j]]: float(briers[j]) for j in range(n_classes)
        }
    except Exception as e:
        out["brier_score_macro"] = None
        out["brier_score_error"] = str(e)

    # Curvas ROC/PR por classe (para plotagem) — pontos amostrados.
    out["roc_curves"] = _roc_curves(y_bin, y_proba, labels)
    out["pr_curves"] = _pr_curves(y_bin, y_proba, labels)
    return out


def _roc_curves(y_bin: np.ndarray, y_proba: np.ndarray, labels: list[int]) -> dict:
    from sklearn.metrics import auc, roc_curve

    curves = {}
    for j, lab in enumerate(labels):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, j], y_proba[:, j])
            curves[ID2LABEL[lab]] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(auc(fpr, tpr)),
            }
        except Exception:
            curves[ID2LABEL[lab]] = None
    return curves


def _pr_curves(y_bin: np.ndarray, y_proba: np.ndarray, labels: list[int]) -> dict:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    curves = {}
    for j, lab in enumerate(labels):
        try:
            prec, rec, _ = precision_recall_curve(y_bin[:, j], y_proba[:, j])
            curves[ID2LABEL[lab]] = {
                "precision": prec.tolist(),
                "recall": rec.tolist(),
                "ap": float(average_precision_score(y_bin[:, j], y_proba[:, j])),
            }
        except Exception:
            curves[ID2LABEL[lab]] = None
    return curves


def flatten_metrics(metrics: dict) -> dict[str, float]:
    """Achata as métricas escalares principais para tabelas comparativas (CSV/XLSX)."""
    flat: dict[str, float] = {}
    scalar_keys = [
        "accuracy",
        "balanced_accuracy",
        "mcc",
        "cohen_kappa",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "precision_macro",
        "precision_weighted",
        "recall_macro",
        "recall_weighted",
    ]
    for k in scalar_keys:
        if k in metrics and isinstance(metrics[k], (int, float)):
            flat[k] = metrics[k]
    prob = metrics.get("probabilistic") or {}
    for k in ("roc_auc_macro_ovr", "pr_auc_macro", "log_loss", "brier_score_macro"):
        if isinstance(prob.get(k), (int, float)):
            flat[k] = prob[k]
    # F1 por classe
    for name, vals in (metrics.get("per_class") or {}).items():
        flat[f"f1_{name}"] = vals["f1"]
    return flat


# ---------------------------------------------------------------------------
# Adaptador para o HuggingFace Trainer
# ---------------------------------------------------------------------------
def compute_metrics_for_trainer(eval_pred):
    """``compute_metrics`` para o HF Trainer — retorna um dicionário plano de escalares.

    Espera ``eval_pred = (logits, labels)``. Calcula probabilidades via softmax e
    expõe as métricas escalares mais importantes (incluindo as probabilísticas).
    O conjunto COMPLETO de métricas/curvas é computado depois, em ``evaluate.py``.
    """
    logits, labels = eval_pred
    logits = np.asarray(logits, dtype=float)
    proba = _softmax(logits)
    preds = proba.argmax(axis=1)
    full = compute_all_metrics(labels, preds, y_proba=proba)
    return flatten_metrics(full)
