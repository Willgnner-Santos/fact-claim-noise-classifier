"""
Evaluation metrics — all metrics from the Métricas_Avaliação_Modelos.xlsx spreadsheet.

Mirrors the BERT pipeline (src/ftr/metrics.py) but adapted for the K-Fold
context: computes per-fold metrics and aggregates mean±std.
"""
from __future__ import annotations

import numpy as np
from typing import Any


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
    """Compute the full set of metrics."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, classification_report,
        cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef,
        precision_score, recall_score,
    )
    from . import CLASS_NAMES, ID2LABEL

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if labels is None:
        labels = sorted(ID2LABEL.keys())
    if class_names is None:
        class_names = [ID2LABEL[i] for i in labels]

    results: dict[str, Any] = {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc":               float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa":       float(cohen_kappa_score(y_true, y_pred)),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0)),
        "f1_micro":          float(f1_score(y_true, y_pred, average="micro",    labels=labels, zero_division=0)),
        "f1_weighted":       float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "precision_macro":   float(precision_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0)),
        "precision_weighted":float(precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "recall_macro":      float(recall_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0)),
        "recall_weighted":   float(recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
    }

    # Per-class metrics
    cr = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                output_dict=True, zero_division=0)
    results["per_class"] = {
        name: {
            "precision": cr[name]["precision"],
            "recall":    cr[name]["recall"],
            "f1":        cr[name]["f1-score"],
            "support":   int(cr[name]["support"]),
        }
        for name in class_names if name in cr
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    results["confusion_matrix"]            = cm.tolist()
    results["confusion_matrix_normalized"] = cm_norm.tolist()
    results["confusion_matrix_labels"]     = class_names

    # Probabilistic metrics
    if y_proba is not None:
        results["probabilistic"] = _probabilistic_metrics(y_true, np.asarray(y_proba), labels)
    else:
        results["probabilistic"] = None

    return results


def _probabilistic_metrics(y_true: np.ndarray, y_proba: np.ndarray, labels: list[int]) -> dict:
    from sklearn.metrics import (
        average_precision_score, brier_score_loss, log_loss, roc_auc_score,
        roc_curve, precision_recall_curve, auc,
    )
    from sklearn.preprocessing import label_binarize

    out: dict = {}
    n_classes = len(labels)
    y_bin = label_binarize(y_true, classes=labels)
    if n_classes == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    try:
        out["roc_auc_macro_ovr"]    = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro",    labels=labels))
        out["roc_auc_weighted_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted", labels=labels))
    except Exception as e:
        out["roc_auc_macro_ovr"] = None; out["roc_auc_error"] = str(e)

    try:
        out["pr_auc_macro"]    = float(average_precision_score(y_bin, y_proba, average="macro"))
        out["pr_auc_weighted"] = float(average_precision_score(y_bin, y_proba, average="weighted"))
    except Exception as e:
        out["pr_auc_macro"] = None; out["pr_auc_error"] = str(e)

    try:
        out["log_loss"] = float(log_loss(y_true, y_proba, labels=labels))
    except Exception as e:
        out["log_loss"] = None

    try:
        # y_proba may have fewer columns than n_classes in folds with missing classes
        n_proba_cols = y_proba.shape[1] if y_proba.ndim == 2 else 0
        if n_proba_cols != n_classes:
            raise ValueError(f"y_proba has {n_proba_cols} columns, expected {n_classes}")
        briers = []
        for j in range(n_classes):
            briers.append(brier_score_loss(y_bin[:, j], y_proba[:, j]))
        out["brier_score_macro"] = float(np.mean(briers))
    except Exception:
        out["brier_score_macro"] = None

    # Per-class ROC and PR curves
    roc_curves, pr_curves = {}, {}
    from . import ID2LABEL
    for j, lab in enumerate(labels):
        cls_name = ID2LABEL.get(lab, str(lab))
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, j], y_proba[:, j])
            roc_curves[cls_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}
        except Exception:
            roc_curves[cls_name] = None
        try:
            prec, rec, _ = precision_recall_curve(y_bin[:, j], y_proba[:, j])
            pr_curves[cls_name] = {"precision": prec.tolist(), "recall": rec.tolist(),
                                    "ap": float(average_precision_score(y_bin[:, j], y_proba[:, j]))}
        except Exception:
            pr_curves[cls_name] = None

    out["roc_curves"] = roc_curves
    out["pr_curves"]  = pr_curves
    return out


def flatten_metrics(metrics: dict) -> dict[str, float]:
    """Flatten metrics for comparative tables (CSV/XLSX)."""
    flat: dict[str, float] = {}
    for k in ("accuracy","balanced_accuracy","mcc","cohen_kappa","f1_macro","f1_micro",
               "f1_weighted","precision_macro","precision_weighted","recall_macro","recall_weighted"):
        if k in metrics and isinstance(metrics[k], (int, float)):
            flat[k] = metrics[k]
    prob = metrics.get("probabilistic") or {}
    for k in ("roc_auc_macro_ovr","pr_auc_macro","log_loss","brier_score_macro"):
        if isinstance(prob.get(k), (int, float)):
            flat[k] = prob[k]
    for name, vals in (metrics.get("per_class") or {}).items():
        flat[f"f1_{name}"] = vals["f1"]
    return flat


def aggregate_cv_metrics(fold_metrics: list[dict]) -> dict[str, Any]:
    """Aggregate metrics from K folds into mean±std."""
    flat_list = [flatten_metrics(m) for m in fold_metrics]
    keys = set(flat_list[0]) & set.intersection(*(set(f) for f in flat_list))
    agg = {}
    for k in sorted(keys):
        vals = np.array([f[k] for f in flat_list if isinstance(f.get(k), (int, float))])
        if len(vals) == 0:
            continue
        agg[k] = {
            "mean": round(float(vals.mean()), 6),
            "std":  round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 6),
            "values": vals.tolist(),
        }
    return agg
