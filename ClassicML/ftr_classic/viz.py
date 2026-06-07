"""Gráficos PT-BR e EN para o pipeline ClassicML."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from . import CLASS_NAMES_PT, CLASS_NAMES_EN

_T = {
    "pt": {
        "confusion_matrix": "Matriz de Confusão",
        "confusion_matrix_norm": "Matriz de Confusão (Normalizada)",
        "predicted": "Predito", "actual": "Real",
        "f1_by_class": "F1-Score por Classe",
        "cv_boxplot": "F1-Macro por Fold (K-Fold)",
        "fold": "Fold", "f1_macro": "F1-Macro",
        "model": "Modelo",
        "comparison": "Comparação dos Modelos — F1-Macro (mean ± std)",
        "roc": "Curva ROC", "pr": "Curva Precisão-Recall",
        "class_": "Classe",
    },
    "en": {
        "confusion_matrix": "Confusion Matrix",
        "confusion_matrix_norm": "Confusion Matrix (Normalized)",
        "predicted": "Predicted", "actual": "Actual",
        "f1_by_class": "F1-Score by Class",
        "cv_boxplot": "F1-Macro per Fold (K-Fold)",
        "fold": "Fold", "f1_macro": "F1-Macro",
        "model": "Model",
        "comparison": "Model Comparison — F1-Macro (mean ± std)",
        "roc": "ROC Curve", "pr": "Precision-Recall Curve",
        "class_": "Class",
    },
}


def _class_names(lang: str) -> list[str]:
    nm = CLASS_NAMES_PT if lang == "pt" else CLASS_NAMES_EN
    return [nm[c] for c in ["tese", "fato", "ruido"]]


def _savefig(fig, path: Path, dpi: int, fmt: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_confusion_matrix(cm, class_names_lang, out_dir: Path, normalized=False,
                          lang="pt", dpi=150, fmt="png") -> str:
    t = _T[lang]
    title = t["confusion_matrix_norm"] if normalized else t["confusion_matrix"]
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names_lang)), yticks=range(len(class_names_lang)),
           xticklabels=class_names_lang, yticklabels=class_names_lang,
           xlabel=t["predicted"], ylabel=t["actual"], title=title)
    thresh = cm.max() / 2.0
    fmt_str = ".2f" if normalized else "d"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt_str),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    suffix = "_norm" if normalized else ""
    fname = f"confusion_matrix{suffix}_{lang}.{fmt}"
    return _savefig(fig, out_dir / fname, dpi, fmt)


def plot_f1_by_class(per_class: dict, out_dir: Path, lang="pt", dpi=150, fmt="png") -> str:
    t = _T[lang]
    nm = CLASS_NAMES_PT if lang == "pt" else CLASS_NAMES_EN
    classes = [nm[c] for c in ["tese", "fato", "ruido"] if c in per_class]
    f1s = [per_class[c]["f1"] for c in ["tese", "fato", "ruido"] if c in per_class]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(classes, f1s, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylim(0, 1.05)
    ax.set_title(t["f1_by_class"])
    ax.set_ylabel("F1-Score")
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return _savefig(fig, out_dir / f"f1_by_class_{lang}.{fmt}", dpi, fmt)


def plot_cv_boxplot(fold_f1s: list[float], experiment_id: str, out_dir: Path,
                    lang="pt", dpi=150, fmt="png") -> str:
    t = _T[lang]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(fold_f1s, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.7))
    ax.set_title(f"{t['cv_boxplot']}\n{experiment_id}")
    ax.set_ylabel(t["f1_macro"])
    ax.set_xticks([1])
    ax.set_xticklabels([experiment_id])
    mean_v = np.mean(fold_f1s)
    ax.axhline(mean_v, color="red", linestyle="--", label=f"mean={mean_v:.4f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _savefig(fig, out_dir / f"cv_boxplot_{lang}.{fmt}", dpi, fmt)


def plot_comparison(results: list[dict], out_dir: Path, lang="pt", dpi=150, fmt="png") -> str:
    """Gráfico comparativo: mean ± std do f1_macro entre todos os modelos."""
    t = _T[lang]
    results = sorted(results, key=lambda r: r.get("f1_macro_mean", 0), reverse=True)
    names   = [r["experiment_id"] for r in results]
    means   = [r.get("f1_macro_mean", 0) for r in results]
    stds    = [r.get("f1_macro_std", 0) for r in results]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="#4C72B0", alpha=0.8, ecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(max(0, min(means) - 0.05), 1.02)
    ax.set_title(t["comparison"])
    ax.set_ylabel(t["f1_macro"])
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) + 0.005,
                f"{m:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return _savefig(fig, out_dir / f"comparison_{lang}.{fmt}", dpi, fmt)


def plot_roc_curves(roc_curves: dict, out_dir: Path, lang="pt", dpi=150, fmt="png") -> str:
    t = _T[lang]
    nm = CLASS_NAMES_PT if lang == "pt" else CLASS_NAMES_EN
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for (cls_en, data), color in zip(roc_curves.items(), colors):
        if data is None:
            continue
        cls_label = nm.get(cls_en, cls_en)
        auc_val = data.get("auc", 0)
        ax.plot(data["fpr"], data["tpr"], color=color, label=f"{cls_label} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(t["roc"]); ax.legend(loc="lower right")
    fig.tight_layout()
    return _savefig(fig, out_dir / f"roc_curves_{lang}.{fmt}", dpi, fmt)


def plot_pr_curves(pr_curves: dict, out_dir: Path, lang="pt", dpi=150, fmt="png") -> str:
    t = _T[lang]
    nm = CLASS_NAMES_PT if lang == "pt" else CLASS_NAMES_EN
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for (cls_en, data), color in zip(pr_curves.items(), colors):
        if data is None:
            continue
        cls_label = nm.get(cls_en, cls_en)
        ap = data.get("ap", 0)
        ax.plot(data["recall"], data["precision"], color=color, label=f"{cls_label} (AP={ap:.3f})")
    ax.set_xlabel(t["pr"].split()[1] if lang == "en" else "Recall")
    ax.set_ylabel("Precision" if lang == "en" else "Precisão")
    ax.set_title(t["pr"]); ax.legend(loc="upper right")
    fig.tight_layout()
    return _savefig(fig, out_dir / f"pr_curves_{lang}.{fmt}", dpi, fmt)
