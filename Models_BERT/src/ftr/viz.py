"""
Visualizações padronizadas (PT-BR e EN), salvas em arquivos.

Cobre EDA e avaliação de modelos:

EDA:
  * distribuição de classes
  * distribuição do tamanho dos textos (palavras / tokens)

Avaliação:
  * matriz de confusão (absoluta e normalizada)
  * curvas de treino/validação (loss, accuracy, F1 por época)
  * métricas por classe (barras)
  * curvas ROC e Precision-Recall
  * comparação entre modelos (barras de métricas finais)

Cada função aceita ``lang in {"pt", "en"}`` e devolve o caminho do arquivo salvo.
Usa backend não-interativo (``Agg``) para rodar no Colab/headless.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import CLASS_NAMES_EN, CLASS_NAMES_PT  # noqa: E402

# Textos de UI por idioma.
_T = {
    "pt": {
        "class_dist_title": "Distribuição das classes",
        "count": "Quantidade",
        "class": "Classe",
        "len_title": "Distribuição do tamanho dos textos",
        "len_x": "Tamanho (nº de {unit})",
        "freq": "Frequência",
        "cm_abs": "Matriz de confusão (absoluta)",
        "cm_norm": "Matriz de confusão (normalizada)",
        "true": "Classe verdadeira",
        "pred": "Classe predita",
        "epoch": "Época",
        "loss": "Perda (loss)",
        "train": "Treino",
        "val": "Validação",
        "per_class_title": "Métricas por classe",
        "score": "Valor",
        "roc_title": "Curvas ROC (OvR)",
        "pr_title": "Curvas Precisão-Revocação",
        "fpr": "Taxa de falsos positivos",
        "tpr": "Taxa de verdadeiros positivos",
        "recall": "Revocação",
        "precision": "Precisão",
        "cmp_title": "Comparação entre modelos",
        "model": "Modelo",
        "tokens": "tokens",
        "words": "palavras",
        "limit512": "limite 512",
    },
    "en": {
        "class_dist_title": "Class distribution",
        "count": "Count",
        "class": "Class",
        "len_title": "Text length distribution",
        "len_x": "Length (# {unit})",
        "freq": "Frequency",
        "cm_abs": "Confusion matrix (absolute)",
        "cm_norm": "Confusion matrix (normalized)",
        "true": "True label",
        "pred": "Predicted label",
        "epoch": "Epoch",
        "loss": "Loss",
        "train": "Train",
        "val": "Validation",
        "per_class_title": "Per-class metrics",
        "score": "Score",
        "roc_title": "ROC curves (OvR)",
        "pr_title": "Precision-Recall curves",
        "fpr": "False positive rate",
        "tpr": "True positive rate",
        "recall": "Recall",
        "precision": "Precision",
        "cmp_title": "Model comparison",
        "model": "Model",
        "tokens": "tokens",
        "words": "words",
        "limit512": "512 limit",
    },
}


def _class_label(name: str, lang: str) -> str:
    return (CLASS_NAMES_PT if lang == "pt" else CLASS_NAMES_EN).get(name, name)


def _save(fig, out_dir, stem: str, lang: str, dpi: int, fmt: str) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_{lang}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def plot_class_distribution(counts: dict[str, int], out_dir, lang="pt", dpi=150, fmt="png") -> str:
    t = _T[lang]
    names = list(counts.keys())
    labels = [_class_label(n, lang) for n in names]
    values = [counts[n] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_title(t["class_dist_title"])
    ax.set_xlabel(t["class"])
    ax.set_ylabel(t["count"])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    return _save(fig, out_dir, "class_distribution", lang, dpi, fmt)


def plot_length_distribution(
    lengths, out_dir, unit="words", lang="pt", dpi=150, fmt="png", vline=None
) -> str:
    t = _T[lang]
    lengths = np.asarray(lengths)
    # clip de cauda para visualização (p99) — anota o real no título.
    p99 = float(np.percentile(lengths, 99))
    clipped = np.clip(lengths, 0, p99)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(clipped, bins=60, color="#4C72B0", alpha=0.85)
    unit_label = t["tokens"] if unit == "tokens" else t["words"]
    ax.set_title(f"{t['len_title']} ({unit_label})")
    ax.set_xlabel(t["len_x"].format(unit=unit_label))
    ax.set_ylabel(t["freq"])
    if vline:
        ax.axvline(vline, color="red", linestyle="--", linewidth=1, label=t["limit512"])
        ax.legend()
    return _save(fig, out_dir, f"length_distribution_{unit}", lang, dpi, fmt)


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm, class_names, out_dir, normalized=False, lang="pt", dpi=150, fmt="png"
) -> str:
    t = _T[lang]
    cm = np.asarray(cm, dtype=float)
    labels = [_class_label(n, lang) for n in class_names]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(t["cm_norm"] if normalized else t["cm_abs"])
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel(t["pred"])
    ax.set_ylabel(t["true"])
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalized else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    stem = "confusion_matrix_norm" if normalized else "confusion_matrix_abs"
    return _save(fig, out_dir, stem, lang, dpi, fmt)


def plot_training_curves(history: list[dict], out_dir, lang="pt", dpi=150, fmt="png") -> list[str]:
    """Plota loss/accuracy/F1 por época a partir do log_history do Trainer.

    ``history`` é uma lista de dicts (cada um pode conter ``epoch``, ``loss``,
    ``eval_loss``, ``eval_accuracy``, ``eval_f1_macro`` ...). Retorna os caminhos
    gerados (um gráfico por métrica disponível).
    """
    t = _T[lang]
    paths = []

    def _series(key):
        xs, ys = [], []
        for h in history:
            if key in h and "epoch" in h and h[key] is not None:
                xs.append(h["epoch"])
                ys.append(h[key])
        return xs, ys

    # Loss treino vs val
    tr_x, tr_y = _series("loss")
    va_x, va_y = _series("eval_loss")
    if tr_y or va_y:
        fig, ax = plt.subplots(figsize=(7, 4))
        if tr_y:
            ax.plot(tr_x, tr_y, marker="o", label=t["train"])
        if va_y:
            ax.plot(va_x, va_y, marker="s", label=t["val"])
        ax.set_title(t["loss"])
        ax.set_xlabel(t["epoch"])
        ax.set_ylabel(t["loss"])
        ax.legend()
        paths.append(_save(fig, out_dir, "curve_loss", lang, dpi, fmt))

    for key, stem in (("eval_accuracy", "curve_accuracy"), ("eval_f1_macro", "curve_f1_macro")):
        x, y = _series(key)
        if y:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(x, y, marker="o", color="#55A868", label=t["val"])
            title = "Accuracy" if "accuracy" in key else "F1-macro"
            ax.set_title(f"{title} ({t['val']})")
            ax.set_xlabel(t["epoch"])
            ax.set_ylabel(t["score"])
            ax.legend()
            paths.append(_save(fig, out_dir, stem, lang, dpi, fmt))
    return paths


def plot_per_class_metrics(per_class: dict, out_dir, lang="pt", dpi=150, fmt="png") -> str:
    t = _T[lang]
    names = list(per_class.keys())
    labels = [_class_label(n, lang) for n in names]
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, m in enumerate(metrics):
        vals = [per_class[n][m] for n in names]
        ax.bar(x + (i - 1) * width, vals, width, label=m.capitalize(), color=colors[i])
    ax.set_title(t["per_class_title"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(t["score"])
    ax.set_ylim(0, 1.05)
    ax.legend()
    return _save(fig, out_dir, "per_class_metrics", lang, dpi, fmt)


def plot_roc_curves(roc_curves: dict, out_dir, lang="pt", dpi=150, fmt="png") -> str | None:
    if not roc_curves:
        return None
    t = _T[lang]
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, c in roc_curves.items():
        if not c:
            continue
        ax.plot(c["fpr"], c["tpr"], label=f"{_class_label(name, lang)} (AUC={c['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title(t["roc_title"])
    ax.set_xlabel(t["fpr"])
    ax.set_ylabel(t["tpr"])
    ax.legend(loc="lower right")
    return _save(fig, out_dir, "roc_curves", lang, dpi, fmt)


def plot_pr_curves(pr_curves: dict, out_dir, lang="pt", dpi=150, fmt="png") -> str | None:
    if not pr_curves:
        return None
    t = _T[lang]
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, c in pr_curves.items():
        if not c:
            continue
        ax.plot(c["recall"], c["precision"], label=f"{_class_label(name, lang)} (AP={c['ap']:.3f})")
    ax.set_title(t["pr_title"])
    ax.set_xlabel(t["recall"])
    ax.set_ylabel(t["precision"])
    ax.legend(loc="lower left")
    return _save(fig, out_dir, "pr_curves", lang, dpi, fmt)


def plot_model_comparison(
    rows: list[dict], metric: str, out_dir, lang="pt", dpi=150, fmt="png"
) -> str | None:
    """Barras comparando um métrica entre modelos. ``rows`` = saída da tabela."""
    rows = [r for r in rows if r.get(metric) is not None]
    if not rows:
        return None
    t = _T[lang]
    rows = sorted(rows, key=lambda r: r[metric], reverse=True)
    names = [r.get("experiment_id", "?") for r in rows]
    vals = [r[metric] for r in rows]
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.1), 4.5))
    bars = ax.bar(names, vals, color="#4C72B0")
    ax.set_title(f"{t['cmp_title']} — {metric}")
    ax.set_xlabel(t["model"])
    ax.set_ylabel(metric)
    ax.set_ylim(0, max(1.0, max(vals) * 1.05))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    return _save(fig, out_dir, f"comparison_{metric}", lang, dpi, fmt)
