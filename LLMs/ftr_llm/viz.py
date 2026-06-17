"""
Visualizations for LLM experiments — PT-BR and EN.

Generated plots:
  1. Confusion matrix (per model x strategy)
  2. F1-Macro by document length range (heatmap)
  3. F1-Macro comparison per model x strategy (bars)
  4. Cost vs. F1-Macro (efficiency scatter)
  5. F1 per class (grouped bar)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# PT-BR <-> EN label translations
# ---------------------------------------------------------------------------
_LABELS = {
    "pt": {
        "confusion_matrix": "Confusion Matrix",
        "predicted": "Predicted",
        "true": "True",
        "f1_by_length": "F1-Macro by Document Length Range",
        "length_bin": "Length range (words)",
        "f1_macro": "F1-Macro",
        "model": "Model",
        "strategy": "Strategy",
        "f1_comparison": "F1-Macro Comparison — Models x Strategies",
        "cost_vs_f1": "Cost vs. F1-Macro (Efficiency)",
        "cost_per_1k": "Cost per 1k docs (USD)",
        "f1_per_class": "F1 per Class",
        "class": "Class",
    },
    "en": {
        "confusion_matrix": "Confusion Matrix",
        "predicted": "Predicted",
        "true": "True",
        "f1_by_length": "F1-Macro by Document Length Range",
        "length_bin": "Length range (words)",
        "f1_macro": "F1-Macro",
        "model": "Model",
        "strategy": "Strategy",
        "f1_comparison": "F1-Macro Comparison — Models × Strategies",
        "cost_vs_f1": "Cost vs. F1-Macro (Efficiency)",
        "cost_per_1k": "Cost per 1k docs (USD)",
        "f1_per_class": "F1 per Class",
        "class": "Class",
    },
}

CLASS_NAMES = ["tese", "fato", "ruido"]
BIN_ORDER = ["A_0-512", "B_512-1k", "C_1k-3k", "D_3k-8k", "E_8k+"]


def _t(lang: str, key: str) -> str:
    return _LABELS.get(lang, _LABELS["en"]).get(key, key)


def _save(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
    title: str,
    out_path: Path,
    lang: str = "pt",
    dpi: int = 150,
) -> None:
    cm_arr = np.array(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, subtitle in zip(
        axes,
        [cm_arr, cm_norm],
        ["d", ".2f"],
        ["(absolute)", "(normalized)"],
    ):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel(_t(lang, "predicted"))
        ax.set_ylabel(_t(lang, "true"))
        ax.set_title(f"{_t(lang, 'confusion_matrix')} {subtitle}")
        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = format(data[i, j], fmt)
                ax.text(j, i, val, ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black", fontsize=9)

    fig.suptitle(title, fontsize=12)
    _save(fig, out_path, dpi)


# ---------------------------------------------------------------------------
# 2. F1-Macro by document length range
# ---------------------------------------------------------------------------
def plot_f1_by_length(
    by_bin: dict[str, dict],      # {bin_name: {f1_macro: float, n: int}}
    models_strategies: list[str], # labels
    matrix: dict[str, dict],      # {model_strategy: {bin: f1}}
    out_path: Path,
    lang: str = "pt",
    dpi: int = 150,
) -> None:
    bins = [b for b in BIN_ORDER if b in by_bin]
    if not bins or not models_strategies:
        return

    data = np.full((len(models_strategies), len(bins)), np.nan)
    for i, ms in enumerate(models_strategies):
        for j, b in enumerate(bins):
            val = matrix.get(ms, {}).get(b)
            if val is not None:
                data[i, j] = val

    fig, ax = plt.subplots(figsize=(max(8, len(bins) * 1.5), max(4, len(models_strategies) * 0.5 + 2)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label=_t(lang, "f1_macro"))
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins, rotation=30, ha="right")
    ax.set_yticks(range(len(models_strategies)))
    ax.set_yticklabels(models_strategies, fontsize=8)
    ax.set_xlabel(_t(lang, "length_bin"))
    ax.set_title(_t(lang, "f1_by_length"))
    for i in range(len(models_strategies)):
        for j in range(len(bins)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=7)
    _save(fig, out_path, dpi)


# ---------------------------------------------------------------------------
# 3. F1-Macro comparison — grouped bars
# ---------------------------------------------------------------------------
def plot_f1_comparison(
    results: list[dict],  # [{model, strategy, f1_macro}]
    out_path: Path,
    lang: str = "pt",
    dpi: int = 150,
) -> None:
    if not results:
        return
    strategies = sorted(set(r["strategy"] for r in results))
    models = sorted(set(r["model"] for r in results))
    x = np.arange(len(models))
    width = 0.8 / max(len(strategies), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.5), 6))
    for i, strat in enumerate(strategies):
        vals = []
        for m in models:
            match = [r for r in results if r["model"] == m and r["strategy"] == strat]
            vals.append(match[0]["f1_macro"] if match else 0.0)
        offset = (i - len(strategies) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=strat)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(_t(lang, "f1_macro"))
    ax.set_title(_t(lang, "f1_comparison"))
    ax.legend(title=_t(lang, "strategy"), bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _save(fig, out_path, dpi)


# ---------------------------------------------------------------------------
# 4. Cost vs F1-Macro scatter (efficiency)
# ---------------------------------------------------------------------------
def plot_cost_vs_f1(
    results: list[dict],  # [{model, strategy, f1_macro, cost_per_1k_docs_usd}]
    out_path: Path,
    lang: str = "pt",
    dpi: int = 150,
) -> None:
    if not results:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in results:
        cost = r.get("cost_per_1k_docs_usd", 0)
        f1 = r.get("f1_macro", 0)
        label = f"{r['model']}\n({r['strategy']})"
        ax.scatter(cost, f1, s=80, zorder=3)
        ax.annotate(label, (cost, f1), textcoords="offset points",
                    xytext=(5, 5), fontsize=6)
    ax.set_xlabel(_t(lang, "cost_per_1k"))
    ax.set_ylabel(_t(lang, "f1_macro"))
    ax.set_title(_t(lang, "cost_vs_f1"))
    ax.grid(True, alpha=0.3)
    _save(fig, out_path, dpi)


# ---------------------------------------------------------------------------
# 5. F1 per class — grouped bars
# ---------------------------------------------------------------------------
def plot_f1_per_class(
    results: list[dict],  # [{model, strategy, f1_tese, f1_fato, f1_ruido}]
    out_path: Path,
    lang: str = "pt",
    dpi: int = 150,
) -> None:
    if not results:
        return
    labels = [f"{r['model']}\n{r['strategy']}" for r in results]
    x = np.arange(len(labels))
    width = 0.25
    colors = {"tese": "#4878CF", "fato": "#6ACC65", "ruido": "#D65F5F"}

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 6))
    for i, cls in enumerate(CLASS_NAMES):
        vals = [r.get(f"f1_{cls}", 0) for r in results]
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width * 0.9, label=cls, color=colors[cls])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(_t(lang, "f1_macro"))
    ax.set_title(_t(lang, "f1_per_class"))
    ax.legend(title=_t(lang, "class"))
    ax.set_ylim(0, 1.05)
    _save(fig, out_path, dpi)


# ---------------------------------------------------------------------------
# Orchestrator: generates all plots for a set of results
# ---------------------------------------------------------------------------
def generate_all_plots(
    summary: list[dict],
    reports_dir: Path,
    dpi: int = 150,
) -> None:
    """Generates all plots in PT and EN."""
    for lang in ("pt", "en"):
        lang_dir = reports_dir / f"plots_{lang}"

        # Global comparison
        plot_f1_comparison(
            summary,
            lang_dir / "f1_comparison.png",
            lang=lang, dpi=dpi,
        )

        # Cost vs F1
        plot_cost_vs_f1(
            summary,
            lang_dir / "cost_vs_f1.png",
            lang=lang, dpi=dpi,
        )

        # F1 per class
        plot_f1_per_class(
            summary,
            lang_dir / "f1_per_class.png",
            lang=lang, dpi=dpi,
        )

        # F1 by length bin — heatmap
        models_strategies = [f"{r['model']}|{r['strategy']}" for r in summary]
        matrix = {
            f"{r['model']}|{r['strategy']}": r.get("by_length_bin", {})
            for r in summary
        }
        all_bins = {b: {} for r in summary for b in r.get("by_length_bin", {})}
        plot_f1_by_length(
            all_bins, models_strategies, matrix,
            lang_dir / "f1_by_length.png",
            lang=lang, dpi=dpi,
        )
