"""
Carrega, anonimiza, pré-processa e faz split do dataset para os experimentos LLM.

Reutiliza:
  - src/ftr/anonymize.py  — Anonymizer + AnonymizationConfig
  - src/ftr/preprocess.py — build_preprocessor + PreprocessConfig

Salva um dataset pré-processado em JSON para evitar reprocessamento.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Adiciona src/ ao path para reutilizar src/ftr sem instalar como pacote
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ftr.anonymize import Anonymizer, AnonymizationConfig
from ftr.preprocess import PreprocessConfig, build_preprocessor
from ftr_llm import LABEL2ID


LENGTH_BINS = [
    (0, 512,   "A_0-512"),
    (512, 1000, "B_512-1k"),
    (1000, 3000, "C_1k-3k"),
    (3000, 8000, "D_3k-8k"),
    (8000, float("inf"), "E_8k+"),
]


def _assign_bin(n_tokens: int) -> str:
    for lo, hi, label in LENGTH_BINS:
        if lo <= n_tokens < hi:
            return label
    return "E_8k+"


def load_raw(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [d for d in data if d.get("text")]


def preprocess_and_anonymize(
    records: list[dict],
    anon_cfg: AnonymizationConfig,
    prep_cfg: PreprocessConfig,
) -> tuple[list[dict], dict]:
    """Aplica pré-processo + anonimização. Retorna (records, audit_summary)."""
    anonymizer = Anonymizer(anon_cfg)
    anonymizer.ensure_ready()
    preprocessor = build_preprocessor(prep_cfg)

    out = []
    for r in records:
        text = preprocessor(r["text"])
        text = anonymizer.anonymize(text)
        words = len(text.split())
        out.append({
            "text": text,
            "label": int(r["label"]),
            "label_text": r["label_text"],
            "n_words": words,
            "length_bin": _assign_bin(words),
        })

    audit = anonymizer.audit_summary(n_docs=len(records))
    return out, audit


def make_splits(
    records: list[dict],
    test_size: float = 0.10,
    val_size: float = 0.10,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Cria splits train/val/test estratificados por label."""
    labels = [r["label"] for r in records]
    idx = list(range(len(records)))

    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, stratify=labels, random_state=seed
    )
    labels_trainval = [labels[i] for i in idx_trainval]
    val_frac = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac, stratify=labels_trainval, random_state=seed
    )

    return {
        "train": [records[i] for i in idx_train],
        "val":   [records[i] for i in idx_val],
        "test":  [records[i] for i in idx_test],
    }


def save_splits(splits: dict[str, list[dict]], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in splits.items():
        p = out_dir / f"{name}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)


def load_splits(split_dir: str | Path) -> dict[str, list[dict]]:
    split_dir = Path(split_dir)
    out = {}
    for name in ("train", "val", "test"):
        p = split_dir / f"{name}.json"
        with open(p, encoding="utf-8") as f:
            out[name] = json.load(f)
    return out


def dataset_stats(records: list[dict]) -> dict[str, Any]:
    from collections import Counter
    import statistics
    words = [r["n_words"] for r in records]
    return {
        "total": len(records),
        "label_dist": dict(Counter(r["label_text"] for r in records)),
        "length_bin_dist": dict(Counter(r["length_bin"] for r in records)),
        "words_mean": round(statistics.mean(words), 1),
        "words_median": statistics.median(words),
        "words_min": min(words),
        "words_max": max(words),
    }
