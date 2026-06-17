# -*- coding: utf-8 -*-
"""Export per-instance predictions of LightGBM on the canonical test set, for
the McNemar paired test (BERT vs Classic ML).

Does NOT retrain: loads the already-saved model (checkpoints/model_best.pkl) that
produced the paper result (F1-macro ~0.973), predicts on the same test.jsonl used
by the BERTs, and saves test_predictions.npz in the SAME format and ORDER as the
BERT npz files.

Usage (from E:\\DIACDE\\Fato_Tese_Ruido\\ClassicML):
    venv\\Scripts\\python.exe scripts\\export_lightgbm_preds.py
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent          # .../ClassicML
EXP = ROOT / "experiments" / "lightgbm__20260603_004319__773fc4b2"
PKL = EXP / "checkpoints" / "model_best.pkl"
TEST_JSONL = ROOT.parent / "Models_BERT" / "Data" / "processed" / "test.jsonl"
OUT = EXP / "metrics" / "test_predictions.npz"

LABEL2ID = {"tese": 0, "fato": 1, "ruido": 2}

def f1_macro(yt, yp):
    f1s = []
    for c in range(3):
        tp = np.sum((yt == c) & (yp == c)); fp = np.sum((yt != c) & (yp == c)); fn = np.sum((yt == c) & (yp != c))
        pr = tp/(tp+fp) if tp+fp else 0.0; re = tp/(tp+fn) if tp+fn else 0.0
        f1s.append(2*pr*re/(pr+re) if pr+re else 0.0)
    return float(np.mean(f1s))

# 1. Load test in the SAME order as the jsonl (identical to the BERT npz files)
rows = [json.loads(l) for l in open(TEST_JSONL, encoding="utf-8")]
texts = [r["text"] for r in rows]
y_true = np.array([LABEL2ID[r["label_text"]] for r in rows], dtype=np.int64)
print(f"test: {len(texts)} trechos | dist y_true={np.bincount(y_true).tolist()}")

# 2. Load saved model + vectorizer
if not PKL.exists():
    sys.exit(f"ERROR: could not find {PKL}")
with open(PKL, "rb") as f:
    obj = pickle.load(f)
model, vec = obj["model"], obj["vectorizer"]

# 3. Predict
X = vec.transform([str(t) for t in texts])
y_pred = model.predict(X).astype(np.int64)
proba = model.predict_proba(X).astype(np.float64) if hasattr(model, "predict_proba") else None

# 4. Validate that this is the paper model (F1 ~0.973)
f1 = f1_macro(y_true, y_pred)
print(f"F1-macro recomputed = {f1:.4f}  (expected ~0.9732)")
if abs(f1 - 0.9732) > 0.01:
    print("WARNING: F1 diverged from the paper; verify the model/splits before use.")

# 5. Save in the BERT npz format
OUT.parent.mkdir(parents=True, exist_ok=True)
if proba is not None:
    np.savez(OUT, y_true=y_true, y_pred=y_pred, proba=proba)
else:
    np.savez(OUT, y_true=y_true, y_pred=y_pred)
print(f"OK -> {OUT}")
