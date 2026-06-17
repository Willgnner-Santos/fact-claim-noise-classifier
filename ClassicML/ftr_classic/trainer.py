"""
Training pipeline with stratified K-Fold and hold-out test evaluation.

Best practices implemented:
- StratifiedKFold (per-class balance across folds)
- Fixed global seed (Python, NumPy)
- Vectorizer fitted ONLY on the training fold (no data leakage)
- Checkpoint of the best model (per fold and overall)
- Detailed per-fold logs
- Full per-fold metrics + mean±std in consolidation
- PT-BR and EN figures saved per experiment
- env.json with library versions and hardware info
- status.json updated in real time (resilient to failures)
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import random
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .data import get_splits, get_xy
from .metrics import aggregate_cv_metrics, compute_all_metrics, flatten_metrics
from .models import build_model
from .vectorizer import build_vectorizer

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _collect_env(seed: int) -> dict:
    import sys, platform
    env: dict[str, Any] = {
        "seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {},
    }
    for pkg in ("sklearn", "numpy", "pandas", "scipy", "xgboost", "lightgbm",
                "matplotlib", "pyyaml"):
        try:
            mod_name = "sklearn" if pkg == "sklearn" else pkg
            import importlib
            mod = importlib.import_module(mod_name.replace("-", "_"))
            env["packages"][pkg] = getattr(mod, "__version__", "?")
        except Exception:
            env["packages"][pkg] = None
    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        env["cuda_available"] = False
    return env


def _make_exp_dir(experiments_root: Path, experiment_id: str) -> Path:
    run_id = uuid.uuid4().hex[:8]
    name = f"{experiment_id}__{_now()}__{run_id}"
    exp_dir = experiments_root / name
    for sub in ("figures", "checkpoints", "logs", "metrics"):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)
    return exp_dir


def _setup_logger(log_dir: Path, experiment_id: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(f"classic.{experiment_id}")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fh = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    lg.addHandler(fh); lg.addHandler(sh)
    lg.propagate = False
    return lg


def _save_checkpoint(model, vectorizer, exp_dir: Path, fold: int | None = None) -> str:
    suffix = f"_fold{fold}" if fold is not None else "_best"
    path = exp_dir / "checkpoints" / f"model{suffix}.pkl"
    with path.open("wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f, protocol=4)
    return str(path)


def run_experiment(
    config_path: str | Path,
    project_root: str | Path,
    experiments_root: str | Path | None = None,
) -> dict:
    """Run a full experiment with K-Fold + hold-out test evaluation."""
    from sklearn.model_selection import StratifiedKFold
    from . import viz

    project_root = Path(project_root).resolve()
    config_path  = Path(config_path)
    configs_dir  = config_path.parent
    config = load_config(config_path, configs_dir)

    seed = config.get("seed", 42)
    _set_seed(seed)

    eid = config.get("experiment_id", config["__config_name__"])
    exp_root = Path(experiments_root) if experiments_root else project_root / "ClassicML" / "experiments"
    exp_dir  = _make_exp_dir(exp_root, eid)
    lg = _setup_logger(exp_dir / "logs", eid)
    lg.info("=== Starting %s ===", eid)

    # Save config + env
    (exp_dir / "config_snapshot.yaml").write_text(
        __import__("yaml").dump(config, allow_unicode=True), encoding="utf-8"
    )
    env = _collect_env(seed)
    (exp_dir / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")

    status: dict[str, Any] = {
        "experiment_id": eid,
        "run_dir": exp_dir.name,
        "model_name": config["model"]["name"],
        "status": "running",
        "started_at": _now(),
        "seed": seed,
    }
    (exp_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    t0 = time.time()
    try:
        splits = get_splits(config, project_root)
        xy = get_xy(splits, config)

        # Concatenate train + val for K-Fold (uses the same dataset)
        if "train" not in xy:
            raise RuntimeError("Split 'train' not found. Check the processed data.")
        X_trainval = xy["train"][0] + xy.get("val", ([], []))[0]
        y_trainval = xy["train"][1] + xy.get("val", ([], []))[1]
        X_test  = xy["test"][0] if "test" in xy else None
        y_test  = xy["test"][1] if "test" in xy else None

        n_splits = config["cv"]["n_splits"]
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        fold_metrics, fold_info = [], []
        best_f1, best_checkpoint = -1.0, None

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval)):
            fold_start = time.time()
            lg.info("Fold %d/%d", fold_idx + 1, n_splits)

            X_tr = [X_trainval[i] for i in train_idx]
            y_tr = [y_trainval[i] for i in train_idx]
            X_val = [X_trainval[i] for i in val_idx]
            y_val = [y_trainval[i] for i in val_idx]

            # Vectorizer fitted ONLY on the training fold
            vec = build_vectorizer(config)
            X_tr_vec  = vec.fit_transform(X_tr)
            X_val_vec = vec.transform(X_val)

            model = build_model(config)
            model.fit(X_tr_vec, y_tr)

            y_pred = model.predict(X_val_vec)
            y_proba = model.predict_proba(X_val_vec) if hasattr(model, "predict_proba") else None

            fm = compute_all_metrics(y_val, y_pred, y_proba=y_proba)
            fold_metrics.append(fm)
            f1 = fm["f1_macro"]

            fold_dur = round(time.time() - fold_start, 1)
            lg.info("  Fold %d — f1_macro=%.4f  acc=%.4f  (%ds)", fold_idx+1, f1, fm["accuracy"], fold_dur)

            # Checkpoint of the best fold
            ckpt = _save_checkpoint(model, vec, exp_dir, fold=fold_idx+1)
            fold_info.append({"fold": fold_idx+1, "f1_macro": f1, "checkpoint": ckpt,
                               "duration_seconds": fold_dur})
            if f1 > best_f1:
                best_f1 = f1
                best_checkpoint = _save_checkpoint(model, vec, exp_dir, fold=None)

            # Figuras por fold
            fold_fig_dir = exp_dir / "figures" / f"fold{fold_idx+1}"
            for lang in config.get("viz", {}).get("languages", ["pt", "en"]):
                cn = [{"pt": CLASS_NAMES_PT, "en": CLASS_NAMES_EN}[lang].get(c, c)
                      for c in ["tese", "fato", "ruido"]]
                viz.plot_confusion_matrix(fm["confusion_matrix"], cn, fold_fig_dir, normalized=False, lang=lang)
                viz.plot_confusion_matrix(fm["confusion_matrix_normalized"], cn, fold_fig_dir, normalized=True, lang=lang)
                viz.plot_f1_by_class(fm["per_class"], fold_fig_dir, lang=lang)

        # Aggregate K-Fold
        cv_agg = aggregate_cv_metrics(fold_metrics)
        f1_macro_mean = cv_agg["f1_macro"]["mean"]
        f1_macro_std  = cv_agg["f1_macro"]["std"]
        lg.info("K-Fold summary: f1_macro=%.4f ± %.4f", f1_macro_mean, f1_macro_std)

        # CV boxplot
        fold_f1s = [m["f1_macro"] for m in fold_metrics]
        for lang in config.get("viz", {}).get("languages", ["pt", "en"]):
            viz.plot_cv_boxplot(fold_f1s, eid, exp_dir / "figures", lang=lang)

        # Evaluation on hold-out test set with the best model
        test_metrics = None
        if X_test and config["cv"].get("final_test_eval", True):
            lg.info("Evaluating on hold-out test set with the best model...")
            with open(best_checkpoint, "rb") as f:
                best = pickle.load(f)
            X_test_vec = best["vectorizer"].transform(X_test)
            y_pred_test = best["model"].predict(X_test_vec)
            y_proba_test = best["model"].predict_proba(X_test_vec) if hasattr(best["model"], "predict_proba") else None
            test_metrics = compute_all_metrics(y_test, y_pred_test, y_proba=y_proba_test)
            (exp_dir / "metrics" / "test_metrics.json").write_text(
                json.dumps(test_metrics, ensure_ascii=False, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                encoding="utf-8"
            )
            import pandas as pd
            pd.DataFrame([flatten_metrics(test_metrics)]).to_csv(
                exp_dir / "metrics" / "test_metrics_flat.csv", index=False
            )
            lg.info("Teste — f1_macro=%.4f  acc=%.4f  mcc=%.4f",
                    test_metrics["f1_macro"], test_metrics["accuracy"], test_metrics["mcc"])

            # Test figures
            test_fig_dir = exp_dir / "figures" / "test"
            for lang in config.get("viz", {}).get("languages", ["pt", "en"]):
                nm = CLASS_NAMES_PT if lang == "pt" else CLASS_NAMES_EN
                cn = [nm[c] for c in ["tese", "fato", "ruido"]]
                viz.plot_confusion_matrix(test_metrics["confusion_matrix"], cn, test_fig_dir, normalized=False, lang=lang)
                viz.plot_confusion_matrix(test_metrics["confusion_matrix_normalized"], cn, test_fig_dir, normalized=True, lang=lang)
                viz.plot_f1_by_class(test_metrics["per_class"], test_fig_dir, lang=lang)
                prob = test_metrics.get("probabilistic") or {}
                if prob.get("roc_curves"):
                    viz.plot_roc_curves(prob["roc_curves"], test_fig_dir, lang=lang)
                if prob.get("pr_curves"):
                    viz.plot_pr_curves(prob["pr_curves"], test_fig_dir, lang=lang)

        # Save CV metrics
        (exp_dir / "metrics" / "cv_metrics.json").write_text(
            json.dumps({"fold_metrics": [flatten_metrics(m) for m in fold_metrics],
                        "aggregated": cv_agg}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        duration = round(time.time() - t0, 1)
        status.update({
            "status": "completed",
            "finished_at": _now(),
            "duration_seconds": duration,
            "f1_macro_mean": f1_macro_mean,
            "f1_macro_std": f1_macro_std,
            "fold_info": fold_info,
            "best_checkpoint": best_checkpoint,
            "test_metrics": flatten_metrics(test_metrics) if test_metrics else None,
        })
        (exp_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        lg.info("=== Completed in %.1fs ===", duration)

    except Exception as e:
        status.update({"status": "failed", "finished_at": _now(),
                       "duration_seconds": round(time.time() - t0, 1),
                       "error": str(e), "traceback": traceback.format_exc()})
        (exp_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        lg.error("Failure: %s\n%s", e, traceback.format_exc())
        raise

    return status


# Import class-name mappings for viz
from . import CLASS_NAMES_PT, CLASS_NAMES_EN
