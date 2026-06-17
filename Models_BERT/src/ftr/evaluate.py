"""
Full evaluation of an experiment on the test set.

Generates predictions (standard truncation or sliding window for long-context),
computes **all** metrics (:mod:`ftr.metrics`), saves to JSON + CSV, and produces
evaluation charts (:mod:`ftr.viz`) in PT-BR and EN.

Main function: :func:`evaluate_split`, called at the end of training. Also provides
:func:`evaluate_from_dir` to re-evaluate an already-trained experiment from the
saved ``best_model``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import CLASS_NAMES
from .metrics import compute_all_metrics, flatten_metrics


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _predict_truncation(trainer, tokenizer, df, config):
    """Predictions via Trainer (truncation tokenisation) — fast/faithful path.

    Important: we do NOT pass the ``labels`` column to the prediction dataset.
    In transformers 4.48, when ``labels`` are present the model computes the
    loss and ``output.predictions`` returns a tuple ``(logits,)`` instead of
    ``logits`` directly. ``np.asarray((logits,))`` becomes shape ``(1, n, 3)``
    and softmax / argmax compute on the wrong axis → model always predicts class 0.
    We extract ``y_true`` directly from the DataFrame (order preserved by the
    parquet index) and pass only text to the inference dataset.
    """
    from datasets import Dataset

    from .tokenization import make_tokenize_function

    text_column = config["data"]["text_column"]
    label_column = config["data"]["label_column"]
    is_longformer = config["model"].get("family", "encoder") == "longformer"
    tok_fn = make_tokenize_function(
        tokenizer, text_column, config["tokenization"]["max_length"],
        set_global_attention_cls=is_longformer,
    )

    # y_true from DataFrame — reset_index ensures alignment with DataLoader.
    y_true = np.asarray(df[label_column].reset_index(drop=True).astype(int))

    # Dataset WITHOUT labels: avoids the tuple-output bug in transformers 4.48.
    ds = Dataset.from_pandas(
        df[[text_column]].reset_index(drop=True),
        preserve_index=False,
    ).map(tok_fn, batched=True, remove_columns=[text_column])

    output = trainer.predict(ds)
    logits = np.asarray(output.predictions, dtype=float)

    # Extra guard: if shape is still (1, n, 3) in some version, fix it.
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]

    proba = _softmax(logits)
    y_pred = proba.argmax(axis=1)
    return y_true, y_pred, proba


def _predict_sliding(trainer, tokenizer, df, config):
    """Predictions via sliding window + aggregation — for long-context models."""
    from tqdm.auto import tqdm

    from .tokenization import predict_long_text

    text_column = config["data"]["text_column"]
    label_column = config["data"]["label_column"]
    tk = config["tokenization"]
    stride = tk.get("sliding_window", {}).get("stride", 128)
    agg = tk.get("sliding_window", {}).get("aggregation", "mean")
    max_length = tk["max_length"]

    model = trainer.model
    model.eval()
    y_true, y_pred, probas = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="sliding-eval"):
        pred, proba = predict_long_text(
            model, tokenizer, str(row[text_column]),
            max_length=max_length, stride=stride, aggregation=agg,
        )
        y_pred.append(pred)
        probas.append(proba)
        y_true.append(int(row[label_column]))
    return np.array(y_true), np.array(y_pred), np.vstack(probas)


def evaluate_split(trainer, tokenizer, df, config, exp, logger=None) -> dict:
    """Evaluates ``df`` (test set), saves metrics + charts, returns the metrics dict."""
    strategy = config["tokenization"].get("long_text_strategy", "truncate")
    if strategy == "sliding_window":
        y_true, y_pred, proba = _predict_sliding(trainer, tokenizer, df, config)
    else:
        y_true, y_pred, proba = _predict_truncation(trainer, tokenizer, df, config)

    metrics = compute_all_metrics(y_true, y_pred, y_proba=proba)

    # Persist metrics.
    exp.write_json("metrics/test_metrics.json", metrics)
    _save_metrics_csv(metrics, exp.metrics / "test_metrics_flat.csv")
    _save_per_class_csv(metrics, exp.metrics / "test_per_class.csv")
    # Raw predictions (for auditing/reuse), without text.
    np.savez_compressed(
        exp.metrics / "test_predictions.npz",
        y_true=y_true, y_pred=y_pred, proba=proba,
    )

    if logger:
        logger.info(
            "Test — acc=%.4f f1_macro=%.4f mcc=%.4f",
            metrics["accuracy"], metrics["f1_macro"], metrics["mcc"],
        )

    _make_eval_figures(metrics, trainer, exp, config)
    return metrics


def _save_metrics_csv(metrics: dict, path: Path):
    import pandas as pd

    flat = flatten_metrics(metrics)
    pd.DataFrame([flat]).to_csv(path, index=False, encoding="utf-8")


def _save_per_class_csv(metrics: dict, path: Path):
    import pandas as pd

    rows = []
    for name, vals in (metrics.get("per_class") or {}).items():
        rows.append({"class": name, **vals})
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def _make_eval_figures(metrics: dict, trainer, exp, config):
    from . import viz

    langs = config.get("viz", {}).get("languages", ["pt", "en"])
    dpi = config.get("viz", {}).get("dpi", 150)
    fmt = config.get("viz", {}).get("format", "png")
    class_names = metrics.get("confusion_matrix_labels", CLASS_NAMES)
    prob = metrics.get("probabilistic") or {}

    # Training/val curves from log_history (if available).
    history = getattr(getattr(trainer, "state", None), "log_history", None)

    for lang in langs:
        viz.plot_confusion_matrix(
            metrics["confusion_matrix"], class_names, exp.figures,
            normalized=False, lang=lang, dpi=dpi, fmt=fmt,
        )
        viz.plot_confusion_matrix(
            metrics["confusion_matrix_normalized"], class_names, exp.figures,
            normalized=True, lang=lang, dpi=dpi, fmt=fmt,
        )
        viz.plot_per_class_metrics(metrics["per_class"], exp.figures, lang=lang, dpi=dpi, fmt=fmt)
        if prob.get("roc_curves"):
            viz.plot_roc_curves(prob["roc_curves"], exp.figures, lang=lang, dpi=dpi, fmt=fmt)
        if prob.get("pr_curves"):
            viz.plot_pr_curves(prob["pr_curves"], exp.figures, lang=lang, dpi=dpi, fmt=fmt)
        if history:
            viz.plot_training_curves(history, exp.figures, lang=lang, dpi=dpi, fmt=fmt)


# ---------------------------------------------------------------------------
# Re-evaluation from an already-trained experiment
# ---------------------------------------------------------------------------
def evaluate_from_dir(exp_dir: str | Path, config: dict, paths) -> dict:
    """Reloads the ``best_model`` from an experiment and re-evaluates on the test set."""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from .data import load_splits
    from .registry import ExperimentDir
    from .train import _detect_precision

    exp_dir = Path(exp_dir)
    best = exp_dir / "best_model"
    tokenizer = AutoTokenizer.from_pretrained(str(best))
    model = AutoModelForSequenceClassification.from_pretrained(str(best))

    # Same precision as used during training — avoids CPU vs GPU discrepancy.
    tr = config.get("train", {})
    prec = _detect_precision(tr.get("precision", "auto"))

    # eval_batch_size: prefer the explicit config value (mandatory for Longformer
    # where the auto-calc train_bs*4 causes OOM at 4096 tokens).
    eval_bs = tr.get("per_device_eval_batch_size") or min(
        tr.get("per_device_train_batch_size", 16) * tr.get("eval_batch_multiplier", 4),
        tr.get("eval_batch_cap", 128),
    )

    try:
        args = TrainingArguments(
            output_dir=str(exp_dir / "tmp_eval"),
            report_to=["none"],
            per_device_eval_batch_size=int(eval_bs),
            bf16=prec["bf16"],
            fp16=prec["fp16"],
            dataloader_num_workers=0,  # avoids fork issues on Colab
            seed=config.get("seed", 42),
        )
    except TypeError:
        # Old transformers version without some parameters
        args = TrainingArguments(
            output_dir=str(exp_dir / "tmp_eval"),
            report_to=["none"],
            per_device_eval_batch_size=int(eval_bs),
        )

    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(model=model, args=args, data_collator=data_collator)
    try:
        trainer.processing_class = tokenizer
    except Exception:
        trainer.tokenizer = tokenizer

    exp = ExperimentDir(
        root=exp_dir,
        experiment_id=config.get("experiment_id", "exp"),
        run_id=exp_dir.name.split("__")[-1],
        timestamp=exp_dir.name.split("__")[1] if "__" in exp_dir.name else "",
    )
    exp.ensure()

    splits = load_splits(paths.data_processed)
    return evaluate_split(trainer, tokenizer, splits["test"], config, exp)
