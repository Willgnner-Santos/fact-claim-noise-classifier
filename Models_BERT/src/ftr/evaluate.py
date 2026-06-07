"""
Avaliação completa de um experimento no conjunto de teste.

Gera predições (truncação padrão ou janela deslizante para long-context),
calcula **todas** as métricas (:mod:`ftr.metrics`), salva em JSON + CSV e produz
os gráficos de avaliação (:mod:`ftr.viz`) em PT-BR e EN.

Função principal: :func:`evaluate_split`, chamada ao final do treino. Também há
:func:`evaluate_from_dir` para reavaliar um experimento já treinado a partir do
``best_model`` salvo.
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
    """Predições via Trainer (tokenização com truncação) — caminho rápido/fiel.

    Importante: NÃO passamos a coluna ``labels`` ao dataset de predição.
    Em transformers 4.48, quando ``labels`` estão presentes o modelo computa o
    loss e ``output.predictions`` retorna uma tupla ``(logits,)`` em vez de
    ``logits`` diretamente. ``np.asarray((logits,))`` vira shape ``(1, n, 3)``
    e o softmax / argmax calculam no eixo errado → modelo sempre prediz classe 0.
    Extraímos ``y_true`` direto do DataFrame (ordem preservada pelo parquet) e
    passamos apenas texto ao dataset de inferência.
    """
    from datasets import Dataset

    from .tokenization import make_tokenize_function

    text_column = config["data"]["text_column"]
    label_column = config["data"]["label_column"]
    tok_fn = make_tokenize_function(tokenizer, text_column, config["tokenization"]["max_length"])

    # y_true do DataFrame — reset_index garante alinhamento com o DataLoader.
    y_true = np.asarray(df[label_column].reset_index(drop=True).astype(int))

    # Dataset SEM labels: evita o bug do tuple-output no transformers 4.48.
    ds = Dataset.from_pandas(
        df[[text_column]].reset_index(drop=True),
        preserve_index=False,
    ).map(tok_fn, batched=True, remove_columns=[text_column])

    output = trainer.predict(ds)
    logits = np.asarray(output.predictions, dtype=float)

    # Defesa extra: se ainda vier como (1, n, 3) por alguma versão, corrige.
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]

    proba = _softmax(logits)
    y_pred = proba.argmax(axis=1)
    return y_true, y_pred, proba


def _predict_sliding(trainer, tokenizer, df, config):
    """Predições via janela deslizante + agregação — para long-context."""
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
    """Avalia o ``df`` (teste), salva métricas + gráficos, retorna o dict de métricas."""
    strategy = config["tokenization"].get("long_text_strategy", "truncate")
    if strategy == "sliding_window":
        y_true, y_pred, proba = _predict_sliding(trainer, tokenizer, df, config)
    else:
        y_true, y_pred, proba = _predict_truncation(trainer, tokenizer, df, config)

    metrics = compute_all_metrics(y_true, y_pred, y_proba=proba)

    # Persistência das métricas.
    exp.write_json("metrics/test_metrics.json", metrics)
    _save_metrics_csv(metrics, exp.metrics / "test_metrics_flat.csv")
    _save_per_class_csv(metrics, exp.metrics / "test_per_class.csv")
    # Predições brutas (para auditoria/reuso), sem o texto.
    np.savez_compressed(
        exp.metrics / "test_predictions.npz",
        y_true=y_true, y_pred=y_pred, proba=proba,
    )

    if logger:
        logger.info(
            "Teste — acc=%.4f f1_macro=%.4f mcc=%.4f",
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

    # Curvas de treino/val a partir do log_history (se houver).
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
# Reavaliação a partir de um experimento já treinado
# ---------------------------------------------------------------------------
def evaluate_from_dir(exp_dir: str | Path, config: dict, paths) -> dict:
    """Recarrega o ``best_model`` de um experimento e reavalia no conjunto de teste."""
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

    # Precisão idêntica à usada no treino — evita discrepância CPU vs GPU.
    tr = config.get("train", {})
    prec = _detect_precision(tr.get("precision", "auto"))

    # eval_batch_size: usa o mesmo teto do treino para não causar OOM.
    train_bs = tr.get("per_device_train_batch_size", 16)
    cap = tr.get("eval_batch_cap", 128)
    eval_bs = min(train_bs * tr.get("eval_batch_multiplier", 4), cap)

    try:
        args = TrainingArguments(
            output_dir=str(exp_dir / "tmp_eval"),
            report_to=["none"],
            per_device_eval_batch_size=int(eval_bs),
            bf16=prec["bf16"],
            fp16=prec["fp16"],
            dataloader_num_workers=0,  # evita problemas de fork no Colab
            seed=config.get("seed", 42),
        )
    except TypeError:
        # Versão antiga do transformers sem alguns parâmetros
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
