"""
Reproducible training of an experiment (HuggingFace ``Trainer`` wrapper).

Faithful to the original notebooks (AdamW, weight_decay=0.01, max_length=512,
truncation, EarlyStopping patience=2, save best, eval/save per epoch) and adds
reproducibility best practices: global seed, config snapshot, ``env.json``,
checkpoint per epoch, automatic resume, memory cleanup, detailed logs,
per-epoch metrics and ``status.json`` with timestamps/errors.

Main function: :func:`train_experiment(config, paths)`.

All torch/transformers imports are lazy so the module can be inspected
locally without a GPU.
"""

from __future__ import annotations

import gc
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any

from .config import save_config_snapshot
from .env_info import collect_environment
from .registry import ExperimentDir, make_experiment_dir
from .seeding import set_seed


def _now_stamp() -> str:
    """Local timestamp 'YYYYmmdd_HHMMSS' (uses system time; ok outside workflows)."""
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _setup_logger(log_dir: Path, name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"ftr.train.{name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def _free_memory():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _detect_precision(requested: str) -> dict:
    """Resolves precision (bf16/fp16/tf32) based on available GPU.

    ``requested`` can be ``"auto"``, ``"bf16"``, ``"fp16"`` or ``"fp32"``.
    Returns kwargs ready for TrainingArguments + tf32 flag. bf16 is only used
    if the GPU supports it (Ampere+ / compute capability >= 8.0, e.g. A100, H100).
    """
    out = {"bf16": False, "fp16": False, "tf32_supported": False}
    try:
        import torch

        if not torch.cuda.is_available():
            return out
        major = torch.cuda.get_device_capability(0)[0]
        bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: major >= 8)())
        out["tf32_supported"] = major >= 8  # Ampere+
        if requested == "auto":
            if bf16_ok:
                out["bf16"] = True       # A100/H100 → bf16 (more stable than fp16)
            else:
                out["fp16"] = True       # T4/V100/Pascal → fp16
        elif requested == "bf16" and bf16_ok:
            out["bf16"] = True
        elif requested == "fp16":
            out["fp16"] = True
        # "fp32" → both False
    except Exception:
        pass
    return out


def _resolve_num_workers(value) -> int:
    if value == "auto" or value is None:
        try:
            return min(8, max(2, (os.cpu_count() or 2)))
        except Exception:
            return 2
    return int(value)


def build_training_arguments(config: dict, output_dir: Path):
    """Builds ``TrainingArguments`` from config.

    Applies GPU-auto-detected throughput optimisations (bf16/tf32, larger eval
    batch, workers, group_by_length) — user's choice to maximise GPU utilisation.
    These options do NOT change training dynamics. Train batch and learning rate
    come from config (see LR scaling note in DECISOES.md).
    """
    from transformers import TrainingArguments

    tr = config["train"]
    train_bs = tr["per_device_train_batch_size"]

    # Eval batch: no gradients, can safely be larger — but with a CAP,
    # otherwise a large train_bs (e.g. 128*4=512 @ 512 tokens) risks OOM
    # precisely at validation epoch 1. Configurable cap (eval_batch_cap).
    eval_bs = tr.get("per_device_eval_batch_size")
    if not eval_bs:
        cap = tr.get("eval_batch_cap", 128)
        eval_bs = min(train_bs * tr.get("eval_batch_multiplier", 4), cap)

    prec = _detect_precision(tr.get("precision", "auto"))

    # TF32 (matmul) on Ampere+ — speed gain with no practical effect on metrics.
    if tr.get("tf32", True) and prec["tf32_supported"]:
        try:
            import torch

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=tr["num_train_epochs"],
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=int(eval_bs),
        gradient_accumulation_steps=tr.get("gradient_accumulation_steps", 1),
        learning_rate=float(tr["learning_rate"]),
        weight_decay=tr.get("weight_decay", 0.01),
        lr_scheduler_type=tr.get("lr_scheduler_type", "linear"),
        warmup_ratio=tr.get("warmup_ratio", 0.0),
        logging_steps=tr.get("logging_steps", 50),
        save_total_limit=tr.get("save_total_limit", 2),
        load_best_model_at_end=tr.get("load_best_model_at_end", True),
        metric_for_best_model=tr.get("metric_for_best_model", "f1_macro"),
        greater_is_better=tr.get("greater_is_better", True),
        bf16=prec["bf16"],
        fp16=prec["fp16"],
        tf32=tr.get("tf32", True) and prec["tf32_supported"],
        gradient_checkpointing=tr.get("gradient_checkpointing", False),
        optim=tr.get("optimizer", "adamw_torch"),
        dataloader_num_workers=_resolve_num_workers(tr.get("dataloader_num_workers", "auto")),
        dataloader_pin_memory=tr.get("dataloader_pin_memory", True),
        group_by_length=tr.get("group_by_length", True),
        seed=config.get("seed", 42),
        report_to=["none"],
        disable_tqdm=False,
    )

    if tr.get("torch_compile", False):
        kwargs["torch_compile"] = True

    # eval_strategy was renamed between transformers versions — try new name first,
    # fall back to old if needed. Remove unsupported keys if required.
    eval_strategy = tr.get("eval_strategy", "epoch")
    save_strategy = tr.get("save_strategy", "epoch")

    def _construct(**extra):
        try:
            return TrainingArguments(eval_strategy=eval_strategy, save_strategy=save_strategy, **kwargs, **extra)
        except TypeError:
            return TrainingArguments(evaluation_strategy=eval_strategy, save_strategy=save_strategy, **kwargs, **extra)

    try:
        return _construct()
    except TypeError:
        # Old transformers version without 'tf32'/'group_by_length'/'torch_compile' — remove and retry.
        for opt in ("tf32", "group_by_length", "torch_compile", "dataloader_pin_memory"):
            kwargs.pop(opt, None)
        return _construct()


def _build_callbacks(config: dict):
    from transformers import EarlyStoppingCallback

    cbs = []
    es = config["train"].get("early_stopping", {})
    if es.get("enabled", True):
        cbs.append(
            EarlyStoppingCallback(
                early_stopping_patience=es.get("patience", 2),
                early_stopping_threshold=es.get("threshold", 0.0),
            )
        )
    return cbs


# ---------------------------------------------------------------------------
# Focal Loss + Class Weights
# ---------------------------------------------------------------------------
def _compute_class_weights(labels_series, num_labels: int):
    """Computes balanced class weights (sklearn) from training labels."""
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.arange(num_labels)
    y = labels_series.to_numpy()
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return weights  # ndarray shape (num_labels,)


class FocalLossTrainer:
    """Mixin that overrides ``compute_loss`` in Trainer with Focal Loss or
    weighted Cross Entropy. Does NOT inherit from Trainer here — combined via
    ``build_custom_trainer`` to avoid circular imports with torch."""

    _class_weights = None   # float tensor, dtype=float32, device=model.device
    _focal_gamma: float = 2.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        import torch
        import torch.nn.functional as F

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self._class_weights is not None:
            w = self._class_weights.to(logits.device)
        else:
            w = None

        if self._focal_gamma > 0:
            # Focal Loss: FL(p_t) = -(1-p_t)^gamma * log(p_t)
            # with optional class weights on the base CE.
            ce = F.cross_entropy(logits, labels, weight=w, reduction="none")
            pt = torch.exp(-ce)
            focal = ((1 - pt) ** self._focal_gamma) * ce
            loss = focal.mean()
        else:
            # Pure Weighted Cross Entropy.
            loss = F.cross_entropy(logits, labels, weight=w)

        return (loss, outputs) if return_outputs else loss


def build_custom_trainer(use_focal: bool, use_class_weights: bool, gamma: float = 2.0):
    """Returns a ``Trainer`` subclass configured according to the flags.

    Uses a factory function to keep torch/transformers imports lazy
    (train.py is importable without GPU).
    """
    from transformers import Trainer

    if not use_focal and not use_class_weights:
        return Trainer  # no modification — default behaviour

    class CustomTrainer(FocalLossTrainer, Trainer):
        pass

    CustomTrainer._focal_gamma = gamma if use_focal else 0.0
    return CustomTrainer


# ---------------------------------------------------------------------------
# Differential Learning Rates (Layer-wise LR Decay)
# ---------------------------------------------------------------------------
def build_optimizer_with_differential_lr(model, config: dict):
    """AdamW with differential LR: deeper layers use a higher LR.

    "Layer-wise decay" strategy: each layer receives an LR equal to
    ``base_lr * decay^(num_layers - layer_idx)``. Classification layers
    (pooler + head) keep the base LR; the embedding layer gets
    ``base_lr * decay^num_layers`` (smallest). Typical values: decay 0.9-0.95.

    Returns ``None`` if differential_lr is not configured (uses Trainer's
    default optimiser).
    """
    tr = config["train"]
    dlr = tr.get("differential_lr", {})
    if not dlr.get("enabled", False):
        return None

    import torch
    from torch.optim import AdamW

    base_lr = float(tr["learning_rate"])
    decay = float(dlr.get("decay", 0.9))
    wd = float(tr.get("weight_decay", 0.01))

    # Collect encoder layers (BERT/RoBERTa have model.encoder.layer)
    encoder = getattr(model, "bert", None) or getattr(model, "roberta", None) or \
              getattr(model, "deberta", None) or getattr(model, "longformer", None)

    if encoder is None or not hasattr(encoder, "encoder"):
        # Explicit warning — differential_lr=true configured but not applied.
        model_name = config.get("model", {}).get("name", "unknown")
        import warnings
        warnings.warn(
            f"differential_lr enabled but architecture not recognised for '{model_name}'. "
            "Uniform LR will be used. Supported: BERT/RoBERTa/DeBERTa/Longformer.",
            UserWarning,
            stacklevel=2,
        )
        return None

    layers = list(encoder.encoder.layer)
    n = len(layers)

    groups = []
    # Embeddings — minimum LR
    groups.append({
        "params": list(encoder.embeddings.parameters()),
        "lr": base_lr * (decay ** n),
        "weight_decay": wd,
    })
    # Transformer layers — LR increases from first to last
    for i, layer in enumerate(layers):
        groups.append({
            "params": list(layer.parameters()),
            "lr": base_lr * (decay ** (n - i - 1)),
            "weight_decay": wd,
        })
    # Pooler + classifier — full base LR
    head_params = []
    for name in ("pooler", "classifier", "pre_classifier"):
        m = getattr(model, name, None)
        if m is not None:
            head_params.extend(list(m.parameters()))
    if head_params:
        groups.append({"params": head_params, "lr": base_lr, "weight_decay": wd})

    return AdamW(groups)


def _to_hf_dataset(df, text_column: str, label_column: str):
    """Converts a DataFrame to a ``datasets.Dataset`` with text/labels columns."""
    from datasets import Dataset

    keep = df[[text_column, label_column]].rename(columns={label_column: "labels"})
    return Dataset.from_pandas(keep, preserve_index=False)


def train_experiment(
    config: dict,
    paths,
    splits: dict | None = None,
    max_steps: int | None = None,
) -> dict:
    """Trains an experiment end-to-end and returns the summary (status dict).

    Args:
        config: merged config (from ``load_config``).
        paths: :class:`ftr.env.Paths` object.
        splits: dict ``{"train": df, "val": df, "test": df}``; if ``None``,
            loads from ``data/processed``.
        max_steps: if provided, limits steps (smoke-test mode).
    """
    from transformers import DataCollatorWithPadding, Trainer

    from .data import load_splits
    from .metrics import compute_all_metrics, compute_metrics_for_trainer, flatten_metrics
    from .models import build_model, build_tokenizer, model_summary
    from .tokenization import make_tokenize_function

    set_seed(config.get("seed", 42), deterministic=config.get("deterministic", False))

    experiment_id = config.get("experiment_id", config.get("__config_name__", "exp"))
    timestamp = _now_stamp()
    exp: ExperimentDir = make_experiment_dir(paths.experiments, experiment_id, timestamp)
    logger = _setup_logger(exp.logs, experiment_id)
    logger.info("=== Starting experiment %s (run %s) ===", experiment_id, exp.run_id)

    # Config and environment snapshot.
    save_config_snapshot(config, exp.root / "config_snapshot.yaml")
    env = collect_environment(seed=config.get("seed", 42))
    exp.write_json("env.json", env)

    status: dict[str, Any] = {
        "experiment_id": experiment_id,
        "run_id": exp.run_id,
        "run_dir": exp.root.name,
        "model_name": config["model"]["name"],
        "is_reproduction": config["model"].get("is_reproduction"),
        "status": "running",
        "started_at": timestamp,
        "model_summary": model_summary(config),
        "smoke_test": max_steps is not None,
    }
    exp.write_status(status)

    t0 = time.time()
    try:
        # --- Data -----------------------------------------------------------
        if splits is None:
            splits = load_splits(paths.data_processed)
        if "train" not in splits or "test" not in splits:
            raise RuntimeError(
                "Splits not found. Run scripts/01_prepare_data.py first."
            )
        text_column = config["data"]["text_column"]
        label_column = config["data"]["label_column"]

        logger.info("Sizes: %s", {k: len(v) for k, v in splits.items()})

        # --- Tokeniser / model ---------------------------------------------
        tokenizer = build_tokenizer(config)
        model = build_model(config)
        is_longformer = config["model"].get("family", "encoder") == "longformer"
        tok_fn = make_tokenize_function(
            tokenizer, text_column, config["tokenization"]["max_length"],
            set_global_attention_cls=is_longformer,
        )

        eval_split = "val" if "val" in splits else "test"
        ds_train = _to_hf_dataset(splits["train"], text_column, label_column).map(
            tok_fn, batched=True, remove_columns=[text_column]
        )
        ds_eval = _to_hf_dataset(splits[eval_split], text_column, label_column).map(
            tok_fn, batched=True, remove_columns=[text_column]
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        args = build_training_arguments(config, exp.checkpoints)
        if max_steps is not None:
            args.max_steps = max_steps

        # --- Focal Loss / Class Weights (opt-in via config) -----------------
        adv_cfg = config["train"].get("advanced", {})
        use_focal = adv_cfg.get("focal_loss", False)
        use_cw = adv_cfg.get("class_weights", False)
        focal_gamma = float(adv_cfg.get("focal_gamma", 2.0))
        TrainerClass = build_custom_trainer(use_focal, use_cw, gamma=focal_gamma)

        class_weights_tensor = None
        if use_cw:
            import torch
            cw = _compute_class_weights(splits["train"][label_column],
                                        config["labels"]["num_labels"])
            class_weights_tensor = torch.tensor(cw, dtype=torch.float32)
            logger.info("Class weights computed: %s", dict(enumerate(cw.round(3).tolist())))

        # --- Differential LR (opt-in via config) ----------------------------
        custom_optimizer = build_optimizer_with_differential_lr(model, config)
        if custom_optimizer is not None:
            logger.info("Differential LR enabled (decay=%.2f)",
                        config["train"].get("differential_lr", {}).get("decay", 0.9))

        trainer = TrainerClass(
            model=model,
            args=args,
            train_dataset=ds_train,
            eval_dataset=ds_eval,
            data_collator=data_collator,
            compute_metrics=compute_metrics_for_trainer,
            callbacks=_build_callbacks(config),
            optimizers=(custom_optimizer, None) if custom_optimizer is not None else (None, None),
        )
        if class_weights_tensor is not None:
            trainer._class_weights = class_weights_tensor
        try:  # recent transformers
            trainer.processing_class = tokenizer
        except Exception:
            trainer.tokenizer = tokenizer

        # --- Training (with resume) ----------------------------------------
        resume = config["train"].get("resume_from_checkpoint", False) and max_steps is None
        resume_arg = _find_checkpoint(exp.checkpoints) if resume else None
        if resume_arg:
            logger.info("Resuming from checkpoint: %s", resume_arg)
        train_result = trainer.train(resume_from_checkpoint=resume_arg)

        # Save best model + tokenizer.
        best_dir = exp.root / "best_model"
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))

        # --- Per-epoch metrics (log_history) --------------------------------
        epoch_metrics = _extract_epoch_metrics(trainer.state.log_history)
        status["epoch_metrics"] = epoch_metrics
        exp.write_json("metrics/epoch_metrics.json", epoch_metrics)
        exp.write_json("trainer_log_history.json", trainer.state.log_history)

        # --- Validation set evaluation (fast, via Trainer) ------------------
        val_eval = trainer.evaluate(ds_eval)
        exp.write_json("metrics/val_metrics_trainer.json", val_eval)

        # --- Full evaluation on test set ------------------------------------
        # IMPORTANT: uses a NEW Trainer loaded from best_model, not the training
        # trainer. The training trainer has internal state (sampler, buffers)
        # that reorders batches in predict(), misaligning y_pred from y_true.
        from .evaluate import evaluate_split, evaluate_from_dir

        test_metrics = evaluate_from_dir(exp.root, config, paths)

        duration = time.time() - t0
        status.update(
            {
                "status": "completed",
                "finished_at": _now_stamp(),
                "duration_seconds": round(duration, 1),
                "train_runtime_seconds": round(train_result.metrics.get("train_runtime", 0), 1),
                "final_train_loss": train_result.metrics.get("train_loss"),
                "best_model_metric": {
                    config["train"].get("metric_for_best_model"):
                        flatten_metrics(test_metrics).get(
                            config["train"].get("metric_for_best_model")
                        )
                },
                "artifacts": {
                    "best_model": str(best_dir),
                    "checkpoints": str(exp.checkpoints),
                    "test_metrics": str(exp.metrics / "test_metrics.json"),
                    "config_snapshot": str(exp.root / "config_snapshot.yaml"),
                    "env": str(exp.root / "env.json"),
                    "figures": str(exp.figures),
                },
            }
        )
        exp.write_status(status)
        logger.info("=== Completed in %.1fs ===", duration)

    except Exception as e:  # noqa: BLE001 — log any failure
        status.update(
            {
                "status": "failed",
                "finished_at": _now_stamp(),
                "duration_seconds": round(time.time() - t0, 1),
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )
        exp.write_status(status)
        if "logger" in dir():
            logger.error("Experiment failed: %s", e)
            logger.error(traceback.format_exc())
        raise
    finally:
        _free_memory()

    return status


def _find_checkpoint(checkpoints_dir: Path) -> str | None:
    """Finds the latest saved checkpoint (for resume)."""
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.exists():
        return None
    cks = [d for d in checkpoints_dir.glob("checkpoint-*") if d.is_dir()]
    if not cks:
        return None
    cks.sort(key=lambda d: int(d.name.split("-")[-1]))
    return str(cks[-1])


def _extract_epoch_metrics(log_history: list[dict]) -> list[dict]:
    """Collects per-epoch training and evaluation metrics from log_history."""
    by_epoch: dict[float, dict] = {}
    for h in log_history:
        ep = h.get("epoch")
        if ep is None:
            continue
        slot = by_epoch.setdefault(round(float(ep), 4), {"epoch": round(float(ep), 4)})
        for k, v in h.items():
            if k in ("epoch", "step"):
                continue
            slot[k] = v
    return [by_epoch[k] for k in sorted(by_epoch)]
