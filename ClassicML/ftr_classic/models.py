"""
Registry of classical sklearn models.

All models are instantiated from the YAML config, ensuring reproducibility.
Models without predict_proba (LinearSVC) are wrapped in CalibratedClassifierCV
to produce calibrated probabilities.
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC

_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "LinearSVC": LinearSVC,
    "SVC": SVC,
    "RandomForestClassifier": RandomForestClassifier,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
    "MultinomialNB": MultinomialNB,
    "ExtraTreesClassifier": ExtraTreesClassifier,
}

# XGBoost and LightGBM — lazy import (optional but recommended)
def _try_register_boosting():
    try:
        from xgboost import XGBClassifier
        _REGISTRY["XGBClassifier"] = XGBClassifier
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        _REGISTRY["LGBMClassifier"] = LGBMClassifier
    except ImportError:
        pass

_try_register_boosting()


def build_model(config: dict):
    """Instantiate the model according to the config.

    If the model does not support predict_proba (e.g. LinearSVC), wraps it in
    CalibratedClassifierCV with isotonic calibration to produce the
    probabilities required for ROC-AUC, Log Loss, etc.
    """
    mc = config["model"]
    cls = _REGISTRY.get(mc["name"])
    if cls is None:
        raise ValueError(f"Unknown model: {mc['name']}. Available: {list(_REGISTRY)}")

    params = mc.get("params", {}) or {}
    model = cls(**params)

    if not mc.get("requires_proba", True):
        # cv=3 may fail if a class has < 3 samples in the fold.
        # Use cv=2 as the safe minimum; on tiny datasets falls back to sigmoid.
        model = CalibratedClassifierCV(model, method="isotonic", cv=2)

    return model
