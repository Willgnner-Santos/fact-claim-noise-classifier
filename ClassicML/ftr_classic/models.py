"""
Registry de modelos clássicos sklearn.

Todos os modelos são instanciados a partir do config YAML, garantindo
reprodutibilidade. Modelos sem predict_proba (LinearSVC) são envolvidos
em CalibratedClassifierCV para produzir probabilidades calibradas.
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

# XGBoost e LightGBM — importação lazy (não obrigatórios, mas recomendados)
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
    """Instancia o modelo conforme o config.

    Se o modelo não suporta predict_proba (ex.: LinearSVC), envolve em
    CalibratedClassifierCV com calibração isotônica para produzir
    probabilidades necessárias para ROC-AUC, Log Loss etc.
    """
    mc = config["model"]
    cls = _REGISTRY.get(mc["name"])
    if cls is None:
        raise ValueError(f"Modelo desconhecido: {mc['name']}. Disponíveis: {list(_REGISTRY)}")

    params = mc.get("params", {}) or {}
    model = cls(**params)

    if not mc.get("requires_proba", True):
        # cv=3 pode falhar se uma classe tiver < 3 amostras no fold.
        # Usamos cv=2 como mínimo seguro; em datasets minúsculos cai para sigmoid.
        model = CalibratedClassifierCV(model, method="isotonic", cv=2)

    return model
