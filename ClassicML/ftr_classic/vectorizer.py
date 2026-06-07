"""
Vetorização TF-IDF para os modelos clássicos.

Ajustado apenas no treino, aplicado consistentemente em val/teste — sem
vazamento de informação do teste para o vetorizador.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def build_vectorizer(config: dict):
    """Instancia o vetorizador conforme o config."""
    vc = config.get("vectorizer", {})
    vtype = vc.get("type", "tfidf")

    params = {
        "max_features": vc.get("max_features", 50000),
        "ngram_range": tuple(vc.get("ngram_range", [1, 2])),
        "min_df": vc.get("min_df", 2),
        "strip_accents": vc.get("strip_accents"),
        "analyzer": vc.get("analyzer", "word"),
    }

    if vtype == "tfidf":
        params["sublinear_tf"] = vc.get("sublinear_tf", True)
        return TfidfVectorizer(**params)
    if vtype == "count":
        return CountVectorizer(**params)
    raise ValueError(f"Tipo de vetorizador desconhecido: {vtype}")
