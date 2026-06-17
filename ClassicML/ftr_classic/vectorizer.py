"""
TF-IDF vectorization for classical models.

Fitted only on training data, applied consistently to val/test — no
information leakage from the test set into the vectorizer.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def build_vectorizer(config: dict):
    """Instantiate the vectorizer according to the config."""
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
    raise ValueError(f"Unknown vectorizer type: {vtype}")
