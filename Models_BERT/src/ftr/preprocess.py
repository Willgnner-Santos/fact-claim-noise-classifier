"""
Pré-processamento textual controlado e reprodutível.

Por padrão reproduz o que os notebooks originais faziam (apenas ``str(t)`` e um
``strip``), porque os modelos são *cased* e qualquer limpeza agressiva mudaria
os experimentos. Estratégias adicionais são opt-in via config:

* ``"none"``              — fidelidade aos originais (default).
* ``"legacy_normalizer"`` — usa o ``normalizer.py`` do projeto de normalização
  vizinho (spaCy/NLTK, normalização jurídica), se disponível.

A função :func:`build_preprocessor` devolve um *callable* ``str -> str``
determinístico, aplicável de forma idêntica a treino/validação/teste.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import dataclass

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class PreprocessConfig:
    to_string: bool = True
    strip: bool = True
    lowercase: bool = False
    normalize_whitespace: bool = True
    strategy: str = "none"  # "none" | "legacy_normalizer"

    @classmethod
    def from_dict(cls, d: dict | None) -> "PreprocessConfig":
        d = d or {}
        return cls(
            to_string=d.get("to_string", True),
            strip=d.get("strip", True),
            lowercase=d.get("lowercase", False),
            normalize_whitespace=d.get("normalize_whitespace", True),
            strategy=d.get("strategy", "none"),
        )


def _basic_clean(text, cfg: PreprocessConfig) -> str:
    if cfg.to_string and not isinstance(text, str):
        text = "" if text is None else str(text)
    if cfg.normalize_whitespace:
        text = _WHITESPACE_RE.sub(" ", text)
    if cfg.lowercase:
        text = text.lower()
    if cfg.strip:
        text = text.strip()
    return text


def _load_legacy_normalizer():
    """Tenta importar o Normalizer do projeto de normalização vizinho."""
    import warnings
    from pathlib import Path

    # Caminho configurável via variável de ambiente — não mais hardcoded Windows.
    env_path = os.environ.get("FTR_NORMALIZER_PATH")
    default_paths = [
        Path(r"e:/Anonymous/projeto_normalizacao_ftp/pipeline_normalizacao/normalizer.py"),
    ]
    candidates = ([Path(env_path)] if env_path else []) + default_paths

    for c in candidates:
        try:
            if c.is_file():
                spec = importlib.util.spec_from_file_location("legacy_normalizer", c)
                mod = importlib.util.module_from_spec(spec)
                sys.modules["legacy_normalizer"] = mod
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                normalizer_cls = getattr(mod, "Normalizer", None)
                if normalizer_cls is not None:
                    return normalizer_cls
        except Exception as e:
            warnings.warn(
                f"Falha ao carregar legacy_normalizer de {c}: {e}",
                UserWarning,
                stacklevel=3,
            )
    warnings.warn(
        "legacy_normalizer não encontrado. Usando limpeza básica como fallback. "
        "Defina FTR_NORMALIZER_PATH para apontar para normalizer.py.",
        UserWarning,
        stacklevel=3,
    )
    return None


def build_preprocessor(cfg: PreprocessConfig):
    """Retorna um callable ``str -> str`` conforme a estratégia configurada."""
    if cfg.strategy == "legacy_normalizer":
        Normalizer = _load_legacy_normalizer()
        if Normalizer is not None:
            try:
                norm = Normalizer()
                _fail_count = [0]

                def _fn(text):
                    text = _basic_clean(text, cfg)
                    try:
                        res = norm.normalizar(fato=text, tese="")
                        return res.get("fato_normalizado", text)
                    except Exception as e:
                        _fail_count[0] += 1
                        if _fail_count[0] <= 3:
                            import warnings
                            warnings.warn(
                                f"normalizar() falhou (#{_fail_count[0]}): {e}. "
                                "Usando texto sem normalização.",
                                UserWarning,
                                stacklevel=2,
                            )
                        return text

                return _fn
            except Exception:
                pass
        # Fallback se o normalizer não puder ser carregado: limpeza básica.

    def _fn(text):
        return _basic_clean(text, cfg)

    return _fn


def preprocess_texts(texts, cfg: PreprocessConfig) -> list[str]:
    fn = build_preprocessor(cfg)
    return [fn(t) for t in texts]
