"""
Carregamento de dados para o pipeline ClassicML.

Reutiliza os splits já gerados pelo pipeline BERT (group_by_document=True,
0% leakage) para comparação justa. Se os splits não existirem, gera do raw
com o mesmo pipeline de limpeza e anonimização.

A anonimização e o pré-processamento são aplicados da mesma forma que no
pipeline BERT — usando src/ftr/anonymize.py e src/ftr/preprocess.py.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_ftr_on_path(project_root: Path) -> None:
    """Adiciona src/ftr ao sys.path — tenta Models_BERT/src primeiro."""
    for candidate in [project_root / "Models_BERT" / "src", project_root / "src"]:
        if (candidate / "ftr").exists():
            s = str(candidate)
            if s not in sys.path:
                sys.path.insert(0, s)
            return


def load_splits_from_bert(processed_dir: Path) -> dict[str, pd.DataFrame] | None:
    """Carrega splits já processados pelo pipeline BERT (parquet/jsonl)."""
    splits = {}
    for name in ("train", "val", "test"):
        for ext, reader in [(".parquet", pd.read_parquet), (".jsonl", lambda p: pd.read_json(p, lines=True))]:
            p = processed_dir / f"{name}{ext}"
            if p.exists():
                splits[name] = reader(p)
                break
    if not splits:
        return None
    logger.info("Splits BERT carregados: %s", {k: len(v) for k, v in splits.items()})
    return splits


def load_raw_and_process(config: dict, project_root: Path) -> dict[str, pd.DataFrame]:
    """Gera splits do zero usando o mesmo pipeline BERT (anonimização + split por doc)."""
    _ensure_ftr_on_path(project_root)

    from ftr.anonymize import Anonymizer, AnonymizationConfig
    from ftr.data import clean, maybe_sample, save_splits, split_dataset, SplitConfig
    from ftr.preprocess import PreprocessConfig, build_preprocessor

    classic_root = Path(__file__).resolve().parents[1]
    raw_path = (classic_root / config["data"]["raw_file"]).resolve()
    text_col = config["data"]["text_column"]
    label_col = config["data"]["label_column"]

    logger.info("Carregando dataset bruto: %s", raw_path)
    with raw_path.open(encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    df, clean_rep = clean(df, text_col, label_col, drop_na=True, drop_duplicates=True)
    logger.info("Limpeza: %s", clean_rep)

    # Anonimização idêntica ao pipeline BERT
    anon_cfg = AnonymizationConfig.from_dict(config.get("anonymization"))
    if anon_cfg.enabled:
        logger.info("Anonimizando PII...")
        anonymizer = Anonymizer(anon_cfg)
        anonymizer.ensure_ready()
        df[text_col] = anonymizer.anonymize_many(df[text_col].astype(str).tolist())
        audit = anonymizer.audit_summary(n_docs=len(df))
        logger.info("Anonimização: %d entidades mascaradas", audit["total_entities_masked"])

    # Pré-processamento idêntico
    pp_cfg = PreprocessConfig.from_dict(config.get("preprocess"))
    pp_fn = build_preprocessor(pp_cfg)
    df[text_col] = [pp_fn(t) for t in df[text_col].tolist()]

    split_cfg = SplitConfig.from_dict(config.get("split", {}))
    splits, meta = split_dataset(df, split_cfg, label_col, text_col)
    logger.info("Split: %s | leakage=%.2f%%", {k: len(v) for k, v in splits.items()}, meta.get("leakage_pct", 0))
    return splits


def get_splits(config: dict, project_root: Path) -> dict[str, pd.DataFrame]:
    """Retorna splits — tenta reutilizar os do BERT, gera do zero se necessário."""
    if config["data"].get("use_bert_splits", True):
        # Resolve caminho relativo (ex: "../Models_BERT/data/processed") a partir de ClassicML/
        classic_root = Path(__file__).resolve().parents[1]
        processed_dir = (classic_root / config["data"]["processed_dir"]).resolve()
        splits = load_splits_from_bert(processed_dir)
        if splits:
            return splits
        logger.warning("Splits BERT não encontrados em %s — gerando do zero.", processed_dir)
    return load_raw_and_process(config, project_root)


def get_xy(splits: dict[str, pd.DataFrame], config: dict) -> dict[str, tuple]:
    """Extrai (X_texts, y_labels) de cada split."""
    tc = config["data"]["text_column"]
    lc = config["data"]["label_column"]
    return {
        name: (df[tc].astype(str).tolist(), df[lc].astype(int).tolist())
        for name, df in splits.items()
    }
