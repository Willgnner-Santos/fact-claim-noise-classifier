"""
01 — Preparação dos dados: limpeza, anonimização e split treino/val/teste.

Pipeline:
    carrega Data/dataset_FINAL.json
      -> limpeza (dropna, remove vazios, dedup, valida rótulos)
      -> (opcional) amostragem estratificada (sample_frac)
      -> anonimização de PII (todos os registros)
      -> pré-processamento textual controlado
      -> split estratificado determinístico
      -> salva em data/processed/ + data_card.json + audit_log.json

Uso:
    python scripts/01_prepare_data.py --config configs/_base.yaml
    python scripts/01_prepare_data.py --config configs/roberta_lexpt.yaml --sample-frac 0.01

O resultado em data/processed/ é COMPARTILHADO por todos os experimentos que usem
o mesmo split/anonimização (a maioria). Configs que mudam sample_frac (ex.:
bertimbau_large com 0.5) devem usar --out-subdir para não colidir.
"""

from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from ftr.anonymize import AnonymizationConfig, Anonymizer  # noqa: E402
from ftr.config import load_config  # noqa: E402
from ftr.data import (  # noqa: E402
    SplitConfig,
    clean,
    load_raw,
    maybe_sample,
    save_splits,
    split_dataset,
    split_report,
)
from ftr.env import get_paths  # noqa: E402
from ftr.preprocess import PreprocessConfig, build_preprocessor  # noqa: E402
from ftr.seeding import set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepara os dados (limpeza + anonimização + split).")
    ap.add_argument("--config", default="configs/_base.yaml", help="config YAML")
    ap.add_argument("--root", default=None, help="PROJECT_ROOT (sobrescreve autodetecção)")
    ap.add_argument("--sample-frac", type=float, default=None, help="sobrescreve data.sample_frac")
    ap.add_argument("--out-subdir", default=None, help="subpasta em data/processed/")
    ap.add_argument("--no-anonymize", action="store_true", help="desativa anonimização")
    args = ap.parse_args()

    paths = get_paths(root=args.root)
    cfg = load_config(args.config, configs_dir=paths.configs)
    set_seed(cfg.get("seed", 42))

    out_dir = paths.data_processed
    if args.out_subdir:
        out_dir = out_dir / args.out_subdir

    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]
    sample_frac = args.sample_frac if args.sample_frac is not None else cfg["data"].get("sample_frac", 1.0)

    print(f"[1/6] Carregando dataset bruto: {paths.raw_dataset}")
    df = load_raw(paths.raw_dataset, text_col, label_col)
    print(f"      registros brutos: {len(df):,}")

    print("[2/6] Limpando (dropna / vazios / duplicatas / validação de rótulos)...")
    df, clean_rep = clean(
        df, text_col, label_col,
        drop_na=cfg["data"].get("drop_na", True),
        drop_duplicates=cfg["data"].get("drop_duplicates", True),
    )
    print(f"      após limpeza: {clean_rep['n_output']:,} | {clean_rep['class_distribution']}")

    if sample_frac and sample_frac < 1.0:
        print(f"[3/6] Amostragem estratificada sample_frac={sample_frac}...")
        df = maybe_sample(df, sample_frac, label_col, cfg["split"]["random_state"])
        print(f"      após amostragem: {len(df):,}")
    else:
        print("[3/6] Sem amostragem (dataset completo).")

    # --- Anonimização ------------------------------------------------------
    anon_cfg = AnonymizationConfig.from_dict(cfg.get("anonymization"))
    if args.no_anonymize:
        anon_cfg.enabled = False
    if anon_cfg.enabled:
        print(f"[4/6] Anonimizando PII (entidades: {list(anon_cfg.entities)})...")
        anonymizer = Anonymizer(anon_cfg)
        df[text_col] = anonymizer.anonymize_many(df[text_col].astype(str).tolist())
        audit = anonymizer.audit_summary(n_docs=len(df))
        print(f"      entidades mascaradas: {audit['total_entities_masked']:,} | {audit['masked_by_type']}")
    else:
        print("[4/6] Anonimização DESATIVADA.")
        audit = {"enabled": False}

    # --- Pré-processamento -------------------------------------------------
    pp_cfg = PreprocessConfig.from_dict(cfg.get("preprocess"))
    print(f"[5/6] Pré-processamento (estratégia={pp_cfg.strategy})...")
    pp_fn = build_preprocessor(pp_cfg)
    df[text_col] = [pp_fn(t) for t in df[text_col].tolist()]

    # --- Split -------------------------------------------------------------
    split_cfg = SplitConfig.from_dict(cfg.get("split"))
    method = "por documento (anti-leakage)" if split_cfg.group_by_document else "aleatório"
    print(f"[6/6] Split {split_cfg.strategy} — método: {method}...")
    splits, split_meta = split_dataset(df, split_cfg, label_col, text_col)
    if split_meta.get("leakage_pct", 0) > 0.5:
        print(f"      ⚠ Vazamento residual: {split_meta['leakage_pct']:.1f}%")
    else:
        print(f"      Vazamento estimado: {split_meta.get('leakage_pct', 0):.2f}% (OK)")
    if "n_doc_groups" in split_meta:
        print(f"      Grupos de documento detectados: {split_meta['n_doc_groups']:,}")
    rep = split_report(splits, label_col)
    for name, info in rep.items():
        print(f"      {name}: n={info['n']:,} {info['class_distribution']}")

    paths_saved = save_splits(splits, out_dir)

    # --- Data card + audit log --------------------------------------------
    data_card = {
        "source": str(paths.raw_dataset),
        "config": cfg.get("__config_name__"),
        "sample_frac": sample_frac,
        "cleaning": clean_rep,
        "split": {"config": vars(split_cfg), "report": rep, "meta": split_meta},
        "preprocess": vars(pp_cfg),
        "anonymization": {"config": {
            "enabled": anon_cfg.enabled,
            "entities": list(anon_cfg.entities),
            "mask_names": anon_cfg.mask_names,
        }},
        "files": paths_saved,
    }
    with (out_dir / "data_card.json").open("w", encoding="utf-8") as f:
        json.dump(data_card, f, ensure_ascii=False, indent=2)
    with (out_dir / "audit_log.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    print(f"\nOK. Dados em: {out_dir}")
    print(f"  - splits: {list(paths_saved.keys())}")
    print(f"  - data_card.json, audit_log.json")


if __name__ == "__main__":
    main()
