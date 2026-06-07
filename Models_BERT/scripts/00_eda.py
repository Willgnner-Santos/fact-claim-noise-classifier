"""
00 — Análise exploratória do dataset (EDA).

Carrega o dataset bruto, calcula estatísticas descritivas e gera gráficos em
PT-BR e EN (distribuição de classes e de tamanho dos textos). Opcionalmente
calcula a distribuição de tokens reais usando um tokenizer (requer transformers).

Roda LOCALMENTE sem GPU. Saídas em results/figures/eda/ e reports/eda_report.json.

Uso:
    python scripts/00_eda.py
    python scripts/00_eda.py --tokenizer neuralmind/bert-base-portuguese-cased --token-sample 2000
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from ftr import CLASS_NAMES, ID2LABEL  # noqa: E402
from ftr.data import clean, load_raw  # noqa: E402
from ftr.env import get_paths  # noqa: E402
from ftr import viz  # noqa: E402


def _describe(arr) -> dict:
    arr = np.asarray(arr)
    return {
        "min": int(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "mean": float(arr.mean()),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA do dataset Fato/Tese/Ruído.")
    ap.add_argument("--root", default=None)
    ap.add_argument("--raw", action="store_true", help="usar dataset BRUTO sem limpeza")
    ap.add_argument("--tokenizer", default=None, help="modelo p/ contar tokens reais")
    ap.add_argument("--token-sample", type=int, default=2000, help="amostra p/ contagem de tokens")
    args = ap.parse_args()

    paths = get_paths(root=args.root)
    eda_dir = paths.results_figures / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    print(f"Carregando {paths.raw_dataset} ...")
    df = load_raw(paths.raw_dataset)
    report = {"n_raw": int(len(df))}

    if not args.raw:
        df, clean_rep = clean(df)
        report["cleaning"] = clean_rep

    label_col = "label"
    text_col = "text"

    # Distribuição de classes
    counts = {ID2LABEL[k]: int(v) for k, v in df[label_col].value_counts().sort_index().items()}
    report["class_distribution"] = counts

    # Tamanhos (chars e palavras)
    texts = df[text_col].astype(str)
    char_len = texts.str.len().to_numpy()
    word_len = texts.str.split().map(len).to_numpy()
    report["char_length"] = _describe(char_len)
    report["word_length"] = _describe(word_len)
    report["pct_words_gt_512"] = float((word_len > 512).mean() * 100)

    # Tokens reais (opcional)
    token_len = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer

            print(f"Contando tokens com {args.tokenizer} (amostra={args.token_sample})...")
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
            sample = texts.sample(min(args.token_sample, len(texts)), random_state=42).tolist()
            token_len = np.array([len(tok.encode(t, truncation=False)) for t in sample])
            report["token_length_sample"] = _describe(token_len)
            report["pct_tokens_gt_512_sample"] = float((token_len > 512).mean() * 100)
            report["token_sample_size"] = len(sample)
            report["tokenizer"] = args.tokenizer
        except Exception as e:
            report["token_length_error"] = str(e)
            print(f"  (não foi possível contar tokens: {e})")

    # Gráficos PT + EN
    print("Gerando gráficos (PT/EN)...")
    figs = []
    for lang in ("pt", "en"):
        figs.append(viz.plot_class_distribution(counts, eda_dir, lang=lang))
        figs.append(viz.plot_length_distribution(word_len, eda_dir, unit="words", lang=lang, vline=512))
        if token_len is not None:
            figs.append(viz.plot_length_distribution(token_len, eda_dir, unit="tokens", lang=lang, vline=512))
    report["figures"] = figs

    out = paths.reports / "eda_report.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Resumo no console
    print("\n=== RESUMO EDA ===")
    print("Classes:", counts)
    print("Palavras: median=%.0f p90=%.0f p99=%.0f max=%d" % (
        report["word_length"]["median"], report["word_length"]["p90"],
        report["word_length"]["p99"], report["word_length"]["max"]))
    print(f"% textos > 512 palavras: {report['pct_words_gt_512']:.1f}%")
    if token_len is not None:
        print(f"% textos > 512 tokens (amostra): {report['pct_tokens_gt_512_sample']:.1f}%")
    print(f"\nRelatório: {out}")
    print(f"Figuras:   {eda_dir}")


if __name__ == "__main__":
    main()
