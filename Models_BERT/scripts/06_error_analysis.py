"""
06_error_analysis.py — Análise qualitativa de erros de alta confiança.

Extrai os exemplos onde o modelo errou com ALTA CONFIANÇA (ex.: predisse "Fato"
com 95% de certeza, mas era "Tese"). Esses casos são cientificamente valiosos:
revelam onde o modelo aprendeu padrões superficiais ou onde há ambiguidade
genuína no corpus.

Para papers de alto impacto em NLP, uma seção de análise de erros qualitativa
é frequentemente a parte mais citada — distingue contribuições técnicas de
contribuições linguísticas reais.

Uso:
    python scripts/06_error_analysis.py --exp roberta_crawlpt [--top 20] [--threshold 0.85]

Saída (em experiments/<exp>/error_analysis/):
    high_confidence_errors.json   — exemplos com texto, pred, true, proba
    error_analysis_report.md      — relatório em Markdown pronto para o paper
    confusion_pairs/              — pares de classe com mais erros
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_src():
    root = _root()
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def run(args):
    import numpy as np

    _add_src()
    from ftr import CLASS_NAMES, ID2LABEL
    from ftr.data import load_splits
    from ftr.env import get_paths
    from ftr.registry import collect_experiments

    paths = get_paths()

    # Encontra o experimento.
    runs = collect_experiments(paths.experiments)
    matching = [r for r in runs
                if r.get("experiment_id") == args.exp and r.get("status") == "completed"]
    if not matching:
        print(f"Experimento '{args.exp}' não encontrado ou não concluído. "
              "Use 04_consolidate.py para listar os disponíveis.")
        return
    # Usa a execução mais recente (última pasta).
    run_info = sorted(matching, key=lambda r: r.get("started_at", ""))[-1]
    exp_dir = paths.experiments / run_info["run_dir"]

    # Carrega probabilidades e rótulos do teste.
    npz_path = exp_dir / "metrics" / "test_predictions.npz"
    if not npz_path.exists():
        print("test_predictions.npz não encontrado. Certifique-se de que 03_evaluate.py foi rodado.")
        return
    npz = np.load(str(npz_path))
    y_proba = npz["proba"]
    y_true = npz["y_true"]
    if y_proba is None or y_true is None:
        print("Probabilidades não encontradas. Certifique-se de que 03_evaluate.py foi rodado.")
        return

    y_pred = y_proba.argmax(axis=1)
    y_conf = y_proba.max(axis=1)

    # Carrega textos do conjunto de teste.
    splits = load_splits(paths.data_processed)
    texts = splits["test"]["text"].tolist() if "test" in splits else None

    # Identifica erros de alta confiança.
    errors = []
    for i, (true, pred, conf) in enumerate(zip(y_true, y_pred, y_conf)):
        if true == pred:
            continue
        if conf < args.threshold:
            continue
        entry = {
            "index": int(i),
            "true_label": int(true),
            "true_class": ID2LABEL[int(true)],
            "pred_label": int(pred),
            "pred_class": ID2LABEL[int(pred)],
            "confidence": round(float(conf), 4),
            "probabilities": {ID2LABEL[j]: round(float(p), 4) for j, p in enumerate(y_proba[i])},
        }
        if texts:
            text = texts[i]
            entry["text_preview"] = text[:500] + ("…" if len(text) > 500 else "")
            entry["text_length_chars"] = len(text)
            entry["text_length_words"] = len(text.split())
        errors.append(entry)

    errors.sort(key=lambda e: e["confidence"], reverse=True)
    top_errors = errors[: args.top]

    # Estatísticas dos erros.
    from collections import Counter
    pair_counts = Counter(
        (ID2LABEL[e["true_label"]], ID2LABEL[e["pred_label"]]) for e in errors
    )

    print(f"\n=== Análise de Erros — {args.exp} ===")
    print(f"Total de erros de alta confiança (≥{args.threshold:.0%}): {len(errors)}")
    print(f"Pares de erro mais frequentes:")
    for (true_cls, pred_cls), count in pair_counts.most_common(6):
        print(f"  {true_cls} → pred {pred_cls}: {count}")

    # Salva resultados.
    out_dir = exp_dir / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "high_confidence_errors.json").write_text(
        json.dumps({"threshold": args.threshold, "n_errors": len(errors),
                    "pair_counts": {f"{t}→{p}": c for (t, p), c in pair_counts.most_common()},
                    "top_errors": top_errors},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Relatório Markdown pronto para o paper.
    report = _make_report(args.exp, run_info, errors, top_errors, pair_counts, args.threshold)
    (out_dir / "error_analysis_report.md").write_text(report, encoding="utf-8")

    print(f"\nArtefatos salvos em: {out_dir}")
    print(f"  high_confidence_errors.json")
    print(f"  error_analysis_report.md  ← copie para o paper")


def _make_report(exp_id, run_info, errors, top_errors, pair_counts, threshold):
    from ftr import ID2LABEL
    lines = [
        f"# Análise de Erros de Alta Confiança — `{exp_id}`",
        "",
        f"**Modelo:** {run_info.get('model_name', '?')}  ",
        f"**Threshold de confiança:** {threshold:.0%}  ",
        f"**Total de erros acima do threshold:** {len(errors)}  ",
        "",
        "## Padrões de Confusão",
        "",
        "| Classe Real | Classe Predita | Ocorrências |",
        "|---|---|---|",
    ]
    for (true_cls, pred_cls), count in pair_counts.most_common():
        lines.append(f"| {true_cls.capitalize()} | {pred_cls.capitalize()} | {count} |")
    lines += [
        "",
        "## Exemplos de Alta Confiança e Erro",
        "",
        "> Estes são os casos onde o modelo errou com maior certeza. "
        "São candidatos a discussão qualitativa no paper.",
        "",
    ]
    for i, e in enumerate(top_errors[:10], 1):
        lines += [
            f"### Erro #{i} — confiança {e['confidence']:.1%}",
            "",
            f"- **Rótulo real:** {e['true_class'].capitalize()}",
            f"- **Predição:** {e['pred_class'].capitalize()} "
            f"(conf. {e['confidence']:.1%})",
            f"- **Probabilidades:** " +
            ", ".join(f"{c}: {p:.1%}" for c, p in e["probabilities"].items()),
        ]
        if "text_preview" in e:
            lines += [
                f"- **Tamanho:** {e.get('text_length_words', '?')} palavras",
                "",
                "**Texto (preview):**",
                "",
                f"> {e['text_preview'].replace(chr(10), ' ').replace(chr(13), '')}",
            ]
        lines.append("")
    lines += [
        "## Perguntas para a Discussão do Paper",
        "",
        "- O modelo confundiu porque o texto continha marcadores linguísticos de ambas as classes?",
        "- Havia ambiguidade que até um jurista humano teria dificuldade em classificar?",
        "- Os erros se concentram em textos de determinado comprimento ou domínio?",
        "- As confusões são assimétricas (ex.: o modelo confunde Tese→Fato muito mais que Fato→Tese)?",
        "- Os erros de alta confiança são distribuídos aleatoriamente ou clustered em regiões específicas?",
    ]
    return "\n".join(lines)


def main():
    _add_src()
    from ftr.registry import collect_experiments
    from ftr.env import get_paths

    ap = argparse.ArgumentParser(description="Análise de erros de alta confiança.")
    ap.add_argument("--exp", type=str, required=True,
                    help="experiment_id a analisar (ex.: roberta_crawlpt)")
    ap.add_argument("--top", type=int, default=20,
                    help="Número de erros a detalhar no relatório (padrão: 20)")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="Confiança mínima para considerar um erro (padrão: 0.85)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
