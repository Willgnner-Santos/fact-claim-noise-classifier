"""
05_ensemble.py — Ensemble soft-voting dos modelos treinados.

Carrega os logits/probabilidades do conjunto de teste de cada experimento
concluído, combina-os por média ponderada (soft-voting) e avalia o resultado
com todas as métricas do pipeline. Tipicamente gera a melhor performance
individual e costuma ser o destaque da seção de resultados do paper.

Uso:
    python scripts/05_ensemble.py [--exp exp1 exp2 ...] [--top 3] [--weights ...]

Exemplos:
    # Ensemble dos 3 melhores modelos por f1_macro (padrão):
    python scripts/05_ensemble.py --top 3

    # Ensemble de modelos específicos:
    python scripts/05_ensemble.py --exp roberta_lexpt roberta_crawlpt xlmr

    # Com pesos manuais (na mesma ordem dos experimentos):
    python scripts/05_ensemble.py --exp roberta_lexpt roberta_crawlpt --weights 0.6 0.4
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


def _load_test_predictions(exp_dir: Path):
    """Carrega probabilidades e rótulos do teste (salvo em test_predictions.npz)."""
    import numpy as np

    p = exp_dir / "metrics" / "test_predictions.npz"
    if p.exists():
        data = np.load(str(p))
        return data["proba"], data["y_true"]
    return None, None


def run(args):
    import numpy as np

    _add_src()
    from ftr.env import get_paths
    from ftr.metrics import compute_all_metrics, flatten_metrics
    from ftr.registry import collect_experiments
    from ftr.viz import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    paths = get_paths()

    # Coleta experimentos concluídos com probabilidades salvas.
    runs = [r for r in collect_experiments(paths.experiments) if r.get("status") == "completed"]

    if not runs:
        print("Nenhum experimento concluído encontrado em experiments/.")
        return

    # Seleciona os experimentos para o ensemble.
    if args.exp:
        selected = [r for r in runs if r["experiment_id"] in args.exp]
        if len(selected) != len(args.exp):
            found = {r["experiment_id"] for r in selected}
            missing = [e for e in args.exp if e not in found]
            print(f"Aviso: experimentos não encontrados: {missing}")
    else:
        # Top-N por f1_macro.
        from ftr.metrics import flatten_metrics

        scored = []
        for r in runs:
            if r.get("test_metrics"):
                f1 = flatten_metrics(r["test_metrics"]).get("f1_macro", 0.0) or 0.0
                scored.append((f1, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [r for _, r in scored[: args.top]]
        print(f"Top-{args.top} por f1_macro: {[r['experiment_id'] for r in selected]}")

    if len(selected) < 2:
        print("Ensemble requer pelo menos 2 modelos. Abortando.")
        return

    # Carrega probabilidades.
    probas_list, labels_list, names = [], [], []
    for r in selected:
        exp_dir = paths.experiments / r["run_dir"]
        p, y = _load_test_predictions(exp_dir)
        if p is None or y is None:
            print(f"  Sem probabilidades salvas para '{r['experiment_id']}' — pulando.")
            continue
        probas_list.append(p)
        labels_list.append(y)
        names.append(r["experiment_id"])

    if len(probas_list) < 2:
        print("Probabilidades não encontradas. Certifique-se de que 03_evaluate.py foi rodado.")
        return

    # Verifica que os rótulos são consistentes.
    y_true = labels_list[0]
    if not all(np.array_equal(y, y_true) for y in labels_list[1:]):
        print("ERRO: rótulos de teste inconsistentes entre experimentos.")
        return

    # Soft-voting: média ponderada das probabilidades.
    weights = args.weights if args.weights else [1.0] * len(probas_list)
    if len(weights) != len(probas_list):
        weights = [1.0] * len(probas_list)
        print("Aviso: número de pesos diferente do número de modelos. Usando pesos iguais.")
    w = np.array(weights, dtype=float)
    w /= w.sum()
    ensemble_proba = sum(w_i * p for w_i, p in zip(w, probas_list))
    y_pred = ensemble_proba.argmax(axis=1)

    # Métricas do ensemble.
    metrics = compute_all_metrics(y_true, y_pred, y_proba=ensemble_proba)
    flat = flatten_metrics(metrics)
    print(f"\n=== Ensemble ({', '.join(names)}) ===")
    for k in ("f1_macro", "f1_weighted", "accuracy", "mcc", "roc_auc_macro_ovr"):
        v = flat.get(k) or (flat.get("probabilistic") or {}).get(k)
        if v is not None:
            print(f"  {k}: {v:.4f}")

    # Salva resultados.
    ensemble_id = "ensemble_" + "_".join(n[:8] for n in names)
    out_dir = paths.results / "ensemble" / ensemble_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "members.json").write_text(
        json.dumps({"members": names, "weights": w.tolist()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Salva métricas em JSON e CSV.
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    (metrics_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
        encoding="utf-8",
    )
    pd.DataFrame([flatten_metrics(metrics)]).to_csv(
        metrics_dir / "test_metrics_flat.csv", index=False, encoding="utf-8"
    )

    # Matriz de confusão usando a assinatura correta de plot_confusion_matrix(cm, class_names, out_dir).
    try:
        from ftr import CLASS_NAMES_PT, CLASS_NAMES_EN
        import numpy as np
        cm_abs = sk_confusion_matrix(y_true, y_pred)
        cm_sum = cm_abs.sum(axis=1, keepdims=True)
        cm_norm = np.where(cm_sum > 0, cm_abs.astype(float) / cm_sum, 0.0)
        for lang, names_map in [("pt", CLASS_NAMES_PT), ("en", CLASS_NAMES_EN)]:
            class_names_lang = [names_map[c] for c in ["tese", "fato", "ruido"]]
            plot_confusion_matrix(cm_abs,  class_names_lang, out_dir, normalized=False, lang=lang)
            plot_confusion_matrix(cm_norm, class_names_lang, out_dir, normalized=True,  lang=lang)
    except Exception as e:
        print(f"  Aviso: não foi possível salvar matriz de confusão: {e}")

    print(f"\nResultados salvos em: {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Ensemble soft-voting dos modelos treinados.")
    ap.add_argument("--exp", nargs="+", help="IDs de experimentos a combinar")
    ap.add_argument("--top", type=int, default=3, help="Top-N modelos por f1_macro (padrão: 3)")
    ap.add_argument("--weights", nargs="+", type=float, help="Pesos para cada modelo (mesma ordem)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
