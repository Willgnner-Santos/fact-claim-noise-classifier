"""
Registro de experimentos: identificador único, pastas isoladas e consolidação.

Cada execução de treino/avaliação cria uma pasta isolada em ``experiments/`` com
nome ``<experiment_id>__<timestamp>__<run_id>``, garantindo que **resultados
antigos nunca sejam sobrescritos**. Dentro dela ficam:

    config_snapshot.yaml   — config exato usado
    env.json               — versões de libs, hardware, seed
    status.json            — estado (running/completed/failed), tempos, erros
    metrics/               — métricas (json/csv) de val e teste
    figures/               — gráficos do experimento
    logs/                  — logs de execução
    checkpoints/           — pesos do modelo (Trainer output_dir)

A função :func:`consolidate` varre ``experiments/`` e produz o JSON consolidado
e a tabela comparativa em ``results/``.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def new_run_id() -> str:
    """Gera um identificador único curto de execução."""
    return uuid.uuid4().hex[:8]


def make_experiment_dir(
    experiments_root: str | Path,
    experiment_id: str,
    timestamp: str,
    run_id: str | None = None,
) -> "ExperimentDir":
    """Cria (e retorna) a pasta isolada de um experimento.

    Args:
        timestamp: string de data/hora gerada por quem chama (ex.: "20260601_1530").
            Mantido como argumento explícito para que o módulo permaneça
            determinístico/serializável.
    """
    run_id = run_id or new_run_id()
    safe_id = re.sub(r"[^A-Za-z0-9_\-]", "_", experiment_id)
    name = f"{safe_id}__{timestamp}__{run_id}"
    path = Path(experiments_root) / name
    exp = ExperimentDir(root=path, experiment_id=experiment_id, run_id=run_id, timestamp=timestamp)
    exp.ensure()
    return exp


@dataclass
class ExperimentDir:
    """Representa a pasta de um experimento e seus subdiretórios."""

    root: Path
    experiment_id: str
    run_id: str
    timestamp: str
    metrics: Path = field(init=False)
    figures: Path = field(init=False)
    logs: Path = field(init=False)
    checkpoints: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.metrics = self.root / "metrics"
        self.figures = self.root / "figures"
        self.logs = self.root / "logs"
        self.checkpoints = self.root / "checkpoints"

    def ensure(self) -> "ExperimentDir":
        for p in (self.root, self.metrics, self.figures, self.logs, self.checkpoints):
            p.mkdir(parents=True, exist_ok=True)
        return self

    # -- escrita de artefatos ------------------------------------------------
    def write_json(self, name: str, obj: Any) -> Path:
        p = self.root / name
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
        return p

    def write_status(self, status: dict) -> Path:
        return self.write_json("status.json", status)


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


# ---------------------------------------------------------------------------
# Consolidação
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect_experiments(experiments_root: str | Path) -> list[dict]:
    """Coleta o resumo de cada experimento encontrado em ``experiments/``."""
    root = Path(experiments_root)
    runs = []
    if not root.exists():
        return runs
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        status = _read_json(d / "status.json") or {}
        test_metrics = _read_json(d / "metrics" / "test_metrics.json")
        env = _read_json(d / "env.json") or {}
        entry = {
            "run_dir": d.name,
            "experiment_id": status.get("experiment_id"),
            "model_name": status.get("model_name"),
            "is_reproduction": status.get("is_reproduction"),
            "status": status.get("status"),
            "started_at": status.get("started_at"),
            "finished_at": status.get("finished_at"),
            "duration_seconds": status.get("duration_seconds"),
            "error": status.get("error"),
            "artifacts": status.get("artifacts"),
            "epoch_metrics": status.get("epoch_metrics"),
            "env": {"packages": env.get("packages"), "hardware": env.get("hardware")},
            "test_metrics": test_metrics,
        }
        runs.append(entry)
    return runs


def consolidate(experiments_root: str | Path, results_dir: str | Path) -> dict:
    """Gera ``consolidated_results.json`` + ``comparison_table.{csv,xlsx}``.

    Retorna um dicionário com os caminhos gerados e a tabela comparativa (lista
    de linhas). Mantém apenas a MELHOR execução por ``experiment_id`` na tabela
    comparativa (por f1_macro), mas o JSON consolidado retém todas as execuções.
    """
    import pandas as pd

    from .metrics import flatten_metrics

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_experiments(experiments_root)

    consolidated = {
        "n_runs": len(runs),
        "runs": runs,
    }
    cpath = results_dir / "consolidated_results.json"
    with cpath.open("w", encoding="utf-8") as f:
        json.dump(consolidated, f, ensure_ascii=False, indent=2, default=_json_default)

    # Tabela comparativa (uma linha por execução concluída, com métricas planas).
    rows = []
    for r in runs:
        if not r.get("test_metrics"):
            continue
        flat = flatten_metrics(r["test_metrics"])
        row = {
            "experiment_id": r.get("experiment_id"),
            "model_name": r.get("model_name"),
            "is_reproduction": r.get("is_reproduction"),
            "run_dir": r.get("run_dir"),
            "status": r.get("status"),
            "duration_seconds": r.get("duration_seconds"),
            **flat,
        }
        rows.append(row)

    table_paths = {}
    if rows:
        df = pd.DataFrame(rows).sort_values(
            by="f1_macro", ascending=False, na_position="last"
        )
        csv_path = results_dir / "comparison_table.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        table_paths["csv"] = str(csv_path)
        try:
            xlsx_path = results_dir / "comparison_table.xlsx"
            df.to_excel(xlsx_path, index=False)
            table_paths["xlsx"] = str(xlsx_path)
        except Exception:
            pass

    multi_seed = aggregate_multi_seed(runs)

    return {
        "consolidated_json": str(cpath),
        "comparison_table": table_paths,
        "n_runs": len(runs),
        "n_with_metrics": len(rows),
        "multi_seed_summary": multi_seed,
    }


def aggregate_multi_seed(runs: list[dict]) -> dict[str, dict]:
    """Agrega múltiplas execuções do mesmo ``experiment_id`` em mean ± std.

    Retorna dicionário ``{experiment_id: {metric: {"mean": x, "std": y, "n": k}}}``.
    Usado para reportar resultados robustos no paper (evita "lucky split").
    Ignora execuções sem métricas ou com status != "completed".
    """
    import numpy as np

    from .metrics import flatten_metrics

    by_exp: dict[str, list[dict]] = {}
    for r in runs:
        if r.get("status") != "completed" or not r.get("test_metrics"):
            continue
        eid = r.get("experiment_id") or "unknown"
        by_exp.setdefault(eid, []).append(flatten_metrics(r["test_metrics"]))

    summary: dict[str, dict] = {}
    for eid, flat_list in by_exp.items():
        if len(flat_list) < 2:
            # Só 1 run: std=0.0 (não None) para compatibilidade com downstream
            # (plotagem, comparação) que espera float.
            m = flat_list[0]
            summary[eid] = {
                k: {"mean": float(v), "std": 0.0, "n": 1}
                for k, v in m.items()
                if isinstance(v, (int, float))
            }
            continue
        # Coleta todos os valores numéricos presentes em TODAS as runs.
        keys = set(flat_list[0]) & set.intersection(*(set(f) for f in flat_list))
        agg: dict[str, dict] = {}
        for k in sorted(keys):
            vals = [f[k] for f in flat_list if isinstance(f.get(k), (int, float))]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            agg[k] = {
                "mean": round(float(arr.mean()), 6),
                "std": round(float(arr.std(ddof=1)), 6),  # std amostral
                "n": len(vals),
                "values": [round(float(v), 6) for v in vals],
            }
        summary[eid] = agg

    return summary
