"""
Carregamento, limpeza e particionamento do dataset Fato/Tese/Ruído.

Fluxo:
    load_raw -> clean -> assign_doc_groups -> split (train/val/test)

SPLIT POR DOCUMENTO (anti-leakage)
===================================
Os textos do dataset são fragmentos de petições judiciais. Um split aleatório
por linha coloca fragmentos do mesmo processo em treino E teste — o modelo
aprende o estilo do advogado/escritório e "reconhece" no teste, inflando as
métricas artificialmente (data leakage por fragmentação).

A solução: agrupar fragmentos do mesmo documento via identificadores únicos
presentes no texto (OAB do advogado + email profissional), e usar
``GroupShuffleSplit`` para garantir que TODOS os fragmentos de um grupo vão
para o mesmo conjunto. Fragmentos sem identificador detectável são tratados
como grupos singleton (split seguro).

Referência: Gorman & Bedrick (2019), "We Need to Talk About Standard Splits".
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import ID2LABEL, LABEL2ID

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    strategy: str = "train_val_test"   # ou "train_test"
    test_size: float = 0.10
    val_size: float = 0.10
    stratify: bool = True
    random_state: int = 42
    # Controle do split por documento (anti-leakage)
    group_by_document: bool = True     # False = split aleatório (compatibilidade)

    @classmethod
    def from_dict(cls, d: dict | None) -> "SplitConfig":
        d = d or {}
        return cls(
            strategy=d.get("strategy", "train_val_test"),
            test_size=d.get("test_size", 0.10),
            val_size=d.get("val_size", 0.10),
            stratify=d.get("stratify", True),
            random_state=d.get("random_state", 42),
            group_by_document=d.get("group_by_document", True),
        )


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def load_raw(
    path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
) -> pd.DataFrame:
    """Carrega o dataset bruto (JSON list-of-records) num DataFrame."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"Esperava uma lista JSON em {path}, obtive {type(data).__name__}"
        )
    df = pd.DataFrame(data)
    missing = {text_column, label_column} - set(df.columns)
    if missing:
        raise ValueError(
            f"Colunas ausentes no dataset: {missing}. Colunas: {list(df.columns)}"
        )
    return df


# ---------------------------------------------------------------------------
# Limpeza
# ---------------------------------------------------------------------------
def clean(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    drop_na: bool = True,
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Limpa o DataFrame e retorna (df_limpo, relatório_de_limpeza)."""
    report: dict = {"n_input": int(len(df))}
    out = df.copy()

    if drop_na:
        before = len(out)
        out = out.dropna(subset=[text_column, label_column]).reset_index(drop=True)
        report["dropped_na"] = int(before - len(out))

    # Remove textos vazios após strip.
    before = len(out)
    out = out[out[text_column].astype(str).str.strip().astype(bool)].reset_index(drop=True)
    report["dropped_empty"] = int(before - len(out))

    if drop_duplicates:
        before = len(out)
        # Duplicatas removidas ANTES do split — evita data leakage por cópia exata.
        out = out.drop_duplicates(subset=[text_column]).reset_index(drop=True)
        report["dropped_duplicates"] = int(before - len(out))

    # Normaliza labels para int e valida o mapeamento canônico.
    out[label_column] = out[label_column].astype(int)
    observed = set(out[label_column].unique().tolist())
    expected = set(ID2LABEL.keys())
    report["labels_observed"] = sorted(observed)
    report["labels_expected"] = sorted(expected)
    report["labels_ok"] = observed.issubset(expected)
    if not observed.issubset(expected):
        raise ValueError(
            f"Rótulos inesperados {observed - expected}. "
            f"Esperado {expected} ({ID2LABEL})."
        )

    # Garante a coluna label_text coerente com o mapeamento canônico.
    out["label_text"] = out[label_column].map(ID2LABEL)

    report["n_output"] = int(len(out))
    report["class_distribution"] = {
        ID2LABEL[k]: int(v)
        for k, v in out[label_column].value_counts().sort_index().items()
    }
    return out, report


def maybe_sample(
    df: pd.DataFrame,
    frac: float,
    label_column: str = "label",
    random_state: int = 42,
) -> pd.DataFrame:
    """Amostragem estratificada por classe (para reproduzir subsets originais)."""
    if frac is None or frac >= 1.0:
        return df
    # Sort explícito antes do groupby para garantir ordem estável entre runs.
    df_sorted = df.sort_values([label_column, df.columns[0]]).reset_index(drop=True)
    sampled = (
        df_sorted.groupby(label_column, group_keys=False)[df_sorted.columns.tolist()]
        .apply(lambda g: g.sample(frac=frac, random_state=random_state))
        .reset_index(drop=True)
    )
    return sampled


# ---------------------------------------------------------------------------
# Agrupamento por documento (anti-leakage)
# ---------------------------------------------------------------------------

# Identificadores específicos o suficiente para identificar um processo/advogado.
# OAB com UF é praticamente único por advogado; email profissional idem.
# NÃO usamos telefone ou CNPJ — muito risco de super-conectar docs distintos.
_GROUP_PATTERNS = {
    "oab":   re.compile(r"OAB\s*/?\s*([A-Z]{2})\s*\.?\s*(\d{4,6})\b", re.I),
    "email": re.compile(r"([a-z0-9._%+\-]{5,}@[a-z0-9.\-]+\.[a-z]{2,})", re.I),
}

# Emails genéricos que aparecem em milhares de docs (não identificam um único processo).
_GENERIC_EMAIL_LOCALS = frozenset({
    "contato", "info", "adm", "admin", "administrativo", "secretaria",
    "atendimento", "recepcao", "financeiro", "juridico", "tj", "jus",
    "ouvidoria", "tribunal", "comarca",
})


def _extract_doc_identifiers(text: str) -> set[str]:
    """Extrai identificadores únicos de documento a partir do texto."""
    ids: set[str] = set()
    # OAB/UF NNNNNN — ex.: "OAB/GO 32827"
    for m in _GROUP_PATTERNS["oab"].finditer(text):
        uf, num = m.group(1).upper(), m.group(2)
        ids.add(f"oab:{uf}{num}")
    # Email profissional
    for m in _GROUP_PATTERNS["email"].finditer(text):
        addr = m.group(1).lower().strip()
        local = addr.split("@")[0]
        if len(local) >= 5 and local not in _GENERIC_EMAIL_LOCALS:
            ids.add(f"email:{addr}")
    return ids


def assign_doc_groups(df: pd.DataFrame, text_column: str = "text") -> pd.Series:
    """Atribui um ``doc_group_id`` inteiro a cada linha do DataFrame.

    Linhas sem identificadores detectável recebem um grupo único (singleton),
    garantindo que nunca causem vazamento.

    Algoritmo: Union-Find sobre identificadores compartilhados. Dois textos que
    compartilham OAB ou email são fundidos no mesmo grupo — assumimos que vieram
    do mesmo processo judicial.

    Complexidade: O(n * k) onde k = nº de identificadores por texto (pequeno).
    """
    n = len(df)
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    # Mapeia identificador -> índices dos textos que o contêm.
    id_to_indices: dict[str, list[int]] = defaultdict(list)
    texts = df[text_column].tolist()
    for i, text in enumerate(texts):
        for pid in _extract_doc_identifiers(str(text)):
            id_to_indices[pid].append(i)

    # Só une se o identificador aparecer em poucos docs (≤ 50).
    # Identificadores que aparecem em 100+ docs são provavelmente boilerplate
    # de tribunal/comarca — não identificam um processo específico.
    n_connected = 0
    for pid, indices in id_to_indices.items():
        if 2 <= len(indices) <= 50:
            for j in range(1, len(indices)):
                union(indices[0], indices[j])
            n_connected += len(indices)

    # Converte roots para IDs compactos (0-based).
    roots = [find(i) for i in range(n)]
    unique_roots = sorted(set(roots))
    root_to_gid = {r: gid for gid, r in enumerate(unique_roots)}
    group_ids = [root_to_gid[r] for r in roots]

    n_groups = len(unique_roots)
    n_singletons = sum(1 for r in unique_roots if roots.count(r) == 1)
    n_multi = n_groups - n_singletons
    pct_connected = n_connected / n * 100 if n > 0 else 0

    logger.info(
        "Agrupamento por documento: %d grupos (%d singletons, %d multi-fragmento); "
        "%d textos linkados (%.1f%%)",
        n_groups, n_singletons, n_multi, n_connected, pct_connected,
    )

    return pd.Series(group_ids, index=df.index, name="doc_group_id")


def _estimate_leakage(groups: pd.Series, test_mask: pd.Series) -> float:
    """Estima % do teste com fragmentos do mesmo grupo no treino (para logging)."""
    df_tmp = pd.DataFrame({"group": groups, "in_test": test_mask})
    grp = df_tmp.groupby("group")["in_test"]
    has_test = grp.any()
    has_train = (~df_tmp.set_index("group")["in_test"]).groupby(level=0).any()
    leakage_groups = has_test.index[has_test & has_train.reindex(has_test.index, fill_value=False)]
    n_leaked = df_tmp[df_tmp["group"].isin(leakage_groups) & df_tmp["in_test"]].shape[0]
    n_test = test_mask.sum()
    return n_leaked / n_test * 100 if n_test > 0 else 0.0


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def split_dataset(
    df: pd.DataFrame,
    cfg: SplitConfig,
    label_column: str = "label",
    text_column: str = "text",
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Divisão em train/val/test respeitando grupos de documento (anti-leakage).

    Quando ``cfg.group_by_document=True`` (padrão), usa ``GroupShuffleSplit``
    para garantir que todos os fragmentos do mesmo processo vão para o mesmo
    conjunto. A estratificação é feita sobre o rótulo majoritário de cada grupo.

    Retorna ``(splits_dict, split_report_dict)``.
    """
    if cfg.group_by_document:
        return _split_by_document(df, cfg, label_column, text_column)
    return _split_random(df, cfg, label_column), {}


def _split_by_document(
    df: pd.DataFrame,
    cfg: SplitConfig,
    label_column: str,
    text_column: str,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Split por grupo de documento (GroupShuffleSplit)."""
    from sklearn.model_selection import GroupShuffleSplit

    groups = assign_doc_groups(df, text_column)
    df = df.copy()
    df["_doc_group"] = groups.values

    # Label majoritário por grupo (para estratificação aproximada).
    # Em caso de empate, usa o label de menor índice (sort garante determinismo).
    group_label = (
        df.groupby("_doc_group")[label_column]
        .agg(lambda x: sorted(x.mode().tolist())[0])
    )
    group_ids = group_label.index.values
    group_labels = group_label.values

    # Separa teste.
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state
    )
    idx_train_val, idx_test = next(
        gss_test.split(group_ids, group_labels, groups=group_ids)
    )
    test_groups = set(group_ids[idx_test])

    report = {}

    if cfg.strategy == "train_test":
        train_mask = ~df["_doc_group"].isin(test_groups)
        test_mask = df["_doc_group"].isin(test_groups)
        leakage = _estimate_leakage(df["_doc_group"], test_mask)
        report["leakage_pct"] = leakage
        if leakage > 1.0:
            warnings.warn(
                f"Vazamento residual estimado: {leakage:.1f}% do teste. "
                "Verifique os grupos detectados.",
                UserWarning,
                stacklevel=3,
            )
        splits = {
            "train": df[train_mask].drop(columns=["_doc_group"]).reset_index(drop=True),
            "test":  df[test_mask].drop(columns=["_doc_group"]).reset_index(drop=True),
        }
        return splits, report

    # Separa validação de dentro do train_val.
    train_val_groups = group_ids[idx_train_val]
    train_val_labels = group_labels[idx_train_val]
    rel_val = cfg.val_size / (1.0 - cfg.test_size)
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=rel_val, random_state=cfg.random_state
    )
    idx_tr, idx_val = next(
        gss_val.split(train_val_groups, train_val_labels, groups=train_val_groups)
    )
    val_groups = set(train_val_groups[idx_val])
    train_groups = set(train_val_groups[idx_tr])

    train_mask = df["_doc_group"].isin(train_groups)
    val_mask   = df["_doc_group"].isin(val_groups)
    test_mask  = df["_doc_group"].isin(test_groups)

    # Verifica leakage residual (deve ser 0 com GroupShuffleSplit correto).
    leakage = _estimate_leakage(df["_doc_group"], test_mask)
    report["leakage_pct"] = leakage
    report["n_doc_groups"] = int(df["_doc_group"].nunique())
    report["split_method"] = "group_by_document"
    if leakage > 0.5:
        warnings.warn(
            f"Vazamento residual estimado: {leakage:.1f}% do teste. "
            "Isso não deveria acontecer com GroupShuffleSplit — verifique os dados.",
            UserWarning,
            stacklevel=3,
        )
    else:
        logger.info("Vazamento estimado: %.2f%% (esperado: ~0%%)", leakage)

    splits = {
        "train": df[train_mask].drop(columns=["_doc_group"]).reset_index(drop=True),
        "val":   df[val_mask].drop(columns=["_doc_group"]).reset_index(drop=True),
        "test":  df[test_mask].drop(columns=["_doc_group"]).reset_index(drop=True),
    }
    return splits, report


def _split_random(
    df: pd.DataFrame,
    cfg: SplitConfig,
    label_column: str,
) -> dict[str, pd.DataFrame]:
    """Split aleatório (mantido para compatibilidade com experimentos originais)."""
    from sklearn.model_selection import train_test_split

    warnings.warn(
        "group_by_document=False: usando split aleatório. "
        "Pode haver data leakage por fragmentação de documentos.",
        UserWarning,
        stacklevel=3,
    )
    strat = df[label_column] if cfg.stratify else None
    train_val, test = train_test_split(
        df, test_size=cfg.test_size, stratify=strat, random_state=cfg.random_state
    )
    if cfg.strategy == "train_test":
        return {
            "train": train_val.reset_index(drop=True),
            "test": test.reset_index(drop=True),
        }
    rel_val = cfg.val_size / (1.0 - cfg.test_size)
    strat_tv = train_val[label_column] if cfg.stratify else None
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=strat_tv, random_state=cfg.random_state
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def split_report(splits: dict[str, pd.DataFrame], label_column: str = "label") -> dict:
    """Estatísticas por split (tamanho e distribuição de classes)."""
    rep = {}
    for name, part in splits.items():
        rep[name] = {
            "n": int(len(part)),
            "class_distribution": {
                ID2LABEL[k]: int(v)
                for k, v in part[label_column].value_counts().sort_index().items()
            },
        }
    return rep


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------
def save_splits(
    splits: dict[str, pd.DataFrame],
    out_dir: str | Path,
    fmt: str = "parquet",
) -> dict:
    """Salva os splits em ``out_dir`` e retorna os caminhos.

    Preferimos **Parquet** (tipado, compacto). Se ``pyarrow``/``fastparquet`` não
    estiverem disponíveis, caímos para **JSON Lines** (NÃO CSV): textos jurídicos
    contêm aspas, vírgulas e quebras de linha que corrompem CSV no round-trip.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    used_fmt = fmt
    for name, part in splits.items():
        if used_fmt == "parquet":
            try:
                p = out_dir / f"{name}.parquet"
                part.to_parquet(p, index=False)
                paths[name] = str(p)
                continue
            except Exception as e:
                logger.warning(
                    "Parquet indisponível (%s) — usando JSONL como fallback.", e
                )
                used_fmt = "jsonl"
        p = out_dir / f"{name}.jsonl"
        part.to_json(p, orient="records", lines=True, force_ascii=False)
        paths[name] = str(p)
    logger.info("Splits salvos em %s (formato: %s)", out_dir, used_fmt)
    return paths


def load_splits(out_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Carrega splits previamente salvos (parquet, jsonl ou csv legado)."""
    out_dir = Path(out_dir)
    splits = {}
    for name in ("train", "val", "test"):
        pq   = out_dir / f"{name}.parquet"
        jl   = out_dir / f"{name}.jsonl"
        csv  = out_dir / f"{name}.csv"
        if pq.exists():
            splits[name] = pd.read_parquet(pq)
        elif jl.exists():
            splits[name] = pd.read_json(jl, orient="records", lines=True)
        elif csv.exists():
            splits[name] = pd.read_csv(csv)
    return splits
