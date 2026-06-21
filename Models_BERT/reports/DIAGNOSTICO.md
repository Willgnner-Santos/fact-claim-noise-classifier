# Diagnóstico dos Experimentos Originais

> Resultado da inspeção dos notebooks em `Notebooks/legacy/`, do dataset e da
> planilha de métricas. Esta é a base de referência para a reprodução fiel.

## 1. Dataset (`Data/dataset_FINAL.json`)

| Propriedade | Valor |
|---|---|
| Formato | Lista JSON de registros (`text`, `label`, `label_text`) |
| Tamanho | 259 MB |
| Registros brutos | **26.781** |
| Registros após limpeza (dropna + dedup) | **25.482** |
| Texto nulo | 1 |
| Duplicatas exatas removidas | 1.298 |
| Mapeamento de rótulos | **0 = tese, 1 = fato, 2 = ruido** |

**Balanceamento.** No bruto é perfeitamente balanceado (8.927/classe). Após
remover duplicatas exatas, fica levemente desbalanceado — as duplicatas se
concentravam em algumas classes:

| Classe | Após limpeza |
|---|---|
| tese | 8.039 |
| fato | 8.517 |
| ruido | 8.926 |

**Tamanho dos textos (em palavras).** mediana ≈ 815, p90 ≈ 3.485, p99 ≈ 7.339,
máx ≈ 45.786. **64,5% dos textos têm mais de 512 palavras** → a maioria excede o
limite dos encoders padrão. Isso motiva a estratégia long-context (Longformer +
janela deslizante) e justifica métricas sensíveis a desbalanceamento.

## 2. Notebooks originais

Todos usam o HuggingFace `Trainer`, otimizador **AdamW** (`weight_decay=0.01`),
**max_length=512** com **truncação simples**, split `train_test_split` **80/20
estratificado** (`random_state=42`), e **accuracy** como métrica de seleção. Não
havia conjunto de validação separado (o teste era usado como `eval_dataset`).
Nenhum aplicava anonimização ou limpeza textual além de `str(t)`.

| Notebook (legacy) | Modelo (checkpoint HF) | Épocas | Batch | LR | Early Stop | Subset | Acc. relatada |
|---|---|---|---|---|---|---|---|
| `RoBERTaLexPT-base.ipynb` | `eduagarcia/RoBERTaLexPT-base` | 10 | 32 | 2e-5 | patience=2 | 100% | ~0,9909 |
| `RoBERTaCrawlPT-base.ipynb` | `eduagarcia/RoBERTaCrawlPT-base` | 10 | 32 | 2e-5 | patience=2 | 100% | ~0,9925 |
| `neuralmindbert-large-...ipynb` | `neuralmind/bert-large-portuguese-cased` | 10 | 8 | 2e-5 | importado, **inativo** | **50%** | ~0,9866 |
| `base-multilingual_30 (1).ipynb` | `bert-base-multilingual-cased` | 10 | 8 | default (≈5e-5) | patience=2 | **30%** | upload HF Hub |
| `training (1).ipynb` | `bert-base-multilingual-cased` (template) | 4 | 8 | 2e-5 | não | balance manual | — |

**Observações relevantes:**
- O notebook `training (1).ipynb` é um **template** com `microsoft/mdeberta-v3-base`,
  `FacebookAI/xlm-roberta-base` e `answerdotai/ModernBERT-base` **comentados** —
  daí a escolha dos modelos novos.
- Os caminhos eram absolutos da máquina original (`C:\Users\anonymousferreira\...`)
  e o dataset era lido como **CSV** (`dataset_FINAL.csv`); aqui padronizamos para
  o JSON existente e caminhos portáveis.

## 3. Planilha `Métricas_Avaliação_Modelos.xlsx`

Métricas solicitadas (todas implementadas em `src/ftr/metrics.py`):

| Tipo | Métrica |
|---|---|
| Essencial | Accuracy |
| Essencial | Precision/Recall/F1 por classe + support |
| Principal | **Macro-F1** |
| Complementar | Weighted-F1 |
| Complementar forte | Balanced Accuracy |
| Forte para artigo | MCC (Matthews) |
| Análise visual | Matriz de confusão normalizada |
| Probabilística | ROC-AUC macro / OvR |
| Desbalanceamento | PR-AUC / Average Precision |
| Calibração | Log Loss |
| Calibração | Brier Score |
| Concordância | Cohen's Kappa |

## 4. Ambiente original (libs)

`transformers 4.48.3`, `datasets 3.3.0`, `scikit-learn 1.6.1`, `pandas 2.2.3`,
`numpy 2.2.3`, `evaluate` — refletidos em `requirements.txt`.
