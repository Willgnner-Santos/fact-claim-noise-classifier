# Pipeline LLM — Fato / Tese / Ruído

Benchmark length-aware para classificação jurídica em português.

## Estrutura

```
LLMs/
├── configs/
│   ├── _base_llm.yaml          # Config base herdada por todos
│   ├── claude_sonnet_46.yaml   # Elite / grande
│   ├── qwen3_32b.yaml          # Forte / médio
│   ├── mistral_large_3.yaml    # Top / grande
│   ├── gemma_3_27b.yaml        # Bom / médio
│   ├── deepseek_v32.yaml       # Top / grande
│   ├── claude_3_haiku.yaml     # Leve / pequeno
│   ├── magistral_small_2509.yaml # Forte / médio
│   └── sabia4.yaml             # Maritaca / brasileiro
├── ftr_llm/                    # Pacote Python do pipeline
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py              # Reutiliza src/ftr/anonymize + preprocess
│   ├── prompts.py              # zero_shot, few_shot, evidence_constrained, decomposed
│   ├── bedrock_client.py       # Retry + fallback + custo
│   ├── maritaca_client.py      # Sabiá-4
│   ├── parser.py               # JSON parser robusto
│   ├── checkpoint.py           # JSONL checkpoint por linha
│   ├── runner.py               # Loop principal com retomada
│   ├── evaluate.py             # Métricas + análise por bin
│   └── viz.py                  # Gráficos PT e EN
├── scripts/
│   ├── 00_prepare_dataset.py   # Pré-processa + anonimiza + split
│   ├── 01_smoke_test_models.py # Testa todos os modelos com "oi"
│   ├── 02_run_experiment.py    # Roda 1 modelo × N estratégias
│   └── 03_consolidate_results.py # Consolida + gráficos finais
├── data/
│   ├── splits/                 # train.json, val.json, test.json
│   ├── checkpoints/            # JSONL por modelo × estratégia
│   ├── results/                # JSON + CSV de métricas
│   └── reports/                # Gráficos PT e EN
├── bedrock_modelos_refatorado.json
├── .env
└── requirements_llm.txt
```

## Quick start

```bash
# 1. Instalar dependências
pip install -r requirements_llm.txt

# 2. Preparar dataset (anonimiza + split — só precisa rodar uma vez)
python scripts/00_prepare_dataset.py

# 3. Testar se os modelos estão acessíveis
python scripts/01_smoke_test_models.py

# 4. Validação rápida do pipeline (30 exemplos, 1 modelo, 1 estratégia)
python scripts/02_run_experiment.py --model claude_3_haiku --sample-n 30 --strategies zero_shot

# 5. Rodar experimento completo no val set
python scripts/02_run_experiment.py --model claude_3_haiku --split val

# 6. Rodar no test set (avaliação final — só depois de ajustar prompts no val)
python scripts/02_run_experiment.py --model claude_sonnet_46 --split test

# 7. Consolidar todos os resultados e gerar gráficos
python scripts/03_consolidate_results.py
```

## Modelos (8)

| Alias | Modelo | Tier | Via |
|---|---|---|---|
| claude_sonnet_46 | Claude Sonnet 4.6 | Elite / grande | Bedrock |
| qwen3_32b | Qwen3 32B | Forte / médio | Bedrock |
| mistral_large_3 | Mistral Large 3 | Top / grande | Bedrock |
| gemma_3_27b | Gemma 3 27B | Bom / médio | Bedrock |
| deepseek_v32 | DeepSeek V3.2 | Top / grande | Bedrock |
| claude_3_haiku | Claude 3 Haiku | Leve / pequeno | Bedrock |
| magistral_small_2509 | Magistral Small | Forte / médio | Bedrock |
| sabia4 | Sabiá-4 | BR / médio | Maritaca API |

## Estratégias (4)

| Estratégia | Descrição |
|---|---|
| zero_shot | Instrução direta sem exemplos |
| few_shot | 3 exemplos fixos por classe |
| evidence_constrained | few-shot + saída com trecho evidência |
| decomposed | 3 perguntas jurídicas intermediárias → classe |

## Retomada de checkpoint

Se o processo cair, basta rodar o mesmo comando novamente.
O runner detecta o checkpoint e continua do último índice salvo.

## Variáveis de ambiente (.env)

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=sa-east-1
API_KEY_MARITACA=...
```
