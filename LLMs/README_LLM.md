# LLM Pipeline — Fato / Tese / Ruído

Length-aware benchmark for legal text classification in Portuguese.

## Structure

```
LLMs/
├── configs/
│   ├── _base_llm.yaml          # Base config inherited by all models
│   ├── claude_sonnet_46.yaml   # Elite / large
│   ├── qwen3_32b.yaml          # Strong / medium
│   ├── mistral_large_3.yaml    # Top / large
│   ├── gemma_3_27b.yaml        # Good / medium
│   ├── deepseek_v32.yaml       # Top / large
│   ├── claude_3_haiku.yaml     # Light / small
│   ├── magistral_small_2509.yaml # Strong / medium
│   └── sabia4.yaml             # Maritaca / Brazilian
├── ftr_llm/                    # Python pipeline package
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py              # Reuses src/ftr/anonymize + preprocess
│   ├── prompts.py              # zero_shot, few_shot, evidence_constrained, decomposed
│   ├── bedrock_client.py       # Retry + fallback + cost
│   ├── maritaca_client.py      # Sabiá-4
│   ├── parser.py               # Robust JSON parser
│   ├── checkpoint.py           # Per-line JSONL checkpoint
│   ├── runner.py               # Main loop with resumption
│   ├── evaluate.py             # Metrics + per-bin analysis
│   └── viz.py                  # Plots (PT and EN)
├── scripts/
│   ├── 00_prepare_dataset.py   # Preprocess + anonymize + split
│   ├── 01_smoke_test_models.py # Test all models with "hi"
│   ├── 02_run_experiment.py    # Run 1 model x N strategies
│   └── 03_consolidate_results.py # Consolidate + final plots
├── data/
│   ├── splits/                 # train.json, val.json, test.json
│   ├── checkpoints/            # JSONL per model x strategy
│   ├── results/                # JSON + CSV metrics
│   └── reports/                # Plots (PT and EN)
├── .env
└── requirements_llm.txt
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements_llm.txt

# 2. Prepare dataset (anonymize + split — only needs to run once)
python scripts/00_prepare_dataset.py

# 3. Test if models are accessible
python scripts/01_smoke_test_models.py

# 4. Fast pipeline validation (30 examples, 1 model, 1 strategy)
python scripts/02_run_experiment.py --model claude_3_haiku --sample-n 30 --strategies zero_shot

# 5. Run full experiment on val set
python scripts/02_run_experiment.py --model claude_3_haiku --split val

# 6. Run on test set (final evaluation — only after tuning prompts on val)
python scripts/02_run_experiment.py --model claude_sonnet_46 --split test

# 7. Consolidate all results and generate plots
python scripts/03_consolidate_results.py
```

## Models (8)

| Alias | Model | Tier | Via |
|---|---|---|---|
| claude_sonnet_46 | Claude Sonnet 4.6 | Elite / large | Bedrock |
| qwen3_32b | Qwen3 32B | Strong / medium | Bedrock |
| mistral_large_3 | Mistral Large 3 | Top / large | Bedrock |
| gemma_3_27b | Gemma 3 27B | Good / medium | Bedrock |
| deepseek_v32 | DeepSeek V3.2 | Top / large | Bedrock |
| claude_3_haiku | Claude 3 Haiku | Light / small | Bedrock |
| magistral_small_2509 | Magistral Small | Strong / medium | Bedrock |
| sabia4 | Sabiá-4 | BR / medium | Maritaca API |

## Strategies (4)

| Strategy | Description |
|---|---|
| zero_shot | Direct instruction without examples |
| few_shot | 3 fixed examples per class |
| evidence_constrained | few-shot + output with evidence excerpt |
| decomposed | 3 intermediate legal questions -> class |

## Checkpoint resumption

If the process crashes, just run the same command again.
The runner detects the checkpoint and continues from the last saved index.

## Environment variables (.env)

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=sa-east-1
API_KEY_MARITACA=...
```
