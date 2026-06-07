# Fact, Claim, Noise — Legal Text Classification (Brazilian Portuguese)

Code and artifacts for the paper submitted for anonymous peer review.

Three families of models are compared for classifying excerpts of Brazilian legal
petitions into three classes: **fato** (fact), **tese** (legal argument/claim),
and **ruído** (noise/irrelevant content).

> The dataset is confidential (provided by a public institution) and cannot be
> released. All other artifacts needed to reproduce the reported results are
> available here or on HuggingFace.

---

## Best model — ready to use

**Model on HuggingFace:** https://huggingface.co/AnonymousModelsRevision/ftr-bertimbau-large

The best fine-tuned model (BERTimbau-large, F1-macro 0.9901). Load and run inference in seconds:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "AnonymousModelsRevision/ftr-bertimbau-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

text = "O réu não compareceu à audiência designada pelo juízo."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    logits = model(**inputs).logits
label = ["tese", "fato", "ruído"][logits.argmax().item()]
print(label)  # fato
```

**Label mapping:** `0 = tese` | `1 = fato` | `2 = ruído`

---

## Reproducing the paper results

All per-instance predictions (`y_true`, `y_pred`, `proba`) for the canonical
test set (n=2,559) are included in this repository as `.npz` files:

```
Models_BERT/experiments/bertimbau_large__*/metrics/test_predictions.npz
Models_BERT/experiments/bertimbau_base__*/metrics/test_predictions.npz
Models_BERT/experiments/mbert__*/metrics/test_predictions.npz
Models_BERT/experiments/mdeberta_v3__*/metrics/test_predictions.npz
Models_BERT/experiments/roberta_crawlpt__*/metrics/test_predictions.npz
Models_BERT/experiments/xlmr__*/metrics/test_predictions.npz
ClassicML/experiments/lightgbm__*/metrics/test_predictions.npz
```

To recompute bootstrap CIs and McNemar tests from these files:

```bash
cd Models_BERT
pip install -r requirements.txt   # numpy, scipy, scikit-learn
python scripts/_bootstrap.py
```

Consolidated results for all three families (Classic ML, BERT, LLMs):

```
consolidated_all.json
```

---

## Repository structure

```
├── Models_BERT/
│   ├── configs/          # one YAML per encoder (7 models)
│   ├── scripts/          # 00_eda → 06_error_analysis + _bootstrap.py
│   ├── src/ftr/          # data, anonymization, training, metrics, viz
│   └── experiments/      # per-run: config, env.json, metrics/, figures/
│                         #   (best_model/ weights are on HuggingFace)
├── ClassicML/
│   ├── configs/          # TF-IDF + classifier configs
│   ├── scripts/          # run_experiment.py, export_lightgbm_preds.py
│   ├── ftr_classic/      # vectorizer, models, metrics
│   └── experiments/      # metrics/ + model_best.pkl (LightGBM, 8.7 MB)
├── LLMs/
│   ├── configs/          # prompt strategies (zero_shot, few_shot, ...)
│   ├── scripts/          # 00_prepare → 04_reeval_on_bert_test.py
│   ├── ftr_llm/          # bedrock/maritaca clients, parser, runner
│   └── data/
│       └── results_bertset/   # per-combination JSON + summary_berttest.csv
└── consolidated_all.json
```

---

## Experimental setup summary

| Family | Best model | F1-macro | 95% CI (bootstrap) |
|--------|-----------|----------|-------------------|
| BERT | BERTimbau-large | 0.9901 | [0.986, 0.994] |
| BERT ensemble (top-3) | BERTimbau-large + base + RoBERTaLexPT | 0.9905 | [0.987, 0.994] |
| Classic ML | LightGBM + TF-IDF | 0.9732 | [0.967, 0.979] |
| LLM (zero-shot) | Qwen3-32B (evidence_constrained) | 0.8526 | [0.839, 0.866] |

- Canonical test set: **2,559 excerpts** (877 fato / 814 tese / 868 ruído)
- Split: 80/10/10, grouped by document, seed=42, exact deduplication
- LLMs evaluated with failure = error (invalid responses count as wrong)
- Bootstrap: 2,000 resamples, seed=42

---

## Dependencies

```bash
# BERT / Classic ML
pip install -r Models_BERT/requirements.txt

# LLMs (AWS Bedrock + Maritaca)
pip install -r LLMs/requirements_llm.txt
```

Training was done on an NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM).
