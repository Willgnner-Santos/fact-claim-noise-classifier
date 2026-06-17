"""
ftr_classic — Classical ML for Fato/Tese/Ruído classification
==============================================================

Classical machine learning pipeline (sklearn) with:
- 7 models: LR, Linear SVM, RBF SVM, Random Forest, Gradient Boosting,
  Naive Bayes, Extra Trees
- Stratified K-Fold (5 folds) + hold-out test evaluation
- All metrics from the Métricas_Avaliação_Modelos.xlsx spreadsheet
- PT-BR and EN figures
- Detailed logs and consolidated results
- Reproducibility: seed, library versions, configurations

Canonical mapping (identical to the BERT pipeline):
    0 -> tese  |  1 -> fato  |  2 -> ruido
"""

ID2LABEL = {0: "tese", 1: "fato", 2: "ruido"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
CLASS_NAMES = [ID2LABEL[i] for i in sorted(ID2LABEL)]
CLASS_NAMES_PT = {"tese": "Tese", "fato": "Fato", "ruido": "Ruído"}
CLASS_NAMES_EN = {"tese": "Thesis", "fato": "Fact", "ruido": "Noise"}
