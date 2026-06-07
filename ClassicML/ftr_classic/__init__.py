"""
ftr_classic — IA Clássica para classificação Fato/Tese/Ruído
=============================================================

Pipeline de machine learning clássico (sklearn) com:
- 7 modelos: LR, SVM Linear, SVM RBF, Random Forest, Gradient Boosting,
  Naive Bayes, Extra Trees
- K-Fold estratificado (5 folds) + avaliação no teste hold-out
- Todas as métricas da planilha Métricas_Avaliação_Modelos.xlsx
- Gráficos PT-BR e EN
- Logs detalhados e resultados consolidados
- Reprodutibilidade: seed, versões, configurações

Mapeamento canônico (idêntico ao pipeline BERT):
    0 -> tese  |  1 -> fato  |  2 -> ruido
"""

ID2LABEL = {0: "tese", 1: "fato", 2: "ruido"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
CLASS_NAMES = [ID2LABEL[i] for i in sorted(ID2LABEL)]
CLASS_NAMES_PT = {"tese": "Tese", "fato": "Fato", "ruido": "Ruído"}
CLASS_NAMES_EN = {"tese": "Thesis", "fato": "Fact", "ruido": "Noise"}
