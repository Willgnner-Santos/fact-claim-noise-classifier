"""
ftr_llm — LLM pipeline for legal classification Fato/Tese/Ruído.

Canonical labels (same as src/ftr):  0=tese  1=fato  2=ruido
"""

ID2LABEL = {0: "tese", 1: "fato", 2: "ruido"}
LABEL2ID = {"tese": 0, "fato": 1, "ruido": 2}
CLASS_NAMES = ["tese", "fato", "ruido"]
