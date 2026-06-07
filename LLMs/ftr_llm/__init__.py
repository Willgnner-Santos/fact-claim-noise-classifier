"""
ftr_llm — pipeline LLM para classificação jurídica Fato/Tese/Ruído.

Labels canônicos (idem src/ftr):  0=tese  1=fato  2=ruido
"""

ID2LABEL = {0: "tese", 1: "fato", 2: "ruido"}
LABEL2ID = {"tese": 0, "fato": 1, "ruido": 2}
CLASS_NAMES = ["tese", "fato", "ruido"]
