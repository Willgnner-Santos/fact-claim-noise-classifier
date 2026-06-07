"""
ftr — Fato / Tese / Ruído
=========================

Pacote de reprodução científica para classificação de textos jurídicos em três
classes: ``fato``, ``tese`` e ``ruido``.

O pacote foi desenhado para rodar de forma idêntica em três ambientes:

* **Local** (Windows/Linux, com ou sem GPU) — para análise de dados,
  pré-processamento, validação de métricas e geração de gráficos.
* **Google Colab** (GPU A100/H100) — para o treino dos modelos, com os dados e
  artefatos no Google Drive montado.
* **Servidor/cluster** — qualquer ambiente onde ``PROJECT_ROOT`` possa ser
  resolvido.

A resolução de caminhos é centralizada em :mod:`ftr.env`, de modo que todos os
artefatos (dados processados, experimentos, resultados) ficam relativos a um
único ``PROJECT_ROOT``.

Mapeamento canônico de rótulos (idêntico aos notebooks originais)::

    0 -> tese
    1 -> fato
    2 -> ruido
"""

from __future__ import annotations

__version__ = "1.0.0"

# Mapeamento canônico de rótulos — NÃO ALTERAR sem reprocessar os dados.
# Espelha o id2label/label2id usado nos notebooks originais.
ID2LABEL: dict[int, str] = {0: "tese", 1: "fato", 2: "ruido"}
LABEL2ID: dict[str, int] = {v: k for k, v in ID2LABEL.items()}

# Ordem canônica das classes para relatórios/gráficos (por índice de label).
CLASS_NAMES: list[str] = [ID2LABEL[i] for i in sorted(ID2LABEL)]  # ["tese", "fato", "ruido"]

# Rótulos "bonitos" para exibição em gráficos PT-BR.
CLASS_NAMES_PT: dict[str, str] = {"tese": "Tese", "fato": "Fato", "ruido": "Ruído"}
CLASS_NAMES_EN: dict[str, str] = {"tese": "Thesis", "fato": "Fact", "ruido": "Noise"}

__all__ = [
    "__version__",
    "ID2LABEL",
    "LABEL2ID",
    "CLASS_NAMES",
    "CLASS_NAMES_PT",
    "CLASS_NAMES_EN",
]
