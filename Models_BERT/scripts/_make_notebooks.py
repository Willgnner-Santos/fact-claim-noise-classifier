"""
Gerador dos notebooks Colab do projeto (executar uma vez, localmente).

Cria notebooks .ipynb válidos em Notebooks/:
  00_Colab_Setup.ipynb        — monta Drive, instala deps, valida GPU/caminhos
  00_EDA.ipynb                — análise exploratória
  01_experiment_template.ipynb— treina UM modelo (parametrizável)
  02_run_all_colab.ipynb      — treina todos os modelos em sequência

Mantém os notebooks finos: eles apenas configuram o ambiente e chamam o pacote
ftr / os scripts. Toda a lógica vive em src/ftr.

Uso:
    python scripts/_make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "Notebooks"

DRIVE_ROOT = "/content/drive/MyDrive/Fato_Tese_Ruido"


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(lines)}


def _src(lines):
    text = "\n".join(lines)
    # nbformat espera lista de linhas terminadas em \n (exceto a última).
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name: str, nb: dict) -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    path = NB_DIR / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  escrito: {path}")


# ---------------------------------------------------------------------------
# Células reutilizáveis
# ---------------------------------------------------------------------------
def cell_mount_and_setup() -> list[dict]:
    return [
        md(
            "## 1. Montar o Google Drive e localizar o projeto",
            "",
            "⚠️ **Suba o PROJETO INTEIRO para o Drive**, não só a pasta `Data`. A pasta",
            f"no Drive (por padrão `{DRIVE_ROOT}`) precisa conter:",
            "",
            "```",
            "Fato_Tese_Ruido/",
            "├── requirements.txt",
            "├── src/ftr/...        ← o módulo 'ftr' (sem isto: No module named 'ftr')",
            "├── configs/",
            "├── scripts/",
            "└── Data/dataset_FINAL.json",
            "```",
            "",
            "A célula abaixo **localiza a pasta automaticamente** (procurando por",
            "`src/ftr`) e mostra um checklist do que está presente/faltando.",
        ),
        code(
            "import os, sys, glob",
            "",
            "# Monta o Drive (somente no Colab).",
            "try:",
            "    from google.colab import drive",
            "    drive.mount('/content/drive')",
            "    IN_COLAB = True",
            "except Exception:",
            "    IN_COLAB = False",
            "    print('Não está no Colab — usando caminho local.')",
            "",
            "",
            "def _looks_like_project(p):",
            "    return os.path.isdir(os.path.join(p, 'src', 'ftr'))",
            "",
            "",
            "def _autodetect_root():",
            f"    candidates = ['{DRIVE_ROOT}']",
            "    # procura qualquer pasta com src/ftr dentro do MyDrive (2 níveis)",
            "    for base in ('/content/drive/MyDrive', '/content/drive/My Drive'):",
            "        candidates += glob.glob(os.path.join(base, '**', 'src', 'ftr'), recursive=False)",
            "        candidates += [os.path.dirname(os.path.dirname(p))",
            "                       for p in glob.glob(os.path.join(base, '*', 'src', 'ftr'))]",
            "    for c in candidates:",
            "        if c and _looks_like_project(c):",
            "            return c",
            "    return candidates[0] if candidates else os.getcwd()",
            "",
            "",
            f"PROJECT_ROOT = _autodetect_root() if IN_COLAB else os.getcwd()",
            "os.environ['FTR_PROJECT_ROOT'] = PROJECT_ROOT",
            "if os.path.join(PROJECT_ROOT, 'src') not in sys.path:",
            "    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))",
            "print('PROJECT_ROOT =', PROJECT_ROOT)",
            "",
            "# Itens OBRIGATÓRIOS (o código + os dados).",
            "required = {'src/ftr': 'pasta (módulo ftr)', 'configs': 'pasta',",
            "            'scripts': 'pasta', 'Data/dataset_FINAL.json': 'dataset'}",
            "# Itens OPCIONAIS (o Drive às vezes converte .txt/.xlsx p/ Google Docs/Sheets,",
            "# removendo a extensão — sem problema, instalamos as deps explicitamente abaixo).",
            "optional = {'requirements.txt': 'arquivo (opcional)'}",
            "print('\\nChecklist da pasta do projeto:')",
            "missing = []",
            "for rel, kind in required.items():",
            "    ok = os.path.exists(os.path.join(PROJECT_ROOT, rel))",
            "    print(f'  [{\"OK\" if ok else \"FALTA\"}] {rel}  ({kind})')",
            "    if not ok:",
            "        missing.append(rel)",
            "for rel, kind in optional.items():",
            "    ok = os.path.exists(os.path.join(PROJECT_ROOT, rel))",
            "    print(f'  [{\"OK\" if ok else \"--\"}] {rel}  ({kind})')",
            "if missing:",
            "    raise SystemExit('\\nFALTAM itens OBRIGATÓRIOS acima. Suba a PASTA INTEIRA do '",
            "                     'projeto para o Drive (não só Data/).')",
            "print('\\nOK — itens obrigatórios presentes. (requirements.txt é opcional.)')",
        ),
        md(
            "## 2. Instalar dependências",
            "",
            "Instala as bibliotecas explicitamente (não depende do `requirements.txt`, que o",
            "Drive às vezes converte para Google Docs). torch já vem no Colab.",
        ),
        code(
            "if IN_COLAB:",
            "    !pip install -q 'transformers==4.48.3' 'datasets==3.3.0' 'scikit-learn==1.6.1' \\",
            "        evaluate sentencepiece sacremoses 'accelerate>=1.3.0' openpyxl seaborn pyarrow",
            "    print('\\nDependências instaladas.')",
            "else:",
            "    print('Local: instale com  pip install -r requirements.txt')",
        ),
        md("## 3. Verificar GPU, ambiente e caminhos"),
        code(
            "import torch",
            "from ftr.env import describe_environment, get_paths",
            "print('CUDA disponível:', torch.cuda.is_available())",
            "if torch.cuda.is_available():",
            "    print('GPU:', torch.cuda.get_device_name(0))",
            "import json",
            "print(json.dumps(describe_environment(), indent=2, ensure_ascii=False))",
        ),
        md("## 4. Verificar que o dataset está no lugar"),
        code(
            "paths = get_paths()",
            "print('Dataset bruto:', paths.raw_dataset)",
            "assert paths.raw_dataset.exists(), (",
            "    f'Dataset não encontrado em {paths.raw_dataset}. '",
            "    'Suba Data/dataset_FINAL.json para o Drive.')",
            "print('OK — dataset encontrado (%.1f MB).' % (paths.raw_dataset.stat().st_size/1e6))",
        ),
    ]


def make_setup_nb() -> dict:
    cells = [
        md(
            "# 00 — Configuração do Colab (Fato / Tese / Ruído)",
            "",
            "Este notebook prepara o ambiente no Google Colab: monta o Drive, instala",
            "as dependências e valida GPU, caminhos e dataset. **Rode-o primeiro.**",
        ),
        *cell_mount_and_setup(),
        md(
            "## 5. Pronto!",
            "",
            "Ambiente configurado. Agora você pode:",
            "- Rodar `00_EDA.ipynb` para a análise exploratória;",
            "- Rodar `01_experiment_template.ipynb` para treinar um modelo;",
            "- Rodar `02_run_all_colab.ipynb` para treinar todos em sequência.",
            "",
            "Antes do treino, prepare os dados (limpeza + anonimização + split):",
        ),
        code(
            "# -W all exibe todos os warnings (importante para ver alertas de leakage)",
            "!cd \"{PROJECT_ROOT}\" && python -W all scripts/01_prepare_data.py --config configs/_base.yaml",
        ),
    ]
    return notebook(cells)


def make_eda_nb() -> dict:
    cells = [
        md("# 00 — Análise Exploratória (EDA)", "", "Roda a EDA e gera gráficos PT/EN."),
        *cell_mount_and_setup(),
        md("## 5. Executar EDA", "Com contagem de tokens reais (usa um tokenizer BERTimbau)."),
        code(
            "!cd \"{PROJECT_ROOT}\" && python scripts/00_eda.py "
            "--tokenizer neuralmind/bert-base-portuguese-cased --token-sample 3000",
        ),
        md("## 6. Visualizar os gráficos gerados"),
        code(
            "from IPython.display import Image, display",
            "import glob",
            "for f in sorted(glob.glob(os.path.join(PROJECT_ROOT, 'results/figures/eda/*_pt.png'))):",
            "    print(f); display(Image(f))",
        ),
    ]
    return notebook(cells)


def make_experiment_nb() -> dict:
    cells = [
        md(
            "# 01 — Treinar UM modelo (template)",
            "",
            "Escolha o `CONFIG` na célula de parâmetros e execute. Os artefatos são",
            "salvos em `experiments/<modelo>__<timestamp>__<runid>/` no Drive.",
        ),
        *cell_mount_and_setup(),
        md(
            "## 5. Parâmetros do experimento",
            "",
            "Configs disponíveis: `roberta_lexpt`, `roberta_crawlpt`, `bertimbau_large`,",
            "`mbert`, `bertimbau_base`, `xlmr`, `mdeberta_v3`, `longformer_pt`.",
        ),
        code(
            "CONFIG = 'configs/roberta_lexpt.yaml'  # @param",
            "SMOKE_TEST = False  # @param  — True faz um teste rápido (5 passos)",
            "max_steps = 5 if SMOKE_TEST else None",
        ),
        md(
            "## 6. Preparar dados (se ainda não preparou)",
            "",
            "Inclui split **por documento** (anti-leakage). Se já rodou no `02_run_all_colab.ipynb`",
            "pode pular esta célula — os splits já existem.",
        ),
        code(
            "from ftr.data import load_splits",
            "splits = load_splits(paths.data_processed)",
            "if 'train' not in splits:",
            "    print('Preparando dados (split por documento)...')",
            "    !cd \"{PROJECT_ROOT}\" && python -W all scripts/01_prepare_data.py --config {CONFIG}",
            "    splits = load_splits(paths.data_processed)",
            "print('Splits:', {k: len(v) for k, v in splits.items()})",
        ),
        md("## 7. Treinar"),
        code(
            "from ftr.config import load_config",
            "from ftr.train import train_experiment",
            "cfg = load_config(CONFIG, configs_dir=paths.configs)",
            "status = train_experiment(cfg, paths, splits=splits, max_steps=max_steps)",
            "print('Status:', status['status'], '| dir:', status['run_dir'])",
        ),
        md("## 8. Ver métricas e gráficos do experimento"),
        code(
            "import json, glob",
            "from IPython.display import Image, display",
            "run_dir = paths.experiments / status['run_dir']",
            "with open(run_dir / 'metrics' / 'test_metrics.json', encoding='utf-8') as f:",
            "    m = json.load(f)",
            "print('Accuracy :', round(m['accuracy'], 4))",
            "print('F1-macro :', round(m['f1_macro'], 4))",
            "print('MCC      :', round(m['mcc'], 4))",
            "for fig in sorted(glob.glob(str(run_dir / 'figures' / '*_pt.png'))):",
            "    display(Image(fig))",
        ),
    ]
    return notebook(cells)


def make_run_all_nb() -> dict:
    cells = [
        md(
            "# 02 — Treinar TODOS os modelos em sequência",
            "",
            "Roda todos os experimentos (ou um subconjunto), de forma resiliente: se um",
            "modelo falhar ou a sessão cair, os demais continuam e o erro fica no",
            "`status.json`. Ao final, consolida os resultados e gera a tabela comparativa.",
            "",
            "⚠️ Treino completo é demorado. Considere começar com `SMOKE_TEST=True`.",
        ),
        *cell_mount_and_setup(),
        md(
            "## 5. Preparar dados",
            "",
            "Aplica limpeza, anonimização e **split por documento** (anti-leakage).",
            "O script agrupa fragmentos do mesmo processo judicial via OAB/email do",
            "advogado e garante que nenhum fragmento aparece em treino E teste.",
            "Ao final, imprime o vazamento estimado — deve ser 0.00%.",
        ),
        code(
            "!cd \"{PROJECT_ROOT}\" && python -W all scripts/01_prepare_data.py --config configs/_base.yaml",
        ),
        md("## 6. Parâmetros"),
        code(
            "SMOKE_TEST        = False  # @param {type:\"boolean\"} — True = 5 steps por modelo (teste rápido)",
            "REPRODUCTIONS_ONLY = False  # @param {type:\"boolean\"} — só os 4 modelos originais",
            "# Alerta de convergência: se os últimos CONVERGENCE_WINDOW modelos",
            "# estiverem dentro de CONVERGENCE_THRESHOLD de f1_macro, pausa e pergunta.",
            "# No Colab (sem terminal interativo) a flag --non-interactive continua automaticamente.",
            "CONVERGENCE_THRESHOLD = 0.005  # @param {type:\"number\"}",
            "CONVERGENCE_WINDOW    = 3      # @param {type:\"integer\"}",
        ),
        md(
            "## 7. Validar código antes de treinar",
            "",
            "Confirma que os arquivos corretos estão carregados.",
            "**Execute esta célula antes de qualquer treino** — ela detecta se há",
            "versões antigas dos módulos e avisa antes de gastar créditos de GPU.",
        ),
        code(
            "import inspect, sys, importlib",
            "",
            "# Recarrega módulos para garantir que o Drive foi lido (não cache)",
            "for mod_name in list(sys.modules.keys()):",
            "    if mod_name.startswith('ftr'):",
            "        del sys.modules[mod_name]",
            "",
            "import ftr.evaluate, ftr.data, ftr.train",
            "",
            "checks = {",
            "    'evaluate.py (bug tuple-output)': ('Dataset SEM labels', inspect.getsource(ftr.evaluate._predict_truncation)),",
            "    'data.py (split por documento)':  ('GroupShuffleSplit',   inspect.getsource(ftr.data.split_dataset)),",
            "    'train.py (import os)':           ('import os',           inspect.getsource(ftr.train)),",
            "}",
            "",
            "all_ok = True",
            "for desc, (keyword, src) in checks.items():",
            "    ok = keyword in src",
            "    status = 'OK' if ok else 'FALHOU — arquivo antigo no Drive!'",
            "    print(f'  [{status}] {desc}')",
            "    if not ok: all_ok = False",
            "",
            "if not all_ok:",
            "    raise SystemExit('Corrija os arquivos acima antes de treinar. '",
            "                     'Suba novamente a pasta src/ do projeto local para o Drive.')",
            "print()",
            "print('Tudo OK — pode prosseguir com o treino.')",
        ),
        md("## 8. Rodar tudo"),
        code(
            "extra = ''",
            "if SMOKE_TEST:        extra += ' --max-steps 5'",
            "if REPRODUCTIONS_ONLY: extra += ' --reproductions-only'",
            "extra += f' --convergence-threshold {CONVERGENCE_THRESHOLD}'",
            "extra += f' --window {CONVERGENCE_WINDOW}'",
            "extra += ' --non-interactive'  # Colab nao tem terminal interativo",
            "",
            "!cd \"{PROJECT_ROOT}\" && python scripts/run_all.py {extra}",
        ),
        md(
            "## 9. Reavaliar experimentos com métricas suspeitas",
            "",
            "Se os resultados no teste estiverem ~0.31 (acaso para 3 classes), significa",
            "que o experimento foi gerado com código antigo. Esta célula detecta e corrige",
            "**sem re-treinar** — apenas recarrega o `best_model` salvo e reavalia.",
        ),
        code(
            "import json",
            "from ftr.registry import collect_experiments",
            "from ftr.config import load_config",
            "from ftr.evaluate import evaluate_from_dir",
            "",
            "REEVAL_THRESHOLD = 0.5  # f1_macro abaixo disso = resultado suspeito",
            "",
            "runs = collect_experiments(paths.experiments)",
            "suspicious = []",
            "for r in runs:",
            "    if r.get('status') != 'completed': continue",
            "    tm = r.get('test_metrics') or {}",
            "    f1 = tm.get('f1_macro', 0) if tm else 0",
            "    if f1 < REEVAL_THRESHOLD:",
            "        suspicious.append(r)",
            "",
            "if not suspicious:",
            "    print('Nenhum resultado suspeito encontrado.')",
            "else:",
            "    print(f'{len(suspicious)} experimento(s) com f1_macro < {REEVAL_THRESHOLD} — reavaliando...')",
            "    for r in suspicious:",
            "        exp_dir = paths.experiments / r['run_dir']",
            "        cfg_path = exp_dir / 'config_snapshot.yaml'",
            "        if not cfg_path.exists():",
            "            print(f'  {r[\"experiment_id\"]}: config_snapshot.yaml nao encontrado — pulando')",
            "            continue",
            "        try:",
            "            cfg = load_config(cfg_path, configs_dir=paths.configs)",
            "            m = evaluate_from_dir(exp_dir, cfg, paths)",
            "            print(f'  {r[\"experiment_id\"]:20s}  f1_macro={m[\"f1_macro\"]:.4f}  acc={m[\"accuracy\"]:.4f}')",
            "        except Exception as e:",
            "            print(f'  {r[\"experiment_id\"]}: ERRO — {e}')",
        ),
        md("## 10. Consolidar e ver a tabela comparativa"),
        code(
            "!cd \"{PROJECT_ROOT}\" && python scripts/04_consolidate.py",
            "import pandas as pd",
            "df = pd.read_csv(paths.results / 'comparison_table.csv')",
            "df[['experiment_id','f1_macro','accuracy','mcc','roc_auc_macro_ovr']].sort_values('f1_macro', ascending=False)",
        ),
        md("## 11. Gráficos comparativos"),
        code(
            "from IPython.display import Image, display",
            "import glob",
            "for f in sorted(glob.glob(str(paths.results_figures / 'comparison' / '*_pt.png'))):",
            "    display(Image(f))",
        ),
        md(
            "## 12. Baixar os resultados",
            "",
            "Os artefatos já estão no seu Drive (em `experiments/`, `results/`, `reports/`).",
            "Baixe essas pastas do Drive e coloque localmente na sua IDE para análise.",
        ),
        md(
            "## 13. Encerrar a sessão do Colab",
            "",
            "Após confirmar que tudo foi salvo no Drive, esta célula aguarda 5 minutos",
            "(tempo de segurança para o Drive sincronizar) e então encerra a sessão",
            "automaticamente, evitando consumo de créditos desnecessários.",
            "",
            "**Só rode esta célula quando tiver certeza de que todos os resultados",
            "foram salvos no Drive.**",
        ),
        code(
            "import time",
            "",
            "WAIT_MINUTES = 5  # @param {type:\"integer\"}",
            "AUTO_SHUTDOWN = True  # @param {type:\"boolean\"}",
            "",
            "if AUTO_SHUTDOWN and IN_COLAB:",
            "    print(f'Aguardando {WAIT_MINUTES} min para o Drive sincronizar...')",
            "    for remaining in range(WAIT_MINUTES * 60, 0, -30):",
            "        print(f'  Encerrando em {remaining//60}m{remaining%60:02d}s...')",
            "        time.sleep(30)",
            "    print('Encerrando sessão do Colab.')",
            "    from google.colab import runtime",
            "    runtime.unassign()",
            "else:",
            "    print('AUTO_SHUTDOWN=False ou fora do Colab — sessão mantida.')",
        ),
    ]
    return notebook(cells)


def main() -> None:
    print("Gerando notebooks Colab em", NB_DIR)
    write("00_Colab_Setup.ipynb", make_setup_nb())
    write("00_EDA.ipynb", make_eda_nb())
    write("01_experiment_template.ipynb", make_experiment_nb())
    write("02_run_all_colab.ipynb", make_run_all_nb())
    print("OK.")


if __name__ == "__main__":
    main()
