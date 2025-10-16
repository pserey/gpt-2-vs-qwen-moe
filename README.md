# GPT-2 vs Qwen-MoE

Comparação entre **GPT-2 baseline** e outro modelo MoE (como Qwen-MoE) em PT-BR.
Inclui perplexidade, tokens/s, pico de GPU, geração qualitativa e curvas de convergência.

## Requisitos e Configuração

Este projeto usa **[uv](https://docs.astral.sh/uv/)** para gerenciamento de dependências e ambiente Python.

### Instalar uv (se não tiver)
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Via pip
pip install uv
```

### Configurar o projeto
```bash
# Clonar e entrar no diretório
cd gpt-2-vs-qwen-moe

# Instalar dependências (vai criar automaticamente o venv)
uv sync
```

## Dados (Gutenberg PT, 100 livros)
```bash
uv run python scripts/fetch_gutenberg_pt.py
```

## Executar

### Treinamento GPT-2 Baseline
```bash
uv run python src/train.py --config configs/baseline.yaml
```

## Saídas
- `results/metrics/*_metrics.json` — métricas (test_ppl, avg_tps, peak_mem_gb)
- `results/plots/*_convergence.png` — curvas treino/val
- `results/generations/*_generations.json` — amostras de geração

## Estrutura do Projeto
```
├── src/
│   ├── dataset.py           # Dataset e DataLoaders
│   ├── train.py            # Script principal de treinamento
│   ├── models/
│   │   └── gpt2_baseline.py # Modelo GPT-2 baseline (sem RoPE)
│   └── utils/
│       └── plotting.py     # Utilitários para gráficos
├── configs/
│   └── baseline.yaml       # Configuração do GPT-2
├── scripts/
│   └── fetch_gutenberg_pt.py # Coleta dados do Project Gutenberg
├── pyproject.toml          # Configuração do projeto e dependências
├── uv.lock                 # Lock file das dependências
├── data/                   # Dados de treinamento (criados pelo script)
└── results/               # Resultados dos experimentos
```

## Notas
- O modelo baseline usa **tiktoken (gpt2)** para tokenização
- MLP pode ser `"gelu"` ou `"swiglu"` (default)