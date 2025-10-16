# GPT-2 vs Qwen-MoE

Comparação entre **GPT-2 baseline** e **Qwen-style MoE** em PT-BR.
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

## Modelos Implementados

### 1. GPT-2 Baseline
Modelo transformer padrão com:
- **Multi-Head Attention** com RoPE
- **SwiGLU** ou GELU para MLP
- **Layer Normalization** pré-camada
- **Positional embeddings** ou RoPE

### 2. Qwen-style MoE
Implementação completa de Mixture of Experts com:
- **Router top-k** (k=1 ou 2) com softmax sobre scores
- **Capacity factor** (1.0-1.25) para limitar tokens por expert
- **Experts independentes** usando SwiGLU
- **Aux loss de balanceamento** (Switch Transformer style)
- **Dispatch/Combine** eficiente de tokens

## Executar Experimentos

### GPT-2 Baseline
```bash
uv run python src/train.py --config configs/baseline.yaml
```

### Qwen-MoE (4 experts, top-1)
```bash
uv run python src/train.py --config configs/qwen_moe_4experts.yaml
# ou use o script:
bash scripts/run_moe_4experts.sh
```

### Qwen-MoE (8 experts, top-2)
```bash
uv run python src/train.py --config configs/qwen_moe_8experts.yaml
# ou use o script:
bash scripts/run_moe_8experts.sh
```

### Qwen-MoE (6 experts, top-1)
```bash
uv run python src/train.py --config configs/qwen_moe_6experts.yaml
```

## Configurações MoE

### 4 Experts (Top-1) - Balanceado
- **4 experts**, top-k=1, capacity_factor=1.0
- **Expert hidden_dim=768** (mesmo que baseline para comparação justa)
- **Aux loss weight=0.01**

### 8 Experts (Top-2) - Alta Capacidade  
- **8 experts**, top-k=2, capacity_factor=1.25
- **Expert hidden_dim=512** (menor para manter parâmetros ativos similares)
- **Aux loss weight=0.01**

### 6 Experts (Top-1) - Intermediário
- **6 experts**, top-k=1, capacity_factor=1.1
- **Expert hidden_dim=640** (balança parâmetros)

## Saídas e Métricas

### Arquivos gerados:
- `results/metrics/*_metrics.json` — métricas completas (PPL, TPS, memória, MoE stats)
- `results/plots/*_convergence.png` — curvas de convergência treino/val
- `results/generations/*_generations.json` — amostras de geração qualitativa

### Métricas de Comparação:
1. **Perplexity (PPL)** — métrica principal de qualidade
2. **Tokens/segundo (TPS)** — throughput durante treinamento
3. **Pico de GPU (GB)** — uso máximo de memória
4. **Parâmetros ativos** — parâmetros usados por forward pass
5. **Geração qualitativa** — coerência e estilo dos textos gerados

## Estrutura do Projeto
```
├── src/
│   ├── dataset.py             # Dataset e DataLoaders
│   ├── train.py               # Script principal (suporta baseline + MoE)
│   ├── models/
│   │   ├── gpt2_baseline.py   # GPT-2 baseline c/ RoPE
│   │   └── qwen_moe.py        # Qwen-style MoE completo
│   └── utils/
│       └── plotting.py        # Utilitários para gráficos
├── configs/
│   ├── baseline.yaml          # GPT-2 baseline
│   └── qwen_moe.yaml          # MoE 6 experts (top-1)  
├── scripts/
│   └── fetch_gutenberg_pt.py  # Coleta dados Project Gutenberg
├── pyproject.toml             # Configuração uv
├── data/                      # Dados (criados pelo script)
└── results/                   # Resultados dos experimentos
```

## Implementação Técnica

### MoE Components:
- **Router**: MLP leve com softmax + top-k selection
- **Expert**: SwiGLU independent networks  
- **Dispatch**: Token routing com capacity factor
- **Combine**: Weighted recombination
- **Aux Loss**: Load balancing (Switch Transformer)

### Características:
- **Mixed precision** training (bfloat16)
- **Gradient accumulation** configurável
- **Learning rate scheduling** (cosine com warmup)
- **Gradient clipping** para estabilidade
- **Comprehensive logging** para debug

## Notas Importantes

- **Parâmetros ativos**: MoE mantém parâmetros ativos similares ao baseline para comparação justa
- **Capacity factor**: Controla quantos tokens cada expert processa (1.0-1.25)
- **Aux loss weight**: Balanceia uso dos experts (0.01 padrão)
- **Gradient checkpointing**: Desabilitado para MoE devido à complexidade
- **RoPE**: Habilitado por padrão em ambos os modelos