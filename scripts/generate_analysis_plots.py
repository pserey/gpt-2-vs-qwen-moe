#!/usr/bin/env python3
"""
Script para gerar plots de análise comparativa entre GPT-2 baseline e Qwen-MoE
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Configurações de estilo
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def load_metrics(metrics_path):
    """Carrega métricas de um arquivo JSON"""
    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_comparative_convergence():
    """Plot comparativo de convergência dos dois modelos"""
    # Carregar dados
    baseline_metrics = load_metrics('results/metrics/baseline_gpt2_small_metrics.json')
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Loss comparativo
    plt.subplot(2, 2, 1)
    steps_baseline = list(range(len(baseline_metrics['train_losses'])))
    steps_moe = list(range(len(moe_metrics['train_losses'])))
    
    plt.plot(steps_baseline, baseline_metrics['train_losses'], 
             label='GPT-2 Baseline', color='#1f77b4', linewidth=2)
    plt.plot(steps_moe, moe_metrics['train_losses'], 
             label='Qwen-MoE', color='#ff7f0e', linewidth=2)
    
    plt.xlabel('Passos de Avaliação')
    plt.ylabel('Loss de Treinamento')
    plt.title('Comparativo de Convergência')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Tokens processados vs Loss
    plt.subplot(2, 2, 2)
    tokens_baseline = baseline_metrics['tokens_seen']
    tokens_moe = moe_metrics['tokens_seen']
    
    plt.plot(np.array(tokens_baseline)/1e6, baseline_metrics['train_losses'], 
             label='GPT-2 Baseline', color='#1f77b4', linewidth=2)
    plt.plot(np.array(tokens_moe)/1e6, moe_metrics['train_losses'], 
             label='Qwen-MoE', color='#ff7f0e', linewidth=2)
    
    plt.xlabel('Tokens Processados (Milhões)')
    plt.ylabel('Loss de Treinamento')
    plt.title('Loss vs Tokens Processados')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Perplexidade final
    plt.subplot(2, 2, 3)
    models = ['GPT-2\nBaseline', 'Qwen-MoE']
    perplexities = [baseline_metrics['test_ppl'], moe_metrics['test_ppl']]
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = plt.bar(models, perplexities, color=colors, alpha=0.7)
    plt.ylabel('Perplexidade de Teste')
    plt.title('Comparativo de Perplexidade Final')
    plt.grid(True, alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, ppl in zip(bars, perplexities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{ppl:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 4: Eficiência computacional
    plt.subplot(2, 2, 4)
    metrics_names = ['Tokens/sec', 'Memória\n(GB)', 'Perplexidade\nFinal']
    baseline_values = [baseline_metrics['avg_tps']/1000, 
                      baseline_metrics['peak_mem_gb'], 
                      baseline_metrics['test_ppl']]
    moe_values = [moe_metrics['avg_tps']/1000, 
                 moe_metrics['peak_mem_gb'], 
                 moe_metrics['test_ppl']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    plt.bar(x - width/2, baseline_values, width, label='GPT-2 Baseline', 
            color='#1f77b4', alpha=0.7)
    plt.bar(x + width/2, moe_values, width, label='Qwen-MoE', 
            color='#ff7f0e', alpha=0.7)
    
    plt.xlabel('Métricas')
    plt.ylabel('Valores')
    plt.title('Comparativo de Eficiência')
    plt.xticks(x, metrics_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/comparative_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico comparativo salvo em: results/plots/comparative_analysis.png")

def plot_learning_efficiency():
    """Plot de eficiência de aprendizado (loss vs tokens)"""
    baseline_metrics = load_metrics('results/metrics/baseline_gpt2_small_metrics.json')
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Curva de aprendizado suavizada
    plt.subplot(1, 2, 1)
    
    # Suavizar curvas com média móvel
    def smooth_curve(y, window=10):
        return np.convolve(y, np.ones(window)/window, mode='valid')
    
    tokens_baseline = np.array(baseline_metrics['tokens_seen'])/1e6
    tokens_moe = np.array(moe_metrics['tokens_seen'])/1e6
    
    losses_baseline_smooth = smooth_curve(baseline_metrics['train_losses'])
    losses_moe_smooth = smooth_curve(moe_metrics['train_losses'])
    
    tokens_baseline_smooth = tokens_baseline[9:]  # Ajustar para o tamanho da suavização
    tokens_moe_smooth = tokens_moe[9:]
    
    plt.plot(tokens_baseline_smooth, losses_baseline_smooth, 
             label='GPT-2 Baseline (suavizado)', color='#1f77b4', linewidth=3)
    plt.plot(tokens_moe_smooth, losses_moe_smooth, 
             label='Qwen-MoE (suavizado)', color='#ff7f0e', linewidth=3)
    
    # Pontos originais mais transparentes
    plt.plot(tokens_baseline, baseline_metrics['train_losses'], 
             color='#1f77b4', alpha=0.3, linewidth=1)
    plt.plot(tokens_moe, moe_metrics['train_losses'], 
             color='#ff7f0e', alpha=0.3, linewidth=1)
    
    plt.xlabel('Tokens Processados (Milhões)')
    plt.ylabel('Loss de Treinamento')
    plt.title('Eficiência de Aprendizado')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Taxa de melhoria por token
    plt.subplot(1, 2, 2)
    
    # Calcular derivada (taxa de melhoria)
    def compute_improvement_rate(tokens, losses, window=5):
        rates = []
        for i in range(window, len(losses)):
            # Taxa de melhoria: -delta_loss / delta_tokens
            delta_loss = losses[i] - losses[i-window]
            delta_tokens = tokens[i] - tokens[i-window]
            if delta_tokens > 0:
                rates.append(-delta_loss / delta_tokens * 1e6)  # Por milhão de tokens
            else:
                rates.append(0)
        return rates
    
    rates_baseline = compute_improvement_rate(tokens_baseline, baseline_metrics['train_losses'])
    rates_moe = compute_improvement_rate(tokens_moe, moe_metrics['train_losses'])
    
    tokens_rates_baseline = tokens_baseline[5:]
    tokens_rates_moe = tokens_moe[5:]
    
    plt.plot(tokens_rates_baseline, rates_baseline, 
             label='GPT-2 Baseline', color='#1f77b4', linewidth=2)
    plt.plot(tokens_rates_moe, rates_moe, 
             label='Qwen-MoE', color='#ff7f0e', linewidth=2)
    
    plt.xlabel('Tokens Processados (Milhões)')
    plt.ylabel('Taxa de Melhoria (Δloss/Mtokens)')
    plt.title('Taxa de Melhoria por Token')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('results/plots/learning_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico de eficiência salvo em: results/plots/learning_efficiency.png")

def plot_model_characteristics():
    """Plot das características específicas dos modelos"""
    baseline_metrics = load_metrics('results/metrics/baseline_gpt2_small_metrics.json')
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Throughput (tokens por segundo)
    models = ['GPT-2\nBaseline', 'Qwen-MoE\n(6 experts)']
    throughputs = [baseline_metrics['avg_tps'], moe_metrics['avg_tps']]
    colors = ['#1f77b4', '#ff7f0e']
    
    bars1 = ax1.bar(models, throughputs, color=colors, alpha=0.7)
    ax1.set_ylabel('Tokens por Segundo')
    ax1.set_title('Throughput de Treinamento')
    ax1.grid(True, alpha=0.3)
    
    for bar, tps in zip(bars1, throughputs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{tps:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Uso de memória
    memory_usage = [baseline_metrics['peak_mem_gb'], moe_metrics['peak_mem_gb']]
    bars2 = ax2.bar(models, memory_usage, color=colors, alpha=0.7)
    ax2.set_ylabel('Memória de Pico (GB)')
    ax2.set_title('Consumo de Memória')
    ax2.grid(True, alpha=0.3)
    
    for bar, mem in zip(bars2, memory_usage):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{mem:.1f} GB', ha='center', va='bottom', fontweight='bold')
    
    # 3. Eficiência (throughput / memória)
    efficiency = [t/m for t, m in zip(throughputs, memory_usage)]
    bars3 = ax3.bar(models, efficiency, color=colors, alpha=0.7)
    ax3.set_ylabel('Tokens/seg por GB')
    ax3.set_title('Eficiência Computacional')
    ax3.grid(True, alpha=0.3)
    
    for bar, eff in zip(bars3, efficiency):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{eff:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Características do MoE (apenas para Qwen-MoE)
    if 'num_experts' in moe_metrics:
        moe_chars = ['Num. Experts', 'Top-K', 'Capacity Factor', 'Aux Loss Weight']
        moe_values = [moe_metrics.get('num_experts', 0), 
                     moe_metrics.get('top_k', 0),
                     moe_metrics.get('capacity_factor', 0),
                     moe_metrics.get('aux_loss_weight', 0)]
        
        bars4 = ax4.bar(moe_chars, moe_values, color='#ff7f0e', alpha=0.7)
        ax4.set_ylabel('Valores')
        ax4.set_title('Parâmetros do Modelo MoE')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        for bar, val in zip(bars4, moe_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(moe_values)*0.02,
                    f'{val}', ha='center', va='bottom', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Dados MoE\nnão disponíveis', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Parâmetros do Modelo MoE')
    
    plt.tight_layout()
    plt.savefig('results/plots/model_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico de características salvo em: results/plots/model_characteristics.png")

def plot_training_stability():
    """Plot de estabilidade do treinamento"""
    baseline_metrics = load_metrics('results/metrics/baseline_gpt2_small_metrics.json')
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Variabilidade do loss (desvio padrão móvel)
    def rolling_std(data, window=10):
        return [np.std(data[max(0, i-window):i+1]) for i in range(len(data))]
    
    std_baseline = rolling_std(baseline_metrics['train_losses'])
    std_moe = rolling_std(moe_metrics['train_losses'])
    
    steps_baseline = list(range(len(std_baseline)))
    steps_moe = list(range(len(std_moe)))
    
    ax1.plot(steps_baseline, std_baseline, label='GPT-2 Baseline', color='#1f77b4', linewidth=2)
    ax1.plot(steps_moe, std_moe, label='Qwen-MoE', color='#ff7f0e', linewidth=2)
    ax1.set_xlabel('Passos de Avaliação')
    ax1.set_ylabel('Desvio Padrão do Loss')
    ax1.set_title('Variabilidade do Treinamento')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribuição dos valores de loss
    ax2.hist(baseline_metrics['train_losses'], bins=30, alpha=0.7, 
             label='GPT-2 Baseline', color='#1f77b4', density=True)
    ax2.hist(moe_metrics['train_losses'], bins=30, alpha=0.7, 
             label='Qwen-MoE', color='#ff7f0e', density=True)
    ax2.set_xlabel('Valor do Loss')
    ax2.set_ylabel('Densidade')
    ax2.set_title('Distribuição dos Valores de Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Taxa de mudança do loss
    def compute_change_rate(losses):
        return [abs(losses[i] - losses[i-1]) if i > 0 else 0 for i in range(len(losses))]
    
    change_baseline = compute_change_rate(baseline_metrics['train_losses'])
    change_moe = compute_change_rate(moe_metrics['train_losses'])
    
    ax3.plot(steps_baseline, change_baseline, label='GPT-2 Baseline', 
             color='#1f77b4', linewidth=2, alpha=0.7)
    ax3.plot(steps_moe, change_moe, label='Qwen-MoE', 
             color='#ff7f0e', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Passos de Avaliação')
    ax3.set_ylabel('|Δ Loss|')
    ax3.set_title('Magnitude da Mudança do Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Box plot comparativo
    data_to_plot = [baseline_metrics['train_losses'], moe_metrics['train_losses']]
    box_plot = ax4.boxplot(data_to_plot, labels=['GPT-2\nBaseline', 'Qwen-MoE'], 
                          patch_artist=True)
    
    colors = ['#1f77b4', '#ff7f0e']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Loss de Treinamento')
    ax4.set_title('Distribuição Comparativa do Loss')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/training_stability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico de estabilidade salvo em: results/plots/training_stability.png")

def plot_performance_summary():
    """Plot resumo de performance"""
    baseline_metrics = load_metrics('results/metrics/baseline_gpt2_small_metrics.json')
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Métricas normalizadas para comparação
    metrics = {
        'Perplexidade Final\n(menor melhor)': {
            'baseline': baseline_metrics['test_ppl'],
            'moe': moe_metrics['test_ppl'],
            'invert': True  # Menor é melhor
        },
        'Throughput\n(maior melhor)': {
            'baseline': baseline_metrics['avg_tps'],
            'moe': moe_metrics['avg_tps'],
            'invert': False
        },
        'Eficiência Memória\n(maior melhor)': {
            'baseline': baseline_metrics['avg_tps'] / baseline_metrics['peak_mem_gb'],
            'moe': moe_metrics['avg_tps'] / moe_metrics['peak_mem_gb'],
            'invert': False
        },
        'Loss Final\n(menor melhor)': {
            'baseline': baseline_metrics['train_losses'][-1],
            'moe': moe_metrics['train_losses'][-1],
            'invert': True
        },
        'Estabilidade\n(menor melhor)': {
            'baseline': np.std(baseline_metrics['train_losses'][-20:]),  # Últimos 20 pontos
            'moe': np.std(moe_metrics['train_losses'][-20:]),
            'invert': True
        }
    }
    
    # Normalizar métricas (0-1 scale)
    normalized_metrics = {}
    for metric, values in metrics.items():
        baseline_val = values['baseline']
        moe_val = values['moe']
        
        if values['invert']:
            # Para métricas onde menor é melhor, invertemos
            max_val = max(baseline_val, moe_val)
            normalized_baseline = 1 - (baseline_val / max_val)
            normalized_moe = 1 - (moe_val / max_val)
        else:
            # Para métricas onde maior é melhor
            max_val = max(baseline_val, moe_val)
            normalized_baseline = baseline_val / max_val
            normalized_moe = moe_val / max_val
        
        normalized_metrics[metric] = {
            'baseline': normalized_baseline,
            'moe': normalized_moe
        }
    
    # Criar radar chart
    labels = list(normalized_metrics.keys())
    baseline_scores = [normalized_metrics[label]['baseline'] for label in labels]
    moe_scores = [normalized_metrics[label]['moe'] for label in labels]
    
    # Ângulos para o radar chart
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    baseline_scores += baseline_scores[:1]  # Fechar o círculo
    moe_scores += moe_scores[:1]
    angles += angles[:1]
    
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, baseline_scores, 'o-', linewidth=2, label='GPT-2 Baseline', color='#1f77b4')
    ax.fill(angles, baseline_scores, alpha=0.25, color='#1f77b4')
    ax.plot(angles, moe_scores, 'o-', linewidth=2, label='Qwen-MoE', color='#ff7f0e')
    ax.fill(angles, moe_scores, alpha=0.25, color='#ff7f0e')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Comparativo Geral de Performance\n(Métricas Normalizadas)', size=16, pad=20)
    
    plt.tight_layout()
    plt.savefig('results/plots/performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico resumo salvo em: results/plots/performance_summary.png")

def main():
    """Função principal para gerar todos os plots"""
    print("🎨 Gerando plots de análise comparativa...")
    print("=" * 50)
    
    # Criar diretório de plots se não existir
    os.makedirs('results/plots', exist_ok=True)
    
    # Gerar todos os plots
    plot_comparative_convergence()
    plot_learning_efficiency() 
    plot_model_characteristics()
    plot_training_stability()
    plot_performance_summary()
    
    print("=" * 50)
    print("✅ Todos os plots foram gerados com sucesso!")
    print("\nArquivos criados:")
    print("📊 results/plots/comparative_analysis.png - Análise comparativa geral")
    print("📈 results/plots/learning_efficiency.png - Eficiência de aprendizado")
    print("🔧 results/plots/model_characteristics.png - Características dos modelos")
    print("📉 results/plots/training_stability.png - Estabilidade do treinamento")
    print("🏆 results/plots/performance_summary.png - Resumo de performance")

if __name__ == "__main__":
    main()