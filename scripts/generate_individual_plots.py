#!/usr/bin/env python3
"""
Script para gerar plots individuais de análise comparativa entre GPT-2 baseline e Qwen-MoE
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

def plot_loss_comparison():
    """Plot comparativo de loss de treinamento"""
    baseline_metrics = load_metrics('results/metrics/baseline_gpt2_small_metrics.json')
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    plt.figure(figsize=(12, 6))
    
    steps_baseline = list(range(len(baseline_metrics['train_losses'])))
    steps_moe = list(range(len(moe_metrics['train_losses'])))
    
    plt.plot(steps_baseline, baseline_metrics['train_losses'], 
             label='GPT-2 Baseline', color='#1f77b4', linewidth=2)
    plt.plot(steps_moe, moe_metrics['train_losses'], 
             label='Qwen-MoE (6 experts)', color='#ff7f0e', linewidth=2)
    
    plt.xlabel('Passos de Avaliação')
    plt.ylabel('Loss de Treinamento')
    plt.title('Comparativo de Convergência - Loss de Treinamento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/individual/loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Loss comparison: results/plots/individual/loss_comparison.png")

def plot_tokens_vs_loss():
    """Plot de loss vs tokens processados"""
    baseline_metrics = load_metrics('results/metrics/baseline_gpt2_small_metrics.json')
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    plt.figure(figsize=(12, 6))
    
    tokens_baseline = np.array(baseline_metrics['tokens_seen'])/1e6
    tokens_moe = np.array(moe_metrics['tokens_seen'])/1e6
    
    plt.plot(tokens_baseline, baseline_metrics['train_losses'], 
             label='GPT-2 Baseline', color='#1f77b4', linewidth=2)
    plt.plot(tokens_moe, moe_metrics['train_losses'], 
             label='Qwen-MoE (6 experts)', color='#ff7f0e', linewidth=2)
    
    plt.xlabel('Tokens Processados (Milhões)')
    plt.ylabel('Loss de Treinamento')
    plt.title('Loss vs Tokens Processados')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/individual/tokens_vs_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Tokens vs Loss: results/plots/individual/tokens_vs_loss.png")

def plot_perplexity_comparison():
    """Plot comparativo de perplexidade final"""
    baseline_metrics = load_metrics('results/metrics/baseline_gpt2_small_metrics.json')
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    plt.figure(figsize=(8, 6))
    
    models = ['GPT-2\nBaseline', 'Qwen-MoE\n(6 experts)']
    perplexities = [baseline_metrics['test_ppl'], moe_metrics['test_ppl']]
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = plt.bar(models, perplexities, color=colors, alpha=0.7, width=0.6)
    plt.ylabel('Perplexidade de Teste')
    plt.title('Comparativo de Perplexidade Final')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, ppl in zip(bars, perplexities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{ppl:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('results/plots/individual/perplexity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Perplexity comparison: results/plots/individual/perplexity_comparison.png")

def plot_throughput_comparison():
    """Plot comparativo de throughput"""
    baseline_metrics = load_metrics('results/metrics/baseline_gpt2_small_metrics.json')
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    plt.figure(figsize=(8, 6))
    
    models = ['GPT-2\nBaseline', 'Qwen-MoE\n(6 experts)']
    throughputs = [baseline_metrics['avg_tps'], moe_metrics['avg_tps']]
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = plt.bar(models, throughputs, color=colors, alpha=0.7, width=0.6)
    plt.ylabel('Tokens por Segundo')
    plt.title('Comparativo de Throughput de Treinamento')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, tps in zip(bars, throughputs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{tps:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/plots/individual/throughput_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Throughput comparison: results/plots/individual/throughput_comparison.png")

def plot_memory_usage():
    """Plot comparativo de uso de memória"""
    baseline_metrics = load_metrics('results/metrics/baseline_gpt2_small_metrics.json')
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    plt.figure(figsize=(8, 6))
    
    models = ['GPT-2\nBaseline', 'Qwen-MoE\n(6 experts)']
    memory_usage = [baseline_metrics['peak_mem_gb'], moe_metrics['peak_mem_gb']]
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = plt.bar(models, memory_usage, color=colors, alpha=0.7, width=0.6)
    plt.ylabel('Memória de Pico (GB)')
    plt.title('Comparativo de Consumo de Memória')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, mem in zip(bars, memory_usage):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{mem:.1f} GB', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('results/plots/individual/memory_usage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Memory usage: results/plots/individual/memory_usage.png")

def plot_efficiency():
    """Plot de eficiência computacional (tokens/seg por GB)"""
    baseline_metrics = load_metrics('results/metrics/baseline_gpt2_small_metrics.json')
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    plt.figure(figsize=(8, 6))
    
    models = ['GPT-2\nBaseline', 'Qwen-MoE\n(6 experts)']
    throughputs = [baseline_metrics['avg_tps'], moe_metrics['avg_tps']]
    memory_usage = [baseline_metrics['peak_mem_gb'], moe_metrics['peak_mem_gb']]
    efficiency = [t/m for t, m in zip(throughputs, memory_usage)]
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = plt.bar(models, efficiency, color=colors, alpha=0.7, width=0.6)
    plt.ylabel('Tokens/seg por GB de Memória')
    plt.title('Eficiência Computacional')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, eff in zip(bars, efficiency):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{eff:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/plots/individual/computational_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Computational efficiency: results/plots/individual/computational_efficiency.png")

def plot_smoothed_learning_curve():
    """Plot de curva de aprendizado suavizada"""
    baseline_metrics = load_metrics('results/metrics/baseline_gpt2_small_metrics.json')
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    plt.figure(figsize=(12, 6))
    
    # Suavizar curvas com média móvel
    def smooth_curve(y, window=10):
        return np.convolve(y, np.ones(window)/window, mode='valid')
    
    tokens_baseline = np.array(baseline_metrics['tokens_seen'])/1e6
    tokens_moe = np.array(moe_metrics['tokens_seen'])/1e6
    
    losses_baseline_smooth = smooth_curve(baseline_metrics['train_losses'])
    losses_moe_smooth = smooth_curve(moe_metrics['train_losses'])
    
    tokens_baseline_smooth = tokens_baseline[9:]  # Ajustar para o tamanho da suavização
    tokens_moe_smooth = tokens_moe[9:]
    
    # Curvas suavizadas
    plt.plot(tokens_baseline_smooth, losses_baseline_smooth, 
             label='GPT-2 Baseline (suavizado)', color='#1f77b4', linewidth=3)
    plt.plot(tokens_moe_smooth, losses_moe_smooth, 
             label='Qwen-MoE (suavizado)', color='#ff7f0e', linewidth=3)
    
    # Pontos originais mais transparentes
    plt.plot(tokens_baseline, baseline_metrics['train_losses'], 
             color='#1f77b4', alpha=0.2, linewidth=1, label='GPT-2 (original)')
    plt.plot(tokens_moe, moe_metrics['train_losses'], 
             color='#ff7f0e', alpha=0.2, linewidth=1, label='Qwen-MoE (original)')
    
    plt.xlabel('Tokens Processados (Milhões)')
    plt.ylabel('Loss de Treinamento')
    plt.title('Curva de Aprendizado Suavizada')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/individual/smoothed_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Smoothed learning curve: results/plots/individual/smoothed_learning_curve.png")

def plot_improvement_rate():
    """Plot de taxa de melhoria por token"""
    baseline_metrics = load_metrics('results/metrics/baseline_gpt2_small_metrics.json')
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    plt.figure(figsize=(12, 6))
    
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
    
    tokens_baseline = np.array(baseline_metrics['tokens_seen'])/1e6
    tokens_moe = np.array(moe_metrics['tokens_seen'])/1e6
    
    rates_baseline = compute_improvement_rate(tokens_baseline, baseline_metrics['train_losses'])
    rates_moe = compute_improvement_rate(tokens_moe, moe_metrics['train_losses'])
    
    tokens_rates_baseline = tokens_baseline[5:]
    tokens_rates_moe = tokens_moe[5:]
    
    plt.plot(tokens_rates_baseline, rates_baseline, 
             label='GPT-2 Baseline', color='#1f77b4', linewidth=2)
    plt.plot(tokens_rates_moe, rates_moe, 
             label='Qwen-MoE (6 experts)', color='#ff7f0e', linewidth=2)
    
    plt.xlabel('Tokens Processados (Milhões)')
    plt.ylabel('Taxa de Melhoria (Δloss/Mtokens)')
    plt.title('Taxa de Melhoria por Token')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('results/plots/individual/improvement_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Improvement rate: results/plots/individual/improvement_rate.png")

def plot_training_variability():
    """Plot de variabilidade do treinamento"""
    baseline_metrics = load_metrics('results/metrics/baseline_gpt2_small_metrics.json')
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    plt.figure(figsize=(12, 6))
    
    # Variabilidade do loss (desvio padrão móvel)
    def rolling_std(data, window=10):
        return [np.std(data[max(0, i-window):i+1]) for i in range(len(data))]
    
    std_baseline = rolling_std(baseline_metrics['train_losses'])
    std_moe = rolling_std(moe_metrics['train_losses'])
    
    steps_baseline = list(range(len(std_baseline)))
    steps_moe = list(range(len(std_moe)))
    
    plt.plot(steps_baseline, std_baseline, label='GPT-2 Baseline', color='#1f77b4', linewidth=2)
    plt.plot(steps_moe, std_moe, label='Qwen-MoE (6 experts)', color='#ff7f0e', linewidth=2)
    plt.xlabel('Passos de Avaliação')
    plt.ylabel('Desvio Padrão do Loss (Janela Móvel)')
    plt.title('Variabilidade do Treinamento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/individual/training_variability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Training variability: results/plots/individual/training_variability.png")

def plot_loss_distribution():
    """Plot de distribuição dos valores de loss"""
    baseline_metrics = load_metrics('results/metrics/baseline_gpt2_small_metrics.json')
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(baseline_metrics['train_losses'], bins=30, alpha=0.7, 
             label='GPT-2 Baseline', color='#1f77b4', density=True)
    plt.hist(moe_metrics['train_losses'], bins=30, alpha=0.7, 
             label='Qwen-MoE (6 experts)', color='#ff7f0e', density=True)
    
    plt.xlabel('Valor do Loss')
    plt.ylabel('Densidade')
    plt.title('Distribuição dos Valores de Loss Durante o Treinamento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/individual/loss_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Loss distribution: results/plots/individual/loss_distribution.png")

def plot_loss_boxplot():
    """Box plot comparativo dos valores de loss"""
    baseline_metrics = load_metrics('results/metrics/baseline_gpt2_small_metrics.json')
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    plt.figure(figsize=(8, 6))
    
    data_to_plot = [baseline_metrics['train_losses'], moe_metrics['train_losses']]
    box_plot = plt.boxplot(data_to_plot, tick_labels=['GPT-2\nBaseline', 'Qwen-MoE\n(6 experts)'], 
                          patch_artist=True)
    
    colors = ['#1f77b4', '#ff7f0e']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('Loss de Treinamento')
    plt.title('Distribuição Comparativa do Loss')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/plots/individual/loss_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Loss boxplot: results/plots/individual/loss_boxplot.png")

def plot_moe_parameters():
    """Plot dos parâmetros específicos do modelo MoE"""
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    plt.figure(figsize=(10, 6))
    
    if 'num_experts' in moe_metrics:
        moe_chars = ['Número de\nExperts', 'Top-K', 'Capacity\nFactor', 'Aux Loss\nWeight']
        moe_values = [moe_metrics.get('num_experts', 0), 
                     moe_metrics.get('top_k', 0),
                     moe_metrics.get('capacity_factor', 0),
                     moe_metrics.get('aux_loss_weight', 0)]
        
        bars = plt.bar(moe_chars, moe_values, color='#ff7f0e', alpha=0.7, width=0.6)
        plt.ylabel('Valores')
        plt.title('Parâmetros do Modelo Mixture of Experts (MoE)')
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, moe_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(moe_values)*0.02,
                    f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/plots/individual/moe_parameters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ MoE parameters: results/plots/individual/moe_parameters.png")

def plot_performance_radar():
    """Radar chart de performance normalizada"""
    baseline_metrics = load_metrics('results/metrics/baseline_gpt2_small_metrics.json')
    moe_metrics = load_metrics('results/metrics/qwen_moe_6experts_top1_metrics.json')
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Métricas normalizadas para comparação
    metrics = {
        'Perplexidade\n(menor melhor)': {
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
    
    ax.plot(angles, baseline_scores, 'o-', linewidth=3, label='GPT-2 Baseline', color='#1f77b4')
    ax.fill(angles, baseline_scores, alpha=0.25, color='#1f77b4')
    ax.plot(angles, moe_scores, 'o-', linewidth=3, label='Qwen-MoE', color='#ff7f0e')
    ax.fill(angles, moe_scores, alpha=0.25, color='#ff7f0e')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Comparativo Geral de Performance\n(Métricas Normalizadas)', size=16, pad=30)
    
    plt.tight_layout()
    plt.savefig('results/plots/individual/performance_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Performance radar: results/plots/individual/performance_radar.png")

def main():
    """Função principal para gerar todos os plots individuais"""
    print("🎨 Gerando plots individuais de análise comparativa...")
    print("=" * 60)
    
    # Criar diretório de plots individuais se não existir
    os.makedirs('results/plots/individual', exist_ok=True)
    
    # Gerar todos os plots individuais
    plot_loss_comparison()
    plot_tokens_vs_loss()
    plot_perplexity_comparison()
    plot_throughput_comparison()
    plot_memory_usage()
    plot_efficiency()
    plot_smoothed_learning_curve()
    plot_improvement_rate()
    plot_training_variability()
    plot_loss_distribution()
    plot_loss_boxplot()
    plot_moe_parameters()
    plot_performance_radar()
    
    print("=" * 60)
    print("✅ Todos os plots individuais foram gerados com sucesso!")
    print("\n📁 Pasta: results/plots/individual/")
    print("🔍 Plots de comparação:")
    print("   • loss_comparison.png - Comparativo de convergência")
    print("   • tokens_vs_loss.png - Loss vs tokens processados")
    print("   • perplexity_comparison.png - Perplexidade final")
    print("   • throughput_comparison.png - Velocidade de processamento")
    print("   • memory_usage.png - Consumo de memória")
    print("   • computational_efficiency.png - Eficiência computacional")
    print("\n📈 Plots de análise de treinamento:")
    print("   • smoothed_learning_curve.png - Curva de aprendizado suavizada")
    print("   • improvement_rate.png - Taxa de melhoria por token")
    print("   • training_variability.png - Estabilidade do treinamento")
    print("   • loss_distribution.png - Distribuição dos valores de loss")
    print("   • loss_boxplot.png - Box plot comparativo")
    print("\n🔧 Plots específicos:")
    print("   • moe_parameters.png - Parâmetros do modelo MoE")
    print("   • performance_radar.png - Radar chart de performance")

if __name__ == "__main__":
    main()