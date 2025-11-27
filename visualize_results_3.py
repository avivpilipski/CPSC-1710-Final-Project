"""
Create visualizations comparing all three GPT-2 models
"""
import sys
sys.path.append('src')

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 9)
plt.rcParams['font.size'] = 10


def load_results(json_path):
    """Load results from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_three_model_comparison(results_dict, output_path):
    """
    Plot all three models side-by-side for all occupations
    """
    occupations = list(results_dict[list(results_dict.keys())[0]]['bias_scores'].keys())
    
    data = []
    for model_name, results in results_dict.items():
        for occupation in occupations:
            if occupation in results['bias_scores']:
                data.append({
                    'Occupation': occupation,
                    'Model': model_name,
                    'Bias Score': results['bias_scores'][occupation]['bias_score']
                })
    
    df = pd.DataFrame(data)
    
    # Sort by absolute bias in large model
    large_data = df[df['Model'] == 'gpt2-large'].copy()
    large_data['abs_bias'] = large_data['Bias Score'].abs()
    occupation_order = large_data.sort_values('abs_bias', ascending=False)['Occupation'].tolist()
    
    fig, ax = plt.subplots(figsize=(16, 11))
    
    x = np.arange(len(occupation_order))
    width = 0.25
    
    colors = {'gpt2': '#3498db', 'gpt2-medium': '#e74c3c', 'gpt2-large': '#f39c12'}
    model_order = ['gpt2', 'gpt2-medium', 'gpt2-large']
    
    for i, model_name in enumerate(model_order):
        model_data = df[df['Model'] == model_name]
        scores = [model_data[model_data['Occupation'] == occ]['Bias Score'].values[0] 
                 for occ in occupation_order]
        
        offset = width * (i - 1)
        bars = ax.bar(x + offset, scores, width, label=model_name, 
                     color=colors.get(model_name, f'C{i}'), alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Occupation', fontsize=13, fontweight='bold')
    ax.set_ylabel('Bias Score (+ = Male, - = Female)', fontsize=13, fontweight='bold')
    ax.set_title('Gender Bias Across All Three GPT-2 Models\n(All 30 Occupations, Sorted by GPT-2 Large Bias)', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(occupation_order, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved three-model comparison plot to {output_path}")
    plt.close()


def plot_weat_exponential(results_dict, output_path):
    """
    Plot WEAT effect sizes showing exponential growth
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Linear plot
    models = ['GPT-2\n(124M)', 'GPT-2 Medium\n(355M)', 'GPT-2 Large\n(774M)']
    model_keys = ['gpt2', 'gpt2-medium', 'gpt2-large']
    effect_sizes = [results_dict[m]['weat_effect_size'] for m in model_keys]
    p_values = [results_dict[m]['weat_p_value'] for m in model_keys]
    params = [124, 355, 774]
    
    colors_list = ['#2ecc71', '#e74c3c', '#f39c12']
    
    # Left: Linear effect size
    bars1 = ax1.bar(models, effect_sizes, color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, effect, p in zip(bars1, effect_sizes, p_values):
        height = bar.get_height()
        sig = "ns" if p > 0.05 else "*" if p < 0.01 else "**" if p < 0.001 else "***"
        ax1.text(bar.get_x() + bar.get_width()/2., height,
               f'{effect:.4f}\n({sig})',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.axhline(y=0.2, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Small Effect (d=0.2)')
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Medium Effect (d=0.5)')
    ax1.axhline(y=0.8, color='darkred', linestyle='--', linewidth=2, alpha=0.5, label='Large Effect (d=0.8)')
    
    ax1.set_ylabel('WEAT Effect Size (Cohen\'s d)', fontsize=12, fontweight='bold')
    ax1.set_title('WEAT Effect Size Across Models', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Log scale to show exponential growth
    ax2.semilogy(params, effect_sizes, 'o-', linewidth=3, markersize=12, 
                color='#e74c3c', markerfacecolor='#f39c12', markeredgecolor='black', markeredgewidth=2)
    
    for x, y, label in zip(params, effect_sizes, models):
        ax2.text(x, y*1.5, f'{y:.4f}', ha='center', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Model Parameters (Millions)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('WEAT Effect Size (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Exponential Growth in Bias with Model Size', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved exponential growth plot to {output_path}")
    plt.close()


def plot_direction_flips(results_dict, output_path):
    """
    Plot occupations that changed direction between models
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    model_keys = ['gpt2', 'gpt2-medium', 'gpt2-large']
    occupations = list(results_dict['gpt2']['bias_scores'].keys())
    
    # Find direction flips
    data = []
    for occ in occupations:
        scores = [results_dict[m]['bias_scores'][occ]['bias_score'] for m in model_keys]
        directions = ['M' if s > 0 else 'F' if s < 0 else 'N' for s in scores]
        
        # Check if direction changed
        if directions[0] != directions[-1]:  # Between GPT-2 and Large
            data.append({
                'Occupation': occ.capitalize(),
                'GPT-2': scores[0],
                'Medium': scores[1],
                'Large': scores[2]
            })
    
    if data:
        df = pd.DataFrame(data).sort_values('Large', key=abs, ascending=False)
        
        x = np.arange(len(df))
        width = 0.25
        
        ax.bar(x - width, df['GPT-2'], width, label='GPT-2', color='#3498db', alpha=0.8, edgecolor='black')
        ax.bar(x, df['Medium'], width, label='GPT-2 Medium', color='#e74c3c', alpha=0.8, edgecolor='black')
        ax.bar(x + width, df['Large'], width, label='GPT-2 Large', color='#f39c12', alpha=0.8, edgecolor='black')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        ax.set_xlabel('Occupation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Bias Score', fontsize=12, fontweight='bold')
        ax.set_title('Occupations with Direction Flips: Female→Male\n(Traditional Female Occupations Becoming Male-Biased)',
                    fontsize=13, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Occupation'], rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add annotations
        for i, (idx, row) in enumerate(df.iterrows()):
            ax.annotate('', xy=(i, row['Large']), xytext=(i, row['GPT-2']),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved direction flips plot to {output_path}")
    plt.close()


def plot_mean_bias_trend(results_dict, output_path):
    """
    Plot mean bias trend across models
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    model_keys = ['gpt2', 'gpt2-medium', 'gpt2-large']
    model_names = ['GPT-2\n(124M)', 'GPT-2 Medium\n(355M)', 'GPT-2 Large\n(774M)']
    params = [124, 355, 774]
    
    mean_biases = [results_dict[m]['mean_bias'] for m in model_keys]
    std_biases = [results_dict[m]['std_bias'] for m in model_keys]
    
    # Plot with error bars
    ax.errorbar(params, mean_biases, yerr=std_biases, fmt='o-', linewidth=3, markersize=12,
               color='#e74c3c', ecolor='#c0392b', elinewidth=2, capsize=8, capthick=2,
               markerfacecolor='#f39c12', markeredgecolor='black', markeredgewidth=2)
    
    # Add value labels
    for x, y, std in zip(params, mean_biases, std_biases):
        ax.text(x, y + std + 0.0005, f'{y:.4f}', ha='center', fontsize=11, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.3)
    ax.set_xlabel('Model Parameters (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Bias Score Across All Occupations', fontsize=12, fontweight='bold')
    ax.set_title('Mean Gender Bias Increases with Model Size', fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Add percentage increase annotations
    pct_increase_1 = ((mean_biases[1] - mean_biases[0]) / abs(mean_biases[0])) * 100
    pct_increase_2 = ((mean_biases[2] - mean_biases[1]) / mean_biases[1]) * 100
    
    ax.text(0.5, 0.95, f'124M→355M: +{pct_increase_1:.0f}%\n355M→774M: +{pct_increase_2:.0f}%',
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved mean bias trend plot to {output_path}")
    plt.close()


def create_three_model_visualizations():
    """Create all three-model visualizations"""
    print("\n" + "="*70)
    print("CREATING THREE-MODEL COMPARISON VISUALIZATIONS")
    print("="*70 + "\n")
    
    results_dir = Path("results")
    
    # Load all three models
    results_dict = {}
    for model_name, pattern in [('gpt2', 'gpt2_*'), ('gpt2-medium', 'gpt2-medium*'), ('gpt2-large', 'gpt2-large*')]:
        files = list(results_dir.glob(f"{pattern}.json"))
        if files:
            # Get the most recent file
            latest_file = max(files, key=lambda p: p.stat().st_mtime)
            print(f"Loading {model_name} from: {latest_file.name}")
            results_dict[model_name] = load_results(latest_file)
        else:
            print(f"⚠ Could not find {model_name} results file")
    
    if len(results_dict) < 3:
        print(f"\n✗ Need all 3 models. Found {len(results_dict)}")
        return
    
    output_dir = Path("results/visualizations")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating three-model comparison plots...\n")
    
    # Create plots
    plot_three_model_comparison(results_dict, output_dir / "06_three_model_comparison.png")
    plot_weat_exponential(results_dict, output_dir / "07_weat_exponential_growth.png")
    plot_direction_flips(results_dict, output_dir / "08_direction_flips.png")
    plot_mean_bias_trend(results_dict, output_dir / "09_mean_bias_trend.png")
    
    print("\n" + "="*70)
    print("✅ THREE-MODEL COMPARISON VISUALIZATIONS CREATED")
    print("="*70)
    print(f"\nSaved to: {output_dir}/")
    print("\nGenerated plots:")
    print("  6. 06_three_model_comparison.png - All occupations across 3 models")
    print("  7. 07_weat_exponential_growth.png - Exponential effect size growth")
    print("  8. 08_direction_flips.png - Occupations that changed direction")
    print("  9. 09_mean_bias_trend.png - Mean bias trend with model size")


if __name__ == "__main__":
    create_three_model_visualizations()