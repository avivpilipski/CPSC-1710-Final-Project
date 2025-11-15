"""
Create visualizations comparing gender bias across models
"""
import sys
sys.path.append('src')

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_results(json_path):
    """Load results from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_bias_comparison(results_dict, output_path):
    """
    Plot 1: Side-by-side comparison of bias scores for all occupations
    """
    # Prepare data
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
    
    # Sort by absolute bias in first model
    first_model = list(results_dict.keys())[0]
    first_model_data = df[df['Model'] == first_model].copy()
    first_model_data['abs_bias'] = first_model_data['Bias Score'].abs()
    occupation_order = first_model_data.sort_values('abs_bias', ascending=False)['Occupation'].tolist()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    x = np.arange(len(occupation_order))
    width = 0.35
    
    colors = {'gpt2': '#3498db', 'gpt2-medium': '#e74c3c'}
    
    for i, model_name in enumerate(results_dict.keys()):
        model_data = df[df['Model'] == model_name]
        scores = [model_data[model_data['Occupation'] == occ]['Bias Score'].values[0] 
                 for occ in occupation_order]
        
        offset = width * (i - 0.5)
        bars = ax.bar(x + offset, scores, width, label=model_name, 
                     color=colors.get(model_name, f'C{i}'), alpha=0.8)
        
        # Add value labels on bars
        for j, (bar, score) in enumerate(zip(bars, scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:+.3f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=7, rotation=90)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Occupation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bias Score (+ = Male, - = Female)', fontsize=12, fontweight='bold')
    ax.set_title('Gender Bias Comparison Across Models\n(All 30 Occupations)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(occupation_order, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved bias comparison plot to {output_path}")
    plt.close()


def plot_weat_comparison(results_dict, output_path):
    """
    Plot 2: WEAT Effect Size comparison bar chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results_dict.keys())
    effect_sizes = [results_dict[m]['weat_effect_size'] for m in models]
    p_values = [results_dict[m]['weat_p_value'] for m in models]
    
    colors = ['#2ecc71' if p > 0.05 else '#e74c3c' for p in p_values]
    
    bars = ax.bar(models, effect_sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, effect, p in zip(bars, effect_sizes, p_values):
        height = bar.get_height()
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{effect:.4f}\n({sig})',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add significance threshold line
    ax.axhline(y=0.2, color='orange', linestyle='--', linewidth=2, 
              label='Small Effect Threshold (d=0.2)')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2,
              label='Medium Effect Threshold (d=0.5)')
    
    ax.set_ylabel('WEAT Effect Size (Cohen\'s d)', fontsize=12, fontweight='bold')
    ax.set_title('Statistical Bias Comparison: WEAT Effect Sizes\n' + 
                'Green = Not Significant (p>0.05), Red = Significant (p<0.05)',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved WEAT comparison plot to {output_path}")
    plt.close()


def plot_top_biased_occupations(results_dict, output_path, top_n=10):
    """
    Plot 3: Top N most biased occupations for each model
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 8))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, results) in zip(axes, results_dict.items()):
        # Get top N by absolute bias
        top_occupations = results['most_biased'][:top_n]
        
        occupations = [occ for occ, _ in top_occupations]
        scores = [data['bias_score'] for _, data in top_occupations]
        
        # Color by direction
        colors = ['#3498db' if score > 0 else '#e74c3c' for score in scores]
        
        bars = ax.barh(range(len(occupations)), scores, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            label = f'{score:+.4f}'
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   label, ha='left' if width > 0 else 'right',
                   va='center', fontsize=9, fontweight='bold')
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_yticks(range(len(occupations)))
        ax.set_yticklabels(occupations)
        ax.set_xlabel('Bias Score (+ = Male, - = Female)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}\nTop {top_n} Most Biased', 
                    fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    
    plt.suptitle('Most Gender-Biased Occupations by Model', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved top biased occupations plot to {output_path}")
    plt.close()


def plot_bias_distribution(results_dict, output_path):
    """
    Plot 4: Distribution of bias scores
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    ax1 = axes[0]
    for model_name, results in results_dict.items():
        scores = [data['bias_score'] for data in results['bias_scores'].values()]
        ax1.hist(scores, bins=15, alpha=0.6, label=model_name, edgecolor='black')
    
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax1.set_xlabel('Bias Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Bias Scores', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    data_for_box = []
    labels = []
    for model_name, results in results_dict.items():
        scores = [data['bias_score'] for data in results['bias_scores'].values()]
        data_for_box.append(scores)
        labels.append(model_name)
    
    bp = ax2.boxplot(data_for_box, labels=labels, patch_artist=True,
                     showmeans=True, meanline=True)
    
    # Color the boxes
    colors_list = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    for patch, color in zip(bp['boxes'], colors_list[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax2.set_ylabel('Bias Score', fontsize=12, fontweight='bold')
    ax2.set_title('Bias Score Distribution by Model', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved distribution plot to {output_path}")
    plt.close()


def plot_amplification_analysis(gpt2_results, gpt2_medium_results, output_path):
    """
    Plot 5: Bias amplification scatter plot (GPT-2 vs GPT-2 Medium)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    occupations = []
    gpt2_scores = []
    medium_scores = []
    
    for occupation in gpt2_results['bias_scores'].keys():
        if occupation in gpt2_medium_results['bias_scores']:
            occupations.append(occupation)
            gpt2_scores.append(gpt2_results['bias_scores'][occupation]['bias_score'])
            medium_scores.append(gpt2_medium_results['bias_scores'][occupation]['bias_score'])
    
    # Create scatter
    colors = ['red' if (g2 * med > 0 and abs(med) > abs(g2)) else 'blue' 
             for g2, med in zip(gpt2_scores, medium_scores)]
    
    scatter = ax.scatter(gpt2_scores, medium_scores, c=colors, s=100, alpha=0.6, edgecolors='black')
    
    # Add occupation labels
    for occ, x, y in zip(occupations, gpt2_scores, medium_scores):
        ax.annotate(occ, (x, y), fontsize=8, alpha=0.7, 
                   xytext=(5, 5), textcoords='offset points')
    
    # Add diagonal line (no change)
    max_val = max(max(abs(min(gpt2_scores)), abs(max(gpt2_scores))),
                  max(abs(min(medium_scores)), abs(max(medium_scores))))
    ax.plot([-max_val, max_val], [-max_val, max_val], 'k--', alpha=0.3, linewidth=2,
           label='No Change Line')
    
    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('GPT-2 Base Bias Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('GPT-2 Medium Bias Score', fontsize=12, fontweight='bold')
    ax.set_title('Bias Amplification: GPT-2 → GPT-2 Medium\n' +
                'Red = Amplified | Blue = Reduced/Flipped',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved amplification analysis plot to {output_path}")
    plt.close()


def create_all_visualizations():
    """Create all visualizations"""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Find result files
    results_dir = Path("results")
    gpt2_file = None
    gpt2_medium_file = None
    
    for f in results_dir.glob("*.json"):
        if "gpt2-medium" in f.name:
            gpt2_medium_file = f
        elif "gpt2" in f.name and "medium" not in f.name:
            gpt2_file = f
    
    if not gpt2_file:
        print("✗ Could not find GPT-2 results file")
        return
    
    print(f"Loading GPT-2 results from: {gpt2_file}")
    gpt2_results = load_results(gpt2_file)
    
    results_dict = {"gpt2": gpt2_results}
    
    if gpt2_medium_file:
        print(f"Loading GPT-2 Medium results from: {gpt2_medium_file}")
        gpt2_medium_results = load_results(gpt2_medium_file)
        results_dict["gpt2-medium"] = gpt2_medium_results
    else:
        print("⚠ GPT-2 Medium results not found, creating single-model visualizations only")
        gpt2_medium_results = None
    
    # Create output directory
    output_dir = Path("results/visualizations")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating plots...\n")
    
    # Plot 1: Bias comparison
    plot_bias_comparison(results_dict, output_dir / "01_bias_comparison.png")
    
    # Plot 2: WEAT comparison
    plot_weat_comparison(results_dict, output_dir / "02_weat_comparison.png")
    
    # Plot 3: Top biased occupations
    plot_top_biased_occupations(results_dict, output_dir / "03_top_biased_occupations.png")
    
    # Plot 4: Distribution
    plot_bias_distribution(results_dict, output_dir / "04_bias_distribution.png")
    
    # Plot 5: Amplification (only if both models exist)
    if gpt2_medium_results:
        plot_amplification_analysis(gpt2_results, gpt2_medium_results, 
                                    output_dir / "05_bias_amplification.png")
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS CREATED")
    print("="*70)
    print(f"\nSaved to: {output_dir}/")
    print("\nGenerated plots:")
    print("  1. 01_bias_comparison.png - Side-by-side bias scores for all occupations")
    print("  2. 02_weat_comparison.png - WEAT effect sizes with significance")
    print("  3. 03_top_biased_occupations.png - Top 10 most biased occupations")
    print("  4. 04_bias_distribution.png - Distribution histograms and box plots")
    if gpt2_medium_results:
        print("  5. 05_bias_amplification.png - Amplification scatter plot")


if __name__ == "__main__":
    create_all_visualizations()