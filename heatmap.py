"""
Create heatmap showing occupation bias across all three models
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

sns.set_style("white")
plt.rcParams['figure.figsize'] = (14, 10)


def load_results(json_path):
    """Load results from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_occupation_heatmap():
    """Create heatmap of occupation biases across models"""
    print("\n" + "="*70)
    print("CREATING OCCUPATION BIAS HEATMAP")
    print("="*70 + "\n")
    
    results_dir = Path("results")
    
    # Load results for all three models
    results_dict = {}
    for model_name, pattern in [('GPT-2\n(124M)', 'gpt2_*'), 
                                  ('GPT-2 Medium\n(355M)', 'gpt2-medium*'), 
                                  ('GPT-2 Large\n(774M)', 'gpt2-large*')]:
        files = list(results_dir.glob(f"{pattern}.json"))
        if files:
            # Get the most recent file
            latest_file = max(files, key=lambda p: p.stat().st_mtime)
            print(f"Loading {model_name.replace(chr(10), ' ')} from: {latest_file.name}")
            results_dict[model_name] = load_results(latest_file)
        else:
            print(f"⚠ Could not find results for {model_name.replace(chr(10), ' ')}")
    
    if len(results_dict) < 2:
        print(f"\n✗ Need at least 2 models. Found {len(results_dict)}")
        return
    
    # Extract bias scores for all occupations
    occupations = list(results_dict[list(results_dict.keys())[0]]['bias_scores'].keys())
    
    # Create dataframe
    data = []
    for model_name in results_dict.keys():
        for occupation in occupations:
            if occupation in results_dict[model_name]['bias_scores']:
                bias_score = results_dict[model_name]['bias_scores'][occupation]['bias_score']
                data.append({
                    'Model': model_name,
                    'Occupation': occupation.capitalize(),
                    'Bias Score': bias_score
                })
    
    df = pd.DataFrame(data)
    
    # Pivot for heatmap
    pivot_df = df.pivot(index='Occupation', columns='Model', values='Bias Score')
    
    # Sort by average absolute bias (most biased at top)
    pivot_df['avg_abs_bias'] = pivot_df.abs().mean(axis=1)
    pivot_df = pivot_df.sort_values('avg_abs_bias', ascending=False)
    pivot_df = pivot_df.drop('avg_abs_bias', axis=1)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.3)
    
    # Main heatmap
    ax1 = fig.add_subplot(gs[0])
    
    # Determine color scale (symmetric around 0)
    vmax = max(abs(pivot_df.min().min()), abs(pivot_df.max().max()))
    
    # Create heatmap
    sns.heatmap(pivot_df, 
                cmap='RdBu_r',  # Red = male bias, Blue = female bias
                center=0,
                vmin=-vmax,
                vmax=vmax,
                annot=True,
                fmt='.4f',
                linewidths=0.5,
                linecolor='gray',
                cbar_kws={'label': 'Bias Score\n(+ = Male, - = Female)', 'shrink': 0.8},
                ax=ax1)
    
    ax1.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Occupation', fontsize=13, fontweight='bold')
    ax1.set_title('Gender Bias Across Models and Occupations\nSorted by Average Absolute Bias', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Rotate labels
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, ha='center')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    
    # Add text annotations for interpretation
    ax1.text(0.5, -0.08, 
            'Red = Male-biased | Blue = Female-biased | White = Neutral',
            transform=ax1.transAxes, ha='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Side panel: Show direction consistency
    ax2 = fig.add_subplot(gs[1])
    
    # Calculate how many models agree on direction
    direction_consistency = []
    for occupation in pivot_df.index:
        row = pivot_df.loc[occupation]
        male_count = (row > 0).sum()
        female_count = (row < 0).sum()
        consistency = max(male_count, female_count) / len(row)
        direction_consistency.append(consistency)
    
    consistency_colors = ['#2ecc71' if c == 1.0 else '#f39c12' if c >= 0.67 else '#e74c3c' 
                         for c in direction_consistency]
    
    ax2.barh(range(len(pivot_df)), direction_consistency, color=consistency_colors, 
            edgecolor='black', linewidth=0.5)
    ax2.set_ylim(-0.5, len(pivot_df) - 0.5)
    ax2.set_xlim(0, 1.05)
    ax2.set_yticks(range(len(pivot_df)))
    ax2.set_yticklabels([])
    ax2.set_xlabel('Direction\nConsistency', fontsize=10, fontweight='bold')
    ax2.set_title('All Models\nAgree?', fontsize=11, fontweight='bold')
    ax2.invert_yaxis()
    ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='100% Agreement'),
        Patch(facecolor='#f39c12', label='67%+ Agreement'),
        Patch(facecolor='#e74c3c', label='<67% Agreement')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    output_dir = Path("results/visualizations")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "13_occupation_bias_heatmap.png"
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved occupation bias heatmap to {output_path}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "-"*70)
    print("SUMMARY STATISTICS")
    print("-"*70)
    
    print("\nOccupations with CONSISTENT direction across all models:")
    for occupation in pivot_df.index:
        row = pivot_df.loc[occupation]
        if all(row > 0) or all(row < 0):
            direction = "Male" if all(row > 0) else "Female"
            print(f"  {occupation:20s} → {direction}")
    
    print("\nOccupations that FLIP direction between models:")
    for occupation in pivot_df.index:
        row = pivot_df.loc[occupation]
        if (row > 0).any() and (row < 0).any():
            print(f"  {occupation}")
    
    print("\n" + "="*70)
    print("✅ HEATMAP CREATED")
    print("="*70)


if __name__ == "__main__":
    create_occupation_heatmap()