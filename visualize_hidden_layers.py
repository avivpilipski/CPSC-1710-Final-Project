"""
Create visualizations for hidden layer bias analysis
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 9)
plt.rcParams['font.size'] = 10


def load_hidden_layer_results(json_path):
    """Load hidden layer analysis results"""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_bias_trajectory_all_models(results_dict, output_path):
    """
    Plot bias trajectory across layers for all models
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    colors = {'gpt2': '#3498db', 'gpt2-medium': '#e74c3c', 'gpt2-large': '#f39c12'}
    
    # Plot 1: WEAT Effect Size
    ax1 = axes[0]
    for model_name, results in results_dict.items():
        layer_indices = sorted([int(k) for k in results['layer_results'].keys()])
        effect_sizes = [results['layer_results'][str(i)]['weat_effect_size'] for i in layer_indices]
        
        ax1.plot(layer_indices, effect_sizes, 'o-', linewidth=2.5, markersize=6,
                label=f"{model_name} ({len(layer_indices)} layers)", 
                color=colors.get(model_name, 'gray'), alpha=0.8)
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax1.axhline(y=0.2, color='orange', linestyle='--', linewidth=1, alpha=0.3, label='Small Effect (d=0.2)')
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.3, label='Medium Effect (d=0.5)')
    ax1.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('WEAT Effect Size (Cohen\'s d)', fontsize=12, fontweight='bold')
    ax1.set_title('Gender Bias Trajectory Across Model Layers\nWEAT Effect Size', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(alpha=0.3)
    
    # Plot 2: Mean Bias Score
    ax2 = axes[1]
    for model_name, results in results_dict.items():
        layer_indices = sorted([int(k) for k in results['layer_results'].keys()])
        mean_biases = [results['layer_results'][str(i)]['mean_bias'] for i in layer_indices]
        
        ax2.plot(layer_indices, mean_biases, 'o-', linewidth=2.5, markersize=6,
                label=model_name, color=colors.get(model_name, 'gray'), alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax2.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Bias Score', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Bias Across Layers\n(+ = Male Bias, - = Female Bias)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved layer trajectory plot to {output_path}")
    plt.close()


def plot_three_phase_architecture(results, model_name, output_path):
    """
    Highlight the three-phase architecture (encode → suppress → re-amplify)
    """
    layer_indices = sorted([int(k) for k in results['layer_results'].keys()])
    mean_biases = [results['layer_results'][str(i)]['mean_bias'] for i in layer_indices]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot the bias trajectory
    ax.plot(layer_indices, mean_biases, 'o-', linewidth=3, markersize=8,
           color='#e74c3c', markerfacecolor='#f39c12', markeredgecolor='black', 
           markeredgewidth=1.5, alpha=0.9)
    
    # Annotate the three phases
    n_layers = len(layer_indices)
    
    # Phase 1: Encoding (first 10% of layers)
    phase1_end = max(3, int(n_layers * 0.1))
    ax.axvspan(0, phase1_end, alpha=0.15, color='blue', label='Phase 1: Encoding')
    
    # Phase 2: Suppression (middle 70% of layers)
    phase2_start = phase1_end
    phase2_end = int(n_layers * 0.9)
    ax.axvspan(phase2_start, phase2_end, alpha=0.15, color='green', label='Phase 2: Suppression')
    
    # Phase 3: Re-amplification (last 10% of layers)
    phase3_start = phase2_end
    ax.axvspan(phase3_start, n_layers - 1, alpha=0.15, color='red', label='Phase 3: Re-amplification')
    
    # Mark key points
    embedding_bias = mean_biases[0]
    final_bias = mean_biases[-1]
    min_bias_idx = np.argmin(np.abs(mean_biases))
    
    ax.scatter([0], [embedding_bias], s=200, color='blue', edgecolor='black', 
              linewidth=2, zorder=5, label=f'Embedding: {embedding_bias:.6f}')
    ax.scatter([min_bias_idx], [mean_biases[min_bias_idx]], s=200, color='green', 
              edgecolor='black', linewidth=2, zorder=5, 
              label=f'Min (Layer {min_bias_idx}): {mean_biases[min_bias_idx]:.6f}')
    ax.scatter([n_layers - 1], [final_bias], s=200, color='red', edgecolor='black', 
              linewidth=2, zorder=5, label=f'Final: {final_bias:.6f}')
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Layer Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Bias Score', fontsize=13, fontweight='bold')
    ax.set_title(f'{model_name.upper()}: Three-Phase Bias Architecture\n' + 
                f'Amplification: {embedding_bias:.6f} → {final_bias:.6f} ' +
                f'({abs(final_bias/embedding_bias - 1)*100:.1f}% change)',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved three-phase architecture plot to {output_path}")
    plt.close()


def plot_amplification_comparison(results_dict, output_path):
    """
    Compare amplification across models
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models = []
    initial_biases = []
    final_biases = []
    amplifications = []
    
    for model_name, results in results_dict.items():
        models.append(model_name)
        initial_biases.append(results['summary']['initial_bias'])
        final_biases.append(results['summary']['final_bias'])
        amplifications.append(results['summary']['amplification'])
    
    x = np.arange(len(models))
    width = 0.35
    
    # Plot 1: Initial vs Final
    bars1 = ax1.bar(x - width/2, initial_biases, width, label='Initial (Embedding)', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, final_biases, width, label='Final (Output)', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=9, fontweight='bold')
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_ylabel('Bias (WEAT Effect Size)', fontsize=11, fontweight='bold')
    ax1.set_title('Initial vs Final Bias', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Amplification percentage
    colors_amp = ['#2ecc71' if amp < 0 else '#e74c3c' for amp in amplifications]
    bars = ax2.bar(models, [abs(results_dict[m]['summary']['amplification_pct']) for m in models], 
                  color=colors_amp, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, amp_pct in zip(bars, [results_dict[m]['summary']['amplification_pct'] for m in models]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
               f'{amp_pct:+.1f}%', ha='center', va='bottom',
               fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Amplification (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Bias Change Through Model\n(Green=Reduced, Red=Amplified)', 
                 fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved amplification comparison to {output_path}")
    plt.close()


def create_hidden_layer_visualizations():
    """Create all hidden layer visualizations"""
    print("\n" + "="*70)
    print("CREATING HIDDEN LAYER VISUALIZATIONS")
    print("="*70 + "\n")
    
    results_dir = Path("results")
    
    # Load hidden layer results
    results_dict = {}
    for model_name in ['gpt2', 'gpt2-medium', 'gpt2-large']:
        pattern = f"hidden_layer_analysis_{model_name}*.json"
        files = list(results_dir.glob(pattern))
        
        if files:
            latest_file = max(files, key=lambda p: p.stat().st_mtime)
            print(f"Loading {model_name} from: {latest_file.name}")
            results = load_hidden_layer_results(latest_file)
            
            # Check if data is valid (not all NaN)
            if results['summary']['initial_bias'] != 0 or results['summary']['final_bias'] != 0:
                results_dict[model_name] = results
            else:
                print(f"  ⚠ Skipping {model_name} - data appears corrupted (all zeros)")
        else:
            print(f"  ⚠ No hidden layer results found for {model_name}")
    
    if not results_dict:
        print("\n✗ No valid hidden layer results found!")
        return
    
    print(f"\nFound valid results for: {list(results_dict.keys())}")
    
    output_dir = Path("results/visualizations")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating plots...\n")
    
    # Plot 1: Trajectory comparison
    if len(results_dict) > 0:
        plot_bias_trajectory_all_models(results_dict, 
                                       output_dir / "10_hidden_layer_trajectory.png")
    
    # Plot 2: Three-phase architecture for each valid model
    for i, (model_name, results) in enumerate(results_dict.items()):
        plot_three_phase_architecture(results, model_name,
                                     output_dir / f"11_three_phase_{model_name}.png")
    
    # Plot 3: Amplification comparison
    if len(results_dict) > 1:
        plot_amplification_comparison(results_dict,
                                     output_dir / "12_amplification_comparison.png")
    
    print("\n" + "="*70)
    print("✅ HIDDEN LAYER VISUALIZATIONS CREATED")
    print("="*70)
    print(f"\nSaved to: {output_dir}/")
    print("\nGenerated plots:")
    print("  10. 10_hidden_layer_trajectory.png - Bias trajectory across all layers")
    for model_name in results_dict.keys():
        print(f"  11. 11_three_phase_{model_name}.png - Three-phase architecture")
    if len(results_dict) > 1:
        print("  12. 12_amplification_comparison.png - Amplification comparison")


if __name__ == "__main__":
    create_hidden_layer_visualizations()