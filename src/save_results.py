"""
Utilities for saving analysis results
"""
import json
import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd


def save_analysis_results(results, model_name, output_dir="results"):
    """
    Save analysis results in multiple formats
    
    Args:
        results: Dictionary with analysis results
        model_name: Name of the model analyzed
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{model_name.replace('/', '_')}_{timestamp}"
    
    # Save as JSON (human-readable)
    json_path = output_dir / f"{base_filename}.json"
    json_safe_results = convert_to_json_safe(results)
    with open(json_path, 'w') as f:
        json.dump(json_safe_results, f, indent=2)
    print(f"✓ Saved JSON results to {json_path}")
    
    # Save as pickle (preserves numpy arrays)
    pickle_path = output_dir / f"{base_filename}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Saved pickle results to {pickle_path}")
    
    # Save bias scores as CSV
    if 'bias_scores' in results:
        csv_path = output_dir / f"{base_filename}_bias_scores.csv"
        df = create_bias_dataframe(results['bias_scores'])
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved bias scores CSV to {csv_path}")
    
    return {
        'json': json_path,
        'pickle': pickle_path,
        'csv': csv_path if 'bias_scores' in results else None
    }


def convert_to_json_safe(obj):
    """Convert numpy arrays and other non-JSON types to JSON-safe formats"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: convert_to_json_safe(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_safe(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, tuple):
        return [convert_to_json_safe(item) for item in obj]
    else:
        return obj


def create_bias_dataframe(bias_scores):
    """Convert bias scores dictionary to pandas DataFrame"""
    data = []
    for occupation, scores in bias_scores.items():
        data.append({
            'occupation': occupation,
            'bias_score': scores['bias_score'],
            'direction': scores['direction'],
            'mean_male_similarity': scores['mean_male_similarity'],
            'mean_female_similarity': scores['mean_female_similarity']
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('bias_score', key=abs, ascending=False)
    return df


def save_embeddings(embeddings, model_name, output_dir="data/processed"):
    """
    Save embeddings for later use
    
    Args:
        embeddings: Dictionary of word embeddings
        model_name: Name of the model
        output_dir: Directory to save embeddings
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"embeddings_{model_name.replace('/', '_')}_{timestamp}.pkl"
    filepath = output_dir / filename
    
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"✓ Saved embeddings to {filepath}")
    return filepath


def load_results(filepath):
    """Load saved results from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_summary_report(all_model_results, output_path="results/summary_report.txt"):
    """
    Create a text summary report comparing multiple models
    
    Args:
        all_model_results: Dictionary mapping model names to their results
        output_path: Path to save the report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GENDER BIAS ANALYSIS SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        for model_name, results in all_model_results.items():
            f.write(f"\nMODEL: {model_name}\n")
            f.write("-" * 70 + "\n")
            f.write(f"WEAT Effect Size: {results['weat_effect_size']:.4f}\n")
            f.write(f"WEAT p-value: {results['weat_p_value']:.4f}\n")
            f.write(f"Mean Bias: {results['mean_bias']:.4f}\n")
            f.write(f"Std Bias: {results['std_bias']:.4f}\n\n")
            
            f.write("Top 10 Most Biased Occupations:\n")
            for occupation, data in results['most_biased'][:10]:
                direction = "→ ♂" if data['direction'] == 'male' else "→ ♀"
                f.write(f"  {occupation:15s} {direction}  {data['bias_score']:+.4f}\n")
            f.write("\n")
    
    print(f"✓ Saved summary report to {output_path}")
    return output_path