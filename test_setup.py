"""
Test script that automatically saves results
"""
import sys
sys.path.append('src')

from embeddings import EmbeddingExtractor
from bias_metrics import analyze_model_bias
from save_results import save_analysis_results, save_embeddings, save_summary_report
import config
from datetime import datetime


def run_full_analysis_and_save():
    """Run analysis on all 30 occupations and save results"""
    
    print("=" * 70)
    print("RUNNING FULL BIAS ANALYSIS WITH AUTO-SAVE")
    print("=" * 70)
    
    model_name = config.MODELS[0]  # Get first model from config
    all_words = config.OCCUPATION_TERMS + config.GENDER_TERMS
    
    print(f"\nAnalyzing {len(config.OCCUPATION_TERMS)} occupations with model: {model_name}")
    print(f"Gender terms: {config.GENDER_TERMS}\n")
    
    # Extract embeddings
    print("Loading model and extracting embeddings...")
    extractor = EmbeddingExtractor(model_name)
    embeddings = extractor.extract_all_embeddings(all_words)
    
    # Save embeddings
    save_embeddings(embeddings, model_name)
    
    # Run bias analysis
    print("\nComputing bias scores...")
    results = analyze_model_bias(
        embeddings,
        config.OCCUPATION_TERMS,
        config.MALE_TERMS,
        config.FEMALE_TERMS
    )
    
    # Add metadata
    results['model_name'] = model_name
    results['timestamp'] = datetime.now().isoformat()
    results['num_occupations'] = len(config.OCCUPATION_TERMS)
    results['occupation_terms'] = config.OCCUPATION_TERMS
    results['male_terms'] = config.MALE_TERMS
    results['female_terms'] = config.FEMALE_TERMS
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nWEAT Effect Size: {results['weat_effect_size']:.4f}")
    print(f"WEAT p-value: {results['weat_p_value']:.4f}")
    print(f"Mean Bias: {results['mean_bias']:.4f}")
    print(f"Std Bias: {results['std_bias']:.4f}")
    
    print("\nTop 10 Most Biased Occupations:")
    print("-" * 70)
    for occupation, data in results['most_biased'][:10]:
        direction_symbol = "→ ♂" if data['direction'] == 'male' else "→ ♀"
        print(f"{occupation:15s} {direction_symbol}  score: {data['bias_score']:+.4f}")
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70 + "\n")
    
    saved_files = save_analysis_results(results, model_name)
    
    # Create summary report
    save_summary_report({model_name: results})
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete and results saved!")
    print("=" * 70)
    
    print(f"\nFiles saved:")
    for file_type, path in saved_files.items():
        if path:
            print(f"  {file_type}: {path}")
    
    return results, saved_files


if __name__ == "__main__":
    print("\n🚀 Starting Full Gender Bias Analysis with Auto-Save\n")
    
    try:
        results, files = run_full_analysis_and_save()
        
        print("\n✅ Success! Next steps:")
        print("1. Check the results/ directory for saved files")
        print("2. Review results/summary_report.txt for a readable summary")
        print("3. Open the CSV file in Excel/Numbers for easy viewing")
        print("4. Run visualizations or compare with other models")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)