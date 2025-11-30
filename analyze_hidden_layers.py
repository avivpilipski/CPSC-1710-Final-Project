"""
Analyze gender bias across hidden layers of the model
FIXED: Uses same mean-pooling logic as original embedding extraction
"""
import sys
sys.path.append('src')

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from bias_metrics import cosine_similarity, compute_weat_score
import config
from datetime import datetime
from pathlib import Path
import json


class HiddenLayerAnalyzer:
    """Analyze bias at each hidden layer of a language model"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()
        
        # Get number of layers
        if hasattr(self.model, 'gpt2'):
            self.n_layers = self.model.gpt2.config.num_hidden_layers
        elif hasattr(self.model, 'transformer'):
            self.n_layers = self.model.transformer.config.num_hidden_layers
        else:
            self.n_layers = self.model.config.num_hidden_layers
            
        print(f"Model: {model_name}")
        print(f"Number of hidden layers: {self.n_layers}")
    
    def extract_hidden_states(self, word):
        """
        Extract hidden states from all layers for a word
        FIXED: Uses mean pooling for multi-token words (same as original analysis)
        """
        tokens = self.tokenizer(word, return_tensors="pt", add_special_tokens=False)
        
        with torch.no_grad():
            outputs = self.model(**tokens, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        
        # Extract representation from each layer
        layer_embeddings = []
        for layer_output in hidden_states:
            # layer_output shape: (batch_size, seq_len, hidden_size)
            # Get first batch
            token_reprs = layer_output[0, :, :].cpu().numpy()  # (seq_len, hidden_size)
            
            # Apply mean pooling for multi-token words (SAME AS ORIGINAL)
            if token_reprs.shape[0] > 1:
                embedding = np.mean(token_reprs, axis=0)
            else:
                embedding = token_reprs[0, :]
            
            layer_embeddings.append(embedding)
        
        return layer_embeddings
    
    def analyze_all_layers(self, words):
        """Extract representations from all layers for multiple words"""
        print(f"\nExtracting hidden states from {len(words)} words across {self.n_layers + 1} layers...")
        print("(Using mean pooling for multi-token words - SAME AS ORIGINAL ANALYSIS)")
        
        # all_layer_reps[layer_idx][word] = embedding vector
        all_layer_reps = [{} for _ in range(self.n_layers + 1)]
        
        for word in words:
            try:
                layer_embeddings = self.extract_hidden_states(word)
                for layer_idx, embedding in enumerate(layer_embeddings):
                    all_layer_reps[layer_idx][word] = embedding
                print(f"✓ {word}")
            except Exception as e:
                print(f"✗ Error with {word}: {e}")
        
        return all_layer_reps
    
    def compute_layer_bias(self, all_layer_reps, occupation_terms, male_terms, female_terms):
        """Compute WEAT score at each layer"""
        layer_results = {}
        
        for layer_idx, layer_reps in enumerate(all_layer_reps):
            # Get embeddings for this layer
            target_embeddings = [layer_reps[occ] for occ in occupation_terms if occ in layer_reps]
            male_embeddings = [layer_reps[term] for term in male_terms if term in layer_reps]
            female_embeddings = [layer_reps[term] for term in female_terms if term in layer_reps]
            
            if target_embeddings and male_embeddings and female_embeddings:
                effect_size, p_value = compute_weat_score(target_embeddings, male_embeddings, female_embeddings)
                
                # Also compute mean bias for each occupation
                bias_scores = {}
                for occ in occupation_terms:
                    if occ in layer_reps:
                        occ_emb = layer_reps[occ]
                        male_sims = [cosine_similarity(occ_emb, male_emb) for male_emb in male_embeddings]
                        female_sims = [cosine_similarity(occ_emb, female_emb) for female_emb in female_embeddings]
                        bias_score = np.mean(male_sims) - np.mean(female_sims)
                        bias_scores[occ] = bias_score
                
                mean_bias = np.mean(list(bias_scores.values()))
                
                layer_results[layer_idx] = {
                    'weat_effect_size': effect_size,
                    'weat_p_value': p_value,
                    'mean_bias': mean_bias,
                    'individual_scores': bias_scores,
                    'layer_name': self._get_layer_name(layer_idx)
                }
                
                print(f"  Layer {layer_idx:2d} ({self._get_layer_name(layer_idx):15s}): "
                      f"WEAT d={effect_size:+.4f}, mean_bias={mean_bias:+.6f}")
        
        return layer_results
    
    def _get_layer_name(self, layer_idx):
        """Get human-readable layer name"""
        if layer_idx == 0:
            return "Embedding"
        elif layer_idx <= self.n_layers:
            return f"Transformer-{layer_idx}"
        else:
            return "Output"


def run_hidden_layer_analysis(model_name):
    """Run complete hidden layer analysis"""
    print("\n" + "="*70)
    print(f"HIDDEN LAYER BIAS ANALYSIS - {model_name.upper()}")
    print("="*70)
    
    all_words = config.OCCUPATION_TERMS + config.GENDER_TERMS
    
    # Initialize analyzer
    analyzer = HiddenLayerAnalyzer(model_name)
    
    # Extract hidden states
    print(f"\nAnalyzing {len(config.OCCUPATION_TERMS)} occupations + {len(config.GENDER_TERMS)} gender terms")
    all_layer_reps = analyzer.analyze_all_layers(all_words)
    
    # Compute bias at each layer
    print("\n" + "-"*70)
    print("BIAS AT EACH LAYER")
    print("-"*70 + "\n")
    
    layer_results = analyzer.compute_layer_bias(
        all_layer_reps,
        config.OCCUPATION_TERMS,
        config.MALE_TERMS,
        config.FEMALE_TERMS
    )
    
    # Analyze trajectory
    print("\n" + "-"*70)
    print("BIAS TRAJECTORY ANALYSIS")
    print("-"*70 + "\n")
    
    layer_indices = sorted(layer_results.keys())
    effect_sizes = [layer_results[l]['weat_effect_size'] for l in layer_indices]
    mean_biases = [layer_results[l]['mean_bias'] for l in layer_indices]
    
    embedding_bias = effect_sizes[0]
    final_bias = effect_sizes[-1]
    
    amplification = final_bias - embedding_bias
    amplification_pct = (amplification / abs(embedding_bias)) * 100 if embedding_bias != 0 else 0
    
    print(f"Initial (Embedding layer): {embedding_bias:+.6f}")
    print(f"Final (Last transformer layer): {final_bias:+.6f}")
    print(f"Amplification: {amplification:+.6f} ({amplification_pct:+.1f}%)")
    
    if amplification > 0:
        print("\n✓ Bias AMPLIFIED through the model")
    elif amplification < 0:
        print("\n✓ Bias REDUCED through the model")
    else:
        print("\n✓ Bias UNCHANGED through the model")
    
    # Find which layers amplify most
    layer_changes = []
    for i in range(1, len(layer_indices)):
        prev_layer = layer_indices[i-1]
        curr_layer = layer_indices[i]
        change = effect_sizes[i] - effect_sizes[i-1]
        layer_changes.append({
            'from': prev_layer,
            'to': curr_layer,
            'change': change,
            'from_name': layer_results[prev_layer]['layer_name'],
            'to_name': layer_results[curr_layer]['layer_name']
        })
    
    print("\nTop 5 layers with MOST amplification:")
    top_changes = sorted(layer_changes, key=lambda x: x['change'], reverse=True)[:5]
    for i, change in enumerate(top_changes, 1):
        print(f"  {i}. {change['from_name']:15s} → {change['to_name']:15s}: {change['change']:+.6f}")
    
    print("\nTop 5 layers with MOST reduction:")
    bottom_changes = sorted(layer_changes, key=lambda x: x['change'])[:5]
    for i, change in enumerate(bottom_changes, 1):
        print(f"  {i}. {change['from_name']:15s} → {change['to_name']:15s}: {change['change']:+.6f}")
    
    # Save results
    print("\n" + "-"*70)
    print("SAVING RESULTS")
    print("-"*70 + "\n")
    
    output = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'n_layers': analyzer.n_layers,
        'extraction_method': 'mean_pooling (FIXED - matches original analysis)',
        'layer_results': {
            str(k): {
                'layer_name': v['layer_name'],
                'weat_effect_size': float(v['weat_effect_size']),
                'weat_p_value': float(v['weat_p_value']),
                'mean_bias': float(v['mean_bias']),
                'individual_scores': {occ: float(score) for occ, score in v['individual_scores'].items()}
            }
            for k, v in layer_results.items()
        },
        'summary': {
            'initial_bias': float(embedding_bias),
            'final_bias': float(final_bias),
            'amplification': float(amplification),
            'amplification_pct': float(amplification_pct),
            'trajectory': 'amplified' if amplification > 0 else 'reduced' if amplification < 0 else 'unchanged'
        }
    }
    
    output_path = Path("results") / f"hidden_layer_analysis_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Saved hidden layer analysis to {output_path}")
    
    print("\n" + "="*70)
    print("✅ HIDDEN LAYER ANALYSIS COMPLETE")
    print("="*70)
    
    return layer_results, output


if __name__ == "__main__":
    print("\n🔬 Starting Fixed Hidden Layer Bias Analysis\n")
    
    models_to_analyze = ["gpt2", "gpt2-medium", "gpt2-large"]
    
    for model_name in models_to_analyze:
        try:
            layer_results, output = run_hidden_layer_analysis(model_name)
            
            print("\n📊 Key Takeaway:")
            print(f"Bias {output['summary']['trajectory']} by {abs(output['summary']['amplification_pct']):.1f}% across all layers")
            print(f"Extraction method: {output['extraction_method']}\n")
            
        except Exception as e:
            print(f"\n❌ Analysis failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()