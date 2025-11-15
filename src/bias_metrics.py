"""
Bias measurement utilities using WEAT and cosine similarity
"""
import numpy as np
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple
from scipy import stats


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    return 1 - cosine(vec1, vec2)


def compute_bias_score(
    word_embedding: np.ndarray,
    male_embeddings: List[np.ndarray],
    female_embeddings: List[np.ndarray]
) -> float:
    """
    Compute bias score for a word
    
    Bias_score = mean(cos_sim(w, male_terms)) - mean(cos_sim(w, female_terms))
    Positive = male-leaning, Negative = female-leaning
    """
    male_sims = [cosine_similarity(word_embedding, male_emb) 
                 for male_emb in male_embeddings]
    
    female_sims = [cosine_similarity(word_embedding, female_emb) 
                   for female_emb in female_embeddings]
    
    bias_score = np.mean(male_sims) - np.mean(female_sims)
    
    return bias_score


def compute_all_bias_scores(
    embeddings: Dict[str, np.ndarray],
    occupation_terms: List[str],
    male_terms: List[str],
    female_terms: List[str]
) -> Dict[str, Dict]:
    """Compute bias scores for all occupation terms"""
    male_embeddings = [embeddings[term] for term in male_terms if term in embeddings]
    female_embeddings = [embeddings[term] for term in female_terms if term in embeddings]
    
    results = {}
    
    for occupation in occupation_terms:
        if occupation not in embeddings:
            continue
            
        word_emb = embeddings[occupation]
        
        male_sims = [cosine_similarity(word_emb, male_emb) 
                     for male_emb in male_embeddings]
        female_sims = [cosine_similarity(word_emb, female_emb) 
                       for female_emb in female_embeddings]
        
        bias_score = np.mean(male_sims) - np.mean(female_sims)
        
        results[occupation] = {
            "bias_score": bias_score,
            "mean_male_similarity": np.mean(male_sims),
            "mean_female_similarity": np.mean(female_sims),
            "male_similarities": male_sims,
            "female_similarities": female_sims,
            "direction": "male" if bias_score > 0 else "female"
        }
    
    return results


def compute_weat_score(
    target_embeddings: List[np.ndarray],
    attribute_a: List[np.ndarray],
    attribute_b: List[np.ndarray]
) -> Tuple[float, float]:
    """
    Compute Word Embedding Association Test (WEAT) score
    
    Returns:
        Tuple of (effect_size, p_value)
    """
    bias_scores = []
    
    for target_emb in target_embeddings:
        male_sim = np.mean([cosine_similarity(target_emb, male_emb) 
                           for male_emb in attribute_a])
        
        female_sim = np.mean([cosine_similarity(target_emb, female_emb) 
                             for female_emb in attribute_b])
        
        bias_scores.append(male_sim - female_sim)
    
    mean_bias = np.mean(bias_scores)
    std_bias = np.std(bias_scores)
    
    effect_size = mean_bias / std_bias if std_bias > 0 else 0
    
    t_stat, p_value = stats.ttest_1samp(bias_scores, 0)
    
    return effect_size, p_value


def analyze_model_bias(
    embeddings: Dict[str, np.ndarray],
    occupation_terms: List[str],
    male_terms: List[str],
    female_terms: List[str]
) -> Dict:
    """Complete bias analysis for a single model"""
    bias_scores = compute_all_bias_scores(
        embeddings, occupation_terms, male_terms, female_terms
    )
    
    target_embeddings = [embeddings[term] for term in occupation_terms 
                        if term in embeddings]
    male_embeddings = [embeddings[term] for term in male_terms 
                      if term in embeddings]
    female_embeddings = [embeddings[term] for term in female_terms 
                        if term in embeddings]
    
    effect_size, p_value = compute_weat_score(
        target_embeddings, male_embeddings, female_embeddings
    )
    
    sorted_occupations = sorted(
        bias_scores.items(), 
        key=lambda x: abs(x[1]["bias_score"]), 
        reverse=True
    )
    
    return {
        "bias_scores": bias_scores,
        "weat_effect_size": effect_size,
        "weat_p_value": p_value,
        "most_biased": sorted_occupations[:10],
        "mean_bias": np.mean([v["bias_score"] for v in bias_scores.values()]),
        "std_bias": np.std([v["bias_score"] for v in bias_scores.values()])
    }