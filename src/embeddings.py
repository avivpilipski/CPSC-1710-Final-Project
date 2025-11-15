"""
Embedding extraction utilities for gender bias analysis
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class EmbeddingExtractor:
    """Extract and manage embeddings from language models"""
    
    def __init__(self, model_name: str):
        """
        Initialize the embedding extractor
        
        Args:
            model_name: HuggingFace model identifier (e.g., 'gpt2')
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        self.embeddings = {}
        
    def get_token_ids(self, word: str) -> List[int]:
        """Get token IDs for a word"""
        return self.tokenizer.encode(word, add_special_tokens=False)
    
    def extract_embedding(self, word: str) -> np.ndarray:
        """
        Extract embedding for a single word
        Uses mean pooling for multi-token words
        """
        tokens = self.tokenizer(word, return_tensors="pt", add_special_tokens=False)
        
        with torch.no_grad():
            outputs = self.model(**tokens, output_hidden_states=True)
            
        embedding_layer = outputs.hidden_states[0]
        
        if embedding_layer.shape[1] > 1:
            embedding = embedding_layer.mean(dim=1).squeeze().numpy()
        else:
            embedding = embedding_layer.squeeze().numpy()
            
        return embedding
    
    def extract_all_embeddings(self, words: List[str]) -> Dict[str, np.ndarray]:
        """Extract embeddings for multiple words"""
        embeddings = {}
        
        for word in words:
            try:
                embeddings[word] = self.extract_embedding(word)
                print(f"✓ Extracted embedding for '{word}' (shape: {embeddings[word].shape})")
            except Exception as e:
                print(f"✗ Error extracting embedding for '{word}': {e}")
                
        self.embeddings = embeddings
        return embeddings
    
    def get_embedding_info(self, word: str) -> Dict:
        """Get detailed information about a word's tokenization and embedding"""
        token_ids = self.get_token_ids(word)
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
        embedding = self.embeddings.get(word)
        
        return {
            "word": word,
            "token_ids": token_ids,
            "tokens": tokens,
            "num_tokens": len(token_ids),
            "embedding_shape": embedding.shape if embedding is not None else None
        }


def compare_tokenization(word: str, model_names: List[str]) -> None:
    """Compare how different models tokenize the same word"""
    print(f"\nTokenization comparison for '{word}':")
    print("-" * 50)
    
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        tokens = [tokenizer.decode([tid]) for tid in token_ids]
        
        print(f"{model_name}:")
        print(f"  Tokens: {tokens}")
        print(f"  Count: {len(tokens)}")
        print()