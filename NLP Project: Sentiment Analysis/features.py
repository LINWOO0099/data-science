"""
Feature extraction module for sentiment analysis.
Includes TF-IDF and Word2Vec feature engineering methods.
"""

from typing import List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np


def get_tfidf_features(train_texts: List[str], test_texts: List[str], 
                       max_features: int = 5000) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Extract TF-IDF features from texts.
    
    Args:
        train_texts: Training texts
        test_texts: Test texts
        max_features: Maximum number of features
        
    Returns:
        Tuple of (X_train, X_test, vectorizer)
    """
    print(f"Extracting TF-IDF features (max_features={max_features})...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()
    print(f"TF-IDF features shape: train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, vectorizer


def get_word2vec_embeddings(train_texts: List[str], test_texts: List[str], 
                            embedding_dim: int = 100) -> Tuple[np.ndarray, np.ndarray, Word2Vec]:
    """
    Extract Word2Vec embeddings from texts.
    Average word vectors for each document.
    
    Args:
        train_texts: Training texts
        test_texts: Test texts
        embedding_dim: Dimension of embeddings
        
    Returns:
        Tuple of (X_train, X_test, w2v_model)
    """
    print(f"Extracting Word2Vec embeddings (dim={embedding_dim})...")
    
    # Tokenize for Word2Vec
    train_tokens = [text.split() for text in train_texts]
    test_tokens = [text.split() for text in test_texts]
    
    # Train Word2Vec model
    w2v = Word2Vec(sentences=train_tokens, vector_size=embedding_dim, 
                   window=5, min_count=1, workers=4)
    
    # Function to get average embedding for a document
    def text_to_vec(tokens: List[str]) -> np.ndarray:
        """Convert tokenized text to average embedding vector."""
        vectors = [w2v.wv[word] for word in tokens if word in w2v.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(embedding_dim)
    
    X_train = np.array([text_to_vec(tokens) for tokens in train_tokens])
    X_test = np.array([text_to_vec(tokens) for tokens in test_tokens])
    
    print(f"Word2Vec embeddings shape: train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, w2v
