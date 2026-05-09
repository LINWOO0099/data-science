"""
Main script for sentiment analysis project.
Orchestrates data loading, preprocessing, model training, and evaluation.
Trains three models: Naive Bayes, LSTM, and BERT.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing import load_imdb_data
from features import get_tfidf_features, get_word2vec_embeddings
from train_nb import train_naive_bayes_tfidf, train_naive_bayes_word2vec
from train_lstm import train_lstm
from train_bert import train_bert
from evaluate import ModelEvaluator
import numpy as np


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS PROJECT - IMDB REVIEWS")
    print("="*70)
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # ---------- Load and Preprocess Data ----------
    print("\nStep 1: Loading and Preprocessing Data")
    print("-" * 70)
    (train_texts, train_labels), (test_texts, test_labels) = load_imdb_data()
    
    evaluation_results = []
    
    # ---------- NAIVE BAYES WITH TF-IDF ----------
    print("\n\nStep 2: Naive Bayes with TF-IDF")
    print("-" * 70)
    X_train_tfidf, X_test_tfidf, vectorizer = get_tfidf_features(
        train_texts, test_texts, max_features=5000
    )
    
    # Ensure non-negative features for MultinomialNB
    X_train_tfidf = np.maximum(X_train_tfidf, 0)
    X_test_tfidf = np.maximum(X_test_tfidf, 0)
    
    nb_model_tfidf = train_naive_bayes_tfidf(
        X_train_tfidf, train_labels, X_test_tfidf, test_labels
    )
    
    # Store results
    y_pred_nb = nb_model_tfidf.predict(X_test_tfidf)
    result_nb = ModelEvaluator.evaluate_model(
        np.array(test_labels), y_pred_nb, "Naive Bayes (TF-IDF)"
    )
    evaluation_results.append(result_nb)
    
    # ---------- NAIVE BAYES WITH WORD2VEC ----------
    print("\n\nStep 3: Naive Bayes with Word2Vec")
    print("-" * 70)
    X_train_w2v, X_test_w2v, w2v_model = get_word2vec_embeddings(
        train_texts, test_texts, embedding_dim=100
    )
    
    nb_model_w2v = train_naive_bayes_word2vec(
        X_train_w2v, train_labels, X_test_w2v, test_labels
    )
    
    # Store results
    y_pred_nb_w2v = nb_model_w2v.predict(X_test_w2v)
    result_nb_w2v = ModelEvaluator.evaluate_model(
        np.array(test_labels), y_pred_nb_w2v, "Naive Bayes (Word2Vec)"
    )
    evaluation_results.append(result_nb_w2v)
    
    # ---------- LSTM MODEL ----------
    print("\n\nStep 4: LSTM Model Training")
    print("-" * 70)
    lstm_model = train_lstm(
        train_texts, train_labels, test_texts, test_labels,
        vocab_size=5000, embedding_dim=100, hidden_dim=128,
        num_epochs=3, batch_size=64
    )
    
    # ---------- BERT MODEL ----------
    print("\n\nStep 5: BERT Model Training")
    print("-" * 70)
    try:
        bert_model = train_bert(
            train_texts, train_labels, test_texts, test_labels,
            model_name='bert-base-uncased', num_epochs=2, batch_size=16
        )
    except Exception as e:
        print(f"Note: BERT training encountered an issue: {e}")
        print("This may be due to memory constraints or missing dependencies.")
        print("Ensure you have transformers and torch installed:")
        print("pip install transformers torch")
    
    # ---------- COMPARISON ----------
    print("\n\nStep 6: Model Comparison")
    print("-" * 70)
    ModelEvaluator.compare_models(evaluation_results)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nModels and vectorizers saved in ./models/")
    print("Training logs saved in ./bert_logs/ and ./bert_results/")
    print("\nTo make predictions on new text:")
    print("  1. Preprocess text using preprocessing.clean_text()")
    print("  2. Use nb_model.predict() for Naive Bayes")
    print("  3. Use lstm_model or bert_model for deep learning models")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
