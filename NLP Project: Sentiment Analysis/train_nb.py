"""
Naive Bayes model training for sentiment analysis.
Supports both MultinomialNB (for TF-IDF) and GaussianNB (for Word2Vec).
"""

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Union
import numpy as np


def train_naive_bayes_tfidf(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> MultinomialNB:
    """
    Train Naive Bayes classifier with TF-IDF features.
    
    Args:
        X_train: Training features (TF-IDF)
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Trained model
    """
    print("Training Naive Bayes with TF-IDF...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("Naive Bayes (TF-IDF) Results")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Negative', 'Positive']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("="*50 + "\n")
    
    return model


def train_naive_bayes_word2vec(X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray) -> GaussianNB:
    """
    Train Gaussian Naive Bayes classifier with Word2Vec embeddings.
    Note: GaussianNB is used instead of MultinomialNB for continuous embeddings.
    
    Args:
        X_train: Training features (Word2Vec embeddings)
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Trained model
    """
    print("Training Gaussian Naive Bayes with Word2Vec...")
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("Gaussian Naive Bayes (Word2Vec) Results")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                               target_names=['Negative', 'Positive']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("="*50 + "\n")
    
    return model


if __name__ == "__main__":
    print("Naive Bayes training module")
