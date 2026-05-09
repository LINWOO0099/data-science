"""
Text preprocessing module for sentiment analysis.
Handles text cleaning, tokenization, stopword removal, and stemming.
"""

import nltk
import re
from typing import List, Tuple
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def clean_text(text: str) -> str:
    """
    Clean raw text by removing HTML tags, special characters, and extra whitespace.
    Apply stemming and stopword removal.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned and processed text
    """
    # Remove HTML tags and non-alphabetical chars
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords and stem
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    return text.split()


def preprocess_batch(texts: List[str]) -> List[str]:
    """
    Preprocess a batch of texts.
    
    Args:
        texts: List of texts to preprocess
        
    Returns:
        List of preprocessed texts
    """
    return [clean_text(text) for text in texts]


def load_imdb_data() -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
    """
    Load and preprocess IMDB dataset.
    
    Returns:
        Tuple of ((train_texts, train_labels), (test_texts, test_labels))
    """
    print("Loading IMDB dataset...")
    from torchtext.datasets import IMDB
    
    train_iter, test_iter = IMDB(split=('train', 'test'))
    
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    
    print("Processing training data...")
    for label, text in train_iter:
        train_texts.append(clean_text(text))
        train_labels.append(1 if label == 'pos' else 0)
    
    print("Processing test data...")
    for label, text in test_iter:
        test_texts.append(clean_text(text))
        test_labels.append(1 if label == 'pos' else 0)
    
    print(f"Data loaded: {len(train_texts)} training, {len(test_texts)} test samples")
    return (train_texts, train_labels), (test_texts, test_labels)
