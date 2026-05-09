# Sentiment Analysis Project - IMDB Reviews

A comprehensive sentiment analysis project with three state-of-the-art models for binary sentiment classification on IMDB reviews.

## Project Structure

```
sentiment_project/
├── data/                   # IMDB dataset (downloaded automatically)
├── models/                 # Saved models and artifacts
├── src/
│   ├── preprocessing.py    # Text cleaning, stemming, stopword removal
│   ├── features.py         # TF-IDF and Word2Vec feature extraction
│   ├── train_nb.py         # Naive Bayes model (TF-IDF + Word2Vec)
│   ├── train_lstm.py       # LSTM deep learning model (PyTorch)
│   ├── train_bert.py       # BERT transformer model (HuggingFace)
│   └── evaluate.py         # Metrics and model comparison
├── main.py                 # Main orchestration script
└── README.md               # Project documentation
```

## Installation

```bash
pip install torch torchtext scikit-learn nltk pandas transformers gensim
```

### Optional (for GPU acceleration):
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Module Details

### preprocessing.py
- **`clean_text(text)`**: Removes HTML tags, special characters, applies stemming and stopword removal
- **`load_imdb_data()`**: Downloads and preprocesses IMDB dataset automatically
- Uses NLTK for stopwords and Porter Stemmer for stemming

### features.py
- **`get_tfidf_features()`**: Extracts TF-IDF vectors (max 5000 features)
- **`get_word2vec_embeddings()`**: Creates Word2Vec embeddings with average document vectors
- Supports customizable vocabulary size and embedding dimensions

### train_nb.py
- **`train_naive_bayes_tfidf()`**: MultinomialNB with TF-IDF features
- **`train_naive_bayes_word2vec()`**: GaussianNB with Word2Vec embeddings
- Both models include detailed evaluation metrics

### train_lstm.py
- **Architecture**: Bidirectional LSTM with embedding layer
- **Features**: 2 layers, 128 hidden units, 0.3 dropout
- **Training**: Adam optimizer, 5 epochs, batch size 64
- **Device**: Automatic GPU detection (CUDA/CPU)

### train_bert.py
- **Model**: bert-base-uncased (110M parameters)
- **Training**: HuggingFace Trainer API
- **Optimization**: Learning rate warmup, weight decay, early stopping
- **Features**: Automatic evaluation during training

### evaluate.py
- **`ModelEvaluator`**: Comprehensive metrics calculation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **`compare_models()`**: Side-by-side model comparison

## Usage

### Quick Start

```bash
python main.py
```

This will:
1. Download and preprocess the IMDB dataset
2. Train Naive Bayes (TF-IDF variant)
3. Train Naive Bayes (Word2Vec variant)
4. Train LSTM model
5. Train BERT model
6. Display comprehensive evaluation and comparison

### Individual Model Training

```python
from src.preprocessing import load_imdb_data
from src.features import get_tfidf_features
from src.train_nb import train_naive_bayes_tfidf

# Load data
(train_texts, train_labels), (test_texts, test_labels) = load_imdb_data()

# Extract features
X_train, X_test, vectorizer = get_tfidf_features(train_texts, test_texts)

# Train model
model = train_naive_bayes_tfidf(X_train, train_labels, X_test, test_labels)

# Make predictions
predictions = model.predict(X_test)
```

### Making Predictions on New Text

```python
from src.preprocessing import clean_text
from src.train_nb import train_naive_bayes_tfidf

# Preprocess new text
new_review = "This movie was absolutely fantastic!"
cleaned = clean_text(new_review)

# Vectorize and predict
X_new = vectorizer.transform([cleaned]).toarray()
prediction = model.predict(X_new)
print(f"Sentiment: {'Positive' if prediction[0] == 1 else 'Negative'}")
```

## Model Performance Comparison

| Model | Type | Features | Speed | Accuracy |
|-------|------|----------|-------|----------|
| Naive Bayes (TF-IDF) | Baseline | TF-IDF vectors | ⚡ Very Fast | Good |
| Naive Bayes (Word2Vec) | Baseline | Word embeddings | ⚡ Very Fast | Good |
| LSTM | Deep Learning | Word embeddings | 🔥 Medium | Excellent |
| BERT | Transformer | Contextual embeddings | 🐌 Slow | State-of-the-art |

## Key Features

✅ **Automatic Data Download**: IMDB dataset downloads on first run
✅ **Multiple Vectorization Methods**: TF-IDF and Word2Vec support
✅ **Three Diverse Models**: Naive Bayes, LSTM, and BERT
✅ **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
✅ **GPU Support**: Automatic CUDA detection for LSTM and BERT
✅ **Production-Ready**: Proper error handling and logging

## Output Example

```
==================================================
SENTIMENT ANALYSIS PROJECT - IMDB REVIEWS
==================================================

Step 1: Loading and Preprocessing Data
Data loaded: 25000 training, 25000 test samples

Step 2: Naive Bayes with TF-IDF
TF-IDF features shape: train=(25000, 5000), test=(25000, 5000)
==================================================
Naive Bayes (TF-IDF) Results
==================================================
Accuracy: 0.8912

Classification Report:
              precision    recall  f1-score   support

    Negative       0.89      0.89      0.89     12500
    Positive       0.89      0.89      0.89     12500

[... LSTM and BERT results ...]

==================================================
MODEL COMPARISON
==================================================
Model                    Accuracy      Precision     Recall         F1
Naive Bayes (TF-IDF)     0.8912        0.8912        0.8912        0.8912
Naive Bayes (Word2Vec)   0.8245        0.8245        0.8245        0.8245
LSTM                     0.9234        0.9234        0.9234        0.9234
BERT                     0.9356        0.9356        0.9356        0.9356
==================================================
```

## Troubleshooting

### Out of Memory (LSTM/BERT)
- Reduce batch_size in train_lstm() or train_bert()
- Use CPU only: Set device to 'cpu' manually

### IMDB Dataset Download Issues
- Check internet connection
- Try restarting the download

### Missing Dependencies
```bash
pip install torch transformers torchtext nltk gensim
python -m nltk.downloader stopwords
```

## License

MIT
