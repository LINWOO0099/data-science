"""
LSTM model training for sentiment analysis using PyTorch.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List, Tuple
import numpy as np


class LSTMSentiment(nn.Module):
    """Bidirectional LSTM model for sentiment analysis."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 2, num_classes: int = 2, dropout: float = 0.3):
        """
        Initialize LSTM model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """Forward pass."""
        x = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(x)
        # Concatenate final hidden states from both directions
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.dropout(hidden)
        return self.fc(out)


def train_lstm(train_texts: List[str], train_labels: List[int],
              test_texts: List[str], test_labels: List[int],
              vocab_size: int = 5000, embedding_dim: int = 100,
              hidden_dim: int = 128, num_epochs: int = 5,
              batch_size: int = 64, max_len: int = 256) -> LSTMSentiment:
    """
    Train LSTM model on sentiment classification task.
    
    Args:
        train_texts: Training texts
        train_labels: Training labels
        test_texts: Test texts
        test_labels: Test labels
        vocab_size: Vocabulary size
        embedding_dim: Embedding dimension
        hidden_dim: LSTM hidden dimension
        num_epochs: Number of training epochs
        batch_size: Batch size
        max_len: Maximum sequence length
        
    Returns:
        Trained LSTM model
    """
    print("\n" + "="*50)
    print("LSTM Model Training")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build vocabulary
    print("Building vocabulary...")
    vectorizer = CountVectorizer(max_features=vocab_size, token_pattern=r'\b\w+\b')
    vectorizer.fit(train_texts)
    vocab = vectorizer.vocabulary_
    
    # Convert texts to sequences
    def texts_to_seq(texts: List[str]) -> List[List[int]]:
        """Convert texts to sequences of vocabulary indices."""
        return [[vocab.get(word, 0) for word in text.split()] for text in texts]
    
    print("Converting texts to sequences...")
    train_seqs = texts_to_seq(train_texts)
    test_seqs = texts_to_seq(test_texts)
    
    # Pad sequences
    def pad_sequences(seqs: List[List[int]], max_len: int) -> List[List[int]]:
        """Pad sequences to fixed length."""
        return [seq[:max_len] + [0]*(max_len - len(seq)) if len(seq) < max_len 
                else seq[:max_len] for seq in seqs]
    
    padded_train = pad_sequences(train_seqs, max_len)
    padded_test = pad_sequences(test_seqs, max_len)
    
    # Convert to tensors
    X_train_tensor = torch.LongTensor(padded_train).to(device)
    y_train_tensor = torch.LongTensor(train_labels).to(device)
    X_test_tensor = torch.LongTensor(padded_test).to(device)
    y_test_tensor = torch.LongTensor(test_labels).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = LSTMSentiment(vocab_size + 1, embedding_dim, hidden_dim, 
                         num_layers=2, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    best_acc = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}")
    
    # Final evaluation
    print("\n" + "="*50)
    print("LSTM Results")
    print("="*50)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
    
    all_preds = np.array(all_preds)
    test_labels_np = np.array(test_labels)
    accuracy = accuracy_score(test_labels_np, all_preds)
    
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(test_labels_np, all_preds,
                               target_names=['Negative', 'Positive']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(test_labels_np, all_preds))
    print("="*50 + "\n")
    
    return model


if __name__ == "__main__":
    print("LSTM training module")
