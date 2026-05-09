"""
BERT model training for sentiment analysis.
Uses HuggingFace transformers library and Hugging Face Trainer API.
"""

from typing import List
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


class IMDBDataset(Dataset):
    """Custom Dataset class for IMDB reviews."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 512):
        """
        Initialize dataset.
        
        Args:
            texts: List of texts
            labels: List of labels
            tokenizer: Tokenizer instance
            max_len: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with tokenized text and label
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred) -> dict:
    """
    Compute accuracy metric for evaluation.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dictionary with accuracy metric
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}


def train_bert(train_texts: List[str], train_labels: List[int],
              test_texts: List[str], test_labels: List[int],
              model_name: str = 'bert-base-uncased',
              num_epochs: int = 3, batch_size: int = 16,
              max_len: int = 256) -> BertForSequenceClassification:
    """
    Train BERT model on sentiment classification task.
    
    Args:
        train_texts: Training texts
        train_labels: Training labels
        test_texts: Test texts
        test_labels: Test labels
        model_name: Pretrained model name
        num_epochs: Number of epochs
        batch_size: Batch size
        max_len: Maximum sequence length
        
    Returns:
        Trained BERT model
    """
    print("\n" + "="*50)
    print("BERT Model Training")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading {model_name} tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, max_len)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, max_len)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./bert_results',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./bert_logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )
    
    # Create trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    print(f"\nTraining for {num_epochs} epochs...")
    trainer.train()
    
    # Evaluate
    print("\nEvaluating on test set...")
    eval_results = trainer.evaluate(test_dataset)
    
    # Get predictions for detailed metrics
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    test_labels_np = np.array(test_labels)
    
    print("\n" + "="*50)
    print("BERT Results")
    print("="*50)
    print(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(test_labels_np, pred_labels,
                               target_names=['Negative', 'Positive']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(test_labels_np, pred_labels))
    print("="*50 + "\n")
    
    return model


if __name__ == "__main__":
    print("BERT training module")
