"""
Model evaluation module for sentiment analysis.
Includes comprehensive metrics, comparisons, and analysis.
"""

from typing import Dict, List
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)


class ModelEvaluator:
    """Evaluate and compare model performance."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        return metrics
    
    @staticmethod
    def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Get confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def calculate_roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Calculate ROC-AUC score.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            ROC-AUC score
        """
        try:
            return roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            return 0.0
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
        """
        Print metrics in formatted way.
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
        """
        print("\n" + "="*50)
        print(f"{model_name} - Evaluation Metrics")
        print("="*50)
        for metric_name, value in metrics.items():
            print(f"{metric_name:.<20} {value:.4f}")
        print("="*50)
    
    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str = "Model") -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation results
        """
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
        cm = ModelEvaluator.get_confusion_matrix(y_true, y_pred)
        
        ModelEvaluator.print_metrics(metrics, model_name)
        
        results = {
            'model': model_name,
            'metrics': metrics,
            'confusion_matrix': cm
        }
        
        return results
    
    @staticmethod
    def compare_models(results_list: List[Dict]) -> None:
        """
        Compare multiple model results.
        
        Args:
            results_list: List of evaluation result dictionaries
        """
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-"*70)
        
        for result in results_list:
            model_name = result['model']
            metrics = result['metrics']
            print(f"{model_name:<20} {metrics['accuracy']:<12.4f} "
                  f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1']:<12.4f}")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    print("Evaluation module")
