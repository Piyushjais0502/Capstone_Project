"""
Evaluation Module for Fake News Detection
Comprehensive metrics and analysis for model performance
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import os

from model import FakeNewsDetector


class ModelEvaluator:
    """Comprehensive evaluation for fake news detection model."""
    
    def __init__(self, model: FakeNewsDetector):
        """
        Initialize evaluator.
        
        Args:
            model: Trained FakeNewsDetector instance
        """
        self.model = model
        self.class_names = ['Real', 'Fake']
    
    def compute_metrics(self, y_true: List[int], y_pred: List[int], 
                       y_proba: List[List[float]] = None) -> Dict:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro')
        }
        
        # Add AUC if probabilities provided
        if y_proba is not None:
            y_proba_positive = [p[1] for p in y_proba]
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba_positive)
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in formatted table."""
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        print(f"{'Metric':<25} {'Value':>10}")
        print("-" * 60)
        
        for metric_name, value in metrics.items():
            formatted_name = metric_name.replace('_', ' ').title()
            print(f"{formatted_name:<25} {value:>10.4f}")
        
        print("=" * 60)
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                             save_path: str = "results/confusion_matrix.png"):
        """
        Plot and save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_roc_curve(self, y_true: List[int], y_proba: List[List[float]], 
                      save_path: str = "results/roc_curve.png"):
        """
        Plot and save ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            save_path: Path to save plot
        """
        y_proba_positive = [p[1] for p in y_proba]
        fpr, tpr, thresholds = roc_curve(y_true, y_proba_positive)
        auc = roc_auc_score(y_true, y_proba_positive)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
        plt.close()
    
    def plot_class_distribution(self, y_true: List[int], y_pred: List[int],
                               save_path: str = "results/class_distribution.png"):
        """
        Plot class distribution comparison.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # True distribution
        true_counts = [y_true.count(0), y_true.count(1)]
        axes[0].bar(self.class_names, true_counts, color=['#2ecc71', '#e74c3c'])
        axes[0].set_title('True Label Distribution', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=10)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Predicted distribution
        pred_counts = [y_pred.count(0), y_pred.count(1)]
        axes[1].bar(self.class_names, pred_counts, color=['#2ecc71', '#e74c3c'])
        axes[1].set_title('Predicted Label Distribution', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
        plt.close()
    
    def evaluate(self, texts: List[str], labels: List[int], 
                save_results: bool = True) -> Dict:
        """
        Complete evaluation pipeline.
        
        Args:
            texts: Test texts
            labels: True labels
            save_results: Whether to save plots
            
        Returns:
            Dictionary of metrics
        """
        print("\n" + "=" * 60)
        print("STARTING EVALUATION")
        print("=" * 60)
        print(f"Test samples: {len(texts)}")
        
        # Get predictions
        print("\nGenerating predictions...")
        predictions = self.model.predict(texts)
        probabilities = self.model.predict_proba(texts)
        
        # Compute metrics
        metrics = self.compute_metrics(labels, predictions, probabilities)
        self.print_metrics(metrics)
        
        # Generate classification report
        print("\nDETAILED CLASSIFICATION REPORT")
        print("-" * 60)
        print(classification_report(labels, predictions, 
                                   target_names=self.class_names))
        
        # Save visualizations
        if save_results:
            print("\nGenerating visualizations...")
            self.plot_confusion_matrix(labels, predictions)
            self.plot_roc_curve(labels, probabilities)
            self.plot_class_distribution(labels, predictions)
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED")
        print("=" * 60)
        
        return metrics


if __name__ == "__main__":
    from preprocessing import TextPreprocessor, create_sample_dataset
    
    # Create sample data
    print("Creating sample dataset...")
    df = create_sample_dataset()
    
    # Preprocess
    preprocessor = TextPreprocessor()
    texts, labels = preprocessor.prepare_dataset(df, 'text', 'label')
    
    # Initialize model (pretrained, not fine-tuned)
    print("\nInitializing model...")
    detector = FakeNewsDetector(max_length=128)
    
    # Evaluate
    print("\nStarting evaluation...")
    evaluator = ModelEvaluator(detector)
    metrics = evaluator.evaluate(texts, labels, save_results=True)
    
    print("\nEvaluation demonstration completed!")
