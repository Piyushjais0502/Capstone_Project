"""
Transformer Model Module for Fake News Detection
Uses DistilBERT - lightweight and suitable for CPU training
"""

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from typing import List, Dict


class FakeNewsDetector:
    """
    Wrapper class for DistilBERT-based fake news detection.
    Optimized for resource-constrained environments.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased", 
                 num_labels: int = 2, max_length: int = 128):
        """
        Initialize the model and tokenizer.
        
        Args:
            model_name: Pretrained model identifier
            num_labels: Number of output classes (2 for binary)
            max_length: Maximum sequence length (shorter = faster)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing model on device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # Load model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
    def tokenize_texts(self, texts: List[str]) -> Dict:
        """
        Tokenize input texts for model consumption.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with input_ids, attention_mask
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encodings
    
    def predict(self, texts: List[str], batch_size: int = 8) -> List[int]:
        """
        Predict labels for input texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for inference
            
        Returns:
            List of predicted labels
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encodings = self.tokenize_texts(batch_texts)
                
                # Move to device
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get predictions
                batch_predictions = torch.argmax(logits, dim=1).cpu().tolist()
                predictions.extend(batch_predictions)
        
        return predictions
    
    def predict_proba(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """
        Predict probability distributions for input texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for inference
            
        Returns:
            List of probability distributions
        """
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encodings = self.tokenize_texts(batch_texts)
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=1).cpu().tolist()
                probabilities.extend(probs)
        
        return probabilities
    
    def save_model(self, save_path: str):
        """Save model and tokenizer to disk."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load model and tokenizer from disk."""
        self.model = DistilBertForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(load_path)
        self.model.to(self.device)
        print(f"Model loaded from {load_path}")


def get_model_info():
    """Display model information and specifications."""
    info = {
        "Model": "DistilBERT",
        "Parameters": "66 million",
        "Size": "~250 MB",
        "Speed": "40% faster than BERT",
        "Memory": "~2 GB RAM during training",
        "Suitable for": "CPU-based training on student laptops"
    }
    
    print("=" * 50)
    print("MODEL SPECIFICATIONS")
    print("=" * 50)
    for key, value in info.items():
        print(f"{key:20s}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    # Display model information
    get_model_info()
    
    # Test model initialization
    print("\nInitializing model...")
    detector = FakeNewsDetector(max_length=128)
    
    # Test prediction
    sample_texts = [
        "Scientists discover new planet in solar system",
        "Miracle diet makes you lose 50 pounds overnight"
    ]
    
    print("\nTesting prediction...")
    predictions = detector.predict(sample_texts)
    probabilities = detector.predict_proba(sample_texts)
    
    for text, pred, prob in zip(sample_texts, predictions, probabilities):
        label = "FAKE" if pred == 1 else "REAL"
        confidence = prob[pred] * 100
        print(f"\nText: {text}")
        print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
