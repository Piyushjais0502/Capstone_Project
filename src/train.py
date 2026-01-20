"""
Training Module for Fake News Detection
Lightweight training approach suitable for CPU and limited resources
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
import os

from model import FakeNewsDetector


class FakeNewsDataset(Dataset):
    """Custom Dataset for fake news detection."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class Trainer:
    """
    Training manager for fake news detection model.
    Optimized for resource-constrained environments.
    """
    
    def __init__(self, model: FakeNewsDetector, learning_rate: float = 2e-5,
                 batch_size: int = 8, epochs: int = 3):
        """
        Initialize trainer.
        
        Args:
            model: FakeNewsDetector instance
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size (small for CPU)
            epochs: Number of training epochs (limited for interim project)
        """
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = model.device
        
        self.train_losses = []
        self.val_losses = []
    
    def prepare_data(self, texts: List[str], labels: List[int], 
                    test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare train and validation dataloaders.
        
        Args:
            texts: List of text samples
            labels: List of labels
            test_size: Validation split ratio
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = FakeNewsDataset(
            train_texts, train_labels, self.model.tokenizer, self.model.max_length
        )
        val_dataset = FakeNewsDataset(
            val_texts, val_labels, self.model.tokenizer, self.model.max_length
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, optimizer, scheduler) -> float:
        """Train for one epoch."""
        self.model.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, texts: List[str], labels: List[int], save_path: str = "models/fake_news_model"):
        """
        Complete training pipeline.
        
        Args:
            texts: Training texts
            labels: Training labels
            save_path: Path to save trained model
        """
        print("=" * 60)
        print("TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Epochs: {self.epochs}")
        print(f"Max Sequence Length: {self.model.max_length}")
        print("=" * 60)
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(texts, labels)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print("-" * 60)
            
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.model.save_model(save_path)
                print(f"✓ Model saved (best validation loss: {val_loss:.4f})")
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)


if __name__ == "__main__":
    from preprocessing import TextPreprocessor, create_sample_dataset
    
    # Create sample data
    print("Creating sample dataset...")
    df = create_sample_dataset()
    
    # Preprocess
    preprocessor = TextPreprocessor()
    texts, labels = preprocessor.prepare_dataset(df, 'text', 'label')
    
    # Initialize model
    print("\nInitializing model...")
    detector = FakeNewsDetector(max_length=128)
    
    # Train (with very small sample - just for demonstration)
    print("\nStarting training...")
    trainer = Trainer(detector, batch_size=2, epochs=2)
    trainer.train(texts, labels)
    
    print("\nTraining demonstration completed!")
