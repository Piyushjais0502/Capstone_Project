"""
Text Preprocessing Module for Fake News Detection
Minimal preprocessing approach suitable for transformer models
"""

import re
import pandas as pd
from typing import List, Tuple


class TextPreprocessor:
    """
    Lightweight text preprocessor for transformer-based models.
    Transformers handle most preprocessing internally, so we keep it minimal.
    """
    
    def __init__(self):
        """Initialize preprocessor with basic cleaning rules."""
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.html_pattern = re.compile(r'<.*?>')
        
    def clean_text(self, text: str) -> str:
        """
        Apply minimal cleaning to text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove HTML tags
        text = self.html_pattern.sub('', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Basic normalization (preserve case for transformers)
        text = text.strip()
        
        return text
    
    def prepare_dataset(self, df: pd.DataFrame, text_column: str, 
                       label_column: str) -> Tuple[List[str], List[int]]:
        """
        Prepare dataset for model training.
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Tuple of (texts, labels)
        """
        # Clean texts
        texts = df[text_column].apply(self.clean_text).tolist()
        
        # Extract labels (assuming binary: 0=Real, 1=Fake)
        labels = df[label_column].tolist()
        
        # Remove empty texts
        filtered_data = [(t, l) for t, l in zip(texts, labels) if len(t) > 0]
        texts, labels = zip(*filtered_data) if filtered_data else ([], [])
        
        return list(texts), list(labels)
    
    def load_sample_data(self, filepath: str, sample_size: int = None) -> pd.DataFrame:
        """
        Load dataset with optional sampling for resource constraints.
        
        Args:
            filepath: Path to CSV file
            sample_size: Number of samples to load (None for all)
            
        Returns:
            Pandas DataFrame
        """
        df = pd.read_csv(filepath)
        
        if sample_size and len(df) > sample_size:
            # Stratified sampling to maintain class balance
            df = df.groupby(df.columns[-1], group_keys=False).apply(
                lambda x: x.sample(min(len(x), sample_size // 2))
            )
        
        return df


def create_sample_dataset(output_path: str = "data/sample_data.csv"):
    """
    Create a small sample dataset for testing (placeholder).
    In actual implementation, use real datasets like LIAR.
    """
    import os
    os.makedirs("data", exist_ok=True)
    
    sample_data = {
        'text': [
            "Scientists confirm new breakthrough in renewable energy technology",
            "BREAKING: Celebrity spotted with alien spacecraft in backyard",
            "Government announces new policy for education reform",
            "Miracle cure discovered that doctors don't want you to know about",
            "Research study shows correlation between exercise and mental health",
            "Shocking truth revealed: Moon landing was definitely fake"
        ],
        'label': [0, 1, 0, 1, 0, 1]  # 0=Real, 1=Fake
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created at {output_path}")
    return df


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = TextPreprocessor()
    
    # Create sample data
    df = create_sample_dataset()
    
    # Test cleaning
    sample_text = "Check this out: https://fake-news.com <b>SHOCKING</b>   news!!!"
    cleaned = preprocessor.clean_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")
    
    # Test dataset preparation
    texts, labels = preprocessor.prepare_dataset(df, 'text', 'label')
    print(f"\nPrepared {len(texts)} samples")
    print(f"Class distribution: Real={labels.count(0)}, Fake={labels.count(1)}")
