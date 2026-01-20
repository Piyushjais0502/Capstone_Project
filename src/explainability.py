"""
Explainability Module using LIME
Provides interpretable explanations for model predictions
"""

import numpy as np
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from typing import List, Tuple
import os

from model import FakeNewsDetector


class ExplainabilityAnalyzer:
    """
    LIME-based explainability for fake news detection.
    Provides word-level importance for predictions.
    """
    
    def __init__(self, model: FakeNewsDetector):
        """
        Initialize explainability analyzer.
        
        Args:
            model: Trained FakeNewsDetector instance
        """
        self.model = model
        self.class_names = ['Real', 'Fake']
        
        # Initialize LIME explainer
        self.explainer = LimeTextExplainer(class_names=self.class_names)
    
    def predict_proba_wrapper(self, texts: List[str]) -> np.ndarray:
        """
        Wrapper function for LIME compatibility.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of probabilities
        """
        probabilities = self.model.predict_proba(texts)
        return np.array(probabilities)
    
    def explain_instance(self, text: str, num_features: int = 10, num_samples: int = 100) -> Tuple:
        """
        Generate explanation for a single text instance.
        
        Args:
            text: Input text to explain
            num_features: Number of top features to show
            num_samples: Number of samples for LIME (lower = faster)
            
        Returns:
            Tuple of (explanation object, prediction, probability, inverted_features)
        """
        # Get prediction
        prediction = self.model.predict([text])[0]
        probability = self.model.predict_proba([text])[0]
        
        # Generate explanation (reduced samples for speed)
        explanation = self.explainer.explain_instance(
            text,
            self.predict_proba_wrapper,
            num_features=num_features,
            num_samples=num_samples,  # Reduced from default 5000 to 100
            top_labels=2
        )
        
        # Get features and invert for intuitive interpretation
        # We want: + = REAL, - = FAKE
        features = explanation.as_list(label=prediction)
        
        # If prediction is FAKE (1), invert the signs
        if prediction == 1:
            inverted_features = [(word, -importance) for word, importance in features]
        else:
            inverted_features = features
        
        return explanation, prediction, probability, inverted_features
    
    def visualize_explanation(self, text: str, num_features: int = 10,
                            save_path: str = None, num_samples: int = 100):
        """
        Visualize explanation for a text instance.
        
        Args:
            text: Input text to explain
            num_features: Number of features to display
            save_path: Path to save visualization (optional)
            num_samples: Number of LIME samples (lower = faster)
        """
        explanation, prediction, probability, features = self.explain_instance(text, num_features, num_samples)
        
        # Print prediction info
        pred_label = self.class_names[prediction]
        confidence = probability[prediction] * 100
        
        print("\n" + "=" * 70)
        print("EXPLAINABILITY ANALYSIS")
        print("=" * 70)
        print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
        print(f"\nPrediction: {pred_label}")
        print(f"Confidence: {confidence:.2f}%")
        print("\n" + "-" * 70)
        print("WORD IMPORTANCE (Top Features)")
        print("-" * 70)
        
        # Use inverted features for intuitive display
        for word, importance in features:
            # Now: positive = REAL, negative = FAKE
            direction = "→ REAL" if importance > 0 else "→ FAKE"
            bar = "█" * min(int(abs(importance) * 50), 20)
            print(f"{word:20s} {importance:+.4f} {bar} {direction}")
        
        print("=" * 70)
        print(f"\nInterpretation:")
        print(f"  + Positive values → Evidence for REAL news")
        print(f"  - Negative values → Evidence for FAKE news")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        words = [f[0] for f in features]
        importances = [f[1] for f in features]
        colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in importances]
        
        bars = ax.barh(words, importances, color=colors)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Word Importance for "{pred_label}" Prediction\n'
                 f'Confidence: {confidence:.2f}% | Green=Real, Red=Fake', 
                 fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return explanation
    
    def compare_explanations(self, texts: List[str], 
                           save_path: str = "results/explanation_comparison.png"):
        """
        Compare explanations for multiple texts.
        
        Args:
            texts: List of texts to compare
            save_path: Path to save comparison plot
        """
        n_texts = len(texts)
        fig, axes = plt.subplots(n_texts, 1, figsize=(12, 4 * n_texts))
        
        if n_texts == 1:
            axes = [axes]
        
        for idx, text in enumerate(texts):
            explanation, prediction, probability = self.explain_instance(text, num_features=8)
            
            # Get features
            features = explanation.as_list(label=prediction)
            words = [f[0] for f in features]
            importances = [f[1] for f in features]
            
            # Plot
            colors = ['#e74c3c' if imp > 0 else '#2ecc71' for imp in importances]
            axes[idx].barh(words, importances, color=colors)
            axes[idx].set_xlabel('Importance Score', fontsize=10)
            axes[idx].set_title(f'Text {idx+1}: {self.class_names[prediction]} '
                              f'({probability[prediction]*100:.1f}% confidence)',
                              fontsize=11, fontweight='bold')
            axes[idx].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to {save_path}")
        plt.close()
    
    def generate_html_explanation(self, text: str, 
                                 save_path: str = "results/explanation.html"):
        """
        Generate interactive HTML explanation.
        
        Args:
            text: Input text
            save_path: Path to save HTML file
        """
        explanation, prediction, probability = self.explain_instance(text)
        
        # Generate HTML
        html = explanation.as_html()
        
        # Add custom styling and header
        pred_label = self.class_names[prediction]
        confidence = probability[prediction] * 100
        
        header = f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px;">
            <h2 style="color: #2c3e50;">Fake News Detection - Explainability Report</h2>
            <p><strong>Prediction:</strong> <span style="color: {'#e74c3c' if prediction == 1 else '#2ecc71'}; font-size: 18px;">{pred_label}</span></p>
            <p><strong>Confidence:</strong> {confidence:.2f}%</p>
            <p><strong>Text:</strong> {text}</p>
        </div>
        """
        
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Explainability Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
            </style>
        </head>
        <body>
            {header}
            {html}
        </body>
        </html>
        """
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"HTML explanation saved to {save_path}")


if __name__ == "__main__":
    from model import FakeNewsDetector
    
    # Initialize model
    print("Initializing model...")
    detector = FakeNewsDetector(max_length=128)
    
    # Initialize explainability analyzer
    analyzer = ExplainabilityAnalyzer(detector)
    
    # Sample texts for explanation
    sample_texts = [
        "Scientists at MIT have developed a new breakthrough in renewable energy technology that could revolutionize solar power efficiency.",
        "SHOCKING: Doctors hate this one weird trick that makes you lose 50 pounds overnight without any effort or exercise required!",
        "The government announced new education reforms aimed at improving student outcomes and teacher training programs."
    ]
    
    print("\n" + "=" * 70)
    print("EXPLAINABILITY DEMONSTRATION")
    print("=" * 70)
    
    # Explain individual instances
    for i, text in enumerate(sample_texts, 1):
        print(f"\n--- Example {i} ---")
        analyzer.visualize_explanation(
            text, 
            num_features=8,
            save_path=f"results/explanation_{i}.png"
        )
    
    # Compare explanations
    print("\nGenerating comparison plot...")
    analyzer.compare_explanations(sample_texts[:2])
    
    # Generate HTML report
    print("\nGenerating HTML report...")
    analyzer.generate_html_explanation(sample_texts[0])
    
    print("\n" + "=" * 70)
    print("EXPLAINABILITY DEMONSTRATION COMPLETED")
    print("=" * 70)
