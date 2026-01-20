"""
Main Script for Fake News Detection Project
Provides a simple interface to run different components
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from preprocessing import TextPreprocessor, create_sample_dataset
from model import FakeNewsDetector, get_model_info
from train import Trainer
from evaluate import ModelEvaluator
from explainability import ExplainabilityAnalyzer


def setup_sample_data():
    """Create sample dataset for testing."""
    print("Creating sample dataset...")
    df = create_sample_dataset('data/sample_data.csv')
    print(f"✓ Sample dataset created with {len(df)} samples")
    return df


def train_model(data_path='data/sample_data.csv', model_path='models/fake_news_model',
                batch_size=8, epochs=3, max_length=128):
    """Train the fake news detection model."""
    import pandas as pd
    
    print("\n" + "="*60)
    print("TRAINING FAKE NEWS DETECTION MODEL")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Preprocess
    print("Preprocessing data...")
    preprocessor = TextPreprocessor()
    texts, labels = preprocessor.prepare_dataset(df, 'text', 'label')
    print(f"✓ Prepared {len(texts)} samples")
    
    # Initialize model
    print("\nInitializing model...")
    detector = FakeNewsDetector(max_length=max_length)
    
    # Train
    print("\nStarting training...")
    trainer = Trainer(detector, batch_size=batch_size, epochs=epochs)
    trainer.train(texts, labels, save_path=model_path)
    
    print("\n✓ Training completed successfully!")
    print(f"Model saved to {model_path}")


def evaluate_model(data_path='data/sample_data.csv', 
                  model_path='models/fake_news_model'):
    """Evaluate the trained model."""
    import pandas as pd
    
    print("\n" + "="*60)
    print("EVALUATING FAKE NEWS DETECTION MODEL")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Preprocess
    print("Preprocessing data...")
    preprocessor = TextPreprocessor()
    texts, labels = preprocessor.prepare_dataset(df, 'text', 'label')
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    detector = FakeNewsDetector()
    detector.load_model(model_path)
    
    # Evaluate
    print("\nEvaluating model...")
    evaluator = ModelEvaluator(detector)
    metrics = evaluator.evaluate(texts, labels, save_results=True)
    
    print("\n✓ Evaluation completed successfully!")
    print("Results saved to results/ folder")


def explain_prediction(text=None, model_path='models/fake_news_model'):
    """Generate explanation for a prediction."""
    print("\n" + "="*60)
    print("GENERATING EXPLANATION")
    print("="*60)
    
    # Default text if none provided
    if text is None:
        text = "Miracle cure discovered that doctors don't want you to know about"
        print(f"\nUsing default text: {text}")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    detector = FakeNewsDetector()
    detector.load_model(model_path)
    
    # Generate explanation
    print("\nGenerating explanation (this may take a few seconds)...")
    analyzer = ExplainabilityAnalyzer(detector)
    analyzer.visualize_explanation(
        text, 
        num_features=10,
        save_path="results/explanation.png"
    )
    
    print("\n✓ Explanation generated successfully!")
    print("Visualization saved to results/explanation.png")


def predict_text(text, model_path='models/fake_news_model'):
    """Make a prediction on custom text."""
    print("\n" + "="*60)
    print("FAKE NEWS PREDICTION")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    detector = FakeNewsDetector()
    detector.load_model(model_path)
    
    # Predict
    print("\nAnalyzing text...")
    prediction = detector.predict([text])[0]
    probability = detector.predict_proba([text])[0]
    
    # Display results
    label = "FAKE" if prediction == 1 else "REAL"
    confidence = probability[prediction] * 100
    
    print("\n" + "-"*60)
    print(f"Text: {text}")
    print("-"*60)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Probabilities: Real={probability[0]*100:.2f}%, Fake={probability[1]*100:.2f}%")
    print("-"*60)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Fake News Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --setup              # Create sample dataset
  python main.py --info               # Show model information
  python main.py --train              # Train model on sample data
  python main.py --evaluate           # Evaluate trained model
  python main.py --explain            # Generate explanation
  python main.py --predict "Your text here"  # Predict custom text
        """
    )
    
    parser.add_argument('--setup', action='store_true',
                       help='Create sample dataset')
    parser.add_argument('--info', action='store_true',
                       help='Display model information')
    parser.add_argument('--train', action='store_true',
                       help='Train the model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the model')
    parser.add_argument('--explain', action='store_true',
                       help='Generate explanation for sample text')
    parser.add_argument('--predict', type=str, metavar='TEXT',
                       help='Predict label for custom text')
    
    parser.add_argument('--data', type=str, default='data/sample_data.csv',
                       help='Path to dataset (default: data/sample_data.csv)')
    parser.add_argument('--model', type=str, default='models/fake_news_model',
                       help='Path to model (default: models/fake_news_model)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs (default: 3)')
    parser.add_argument('--max-length', type=int, default=128,
                       help='Maximum sequence length (default: 128)')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    try:
        if args.setup:
            setup_sample_data()
        
        if args.info:
            get_model_info()
        
        if args.train:
            train_model(
                data_path=args.data,
                model_path=args.model,
                batch_size=args.batch_size,
                epochs=args.epochs,
                max_length=args.max_length
            )
        
        if args.evaluate:
            evaluate_model(
                data_path=args.data,
                model_path=args.model
            )
        
        if args.explain:
            explain_prediction(model_path=args.model)
        
        if args.predict:
            predict_text(args.predict, model_path=args.model)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease check:")
        print("1. All dependencies are installed (pip install -r requirements.txt)")
        print("2. Dataset exists (run with --setup to create sample data)")
        print("3. Model is trained (run with --train first)")
        sys.exit(1)


if __name__ == "__main__":
    main()
