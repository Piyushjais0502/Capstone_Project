"""
Fake News Detection Web Interface
A user-friendly GUI for testing the fake news detector
"""

import gradio as gr
import sys
sys.path.append('src')

from model import FakeNewsDetector
from explainability import ExplainabilityAnalyzer
import matplotlib.pyplot as plt
import io
from PIL import Image

# Load model
print("Loading model...")
detector = FakeNewsDetector()
try:
    detector.load_model('models/fake_news_model')
    analyzer = ExplainabilityAnalyzer(detector)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"⚠ Warning: {e}")
    print("Please train the model first: python main.py --train")

def predict_news(text):
    """Predict if news is fake or real"""
    if not text or len(text.strip()) == 0:
        return "⚠️ Please enter some text", "", None
    
    try:
        # Get prediction
        prediction = detector.predict([text])[0]
        probabilities = detector.predict_proba([text])[0]
        
        # Format result
        label = "🚨 FAKE NEWS" if prediction == 1 else "✅ REAL NEWS"
        confidence = probabilities[prediction] * 100
        
        # Create detailed result
        result = f"""
### {label}
**Confidence:** {confidence:.1f}%

**Probability Breakdown:**
- Real News: {probabilities[0]*100:.1f}%
- Fake News: {probabilities[1]*100:.1f}%
        """
        
        # Create confidence bar chart
        fig, ax = plt.subplots(figsize=(8, 3))
        categories = ['Real', 'Fake']
        values = [probabilities[0]*100, probabilities[1]*100]
        colors = ['#2ecc71' if prediction == 0 else '#95a5a6', 
                  '#e74c3c' if prediction == 1 else '#95a5a6']
        
        bars = ax.barh(categories, values, color=colors)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Confidence (%)', fontsize=12)
        ax.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
        
        # Add percentage labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 2, i, f'{val:.1f}%', va='center', fontsize=11)
        
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return result, label, img
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", None

def explain_prediction(text):
    """Generate explanation for the prediction"""
    if not text or len(text.strip()) == 0:
        return "⚠️ Please enter some text", None
    
    try:
        # Generate explanation (faster with fewer samples)
        explanation, prediction, probability, features = analyzer.explain_instance(text, num_features=10, num_samples=100)
        
        # Create explanation text
        label = "FAKE" if prediction == 1 else "REAL"
        confidence = probability[prediction] * 100
        
        explain_text = f"""
### Explanation for "{label}" Prediction
**Confidence:** {confidence:.1f}%

**Top Words Contributing to the Decision:**

"""
        
        # Now features are already inverted: + = REAL, - = FAKE
        for word, importance in features:
            direction = "→ REAL" if importance > 0 else "→ FAKE"
            bar = "█" * min(int(abs(importance) * 100), 20)
            explain_text += f"- **{word}**: {importance:+.4f} {bar} {direction}\n"
        
        explain_text += f"""

**How to Read:**
- **Positive values (+)** → Evidence for REAL news (Green)
- **Negative values (-)** → Evidence for FAKE news (Red)
- **Larger magnitude** = stronger influence

**Interpretation:** Words with negative values made the model think this is fake news, 
while words with positive values suggested it might be real.
"""
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        words = [f[0] for f in features]
        importances = [f[1] for f in features]
        
        # Green for positive (REAL), Red for negative (FAKE)
        colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in importances]
        
        bars = ax.barh(words, importances, color=colors)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Word Importance for "{label}" Prediction\n'
                    f'Confidence: {confidence:.1f}% | Green=Real Evidence, Red=Fake Evidence', 
                    fontsize=13, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Evidence for REAL (+)'),
            Patch(facecolor='#e74c3c', label='Evidence for FAKE (-)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return explain_text, img
        
    except Exception as e:
        return f"❌ Error generating explanation: {str(e)}", None

# Sample texts for quick testing
examples = [
    ["Scientists at MIT develop new breakthrough in renewable energy technology"],
    ["SHOCKING: Miracle cure discovered that doctors don't want you to know about"],
    ["Government announces new education policy reforms"],
    ["You won't believe this one weird trick to lose 50 pounds overnight"],
    ["Research study shows correlation between exercise and mental health"],
    ["BREAKING: Celebrity spotted with alien spacecraft in backyard"]
]

# Create Gradio interface
with gr.Blocks(title="Fake News Detector") as demo:
    
    gr.Markdown("""
    # 🔍 Fake News Detection System
    ### An Explainable AI Approach using DistilBERT + LIME
    
    Enter any news text below to check if it's fake or real. The system will:
    1. **Predict** if the news is fake or real
    2. **Explain** which words influenced the decision
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="📝 Enter News Text",
                placeholder="Type or paste news text here...",
                lines=5
            )
            
            with gr.Row():
                predict_btn = gr.Button("🔍 Check News", variant="primary", size="lg")
                explain_btn = gr.Button("💡 Explain Prediction", variant="secondary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", size="lg")
    
    gr.Markdown("### 📊 Results")
    
    with gr.Row():
        with gr.Column():
            prediction_output = gr.Markdown(label="Prediction")
            prediction_label = gr.Textbox(label="Result", interactive=False)
            confidence_plot = gr.Image(label="Confidence Visualization")
    
    gr.Markdown("### 💡 Explanation (Why this prediction?)")
    
    with gr.Row():
        with gr.Column():
            explanation_output = gr.Markdown(label="Explanation")
            explanation_plot = gr.Image(label="Word Importance")
    
    gr.Markdown("---")
    gr.Markdown("### 🎯 Try These Examples:")
    gr.Examples(
        examples=examples,
        inputs=input_text,
        label="Click any example to test"
    )
    
    gr.Markdown("""
    ---
    ### ℹ️ About This System
    
    **Technology Stack:**
    - **Model:** DistilBERT (Lightweight Transformer)
    - **Explainability:** LIME (Local Interpretable Model-agnostic Explanations)
    - **Training:** Fine-tuned on fake news dataset
    
    **How It Works:**
    1. Text is processed and tokenized
    2. DistilBERT analyzes the content
    3. Classification head predicts Fake/Real
    4. LIME explains which words mattered most
    
    **Note:** This is a demonstration system for academic purposes. 
    Always verify important news from multiple reliable sources.
    
    ---
    *7th Semester CSE Interim Project | Explainable Fake News Detection*
    """)
    
    # Button actions
    predict_btn.click(
        fn=predict_news,
        inputs=input_text,
        outputs=[prediction_output, prediction_label, confidence_plot]
    )
    
    explain_btn.click(
        fn=explain_prediction,
        inputs=input_text,
        outputs=[explanation_output, explanation_plot]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", "", None, "", None),
        outputs=[input_text, prediction_output, prediction_label, 
                confidence_plot, explanation_output, explanation_plot]
    )

# Launch the app
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Fake News Detection Web Interface")
    print("="*60)
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="127.0.0.1",
        server_port=7861,  # Changed port
        show_error=True,
        inbrowser=True,  # Automatically open browser
        theme=gr.themes.Soft()  # Moved theme here for Gradio 6.0
    )
