# Quick Reference Card
## Fake News Detection Project

### Common Commands

```bash
# Setup
python main.py --setup              # Create sample dataset
python main.py --info               # Show model info

# Training
python main.py --train              # Train on sample data
python main.py --train --epochs 5   # Train for 5 epochs
python main.py --train --batch-size 4  # Smaller batch size

# Evaluation
python main.py --evaluate           # Evaluate model

# Prediction
python main.py --predict "Your text here"  # Predict custom text

# Explainability
python main.py --explain            # Generate explanation
```

### Project Structure

```
fake-news-detection/
├── src/                 # Source code
│   ├── preprocessing.py # Text cleaning
│   ├── model.py        # DistilBERT wrapper
│   ├── train.py        # Training pipeline
│   ├── evaluate.py     # Metrics & plots
│   └── explainability.py # LIME integration
├── data/               # Datasets
├── models/             # Saved models
├── results/            # Visualizations
├── notebooks/          # Jupyter notebooks
└── report/             # Documentation
```

### Key Concepts

**DistilBERT:**
- 66M parameters
- 40% faster than BERT
- 97% of BERT's performance
- Runs on CPU

**LIME:**
- Local explanations
- Word-level importance
- Model-agnostic
- Interpretable

**Metrics:**
- Accuracy: Overall correctness
- Precision: Fake prediction accuracy
- Recall: Fake news coverage
- F1-Score: Balanced measure

### Python API

```python
# Preprocessing
from src.preprocessing import TextPreprocessor
preprocessor = TextPreprocessor()
texts, labels = preprocessor.prepare_dataset(df, 'text', 'label')

# Model
from src.model import FakeNewsDetector
detector = FakeNewsDetector(max_length=128)
predictions = detector.predict(texts)
probabilities = detector.predict_proba(texts)

# Training
from src.train import Trainer
trainer = Trainer(detector, batch_size=8, epochs=3)
trainer.train(texts, labels, save_path="models/my_model")

# Evaluation
from src.evaluate import ModelEvaluator
evaluator = ModelEvaluator(detector)
metrics = evaluator.evaluate(test_texts, test_labels)

# Explainability
from src.explainability import ExplainabilityAnalyzer
analyzer = ExplainabilityAnalyzer(detector)
analyzer.visualize_explanation(text, save_path="results/exp.png")
```

### Hyperparameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| batch_size | 8 | 4-16 | Lower for less RAM |
| learning_rate | 2e-5 | 1e-5 to 5e-5 | Standard for transformers |
| epochs | 3 | 2-10 | More = better but slower |
| max_length | 128 | 64-512 | Shorter = faster |

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce batch_size to 4 |
| Slow training | Reduce max_length to 64 |
| Low accuracy | Increase epochs to 5-10 |
| LIME too slow | Reduce num_samples to 500 |

### File Sizes

- DistilBERT model: ~250 MB
- Sample dataset: < 1 KB
- LIAR dataset: ~2 MB
- Trained model: ~250 MB
- Results (plots): ~1 MB

### Expected Performance

| Metric | Expected Range |
|--------|---------------|
| Accuracy | 75-85% |
| Precision | 73-83% |
| Recall | 75-85% |
| F1-Score | 74-84% |
| Training Time | 15-20 min/epoch |
| Inference Time | 0.5 sec/sample |

### Important Notes

✅ **Do:**
- Start with sample data
- Monitor RAM usage
- Save model checkpoints
- Document experiments
- Use explainability

❌ **Don't:**
- Train on full dataset without testing
- Use batch_size > 16 on CPU
- Ignore validation loss
- Skip preprocessing
- Over-claim results

### Viva Preparation

**Key Points to Remember:**
1. Why DistilBERT? (Efficiency + Performance)
2. How LIME works? (Perturbation + Local model)
3. Metrics meaning? (Accuracy vs F1)
4. Limitations? (English only, 128 tokens, CPU)
5. Future work? (Multimodal, multilingual, real-time)

**Demo Checklist:**
- [ ] Sample predictions ready
- [ ] Explanation visualizations saved
- [ ] Metrics computed
- [ ] Code walkthrough prepared
- [ ] Architecture diagram ready

### Useful Links

- **Hugging Face:** https://huggingface.co/distilbert-base-uncased
- **LIME Docs:** https://lime-ml.readthedocs.io
- **LIAR Dataset:** https://www.cs.ucsb.edu/~william/data/
- **PyTorch Docs:** https://pytorch.org/docs/

### Contact & Support

- Project Guide: [Guide Name]
- Email: [Your Email]
- GitHub: [Repository Link]

---

**Last Updated:** January 2026  
**Version:** 1.0 (Interim Phase)
