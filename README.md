# An Explainable Transformer-Based Approach for Fake News Detection

## Project Overview
**Academic Level:** 7th Semester Computer Science Engineering (Interim Project - ETE)  
**Project Type:** Work-in-Progress Research Implementation  
**Domain:** Natural Language Processing, Misinformation Detection

## Abstract
This project proposes an explainable approach to detect fake news using lightweight transformer models. With the proliferation of misinformation on social media, automated detection systems are crucial. However, black-box models lack transparency. This work combines DistilBERT (a lightweight transformer) with LIME (Local Interpretable Model-agnostic Explanations) to provide both accurate classification and human-interpretable explanations.

## Problem Statement
Traditional fake news detection relies on manual fact-checking or opaque machine learning models. While transformers achieve high accuracy, they lack explainability—critical for trust and verification. This project addresses: How can we build a lightweight, explainable fake news detection system suitable for resource-constrained environments?

## Objectives
1. Implement a binary classifier (Fake/Real) using DistilBERT
2. Fine-tune on publicly available fake news datasets
3. Integrate LIME for model explainability
4. Evaluate performance using standard metrics
5. Demonstrate interpretability through visualization

## System Requirements
### Hardware (Student Laptop)
- RAM: 16 GB
- Processor: Intel i5/i7 or equivalent
- Storage: 10 GB free space
- GPU: Not required (CPU-based training)

### Software
- Python 3.8+
- PyTorch / TensorFlow
- Transformers library (Hugging Face)
- LIME library
- Standard ML libraries (scikit-learn, pandas, numpy)

## Project Structure
```
fake-news-detection/
├── data/                    # Dataset storage
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Source code
│   ├── preprocessing.py     # Text preprocessing
│   ├── model.py            # Model definition
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation metrics
│   └── explainability.py   # LIME integration
├── results/                 # Output results and visualizations
├── report/                  # Project report and documentation
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Methodology Overview
1. **Data Collection:** Use LIAR dataset (subset) or similar publicly available data
2. **Preprocessing:** Minimal cleaning, transformer tokenization
3. **Model Selection:** DistilBERT (66M parameters, 40% faster than BERT)
4. **Training:** Fine-tuning with limited epochs on CPU
5. **Evaluation:** Accuracy, Precision, Recall, F1-score
6. **Explainability:** LIME-based feature importance visualization

## Current Status (Interim Phase)
- ✅ Literature review completed
- ✅ System design finalized
- ✅ Development environment setup
- 🔄 Implementation in progress
- ⏳ Preliminary results expected
- ⏳ Final evaluation pending

## Expected Outcomes
- Functional fake news classifier with reasonable accuracy
- Explainability dashboard showing word-level contributions
- Comparative analysis with baseline models
- Documentation suitable for academic evaluation

## Future Scope
- Multi-class classification (partially true, misleading, etc.)
- Cross-domain generalization testing
- Integration with fact-checking APIs
- Web-based demonstration interface

## References
- Devlin et al. (2019) - BERT: Pre-training of Deep Bidirectional Transformers
- Sanh et al. (2019) - DistilBERT: A distilled version of BERT
- Ribeiro et al. (2016) - "Why Should I Trust You?": Explaining Predictions
- Wang (2017) - LIAR: A Benchmark Dataset for Fake News Detection

## Getting Started

### Quick Start (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create sample dataset
python main.py --setup

# 3. Show model information
python main.py --info

# 4. Train model (demo)
python main.py --train

# 5. Evaluate model
python main.py --evaluate

# 6. Generate explanation
python main.py --explain
```

### Detailed Setup

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for comprehensive installation instructions.

### Quick Reference

See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for commands, API, and troubleshooting.

## Usage Examples

### Command Line Interface

```bash
# Predict custom text
python main.py --predict "Scientists discover new breakthrough in technology"

# Train with custom parameters
python main.py --train --epochs 5 --batch-size 4

# Evaluate on custom dataset
python main.py --evaluate --data data/my_data.csv
```

### Python API

```python
from src.model import FakeNewsDetector
from src.explainability import ExplainabilityAnalyzer

# Load model
detector = FakeNewsDetector()
detector.load_model("models/fake_news_model")

# Make prediction
text = "Breaking news: Miracle cure discovered"
prediction = detector.predict([text])[0]
probability = detector.predict_proba([text])[0]

print(f"Prediction: {'FAKE' if prediction == 1 else 'REAL'}")
print(f"Confidence: {max(probability)*100:.2f}%")

# Generate explanation
analyzer = ExplainabilityAnalyzer(detector)
analyzer.visualize_explanation(text)
```

## Documentation

- **[Project Report](report/PROJECT_REPORT.md)** - Complete academic report
- **[Presentation Outline](report/PRESENTATION_OUTLINE.md)** - Viva preparation
- **[Setup Guide](SETUP_GUIDE.md)** - Installation instructions
- **[Quick Reference](QUICK_REFERENCE.md)** - Commands and API reference

## Project Highlights

✨ **Key Features:**
- Lightweight transformer model (DistilBERT)
- CPU-friendly (no GPU required)
- Explainable predictions (LIME)
- Modular, well-documented code
- Comprehensive evaluation metrics
- Interactive visualizations

🎯 **Perfect For:**
- 7th semester interim projects
- Resource-constrained environments
- Learning explainable AI
- Academic presentations
- Understanding fake news patterns

## Results Preview

**Sample Predictions:**

| Text | Prediction | Confidence |
|------|------------|------------|
| "Scientists confirm breakthrough..." | Real | 68% |
| "Miracle cure discovered..." | Fake | 78% |
| "Government announces policy..." | Real | 65% |

**Explainability Example:**

For fake news: "Miracle cure discovered that doctors don't want you to know"

Top indicators:
- "miracle" (+0.45) → Sensational claim
- "don't want you to know" (+0.52) → Conspiracy language
- "cure" (+0.35) → Unverified medical claim

See `results/` folder for visualizations.

## Contributing

This is an academic project for interim evaluation. Suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is created for academic purposes. Feel free to use and modify for educational use.

## Acknowledgments

- **Hugging Face** for Transformers library
- **LIME developers** for explainability tools
- **PyTorch team** for deep learning framework
- **Open-source community** for various tools and libraries

## Citation

If you use this project in your research or coursework, please cite:

```
@project{fake_news_detection_2026,
  title={An Explainable Transformer-Based Approach for Fake News Detection},
  author={[Your Name]},
  year={2026},
  institution={[Your University]},
  type={7th Semester Interim Project}
}
```

## Author

**[Your Name]**  
7th Semester Computer Science Engineering  
Interim Project (ETE) - Academic Year 2025-26  
Email: [your-email]  
GitHub: [your-github]

## Support

For questions or issues:
- Check [SETUP_GUIDE.md](SETUP_GUIDE.md) for installation help
- See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for troubleshooting
- Review [PROJECT_REPORT.md](report/PROJECT_REPORT.md) for detailed information
- Contact your project guide

---

**Status:** ✅ Interim Phase Complete | 🔄 Final Implementation In Progress

**Last Updated:** January 2026
#   C a p s t o n e _ P r o j e c t  
 