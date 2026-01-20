# Setup Guide for Fake News Detection Project

This guide will help you set up and run the project on your laptop.

## Prerequisites

- Python 3.8 or higher
- 16 GB RAM (minimum 8 GB)
- 10 GB free disk space
- Internet connection (for downloading pre-trained models)

## Step-by-Step Installation

### 1. Install Python

**Windows:**
- Download from https://www.python.org/downloads/
- During installation, check "Add Python to PATH"
- Verify: Open Command Prompt and type `python --version`

**Linux/Mac:**
```bash
# Usually pre-installed, verify with:
python3 --version

# If not installed:
# Ubuntu/Debian: sudo apt-get install python3 python3-pip
# Mac: brew install python3
```

### 2. Create Project Directory

```bash
# Navigate to desired location
cd Documents

# Create and enter project directory
mkdir fake-news-detection
cd fake-news-detection
```

### 3. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

### 4. Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# This will take 5-10 minutes depending on your internet speed
```

### 5. Verify Installation

```bash
# Test imports
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import lime; print('LIME installed successfully')"
```

### 6. Download Pre-trained Model (Optional)

The model will be downloaded automatically on first use, but you can pre-download:

```python
python -c "from transformers import DistilBertTokenizer, DistilBertForSequenceClassification; DistilBertTokenizer.from_pretrained('distilbert-base-uncased'); DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2); print('Model downloaded successfully')"
```

This downloads ~250 MB and may take a few minutes.

## Quick Start

### Test the Installation

```bash
# Create sample dataset
python src/preprocessing.py

# Test model initialization
python src/model.py

# You should see model information and sample predictions
```

### Run Data Exploration

```bash
# Start Jupyter Notebook
jupyter notebook

# Open notebooks/01_data_exploration.ipynb
# Run all cells to explore the sample dataset
```

### Train the Model (Demo)

```bash
# Train on sample data (quick demo)
python src/train.py

# This will:
# 1. Create sample dataset
# 2. Initialize DistilBERT
# 3. Run 2 epochs of training
# 4. Save model to models/fake_news_model/
```

### Evaluate the Model

```bash
# Evaluate on sample data
python src/evaluate.py

# This will:
# 1. Load the model
# 2. Generate predictions
# 3. Compute metrics
# 4. Save visualizations to results/
```

### Generate Explanations

```bash
# Generate LIME explanations
python src/explainability.py

# This will:
# 1. Load the model
# 2. Explain sample predictions
# 3. Save visualizations to results/
```

## Working with Real Data

### Download LIAR Dataset

1. Visit: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
2. Download and extract to `data/` folder
3. You should have:
   - `data/train.tsv`
   - `data/test.tsv`
   - `data/valid.tsv`

### Prepare Dataset

```python
import pandas as pd
from src.preprocessing import TextPreprocessor

# Load LIAR dataset
df = pd.read_csv('data/train.tsv', sep='\t', header=None)
df.columns = ['id', 'label', 'statement', 'subject', 'speaker', 
              'job', 'state', 'party', 'context', 'history']

# Convert to binary labels
label_map = {
    'true': 0, 'mostly-true': 0, 'half-true': 0,
    'barely-true': 1, 'false': 1, 'pants-fire': 1
}
df['binary_label'] = df['label'].map(label_map)

# Save processed dataset
df[['statement', 'binary_label']].to_csv('data/processed_train.csv', index=False)
```

### Train on Real Data

```python
from src.preprocessing import TextPreprocessor
from src.model import FakeNewsDetector
from src.train import Trainer
import pandas as pd

# Load data
df = pd.read_csv('data/processed_train.csv')

# Preprocess
preprocessor = TextPreprocessor()
texts, labels = preprocessor.prepare_dataset(df, 'statement', 'binary_label')

# Sample for quick training (optional)
# texts = texts[:5000]
# labels = labels[:5000]

# Initialize model
detector = FakeNewsDetector(max_length=128)

# Train
trainer = Trainer(detector, batch_size=8, epochs=3)
trainer.train(texts, labels, save_path="models/liar_model")
```

## Troubleshooting

### Issue: "No module named 'torch'"

**Solution:**
```bash
pip install torch
```

### Issue: Out of Memory Error

**Solution:** Reduce batch size in training
```python
trainer = Trainer(detector, batch_size=4, epochs=3)  # Reduced from 8
```

### Issue: Training is Very Slow

**Solutions:**
1. Reduce dataset size:
   ```python
   texts = texts[:1000]  # Use only 1000 samples
   labels = labels[:1000]
   ```

2. Reduce sequence length:
   ```python
   detector = FakeNewsDetector(max_length=64)  # Reduced from 128
   ```

3. Reduce epochs:
   ```python
   trainer = Trainer(detector, epochs=2)  # Reduced from 3
   ```

### Issue: LIME Takes Too Long

**Solution:** Reduce number of samples
```python
explanation = explainer.explain_instance(
    text, 
    model.predict_proba, 
    num_samples=500  # Reduced from 1000
)
```

### Issue: Model Download Fails

**Solution:** Check internet connection and try manual download
```bash
# Set cache directory
export TRANSFORMERS_CACHE=./cache

# Try downloading again
python -c "from transformers import DistilBertTokenizer; DistilBertTokenizer.from_pretrained('distilbert-base-uncased')"
```

## Project Structure

```
fake-news-detection/
├── data/                    # Dataset files
│   └── sample_data.csv
├── models/                  # Saved models
│   └── fake_news_model/
├── notebooks/               # Jupyter notebooks
│   └── 01_data_exploration.ipynb
├── results/                 # Output visualizations
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── explanation_*.png
├── src/                     # Source code
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── explainability.py
├── report/                  # Documentation
│   ├── PROJECT_REPORT.md
│   └── PRESENTATION_OUTLINE.md
├── requirements.txt         # Dependencies
├── README.md               # Project overview
└── SETUP_GUIDE.md          # This file
```

## Next Steps

1. **Explore the code:** Read through the source files to understand the implementation
2. **Run experiments:** Try different hyperparameters and configurations
3. **Analyze results:** Study the evaluation metrics and explanations
4. **Extend the project:** Implement features from the Future Scope section
5. **Prepare presentation:** Use the report and presentation outline for your viva

## Getting Help

- **Documentation:** Read the PROJECT_REPORT.md for detailed information
- **Code comments:** All source files are well-commented
- **Jupyter notebooks:** Interactive examples in the notebooks/ folder
- **Online resources:**
  - Hugging Face Transformers: https://huggingface.co/docs/transformers
  - LIME documentation: https://lime-ml.readthedocs.io
  - PyTorch tutorials: https://pytorch.org/tutorials

## Tips for Success

1. **Start small:** Test with sample data before using full dataset
2. **Monitor resources:** Keep an eye on RAM and CPU usage
3. **Save frequently:** Save model checkpoints during training
4. **Document changes:** Keep notes on experiments and results
5. **Ask questions:** Don't hesitate to seek help from your guide

Good luck with your project! 🚀
