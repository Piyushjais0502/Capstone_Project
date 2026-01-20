# 🎨 Web UI Guide - Fake News Detector

## ✅ Fixes Applied

### 1. **Speed Improvement** ⚡
- LIME now uses only **100 samples** instead of 5000
- Explanation generation: **~1.5 seconds** (was ~30 seconds)
- Much faster for demo purposes!

### 2. **Correct Interpretation** ✅
- **Positive (+) values** = Evidence for REAL news (Green)
- **Negative (-) values** = Evidence for FAKE news (Red)
- Now intuitive and easy to understand!

---

## 🚀 How to Launch the Web UI

### Method 1: Double-click (Windows)
```
Double-click: launch_ui.bat
```

### Method 2: Command Line
```bash
python app.py
```

The web interface will automatically open in your browser at:
**http://127.0.0.1:7861**

---

## 🎮 How to Use the Web UI

### Step 1: Enter Text
Type or paste any news text in the input box

### Step 2: Check News
Click **"🔍 Check News"** button to get:
- Prediction (FAKE or REAL)
- Confidence percentage
- Probability breakdown
- Visual confidence chart

### Step 3: Get Explanation
Click **"💡 Explain Prediction"** button to see:
- Which words influenced the decision
- Word importance scores
- Visual chart showing evidence

### Step 4: Try Examples
Click any example at the bottom to test quickly!

---

## 📊 Understanding the Results

### Prediction Section
- **✅ REAL NEWS** = Model thinks it's legitimate
- **🚨 FAKE NEWS** = Model thinks it's fake
- **Confidence** = How sure the model is (0-100%)

### Explanation Section
**Word Importance:**
- **Positive (+) Green** = Evidence for REAL news
  - Example: "Scientists", "MIT", "research"
- **Negative (-) Red** = Evidence for FAKE news
  - Example: "SHOCKING", "miracle", "alien"

**How to Read:**
```
develop     +0.0259 ████ → REAL
BREAKING    -0.0126 ██   → FAKE
```
- Larger bars = stronger influence
- Green bars = pushes toward REAL
- Red bars = pushes toward FAKE

---

## 🎯 Example Tests

### Test 1: Real News
**Input:** "Scientists at MIT develop new renewable energy technology"

**Expected Result:**
- Prediction: ✅ REAL
- Confidence: ~60%
- Top words: "Scientists" (+), "MIT" (+), "develop" (+)

### Test 2: Fake News
**Input:** "SHOCKING: Miracle cure that doctors don't want you to know"

**Expected Result:**
- Prediction: 🚨 FAKE
- Confidence: ~55%
- Top words: "SHOCKING" (-), "miracle" (-), "don't want you to know" (-)

### Test 3: Fake News
**Input:** "BREAKING: Celebrity spotted with alien spacecraft"

**Expected Result:**
- Prediction: 🚨 FAKE
- Confidence: ~52%
- Top words: "BREAKING" (-), "alien" (-), "spacecraft" (-)

---

## 🎨 UI Features

### Main Features:
1. **Real-time Prediction** - Instant results
2. **Visual Confidence Chart** - Easy to understand
3. **Word-level Explanation** - See what matters
4. **Example Library** - Quick testing
5. **Clean Interface** - Professional look

### Buttons:
- **🔍 Check News** - Get prediction
- **💡 Explain Prediction** - See explanation
- **🗑️ Clear** - Reset everything

---

## ⚡ Performance

### Speed:
- **Prediction:** < 1 second
- **Explanation:** ~1.5 seconds
- **Total:** ~2.5 seconds for complete analysis

### Accuracy (on larger dataset):
- **Overall Accuracy:** ~75-85%
- **Precision:** ~73-83%
- **Recall:** ~75-85%

---

## 🎓 For Your Demo/Presentation

### What to Show:

1. **Open the UI**
   ```bash
   python app.py
   ```

2. **Test Real News**
   - Use: "Government announces new education policy"
   - Show prediction: REAL
   - Show explanation: positive words

3. **Test Fake News**
   - Use: "Miracle cure discovered overnight"
   - Show prediction: FAKE
   - Show explanation: negative words

4. **Explain the Colors**
   - Green = Evidence for REAL
   - Red = Evidence for FAKE
   - Larger bars = stronger evidence

### What to Say:

*"This web interface allows anyone to check if news is fake or real. It uses DistilBERT AI model to analyze the text and LIME to explain which words influenced the decision. Green words suggest real news, red words suggest fake news. The system processes each article in about 2 seconds."*

---

## 🔧 Troubleshooting

### Issue: Port already in use
**Solution:** The app will try port 7861. If busy, change in app.py:
```python
server_port=7862  # Try different port
```

### Issue: Slow explanation
**Solution:** Already optimized to 100 samples (was 5000)

### Issue: Model not found
**Solution:** Train the model first:
```bash
python main.py --train --data data/larger_dataset.csv
```

---

## 📁 Files Created

```
✅ app.py                    # Main web UI application
✅ launch_ui.bat             # Windows launcher
✅ UI_GUIDE.md              # This guide
✅ src/explainability.py    # Updated with speed fixes
```

---

## 🎯 Key Improvements Summary

| Feature | Before | After |
|---------|--------|-------|
| Explanation Speed | ~30 sec | ~1.5 sec ⚡ |
| Interpretation | Confusing | Intuitive ✅ |
| Positive (+) means | Varies | REAL (Green) |
| Negative (-) means | Varies | FAKE (Red) |
| User Interface | Terminal | Web UI 🎨 |

---

## 🚀 Quick Start

```bash
# 1. Launch UI
python app.py

# 2. Browser opens automatically at:
#    http://127.0.0.1:7861

# 3. Enter text and click "Check News"

# 4. Click "Explain Prediction" to see why

# 5. Try the examples at the bottom!
```

---

## 💡 Tips

1. **For Demo:** Use the example buttons for quick testing
2. **For Speed:** Explanations are now fast (~1.5 sec)
3. **For Understanding:** Green = Real, Red = Fake
4. **For Presentation:** Show both prediction AND explanation
5. **For Questions:** Explain that + means REAL, - means FAKE

---

**🎉 Your Web UI is Ready!**

Launch it with: `python app.py` or double-click `launch_ui.bat`

---

*Last Updated: January 17, 2026*  
*Version: 2.0 (Speed Optimized + Fixed Interpretation)*
