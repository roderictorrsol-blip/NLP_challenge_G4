# NLP Fake News Detection - Logistic Regression Experiments

## Project Overview

**Objective:** Build a classifier to distinguish between real and fake news headlines.

**Dataset:**
- Training data: 34,152 headlines (51.5% fake, 48.5% real)
- Test data: 9,984 headlines (labels predicted)

---

## Methodology

### Model
- **Algorithm:** Logistic Regression
- **Vectorization:** TF-IDF
- **Train/Test Split:** 80/20 (random_state=42)
- **Class Balancing:** `class_weight='balanced'`

### Preprocessing Experiments

We tested 9 different preprocessing configurations to find the optimal approach:

| # | Configuration | Encoding | Punctuation | Stopwords | Stem/Lemma | N-grams |
|---|---------------|----------|-------------|-----------|------------|---------|
| 1 | Baseline | ✓ | ✓ | ✗ | ✗ | (1,2) |
| 2 | Manual stopwords | ✓ | ✓ | ✓ manual | ✗ | (1,2) |
| 3 | TF-IDF stopwords | ✓ | ✓ | ✓ TF-IDF | ✗ | (1,2) |
| 4 | Keep punctuation | ✓ | ✗ | ✗ | ✗ | (1,2) |
| 5 | Stemming | ✓ | ✓ | ✓ | Stemming | (1,2) |
| 6 | Lemmatization | ✓ | ✓ | ✓ | Lemmatization | (1,2) |
| 7 | Unigrams only | ✓ | ✓ | ✗ | ✗ | (1,1) |
| 8 | Bigrams | ✓ | ✓ | ✗ | ✗ | (1,2) |
| 9 | Trigrams | ✓ | ✓ | ✗ | ✗ | (1,3) |

---

## Results

### Accuracy Comparison (sorted by performance)

| Rank | Configuration | Accuracy | Features |
|------|---------------|----------|----------|
| 🥇 | **Unigrams only (1,1)** | **94.63%** | 12,120 |
| 🥈 | Trigrams (1,3) | 94.61% | 76,476 |
| 🥉 | Baseline (1,2) | 94.60% | 50,624 |
| 4 | Keep punctuation | 94.60% | 50,624 |
| 5 | Bigrams (1,2) | 94.60% | 50,624 |
| 6 | With stemming | 93.63% | 35,553 |
| 7 | TF-IDF stopwords | 93.59% | 50,395 |
| 8 | Manual stopwords | 93.44% | 35,908 |
| 9 | With lemmatization | 93.40% | 36,553 |

### Best Configuration

```
Model:          Logistic Regression
Preprocessing:  Encoding fix + Punctuation removal
Stopwords:      NOT removed
Stemming/Lemma: NOT applied
N-grams:        Unigrams only (1,1)
Accuracy:       94.63%
```

### Classification Report (Best Model)

```
              precision    recall  f1-score   support

    Fake (0)     0.96      0.94      0.95      3529
    Real (1)     0.93      0.96      0.95      3302

    accuracy                         0.95      6831
```

---

## Key Insights

### 1. Simpler preprocessing wins
- Unigrams alone outperformed bigrams and trigrams
- Fewer features (12K vs 76K) with better accuracy

### 2. DO NOT remove stopwords
- Removing stopwords decreased accuracy by ~1%
- Common words like "the", "is", "just" are predictive for fake news

### 3. DO NOT use stemming/lemmatization
- Both techniques hurt performance
- Lemmatization had the worst impact (-1.23%)

### 4. Punctuation has minimal impact
- Keeping vs removing punctuation produced identical results

### Top Predictive Features

**Real News Indicators (positive coefficients):**
| Word | Coefficient |
|------|-------------|
| says | +9.10 |
| us | +8.07 |
| factbox | +6.37 |
| house | +4.30 |
| china | +3.60 |

**Fake News Indicators (negative coefficients):**
| Word | Coefficient |
|------|-------------|
| video | -15.26 |
| breaking | -8.11 |
| just | -7.19 |
| gop | -6.77 |
| hillary | -6.36 |

---

## Output Files

| File | Description |
|------|-------------|
| `NLP_G4_LogisticRegression_Comparison.ipynb` | Full experiment notebook with results |
| `testing_data_predictions_LR.csv` | Predictions on test data (9,984 rows) |

### Test Data Predictions Distribution
- Fake (0): 4,610 (46.2%)
- Real (1): 5,374 (53.8%)

---

## How to Run

1. Place `training_data_lowercase.csv` in the same folder as the notebook
2. Run all cells in `NLP_G4_LogisticRegression_Comparison.ipynb`
3. For predictions, place `testing_data_lowercase_nolabels.csv` and run Section 9

### Requirements
```
pandas
numpy
scikit-learn
nltk
matplotlib
```

---

## Estimated Performance

Based on validation results, we estimate the model will achieve approximately **94-95% accuracy** on unseen test data.

---

*Note: This README covers Logistic Regression experiments only. BERT experiments are being conducted separately and will be combined in the final report.*
