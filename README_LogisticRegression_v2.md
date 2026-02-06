# NLP Fake News Detection - Logistic Regression Experiments

## Project Overview

**Objective:** Build a classifier to distinguish between real and fake news headlines.

**Dataset:**
- Training data: 34,152 headlines (51.5% fake, 48.5% real)
- After deduplication: 32,197 headlines (1,955 duplicates removed, 5.72%)
- Test data: 9,984 headlines (labels predicted)

---

## Methodology

### Model
- **Algorithm:** Logistic Regression
- **Vectorization:** TF-IDF
- **Train/Test Split:** 80/20 (random_state=42)
- **Class Balancing:** `class_weight='balanced'`

### Data Quality: Deduplication & Leakage Check

Before running experiments, we identified and removed duplicate headlines from the training data. Deduplication was performed on **preprocessed text** rather than raw text, because encoding differences (smart quotes, tab characters, extra whitespace) caused near-identical headlines to survive raw deduplication.

**Examples of near-duplicates caught:**
- Tab vs space: `"afghan capital⟶kabul"` vs `"afghan capital kabul"`
- Colon vs dash: `"next week: statement"` vs `"next week - statement"`
- Encoding characters: `"‚racist"` vs `"racist"`

**Leakage verification** was built into every experiment run, checking for both index overlap and identical text content between train and test sets. After deduplication, 7 of 9 configurations showed zero leakage. The 2 remaining overlaps (configs with stopword removal) were caused by preprocessing itself — titles like "once again" and "so" become empty strings after stopword removal, creating artificial matches. These affected only 2 out of 6,440 test samples and had negligible impact.

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

### Accuracy Comparison (after deduplication, sorted by performance)

| Rank | Configuration | Accuracy | Features | Leakage |
|------|---------------|----------|----------|---------|
| 🥇 | **Unigrams only (1,1)** | **93.93%** | 11,274 | ✅ None |
| 🥈 | Baseline (1,2) | 93.85% | 41,583 | ✅ None |
| 🥉 | Bigrams (1,2) | 93.85% | 41,583 | ✅ None |
| 4 | Trigrams (1,3) | 93.82% | 54,251 | ✅ None |
| 5 | Keep punctuation | 93.74% | 41,837 | ✅ None |
| 6 | With stemming | 93.07% | 28,568 | ⚠️ 2 |
| 7 | Manual stopwords | 93.04% | 29,551 | ⚠️ 2 |
| 8 | TF-IDF stopwords | 92.62% | 27,934 | ✅ None |
| 9 | With lemmatization | 92.56% | 28,638 | ⚠️ 2 |

### Best Configuration

```
Model:          Logistic Regression
Preprocessing:  Encoding fix + Punctuation removal
Stopwords:      NOT removed
Stemming/Lemma: NOT applied
N-grams:        Unigrams only (1,1)
Accuracy:       93.93%
```

### Classification Report (Best Model)

```
              precision    recall  f1-score   support

    Fake (0)     0.95      0.93      0.94      3259
    Real (1)     0.93      0.95      0.94      3181

    accuracy                         0.94      6440
```

---

## Impact of Deduplication

Removing 1,955 duplicate headlines (5.72% of data) revealed that previous accuracy scores were inflated by ~0.7–1.0%. Duplicate headlines appearing in both train and test sets allowed the model to "memorize" answers rather than generalize.

### Before vs After Comparison

| # | Configuration | Before | After | Change |
|---|---------------|--------|-------|--------|
| 7 | **Unigrams** | **94.63%** | **93.93%** | **-0.70%** |
| 1 | Baseline | 94.60% | 93.85% | -0.75% |
| 8 | Bigrams | 94.60% | 93.85% | -0.75% |
| 9 | Trigrams | 94.61% | 93.82% | -0.79% |
| 4 | Keep punctuation | 94.60% | 93.74% | -0.86% |
| 5 | Stemming | 93.63% | 93.07% | -0.56% |
| 2 | Manual stopwords | 93.44% | 93.04% | -0.40% |
| 3 | TF-IDF stopwords | 93.59% | 92.62% | -0.97% |
| 6 | Lemmatization | 93.40% | 92.56% | -0.84% |

**Key finding:** While absolute accuracy dropped across all configurations, the relative ranking remained nearly identical. This confirms that our preprocessing conclusions were not artifacts of data leakage — they reflect genuine model behavior.

---

## Key Insights

### 1. Simpler preprocessing wins
- Unigrams alone outperformed bigrams and trigrams
- Fewer features (11K vs 54K) with better accuracy
- This held true both before and after deduplication

### 2. DO NOT remove stopwords
- Removing stopwords decreased accuracy by ~0.8–1.3%
- Common words like "the", "is", "just" are predictive for fake news
- Stopword removal also introduces minor leakage by collapsing distinct titles into identical processed strings

### 3. DO NOT use stemming/lemmatization
- Both techniques hurt performance
- Lemmatization had the worst impact (-1.37% vs best model)

### 4. Punctuation has minimal impact
- Keeping vs removing punctuation produced nearly identical results (93.74% vs 93.85%)

### 5. Data quality matters
- 5.72% of the dataset were duplicates (exact or near-duplicate after preprocessing)
- Removing them reduced accuracy by ~0.7% but gave more honest performance estimates
- Always deduplicate on processed text, not raw text, to catch encoding variants

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
| `NLP_G4_LogisticRegression_Comparison.ipynb` | Full experiment notebook with deduplication and leakage checks |
| `testing_data_predictions_LR.csv` | Predictions on test data (9,984 rows) |

### Test Data Predictions Distribution
- Fake (0): 4,586 (45.9%)
- Real (1): 5,398 (54.1%)

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

Based on deduplicated validation results with verified zero data leakage, we estimate the model will achieve approximately **93–94% accuracy** on unseen test data.

---

*Note: This README covers Logistic Regression experiments only. BERT experiments are being conducted separately and will be combined in the final report.*
