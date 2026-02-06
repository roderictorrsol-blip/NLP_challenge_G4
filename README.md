# NLP Challenge - Fake News Detection | Group 4

## Project Overview

**Objective:** Classify news headlines as real or fake using Natural Language Processing techniques, progressing from traditional machine learning to transformer-based deep learning models.

**Team:** Group 4 (IronHack NLP Challenge, Week 4)

**Final Submission Model:** DistilBERT (fine-tuned, Epoch 2 checkpoint)

**Dataset:**
- Training data: 34,152 headlines (51.5% fake / 48.5% real)
- After deduplication: 32,206 unique headlines
- Test data: 9,984 headlines (unlabeled)

---

## Final Model Selection: DistilBERT

After evaluating three models across accuracy, cost, and training time, we selected **DistilBERT** (`distilbert-base-uncased`) as our final submission model. Specifically, we favor the **Epoch 2 checkpoint** (98.14% accuracy, 0.9814 F1) over Epoch 3, because:

- Epoch 2 has the lowest validation loss (0.0795), meaning the model is most well-calibrated at that point
- Epoch 3 shows rising validation loss (0.0946) despite a marginal accuracy gain (+0.09%), which is a classic overfitting signal
- Epoch 2 performance is more likely to reflect what we will see on truly unseen test data

RoBERTa was not selected despite achieving higher validation accuracy (98.60%) because it doubles the training time and cost for only 24 fewer misclassifications out of 6,442. The performance gain does not justify the resource increase.

---

## Model Comparison

### Accuracy and Performance

| Model | Architecture | Accuracy | F1 Score | Errors / 6,442 | Precision | Recall |
|-------|-------------|----------|----------|-----------------|-----------|--------|
| Logistic Regression | TF-IDF + LogReg | 93.93% | 0.9393 | ~391 | 0.9389 | 0.9393 |
| **DistilBERT (Epoch 2)** | **Transformer (66M params)** | **98.14%** | **0.9814** | **~120** | **0.9814** | **0.9814** |
| DistilBERT (Epoch 3) | Transformer (66M params) | 98.23% | 0.9823 | 114 | 0.9823 | 0.9823 |
| RoBERTa | Transformer (125M params) | 98.60% | 0.9860 | 90 | 0.9860 | 0.9860 |

### Cost and Training Time

All GPU-based models were trained on a Google Colab T4 GPU. Cost estimates use a mid-range T4 on-demand rate of approximately $0.40/hr.

| Model | Hardware | Training Time | Approx. Cost | Iterations Tested |
|-------|----------|---------------|-------------|-------------------|
| Logistic Regression | CPU only | ~5 seconds | $0.00 | 9 configurations |
| **DistilBERT** | **T4 GPU** | **~9 min** | **~$0.06** | **1 configuration** |
| RoBERTa | T4 GPU | ~16.5 min | ~$0.11 | 1 configuration |

### Cost-Effectiveness Analysis

| Model | Cost | Accuracy Gain vs Baseline | Errors Eliminated vs LR | Cost per Error Eliminated |
|-------|------|--------------------------|------------------------|--------------------------|
| Logistic Regression | $0.00 | baseline | baseline | -- |
| **DistilBERT** | **~$0.06** | **+4.21%** | **~271** | **~$0.0002** |
| RoBERTa | ~$0.11 | +4.67% | ~301 | ~$0.0004 |

Going from DistilBERT to RoBERTa costs roughly double ($0.06 to $0.11) for only 30 additional errors eliminated -- sharply diminishing returns.

### Inference Cost at Scale

Training is a one-time cost. For production deployment, inference speed matters more:

| Model | Parameters | Inference on 10K Headlines (T4) | Relative Speed |
|-------|-----------|--------------------------------|----------------|
| Logistic Regression | N/A (sparse matrix) | milliseconds (CPU) | fastest |
| DistilBERT | 66M | ~11 sec | 1x |
| RoBERTa | 125M | ~18-20 sec | ~1.7x slower |

DistilBERT is approximately 40% smaller and faster than RoBERTa at inference, which compounds significantly when classifying headlines at scale.

---

## Why Logistic Regression Is Still Worth Considering

Logistic Regression deserves recognition as an excellent baseline choice. At zero GPU cost, we were able to rapidly test 9 different preprocessing configurations and compare their performance -- something that would have been prohibitively time-consuming with transformer models. For teams or individuals without access to GPU resources or cloud budgets, Logistic Regression with TF-IDF achieves 93.93% accuracy, which is strong performance for a model that trains in seconds on a laptop CPU.

The LR experiments also proved valuable for understanding the data. Through systematic comparison of preprocessing pipelines, we discovered that minimal preprocessing (unigrams, no stopword removal, no stemming) consistently outperformed complex feature engineering -- a finding that informed our decision to use minimal preprocessing for the transformer models as well.

---

## Data Quality and Deduplication

Across all three models, we addressed data quality through deduplication:

| Model | Dedup Method | Duplicates Removed | Leakage Check |
|-------|-------------|-------------------|---------------|
| Logistic Regression | Preprocessed-text dedup | 1,955 (5.72%) | Yes -- 7 of 9 configs: 0 leaks |
| DistilBERT | Full-row `drop_duplicates()` | 1,946 (5.70%) | Not performed |
| RoBERTa | Title-column `drop_duplicates(subset=["title"])` | 1,946 (5.70%) | Yes -- 0 overlap confirmed |

The Logistic Regression analysis identified that preprocessed-text deduplication catches 9 additional near-duplicates (encoding/whitespace variants) missed by raw-text dedup. This is a minor gap that does not materially affect transformer results.

A shared limitation across all transformer experiments is that the test set was used as the validation set (`eval_dataset=X_test_ds` with `load_best_model_at_end=True`), meaning reported accuracies are slightly optimistic. A proper three-way split (train/validation/test) would provide more conservative estimates.

---

## Key Insights

1. **Simpler preprocessing wins.** Across both LR and transformer models, minimal text preprocessing consistently outperformed complex pipelines. BERT and RoBERTa tokenizers handle subword segmentation internally, making manual stemming, lemmatization, and stopword removal unnecessary and potentially harmful.

2. **Diminishing returns on model complexity.** The jump from LR to DistilBERT cuts errors by ~70% (391 to 120). The jump from DistilBERT to RoBERTa cuts errors by only ~25% more (120 to 90) at double the cost. The biggest gains come from adopting transformers at all, not from picking the largest one.

3. **Validation loss matters more than accuracy.** Both transformer models showed rising validation loss at Epoch 3 despite marginal accuracy improvements. Monitoring loss curves -- not just accuracy -- is critical for selecting the checkpoint that will generalize best.

4. **Data quality is a universal concern.** Deduplication improved result integrity across all models. The LR deduplication analysis revealed that our pre-dedup accuracy figures were inflated by approximately 0.7%, demonstrating that even small data issues compound into misleading metrics.

5. **LR enables rapid experimentation.** The ability to test 9 preprocessing configurations in minutes (vs hours for transformers) made LR indispensable for understanding the dataset before committing GPU resources to deeper models.

---

## Repository Structure

### Final Submission Files

| File | Description |
|------|-------------|
| `NLP_G4_final.ipynb` | **Final submission notebook** (copy of DistilBERT notebook) |
| `testing_data_lowercase_nolabels_final.csv` | **Final submission predictions** (copy of DistilBERT predictions) |

### Notebooks (Experiments)

| File | Model | Description |
|------|-------|-------------|
| `NLP_G4_LogisticRegression_Comparison.ipynb` | Logistic Regression | 9 preprocessing configs compared with deduplication |
| `NLP_G4_BERT2.0.ipynb` | DistilBERT | Fine-tuned transformer (selected model) |
| `NLP_G4_RoBERTa.ipynb` | RoBERTa | Fine-tuned transformer (evaluated, not selected) |

### Data Files

| File | Description |
|------|-------------|
| `training_data_lowercase.csv` | Training dataset (34,152 labeled headlines) |
| `testing_data_lowercase_nolabels.csv` | Test dataset (9,984 unlabeled headlines) |
| `testing_data_predictions_LR.csv` | Logistic Regression predictions on test data |
| `validation_data_predictions_BERT20.csv` | DistilBERT predictions on test data |
| `validation_data_predictions_RoBERTa.csv` | RoBERTa predictions on test data |

### Documentation

| File | Description |
|------|-------------|
| `README.md` | This file -- project overview and final comparison |
| `README_LogisticRegression.md` | Detailed LR experiment documentation |
| `README_LogisticRegression_v2.md` | Updated LR documentation with deduplication analysis |
| `README_BERT.md` | Detailed DistilBERT experiment documentation |
| `README_RoBERTa.md` | Detailed RoBERTa experiment documentation |

### Other

| File | Description |
|------|-------------|
| `Presentation template.pptx` | Slide deck for project presentation |

---

## How to Reproduce

### Logistic Regression (no GPU required)
1. Open `NLP_G4_LogisticRegression_Comparison.ipynb` in Jupyter or Colab
2. Place `training_data_lowercase.csv` in the working directory
3. Run all cells -- trains in seconds on CPU

### DistilBERT / RoBERTa (GPU required)
1. Open the respective notebook in Google Colab
2. Enable GPU runtime: Runtime > Change runtime type > GPU
3. Upload `training_data_lowercase.csv` when prompted (or place in `/content/`)
4. Run all cells
5. For test predictions, upload `testing_data_lowercase_nolabels.csv`

### Requirements
```
transformers
datasets
accelerate
evaluate
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
torch (provided by Colab)
```

---

## Technical Summary

| Aspect | Logistic Regression | DistilBERT | RoBERTa |
|--------|-------------------|------------|---------|
| Model type | Linear classifier | Transformer (distilled) | Transformer |
| Parameters | Sparse TF-IDF weights | 66M | 125M |
| Tokenization | TF-IDF vectorizer | WordPiece (max_length=64) | BPE (max_length=64) |
| Preprocessing | 9 configs tested | Minimal (tokenizer handles it) | Minimal (tokenizer handles it) |
| Train/test split | 80/20, stratified | 80/20, stratified | 80/20, stratified |
| Training epochs | N/A | 3 | 3 |
| Learning rate | N/A | 2e-5 | 2e-5 |
| Batch size | N/A | 16 train / 32 eval | 16 train / 32 eval |
| Best accuracy | 93.93% (unigrams) | 98.23% (Epoch 3) | 98.60% (Epoch 3) |
| Recommended checkpoint | N/A | Epoch 2 (98.14%) | Epoch 2 (98.39%) |

---

*Group 4 -- IronHack Data Science Bootcamp, Week 4 NLP Challenge*
