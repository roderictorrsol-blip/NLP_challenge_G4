# NLP Fake News Detection - DistilBERT Experiments

## Project Overview

**Objective:** Build a classifier to distinguish between real and fake news headlines using a fine-tuned transformer model.

**Dataset:**
- Training data: 34,152 headlines (51.5% fake, 48.5% real)
- After deduplication: 32,206 headlines (1,946 exact duplicates removed, 5.70%)
- Test data: 9,984 headlines (labels predicted)

---

## Methodology

### Model
- **Architecture:** DistilBERT (`distilbert-base-uncased`)
- **Task:** Binary sequence classification (num_labels=2)
- **Tokenization:** BERT WordPiece tokenizer, max_length=64, padding to max_length
- **Train/Test Split:** 80/20 (random_state=42, stratified)
- **Environment:** Google Colab with GPU

DistilBERT is a distilled version of BERT-base that retains approximately 97% of BERT's language understanding while being 60% faster and 40% smaller. It was chosen for its balance of performance and training speed within the constraints of a Colab GPU session.

### Preprocessing

Unlike the Logistic Regression experiments which tested 9 preprocessing configurations, the BERT pipeline uses minimal preprocessing. BERT's tokenizer handles subword segmentation, casing (uncased model lowercases automatically), and special tokens internally. The only manual preprocessing steps were:

- Replacing tab characters with spaces
- Fixing encoding issues with apostrophes (smart quotes to standard quotes)
- Removing exact duplicate rows

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-5 |
| Train batch size | 16 |
| Eval batch size | 32 |
| Epochs | 3 |
| Weight decay | 0.01 |
| Best model selection | F1 score |
| Optimizer | AdamW (default) |
| Scheduler | Linear warmup (default) |

### Data Quality: Deduplication

Deduplication was performed using `df.drop_duplicates()` on raw text, removing 1,946 exact duplicate rows. This is less thorough than the preprocessed-text deduplication used in the Logistic Regression notebook (which caught 1,955 duplicates including near-duplicates differing only in encoding or whitespace). See Known Issues below for implications.

---

## Results

### Training Progress

| Epoch | Training Loss | Validation Loss | Accuracy | F1 |
|-------|--------------|-----------------|----------|------|
| 1 | 0.0648 | 0.0978 | 97.24% | 0.9723 |
| 2 | 0.0659 | 0.0795 | 98.14% | 0.9814 |
| 3 | 0.0044 | 0.0946 | 98.23% | 0.9823 |

Best model selected: Epoch 3 (highest F1). Total training time: approximately 9 minutes on Colab GPU.

Note: Validation loss increased from epoch 2 to 3 (0.0795 to 0.0946) while accuracy only marginally improved (+0.09%). This pattern indicates early overfitting -- the model is becoming more confident in its predictions but less well-calibrated. Epoch 2 may represent a more robust checkpoint.

### Classification Report (Validation Set)

```
              precision    recall  f1-score   support

    Fake (0)     0.9840    0.9803    0.9822      3205
    Real (1)     0.9806    0.9842    0.9824      3237

    accuracy                         0.9823      6442
   macro avg     0.9823    0.9823    0.9823      6442
weighted avg     0.9823    0.9823    0.9823      6442
```

### Confusion Matrix

```
              Predicted 0    Predicted 1
Actual 0         3142            63
Actual 1           51          3186
```

Total errors: 114 out of 6,442 (63 false positives, 51 false negatives).

---

## Known Issues

### 1. Test set used as validation set

The Trainer's `eval_dataset` is set to `X_test_ds`, and `load_best_model_at_end=True` selects the epoch with the best F1 on this same set. This means the reported test accuracy (98.23%) was indirectly optimized -- the test set influenced which model checkpoint was kept. A more rigorous approach would use a separate validation split for epoch selection and reserve the test set for final evaluation only.

### 2. Raw-text deduplication

The notebook removes duplicates on raw text (`df.drop_duplicates()`), catching 1,946 exact matches. However, the Logistic Regression analysis showed that 9 additional near-duplicates exist that only become identical after preprocessing (encoding normalization, whitespace collapsing). These near-duplicates could appear in both train and test sets, though the impact on a 32K-row dataset is negligible.

### 3. No explicit leakage verification

Unlike the updated Logistic Regression notebook, no programmatic check confirms zero text overlap between train and test sets after the split.

### 4. Overfitting signal at epoch 3

Validation loss increased between epochs 2 and 3 while accuracy barely improved. The selected model (epoch 3) may be slightly overfit. For production use, epoch 2 might generalize better.

### 5. Extra columns in pipeline

EDA-derived columns (`char_count`, `word_count`, `has_tab`, `has_url`) are present in the DataFrame during tokenization. They are filtered out by `set_format` and do not affect model training or predictions, but they add unnecessary data to the pipeline.

---

## Output Files

| File | Description |
|------|-------------|
| `NLP_G4_BERT2_0.ipynb` | Full experiment notebook (Colab) |
| `validation_data_predictions.csv` | Predictions on test data (9,984 rows) |
| `bert_headlines/` | Saved model and tokenizer (Colab local) |

---

## How to Run

1. Open `NLP_G4_BERT2_0.ipynb` in Google Colab
2. Enable GPU runtime: Runtime > Change runtime type > GPU
3. Upload `training_data_lowercase.csv` to `/content/`
4. Run all cells
5. For test predictions, upload `testing_data_lowercase_nolabels.csv` to `/content/`

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

## Estimated Performance

Based on validation results, the model achieves approximately 98% accuracy on the validation split. Due to the test-as-validation issue described above, real-world performance on truly unseen data may be slightly lower, likely in the range of 97-98%.

---

*Note: This README covers the DistilBERT experiment only. Logistic Regression experiments are documented separately. A combined comparison will follow.*
