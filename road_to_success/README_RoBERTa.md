# NLP Fake News Detection - RoBERTa Experiments

## Project Overview

**Objective:** Build a classifier to distinguish between real and fake news headlines using a fine-tuned RoBERTa transformer model.

**Dataset:**
- Training data: 34,152 headlines (51.5% fake, 48.5% real)
- After deduplication: 32,206 headlines (1,946 title-level duplicates removed, 5.70%)
- Test data: 9,984 headlines (labels predicted)

---

## Methodology

### Model
- **Architecture:** RoBERTa (`roberta-base`)
- **Task:** Binary sequence classification (num_labels=2)
- **Tokenization:** RoBERTa BPE tokenizer, max_length=64, padding to max_length
- **Train/Test Split:** 80/20 (random_state=42, stratified)
- **Environment:** Google Colab with GPU

RoBERTa (Robustly Optimized BERT Approach) improves on BERT by using dynamic masking, removing next-sentence prediction, training with larger batches, and using more training data. It uses Byte-Pair Encoding (BPE) rather than WordPiece tokenization and has a different set of special tokens, all handled automatically by the tokenizer.

### Preprocessing

Minimal manual preprocessing was applied, consistent with transformer best practices. The tokenizer handles subword segmentation, special tokens, and attention masks internally. Manual steps were limited to:

- Dropping rows with missing title or label values
- Casting label to integer, title to string
- Removing duplicate rows by title (after checking for label conflicts -- none were found)

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

### Data Quality

**Deduplication:** Performed on the `title` column using `drop_duplicates(subset=["title"])`, removing 1,946 duplicates. Before dropping, label conflicts were checked -- no title mapped to both labels, so deduplication was safe. This approach is more targeted than the DistilBERT notebook (which used full-row dedup) but still operates on raw text rather than preprocessed text.

**Leakage check:** An explicit check for overlapping titles between train and test sets was performed both before and after training:
```
len(set(X_train["title"]).intersection(set(X_test["title"]))) = 0
```
This confirms zero raw-text leakage between splits. However, this check does not catch near-duplicates that differ only in encoding or whitespace (the same limitation identified in the Logistic Regression analysis, where 9 additional near-duplicates were found using preprocessed-text deduplication).

---

## Results

### Training Progress

| Epoch | Training Loss | Validation Loss | Accuracy | F1 |
|-------|--------------|-----------------|----------|------|
| 1 | 0.0632 | 0.0881 | 97.95% | 0.9795 |
| 2 | 0.0792 | 0.0720 | 98.39% | 0.9839 |
| 3 | 0.0205 | 0.0777 | 98.60% | 0.9860 |

Best model selected: Epoch 3 (highest F1). Total training time: approximately 16.5 minutes on Colab GPU.

Note: As with the DistilBERT experiment, validation loss increased from epoch 2 to 3 (0.0720 to 0.0777) while accuracy improved by only 0.21%. This suggests early overfitting. Epoch 2 may represent a more robust checkpoint for generalization.

### Classification Report (Validation Set)

```
              precision    recall  f1-score   support

    Fake (0)     0.9857    0.9863    0.9860      3205
    Real (1)     0.9864    0.9858    0.9861      3237

    accuracy                         0.9860      6442
   macro avg     0.9860    0.9860    0.9860      6442
weighted avg     0.9860    0.9860    0.9860      6442
```

### Confusion Matrix

```
              Predicted 0    Predicted 1
Actual 0         3161            44
Actual 1           46          3191
```

Total errors: 90 out of 6,442 (44 false positives, 46 false negatives). This is a reduction of 24 errors compared to DistilBERT (114 errors).

### ROC-AUC

An ROC curve was generated from the model's logits. The AUC value is visible in the notebook's plot output but was not printed numerically. Based on the near-perfect classification metrics, AUC is expected to be above 0.99.

---

## Known Issues

### 1. Test set used as validation set

Same issue as the DistilBERT experiment: `eval_dataset=X_test_ds` combined with `load_best_model_at_end=True` means the test set influenced epoch selection. The reported 98.60% accuracy was indirectly optimized against the evaluation set. A proper three-way split (train/validation/test) would give a more conservative estimate.

### 2. Raw-text deduplication

Title-level dedup on raw text catches exact matches but misses the 9 near-duplicates identified through preprocessed-text deduplication in the Logistic Regression analysis. Impact is negligible on a 32K dataset.

### 3. Overfitting signal at epoch 3

Validation loss increased between epochs 2 and 3 while accuracy gained only 0.21%. The selected model may be slightly overfit. Epoch 2 could generalize better to unseen data.

### 4. Unused compute_metrics redefinition

Cell 38 redefines `compute_metrics` using `np.softmax` (which does not exist in NumPy -- it should be `scipy.special.softmax` or a manual implementation). This cell was likely experimental and was not used during training, as the Trainer was already instantiated with the original `compute_metrics` from cell 35. No impact on results, but the cell would error if run independently.

### 5. Extra columns in pipeline

EDA-derived columns (`char_count`, `word_count`, `has_tab`, `has_url`) are present in the DataFrame during tokenization. They are excluded by `set_format` and do not affect model training or predictions.

### 6. Output directory naming

The training output directory is named `./bert_headlines` rather than something RoBERTa-specific, carried over from the DistilBERT notebook. The final saved model is correctly placed in `./trained_roberta_model`.

---

## Improvements Over DistilBERT Notebook

| Aspect | DistilBERT | RoBERTa |
|--------|-----------|---------|
| Leakage check | Not performed | Explicit check (0 overlap) |
| Label conflict check | Not performed | Verified before dedup |
| Dedup method | Full-row `drop_duplicates()` | Title-column `drop_duplicates(subset=["title"])` |
| ROC-AUC | Not computed | Plotted from logits |
| Model export | Saved to Colab local | Saved + zipped for download |

---

## Output Files

| File | Description |
|------|-------------|
| `NLP_G4_RoBERTa.ipynb` | Full experiment notebook (Colab) |
| `validation_data_predictions_RoBERTa.csv` | Predictions on test data (9,984 rows) |
| `trained_roberta_model/` | Saved model weights and tokenizer |

---

## How to Run

1. Open `NLP_G4_RoBERTa.ipynb` in Google Colab
2. Enable GPU runtime: Runtime > Change runtime type > GPU
3. Upload `training_data_lowercase.csv` when prompted
4. Run all cells
5. For test predictions, upload `testing_data_lowercase_nolabels.csv` when prompted

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

Based on validation results, the model achieves approximately 98.6% accuracy. Due to the test-as-validation issue, real-world performance on truly unseen data may be slightly lower, likely in the range of 97.5-98.5%.

---

*Note: This README covers the RoBERTa experiment only. Logistic Regression and DistilBERT experiments are documented separately. A combined comparison will follow.*
