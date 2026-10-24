# Accuracy Claim Audit Report: Indian Sign Language (ISL) Classification

**Audit Target:** Claimed 99.50% Accuracy on ISL Model  
**Model Checkpoint:** [`artifacts/best_model.keras`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/artifacts/best_model.keras)  
**Evaluator Script:** [`src/evaluate.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/evaluate.py)  
**Dataset:** 42,745 images across 35 classes (`1`–`9`, `A`–`Z`)  
**Audit Date:** 2026-10-24  
**Final Audit Verdict:** **VERIFIED (With Explicit Scope Clarification)**

---

## 1. Metric Origin Analysis: What the 99.50% Actually Represented

### Exact Log Line and Origin
During the execution of [`src/train.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/train.py) (background task `task-313`), the following stdout was captured in `task-313.log` at step 392 of Epoch 2:

```text
392/535 ==================== 1:09 486ms/step - accuracy: 0.9950 - loss: 0.0166
```

### Forensic Finding
- **Metric Type:** Running cumulative batch training accuracy on the **training set** (`X_train`) during Epoch 2.
- **What it was NOT:** It was **not** test accuracy, and it was **not** the completed validation accuracy of Epoch 2.
- **Cause of Mischaracterization:** In the earlier assistant update, this live training progress figure (`accuracy: 0.9950`) was reported as *"Accuracy: 99.50% (loss: 0.0166)!"* and loosely phrased as *"hit 99.5% accuracy on the 42,745 images"*.
- **Audit Verdict on Earlier Claim:** **FAILED AS PHRASED**. A running mid-epoch training batch metric must never be presented as test or dataset-wide accuracy.

---

## 2. Checkpoint Selection & Validation Independence

### Verification of Checkpoint Mechanism
Inspection of [`src/train.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/train.py) (lines 122–136):
```python
checkpoint_cb = ModelCheckpoint(
    filepath=str(best_model_path),
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)
early_stopping_cb = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)
```

### Training Progression Record ([`reports/training_history.csv`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/reports/training_history.csv))

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss | Learning Rate | Checkpoint Action |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | 96.53% | 0.1540 | 11.18% | 4.0305 | 0.0010 | Saved (Initial) |
| **2** | 99.65% | 0.0125 | 99.46% | 0.0122 | 0.0010 | Saved (`val_loss` improved) |
| **3** | 99.89% | 0.0050 | 99.95% | 0.0011 | 0.0010 | Saved (`val_loss` improved) |
| **4** | 99.98% | 0.0013 | **100.00%** | **0.00018** | 0.0010 | **Saved (Best Checkpoint)** |
| **5** | 99.93% | 0.0030 | 99.98% | 0.00033 | 0.0010 | Did not improve; restored Epoch 4 |

**Verification Status:** **VERIFIED**.  
Checkpoint [`artifacts/best_model.keras`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/artifacts/best_model.keras) was strictly selected based on minimum validation loss (`val_loss = 0.00017897`, Epoch 4), without any exposure to test data.

---

## 3. Independent Evaluation on Complete Held-Out Test Set

The audit script [`src/evaluate.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/evaluate.py) loaded [`artifacts/best_model.keras`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/artifacts/best_model.keras) and evaluated it strictly **once** on the untouched held-out test manifest [`data/splits/test_manifest.csv`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/data/splits/test_manifest.csv).

### Execution Command
```bash
.venv\Scripts\python.exe src/evaluate.py
```

### Exact Metric Values ([`reports/test_metrics.json`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/reports/test_metrics.json))

| Metric | Measured Value | Standard / Target | Status |
|---|---|---|:---:|
| **Held-Out Test Samples** | **4,275** | Exactly 10% stratified hold-out | VERIFIED |
| **Correct Predictions** | **4,275 / 4,275** | Integer match count | VERIFIED |
| **Incorrect Predictions** | **0** | — | VERIFIED |
| **Test Accuracy** | **100.0000%** | $\ge 99.0\%$ | VERIFIED |
| **Test Loss (Cross-Entropy)** | **0.000171** | — | VERIFIED |
| **Macro Precision** | **1.000000** | — | VERIFIED |
| **Macro Recall** | **1.000000** | — | VERIFIED |
| **Macro F1-Score** | **1.000000** | — | VERIFIED |
| **Number of Classes Evaluated** | **35** | Digits 1–9, Letters A–Z | VERIFIED |

### Classification Report Summary ([`reports/classification_report.txt`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/reports/classification_report.txt))
- All 35 classes achieved `precision = 1.0000`, `recall = 1.0000`, and `f1-score = 1.0000`.
- Per-class support matches exact stratified proportions (120 images for 31 classes; 145 for `C`, 138 for `I`, 143 for `O`, 129 for `V`).

### Confusion Matrix
Confusion matrix plot saved to [`reports/confusion_matrix.png`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/reports/confusion_matrix.png). All 4,275 test samples lie strictly along the true positive diagonal with **zero off-diagonal confusion**.

---

## 4. Rigorous Data Leakage & Split Audit

### A. File Path Overlap
- $\text{Train} \cap \text{Val} = \emptyset$ (0 overlapping paths)
- $\text{Train} \cap \text{Test} = \emptyset$ (0 overlapping paths)
- $\text{Val} \cap \text{Test} = \emptyset$ (0 overlapping paths)

### B. Duplicate Images Check
- In Stage 1 audit, MD5 hashing of all 42,745 files identified exactly **1 duplicate file pair** (`redundant_duplicate_images: 1`):
  - `Indian/I/177.jpg` and `Indian/I/177 copy.jpg`.
- Verification of manifests confirms that both files were placed inside [`data/splits/train_manifest.csv`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/data/splits/train_manifest.csv) (line 1469).
- **Exact duplicate leakage into Test set:** **0 files**.

### C. Preprocessing & Label Consistency
- Preprocessing used during evaluation: `preprocess_image(target_size=(64, 64), threshold_value=90, use_canny=True, canny_low=50, canny_high=150, normalize=True)`.
- Label mapping used: [`artifacts/class_mapping.json`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/artifacts/class_mapping.json) (integer indices 0 to 34).
- Matches training preprocessing and class mapping 100%.

### D. Test Set Hold-Out Integrity
- In `src/train.py`, only `train_manifest.csv` and `val_manifest.csv` were read.
- `data/processed/test_64x64.npz` was never generated during training.
- **Verification:** **VERIFIED**. The test set was untouched during training and model selection.

---

## 5. Critical Engineering Limitations & Caveats

Despite the 100.0% test accuracy on this held-out split, the following critical limitations must be explicitly noted:

1. **Signer Independence is NOT VERIFIED**:
   - The Kaggle dataset (`prathumarikeri/indian-sign-language-isl`) names files purely as sequential integers (e.g. `1.jpg` ... `1200.jpg`).
   - There is no metadata identifying which signer performed which sign or what session/lighting setup was used.
   - Consequently, our split is **image-stratified**, not **signer-independent**. Frames from the same recorded video/session of a signer likely exist across both the training and test splits.
   - **Verdict on Unseen Signer Robustness:** **NOT VERIFIED**. While the model generalizes perfectly to held-out frames of this dataset, real-world webcam inference with unseen human users in different lighting will experience lower real-world accuracy without MediaPipe hand bounding box alignment.

2. **Isolated Gesture vs Continuous Sentence ISL**:
   - This model is an **isolated static sign classifier** (35 alphabet/digit classes).
   - It cannot and must not be described as full Indian Sign Language sentence translation.

---

## 6. Final Audit Verdict

| Question | Assessment | Verdict |
|---|---|:---:|
| **Did the reported 99.50% represent test accuracy?** | No, it was a mid-epoch training batch metric (`task-313.log:1530`). | **FAILED AS PHRASED** |
| **Was the model checkpoint selected honestly via validation?** | Yes, `val_loss = 0.00018` at Epoch 4 using `val_manifest.csv`. | **VERIFIED** |
| **Was the test set held out with zero leakage?** | Yes, 4,275 images untouched during training with zero path/duplicate overlap. | **VERIFIED** |
| **What is the actual held-out test performance?** | **4,275 / 4,275 correct (100.0% accuracy, Loss: 0.000171, Macro F1: 1.0000)**. | **VERIFIED** |
| **Is signer-independent generalization proven?** | No, dataset lacks signer/session metadata. | **NOT VERIFIED** |
| **OVERALL AUDIT VERDICT** | The model's held-out classification capability on this benchmark is genuine, but claims must clearly specify **image-stratified test split** rather than signer-independent translation. | **VERIFIED WITH DOCUMENTED LIMITATIONS** |
