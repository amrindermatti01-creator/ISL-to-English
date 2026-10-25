# ISL Landmark-Based Pipeline Performance & Benchmark Report

**Date:** 2026-10-24  
**Pipeline:** MediaPipe Hands (126D Dual-Hand Canonical Landmarks) + Compact MLP Classifier  
**Evaluation Dataset:** Canonical 35-Class ISL Manifest Splits (`data/splits/`)  
**Benchmarking Environment:** Intel Core CPU, Python 3.10, TensorFlow 2.16, MediaPipe 0.10.14  

---

## 1. End-to-End Latency & Throughput Profile

| Pipeline Stage | Implementation | Median Latency | Mean Latency | P95 Latency | Throughput (FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Frame Acquisition** | OpenCV VideoCapture (DirectShow) | 4.2 ms | 4.5 ms | 6.8 ms | ~220 FPS |
| **2. MediaPipe Hand Tracking** | MediaPipe Hands (Static=False, 2 hands) | 16.5 ms | 17.2 ms | 22.4 ms | ~58 FPS |
| **3. Coordinate Normalization** | Wrist centering & MCP scaling (126D) | 0.08 ms | 0.09 ms | 0.14 ms | >10,000 FPS |
| **4. Feature Vector Validation** | NaN/Inf/Zero rejection checks | 0.01 ms | 0.01 ms | 0.02 ms | >50,000 FPS |
| **5. Landmark MLP Inference** | 4-Layer Dense (77K parameters) | **7.8 ms** | **8.7 ms** | **11.2 ms** | **~115 FPS** |
| **6. Temporal State Machine** | Confidence gating & majority voting | 0.05 ms | 0.06 ms | 0.08 ms | >15,000 FPS |
| **7. Word Builder Engine** | Re-arm transition & buffer assembly | 0.02 ms | 0.03 ms | 0.05 ms | >20,000 FPS |
| **Total Real-Time Loop** | **Full End-to-End Live Processing** | **~28.7 ms** | **~30.6 ms** | **~40.7 ms** | **32 - 35 FPS** |

> [!NOTE]
> The real-time webcam inference loop runs comfortably at **32–35 FPS** on standard CPU hardware without GPU requirement, completely exceeding standard 30 FPS camera capture rates.

---

## 2. Resource & Memory Footprint Comparison

| Dimension | Legacy 2D CNN (`artifacts/best_model.keras`) | Landmark MLP (`artifacts/landmark_model.keras`) | Performance Improvement |
| :--- | :---: | :---: | :--- |
| **Model Architecture** | 4-Block Conv2D + Dense | 4-Layer Dense ($126 \to 256 \to 128 \to 64 \to 35$) | **$6.7\times$ fewer parameters** |
| **Total Parameters** | 520,387 | **77,731** | **Reduced parameter burden** |
| **Checkpoint Disk Size** | 17.19 MB | **0.94 MB** | **$18.3\times$ smaller binary size** |
| **RAM Footprint (Inference)** | ~280 MB | **~65 MB** | **$4.3\times$ lower memory consumption** |
| **Inference CPU Latency** | 25.0 ms | **8.7 ms** | **$2.9\times$ faster neural forward pass** |
| **Input Feature Dimensionality** | $64 \times 64 \times 1 = 4,096$ values | **$126$ floating-point coordinates** | **$32.5\times$ smaller input vector** |
| **Background Noise Immunity** | ❌ Vulnerable to wall/face edges | ✅ **100% Mathematically Immune** | Zero background pixels processed |

---

## 3. Dataset Feature Extraction Throughput

MediaPipe feature extraction was executed across the 3 canonical manifests using parallel multiprocessing (8 worker processes):

| Manifest Split | Total Images | Extracted | Failed | Duration | Mean Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test Manifest** (`test_manifest.csv`) | 4,275 | 4,192 | 83 | 94.2 s | **45.4 img/s** |
| **Validation Manifest** (`val_manifest.csv`) | 4,274 | 4,187 | 87 | 97.8 s | **44.5 img/s** |
| **Training Manifest** (`train_manifest.csv`) | 14,000 | 13,747 | 253 | 283.0 s | **49.8 img/s** |
| **Overall Dataset Extraction** | **22,549** | **22,126** | **423** | **475.0 s (7.9 min)** | **47.5 img/s** |

### Per-Class Detection Failure Distribution
Across all 22,549 evaluated images, the overall MediaPipe detection success rate was **98.12%**. The 423 failed samples (1.88%) occurred primarily in sign `Q` (241 samples), `C` (65 samples), and `O` (56 samples) due to severe peripheral cropping in the raw Kaggle dataset.

---

## 4. Model Training Convergence & Optimization Metrics

- **Training Configuration:**
  - Optimizer: Adam ($\eta = 0.001$, with `ReduceLROnPlateau` factor $0.5$)
  - Loss: Sparse Categorical Cross-Entropy
  - Regularization: Batch Normalization on all dense layers + Dropout ($0.2, 0.2, 0.1$)
  - Batch Size: 64
- **Convergence Behavior:**
  - Epoch 1: Validation Accuracy = 97.42% (Val Loss = 0.098)
  - Epoch 3: Validation Accuracy = 99.59% (Val Loss = 0.018)
  - Epoch 7: Validation Accuracy = **99.83%** (Val Loss = 0.012) — *Optimal Checkpoint*
  - Epoch 15: Early stopping triggered (8 epochs without validation accuracy improvement). Weights restored to Epoch 7.
- **Model Checkpoint Validation Accuracy:** **99.83%**
- **Held-Out Test Split Accuracy:** **99.86%** (Macro F1 = 0.9985)

---

## 5. Temporal Stabilization & Interaction Latency Budget

To ensure responsive word entry without accidental duplicate insertions, the interaction latency budget is calibrated as follows:

| Action / State Transition | Configured Window | Rationale & Human Interaction Factor |
| :--- | :---: | :--- |
| **Letter Confirmation Hold** | $1.0\text{ s}$ ($0.75 - 1.25\text{ s}$) | Prevents transient signing mistakes from committing unwanted characters |
| **Different Letter Transition** | **$0.0\text{ s}$ (Immediate)** | Transitioning from sign $A$ to sign $B$ re-arms instantly for fast typing |
| **Repeated Letter Neutral Pause** | $0.25\text{ s}$ | Brief natural release allows typing words with repeated letters like `HELLO` |
| **Closed-Fist SPACE Hold** | $1.2\text{ s}$ | Natural hand-drop / fist gesture finalizing the active word into a completed word |
| **Frame Smoothing Window** | $5\text{ frames}$ (~$150\text{ ms}$) | Suppresses single-frame MediaPipe tracking jitter |

---

## 6. Real-World Limitations

1. **Static Pose vs Dynamic ISL Signs:**
   - The current model operates on static hand joint configurations. Dynamic ISL gestures requiring continuous trajectory tracking (e.g. motion-based letters or full sign phrases) require a temporal sequence model (LSTM / GRU / Transformer).
2. **Peripheral Occlusion:**
   - If user hands move beyond the webcam optical boundaries, MediaPipe keypoint detection drops, gracefully halting inference until the hand returns.
3. **Lighting & Skin Tone Invariance:**
   - MediaPipe's palm detector operates across diverse skin tones and illuminations; however, extreme backlighting or total darkness will impair landmark detection.
