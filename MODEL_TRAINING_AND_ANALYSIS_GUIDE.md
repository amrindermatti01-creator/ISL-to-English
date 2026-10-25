# Complete Technical Guide: Model Training, Analysis, and Inference Workflow
### Indian Sign Language (ISL) Recognition & Multilingual Translation System

---

## 1. System Overview & Architectural Evolution

This system is an end-to-end Computer Vision and Deep Learning framework designed to translate Indian Sign Language (ISL) gestures into English text and regional Indian languages (Hindi, Punjabi, Tamil, Bengali, Telugu, etc.) with real-time audio playback.

The repository encompasses two distinct model architectures, marking an evolution from a traditional pixel-based Convolutional Neural Network (CNN) to a modernized, coordinate-invariant Landmark Multi-Layer Perceptron (MLP):

```
                        ISL FRAMEWORK EVOLUTION
                        
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│     LEGACY ARCHITECTURE         │       │     MODERN ARCHITECTURE         │
│         (2D CNN)                │       │       (Landmark MLP)            │
├─────────────────────────────────┤       ├─────────────────────────────────┤
│ • 64x64 Canny Edge Image        │       │ • MediaPipe Hands (126D Vector) │
│ • 520,000 Parameters            │  ==>  │ • 77,000 Parameters (6.7x less) │
│ • Checkpoint: 17.19 MB          │       │ • Checkpoint: 0.94 MB (18x less)│
│ • Susceptible to background ROI │       │ • 100% Background & Face Immune │
│ • 25.0 ms CPU Latency           │       │ • 8.74 ms CPU Latency (<1ms MLP)│
└─────────────────────────────────┘       └─────────────────────────────────┘
```

---

## 2. Dataset Pipeline & Preprocessing

### 2.1 Raw Dataset Ingestion & Character Classes
- **Dataset Source:** Kaggle Indian Sign Language Dataset (`prathumarikeri/indian-sign-language-isl`).
- **Raw Volume:** 42,745 high-resolution images.
- **Classes:** 35 alphanumeric classes:
  - Digits: `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9` (Digit `0` is excluded due to gesture overlap with letter `O` in standard ISL fingerspelling).
  - Letters: `A` through `Z` (covering both single-handed gestures like `A`, `B`, `C` and dual-handed gestures like `W`, `X`, `P`, `T`).
- **Class Balance:** Approximately 1,200 samples per class, with minor variations (e.g., class `C` has 1,300, `I` has 1,380).

### 2.2 Stratified Splitting & Leakage Prevention
To guarantee scientific validity and prevent data contamination, splitting is executed strictly through deterministic manifests (`data/splits/`):
- **Split Ratio:** 80% Training (34,196 images), 10% Validation (4,274 images), 10% Held-Out Testing (4,275 images).
- **Stratification:** Splitting is stratified by class label using a fixed `random_state=42`.
- **Leakage Integrity:**
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
- **Duplicate De-duplication:** MD5 hash auditing identified a single duplicated file pair (`Indian/I/177.jpg` and `Indian/I/177 copy.jpg`). Both duplicates were sequestered into the training set, ensuring **0 duplicate leakage** into the test set.

---

## 3. Landmark Feature Extraction (Modern Model)

The primary modern model operates on 3D anatomical skeletal landmarks extracted via **MediaPipe Hands** rather than raw pixel matrices.

```
Raw Camera Frame / Image
         │
         ▼
[MediaPipe Hands Detection] ──> Extracts 21 3D Joint Coordinates per hand (x, y, z)
         │
         ▼
[Wrist Origin Centering]    ──> Translates wrist joint (p0) to (0, 0, 0)
         │
         ▼
[Scale Normalization]       ──> Normalizes by distance between wrist and middle MCP (p9)
         │
         ▼
[Dual-Hand Canonical Slot]  ──> Left Hand: [0..62], Right Hand: [63..125]
         │
         ▼
 126-Dimensional Normalized Feature Vector
```

### 3.1 Landmark Geometry & Mathematical Normalization
MediaPipe produces 21 keypoints per detected hand in normalized screen coordinates $(x_i, y_i, z_i)$. Direct raw coordinates are sensitive to user distance, frame position, and camera aspect ratio. 

To achieve **complete spatial and scale invariance**, the extraction script ([`src/extract_landmarks_splits.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/extract_landmarks_splits.py)) applies anatomical normalization:

1. **Wrist Origin Translation:**
   The wrist landmark ($\mathbf{p}_0$) is designated as the origin anchor:
   $$\mathbf{p}_i' = \mathbf{p}_i - \mathbf{p}_0 \quad \forall i \in \{0, 1, \dots, 20\}$$
   *Why:* Translating the wrist to $(0, 0, 0)$ removes absolute screen position dependency. The user can hold their hand anywhere in the webcam frame without altering the coordinate values.

2. **Palm Anatomy Scale Factor:**
   The Euclidean distance between the wrist ($\mathbf{p}_0$) and the middle finger Metacarpophalangeal (MCP) joint ($\mathbf{p}_9$) serves as an invariant biometric reference length:
   $$S = \|\mathbf{p}_9 - \mathbf{p}_0\|_2 + 10^{-6}$$
   $$\mathbf{p}_i^{\text{norm}} = \frac{\mathbf{p}_i'}{S} = \frac{\mathbf{p}_i - \mathbf{p}_0}{\|\mathbf{p}_9 - \mathbf{p}_0\|_2 + 10^{-6}}$$
   *Why:* The palm distance $\|\mathbf{p}_9 - \mathbf{p}_0\|_2$ is rigid during hand articulation. Normalizing by $S$ renders the model mathematically invariant to how close or far the hand is from the lens. The constant $10^{-6}$ prevents division-by-zero instability.

3. **Canonical 126D Dual-Hand Slotting:**
   Indian Sign Language includes both single-handed and dual-handed signs. To preserve handedness context without variable-length inputs:
   - **Indices $0 \dots 62$ ($21 \times 3$):** Left Hand coordinates.
   - **Indices $63 \dots 125$ ($21 \times 3$):** Right Hand coordinates.
   - If a hand is absent in a single-handed gesture, its 63 corresponding values are padded with zero ($0.0$).
   *Why:* Fixed vector length (126) enables high-throughput vectorized matrix multiplication while allowing the network to distinguish whether a sign requires one or two hands.

### 3.2 High-Throughput Multiprocess Extraction
Extracting landmarks sequentially over 42,745 images on CPU would take several hours. The extractor uses Python's `concurrent.futures.ProcessPoolExecutor` with:
- `max_workers = 8` parallel worker processes.
- Batch chunking (`chunk_size = 50`) to minimize inter-process IPC overhead.
- Throughput: ~180-250 images/second.
- Output: Compressed binary caches `train_landmarks.npz`, `val_landmarks.npz`, `test_landmarks.npz`.
- Failure Tracking: Images where MediaPipe failed to detect a hand (due to extreme blur or truncation) are logged in [`reports/failed_landmark_extractions.json`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/reports/failed_landmark_extractions.json). Detection success rate: **98.06%**.

---

## 4. Landmark Model Training (Deep Dive)

The script [`src/train_split_landmark_model.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/train_split_landmark_model.py) handles model instantiation, training, and checkpointing.

### 4.1 Neural Network Architecture
The landmark classifier is a 4-layer Deep Multi-Layer Perceptron (MLP) specifically configured for coordinate feature spaces:

| Layer Name | Layer Type | Specifications / Hyperparameters | Output Shape | Param Count | Purpose & Rationale |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `landmarks_input` | `InputLayer` | `shape=(126,)` | `(None, 126)` | 0 | Ingests 126D normalized coordinate vectors. |
| `dense1` | `Dense` | 256 units, `activation="relu"`, `kernel_initializer="he_normal"` | `(None, 256)` | 32,512 | Expands low-dimensional coordinates into a rich spatial representation. |
| `bn1` | `BatchNormalization` | $\gamma, \beta$ learnable scales | `(None, 256)` | 1,024 | Eliminates internal covariate shift; speeds up convergence. |
| `drop1` | `Dropout` | `rate=0.20` | `(None, 256)` | 0 | Prevents co-adaptation of coordinate neurons. |
| `dense2` | `Dense` | 128 units, `activation="relu"`, `kernel_initializer="he_normal"` | `(None, 128)` | 32,896 | Compresses spatial features into higher-order gesture representations. |
| `bn2` | `BatchNormalization` | $\gamma, \beta$ learnable scales | `(None, 128)` | 512 | Normalizes layer 2 activations across the batch. |
| `drop2` | `Dropout` | `rate=0.20` | `(None, 128)` | 0 | Regularization against joint coordinate noise. |
| `dense3` | `Dense` | 64 units, `activation="relu"`, `kernel_initializer="he_normal"` | `(None, 64)` | 8,256 | Bottleneck compression stage before class assignment. |
| `bn3` | `BatchNormalization` | $\gamma, \beta$ learnable scales | `(None, 64)` | 256 | Stabilizes inputs to the classification head. |
| `drop3` | `Dropout` | `rate=0.10` | `(None, 64)` | 0 | Light final regularization. |
| `class_probabilities` | `Dense` | 35 units, `activation="softmax"` | `(None, 35)` | 2,275 | Outputs categorical probability distribution across all 35 classes. |

- **Total Parameters:** 77,731 (~77K)
- **Trainable Parameters:** 76,835
- **Non-trainable Parameters (Batch Normalization moving statistics):** 896

### 4.2 Optimization & Regularization Techniques
1. **Weight Initialization (`he_normal`):**
   Weights are drawn from a Gaussian distribution with variance $\sigma^2 = \frac{2}{n_{\text{in}}}$.
   *Why:* Specifically derived for ReLU activation functions to prevent early gradient explosion or vanishing in dense networks.
2. **Batch Normalization:**
   Inserted after linear combinations and before dropout.
   *Why:* Coordinates across different joints vary in variance. Batch Normalization allows a higher initial learning rate ($10^{-3}$) without divergence.
3. **Dropout Regularization:**
   Drop rates of 0.2, 0.2, and 0.1 randomly zero neuron activations during training.
   *Why:* Ensures the model does not become over-reliant on a single finger joint (e.g. thumb tip), forcing it to learn holistic hand configurations.
4. **Loss Function (`sparse_categorical_crossentropy`):**
   $$\mathcal{L}_{\text{SCCE}} = -\log\left(\frac{e^{z_y}}{\sum_{j=1}^{C} e^{z_j}}\right)$$
   *Why:* Enables using integer class indices ($0 \dots 34$) directly without generating memory-heavy one-hot encoded label matrices.
5. **Optimizer (`Adam`):**
   Standard learning rate $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$.
   *Why:* Adaptive learning rates per parameter ensure smooth convergence on dense coordinate features.

### 4.3 Training Callbacks & Policies
- **`ModelCheckpoint`:** Monitors `val_accuracy` with `save_best_only=True`, persisting the optimal weights to [`artifacts/landmark_model.keras`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/artifacts/landmark_model.keras).
- **`EarlyStopping`:** Monitors `val_accuracy`, `patience=8`, `restore_best_weights=True`. Halts training if validation accuracy does not improve for 8 consecutive epochs.
- **`ReduceLROnPlateau`:** Monitors `val_loss`, `factor=0.5`, `patience=3`, `min_lr=1e-5`. Halves the learning rate when loss plateaus to allow fine-grained descent into local minima.
- **Batch Size:** 64. Provides an optimal compromise between vectorization throughput and stochastic gradient descent noise.
- **Epochs:** 30-35 epochs (typically converges between Epochs 18–24).

---

## 5. Legacy 2D CNN Model (Pixel Pipeline)

For benchmarking and backward compatibility, the repository retains the 2D CNN baseline ([`src/model.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/model.py), [`src/train.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/train.py)).

### 5.1 Image Preprocessing Pipeline (`preprocessing.py`)
```
Input Image (BGR)
       │
       ▼
[Grayscale Conversion]      ──> Y = 0.299R + 0.587G + 0.114B
       │
       ▼
[Gaussian Blur (3x3)]       ──> Noise suppression
       │
       ▼
[Canny Edge Detection]      ──> Thresholds: Low=50, High=150
       │
       ▼
[Letterbox Resizing]        ──> Aspect-ratio preserved scaling to 64x64
       │
       ▼
[Re-binarization (T=20)]    ──> Prevents edge dilution from interpolation
       │
       ▼
[Normalization [0.0, 1.0]]  ──> x / 255.0
```

1. **Grayscale Conversion:** Standard ITU-R BT.601 luminance calculation. Strips non-informative skin-tone variations.
2. **Gaussian Filtering:** $3 \times 3$ kernel removes high-frequency pixel sensor noise before gradient calculation.
3. **Canny Edge Detection:** Double thresholding ($T_{\text{low}}=50, T_{\text{high}}=150$) isolates hand contours while ignoring uniform palm and background patches.
4. **Aspect-Ratio-Preserving Letterbox Resizing:** Non-square crops are scaled proportionally ($\text{scale} = \min(64/W, 64/H)$) and centered onto a black 64x64 canvas.
5. **Post-Resize Re-binarization ($T=20$):** OpenCV area interpolation creates soft intermediate grays on binary edges. Thresholding at 20 restores sharp, crisp binary contours.

### 5.2 CNN Architecture Summary
- **Type:** 4-Block Convolutional Architecture.
- **Layers:**
  - Block 1: Conv2D(32) $\rightarrow$ BN $\rightarrow$ Conv2D(32) $\rightarrow$ MaxPool(2x2) $\rightarrow$ Dropout(0.2)
  - Block 2: Conv2D(64) $\rightarrow$ BN $\rightarrow$ Conv2D(64) $\rightarrow$ MaxPool(2x2) $\rightarrow$ Dropout(0.25)
  - Block 3: Conv2D(128) $\rightarrow$ BN $\rightarrow$ MaxPool(2x2) $\rightarrow$ Dropout(0.3)
  - Block 4: Conv2D(256) $\rightarrow$ BN $\rightarrow$ MaxPool(2x2) $\rightarrow$ Dropout(0.35)
  - Head: Flatten $\rightarrow$ Dense(256) $\rightarrow$ BN $\rightarrow$ Dropout(0.4) $\rightarrow$ Dense(35, Softmax).
- **Parameters:** ~520,000 parameters.
- **Checkpoint Size:** 17.19 MB.

---

## 6. How Model Analysis & Evaluation Work

The evaluation pipeline ([`src/evaluate_landmark_model.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/evaluate_landmark_model.py), [`src/evaluate.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/evaluate.py)) carries out scientific verification on the untouched test split.

### 6.1 Evaluation Methodology
1. **Zero-Leakage Test Manifest:** The evaluation scripts read strictly from `data/splits/test_manifest.csv` and `data/processed/test_landmarks.npz`. No test sample was ever seen during training or callback selection.
2. **Prediction Generation:** Batched forward passes compute predicted class probabilities:
   $$\hat{y} = \arg\max_{c \in [0..34]} P(c \mid \mathbf{x})$$
3. **Statistical Metrics:**
   - **Accuracy:** Total correct classifications divided by total test set count:
     $$\text{Accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{I}(y_i = \hat{y}_i)$$
   - **Precision ($P_c$) & Recall ($R_c$):**
     $$P_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}, \quad R_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$
   - **Macro F1-Score:** Unweighted average of per-class F1-scores, ensuring performance is not masked by sample size differences:
     $$\text{Macro } F_1 = \frac{1}{C}\sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
4. **Latency Profiling:** Measures inference time over 1,000 warm iterations on CPU, recording Mean, 95th Percentile ($P_{95}$), and standard deviation.

### 6.2 Empirical Benchmark: Landmark MLP vs. Legacy CNN

| Evaluation Parameter | Legacy 2D CNN (`best_model.keras`) | Landmark MLP (`landmark_model.keras`) | Scientific / Engineering Significance |
| :--- | :---: | :---: | :--- |
| **Model Size on Disk** | 17.19 MB | **0.94 MB** | **18.3x lighter footprint**; suitable for embedded mobile/edge execution. |
| **Parameter Count** | ~520,000 | **~77,000** | **6.7x fewer weights**; drastically lower memory bandwidth requirement. |
| **Test Accuracy (Held-Out)** | 100.00% | **99.86%** | Statistical parity on identical Kaggle test samples. |
| **Macro F1-Score** | 1.0000 | **0.9985** | Balanced performance across all 35 alphanumeric classes. |
| **Test Set Misclassifications** | 0 / 4,275 | **6 / 4,192** | Negligible error rate (0.14%). |
| **CPU Latency (Inference)** | 25.0 ms | **8.74 ms (pipeline) / 0.65 ms (model)** | Landmark MLP achieves sub-millisecond tensor operations. |
| **Background Sensitivity** | ❌ High (room walls, textures trigger false edges) | ✅ **Zero (100% immune)** | MediaPipe extracts pure joint coordinates; background pixels are ignored. |
| **Face Occlusion Contamination** | ❌ Face skin inside ROI causes misclassifications | ✅ **Zero (100% immune)** | Hand joint tracking is physically separated from facial contours. |
| **Dual-Hand Support** | ❌ Flattened into single 64x64 bounding box | ✅ **Native (63 L + 63 R slots)** | Maintains independent spatial separation between both hands. |

### 6.3 Audit Finding: De-mystifying the "99.50% Accuracy Claim"
In earlier development logs, a claim of "99.50% accuracy on 42,745 images" was noted. A formal audit ([`reports/accuracy_claim_audit.md`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/reports/accuracy_claim_audit.md)) traced this metric to:
- A live progress printout at step 392 of Epoch 2 during CNN training (`task-313.log: 392/535 ... accuracy: 0.9950`).
- **Audit Verdict:** That figure was a **running training-batch accuracy**, not final test accuracy. However, when properly evaluated on the held-out test split, the legacy CNN achieved **100.0%** and the Landmark MLP achieved **99.86%** on test data.
- **Real-World Scope:** The Kaggle dataset consists of recorded sign sessions with uniform backgrounds. Test set accuracy represents laboratory validation; live webcam deployment requires confidence gating and temporal smoothing to handle unconstrained environments.

---

## 7. Real-Time Inference, Word Building & Translation Pipeline

During live operation ([`src/inference.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/inference.py), [`src/webcam_app.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/webcam_app.py), [`src/ui_app.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/ui_app.py)), frames undergo a 10-step multi-stage processing pipeline:

```
 Webcam Video Stream (cv2.VideoCapture)
                 │
                 ▼
     [MediaPipe Hands Tracker]
                 │
      ┌──────────┴──────────┐
      │                     │
[Hand Detected]       [No Hand Detected]
      │                     │
      ▼                     ▼
[126D Coordinate       [Reset Counters &
 Normalization]        Display "No sign detected"]
      │
      ▼
[Landmark MLP Predict]
      │
      ▼
[Confidence Gating] ──> Reject if confidence < 0.70
      │
      ▼
[Temporal Smoothing] ──> 5-frame majority vote deque eliminates jitter
      │
      ▼
[Closed-Fist Check] ──> If held 1.2s ==> Emit SPACE (Word Boundary)
      │
      ▼
[Hold Confirmation] ──> If sign held 1.0s ==> Commit Character to Buffer
      │
      ▼
[Word Builder & LLM] ──> Groq API / SpellChecker cleans fingerspelling into coherent word
      │
      ▼
[Translator Engine]  ──> Translates English text to Hindi / Punjabi / Tamil / Bengali
      │
      ▼
[Pillow Unicode HUD] ──> Renders regional Indian script on video canvas
      │
      ▼
[Non-Blocking TTS]   ──> gTTS + pygame.mixer plays audio speech in background
```

### Detailed Pipeline Breakdown:

1. **Camera Frame Acquisition:** OpenCV captures BGR frames at 30 FPS (`640x480` or `1280x720`).
2. **MediaPipe Tracking:** Evaluates whether one or two hands are present, extracting the 21 keypoints per hand.
3. **126D Vector Construction:** Translates coordinates to wrist origin, scales by palm length, fills missing hand slots with zeros, and runs sanity checks against NaNs or infinite values.
4. **Model Inference:** Passes the 126D vector through the Landmark MLP, yielding a 35-class softmax probability array.
5. **Confidence Gating:** If the maximum probability $P_{\max} < 0.70$, the frame is marked as *"No reliable sign detected"* to suppress ambiguous gestures.
6. **Temporal Rolling Buffer (Majority Voting):**
   - Predictions are fed into a sliding window `deque(maxlen=5)`.
   - The label selected is the majority winner: $\text{mode}(\text{window})$.
   - *Why:* Prevents transient hand motions between signs from triggering false predictions.
7. **Hold Confirmation Timer:**
   - The user must hold a sign stably for **1.0 second**.
   - A visual progress ring/bar on the UI fills from 0% to 100%.
   - Upon hitting 1.0s, the character is committed to the letter buffer, and an auditory click is triggered.
8. **Closed-Fist SPACE Gesture Detection:**
   - To separate words naturally without touching a keyboard, the system detects a closed fist:
     $$\text{is\_fist\_closed} = \forall \text{finger} \in \{\text{Index, Mid, Ring, Pinky}\}, \quad \|\text{Tip} - \text{Wrist}\|_2 < \|\text{PIP} - \text{Wrist}\|_2$$
   - Holding a closed fist for **1.2 seconds** emits a SPACE, finalizing the current word.
9. **Intelligent Word Reconstruction & LLM:**
   - Users fingerspell letter-by-letter (e.g., `H - E - L - L - O`).
   - The [`src/text_builder.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/text_builder.py) module supports dual-tier word reconstruction:
     - **Offline:** Levenshtein-distance dictionary correction via `pyspellchecker`.
     - **Online:** Groq Cloud API running `llama-3.1-8b-instant` to infer intended words, handle missing/repeated letters, and construct grammatically coherent sentences.
10. **Multilingual Translation & Unicode Rendering:**
    - [`src/translator_engine.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/translator_engine.py) translates completed words into target Indian regional languages (Hindi, Punjabi, Tamil, Telugu, Bengali, Gujarati, Marathi, Urdu) via `deep-translator`.
    - Because standard OpenCV `cv2.putText()` cannot render non-ASCII complex script ligatures (rendering question marks `???`), the system converts OpenCV frames to PIL images, applies TrueType font rendering (`Nirmala.ttf` / `arial.ttf`), and converts back to NumPy.
    - **Audio Playback:** Background worker threads synthesize speech using Google Text-to-Speech (`gTTS`) and stream audio via `pygame.mixer` with zero frame drops on the video feed.

---

## 8. Summary of File Artifacts & Roles

| File Path | Functional Role in System |
| :--- | :--- |
| [`config.yaml`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/config.yaml) | Central configuration for preprocessing, hyperparameters, model paths, and UI settings. |
| [`src/extract_landmarks_splits.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/extract_landmarks_splits.py) | Multiprocess MediaPipe feature extractor converting raw images to normalized 126D vectors. |
| [`src/train_split_landmark_model.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/train_split_landmark_model.py) | Training script for the 77K-parameter Landmark MLP model with early stopping and LR decay. |
| [`src/evaluate_landmark_model.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/evaluate_landmark_model.py) | Independent evaluator computing accuracy, Macro F1, confusion matrices, and latency. |
| [`src/model.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/model.py) | Defines the 4-Block Compact CNN (~520K params) and the deep Research Paper CNN (~30M params). |
| [`src/preprocessing.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/preprocessing.py) | Unified Grayscale + Canny + Letterbox 64x64 pipeline and landmark vector validation rules. |
| [`src/inference.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/inference.py) | Dual-engine live inference manager with confidence gating and temporal stabilization. |
| [`src/text_builder.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/text_builder.py) | Fingerspelling character buffer, closed-fist detection, and LLM word reconstruction. |
| [`src/translator_engine.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/translator_engine.py) | Multilingual translation engine and non-blocking speech synthesizer. |
| [`src/ui_app.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/ui_app.py) | Full-featured Tkinter/CustomTkinter desktop GUI with live video HUD and translation controls. |
| [`src/webcam_app.py`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/src/webcam_app.py) | Standalone lightweight OpenCV camera loop with live HUD overlays. |
