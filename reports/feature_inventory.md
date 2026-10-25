# Repository Feature Inventory & Scope Audit

**Project:** Indian Sign Language (ISL) Recognition System  
**Audit Date:** October 2026  
**Auditor:** Senior Computer Vision & Machine Learning Engineer  
**Scope Goal:** Accurate ISL-to-English Word Recognition Using MediaPipe Landmarks  

---

## 1. Executive Summary

This inventory documents all existing features, modules, dependencies, and artifacts currently in the repository. Each feature is categorized by its technical role, active status, dependencies, and whether it belongs strictly to the **English Word Recognition Scope**.

In accordance with Section 3 of the Master Implementation Specification, **no features are deleted, disabled, or destructively modified** during this audit. A formal removal proposal is presented separately in [reports/proposed_feature_removals.md](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/reports/proposed_feature_removals.md) for explicit user approval.

---

## 2. Comprehensive Feature Inventory Matrix

| Feature / Component | Files Involved | Core Dependencies | Currently In Use? | Belongs to English Word Scope? | Assessment & Retention Status |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **MediaPipe Hands Landmark Tracking** | `src/inference.py`<br>`src/extract_landmarks.py`<br>`src/webcam_app.py` | `mediapipe`<br>`opencv-python` | **Yes** (Primary) | **YES** | **Keep & Enhance**: Foundation of coordinate-based classification. Provides 21 3D joint coordinates per hand. |
| **Landmark Feature Normalization** | `src/extract_landmarks.py`<br>`src/inference.py` | `numpy` | **Yes** | **YES** | **Keep & Audit**: Converts raw coordinates to wrist-relative normalized 126D vectors. Must be verified across splits. |
| **Landmark Neural Network (MLP)** | `src/train_landmark_model.py`<br>`artifacts/landmark_model.keras`<br>`artifacts/landmark_class_mapping.json` | `tensorflow`<br>`keras`<br>`scikit-learn` | **Yes** | **YES** | **Keep & Re-Train on Splits**: Compact MLP (77K params). Needs formal re-training and evaluation strictly on split manifests (`data/splits/`). |
| **Legacy 2D CNN Model & Preprocessing** | `src/model.py`<br>`src/preprocessing.py`<br>`src/train.py`<br>`src/evaluate.py`<br>`artifacts/best_model.keras`<br>`artifacts/class_mapping.json` | `tensorflow`<br>`keras`<br>`opencv-python`<br>`pyyaml` | **Yes** (Baseline) | **YES** (Baseline Comparison) | **Preserve Intact**: Serves as the required benchmark baseline to compare against the new landmark model. |
| **Dataset Splits & Manifests** | `data/splits/train_manifest.csv`<br>`data/splits/val_manifest.csv`<br>`data/splits/test_manifest.csv` | `pandas` | **Yes** | **YES** | **Preserve Strictly**: Canonical 80/10/10 splits across 42,745 Kaggle images. Landmarks must be extracted strictly per split. |
| **English Word Builder (Fingerspelling)** | `src/text_builder.py`<br>`src/inference.py` | Python standard library (`time`, `collections`, `difflib`) | **Yes** | **YES** | **Keep & Refactor**: Must maintain raw recognized letters, support repeated letters (re-arm mechanism), and word boundary controls without cloud LLMs. |
| **Webcam Real-Time Inference Application** | `src/webcam_app.py`<br>`src/inference.py` | `opencv-python`<br>`mediapipe`<br>`numpy` | **Yes** | **YES** | **Keep & Clean**: Real-time camera loop with temporal smoothing and English text HUD. |
| **Desktop GUI Application** | `src/ui_app.py` | `tkinter`<br>`PIL`<br>`opencv-python` | **Yes** | **YES** | **Keep & Clean**: Desktop UI for displaying webcam stream, current letter, current word, and finalized words. |
| **Cloud LLM Word Autocorrection** | `src/text_builder.py` | `groq` | Optional | **NO** (Out of Scope) | **Marked for Removal Approval**: Replaces raw recognized spelling. Must be bypassed so classifier errors remain measurable. |
| **Regional Language Translation** | `src/translator_engine.py`<br>`src/inference.py`<br>`src/webcam_app.py`<br>`src/ui_app.py` | `deep-translator` | Active | **NO** (Out of Scope) | **Marked for Removal Approval**: Multi-language translation into Hindi, Punjabi, Tamil, etc., is outside the English word recognition scope. |
| **Text-to-Speech (TTS) Voice Synthesis** | `src/translator_engine.py`<br>`src/inference.py`<br>`src/webcam_app.py`<br>`src/ui_app.py` | `gTTS`<br>`pygame` | Active | **NO** (Out of Scope) | **Marked for Removal Approval**: Spoken audio generation is outside the English word recognition scope. |
| **TrueType Unicode Font Bundle** | `assets/fonts/*.ttf` | `PIL.ImageFont` | Active | **NO** (Out of Scope) | **Marked for Removal Approval**: Used exclusively for rendering non-ASCII Indic scripts on OpenCV frames. |

---

## 3. Detailed Audit Findings

### 3.1 Model Architecture & Input Formats
1. **Legacy CNN (`artifacts/best_model.keras`)**:
   - 4-Block Conv2D + BatchNorm + Dropout network (520,387 parameters, 17.19 MB).
   - Input shape: `(64, 64, 1)` grayscale Canny edge letterboxed image.
   - 35 classes: digits `1`–`9` (indices 0–8) and letters `A`–`Z` (indices 9–34).
2. **Current Landmark Model (`artifacts/landmark_model.keras`)**:
   - 4-layer Keras MLP: Dense(256) -> BatchNorm -> Dropout(0.2) -> Dense(128) -> BatchNorm -> Dropout(0.2) -> Dense(64) -> Dropout(0.1) -> Dense(35, softmax).
   - Input shape: `(126,)` float32 vector (Left Hand 63D + Right Hand 63D).
   - Parameters: 77,731 (0.94 MB).

### 3.2 Hand Landmark & Bounding Box Behavior
- Previously, `extract_hand_roi` computed `max(box_w, box_h) * 1.15`, expanding vertical hands horizontally by 150–200 pixels and drawing face, ear, neck, and wall edges into the 64×64 Canny image.
- By shifting to landmark coordinate classification, pixel cropping is completely bypassed during inference. Hand detection relies strictly on MediaPipe keypoints.

### 3.3 Dataset Splits & Independence
- Split manifests exist in `data/splits/`:
  - `train_manifest.csv`: 34,197 samples (80.0%)
  - `val_manifest.csv`: 4,274 samples (10.0%)
  - `test_manifest.csv`: 4,274 samples (10.0%)
  - Total: 42,745 images across 35 directories in `C:/Users/hp/Indian`.
- **Requirement**: Landmark extraction must be executed strictly following these split manifests to prevent data leakage and enable direct, unbiased comparison with the legacy CNN.

### 3.4 Unrelated Out-of-Scope Modules
The following modules fall outside the strict scope of English word recognition:
1. `src/translator_engine.py` (translation, TTS, audio playback)
2. `groq` integration in `src/text_builder.py` (cloud LLM autocorrection)
3. `deep-translator`, `gTTS`, `pygame`, `groq` dependencies in `requirements.txt`
4. Indian TrueType fonts in `assets/fonts/`

These components will be deactivated in the English word pipeline without destructive deletion, pending explicit user approval.
