# Proposed Feature Removals & Decommissioning Plan
### Streamlining to Strict Scope: English Word Recognition from ISL Landmarks

**Target System:** Indian Sign Language (ISL) Recognition  
**Date:** October 2026  
**Status:** Awaiting User Approval  

---

## 1. Approval Gate Overview

In strict accordance with Section 2 and Section 13 of the Master Implementation Specification, features outside the English word recognition scope (Text-to-Speech, Speech Recognition, Multilingual Regional Translation, Cloud LLM Word Autocorrection, Audio Recording/Playback) are inventoried below.

**Policy:**
- **No files are deleted, dependencies uninstalled, or code destructively removed** at this stage.
- The new landmark-based recognition and English word builder are implemented non-destructively alongside existing components.
- The user can review this proposal and approve or reject each item.

---

## 2. Removal Candidates Matrix

| Candidate # | Component / Feature | File Paths Involved | Dependencies | Current Function | Impact of Removal | Recommendation |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- |
| **C1** | **Regional Language Translation** | `src/translator_engine.py`<br>`src/inference.py`<br>`src/webcam_app.py`<br>`src/ui_app.py` | `deep-translator` | Translates English text into Hindi, Punjabi, Bengali, Tamil, etc. | Removes regional language translation; restricts application output strictly to English words. | **Propose Removal**: Focuses pipeline purely on accurate English word building. |
| **C2** | **Text-to-Speech (TTS) Voice Synthesis** | `src/translator_engine.py`<br>`src/inference.py`<br>`src/webcam_app.py`<br>`src/ui_app.py` | `gTTS`<br>`pygame` | Generates spoken audio responses in background thread. | Eliminates audio playback and background sound mixer threads. | **Propose Removal**: Not required for visual English word recognition. |
| **C3** | **Cloud LLM Word Autocorrection** | `src/text_builder.py` | `groq` | Sends noisy letters to cloud LLM to reconstruct words. | Eliminates cloud API dependency and latency; preserves raw letter sequences so classifier errors remain visible. | **Propose Removal**: Guarantees true offline operation and unmasked model measurement. |
| **C4** | **Indian Script TrueType Fonts** | `assets/fonts/*.ttf` | Pillow font loader | Renders Devanagari, Gurmukhi, Tamil, etc., on video frames. | Reverts video frame text rendering to clean standard English typography. | **Propose Removal**: Reduces asset footprint. |
| **C5** | **Out-of-Scope Dependencies in `requirements.txt`** | `requirements.txt` | `deep-translator`<br>`gTTS`<br>`pygame`<br>`groq`<br>`symspellpy` | External NLP, audio, and translation packages. | Reduces virtual environment footprint and prevents version conflicts. | **Propose Removal**: Cleans dependency tree. |

---

## 3. Preserved Baseline Components (Strictly Retained)

The following components will **NOT** be touched or removed:
1. `artifacts/best_model.keras` (Legacy 2D CNN baseline checkpoint).
2. `artifacts/class_mapping.json` (Canonical class indices).
3. `src/model.py`, `src/preprocessing.py`, `src/train.py`, `src/evaluate.py`.
4. `data/splits/*.csv` (Canonical training, validation, and test split manifests).

---

## 4. User Action Requested

Please indicate your decision:
- **Approve All Removals**: Decommission C1–C5 after the new English word recognition pipeline is validated.
- **Selective Retention**: Specify which components (if any) you wish to retain for future phases.
