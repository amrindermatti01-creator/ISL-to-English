# Indian Sign Language (ISL) Webcam Inference Benchmark & Validation Report

- **Date**: 2026-10-24 15:50:42
- **Model Checkpoint**: `artifacts\best_model.keras`
- **Class Count**: 35 classes (`1`-`9`, `A`-`Z`)
- **Confidence Threshold**: `0.7`
- **Temporal Smoothing Window**: `5` frames (majority vote)
- **Stability Duration**: `1.0` seconds

---

## 1. Latency & Throughput Performance

Tested over **50 sequential full-pipeline frames** (MediaPipe detection $\to$ Dynamic ROI crop $\to$ 64×64 Canny edge normalization $\to$ CNN forward pass $\to$ Temporal vote $\to$ HUD overlay).

| Metric | Measured Value | Standard Target | Status |
| :--- | :--- | :--- | :--- |
| **Average Throughput (FPS)** | **26.2 FPS** | $\ge 20$ FPS | **PASSED** |
| **Mean End-to-End Latency** | **36.9 ms** | $\le 50$ ms | **PASSED** |
| **Median (P50) Latency** | **34.6 ms** | $\le 45$ ms | **PASSED** |
| **95th Percentile (P95) Latency** | **48.3 ms** | $\le 65$ ms | **PASSED** |
| **Min Latency** | **25.8 ms** | - | - |
| **Max Latency** | **95.7 ms** | - | - |

---

## 2. Pipeline Integrity & Edge Case Verification

1. **Preprocessing Parity**:
   - Live crops undergo identical pipeline: Grayscale ($Y=0.299R+0.587G+0.114B$) $\to$ Gaussian Blur ($5\times 5$, $\sigma=0$) $\to$ Canny Edge ($50, 150$) $\to 64\times 64$ resize $\to$ float32 normalization $[0, 1]$.
   - PiP inset visually verifies edges fed to the model in real-time.
2. **Rejection of Uncertain Gestures**:
   - Frames where hand is absent or confidence falls below `0.7` are rejected rather than forced into a random class, displaying `"No reliable sign detected"`.
3. **Temporal Flicker Suppression**:
   - Majority voting over a rolling deque window of 5 frames eliminates frame-to-frame label oscillations.
4. **Fingerspelling Buffer**:
   - Characters are only committed to the sentence buffer if held continuously for 1.0s, preventing accidental insertions during hand transitions.

---

## 3. Real-Time Execution Instructions

Launch the live camera app:
```bash
.venv\Scripts\python.exe src/webcam_app.py --camera 0
```
Interactive Controls:
- **`[SPACE]`**: Insert space in spelled text
- **`[BACKSPACE]`**: Delete last character
- **`[C]`**: Clear entire text buffer
- **`[F]`**: Toggle between MediaPipe Dynamic Hand Tracker and Central Fixed ROI
- **`[S]`**: Save snapshot to `reports/snapshots/`
- **`[Q]` / `[ESC]`**: Clean exit
