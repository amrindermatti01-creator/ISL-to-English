# MediaPipe Landmark Extraction & Split Audit Report

**Date:** 2026-10-24 17:27:17  
**Target:** Indian Sign Language (ISL) 35-Class Dataset  
**Framework:** MediaPipe Hands (static image mode, 21 3D landmarks per hand, normalized 126D)  

---

## 1. Split Extraction Summary

Landmarks were extracted strictly following the canonical manifests in `data/splits/`.
No images were reassigned or leaked between splits.

| Split Name | Manifest Total | Successfully Extracted | Failed Detections | Detection Rate | Cache Path |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Test Split** | **4,275** | **4,192** | 83 | **98.06%** | `data\processed\test_landmarks.npz` |
| **Validation Split** | **4,274** | **4,187** | 87 | **97.96%** | `data\processed\val_landmarks.npz` |
| **Training Split** | **14,000** | **13,747** | 253 | **98.19%** | `data\processed\train_landmarks.npz` |

---

## 2. Detection Failure Analysis

Total recorded detection failures across processed samples: **423**.
Full per-sample failure logs are archived in [`reports/failed_landmark_extractions.json`](file:///c:/Users/hp/ISL%20Detetcion%20and%20English%20Converter/reports/failed_landmark_extractions.json).

### Failure Breakdown by Class:
| Class | Total Failed Samples | Primary Cause |
| :---: | :---: | :--- |
| **B** | 7 | Hand partially cropped or low edge contrast in Kaggle image |
| **C** | 65 | Hand partially cropped or low edge contrast in Kaggle image |
| **G** | 1 | Hand partially cropped or low edge contrast in Kaggle image |
| **O** | 56 | Hand partially cropped or low edge contrast in Kaggle image |
| **P** | 33 | Hand partially cropped or low edge contrast in Kaggle image |
| **Q** | 241 | Hand partially cropped or low edge contrast in Kaggle image |
| **S** | 15 | Hand partially cropped or low edge contrast in Kaggle image |
| **U** | 1 | Hand partially cropped or low edge contrast in Kaggle image |
| **V** | 4 | Hand partially cropped or low edge contrast in Kaggle image |

---

## 3. Landmark Representation & Normalization Formula

1. **Dual-Hand Slot Allocation**:
   - `0..62`: Left hand landmarks ($21 \times [x, y, z]$)
   - `63..125`: Right hand landmarks ($21 \times [x, y, z]$)
   - If a hand is missing, its 63 values are padded with 0.0.
2. **Wrist Centering & Geometric Scale Normalization**:
   $$\mathbf{p}_i^{\text{norm}} = \frac{\mathbf{p}_i - \mathbf{p}_0}{\|\mathbf{p}_9 - \mathbf{p}_0\|_2 + 10^{-6}}$$
   where $\mathbf{p}_0$ is the wrist landmark, and $\mathbf{p}_9$ is the middle-finger MCP joint.
   This guarantees complete mathematical invariance to camera distance and user position in frame.
