"""
src/cache_mendeley_dataset.py - Step 3: MediaPipe Hand ROI Pre-Extraction & Binary Caching
Extracts tight hand ROIs with convex hull masking, standardizes to 64x64 Adaptive Canny edges,
and caches arrays to data/processed/*.npz for high-speed CNN training.

Features:
  1. Multi-threaded pipeline (Thread-local MediaPipe Hands).
  2. Dual-hand bounding box union + 10% proportional padding.
  3. Dilated convex hull masking to eradicate background, torso, and clothing contours.
  4. Adaptive Lighting Canny filter with median luminance estimation.
  5. Aspect-ratio preserving square letterboxing.
  6. Fallback for unobserved hands to ensure zero dropped samples.
  7. Visual verification artifact generation: reports/mendeley_sample_preprocessed.png.
  8. Compact uint8 storage in .npz.
"""

import os
import sys
import time
import json
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
from tqdm import tqdm

from src.preprocessing import preprocess_image

_thread_local = threading.local()

def get_thread_detector():
    """Initializes or retrieves the thread-local MediaPipe Hands detector."""
    if not hasattr(_thread_local, "hands"):
        _thread_local.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _thread_local.hands

def extract_hand_roi(frame_bgr, landmarks_list, margin_scale=0.10):
    """
    Computes a tight bounding box around all detected hand landmarks,
    masks out background / clothing clutter using a dilated convex hull,
    and letterboxes onto a square canvas with black padding.
    """
    h, w = frame_bgr.shape[:2]
    all_pts = []

    for hand_landmarks in landmarks_list:
        for lm in hand_landmarks.landmark:
            all_pts.append([int(lm.x * w), int(lm.y * h)])

    if not all_pts:
        return None

    pts_arr = np.array(all_pts, dtype=np.int32)
    x_min, y_min = np.min(pts_arr, axis=0)
    x_max, y_max = np.max(pts_arr, axis=0)

    box_w = x_max - x_min
    box_h = y_max - y_min

    pad_x = max(12, int(box_w * margin_scale))
    pad_y = max(15, int(box_h * margin_scale))

    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)

    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return None

    roi = frame_bgr[y1:y2, x1:x2].copy()
    rh, rw = roi.shape[:2]

    # Dilated convex hull mask
    rel_pts = pts_arr - [x1, y1]
    hull = cv2.convexHull(rel_pts)
    mask = np.zeros((rh, rw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=1)
    masked_roi = cv2.bitwise_and(roi, roi, mask=mask)

    # Square letterbox
    max_side = max(rh, rw)
    square_canvas = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    dx = (max_side - rw) // 2
    dy = (max_side - rh) // 2
    square_canvas[dy:dy + rh, dx:dx + rw] = masked_roi

    return square_canvas

def process_single_image(img_path, target_size=(64, 64)):
    """Worker function to process one image."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return np.zeros(target_size, dtype=np.uint8), False

    detector = get_thread_detector()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = detector.process(img_rgb)

    detected = False
    if results.multi_hand_landmarks:
        roi = extract_hand_roi(img_bgr, results.multi_hand_landmarks)
        if roi is not None:
            detected = True
            edges = preprocess_image(roi, target_size=target_size, use_canny=True, normalize=False)
            if len(edges.shape) == 3 and edges.shape[-1] == 1:
                edges = edges[:, :, 0]
            return edges, True

    # Fallback crop
    h, w = img_bgr.shape[:2]
    crop = img_bgr[int(h * 0.15):int(h * 0.90), int(w * 0.10):int(w * 0.90)]
    edges = preprocess_image(crop, target_size=target_size, use_canny=True, normalize=False)
    if len(edges.shape) == 3 and edges.shape[-1] == 1:
        edges = edges[:, :, 0]
    return edges, False

def cache_split_dataset(manifest_csv_path, class_to_idx, output_npz_path, num_workers=8):
    """Processes an entire manifest split and saves to .npz."""
    df = pd.read_csv(manifest_csv_path)
    print(f"\nProcessing {len(df):,} samples from: {manifest_csv_path}")

    paths = df["path"].tolist()
    labels = [class_to_idx[c] for c in df["class_name"]]

    features = []
    detection_flags = []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_single_image, paths), total=len(paths), desc=f"Caching {output_npz_path.name}"))

    for feat, flag in results:
        features.append(feat)
        detection_flags.append(flag)

    elapsed = time.time() - t0
    x_arr = np.array(features, dtype=np.uint8)
    y_arr = np.array(labels, dtype=np.int32)
    flags_arr = np.array(detection_flags, dtype=bool)

    det_rate = np.mean(flags_arr) * 100
    fps = len(paths) / elapsed if elapsed > 0 else 0
    print(f"  Processed in {elapsed:.1f}s ({fps:.1f} imgs/s)")
    print(f"  MediaPipe Hand Detection Rate: {det_rate:.2f}% ({np.sum(flags_arr):,} / {len(paths):,})")

    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz_path,
        x=x_arr,
        y=y_arr,
        detection_flags=flags_arr
    )
    print(f"  Saved to: {output_npz_path} ({output_npz_path.stat().st_size / (1024*1024):.2f} MB)")
    return x_arr, y_arr, det_rate

def generate_sample_mosaic(manifest_csv_path, output_png_path, num_samples=16):
    """Creates a sample visualization mosaic of preprocessed Mendeley images."""
    df = pd.read_csv(manifest_csv_path).sample(n=num_samples, random_state=42)
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for ax, (_, row) in zip(axes.flatten(), df.iterrows()):
        edges, detected = process_single_image(row["path"])
        ax.imshow(edges, cmap="gray")
        tag = "MP" if detected else "Fallback"
        ax.set_title(f"{row['class_name']} ({row['age_group'][0]}) [{tag}]", fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VISUAL] Sample visualization mosaic saved to: {output_png_path}")

def main():
    parser = argparse.ArgumentParser(description="Cache Mendeley ISL dataset features into NPZ archives.")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent worker threads")
    parser.add_argument("--skip-test", action="store_true", help="Skip caching test split")
    args = parser.parse_args()

    splits_dir = PROJECT_ROOT / "data/splits"
    processed_dir = PROJECT_ROOT / "data/processed"
    reports_dir = PROJECT_ROOT / "reports"
    artifacts_dir = PROJECT_ROOT / "artifacts"

    mapping_file = artifacts_dir / "class_mapping_mendeley.json"
    if not mapping_file.exists():
        # Fallback to real camera mapping (both are 26 classes A-Z)
        mapping_file = artifacts_dir / "class_mapping_real.json"

    with open(mapping_file, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)["class_to_idx"]

    val_manifest = splits_dir / "mendeley_val_manifest.csv"
    train_manifest = splits_dir / "mendeley_train_manifest.csv"
    test_manifest = splits_dir / "mendeley_test_manifest.csv"

    print("=" * 75)
    print("MENDELEY ISL FEATURE EXTRACTION & CACHING PIPELINE")
    print("=" * 75)

    if val_manifest.exists():
        cache_split_dataset(val_manifest, class_to_idx, processed_dir / "mendeley_val_64x64.npz", num_workers=args.workers)

    if train_manifest.exists():
        cache_split_dataset(train_manifest, class_to_idx, processed_dir / "mendeley_train_64x64.npz", num_workers=args.workers)

    if not args.skip_test and test_manifest.exists():
        cache_split_dataset(test_manifest, class_to_idx, processed_dir / "mendeley_test_64x64.npz", num_workers=args.workers)

    if val_manifest.exists():
        generate_sample_mosaic(val_manifest, reports_dir / "mendeley_sample_preprocessed.png")

    print("\n" + "=" * 75)
    print("FEATURE CACHING COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    main()
