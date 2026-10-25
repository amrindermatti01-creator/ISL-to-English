"""
src/cache_real_dataset.py - Step 2: MediaPipe Hand ROI Pre-Extraction & Binary Caching
Extracts tight hand ROIs with convex hull masking, standardizes to 64x64 Canny edges,
and caches arrays to data/processed/*.npz for high-speed CNN training.

Features:
  1. Multi-threaded pipeline (Thread-local MediaPipe Hands).
  2. Dual-hand bounding box union + 10% proportional padding.
  3. Dilated convex hull masking to eradicate background, torso, and face edges.
  4. Aspect-ratio preserving square letterboxing.
  5. Fallback for unobserved hands to ensure zero dropped samples.
  6. Visual verification artifact generation: reports/real_camera_sample_preprocessed.png.
  7. Compact uint8 storage in .npz (loads in milliseconds into RAM).
"""

import os
import sys
import time
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

# Thread-local storage for MediaPipe Hands
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
    masks out background / face clutter using a dilated convex hull,
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

    # 10% proportional padding around hand cluster
    pad_x = max(12, int(box_w * margin_scale))
    pad_y = max(15, int(box_h * margin_scale))

    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)

    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return None

    # Crop tight hand region
    roi = frame_bgr[y1:y2, x1:x2].copy()
    rh, rw = roi.shape[:2]

    # Hand convex hull mask to eradicate shirt/face/room contours
    rel_pts = pts_arr - [x1, y1]
    hull = cv2.convexHull(rel_pts)
    mask = np.zeros((rh, rw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Dilate slightly so finger edges and flesh contours remain intact
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=1)
    masked_roi = cv2.bitwise_and(roi, roi, mask=mask)

    # Letterbox to square canvas with clean black borders
    max_dim = max(rh, rw)
    square_roi = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    off_x = (max_dim - rw) // 2
    off_y = (max_dim - rh) // 2
    square_roi[off_y:off_y + rh, off_x:off_x + rw] = masked_roi

    return square_roi

def fallback_center_letterbox(img_bgr):
    """Fallback when hand detector fails (e.g. extreme blur): center square crop."""
    h, w = img_bgr.shape[:2]
    max_dim = max(h, w)
    square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    off_x = (max_dim - w) // 2
    off_y = (max_dim - h) // 2
    square[off_y:off_y + h, off_x:off_x + w] = img_bgr
    return square

def _process_single_image(args):
    """Worker function to process one image row."""
    idx, img_path, class_idx, target_size, canny_low, canny_high = args
    
    img = cv2.imread(img_path)
    if img is None:
        # Fallback empty black image
        return idx, np.zeros((target_size[1], target_size[0], 1), dtype=np.uint8), class_idx, False

    detector = get_thread_detector()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_res = detector.process(rgb)

    hand_detected = False
    processed_crop = None

    if mp_res.multi_hand_landmarks:
        roi = extract_hand_roi(img, mp_res.multi_hand_landmarks)
        if roi is not None:
            processed_crop = roi
            hand_detected = True

    if processed_crop is None:
        # Fallback to center-square letterbox
        processed_crop = fallback_center_letterbox(img)

    # Apply unified Canny edge preprocessing
    proc = preprocess_image(
        processed_crop,
        target_size=target_size,
        use_canny=True,
        canny_low=canny_low,
        canny_high=canny_high,
        normalize=False # Keep uint8 for compact disk cache
    )

    return idx, proc, class_idx, hand_detected

def cache_real_split(
    manifest_csv: Path,
    output_npz: Path,
    target_size=(64, 64),
    canny_low=50,
    canny_high=150,
    max_workers=8
):
    """Processes an entire manifest into a compressed NPZ binary cache."""
    df = pd.read_csv(manifest_csv)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    n_samples = len(df)
    
    print("\n" + "=" * 75)
    print(f"PROCESSING SPLIT: {manifest_csv.name} -> {output_npz.name}")
    print(f"Total samples:    {n_samples:,}")
    print(f"Target size:      {target_size}")
    print(f"Parallel threads: {max_workers}")
    print("=" * 75)

    X = np.empty((n_samples, target_size[1], target_size[0], 1), dtype=np.uint8)
    y = np.empty(n_samples, dtype=np.int32)
    detected_count = 0

    task_args = [
        (i, row.path, row.class_idx, target_size, canny_low, canny_high)
        for i, row in enumerate(df.itertuples())
    ]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, proc, class_idx, detected in tqdm(
            executor.map(_process_single_image, task_args),
            total=n_samples,
            desc=f"Caching {output_npz.stem}"
        ):
            X[idx] = proc
            y[idx] = class_idx
            if detected:
                detected_count += 1

    t1 = time.time()
    elapsed = t1 - t0
    detection_rate = (detected_count / n_samples) * 100.0

    print(f"\n[Split Caching Finished]")
    print(f"  Processed {n_samples:,} images in {elapsed:.1f}s ({n_samples/elapsed:.1f} imgs/s, {elapsed/n_samples*1000:.1f}ms/img)")
    print(f"  MediaPipe Hand Detection Success: {detected_count:,} / {n_samples:,} ({detection_rate:.2f}%)")
    print(f"  Fallback crops used:             {n_samples - detected_count} ({(100 - detection_rate):.2f}%)")

    print(f"Saving compressed NumPy archive to: {output_npz}...")
    np.savez_compressed(output_npz, X=X, y=y)
    file_mb = output_npz.stat().st_size / (1024 * 1024)
    print(f"[OK] Cache file saved successfully! Size: {file_mb:.2f} MB\n")

    return {
        "manifest": str(manifest_csv),
        "output_npz": str(output_npz),
        "samples": n_samples,
        "detection_rate": detection_rate,
        "elapsed_sec": elapsed,
        "cache_mb": file_mb
    }

def generate_sample_preprocessed_visual(
    manifest_csv: Path,
    output_png: Path,
    num_samples=10,
    target_size=(64, 64)
):
    """
    Renders raw camera captures side-by-side with their preprocessed
    MediaPipe ROI 64x64 Canny edge outputs for visual inspection.
    """
    df = pd.read_csv(manifest_csv)
    sample_df = df.groupby("class_name").first().reset_index().head(num_samples)

    fig, axes = plt.subplots(num_samples, 2, figsize=(6, num_samples * 2.2))
    plt.suptitle("MediaPipe Hand ROI + 64x64 Canny Edge Verification", fontsize=12, fontweight="bold")

    detector = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    for i, row in sample_df.iterrows():
        img_bgr = cv2.imread(row["path"])
        c_name = row["class_name"]

        # Raw RGB
        raw_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Process ROI
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)
        if res.multi_hand_landmarks:
            crop = extract_hand_roi(img_bgr, res.multi_hand_landmarks)
        else:
            crop = fallback_center_letterbox(img_bgr)
            
        canny_out = preprocess_image(crop, target_size=target_size, normalize=False)
        canny_view = canny_out.squeeze()

        # Plot Raw
        axes[i, 0].imshow(raw_rgb)
        axes[i, 0].set_title(f"Raw [{c_name}]", fontsize=10)
        axes[i, 0].axis("off")

        # Plot Preprocessed
        axes[i, 1].imshow(canny_view, cmap="gray")
        axes[i, 1].set_title(f"ROI Canny [{c_name}]", fontsize=10)
        axes[i, 1].axis("off")

    detector.close()
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"[OK] Preprocessing verification image saved to: {output_png}")

def main():
    parser = argparse.ArgumentParser(description="MediaPipe Hand ROI Pre-Extraction and Caching")
    parser.add_argument("--train-manifest", type=str, default="data/splits/real_train_manifest.csv")
    parser.add_argument("--val-manifest", type=str, default="data/splits/real_val_manifest.csv")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads (default 8)")
    parser.add_argument("--val-only", action="store_true", help="Cache validation split only")
    args = parser.parse_args()

    train_csv = Path(args.train_manifest)
    val_csv = Path(args.val_manifest)
    out_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)

    train_npz = out_dir / "real_train_64x64.npz"
    val_npz = out_dir / "real_val_64x64.npz"

    # 1. Validation split caching (faster, provides immediate feedback)
    val_stats = cache_real_split(
        manifest_csv=val_csv,
        output_npz=val_npz,
        target_size=(64, 64),
        max_workers=args.workers
    )

    # 2. Visual sample report
    visual_png = reports_dir / "real_camera_sample_preprocessed.png"
    print("\nGenerating sample preprocessing comparison image...")
    generate_sample_preprocessed_visual(val_csv, visual_png, num_samples=8)

    # 3. Training split caching
    if not args.val_only:
        train_stats = cache_real_split(
            manifest_csv=train_csv,
            output_npz=train_npz,
            target_size=(64, 64),
            max_workers=args.workers
        )
    else:
        train_stats = None

    print("\n" + "=" * 75)
    print("STEP 2 COMPLETE: ROI PRE-EXTRACTION AND CACHING SUCCESSFUL!")
    print(f"Validation cache: {val_npz} ({val_stats['cache_mb']:.1f} MB, {val_stats['detection_rate']:.1f}% detection)")
    if train_stats:
        print(f"Training cache:   {train_npz} ({train_stats['cache_mb']:.1f} MB, {train_stats['detection_rate']:.1f}% detection)")
    print(f"Sample preview:   {visual_png}")
    print("=" * 75)

if __name__ == "__main__":
    main()
