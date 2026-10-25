"""
src/extract_landmarks.py
High-Performance Multiprocess Feature Extraction Pipeline for Indian Sign Language (ISL).
Converts raw images into canonical 126-dimensional coordinate feature vectors
using MediaPipe Hands with wrist-relative translation normalization and
left-hand reflection augmentation.
"""

import os
import sys
import glob
import argparse
import logging
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import cv2
import mediapipe as mp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ISLLandmarkExtractor")


def _extract_from_image_array(
    image_bgr: np.ndarray,
    hands_detector: mp.solutions.hands.Hands,
    normalize_wrist: bool = True
) -> Optional[np.ndarray]:
    if image_bgr is None or image_bgr.size == 0:
        return None

    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_image)

    if not results.multi_hand_landmarks:
        return None

    left_pts = None
    right_pts = None

    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
        handedness_label = results.multi_handedness[i].classification[0].label.lower()
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32
        )

        if handedness_label == "left":
            left_pts = coords
        else:
            right_pts = coords

    if left_pts is None and right_pts is None:
        return None

    if normalize_wrist:
        if left_pts is not None and right_pts is not None:
            ref = left_pts[0].copy()
            left_vec = (left_pts - ref).flatten()
            right_vec = (right_pts - ref).flatten()
        elif left_pts is not None:
            ref = left_pts[0].copy()
            left_vec = (left_pts - ref).flatten()
            right_vec = np.zeros(63, dtype=np.float32)
        else:
            ref = right_pts[0].copy()
            left_vec = np.zeros(63, dtype=np.float32)
            right_vec = (right_pts - ref).flatten()
    else:
        left_vec = left_pts.flatten() if left_pts is not None else np.zeros(63, dtype=np.float32)
        right_vec = right_pts.flatten() if right_pts is not None else np.zeros(63, dtype=np.float32)

    return np.concatenate([left_vec, right_vec]).astype(np.float32)


def _worker_process_class(
    cls_name: str,
    cls_idx: int,
    cls_folder: str,
    max_samples: Optional[int],
    confidence: float
) -> Tuple[str, int, List[np.ndarray], int]:
    """
    Worker function executed in separate process.
    Initializes its own MediaPipe detector and extracts features for one class.
    """
    # Suppress verbose TF/MediaPipe logs in worker
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GLOG_minloglevel"] = "3"

    img_files = glob.glob(os.path.join(cls_folder, "*.jpg")) + \
                glob.glob(os.path.join(cls_folder, "*.png")) + \
                glob.glob(os.path.join(cls_folder, "*.jpeg"))
    img_files.sort()

    if max_samples and len(img_files) > max_samples:
        indices = np.linspace(0, len(img_files) - 1, max_samples, dtype=int)
        selected_files = [img_files[idx] for idx in indices]
    else:
        selected_files = img_files

    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=confidence
    )

    extracted_vectors = []
    try:
        for fpath in selected_files:
            img = cv2.imread(fpath)
            vec = _extract_from_image_array(img, detector, normalize_wrist=True)
            if vec is not None:
                extracted_vectors.append(vec)
    finally:
        detector.close()

    return cls_name, cls_idx, extracted_vectors, len(selected_files)


def augment_mirror_landmarks(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates mirrored samples for left-handed signers:
    1. Swaps Left Hand (0..62) and Right Hand (63..125) slots.
    2. Reflects x-coordinates: x -> -x.
    """
    left_slot = features[:, 0:63].copy()
    right_slot = features[:, 63:126].copy()

    aug_features = np.hstack((right_slot, left_slot))
    aug_features[:, 0::3] = -aug_features[:, 0::3]

    combined_features = np.vstack((features, aug_features))
    combined_labels = np.concatenate((labels, labels.copy()))
    return combined_features, combined_labels


def process_dataset_multiprocess(
    dataset_dir: str,
    output_npz_path: str,
    max_samples_per_class: Optional[int] = 300,
    augment_mirror: bool = True,
    confidence: float = 0.5,
    num_workers: int = 8
) -> Dict[str, any]:
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    classes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
    ])
    logger.info(f"Discovered {len(classes)} classes across {dataset_dir}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}

    logger.info(f"Dispatching extraction across {num_workers} worker processes...")

    all_features = []
    all_labels = []
    stats_per_class = {}

    futures = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for cls_name in classes:
            cls_folder = os.path.join(dataset_dir, cls_name)
            cls_idx = class_to_idx[cls_name]
            f = executor.submit(
                _worker_process_class,
                cls_name, cls_idx, cls_folder, max_samples_per_class, confidence
            )
            futures.append(f)

        for future in as_completed(futures):
            cls_name, cls_idx, vectors, total_imgs = future.result()
            detected_count = len(vectors)
            for vec in vectors:
                all_features.append(vec)
                all_labels.append(cls_idx)

            rate = (detected_count / max(1, total_imgs)) * 100
            stats_per_class[cls_name] = {
                "total_images": total_imgs,
                "detected": detected_count,
                "detection_rate": rate / 100.0
            }
            logger.info(f"[Worker Done] Class '{cls_name}': {detected_count}/{total_imgs} ({rate:.1f}%)")

    X_raw = np.array(all_features, dtype=np.float32)
    y_raw = np.array(all_labels, dtype=np.int64)

    logger.info(f"Parallel extraction finished: {X_raw.shape[0]} base samples extracted.")

    if augment_mirror and len(X_raw) > 0:
        logger.info("Applying left-handed mirror augmentation...")
        X, y = augment_mirror_landmarks(X_raw, y_raw)
        logger.info(f"Total dataset after augmentation: {X.shape[0]} samples.")
    else:
        X, y = X_raw, y_raw

    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    np.savez_compressed(
        output_npz_path,
        X=X,
        y=y,
        classes=classes,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class
    )
    file_size_kb = os.path.getsize(output_npz_path) / 1024
    logger.info(f"Saved dataset artifact: {output_npz_path} ({file_size_kb:.1f} KB)")

    return {
        "output_path": output_npz_path,
        "total_samples": len(X),
        "num_classes": len(classes),
        "classes": classes,
        "stats": stats_per_class
    }


def main():
    parser = argparse.ArgumentParser(description="Multiprocess 126D MediaPipe Hand Feature Extraction.")
    parser.add_argument("--dataset-dir", type=str, default="C:/Users/hp/Indian", help="Dataset directory")
    parser.add_argument("--output", type=str, default="data/processed/landmarks_126d.npz", help="NPZ output path")
    parser.add_argument("--max-samples", type=int, default=300, help="Max samples per class (default: 300)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirror augmentation")
    parser.add_argument("--confidence", type=float, default=0.5, help="MediaPipe min confidence")
    args = parser.parse_args()

    process_dataset_multiprocess(
        dataset_dir=args.dataset_dir,
        output_npz_path=args.output,
        max_samples_per_class=args.max_samples,
        augment_mirror=not args.no_mirror,
        confidence=args.confidence,
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()
