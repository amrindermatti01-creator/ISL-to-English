"""
src/prepare_real_dataset.py - Step 1: Dataset Sanitation & Manifest Generation
For External Real Camera ISL Dataset: C:\\Users\\hp\\Real ISL Camera dataset

Responsibilities:
  1. Verifies python environment and core dependencies (MediaPipe, TensorFlow, OpenCV, Pandas).
  2. Scans splits (Training, Validation, Testing, Letters) across 26 classes (A-Z).
  3. Sanitizes dataset by filtering out non-image files, corrupted images, and script artifacts (<1KB).
  4. Generates artifacts/class_mapping_real.json (26 classes, 0-25).
  5. Produces validated CSV manifests:
       - data/splits/real_train_manifest.csv
       - data/splits/real_val_manifest.csv
       - data/splits/real_test_manifest.csv
       - data/splits/real_letters_manifest.csv
  6. Validates zero leakage across splits and complete class coverage.
"""

import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import pandas as pd
import numpy as np


def verify_environment():
    """Verifies that all required packages are present and operational."""
    print("=" * 75)
    print("STEP 1: ENVIRONMENT & DATASET AUDIT")
    print("=" * 75)
    print(f"Python executable: {sys.executable}")
    print(f"Python version:    {sys.version.split()[0]}")
    
    # Check OpenCV
    print(f"OpenCV version:    {cv2.__version__}")
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow ver:    {tf.__version__}")
    except ImportError as e:
        print(f"[ERROR] TensorFlow not found: {e}")
        sys.exit(1)
        
    # Check MediaPipe
    try:
        import mediapipe as mp
        print(f"MediaPipe ver:     {mp.__version__}")
    except ImportError as e:
        print(f"[ERROR] MediaPipe not found: {e}")
        sys.exit(1)
        
    print("[OK] Environment verification passed.\n")


def scan_and_sanitize_split(split_dir: Path, class_to_idx: dict, supported_exts=(".jpg", ".jpeg", ".png")):
    """
    Scans a split directory (e.g. Training, Validation, Testing),
    filters out corrupt / non-image text files, and returns valid records.
    """
    valid_records = []
    corrupted_files = []

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    for class_name in sorted(class_to_idx.keys()):
        class_folder = split_dir / class_name
        if not class_folder.exists() or not class_folder.is_dir():
            print(f"  [WARNING] Class directory missing: {class_folder}")
            continue

        for img_path in sorted(class_folder.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in supported_exts:
                continue

            # Check file size (filter scripts like 301.jpg which are < 1000 bytes)
            file_size = img_path.stat().st_size
            if file_size < 1000:
                corrupted_files.append((str(img_path), f"File too small ({file_size} bytes - likely text/script)"))
                continue

            # Quick image decode check
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0 or len(img.shape) != 3:
                corrupted_files.append((str(img_path), "Failed OpenCV image decode"))
                continue

            valid_records.append({
                "path": str(img_path.resolve()),
                "class_name": class_name,
                "class_idx": class_to_idx[class_name]
            })

    return pd.DataFrame(valid_records), corrupted_files


def scan_letters_reference(letters_dir: Path, class_to_idx: dict, supported_exts=(".jpg", ".jpeg", ".png")):
    """Scans canonical reference images in Letters/ (A.jpg to Z.jpg)."""
    valid_records = []
    corrupted_files = []

    if not letters_dir.exists():
        return pd.DataFrame(), []

    for img_path in sorted(letters_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in supported_exts:
            continue
        c_name = img_path.stem.upper()
        if c_name not in class_to_idx:
            continue

        file_size = img_path.stat().st_size
        if file_size < 1000:
            corrupted_files.append((str(img_path), f"File too small ({file_size} bytes)"))
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            corrupted_files.append((str(img_path), "Failed OpenCV decode"))
            continue

        valid_records.append({
            "path": str(img_path.resolve()),
            "class_name": c_name,
            "class_idx": class_to_idx[c_name]
        })

    return pd.DataFrame(valid_records), corrupted_files


def main():
    parser = argparse.ArgumentParser(description="Sanitize Real ISL Camera Dataset and Generate Manifests")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=r"C:\Users\hp\Real ISL Camera dataset",
        help="Path to external real camera dataset root"
    )
    parser.add_argument(
        "--output-splits-dir",
        type=str,
        default="data/splits",
        help="Directory to store generated manifest CSVs"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory to store class_mapping_real.json"
    )
    args = parser.parse_args()

    verify_environment()

    dataset_root = Path(args.dataset_dir)
    splits_dir = Path(args.output_splits_dir)
    artifacts_dir = Path(args.artifacts_dir)

    splits_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root:        {dataset_root}")
    print(f"Splits output dir:   {splits_dir}")
    print(f"Artifacts output:    {artifacts_dir}\n")

    # 1. Discover classes from Training directory
    train_dir = dataset_root / "Training"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found in {dataset_root}")

    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir() and d.name.isalpha()])
    num_classes = len(classes)
    print(f"Discovered {num_classes} classes: {classes}")
    assert num_classes == 26, f"Expected 26 classes (A-Z), found {num_classes}!"

    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {str(i): c for i, c in enumerate(classes)}

    mapping_payload = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_classes": num_classes,
        "classes": classes
    }

    mapping_path = artifacts_dir / "class_mapping_real.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping_payload, f, indent=2)
    print(f"[OK] Saved 26-class mapping to: {mapping_path}")

    # 2. Scan and sanitize all splits
    all_corrupted = []
    
    print("\nScanning Training split...")
    train_df, train_corrupt = scan_and_sanitize_split(dataset_root / "Training", class_to_idx)
    all_corrupted.extend(train_corrupt)
    print(f"  -> Valid Training images: {len(train_df):,}")

    print("Scanning Validation split...")
    val_df, val_corrupt = scan_and_sanitize_split(dataset_root / "Validation", class_to_idx)
    all_corrupted.extend(val_corrupt)
    print(f"  -> Valid Validation images: {len(val_df):,}")

    print("Scanning Testing split...")
    test_df, test_corrupt = scan_and_sanitize_split(dataset_root / "Testing", class_to_idx)
    all_corrupted.extend(test_corrupt)
    print(f"  -> Valid Testing images: {len(test_df):,}")

    print("Scanning Canonical Letters split...")
    letters_df, letters_corrupt = scan_letters_reference(dataset_root / "Letters", class_to_idx)
    all_corrupted.extend(letters_corrupt)
    print(f"  -> Valid Reference Letters: {len(letters_df)}")

    # 3. Report corrupted / excluded files
    print("\n" + "-" * 75)
    print(f"DATASET INTEGRITY AUDIT: Discovered {len(all_corrupted)} corrupted / non-image artifact files:")
    for path_str, reason in all_corrupted:
        print(f"  [EXCLUDED] {path_str} -> {reason}")
    print("-" * 75)

    # 4. Save manifest CSVs
    train_csv = splits_dir / "real_train_manifest.csv"
    val_csv = splits_dir / "real_val_manifest.csv"
    test_csv = splits_dir / "real_test_manifest.csv"
    letters_csv = splits_dir / "real_letters_manifest.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    letters_df.to_csv(letters_csv, index=False)

    print("\n[OK] Manifest CSVs written successfully:")
    print(f"  - Training:   {train_csv} ({len(train_df):,} rows)")
    print(f"  - Validation: {val_csv} ({len(val_df):,} rows)")
    print(f"  - Testing:    {test_csv} ({len(test_df):,} rows)")
    print(f"  - Letters:    {letters_csv} ({len(letters_df)} rows)")

    # 5. Split Verification & Assertions
    print("\nVerifying split integrity & class coverage:")
    for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        classes_in_split = df["class_name"].nunique()
        print(f"  {split_name:<5}: {len(df):>5} images across {classes_in_split:>2} classes. Range: [{df['class_name'].value_counts().min()} - {df['class_name'].value_counts().max()}] per class.")
        assert classes_in_split == 26, f"Missing classes in {split_name} split!"

    # Zero leakage assertion
    train_paths = set(train_df["path"])
    val_paths = set(val_df["path"])
    test_paths = set(test_df["path"])
    
    overlap_tv = train_paths.intersection(val_paths)
    overlap_tt = train_paths.intersection(test_paths)
    overlap_vt = val_paths.intersection(test_paths)

    assert len(overlap_tv) == 0, f"Data leakage between Train and Val ({len(overlap_tv)} files)!"
    assert len(overlap_tt) == 0, f"Data leakage between Train and Test ({len(overlap_tt)} files)!"
    assert len(overlap_vt) == 0, f"Data leakage between Val and Test ({len(overlap_vt)} files)!"
    print("  [PASSED] Exactly 0% data leakage across Train, Val, and Test splits.")

    print("\n" + "=" * 75)
    print("STEP 1 COMPLETE: DATASET SANITIZED AND READY FOR ROI PRE-EXTRACTION!")
    print("=" * 75)


if __name__ == "__main__":
    main()
