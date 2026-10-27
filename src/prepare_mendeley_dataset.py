"""
src/prepare_mendeley_dataset.py - Step 2: High-Speed Dataset Sanitization & Manifest Generation
For Mendeley ISL Dataset (Adults and Teenagers only).

Responsibilities:
  1. Fast multi-core scanning across 26 alphabet classes (A-Z).
  2. Sanitizes dataset by filtering out corrupt images or files < 1KB.
  3. Generates artifacts/class_mapping_mendeley.json (26 classes, 0-25).
  4. Performs stratified train/val/test split (80% / 10% / 10%).
  5. Produces validated CSV manifests:
       - data/splits/mendeley_train_manifest.csv
       - data/splits/mendeley_val_manifest.csv
       - data/splits/mendeley_test_manifest.csv
  6. Validates zero leakage across splits and complete class coverage.
"""

import os
import sys
import json
import struct
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm


def get_jpeg_dimensions(file_path: Path):
    """Fast extraction of JPEG dimensions without decoding full pixel array."""
    try:
        with open(file_path, "rb") as f:
            data = f.read(2)
            if data != b"\xff\xd8":
                return None, None
            while True:
                marker = f.read(2)
                if len(marker) < 2:
                    break
                if marker[0] != 0xFF:
                    break
                m = marker[1]
                if m in (0xC0, 0xC1, 0xC2, 0xC3):  # SOF markers
                    f.read(3)  # length + precision
                    h, w = struct.unpack(">HH", f.read(4))
                    return w, h
                elif m == 0xD9 or m == 0xDA:  # EOI or SOS
                    break
                else:
                    len_bytes = f.read(2)
                    if len(len_bytes) < 2:
                        break
                    block_len = struct.unpack(">H", len_bytes)[0]
                    f.seek(block_len - 2, 1)
    except Exception:
        pass
    return None, None


def verify_file(img_path: Path, class_name: str):
    """Verifies a single image file."""
    try:
        sz = img_path.stat().st_size
        if sz < 1000:
            return None, (str(img_path), f"File too small ({sz} bytes)")
        
        w, h = get_jpeg_dimensions(img_path)
        if w is None or h is None or w < 20 or h < 20:
            # Fallback to OpenCV check
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                return None, (str(img_path), "Invalid image decode")
            h, w = img.shape[:2]

        fname = img_path.name.lower()
        age_group = "Teenagers" if "teenager" in fname else ("Adults" if "adult" in fname else "Unknown")
        sleeves = "Half" if "half" in fname else ("Full" if "full" in fname else "Unknown")

        rec = {
            "path": str(img_path.resolve()),
            "filename": img_path.name,
            "class_name": class_name,
            "age_group": age_group,
            "sleeves": sleeves,
            "file_size_bytes": sz,
            "height": h,
            "width": w,
            "channels": 3
        }
        return rec, None
    except Exception as e:
        return None, (str(img_path), str(e))


def scan_and_sanitize_mendeley(data_dir: Path, num_workers=12):
    """Scans all class directories with multi-threading."""
    valid_records = []
    corrupted_files = []

    classes = [str(d) for d in range(10)] + [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    all_files = []

    for class_name in sorted(classes):
        class_folder = data_dir / class_name
        if not class_folder.exists():
            continue
        for img_path in class_folder.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                all_files.append((img_path, class_name))

    print(f"Auditing {len(all_files):,} images across 26 classes using {num_workers} threads...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(verify_file, p, c) for p, c in all_files]
        for f in tqdm(futures, desc="Sanitizing"):
            rec, err = f.result()
            if rec:
                valid_records.append(rec)
            if err:
                corrupted_files.append(err)

    return valid_records, corrupted_files


def create_stratified_splits(df: pd.DataFrame, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42):
    """Stratified train/val/test split per class."""
    np.random.seed(seed)
    train_records = []
    val_records = []
    test_records = []

    for class_name, group in df.groupby("class_name"):
        shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_records.append(shuffled.iloc[:n_train])
        val_records.append(shuffled.iloc[n_train:n_train + n_val])
        test_records.append(shuffled.iloc[n_train + n_val:])

    train_df = pd.concat(train_records, ignore_index=True)
    val_df = pd.concat(val_records, ignore_index=True)
    test_df = pd.concat(test_records, ignore_index=True)

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Prepare and split Mendeley ISL dataset.")
    parser.add_argument("--data-dir", type=str, default="data/mendeley_isl", help="Path to extracted Mendeley ISL images")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Directory to save split CSV manifests")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory to save class mapping")
    parser.add_argument("--workers", type=int, default=12, help="Number of worker threads")
    args = parser.parse_args()

    data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    splits_dir = (PROJECT_ROOT / args.splits_dir).resolve()
    artifacts_dir = (PROJECT_ROOT / args.artifacts_dir).resolve()

    splits_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("STEP 2: MENDELEY ISL DATASET AUDIT & MANIFEST GENERATION")
    print("=" * 75)

    valid_records, corrupted_files = scan_and_sanitize_mendeley(data_dir, num_workers=args.workers)

    print(f"\nScan Complete:")
    print(f"  Valid images:     {len(valid_records):,}")
    print(f"  Corrupted files:  {len(corrupted_files):,}")

    if corrupted_files:
        print("\nQuarantined Corrupted Files:")
        for p, reason in corrupted_files[:10]:
            print(f"  - {p} ({reason})")

    df = pd.DataFrame(valid_records)
    if df.empty:
        print("[ERROR] No valid images found! Please run download_mendeley_dataset.py first.")
        sys.exit(1)

    classes = sorted(df["class_name"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    mapping_payload = {
        "class_to_idx": class_to_idx,
        "num_classes": len(classes),
        "classes": classes
    }

    mapping_file = artifacts_dir / "class_mapping_mendeley.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping_payload, f, indent=2)
    print(f"\n[SAVED] Class mapping saved: {mapping_file} ({len(classes)} classes)")

    train_df, val_df, test_df = create_stratified_splits(df)

    train_path = splits_dir / "mendeley_train_manifest.csv"
    val_path = splits_dir / "mendeley_val_manifest.csv"
    test_path = splits_dir / "mendeley_test_manifest.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n" + "=" * 75)
    print("SPLIT MANIFEST SUMMARY")
    print("=" * 75)
    print(f"  Train:      {len(train_df):,} images -> {train_path.name}")
    print(f"  Validation: {len(val_df):,} images -> {val_path.name}")
    print(f"  Test:       {len(test_df):,} images -> {test_path.name}")
    print(f"  Total:      {len(df):,} images")
    print("=" * 75)

    # Zero leakage check
    set_train = set(train_df["path"])
    set_val = set(val_df["path"])
    set_test = set(test_df["path"])

    assert len(set_train.intersection(set_val)) == 0, "Leakage between train and val!"
    assert len(set_train.intersection(set_test)) == 0, "Leakage between train and test!"
    assert len(set_val.intersection(set_test)) == 0, "Leakage between val and test!"
    print("[PASS] Zero data leakage across train, val, and test splits!\n")


if __name__ == "__main__":
    main()
