"""
src/audit_dataset.py - Fast & Thorough Stage 1 Dataset Audit for Indian Sign Language (ISL)
Audits the Kaggle dataset 'prathumarikeri/indian-sign-language-isl'.
Produces:
  - reports/dataset_report.json
  - reports/class_distribution.csv
  - reports/sample_images.png
  - artifacts/class_mapping.json
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import yaml
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import kagglehub

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_or_locate_dataset(raw_dir=None, dataset_slug=None):
    if raw_dir and Path(raw_dir).exists():
        print(f"[Stage 1] Using local dataset directory: {raw_dir}")
        return Path(raw_dir)
    print(f"[Stage 1] Downloading / locating dataset from KaggleHub: '{dataset_slug}'...")
    dataset_path = kagglehub.dataset_download(dataset_slug)
    print(f"[Stage 1] Dataset located at: {dataset_path}")
    return Path(dataset_path)

def compute_md5(file_path, chunk_size=65536):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def audit_dataset(root_path: Path, output_reports_dir: Path, artifacts_dir: Path, supported_exts=(".jpg", ".jpeg", ".png")):
    output_reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Stage 1] Auditing dataset directory: {root_path}")

    # Discover class directories
    # Check if 'Indian' exists or direct folders
    indian_sub = root_path / "Indian"
    if indian_sub.exists() and indian_sub.is_dir():
        active_root = indian_sub
    else:
        active_root = root_path

    active_class_dirs = sorted([d for d in active_root.iterdir() if d.is_dir()], key=lambda x: x.name)
    print(f"[Stage 1] Selected active class root: {active_root}")
    print(f"[Stage 1] Total class folders found: {len(active_class_dirs)}")

    class_distribution = []
    corrupt_files = []
    image_shapes = defaultdict(int)
    class_samples = {}
    
    # Store (size, file_path) to fast-filter duplicate candidates
    size_to_files = defaultdict(list)
    total_images = 0

    print("[Stage 1] Scanning images for corruption, resolutions, and file sizes...")
    for c_dir in tqdm(active_class_dirs, desc="Auditing classes"):
        c_name = c_dir.name
        img_files = [f for f in c_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_exts]
        valid_count = 0
        
        for img_path in img_files:
            total_images += 1
            try:
                # Fast header verification
                with Image.open(img_path) as img:
                    shape_key = f"{img.width}x{img.height}_{img.mode}"
                    image_shapes[shape_key] += 1
                
                f_size = os.path.getsize(img_path)
                size_to_files[f_size].append(str(img_path))
                valid_count += 1

                if c_name not in class_samples:
                    class_samples[c_name] = str(img_path)

            except Exception as e:
                corrupt_files.append({"file": str(img_path), "error": str(e), "class": c_name})

        class_distribution.append({
            "class_name": c_name,
            "image_count": valid_count,
            "corrupt_count": len([x for x in corrupt_files if x["class"] == c_name]),
            "path": str(c_dir)
        })

    # Class distribution dataframe
    df_dist = pd.DataFrame(class_distribution)
    csv_path = output_reports_dir / "class_distribution.csv"
    df_dist.to_csv(csv_path, index=False)
    print(f"[Stage 1] Class distribution saved to: {csv_path}")

    # Duplicates analysis: Only hash files that share identical byte sizes
    print("[Stage 1] Checking duplicate candidates among matching file sizes...")
    duplicate_groups = {}
    candidate_size_groups = [paths for size, paths in size_to_files.items() if len(paths) > 1]
    
    for paths in candidate_size_groups:
        hash_map = defaultdict(list)
        for p in paths:
            h = compute_md5(p)
            hash_map[h].append(p)
        for h, matching_paths in hash_map.items():
            if len(matching_paths) > 1:
                duplicate_groups[h] = matching_paths

    total_duplicate_files = sum(len(paths) - 1 for paths in duplicate_groups.values())
    print(f"[Stage 1] Exact duplicate files found: {total_duplicate_files} across {len(duplicate_groups)} distinct groups.")

    # Class Mapping (alphabetical / sorted integer IDs)
    classes = sorted([d["class_name"] for d in class_distribution])
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    idx_to_class = {idx: name for idx, name in enumerate(classes)}
    
    mapping_payload = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_classes": len(classes),
        "classes": classes
    }
    mapping_path = artifacts_dir / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping_payload, f, indent=2)
    print(f"[Stage 1] Class mapping saved to: {mapping_path}")

    # Save comprehensive audit report
    report = {
        "dataset_slug": "prathumarikeri/indian-sign-language-isl",
        "dataset_root": str(root_path),
        "class_root": str(active_root),
        "total_images_scanned": total_images,
        "valid_images": total_images - len(corrupt_files),
        "corrupt_images": len(corrupt_files),
        "corrupt_file_details": corrupt_files,
        "num_classes": len(classes),
        "classes": classes,
        "image_shapes_detected": dict(image_shapes),
        "exact_duplicate_groups": len(duplicate_groups),
        "redundant_duplicate_images": total_duplicate_files,
        "class_distribution_summary": {
            "min_images_per_class": int(df_dist["image_count"].min()) if not df_dist.empty else 0,
            "max_images_per_class": int(df_dist["image_count"].max()) if not df_dist.empty else 0,
            "mean_images_per_class": float(df_dist["image_count"].mean()) if not df_dist.empty else 0,
            "std_images_per_class": float(df_dist["image_count"].std()) if not df_dist.empty else 0
        }
    }

    report_path = output_reports_dir / "dataset_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Stage 1] Full dataset report saved to: {report_path}")

    # Generate sample grid image
    if class_samples:
        print("[Stage 1] Generating sample images grid...")
        cols = 7
        rows = int(np.ceil(len(class_samples) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
        axes = axes.flatten()

        for idx, (c_name, s_path) in enumerate(class_samples.items()):
            ax = axes[idx]
            img_bgr = cv2.imread(s_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
            ax.set_title(c_name, fontsize=11, fontweight="bold")
            ax.axis("off")

        # Hide any unused axes
        for j in range(idx + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        sample_img_path = output_reports_dir / "sample_images.png"
        plt.savefig(sample_img_path, dpi=150)
        plt.close()
        print(f"[Stage 1] Sample image grid saved to: {sample_img_path}")

    print("\n=======================================================")
    print(f"AUDIT SUMMARY: {len(classes)} classes | {total_images} images | {len(corrupt_files)} corrupt")
    print(f"Classes: {classes}")
    print("=======================================================\n")
    return report

if __name__ == "__main__":
    cfg = load_config()
    slug = cfg["dataset"].get("kaggle_slug")
    raw_dir = cfg["dataset"].get("raw_dir")
    dataset_dir = download_or_locate_dataset(raw_dir=raw_dir, dataset_slug=slug)
    audit_dataset(
        root_path=dataset_dir,
        output_reports_dir=Path(cfg["artifacts"]["reports_dir"]),
        artifacts_dir=Path(cfg["artifacts"]["model_dir"]),
        supported_exts=tuple(cfg["dataset"]["supported_extensions"])
    )
