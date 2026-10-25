"""
src/evaluate_real_retrained.py - Step 4: Independent Evaluation of Retrained Real Camera Model
Evaluates artifacts/best_model_real.keras on:
  1. The 5,198 held-out real camera images in data/splits/real_test_manifest.csv
  2. The 26 canonical reference gesture images in data/splits/real_letters_manifest.csv

Generates:
  - reports/real_camera_retrained_metrics.json
  - reports/real_camera_retrained_per_class_metrics.csv
  - reports/real_camera_retrained_confusion_matrix.png
  - reports/real_camera_retrained_per_class_accuracy.png
"""

import os
import sys
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

from src.cache_real_dataset import _process_single_image

def main():
    start_time = time.time()
    artifacts_dir = Path("artifacts")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "best_model_real.keras"
    mapping_path = artifacts_dir / "class_mapping_real.json"
    test_manifest_path = Path("data/splits/real_test_manifest.csv")
    letters_manifest_path = Path("data/splits/real_letters_manifest.csv")

    print("=" * 80)
    print("STEP 4: INDEPENDENT EVALUATION OF RETRAINED REAL CAMERA MODEL")
    print(f"Model Checkpoint: {model_path}")
    print(f"Class Mapping:    {mapping_path}")
    print(f"Test Manifest:    {test_manifest_path}")
    print("=" * 80)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Class mapping not found: {mapping_path}")

    # 1. Load class mapping
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    classes = mapping["classes"]
    class_to_idx = mapping["class_to_idx"]
    idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
    num_classes = len(classes)

    # 2. Load model
    print(f"\n[1/4] Loading retrained model checkpoint...")
    model = tf.keras.models.load_model(str(model_path))

    # 3. Preprocess Test Set with MediaPipe Hand ROI pipeline
    test_df = pd.read_csv(test_manifest_path)
    total_test_samples = len(test_df)
    print(f"\n[2/4] Preprocessing {total_test_samples:,} unseen test images via MediaPipe ROI...")

    X_test = np.empty((total_test_samples, 64, 64, 1), dtype=np.float32)
    y_test = np.empty(total_test_samples, dtype=np.int32)
    hand_detected_count = 0

    task_args = [
        (i, row.path, row.class_idx, (64, 64), 50, 150)
        for i, row in enumerate(test_df.itertuples())
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        for idx, proc, class_idx, detected in tqdm(
            executor.map(_process_single_image, task_args),
            total=total_test_samples,
            desc="Preprocessing Test Set"
        ):
            X_test[idx] = proc.astype(np.float32) / 255.0
            y_test[idx] = class_idx
            if detected:
                hand_detected_count += 1

    print(f"MediaPipe detection success on test set: {hand_detected_count:,} / {total_test_samples:,} ({hand_detected_count/total_test_samples*100:.1f}%)")

    # 4. Run Model Predictions
    print(f"\n[3/4] Running forward inference across {total_test_samples:,} samples...")
    t_pred0 = time.time()
    preds_prob = model.predict(X_test, batch_size=128, verbose=1)
    t_pred1 = time.time()
    latency_ms = ((t_pred1 - t_pred0) / total_test_samples) * 1000.0

    y_pred = np.argmax(preds_prob, axis=1)
    y_conf = np.max(preds_prob, axis=1)

    overall_acc = float(accuracy_score(y_test, y_pred))
    mean_conf = float(np.mean(y_conf))

    print(f"\n" + "=" * 80)
    print(f"OVERALL RETRAINED MODEL ACCURACY: {overall_acc * 100:.2f}% ({np.sum(y_pred == y_test)} / {total_test_samples})")
    print(f"Mean Prediction Confidence:       {mean_conf * 100:.2f}%")
    print(f"Inference Latency:                {latency_ms:.2f} ms / frame ({1000.0/latency_ms:.1f} FPS)")
    print("=" * 80)

    # 5. Per-class metrics
    per_class_records = []
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    
    for c_idx, c_name in enumerate(classes):
        mask = (y_test == c_idx)
        n_c = int(np.sum(mask))
        if n_c > 0:
            c_correct = int(np.sum(y_pred[mask] == c_idx))
            c_acc = (c_correct / n_c) * 100.0
            c_conf = float(np.mean(y_conf[mask]))
        else:
            c_correct = 0
            c_acc = 0.0
            c_conf = 0.0
        per_class_records.append({
            "class_name": c_name,
            "total_samples": n_c,
            "correct": c_correct,
            "accuracy_percent": c_acc,
            "mean_confidence": c_conf
        })

    per_class_df = pd.DataFrame(per_class_records)
    csv_out = reports_dir / "real_camera_retrained_per_class_metrics.csv"
    per_class_df.to_csv(csv_out, index=False)
    print(f"\nPer-class metrics saved to: {csv_out}")

    # 6. Generate Visual Charts
    # A. Confusion Matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Retrained Real Camera ISL Confusion Matrix (Acc: {overall_acc*100:.2f}%)", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("True Class", fontsize=10)
    plt.tight_layout()
    cm_png = reports_dir / "real_camera_retrained_confusion_matrix.png"
    plt.savefig(cm_png, dpi=150)
    plt.close()

    # B. Per-class accuracy bar chart
    plt.figure(figsize=(12, 5))
    bars = plt.bar(per_class_df["class_name"], per_class_df["accuracy_percent"], color="#3b82f6", edgecolor="#1e3a8a")
    plt.axhline(overall_acc * 100, color="crimson", linestyle="--", linewidth=1.5, label=f"Overall Accuracy ({overall_acc*100:.1f}%)")
    plt.title("Per-Class Recognition Accuracy (Retrained Real Camera Model)", fontsize=12, fontweight="bold")
    plt.xlabel("ISL Letter", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    bar_png = reports_dir / "real_camera_retrained_per_class_accuracy.png"
    plt.savefig(bar_png, dpi=150)
    plt.close()

    # 7. Evaluate on canonical Letters/
    letters_correct = 0
    letters_total = 0
    letters_results = []
    if letters_manifest_path.exists():
        let_df = pd.read_csv(letters_manifest_path)
        print(f"\n[4/4] Evaluating {len(let_df)} Canonical Reference Letters in Letters/...")
        for _, row in let_df.iterrows():
            c_name = row["class_name"]
            c_idx = row["class_idx"]
            img = cv2.imread(row["path"])
            if img is None:
                continue
            # Process
            _, proc, _, _ = _process_single_image((0, row["path"], c_idx, (64, 64), 50, 150))
            inp = np.expand_dims(proc.astype(np.float32) / 255.0, axis=0)
            pred = model.predict(inp, verbose=0)[0]
            pred_idx = int(np.argmax(pred))
            pred_class = idx_to_class[pred_idx]
            conf = float(pred[pred_idx])
            is_corr = (pred_class == c_name)
            if is_corr:
                letters_correct += 1
            letters_total += 1
            letters_results.append({
                "letter": c_name,
                "predicted": pred_class,
                "confidence": conf,
                "is_correct": is_corr
            })
            mark = "PASS" if is_corr else "FAIL"
            print(f"  Letter {c_name}: Pred={pred_class:<2} ({conf*100:5.1f}%) [{mark}]")

    letters_acc = (letters_correct / letters_total) * 100.0 if letters_total > 0 else 0.0
    print(f"\nCanonical Letters Accuracy: {letters_acc:.1f}% ({letters_correct}/{letters_total})")

    # 8. Save Metrics JSON
    final_metrics = {
        "model": str(model_path),
        "test_samples": total_test_samples,
        "correct_predictions": int(np.sum(y_pred == y_test)),
        "overall_accuracy_percent": overall_acc * 100.0,
        "mean_confidence_percent": mean_conf * 100.0,
        "inference_latency_ms": latency_ms,
        "letters_accuracy_percent": letters_acc,
        "letters_correct": letters_correct,
        "letters_total": letters_total,
        "canonical_letters_results": letters_results
    }
    json_out = reports_dir / "real_camera_retrained_metrics.json"
    with open(json_out, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("STEP 4 COMPLETE: EVALUATION FINISHED!")
    print(f"Metrics saved to: {json_out}")
    print(f"Confusion Matrix: {cm_png}")
    print(f"Accuracy Chart:   {bar_png}")
    print("=" * 80)

if __name__ == "__main__":
    main()
