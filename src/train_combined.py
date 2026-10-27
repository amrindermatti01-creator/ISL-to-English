"""
src/train_combined.py - Step 5: Combined Model Training (Real Camera + Mendeley Adults & Teenagers)
Merges Real Camera Dataset (18,198 train) with Mendeley ISL Dataset (~39,560 train)
Totaling ~57,758 diversified real-world sign language training samples across 26 alphabet classes (A-Z).

Features:
  1. Dataset unification: Concatenates real_train_64x64.npz and mendeley_train_64x64.npz.
  2. Multi-domain validation: Validates against a balanced mix of real-camera and Mendeley validation splits.
  3. Safe data augmentation (rotation +/-10 deg, translation +/-3px, zoom 0.95-1.05; NO horizontal flip).
  4. ModelCheckpoint saving best weights to artifacts/best_model_combined.keras.
  5. EarlyStopping with patience=5 restoring best weights.
  6. ReduceLROnPlateau learning rate decay.
  7. Detailed CSV logging to reports/combined_training_history.csv.
  8. Training curve visualization saved to reports/combined_training_curves.png.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from src.train_real import build_real_compact_cnn, plot_real_training_curves
from src.train_mendeley import create_augmented_dataset


def main():
    parser = argparse.ArgumentParser(description="Train unified CNN on combined Real Camera + Mendeley ISL dataset.")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial Adam learning rate")
    args = parser.parse_args()

    processed_dir = PROJECT_ROOT / "data/processed"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    reports_dir = PROJECT_ROOT / "reports"

    real_train_cache = processed_dir / "real_train_64x64.npz"
    real_val_cache = processed_dir / "real_val_64x64.npz"
    mendeley_train_cache = processed_dir / "mendeley_train_64x64.npz"
    mendeley_val_cache = processed_dir / "mendeley_val_64x64.npz"

    if not real_train_cache.exists() or not mendeley_train_cache.exists():
        print("[ERROR] Required dataset caches missing. Ensure both Real Camera and Mendeley caches exist.")
        sys.exit(1)

    print("=" * 75)
    print("STEP 5: UNIFIED COMBINED MODEL TRAINING (REAL CAMERA + MENDELEY ISL)")
    print("=" * 75)

    # 1. Load Real Camera dataset
    print(f"Loading Real Camera train cache: {real_train_cache.name} ...")
    r_tr = np.load(real_train_cache)
    x_tr_k = "x" if "x" in r_tr else "X"
    x_real_tr, y_real_tr = r_tr[x_tr_k], r_tr["y"]
    r_val = np.load(real_val_cache)
    x_val_k = "x" if "x" in r_val else "X"
    x_real_val, y_real_val = r_val[x_val_k], r_val["y"]

    # 2. Load Mendeley ISL dataset
    print(f"Loading Mendeley ISL train cache:  {mendeley_train_cache.name} ...")
    m_tr = np.load(mendeley_train_cache)
    x_m_k = "x" if "x" in m_tr else "X"
    x_mend_tr, y_mend_tr = m_tr[x_m_k], m_tr["y"]
    m_val = np.load(mendeley_val_cache)
    x_mv_k = "x" if "x" in m_val else "X"
    x_mend_val, y_mend_val = m_val[x_mv_k], m_val["y"]

    # Remap Real Camera labels (0-25 -> 36-class scheme)
    with open(artifacts_dir / "class_mapping_real.json", "r") as f:
        real_classes = json.load(f)["classes"]
    with open(artifacts_dir / "class_mapping_mendeley.json", "r") as f:
        mend_map = json.load(f)
        mend_c2i = mend_map["class_to_idx"]
        num_classes = mend_map["num_classes"]

    remap_lut = np.array([mend_c2i[c] for c in real_classes], dtype=np.int32)
    y_real_tr = remap_lut[y_real_tr]
    y_real_val = remap_lut[y_real_val]

    # Ensure consistent 3D shape (N, 64, 64) before concatenation
    if x_real_tr.ndim == 4:
        x_real_tr = np.squeeze(x_real_tr, axis=-1)
    if x_real_val.ndim == 4:
        x_real_val = np.squeeze(x_real_val, axis=-1)
    if x_mend_tr.ndim == 4:
        x_mend_tr = np.squeeze(x_mend_tr, axis=-1)
    if x_mend_val.ndim == 4:
        x_mend_val = np.squeeze(x_mend_val, axis=-1)

    # 3. Concatenate Training and Validation
    x_train = np.concatenate([x_real_tr, x_mend_tr], axis=0)
    y_train = np.concatenate([y_real_tr, y_mend_tr], axis=0)

    x_val = np.concatenate([x_real_val, x_mend_val], axis=0)
    y_val = np.concatenate([y_real_val, y_mend_val], axis=0)

    # Shuffle training set to interleave domains
    np.random.seed(42)
    perm = np.random.permutation(len(x_train))
    x_train = x_train[perm]
    y_train = y_train[perm]

    print("\nCombined Dataset Summary:")
    print(f"  Real Camera Train Samples:    {len(x_real_tr):,}")
    print(f"  Mendeley ISL Train Samples:   {len(x_mend_tr):,}")
    print(f"  TOTAL Unified Train Samples:  {len(x_train):,} (shape: {x_train.shape})")
    print(f"  TOTAL Unified Val Samples:    {len(x_val):,} (shape: {x_val.shape})")

    # Save cached combined npz
    comb_train_cache = processed_dir / "combined_train_64x64.npz"
    comb_val_cache = processed_dir / "combined_val_64x64.npz"
    np.savez_compressed(comb_train_cache, x=x_train, y=y_train)
    np.savez_compressed(comb_val_cache, x=x_val, y=y_val)
    print(f"  Cached combined datasets saved to: {comb_train_cache.name} & {comb_val_cache.name}")

    # Build model (36 classes)
    print(f"\nBuilding Compact CNN for {num_classes} classes...")
    model = build_real_compact_cnn(input_shape=(64, 64, 1), num_classes=num_classes, learning_rate=args.lr)
    model.summary()

    train_ds = create_augmented_dataset(x_train, y_train, batch_size=args.batch_size, is_training=True)
    val_ds = create_augmented_dataset(x_val, y_val, batch_size=args.batch_size, is_training=False)

    model_checkpoint_path = artifacts_dir / "best_model_combined.keras"
    history_csv_path = reports_dir / "combined_training_history.csv"
    training_curves_path = reports_dir / "combined_training_curves.png"
    config_path = artifacts_dir / "training_config_combined.json"

    callbacks = [
        ModelCheckpoint(
            filepath=str(model_checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        CSVLogger(str(history_csv_path), separator=",", append=False)
    ]

    print("\nStarting combined training loop...")
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )
    train_time = time.time() - t0
    print(f"\n[OK] Combined training completed in {train_time:.1f}s ({train_time/60:.2f} min).")

    # Save training curves
    try:
        plot_real_training_curves(history_csv_path, training_curves_path)
        print(f"[SAVED] Combined training curves plot saved to: {training_curves_path}")
    except Exception as e:
        print(f"[WARNING] Could not plot training curves: {e}")

    # Best metrics
    best_val_acc = float(np.max(history.history["val_accuracy"]))
    best_val_loss = float(np.min(history.history["val_loss"]))
    print(f"  Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"  Best Validation Loss:     {best_val_loss:.4f}")

    # Evaluate best model on test splits
    best_model = tf.keras.models.load_model(str(model_checkpoint_path))
    mend_test_cache = processed_dir / "mendeley_test_64x64.npz"
    real_test_cache = processed_dir / "real_test_64x64.npz"

    mend_test_acc = None
    if mend_test_cache.exists():
        print(f"\nEvaluating on Mendeley test set ({mend_test_cache.name})...")
        d_test = np.load(mend_test_cache)
        x_m_k = "x" if "x" in d_test else "X"
        x_m_test = np.expand_dims(d_test[x_m_k], axis=-1).astype(np.float32) / 255.0
        y_m_test = d_test["y"].astype(np.int32)
        ev_m = best_model.evaluate(x_m_test, y_m_test, batch_size=args.batch_size, verbose=1)
        mend_test_acc = float(ev_m[1])
        print(f"  >>> COMBINED MODEL ON MENDELEY TEST: {mend_test_acc * 100:.2f}% <<<")

    real_test_acc = None
    if real_test_cache.exists():
        print(f"\nEvaluating on Real Camera test set ({real_test_cache.name})...")
        d_r_test = np.load(real_test_cache)
        x_r_k = "x" if "x" in d_r_test else "X"
        x_raw = d_r_test[x_r_k].astype(np.float32) / 255.0
        x_r_test = np.expand_dims(x_raw, axis=-1) if x_raw.ndim == 3 else x_raw
        y_r_test = d_r_test["y"].astype(np.int32)
        # Remap real camera test labels (0-25 -> 10-35)
        y_r_test_36 = remap_lut[y_r_test]
        ev_r = best_model.evaluate(x_r_test, y_r_test_36, batch_size=args.batch_size, verbose=1)
        real_test_acc = float(ev_r[1])
        print(f"  >>> COMBINED MODEL ON REAL CAMERA TEST: {real_test_acc * 100:.2f}% <<<")

    # Save metadata
    config = {
        "dataset": "Combined (Real ISL Camera + Mendeley Adults & Teenagers)",
        "num_classes": num_classes,
        "input_shape": [64, 64, 1],
        "real_train_samples": int(len(x_real_tr)),
        "mendeley_train_samples": int(len(x_mend_tr)),
        "total_train_samples": int(len(x_train)),
        "total_val_samples": int(len(x_val)),
        "batch_size": args.batch_size,
        "epochs_trained": len(history.history["accuracy"]),
        "best_val_accuracy": round(best_val_acc * 100, 2),
        "best_val_loss": round(best_val_loss, 4),
        "mendeley_test_accuracy": round(mend_test_acc * 100, 2) if mend_test_acc is not None else None,
        "real_test_accuracy": round(real_test_acc * 100, 2) if real_test_acc is not None else None,
        "training_duration_seconds": round(train_time, 1),
        "model_path": str(model_checkpoint_path.resolve())
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[SAVED] Combined training configuration metadata saved to: {config_path}")


if __name__ == "__main__":
    main()
