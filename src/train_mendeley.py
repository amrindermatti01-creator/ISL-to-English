"""
src/train_mendeley.py - Step 4: High-Performance CNN Training on Mendeley ISL Dataset
Trains on cached 64x64 MediaPipe Hand ROI Canny edge features from Adults & Teenagers across 26 alphabet classes (A-Z).

Features:
  1. Fast in-memory training from RAM.
  2. Native 26-class output head.
  3. Safe data augmentation (rotation +/-10 deg, translation +/-3px, zoom 0.95-1.05; NO horizontal flip).
  4. ModelCheckpoint saving best weights to artifacts/best_model_mendeley.keras.
  5. EarlyStopping with patience=5 restoring best weights.
  6. ReduceLROnPlateau learning rate decay.
  7. Detailed CSV logging to reports/mendeley_training_history.csv.
  8. Training curve visualization saved to reports/mendeley_training_curves.png.
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
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from src.train_real import build_real_compact_cnn, plot_real_training_curves


def create_augmented_dataset(x_data, y_data, batch_size=64, is_training=True):
    """Creates a tf.data pipeline with cached arrays and safe geometric augmentation."""
    x_tensor = np.expand_dims(x_data, axis=-1).astype(np.float32) / 255.0
    y_tensor = y_data.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(x_data), 10000), seed=42, reshuffle_each_iteration=True)

        data_augmentation = tf.keras.Sequential([
            layers.RandomRotation(factor=(-0.03, 0.03), fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05), fill_mode="constant", fill_value=0.0),
            layers.RandomZoom(height_factor=(-0.05, 0.05), fill_mode="constant", fill_value=0.0)
        ], name="safe_augmentation")

        ds = ds.batch(batch_size)
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch_size)

    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser(description="Train CNN on Mendeley ISL dataset (Adults & Teenagers only).")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial Adam learning rate")
    args = parser.parse_args()

    processed_dir = PROJECT_ROOT / "data/processed"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    reports_dir = PROJECT_ROOT / "reports"

    train_cache = processed_dir / "mendeley_train_64x64.npz"
    val_cache = processed_dir / "mendeley_val_64x64.npz"

    if not train_cache.exists() or not val_cache.exists():
        print(f"[ERROR] Processed data not found. Please run src/cache_mendeley_dataset.py first.")
        sys.exit(1)

    print("=" * 75)
    print("STEP 4: MENDELEY ISL CNN TRAINING (ADULTS & TEENAGERS ONLY)")
    print("=" * 75)

    print(f"Loading training cache:   {train_cache.name} ...")
    data_train = np.load(train_cache)
    x_train, y_train = data_train["x"], data_train["y"]

    print(f"Loading validation cache: {val_cache.name} ...")
    data_val = np.load(val_cache)
    x_val, y_val = data_val["x"], data_val["y"]

    print(f"  Train set: {len(x_train):,} samples, shape: {x_train.shape}")
    print(f"  Val set:   {len(x_val):,} samples, shape: {x_val.shape}")

    mapping_file = artifacts_dir / "class_mapping_mendeley.json"
    if mapping_file.exists():
        with open(mapping_file, "r", encoding="utf-8") as f:
            num_classes = json.load(f)["num_classes"]
    else:
        num_classes = int(max(np.max(y_train), np.max(y_val)) + 1)
    print(f"  Vocabulary: {num_classes} classes")
    model = build_real_compact_cnn(input_shape=(64, 64, 1), num_classes=num_classes, learning_rate=args.lr)
    model.summary()

    train_ds = create_augmented_dataset(x_train, y_train, batch_size=args.batch_size, is_training=True)
    val_ds = create_augmented_dataset(x_val, y_val, batch_size=args.batch_size, is_training=False)

    model_checkpoint_path = artifacts_dir / "best_model_mendeley.keras"
    history_csv_path = reports_dir / "mendeley_training_history.csv"
    training_curves_path = reports_dir / "mendeley_training_curves.png"
    config_path = artifacts_dir / "training_config_mendeley.json"

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

    print("\nStarting training loop...")
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )
    train_time = time.time() - t0
    print(f"\n[OK] Training completed in {train_time:.1f}s ({train_time/60:.2f} min).")

    # Save training curves
    plot_real_training_curves(history_csv_path, training_curves_path)
    print(f"[SAVED] Training curves plot saved to: {training_curves_path}")

    # Best metrics
    best_val_acc = float(np.max(history.history["val_accuracy"]))
    best_val_loss = float(np.min(history.history["val_loss"]))
    print(f"  Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"  Best Validation Loss:     {best_val_loss:.4f}")

    # Evaluate on held-out test split
    test_cache = processed_dir / "mendeley_test_64x64.npz"
    test_acc = None
    if test_cache.exists():
        print(f"\nEvaluating on held-out Mendeley test set ({test_cache.name})...")
        best_model = tf.keras.models.load_model(str(model_checkpoint_path))
        d_test = np.load(test_cache)
        x_test = np.expand_dims(d_test["x"], axis=-1).astype(np.float32) / 255.0
        y_test = d_test["y"].astype(np.int32)
        eval_res = best_model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=1)
        test_acc = float(eval_res[1])
        print(f"  >>> MENDELEY TEST ACCURACY: {test_acc * 100:.2f}% (Loss: {eval_res[0]:.4f}) <<<")

    # Save metadata
    config = {
        "dataset": "Mendeley ISL (Adults + Teenagers ONLY)",
        "num_classes": num_classes,
        "input_shape": [64, 64, 1],
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "batch_size": args.batch_size,
        "epochs_trained": len(history.history["accuracy"]),
        "best_val_accuracy": round(best_val_acc * 100, 2),
        "best_val_loss": round(best_val_loss, 4),
        "test_accuracy": round(test_acc * 100, 2) if test_acc is not None else None,
        "training_duration_seconds": round(train_time, 1),
        "model_path": str(model_checkpoint_path.resolve())
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[SAVED] Training configuration metadata saved to: {config_path}")


if __name__ == "__main__":
    main()
