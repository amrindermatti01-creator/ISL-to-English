"""
src/train.py - Stage 3 CNN Training Pipeline
High-Performance In-Memory Training with Cache Verification
Supports:
  - Automatic pre-caching to data/processed/*.npz (processes images once)
  - Lightning-fast training from RAM (AVX2/FMA accelerated)
  - ModelCheckpoint saving best model to artifacts/best_model.keras
  - EarlyStopping with patience 5 (restores best weights)
  - Learning rate reduction via ReduceLROnPlateau
  - Real-time CSV logging to reports/training_history.csv
  - Training curves plotting to reports/training_curves.png
  - Training configuration artifact saving to artifacts/training_config.json
  - Reloading checkpoint verification test
"""

import os
import sys
import time
import json
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from src.model import get_model
from src.data_loader import cache_split_to_npz, load_cached_split

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def plot_training_curves(history_csv_path, output_png_path):
    df = pd.read_csv(history_csv_path)
    epochs = range(1, len(df) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy Curve
    ax1.plot(epochs, df["accuracy"], "b-o", label="Training Accuracy", linewidth=2)
    ax1.plot(epochs, df["val_accuracy"], "r-s", label="Validation Accuracy", linewidth=2)
    ax1.set_title("ISL Model Accuracy Over Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # Loss Curve
    ax2.plot(epochs, df["loss"], "b-o", label="Training Loss", linewidth=2)
    ax2.plot(epochs, df["val_loss"], "r-s", label="Validation Loss", linewidth=2)
    ax2.set_title("ISL Model Loss Over Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    Path(output_png_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png_path, dpi=150)
    plt.close()
    print(f"[Stage 3] Training curves saved to: {output_png_path}")

def train_model(architecture="compact", epochs=5, batch_size=64):
    cfg = load_config()
    artifacts_dir = Path(cfg["artifacts"]["model_dir"])
    reports_dir = Path(cfg["artifacts"]["reports_dir"])
    splits_dir = Path(cfg["dataset"]["splits_dir"])
    processed_dir = Path(cfg["dataset"]["processed_dir"])

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_csv = splits_dir / "train_manifest.csv"
    val_csv = splits_dir / "val_manifest.csv"

    # Class mapping
    class_map_file = artifacts_dir / "class_mapping.json"
    with open(class_map_file, "r") as f:
        class_mapping = json.load(f)
    num_classes = class_mapping["num_classes"]

    p_cfg = cfg["preprocessing"]
    target_size = tuple(p_cfg["target_size"])

    train_npz = processed_dir / f"train_{target_size[0]}x{target_size[1]}.npz"
    val_npz = processed_dir / f"val_{target_size[0]}x{target_size[1]}.npz"

    # Ensure preprocessed cache exists
    if not train_npz.exists():
        cache_split_to_npz(
            train_csv, train_npz,
            target_size=target_size,
            threshold_value=p_cfg["threshold_value"],
            use_canny=p_cfg["use_canny"],
            canny_low=p_cfg["canny_low"],
            canny_high=p_cfg["canny_high"]
        )
    if not val_npz.exists():
        cache_split_to_npz(
            val_csv, val_npz,
            target_size=target_size,
            threshold_value=p_cfg["threshold_value"],
            use_canny=p_cfg["use_canny"],
            canny_low=p_cfg["canny_low"],
            canny_high=p_cfg["canny_high"]
        )

    # Load high-speed RAM arrays
    X_train, y_train = load_cached_split(train_npz)
    X_val, y_val = load_cached_split(val_npz)

    print(f"\n=======================================================")
    print(f"[Stage 3] STARTING IN-MEMORY CNN TRAINING")
    print(f"  Architecture:     {architecture.upper()}")
    print(f"  Input Shape:      ({target_size[0]}, {target_size[1]}, 1)")
    print(f"  Number of Classes: {num_classes}")
    print(f"  Batch Size:       {batch_size}")
    print(f"  Max Epochs:       {epochs}")
    print(f"  Train Samples:    {len(X_train)}")
    print(f"  Val Samples:      {len(X_val)}")
    print(f"=======================================================\n")

    # Build model
    model = get_model(
        architecture=architecture,
        input_shape=(target_size[0], target_size[1], 1),
        num_classes=num_classes,
        learning_rate=cfg["training"]["learning_rate"]
    )
    model.summary()

    # Callbacks
    best_model_path = artifacts_dir / "best_model.keras"
    history_csv = reports_dir / "training_history.csv"

    checkpoint_cb = ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )

    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        patience=cfg["training"]["patience_early_stopping"],
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr_cb = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    csv_logger_cb = CSVLogger(str(history_csv), append=False)
    callbacks = [checkpoint_cb, early_stopping_cb, reduce_lr_cb, csv_logger_cb]

    # Training
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )
    elapsed_time = time.time() - start_time
    print(f"\n[Stage 3] Training finished in {elapsed_time/60:.2f} minutes ({elapsed_time:.1f}s)")

    # Plot curves
    curves_png = reports_dir / "training_curves.png"
    plot_training_curves(history_csv, curves_png)

    # Save training configuration artifact
    final_train_acc = float(history.history["accuracy"][-1])
    final_val_acc = float(history.history["val_accuracy"][-1])
    best_val_loss = float(min(history.history["val_loss"]))
    best_val_acc = float(max(history.history["val_accuracy"]))

    training_config = {
        "architecture": architecture,
        "input_shape": [target_size[0], target_size[1], 1],
        "num_classes": num_classes,
        "batch_size": batch_size,
        "epochs_run": len(history.history["loss"]),
        "elapsed_seconds": round(elapsed_time, 2),
        "elapsed_minutes": round(elapsed_time / 60, 2),
        "final_train_acc": round(final_train_acc, 4),
        "final_val_acc": round(final_val_acc, 4),
        "best_val_acc": round(best_val_acc, 4),
        "best_val_loss": round(best_val_loss, 4),
        "best_model_path": str(best_model_path),
        "history_csv": str(history_csv),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    config_artifact = artifacts_dir / "training_config.json"
    with open(config_artifact, "w") as f:
        json.dump(training_config, f, indent=2)
    print(f"[Stage 3] Training configuration saved to: {config_artifact}")

    # Reload checkpoint validation
    print("[Stage 3] Validating best model reloadability...")
    reloaded_model = tf.keras.models.load_model(str(best_model_path))
    sample_preds = reloaded_model.predict(X_val[:5], verbose=0)
    assert sample_preds.shape == (5, num_classes), f"Unexpected output shape: {sample_preds.shape}"
    predicted_classes = np.argmax(sample_preds, axis=1)
    print(f"[Stage 3] Reload validation PASSED! Predicted labels: {predicted_classes.tolist()}, Actual: {y_val[:5].tolist()}")

    return training_config

if __name__ == "__main__":
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Stage 3 ISL CNN Training")
    parser.add_argument("--arch", type=str, default="compact", choices=["compact", "paper"])
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    train_model(
        architecture=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
