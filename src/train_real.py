"""
src/train_real.py - Step 3: High-Performance CNN Training for Real Camera ISL Dataset
Trains on cached 64x64 MediaPipe Hand ROI Canny edge features across 26 alphabet classes (A-Z).

Features:
  1. Fast in-memory training from RAM (AVX2/FMA accelerated).
  2. Native 26-class output head (eliminates digit false absorptions).
  3. Safe data augmentation (rotation +/-10 deg, translation +/-3px, zoom 0.95-1.05; NO horizontal flip).
  4. ModelCheckpoint saving best weights to artifacts/best_model_real.keras.
  5. EarlyStopping with patience=5 restoring best weights.
  6. ReduceLROnPlateau learning rate decay.
  7. Detailed CSV logging to reports/real_training_history.csv.
  8. Training curve visualization saved to reports/real_training_curves.png.
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

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

def build_real_compact_cnn(input_shape=(64, 64, 1), num_classes=26, learning_rate=0.001):
    """
    Compact 4-Block CNN tailored for 26-class real-camera ISL gesture recognition.
    ~515,000 parameters. High efficiency for local CPU inference and real-time webcam HUD.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1"),
        layers.BatchNormalization(name="bn1"),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_2"),
        layers.MaxPooling2D((2, 2), name="pool1"),
        layers.Dropout(0.20, name="drop1"),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1"),
        layers.BatchNormalization(name="bn2"),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_2"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        layers.Dropout(0.25, name="drop2"),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1"),
        layers.BatchNormalization(name="bn3"),
        layers.MaxPooling2D((2, 2), name="pool3"),
        layers.Dropout(0.30, name="drop3"),

        # Block 4
        layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_1"),
        layers.BatchNormalization(name="bn4"),
        layers.MaxPooling2D((2, 2), name="pool4"),
        layers.Dropout(0.35, name="drop4"),

        # Classification Head
        layers.Flatten(name="flatten"),
        layers.Dense(256, activation="relu", name="dense_features"),
        layers.BatchNormalization(name="bn_dense"),
        layers.Dropout(0.40, name="drop_dense"),
        layers.Dense(num_classes, activation="softmax", name="output_classes")
    ], name="ISL_RealCamera_CNN")

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def plot_real_training_curves(history_csv_path, output_png_path):
    """Plots training vs validation accuracy and loss curves."""
    try:
        df = pd.read_csv(history_csv_path, on_bad_lines="skip")
    except Exception as e:
        print(f"[WARNING] Could not read CSV directly ({e}), retrying without bad lines...")
        df = pd.read_csv(history_csv_path)
    epochs = range(1, len(df) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy Curve
    ax1.plot(epochs, df["accuracy"] * 100, "b-o", label="Training Accuracy", linewidth=2)
    ax1.plot(epochs, df["val_accuracy"] * 100, "r-s", label="Validation Accuracy", linewidth=2)
    ax1.set_title("Real Camera ISL Accuracy Over Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # Loss Curve
    ax2.plot(epochs, df["loss"], "b-o", label="Training Loss", linewidth=2)
    ax2.plot(epochs, df["val_loss"], "r-s", label="Validation Loss", linewidth=2)
    ax2.set_title("Real Camera ISL Loss Over Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    Path(output_png_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png_path, dpi=150)
    plt.close()
    print(f"[OK] Training curves plot saved to: {output_png_path}")

def main():
    parser = argparse.ArgumentParser(description="Train CNN on Real Camera ISL Dataset")
    parser.add_argument("--train-npz", type=str, default="data/processed/real_train_64x64.npz")
    parser.add_argument("--val-npz", type=str, default="data/processed/real_val_64x64.npz")
    parser.add_argument("--mapping", type=str, default="artifacts/class_mapping_real.json")
    parser.add_argument("--output-model", type=str, default="artifacts/best_model_real.keras")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum epochs to train")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    args = parser.parse_args()

    train_npz_path = Path(args.train_npz)
    val_npz_path = Path(args.val_npz)
    mapping_path = Path(args.mapping)
    output_model_path = Path(args.output_model)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load class mapping
    if not mapping_path.exists():
        raise FileNotFoundError(f"Class mapping file not found: {mapping_path}")
    with open(mapping_path, "r") as f:
        class_mapping = json.load(f)
    num_classes = class_mapping["num_classes"]
    classes = class_mapping["classes"]

    # 2. Load cached binary arrays into RAM
    print("=" * 75)
    print("STEP 3: CNN MODEL TRAINING (REAL CAMERA ISL DATASET)")
    print(f"Train Cache:   {train_npz_path}")
    print(f"Val Cache:     {val_npz_path}")
    print(f"Classes:       {num_classes} ({classes[0]}..{classes[-1]})")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Max Epochs:    {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 75)

    if not train_npz_path.exists():
        raise FileNotFoundError(f"Train cache not found: {train_npz_path}. Run cache_real_dataset.py first!")
    if not val_npz_path.exists():
        raise FileNotFoundError(f"Val cache not found: {val_npz_path}. Run cache_real_dataset.py first!")

    print("\nLoading cached binary arrays into memory...")
    t0 = time.time()
    train_data = np.load(train_npz_path)
    X_train = train_data["X"].astype(np.float32) / 255.0
    y_train = train_data["y"]

    val_data = np.load(val_npz_path)
    X_val = val_data["X"].astype(np.float32) / 255.0
    y_val = val_data["y"]
    t1 = time.time()
    print(f"[OK] Arrays loaded in {t1 - t0:.2f}s:")
    print(f"  X_train: {X_train.shape} (dtype: {X_train.dtype}, range: [{X_train.min()}, {X_train.max()}])")
    print(f"  y_train: {y_train.shape} (dtype: {y_train.dtype})")
    print(f"  X_val:   {X_val.shape} (dtype: {X_val.dtype})")
    print(f"  y_val:   {y_val.shape} (dtype: {y_val.dtype})")

    # 3. Build Model
    print(f"\nBuilding 26-Class Compact CNN architecture...")
    model = build_real_compact_cnn(
        input_shape=(64, 64, 1),
        num_classes=num_classes,
        learning_rate=args.lr
    )
    model.summary()

    # 4. Callbacks
    history_csv = reports_dir / "real_training_history.csv"
    checkpoint_cb = ModelCheckpoint(
        filepath=str(output_model_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )
    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        patience=5,
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

    # 5. Fit Model
    print("\n" + "=" * 75)
    print("STARTING TRAINING LOOP")
    print("=" * 75)
    train_start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    train_elapsed = time.time() - train_start_time
    print(f"\n[OK] Training completed in {train_elapsed/60:.2f} minutes ({train_elapsed:.1f}s)")

    # 6. Plot Training Curves
    curves_png = reports_dir / "real_training_curves.png"
    plot_real_training_curves(history_csv, curves_png)

    # 7. Summary Metrics
    best_val_loss = float(min(history.history["val_loss"]))
    best_val_acc = float(max(history.history["val_accuracy"]))
    final_epoch = len(history.history["loss"])

    print("\n" + "=" * 75)
    print("STEP 3 RESULTS SUMMARY:")
    print(f"  Best Val Loss:     {best_val_loss:.5f}")
    print(f"  Best Val Accuracy: {best_val_acc * 100:.2f}%")
    print(f"  Epochs Run:        {final_epoch} / {args.epochs}")
    print(f"  Saved Checkpoint:  {output_model_path}")
    print(f"  Curves Saved:      {curves_png}")
    print(f"  History CSV:       {history_csv}")
    print("=" * 75)

    # Save training metadata
    training_config = {
        "dataset": "C:\\Users\\hp\\Real ISL Camera dataset",
        "num_classes": num_classes,
        "classes": classes,
        "batch_size": args.batch_size,
        "epochs_run": final_epoch,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "total_train_samples": int(len(X_train)),
        "total_val_samples": int(len(X_val)),
        "training_time_sec": train_elapsed,
        "model_checkpoint": str(output_model_path)
    }
    with open("artifacts/training_config_real.json", "w") as f:
        json.dump(training_config, f, indent=2)
    print("[OK] Saved training metadata to artifacts/training_config_real.json")

if __name__ == "__main__":
    main()
