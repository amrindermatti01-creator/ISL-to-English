"""
src/train_split_landmark_model.py
Trains a compact feedforward neural network (MLP) on canonical split landmarks:
  - Train: data/processed/train_landmarks.npz
  - Validation: data/processed/val_landmarks.npz

Strictly respects split boundaries (zero test data exposure).
Saves model checkpoint to artifacts/landmark_model.keras and generates training curves.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LandmarkModelTrainer")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Set deterministic random seeds
tf.random.set_seed(42)
np.random.seed(42)


def build_compact_landmark_mlp(input_dim: int = 126, num_classes: int = 35) -> tf.keras.Model:
    """
    Compact 4-layer MLP tailored for landmark coordinate classification.
    Parameter count: ~77,000 parameters.
    """
    inputs = layers.Input(shape=(input_dim,), name="landmarks_input")

    # Layer 1
    x = layers.Dense(256, activation="relu", kernel_initializer="he_normal", name="dense1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(0.2, name="drop1")(x)

    # Layer 2
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal", name="dense2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dropout(0.2, name="drop2")(x)

    # Layer 3
    x = layers.Dense(64, activation="relu", kernel_initializer="he_normal", name="dense3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Dropout(0.1, name="drop3")(x)

    # Classification output (probabilities)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ISL_Landmark_MLP")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_landmark_model(
    train_npz_path: str = "data/processed/train_landmarks.npz",
    val_npz_path: str = "data/processed/val_landmarks.npz",
    model_output_path: str = "artifacts/landmark_model.keras",
    config_output_path: str = "artifacts/landmark_training_config.json",
    reports_dir: str = "reports",
    epochs: int = 35,
    batch_size: int = 64
) -> Dict[str, Any]:
    logger.info(f"Loading training split from {train_npz_path}...")
    train_data = np.load(train_npz_path)
    X_train, y_train = train_data["X"], train_data["y"]

    logger.info(f"Loading validation split from {val_npz_path}...")
    val_data = np.load(val_npz_path)
    X_val, y_val = val_data["X"], val_data["y"]

    num_classes = 35
    input_dim = X_train.shape[1]

    logger.info(f"Training samples: {len(X_train)} | Validation samples: {len(X_val)} | Input Dim: {input_dim}")

    model = build_compact_landmark_mlp(input_dim=input_dim, num_classes=num_classes)
    model.summary(print_fn=lambda x: logger.info(x))

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_output_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    logger.info("Starting model training on canonical training split...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1
    )

    # Save training curves plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss", color="#0284c7")
    plt.plot(history.history["val_loss"], label="Val Loss", color="#ea580c")
    plt.title("Landmark Model Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy", color="#10b981")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy", color="#f59e0b")
    plt.title("Landmark Model Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    curves_path = os.path.join(reports_dir, "landmark_training_curves.png")
    plt.tight_layout()
    plt.savefig(curves_path, dpi=200)
    plt.close()
    logger.info(f"Saved training curves to {curves_path}")

    # Save history dataframe
    hist_df = pd.DataFrame(history.history)
    hist_path = os.path.join(reports_dir, "landmark_training_history.csv")
    hist_df.to_csv(hist_path, index=False)

    # Save training config
    best_val_acc = float(np.max(history.history["val_accuracy"]))
    final_train_acc = float(history.history["accuracy"][-1])
    config_dict = {
        "architecture": "4-Layer Dense MLP",
        "input_dim": input_dim,
        "num_classes": num_classes,
        "epochs_run": len(history.history["loss"]),
        "batch_size": batch_size,
        "best_val_accuracy": best_val_acc,
        "final_train_accuracy": final_train_acc,
        "optimizer": "Adam (lr=0.001)",
        "regularization": "BatchNormalization + Dropout(0.2, 0.2, 0.1)",
        "train_samples": len(X_train),
        "val_samples": len(X_val)
    }
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Model saved to {model_output_path}. Best Validation Accuracy: {best_val_acc*100:.2f}%")
    return config_dict


def main():
    parser = argparse.ArgumentParser(description="Train Landmark MLP on split manifests.")
    parser.add_argument("--train-data", type=str, default="data/processed/train_landmarks.npz")
    parser.add_argument("--val-data", type=str, default="data/processed/val_landmarks.npz")
    parser.add_argument("--model-out", type=str, default="artifacts/landmark_model.keras")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    train_landmark_model(
        train_npz_path=args.train_data,
        val_npz_path=args.val_data,
        model_output_path=args.model_out,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
