"""
src/preprocessing.py - Unified ISL Image Preprocessing Pipeline
Matches research paper specifications:
  1. Grayscale Conversion (Luminance model: 0.299R + 0.587G + 0.114B)
  2. Gaussian Blur (subtle noise suppression)
  3. Binary Thresholding (T=90) and/or Canny Edge Detection
  4. Resizing to 64x64
  5. Pixel Normalization to [0.0, 1.0]
Used identically in both training and live webcam inference.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import yaml
import matplotlib.pyplot as plt

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def preprocess_image(
    image_input,
    target_size=(64, 64),
    threshold_value=90,
    use_canny=True,
    canny_low=50,
    canny_high=150,
    normalize=True,
    preserve_aspect_ratio=True
):
    """
    Unified preprocessing function used identically for training, evaluation,
    and real-time webcam inference.

    Args:
        image_input: File path (str or Path) or numpy array (BGR or RGB or Grayscale).
        target_size: Tuple (width, height), default (64, 64).
        threshold_value: Binary threshold cutoff T=90 (Paper Section 3.2.2).
        use_canny: Whether to apply Canny edge detection (Paper Section 3.2.3).
        canny_low: Lower gradient threshold for Canny.
        canny_high: Upper gradient threshold for Canny.
        normalize: If True, scales pixel values to range [0.0, 1.0].
        preserve_aspect_ratio: If True, scales rectangular inputs proportionally
                               and letterboxes/pads onto target_size canvas with 0.

    Returns:
        np.ndarray of shape (target_size[1], target_size[0], 1) with dtype float32 (if normalized) or uint8.
    """
    # 1. Handle input types
    if isinstance(image_input, (str, Path)):
        # Load image via cv2 (returns BGR)
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"Unable to read image at: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # 2. Grayscale conversion (Luminance model)
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Standard OpenCV BGR to Gray follows Y = 0.299R + 0.587G + 0.114B
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif len(img.shape) == 2:
        gray = img
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    # 3. Gaussian Blur to reduce high-frequency noise before thresholding
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 4. Feature extraction: Binary thresholding & Canny edge detection
    if use_canny:
        # Canny edge detection highlighting hand boundary contours
        processed = cv2.Canny(blurred, canny_low, canny_high)
    elif threshold_value is not None:
        # Binary thresholding with set cutoff T=90 (Equation 2 in Paper)
        _, processed = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    else:
        processed = blurred

    # 5. Resize to target dimensions (with aspect-ratio preservation option)
    if preserve_aspect_ratio:
        h_p, w_p = processed.shape[:2]
        tw, th = target_size
        scale = min(tw / w_p, th / h_p)
        new_w = max(1, int(round(w_p * scale)))
        new_h = max(1, int(round(h_p * scale)))

        resized = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Re-binarize after interpolation to ensure edges remain crisp (prevents dilution bug)
        if use_canny or threshold_value is not None:
            _, resized = cv2.threshold(resized, 20, 255, cv2.THRESH_BINARY)

        canvas = np.zeros((th, tw), dtype=resized.dtype)
        ox = (tw - new_w) // 2
        oy = (th - new_h) // 2
        canvas[oy:oy + new_h, ox:ox + new_w] = resized
        processed = canvas
    elif (processed.shape[1], processed.shape[0]) != target_size:
        processed = cv2.resize(processed, target_size, interpolation=cv2.INTER_AREA)
        # Re-binarize after interpolation to ensure edges remain crisp (prevents dilution bug)
        if use_canny or threshold_value is not None:
            _, processed = cv2.threshold(processed, 20, 255, cv2.THRESH_BINARY)

    # 6. Normalize pixel values [0, 1]
    if normalize:
        processed_arr = processed.astype(np.float32) / 255.0
    else:
        processed_arr = processed

    # 7. Add channel dimension -> (H, W, 1) for Conv2D input
    if len(processed_arr.shape) == 2:
        processed_arr = np.expand_dims(processed_arr, axis=-1)

    return processed_arr

def visualize_sample_preprocessing(manifest_csv_path, output_image_path="reports/sample_preprocessed.png", num_samples=10):
    """
    Visualizes raw input images side-by-side with their preprocessed outputs
    to verify pipeline correctness and preserve hand contours.
    """
    import pandas as pd
    df = pd.read_csv(manifest_csv_path)
    # Pick distinct classes for visual verification
    sample_df = df.groupby("class_name").first().reset_index().head(num_samples)

    fig, axes = plt.subplots(num_samples, 2, figsize=(6, num_samples * 2.2))

    for idx, row in sample_df.iterrows():
        img_path = row["path"]
        c_name = row["class_name"]

        # Raw image
        raw_bgr = cv2.imread(img_path)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        # Preprocessed image (64x64, 1 channel)
        prep = preprocess_image(img_path, target_size=(64, 64), normalize=True)
        prep_view = (prep.squeeze() * 255).astype(np.uint8)

        # Plot Raw
        axes[idx, 0].imshow(raw_rgb)
        axes[idx, 0].set_title(f"Raw [{c_name}]", fontsize=10)
        axes[idx, 0].axis("off")

        # Plot Preprocessed
        axes[idx, 1].imshow(prep_view, cmap="gray")
        axes[idx, 1].set_title(f"Preprocessed (64x64 Canny) [{c_name}]", fontsize=10)
        axes[idx, 1].axis("off")

    plt.tight_layout()
    Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image_path, dpi=150)
    plt.close()
    print(f"[Stage 2] Preprocessing verification image saved to: {output_image_path}")

if __name__ == "__main__":
    cfg = load_config()
    p_cfg = cfg["preprocessing"]
    test_img = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.circle(test_img, (64, 64), 30, (255, 255, 255), -1)
    out = preprocess_image(
        test_img,
        target_size=tuple(p_cfg["target_size"]),
        threshold_value=p_cfg["threshold_value"],
        use_canny=p_cfg["use_canny"],
        canny_low=p_cfg["canny_low"],
        canny_high=p_cfg["canny_high"],
        normalize=p_cfg["normalize"]
    )
    print(f"[Stage 2] Preprocessing unit check passed! Output shape: {out.shape}, dtype: {out.dtype}, range: [{out.min()}, {out.max()}]")


# ============================================================================
# MediaPipe 126D Landmark Normalization & Validation (Dual-Hand ISL Canonical)
# ============================================================================

def normalize_landmarks_126d(landmarks, handedness="Right"):
    """
    Normalizes hand landmarks to a canonical 126D coordinate vector.
    - Preserves ISL dual-handed alphabet requirements:
        Left hand occupies indices [0:63]
        Right hand occupies indices [63:126]
    - Translates landmarks so wrist (landmark 0) is at origin (0, 0, 0).
    - Scales coordinates by Euclidean distance between wrist and middle MCP (landmark 9).
    - Missing hand is padded with zeros.

    Args:
        landmarks: List of 21 landmark objects/dicts or (21, 3) ndarray.
        handedness: 'Left' or 'Right' (or 'left'/'right').

    Returns:
        np.ndarray of shape (126,) with dtype float32.
    """
    if isinstance(landmarks, np.ndarray):
        pts = landmarks.astype(np.float32)
    else:
        pts = []
        for lm in landmarks:
            if hasattr(lm, 'x'):
                pts.append([lm.x, lm.y, lm.z])
            elif isinstance(lm, dict):
                pts.append([lm['x'], lm['y'], lm['z']])
            else:
                pts.append(list(lm)[:3])
        pts = np.array(pts, dtype=np.float32)

    if pts.shape != (21, 3):
        raise ValueError(f"Expected 21 landmarks with (x, y, z), got shape {pts.shape}")

    wrist = pts[0]
    mcp = pts[9]
    scale = float(np.linalg.norm(mcp - wrist)) + 1e-6
    pts_norm = (pts - wrist) / scale

    vec = np.zeros(126, dtype=np.float32)
    h_lower = str(handedness).lower()
    if "left" in h_lower:
        vec[0:63] = pts_norm.flatten()
    else:
        vec[63:126] = pts_norm.flatten()

    return vec


def validate_landmarks_vector(vec):
    """
    Validates a 126D landmark feature vector prior to model inference.
    Rejects:
    - Incorrect type or shape != (126,)
    - NaN or Inf values
    - Completely empty vectors (all zeros, indicating no hand detected)

    Returns:
        tuple (is_valid: bool, error_message: str)
    """
    if not isinstance(vec, np.ndarray):
        return False, "Input must be a numpy ndarray"
    if vec.shape != (126,):
        return False, f"Expected shape (126,), got {vec.shape}"
    if np.isnan(vec).any():
        return False, "Vector contains NaN values"
    if np.isinf(vec).any():
        return False, "Vector contains Inf values"
    if np.all(vec == 0.0):
        return False, "Vector is empty (all zeros - no hand detected)"
    return True, "Valid"

