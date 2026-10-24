"""
src/inference.py - Real-Time Indian Sign Language (ISL) Inference Engine
Features:
  1. MediaPipe Hands detection (supports single and dual-handed ISL gestures)
  2. Dynamic Hand ROI bounding box with margin padding
  3. Exact Unified Preprocessing (64x64 Canny edge matching trained CNN)
  4. Temporal smoothing buffer (rolling majority vote to eliminate flicker)
  5. Confidence threshold gating ("No reliable sign detected" when uncertain)
  6. Fingerspelling text buffer with stability timer
"""

import os
import sys
import time
import json
from collections import deque, Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from src.preprocessing import preprocess_image

class ISLInferenceEngine:
    def __init__(
        self,
        model_path="artifacts/best_model.keras",
        class_mapping_path="artifacts/class_mapping.json",
        confidence_threshold=0.70,
        smoothing_window=5,
        target_size=(64, 64),
        threshold_value=90,
        use_canny=True,
        canny_low=50,
        canny_high=150,
        stability_duration=1.0,  # seconds to hold sign for spelling buffer
        margin_scale=1.15        # tightened from 1.3 to reduce background contamination
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.target_size = target_size
        self.threshold_value = threshold_value
        self.use_canny = use_canny
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.stability_duration = stability_duration
        self.margin_scale = margin_scale

        # Load class mapping
        if not Path(class_mapping_path).exists():
            raise FileNotFoundError(f"Class mapping file not found: {class_mapping_path}")
        with open(class_mapping_path, "r") as f:
            mapping = json.load(f)
        self.idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
        self.classes = mapping["classes"]

        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        print(f"[InferenceEngine] Loading model from: {self.model_path}")
        self.model = tf.keras.models.load_model(str(self.model_path))

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

        # Temporal smoothing deque
        self.prediction_buffer = deque(maxlen=self.smoothing_window)
        self.confidence_buffer = deque(maxlen=self.smoothing_window)

        # Fingerspelling accumulator state
        self.spelled_text = ""
        self.last_predicted_char = None
        self.char_hold_start_time = None
        self.char_already_added = False

        # Profiling stats
        self.fps = 0.0
        self.frame_count = 0
        self.start_fps_time = time.time()
        self.latency_ms = 0.0

    def extract_hand_roi(self, frame_bgr, landmarks_list, margin_scale=None):
        """
        Calculates bounding box enclosing all detected hands (single or dual),
        adds a margin (default tightened to 1.15x), and crops ROI.
        """
        if margin_scale is None:
            margin_scale = self.margin_scale
        h, w = frame_bgr.shape[:2]
        all_xs = []
        all_ys = []

        for hand_landmarks in landmarks_list:
            for lm in hand_landmarks.landmark:
                all_xs.append(int(lm.x * w))
                all_ys.append(int(lm.y * h))

        if not all_xs or not all_ys:
            return None, None

        x_min, x_max = min(all_xs), max(all_xs)
        y_min, y_max = min(all_ys), max(all_ys)

        box_w = x_max - x_min
        box_h = y_max - y_min
        center_x = x_min + box_w // 2
        center_y = y_min + box_h // 2

        # Make square crop based on maximum dimension
        side = int(max(box_w, box_h) * margin_scale)
        half_side = side // 2

        x1 = max(0, center_x - half_side)
        y1 = max(0, center_y - half_side)
        x2 = min(w, center_x + half_side)
        y2 = min(h, center_y + half_side)

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return None, None

        roi = frame_bgr[y1:y2, x1:x2]
        bbox = (x1, y1, x2, y2)
        return roi, bbox

    def process_frame(self, frame_bgr):
        """
        Main inference entry point for a single frame.
        Returns:
          annotated_frame: frame with overlays
          prep_preview: 64x64 Canny edge crop for Picture-in-Picture display
          result_dict: {
             'predicted_class': str,
             'confidence': float,
             'raw_class': str,
             'raw_confidence': float,
             'is_reliable': bool,
             'latency_ms': float,
             'fps': float,
             'spelled_text': str,
             'hold_progress': float
          }
        """
        t0 = time.time()
        self.frame_count += 1

        # Calculate FPS
        if time.time() - self.start_fps_time >= 1.0:
            self.fps = self.frame_count / (time.time() - self.start_fps_time)
            self.frame_count = 0
            self.start_fps_time = time.time()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_results = self.hands.process(frame_rgb)

        annotated_frame = frame_bgr.copy()
        prep_preview = None
        raw_pred_class = None
        raw_confidence = 0.0

        if mp_results.multi_hand_landmarks:
            # Draw hand skeleton
            for hand_landmarks in mp_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

            # Crop hand ROI
            roi_bgr, bbox = self.extract_hand_roi(frame_bgr, mp_results.multi_hand_landmarks)

            if roi_bgr is not None and bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 100), 2)

                # Preprocess cropped hand
                preprocessed = preprocess_image(
                    roi_bgr,
                    target_size=self.target_size,
                    threshold_value=self.threshold_value,
                    use_canny=self.use_canny,
                    canny_low=self.canny_low,
                    canny_high=self.canny_high,
                    normalize=True
                )
                prep_preview = (preprocessed.squeeze() * 255).astype(np.uint8)

                # Model forward pass
                tensor_input = np.expand_dims(preprocessed, axis=0)  # (1, 64, 64, 1)
                predictions = self.model(tensor_input, training=False).numpy()[0]

                best_idx = int(np.argmax(predictions))
                raw_confidence = float(predictions[best_idx])
                raw_pred_class = self.idx_to_class.get(best_idx, "Unknown")

        # Update temporal buffer
        if raw_pred_class and raw_confidence >= self.confidence_threshold:
            self.prediction_buffer.append(raw_pred_class)
            self.confidence_buffer.append(raw_confidence)
        else:
            self.prediction_buffer.append(None)
            self.confidence_buffer.append(0.0)

        # Smooth prediction via majority voting
        valid_predictions = [p for p in self.prediction_buffer if p is not None]
        if valid_predictions and len(valid_predictions) >= (self.smoothing_window // 2 + 1):
            majority_class, count = Counter(valid_predictions).most_common(1)[0]
            avg_conf = float(np.mean([c for p, c in zip(self.prediction_buffer, self.confidence_buffer) if p == majority_class]))
            smooth_class = majority_class
            smooth_conf = avg_conf
            is_reliable = True
        else:
            smooth_class = "No reliable sign detected"
            smooth_conf = 0.0
            is_reliable = False

        # Update fingerspelling accumulator
        hold_progress = 0.0
        if is_reliable:
            if smooth_class == self.last_predicted_char:
                if self.char_hold_start_time is None:
                    self.char_hold_start_time = time.time()
                elapsed_hold = time.time() - self.char_hold_start_time
                hold_progress = min(1.0, elapsed_hold / self.stability_duration)

                if elapsed_hold >= self.stability_duration and not self.char_already_added:
                    self.spelled_text += smooth_class
                    self.char_already_added = True
            else:
                self.last_predicted_char = smooth_class
                self.char_hold_start_time = time.time()
                self.char_already_added = False
        else:
            self.char_hold_start_time = None
            self.last_predicted_char = None
            self.char_already_added = False

        self.latency_ms = (time.time() - t0) * 1000.0

        result = {
            "predicted_class": smooth_class,
            "confidence": smooth_conf,
            "raw_class": raw_pred_class or "None",
            "raw_confidence": raw_confidence,
            "is_reliable": is_reliable,
            "latency_ms": round(self.latency_ms, 1),
            "fps": round(self.fps, 1),
            "spelled_text": self.spelled_text,
            "hold_progress": round(hold_progress, 2)
        }
        return annotated_frame, prep_preview, result

    def add_space(self):
        self.spelled_text += " "

    def backspace(self):
        if self.spelled_text:
            self.spelled_text = self.spelled_text[:-1]

    def clear_text(self):
        self.spelled_text = ""

    def release(self):
        if hasattr(self, "hands") and self.hands:
            self.hands.close()
