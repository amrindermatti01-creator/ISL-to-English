"""
src/inference.py - Unified Dual-Engine Indian Sign Language (ISL) Inference Engine
Features:
  1. Primary Engine: Ultra-Fast 126D Coordinate-Invariant Landmark MLP (<1ms latency, 100% background-immune)
  2. Legacy Engine: 64x64 Canny Edge 2D CNN with Aspect-Ratio-Preserving Letterbox ROI
  3. MediaPipe Hands detection (single & dual-handed gestures)
  4. Closed-Fist SPACE Gesture Detection for Natural Word Boundary Delimitation
  5. Intelligent Fingerspelling Buffer (TextBuilder) with Groq LLM & Offline Spell Correction
  6. Real-Time Multilingual Translation (Hindi, Punjabi, Tamil, Bengali, Telugu, etc.)
  7. Non-Blocking Text-to-Speech (gTTS + pygame.mixer)
  8. Pillow-based Unicode Text Overlay for live Indian script rendering
"""

import os
import sys
import time
import json
from collections import deque, Counter
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from src.preprocessing import preprocess_image
from src.text_builder import TextBuilder, is_fist_closed
from src.translator_engine import TranslatorEngine, SUPPORTED_LANGUAGES


class ISLInferenceEngine:
    def __init__(
        self,
        active_engine: str = "landmark_mlp",
        landmark_model_path: str = "artifacts/landmark_model.keras",
        landmark_mapping_path: str = "artifacts/landmark_class_mapping.json",
        cnn_model_path: str = "artifacts/best_model.keras",
        cnn_mapping_path: str = "artifacts/class_mapping.json",
        confidence_threshold: float = 0.70,
        smoothing_window: int = 5,
        target_size: Tuple[int, int] = (64, 64),
        threshold_value: int = 90,
        use_canny: bool = True,
        canny_low: int = 50,
        canny_high: int = 150,
        stability_duration: float = 1.0,
        pad_x_ratio: float = 0.15,
        pad_y_ratio: float = 0.15,
        edge_margin: int = 8,
        enable_segmentation: bool = False,
        target_language: str = "hi",
        tts_enabled: bool = False,
        model_path: Optional[str] = None,       # Backward compatibility
        class_mapping_path: Optional[str] = None # Backward compatibility
    ):
        self.active_engine = active_engine
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.target_size = target_size
        self.threshold_value = threshold_value
        self.use_canny = use_canny
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.stability_duration = stability_duration
        self.pad_x_ratio = pad_x_ratio
        self.pad_y_ratio = pad_y_ratio
        self.edge_margin = edge_margin
        self.enable_segmentation = enable_segmentation
        self.target_language = target_language
        self.tts_enabled = tts_enabled

        # Handle backwards compatibility parameters
        if model_path:
            cnn_model_path = model_path
        if class_mapping_path:
            cnn_mapping_path = class_mapping_path

        # 1. Load Landmark MLP Model
        self.landmark_model = None
        self.landmark_idx_to_class = {}
        self.landmark_classes = []
        lm_path = Path(landmark_model_path)
        if lm_path.exists():
            try:
                self.landmark_model = tf.keras.models.load_model(str(lm_path))
                print(f"[InferenceEngine] Loaded Landmark MLP from: {lm_path}")
            except Exception as e:
                print(f"[InferenceEngine] Warning: Could not load landmark model: {e}")

        lm_map_path = Path(landmark_mapping_path)
        if lm_map_path.exists():
            with open(lm_map_path, "r", encoding="utf-8") as f:
                lm_data = json.load(f)
            self.landmark_idx_to_class = {int(k): v for k, v in lm_data["idx_to_class"].items()}
            self.landmark_classes = lm_data["classes"]

        # 2. Load Legacy 2D CNN Model
        self.cnn_model = None
        self.cnn_idx_to_class = {}
        self.cnn_classes = []
        cnn_p = Path(cnn_model_path)
        if cnn_p.exists():
            try:
                self.cnn_model = tf.keras.models.load_model(str(cnn_p))
                print(f"[InferenceEngine] Loaded Legacy CNN from: {cnn_p}")
            except Exception as e:
                print(f"[InferenceEngine] Warning: Could not load CNN model: {e}")

        cnn_map_p = Path(cnn_mapping_path)
        if cnn_map_p.exists():
            with open(cnn_map_p, "r", encoding="utf-8") as f:
                cnn_data = json.load(f)
            self.cnn_idx_to_class = {int(k): v for k, v in cnn_data["idx_to_class"].items()}
            self.cnn_classes = cnn_data["classes"]

        # Default class lookup delegates to active engine
        self._sync_class_mapping()

        # 3. MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

        # 4. Optional Selfie Segmentation
        self.segmenter = None
        if self.enable_segmentation:
            self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

        # 5. Temporal Smoothing Buffers
        self.prediction_buffer = deque(maxlen=self.smoothing_window)
        self.confidence_buffer = deque(maxlen=self.smoothing_window)

        # 6. Intelligent Fingerspelling Buffer (TextBuilder)
        self.text_builder = TextBuilder(
            confirm_threshold=self.stability_duration,
            space_threshold=1.2
        )

        # 7. Translation & Speech Engine
        self.translator = TranslatorEngine()

        # Profiling stats
        self.fps = 0.0
        self.frame_count = 0
        self.start_fps_time = time.time()
        self.latency_ms = 0.0
        self.translated_word = ""
        self.translated_sentence = ""

    def _sync_class_mapping(self):
        if self.active_engine == "landmark_mlp" and self.landmark_idx_to_class:
            self.idx_to_class = self.landmark_idx_to_class
            self.classes = self.landmark_classes
            self.model = self.landmark_model
        else:
            self.idx_to_class = self.cnn_idx_to_class
            self.classes = self.cnn_classes
            self.model = self.cnn_model

    def set_engine_mode(self, mode: str):
        """Switches active engine between 'landmark_mlp' and 'cnn_legacy'."""
        if mode in ("landmark_mlp", "cnn_legacy"):
            self.active_engine = mode
            self._sync_class_mapping()
            print(f"[InferenceEngine] Active engine switched to: {self.active_engine}")

    def set_target_language(self, lang_code: str):
        """Sets target language for translation ('hi', 'pa', 'ta', 'bn', etc.)."""
        self.target_language = lang_code
        self._update_translation()

    def toggle_tts(self) -> bool:
        """Toggles audio speech on/off."""
        self.tts_enabled = not self.tts_enabled
        return self.tts_enabled

    def extract_126d_landmarks(self, landmarks_list, handedness_list) -> Optional[np.ndarray]:
        """
        Extracts wrist-normalized 126D landmark vector:
          Indices 0..62:  Left hand
          Indices 63..125: Right hand
        """
        if not landmarks_list or not handedness_list:
            return None

        left_pts = None
        right_pts = None

        for i, hand_landmarks in enumerate(landmarks_list):
            label = handedness_list[i].classification[0].label.lower()
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32
            )
            if label == "left":
                left_pts = coords
            else:
                right_pts = coords

        if left_pts is None and right_pts is None:
            return None

        # Normalize relative to left wrist (or respective wrist)
        if left_pts is not None and right_pts is not None:
            ref = left_pts[0].copy()
            left_vec = (left_pts - ref).flatten()
            right_vec = (right_pts - ref).flatten()
        elif left_pts is not None:
            ref = left_pts[0].copy()
            left_vec = (left_pts - ref).flatten()
            right_vec = np.zeros(63, dtype=np.float32)
        else:
            ref = right_pts[0].copy()
            left_vec = np.zeros(63, dtype=np.float32)
            right_vec = (right_pts - ref).flatten()

        return np.concatenate([left_vec, right_vec]).astype(np.float32)

    def extract_hand_roi(self, frame_bgr, landmarks_list, pad_x_ratio=None, pad_y_ratio=None):
        """Bounding box logic for Legacy CNN mode."""
        if pad_x_ratio is None:
            pad_x_ratio = self.pad_x_ratio
        if pad_y_ratio is None:
            pad_y_ratio = self.pad_y_ratio

        if not landmarks_list:
            return None, None, "no_hand"

        h, w = frame_bgr.shape[:2]
        all_xs = []
        all_ys = []

        for hand_landmarks in landmarks_list:
            for lm in hand_landmarks.landmark:
                if np.isnan(lm.x) or np.isnan(lm.y):
                    continue
                all_xs.append(int(round(lm.x * w)))
                all_ys.append(int(round(lm.y * h)))

        if not all_xs or not all_ys:
            return None, None, "no_hand"

        x_min, x_max = min(all_xs), max(all_xs)
        y_min, y_max = min(all_ys), max(all_ys)
        box_w = x_max - x_min
        box_h = y_max - y_min

        if (x_min <= self.edge_margin or y_min <= self.edge_margin or
            x_max >= (w - self.edge_margin) or y_max >= (h - self.edge_margin)):
            return None, (max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)), "truncated"

        if box_w < 20 or box_h < 20:
            return None, (max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)), "too_small"

        pad_x = max(6, int(round(box_w * pad_x_ratio)))
        pad_y = max(6, int(round(box_h * pad_y_ratio)))

        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_max + pad_x)
        y2 = min(h, y_max + pad_y)

        roi = frame_bgr[y1:y2, x1:x2].copy()
        bbox = (x1, y1, x2, y2)
        return roi, bbox, "ok"

    def process_frame(self, frame_bgr: np.ndarray):
        """
        Processes a single webcam/video frame.
        Executes active engine (Landmark MLP or CNN Legacy), updates TextBuilder,
        computes multilingual translation, and returns enriched outputs.
        """
        t0 = time.time()
        self.frame_count += 1

        if time.time() - self.start_fps_time >= 1.0:
            self.fps = self.frame_count / (time.time() - self.start_fps_time)
            self.frame_count = 0
            self.start_fps_time = time.time()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_results = self.hands.process(frame_rgb)

        annotated_frame = frame_bgr.copy()
        source_roi = None
        prep_preview = None
        raw_pred_class = None
        raw_confidence = 0.0
        roi_status = "no_hand"
        bbox = None
        fist_detected = False

        if mp_results.multi_hand_landmarks:
            # Draw skeletons
            for hand_landmarks in mp_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                if is_fist_closed(hand_landmarks):
                    fist_detected = True

            # ==========================================
            # ENGINE 1: Landmark MLP (Primary, <1ms)
            # ==========================================
            if self.active_engine == "landmark_mlp" and self.landmark_model is not None:
                roi_status = "ok"
                vec126 = self.extract_126d_landmarks(
                    mp_results.multi_hand_landmarks,
                    mp_results.multi_handedness
                )

                if vec126 is not None:
                    input_tensor = np.expand_dims(vec126, axis=0)  # (1, 126)
                    predictions = self.landmark_model(input_tensor, training=False).numpy()[0]
                    best_idx = int(np.argmax(predictions))
                    raw_confidence = float(predictions[best_idx])
                    raw_pred_class = self.landmark_idx_to_class.get(best_idx, "Unknown")

                # Also grab hand ROI bounding box for debug visual display
                _, bbox, _ = self.extract_hand_roi(frame_bgr, mp_results.multi_hand_landmarks)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    source_roi = frame_bgr[y1:y2, x1:x2].copy()
                    color = (255, 100, 0) if fist_detected else (0, 255, 100)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    tag = "CLOSED FIST (SPACE)" if fist_detected else "LANDMARKS 126D"
                    cv2.putText(
                        annotated_frame,
                        tag,
                        (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        color,
                        1,
                        cv2.LINE_AA
                    )

            # ==========================================
            # ENGINE 2: Legacy 2D CNN (Fallback)
            # ==========================================
            else:
                roi_bgr, bbox, roi_status = self.extract_hand_roi(frame_bgr, mp_results.multi_hand_landmarks)
                if roi_status == "ok" and roi_bgr is not None and bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
                    source_roi = roi_bgr.copy()

                    preprocessed = preprocess_image(
                        roi_bgr,
                        target_size=self.target_size,
                        threshold_value=self.threshold_value,
                        use_canny=self.use_canny,
                        canny_low=self.canny_low,
                        canny_high=self.canny_high,
                        normalize=True,
                        preserve_aspect_ratio=True
                    )
                    prep_preview = (preprocessed.squeeze() * 255).astype(np.uint8)

                    tensor_input = np.expand_dims(preprocessed, axis=0)
                    predictions = self.cnn_model(tensor_input, training=False).numpy()[0]
                    best_idx = int(np.argmax(predictions))
                    raw_confidence = float(predictions[best_idx])
                    raw_pred_class = self.cnn_idx_to_class.get(best_idx, "Unknown")
                elif bbox is not None:
                    x1, y1, x2, y2 = bbox
                    warn_color = (0, 165, 255) if roi_status == "truncated" else (100, 100, 255)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), warn_color, 2)

        # Update temporal buffer
        if raw_pred_class and raw_confidence >= self.confidence_threshold:
            self.prediction_buffer.append(raw_pred_class)
            self.confidence_buffer.append(raw_confidence)
        else:
            self.prediction_buffer.append(None)
            self.confidence_buffer.append(0.0)

        # Temporal majority voting
        valid_predictions = [p for p in self.prediction_buffer if p is not None]
        if valid_predictions and len(valid_predictions) >= (self.smoothing_window // 2 + 1):
            majority_class, _ = Counter(valid_predictions).most_common(1)[0]
            avg_conf = float(np.mean([c for p, c in zip(self.prediction_buffer, self.confidence_buffer) if p == majority_class]))
            smooth_class = majority_class
            smooth_conf = avg_conf
            is_reliable = True
        else:
            if roi_status == "truncated":
                smooth_class = "Hand near edge — move into view"
            elif roi_status == "too_small":
                smooth_class = "Hand too small"
            elif roi_status == "no_hand":
                smooth_class = "Hand not detected"
            else:
                smooth_class = "Uncertain / No reliable sign"
            smooth_conf = 0.0
            is_reliable = False

        # Feed to intelligent TextBuilder
        active_sign = smooth_class if is_reliable else None
        tb_state = self.text_builder.update(active_sign, fist_detected=fist_detected)

        # Trigger TTS on committed word
        if tb_state["event"] == "WORD_COMMITTED":
            committed = tb_state["committed_word"]
            if self.target_language != "en":
                translated_w = self.translator.translate(committed, self.target_language)
            else:
                translated_w = committed

            self.translated_word = translated_w
            self._update_translation()

            if self.tts_enabled:
                self.translator.speak_async(translated_w, self.target_language)

        self.latency_ms = (time.time() - t0) * 1000.0

        # Render regional Unicode text on frame if not English
        if self.target_language != "en" and (self.translated_word or self.translated_sentence):
            display_str = f"[{self.target_language.upper()}]: {self.translated_word or self.translated_sentence}"
            annotated_frame = self.translator.render_unicode_text(
                annotated_frame,
                display_str,
                pos=(15, 60),
                lang_code=self.target_language,
                font_size=26,
                color=(255, 255, 50),
                bg_color=(20, 20, 20)
            )

        result = {
            "predicted_class": smooth_class,
            "confidence": smooth_conf,
            "raw_class": raw_pred_class or "None",
            "raw_confidence": raw_confidence,
            "is_reliable": is_reliable,
            "latency_ms": round(self.latency_ms, 1),
            "fps": round(self.fps, 1),
            "spelled_text": tb_state["raw_word"],
            "committed_word": tb_state["committed_word"],
            "sentence": tb_state["sentence"],
            "translated_word": self.translated_word,
            "translated_sentence": self.translated_sentence,
            "target_language": self.target_language,
            "tts_enabled": self.tts_enabled,
            "active_engine": self.active_engine,
            "fist_detected": fist_detected,
            "source_roi": source_roi,
            "roi_status": roi_status,
            "bbox": bbox
        }
        return annotated_frame, source_roi, prep_preview, result

    def _update_translation(self):
        sentence = " ".join(self.text_builder.sentence_words)
        if sentence and self.target_language != "en":
            self.translated_sentence = self.translator.translate(sentence, self.target_language)
        else:
            self.translated_sentence = sentence

    def add_space(self):
        committed = self.text_builder.commit_word()
        if committed:
            self._update_translation()
            if self.tts_enabled:
                self.translator.speak_async(self.translated_word, self.target_language)

    def backspace(self):
        self.text_builder.backspace()
        self._update_translation()

    def clear_text(self):
        self.text_builder.clear()
        self.translated_word = ""
        self.translated_sentence = ""

    def release(self):
        if hasattr(self, "hands") and self.hands:
            self.hands.close()
        if hasattr(self, "segmenter") and self.segmenter:
            self.segmenter.close()
