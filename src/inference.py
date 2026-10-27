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

def capture_fullscreen_desktop():
    """
    Captures the complete native desktop full screen across primary display / virtual screen.
    Uses PIL ImageGrab with winsta0 attachment on Windows, with fallback to GDI BitBlt.
    Returns:
        numpy.ndarray (BGR format, uint8) or None if desktop DC is unavailable.
    """
    if os.name == "nt":
        import ctypes
        try:
            user32 = ctypes.windll.user32
            # Attach to default desktop station if in background shell
            h_winsta = user32.OpenWindowStationW('winsta0', False, 0x000F037F)
            if h_winsta:
                user32.SetProcessWindowStation(h_winsta)
            h_desk = user32.OpenDesktopW('default', 0, False, 0x01FF)
            if h_desk:
                user32.SetThreadDesktop(h_desk)
        except Exception:
            pass

    # 1. Primary Attempt: PIL ImageGrab (handles DPI scaling, all monitors, and compositing)
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab(all_screens=True)
        arr = np.array(img)
        if arr is not None and arr.size > 0:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            if bgr.shape[0] >= 100 and bgr.shape[1] >= 100 and bgr.max() > 0:
                return bgr
    except Exception:
        pass

    # 2. Secondary Attempt: Direct Windows GDI BitBlt
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32

            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    user32.SetProcessDPIAware()
                except Exception:
                    pass

            x = user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
            y = user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN
            w = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
            h = user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN

            if w <= 0 or h <= 0:
                w = user32.GetSystemMetrics(0)
                h = user32.GetSystemMetrics(1)
                x, y = 0, 0

            hdc_screen = user32.GetDC(0)
            hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
            hbm = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
            gdi32.SelectObject(hdc_mem, hbm)

            # 0x00CC0020 = SRCCOPY
            gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, x, y, 0x00CC0020)

            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ('biSize', wintypes.DWORD),
                    ('biWidth', wintypes.LONG),
                    ('biHeight', wintypes.LONG),
                    ('biPlanes', wintypes.WORD),
                    ('biBitCount', wintypes.WORD),
                    ('biCompression', wintypes.DWORD),
                    ('biSizeImage', wintypes.DWORD),
                    ('biXPelsPerMeter', wintypes.LONG),
                    ('biYPelsPerMeter', wintypes.LONG),
                    ('biClrUsed', wintypes.DWORD),
                    ('biClrImportant', wintypes.DWORD)
                ]

            bmi = BITMAPINFOHEADER()
            bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.biWidth = w
            bmi.biHeight = -h
            bmi.biPlanes = 1
            bmi.biBitCount = 32
            bmi.biCompression = 0

            buf = ctypes.create_string_buffer(w * h * 4)
            gdi32.GetDIBits(hdc_mem, hbm, 0, h, buf, ctypes.byref(bmi), 0)

            gdi32.DeleteObject(hbm)
            gdi32.DeleteDC(hdc_mem)
            user32.ReleaseDC(0, hdc_screen)

            arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
            bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            return bgr
        except Exception:
            pass

    return None

class ISLInferenceEngine:
    def __init__(
        self,
        model_path=str(Path("artifacts/best_model_combined.keras") if Path("artifacts/best_model_combined.keras").exists() else (Path("artifacts/best_model_real.keras") if Path("artifacts/best_model_real.keras").exists() else Path("artifacts/best_model.keras"))),
        class_mapping_path=str(Path("artifacts/class_mapping_mendeley.json") if Path("artifacts/best_model_combined.keras").exists() else (Path("artifacts/class_mapping_real.json") if Path("artifacts/best_model_real.keras").exists() else Path("artifacts/class_mapping.json"))),
        confidence_threshold=0.55,
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
        if "idx_to_class" in mapping:
            self.idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
        elif "class_to_idx" in mapping:
            self.idx_to_class = {int(v): k for k, v in mapping["class_to_idx"].items()}
        else:
            self.idx_to_class = {i: c for i, c in enumerate(mapping["classes"])}
        self.classes = mapping.get("classes", [self.idx_to_class[i] for i in range(len(self.idx_to_class))])

        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        print(f"[InferenceEngine] Loading model from: {self.model_path}")
        self.model = tf.keras.models.load_model(str(self.model_path))

        # Initialize MediaPipe Hands (Stream mode for webcam/video, Static mode for photo)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.48,
            min_tracking_confidence=0.45
        )
        self.static_hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.40
        )

        # Temporal smoothing deques
        self.prediction_buffer = deque(maxlen=self.smoothing_window)
        self.confidence_buffer = deque(maxlen=self.smoothing_window)
        self.prob_buffer = deque(maxlen=self.smoothing_window)

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
        Calculates a tight bounding box around detected hands,
        masks out background/face clutter using per-hand convex hulls,
        and pads to square with clean black borders (letterboxing).
        Prevents sideways expansion that captures face, hair, and wall plates.
        """
        h, w = frame_bgr.shape[:2]
        all_pts = []

        for hand_landmarks in landmarks_list:
            for lm in hand_landmarks.landmark:
                all_pts.append([int(lm.x * w), int(lm.y * h)])

        if not all_pts:
            return None, None

        pts_arr = np.array(all_pts, dtype=np.int32)
        x_min, y_min = np.min(pts_arr, axis=0)
        x_max, y_max = np.max(pts_arr, axis=0)

        box_w = x_max - x_min
        box_h = y_max - y_min

        # Modest proportional padding (tight fit to hand, ~10% margin)
        pad_x = max(12, int(box_w * 0.10))
        pad_y = max(15, int(box_h * 0.10))

        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_max + pad_x)
        y2 = min(h, y_max + pad_y)

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return None, None

        # 1. Crop tight hand ROI
        roi = frame_bgr[y1:y2, x1:x2].copy()
        rh, rw = roi.shape[:2]

        # 2. Hand-only convex hull mask per hand to isolate fingers and eliminate facial clutter
        mask = np.zeros((rh, rw), dtype=np.uint8)
        for hand_landmarks in landmarks_list:
            hand_pts = np.array([[int(lm.x * w) - x1, int(lm.y * h) - y1] for lm in hand_landmarks.landmark], dtype=np.int32)
            if len(hand_pts) > 0:
                hull = cv2.convexHull(hand_pts)
                cv2.fillConvexPoly(mask, hull, 255)

        # Dilate slightly so finger boundaries and flesh are preserved
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=1)
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask)

        # 3. Letterbox to square with black background (no background noise introduced)
        max_dim = max(rh, rw)
        square_roi = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        off_x = (max_dim - rw) // 2
        off_y = (max_dim - rh) // 2
        square_roi[off_y:off_y + rh, off_x:off_x + rw] = masked_roi

        bbox = (x1, y1, x2, y2)
        return square_roi, bbox

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
             'hold_progress': float,
             'top3': list of dicts,
             'num_hands': int,
             'hand_mode': str
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
        raw_predictions = None
        num_hands = 0
        top3_classes = []

        if mp_results.multi_hand_landmarks:
            num_hands = len(mp_results.multi_hand_landmarks)
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
                raw_predictions = self.model(tensor_input, training=False).numpy()[0]

                best_idx = int(np.argmax(raw_predictions))
                raw_confidence = float(raw_predictions[best_idx])
                raw_pred_class = self.idx_to_class.get(best_idx, "Unknown")

        # Soft probability smoothing across frames (eliminates jitter and rapid dropouts)
        if raw_predictions is not None:
            self.prob_buffer.append(raw_predictions)
            avg_probs = np.mean(self.prob_buffer, axis=0)
            top3_idx = np.argsort(avg_probs)[::-1][:3]
            top3_classes = [
                {"class": self.idx_to_class.get(int(i), "?"), "prob": float(avg_probs[i])}
                for i in top3_idx
            ]
            smooth_best_idx = int(top3_idx[0])
            candidate_class = self.idx_to_class.get(smooth_best_idx, "Unknown")
            candidate_conf = float(avg_probs[smooth_best_idx])

            if candidate_conf >= self.confidence_threshold:
                smooth_class = candidate_class
                smooth_conf = candidate_conf
                is_reliable = True
            else:
                smooth_class = "No reliable sign detected"
                smooth_conf = candidate_conf
                is_reliable = False
        else:
            self.prob_buffer.clear()
            smooth_class = "No reliable sign detected"
            smooth_conf = 0.0
            is_reliable = False
            top3_classes = []

        # Determine hand mode description
        if num_hands == 1:
            hand_mode = "Single-Hand (Digits 1-9, C, I, L, O, U, V)"
        elif num_hands >= 2:
            hand_mode = "Dual-Hand (Letters A, B, D-H, J-K, M-N, P-T, W-Z)"
        else:
            hand_mode = "No Hands in View"

        # Update fingerspelling accumulator
        hold_progress = 0.0
        new_char_confirmed = False
        confirmed_char = None
        confirmed_confidence = 0.0

        if is_reliable:
            if smooth_class == self.last_predicted_char:
                if self.char_hold_start_time is None:
                    self.char_hold_start_time = time.time()
                elapsed_hold = time.time() - self.char_hold_start_time
                hold_progress = min(1.0, elapsed_hold / self.stability_duration)

                if elapsed_hold >= self.stability_duration and not self.char_already_added:
                    self.spelled_text += smooth_class
                    self.char_already_added = True
                    new_char_confirmed = True
                    confirmed_char = smooth_class
                    confirmed_confidence = smooth_conf
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
            "hold_progress": round(hold_progress, 2),
            "top3": top3_classes,
            "num_hands": num_hands,
            "hand_mode": hand_mode,
            "new_char_confirmed": new_char_confirmed,
            "confirmed_char": confirmed_char,
            "confirmed_confidence": round(confirmed_confidence, 4)
        }
        return annotated_frame, prep_preview, result

    def predict_image(self, frame_bgr):
        """
        Specialized offline one-shot prediction for static photos.
        Uses static_image_mode MediaPipe detector for maximal accuracy on independent images.
        Returns:
            annotated_frame: Image with skeleton overlays and ROI boxes.
            prep_preview: 64x64 Canny edge input fed to the CNN.
            result: Dictionary containing predicted class, confidence, top 3, latency, and status.
        """
        t0 = time.time()
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 1. Detect using static detector first
        mp_results = self.static_hands.process(frame_rgb)
        if not mp_results.multi_hand_landmarks:
            # Fallback to stream detector
            mp_results = self.hands.process(frame_rgb)

        annotated_frame = frame_bgr.copy()
        prep_preview = None
        num_hands = 0
        top3_classes = []

        if mp_results.multi_hand_landmarks:
            num_hands = len(mp_results.multi_hand_landmarks)
            for hand_landmarks in mp_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

            roi_bgr, bbox = self.extract_hand_roi(frame_bgr, mp_results.multi_hand_landmarks)
            if roi_bgr is not None and bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 100), 2)

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

                tensor_input = np.expand_dims(preprocessed, axis=0)
                raw_predictions = self.model(tensor_input, training=False).numpy()[0]

                top3_idx = np.argsort(raw_predictions)[::-1][:3]
                top3_classes = [
                    {"class": self.idx_to_class.get(int(i), "?"), "prob": float(raw_predictions[i])}
                    for i in top3_idx
                ]

                best_idx = int(top3_idx[0])
                pred_class = self.idx_to_class.get(best_idx, "Unknown")
                conf = float(raw_predictions[best_idx])
                is_reliable = (conf >= self.confidence_threshold)
            else:
                pred_class = "Bounding box failed"
                conf = 0.0
                is_reliable = False
        else:
            # Central crop fallback if hands are not recognized by MediaPipe (e.g. cropped dataset photo)
            box_side = min(h, w, 280)
            x1 = (w - box_side) // 2
            y1 = (h - box_side) // 2
            roi_bgr = frame_bgr[y1:y1 + box_side, x1:x1 + box_side]

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
            tensor_input = np.expand_dims(preprocessed, axis=0)
            raw_predictions = self.model(tensor_input, training=False).numpy()[0]
            top3_idx = np.argsort(raw_predictions)[::-1][:3]
            top3_classes = [
                {"class": self.idx_to_class.get(int(i), "?"), "prob": float(raw_predictions[i])}
                for i in top3_idx
            ]
            best_idx = int(top3_idx[0])
            pred_class = self.idx_to_class.get(best_idx, "Unknown")
            conf = float(raw_predictions[best_idx])
            is_reliable = (conf >= self.confidence_threshold)
            cv2.rectangle(annotated_frame, (x1, y1), (x1 + box_side, y1 + box_side), (255, 165, 0), 2)
            cv2.putText(annotated_frame, "Center-crop fallback (No MP Landmarks)", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1, cv2.LINE_AA)

        latency = (time.time() - t0) * 1000.0

        if num_hands == 1:
            hand_mode = "Single-Hand Gesture"
        elif num_hands >= 2:
            hand_mode = "Dual-Hand Gesture"
        else:
            hand_mode = "Fallback Center-Crop"

        result = {
            "predicted_class": pred_class if is_reliable else f"{pred_class} (Low Conf)",
            "sign": pred_class,
            "confidence": round(conf, 4),
            "is_reliable": is_reliable,
            "latency_ms": round(latency, 1),
            "top3": top3_classes,
            "num_hands": num_hands,
            "hand_mode": hand_mode
        }
        return annotated_frame, prep_preview, result

    def save_accurate_snapshot(
        self,
        frame_bgr=None,
        prep_preview=None,
        result=None,
        prefix="auto_result",
        snapshots_dir="reports/snapshots",
        capture_fullscreen=True,
        full_screen_img=None,
        app_window_img=None
    ):
        """
        Saves a pixel-perfect, high-resolution screenshot when a sign is recognized.
        Ensures the entire full screen is captured (not only the result frame) by:
          1. Capturing the complete native desktop full screen (all monitors/windows).
          2. Saving the full application window / HUD frame if available.
          3. Saving the annotated camera frame and 64x64 Canny edge input.
          4. Naming the primary file informatively: fullscreen_{prefix}_{char}_{conf}pct_{YYYYMMDD_HHMMSS}.png
        """
        out_dir = Path(snapshots_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        char_str = "SIGN"
        conf_pct = 0
        if result:
            char_str = str(result.get("confirmed_char") or result.get("predicted_class") or "SIGN").replace(" ", "_")
            conf_pct = int(round((result.get("confidence") or 0.0) * 100))

        # 1. FULL SCREEN CAPTURE (Whole Monitor / Desktop)
        fs_img = full_screen_img
        if fs_img is None and capture_fullscreen:
            fs_img = capture_fullscreen_desktop()

        fs_path = None
        fs_filename = None
        if fs_img is not None and getattr(fs_img, "size", 0) > 0:
            fs_filename = f"fullscreen_{prefix}_{char_str}_{conf_pct}pct_{timestamp}.png"
            fs_path = out_dir / fs_filename
            cv2.imwrite(str(fs_path), fs_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        # 2. FULL APPLICATION WINDOW / HUD FRAME
        win_path = None
        win_filename = None
        if app_window_img is not None and getattr(app_window_img, "size", 0) > 0:
            win_filename = f"window_{prefix}_{char_str}_{conf_pct}pct_{timestamp}.png"
            win_path = out_dir / win_filename
            cv2.imwrite(str(win_path), app_window_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if fs_path is None:
                fs_path = win_path
                fs_filename = win_filename

        # 3. CAMERA RESULT COMPOSITE (with metadata footer)
        res_path = None
        res_filename = None
        if frame_bgr is not None and getattr(frame_bgr, "size", 0) > 0:
            h, w = frame_bgr.shape[:2]
            composite = frame_bgr.copy()

            banner_h = 44
            footer = np.zeros((banner_h, w, 3), dtype=np.uint8)
            footer[:] = (20, 24, 32)
            cv2.line(footer, (0, 0), (w, 0), (56, 189, 248), 2)  # Cyan accent line

            sign_info = f"RESULT: [{char_str}] ({conf_pct}%)"
            text_info = f"Spelled: '{self.spelled_text}'"
            meta_info = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | FPS: {result.get('fps', 0):.1f} | Latency: {result.get('latency_ms', 0):.1f}ms" if result else ""

            cv2.putText(footer, sign_info, (14, 29), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 180), 1, cv2.LINE_AA)
            cv2.putText(footer, text_info, (240, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(footer, meta_info, (max(w - 420, 480), 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 220), 1, cv2.LINE_AA)

            final_composite = np.vstack([composite, footer])
            res_filename = f"result_{prefix}_{char_str}_{conf_pct}pct_{timestamp}.png"
            res_path = out_dir / res_filename
            cv2.imwrite(str(res_path), final_composite, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if fs_path is None:
                fs_path = res_path
                fs_filename = res_filename

        # 4. CANNY EDGE 64x64 PREVIEW
        canny_path = None
        if prep_preview is not None and getattr(prep_preview, "size", 0) > 0:
            canny_filename = f"canny_{char_str}_{conf_pct}pct_{timestamp}.png"
            canny_path = out_dir / canny_filename
            scaled_canny = cv2.resize(prep_preview, (128, 128), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(canny_path), scaled_canny, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        primary_path = fs_path or win_path or res_path
        primary_filename = fs_filename or win_filename or res_filename or "snapshot.png"

        return {
            "fullscreen_path": str(fs_path) if fs_path else None,
            "window_path": str(win_path) if win_path else None,
            "result_path": str(res_path) if res_path else None,
            "canny_path": str(canny_path) if canny_path else None,
            "snapshot_path": str(primary_path),
            "filename": primary_filename,
            "char": char_str,
            "confidence": conf_pct
        }

    def add_space(self):
        self.spelled_text += " "

    def backspace(self):
        if self.spelled_text:
            self.spelled_text = self.spelled_text[:-1]

    def reset_smoothing(self):
        """Resets temporal smoothing queues and stability timers without clearing spelled text."""
        self.last_predicted_char = None
        self.char_hold_start_time = None
        self.char_already_added = False
        self.prob_buffer.clear()
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()

    def clear_text(self):
        self.spelled_text = ""
        self.reset_smoothing()

    def release(self):
        if hasattr(self, "hands") and self.hands:
            self.hands.close()
        if hasattr(self, "static_hands") and self.static_hands:
            self.static_hands.close()
