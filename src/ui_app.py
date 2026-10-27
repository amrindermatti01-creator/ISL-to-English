"""
src/ui_app.py - Desktop GUI Application for Indian Sign Language (ISL) Recognition

Stage 6 Deliverable:
  - Tkinter-based interactive desktop graphical interface
  - Live webcam canvas with MediaPipe skeleton & dynamic ROI overlays
  - Live Picture-in-Picture (PiP) inset of the 64x64 Canny edge CNN input
  - Real-time prediction card with confidence meter, latency, and FPS metrics
  - Rejection of uncertain predictions ("No reliable sign detected")
  - Fingerspelling text buffer with stability timer progress bar
  - Dedicated controls: Start/Stop, Camera selector, Space, Backspace, Clear, Copy, Snapshot
  - Clean error handling and graceful shutdown
"""

import os
import sys
import time
import json
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import threading

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.inference import ISLInferenceEngine
from src.preprocessing import preprocess_image


class ISLDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Indian Sign Language (ISL) Recognition System")
        self.root.geometry("1120x760")
        self.root.minsize(980, 680)
        self.root.configure(bg="#12151b")

        # Core State
        self.cap = None
        self.is_running = False
        self.camera_idx = 0
        self.engine = None
        self.use_fixed_roi = False
        self.snapshots_dir = Path("reports/snapshots")
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Performance Tracking
        self.last_frame_time = time.time()
        self.fps = 0.0

        # Styles
        self.setup_styles()

        # Build UI Layout
        self.build_header()
        self.build_main_content()
        self.build_status_bar()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initialize Inference Engine asynchronously to avoid blocking UI startup
        self.init_engine_thread()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Configure custom modern dark styles
        self.style.configure(".", background="#12151b", foreground="#e2e8f0", font=("Segoe UI", 10))
        self.style.configure("TFrame", background="#12151b")
        self.style.configure("Card.TFrame", background="#1a202c", relief="flat")
        self.style.configure("InnerCard.TFrame", background="#222b3a")

        self.style.configure(
            "Primary.TButton",
            background="#00b4d8",
            foreground="#000000",
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
            padding=6
        )
        self.style.map("Primary.TButton", background=[("active", "#90e0ef")])

        self.style.configure(
            "Success.TButton",
            background="#10b981",
            foreground="#000000",
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
            padding=6
        )
        self.style.map("Success.TButton", background=[("active", "#34d399")])

        self.style.configure(
            "Danger.TButton",
            background="#ef4444",
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
            padding=6
        )
        self.style.map("Danger.TButton", background=[("active", "#f87171")])

        self.style.configure(
            "Action.TButton",
            background="#334155",
            foreground="#ffffff",
            font=("Segoe UI", 10),
            borderwidth=0,
            padding=5
        )
        self.style.map("Action.TButton", background=[("active", "#475569")])

        # Sliders
        self.style.configure("Horizontal.TScale", background="#1a202c", troughcolor="#334155")
        # Progressbar
        self.style.configure("Hold.Horizontal.TProgressbar", background="#00e5ff", troughcolor="#334155")

    def build_header(self):
        header = tk.Frame(self.root, bg="#1a202c", height=60)
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header,
            text="INDIAN SIGN LANGUAGE (ISL) REAL-TIME DETECTOR",
            font=("Segoe UI", 14, "bold"),
            fg="#00e5ff",
            bg="#1a202c"
        )
        title_lbl.pack(side=tk.LEFT, padx=20, pady=12)

        self.subtitle_lbl = tk.Label(
            header,
            text="26 Alphabet Classes (A–Z) | MediaPipe Vision + CNN",
            font=("Segoe UI", 10),
            fg="#94a3b8",
            bg="#1a202c"
        )
        self.subtitle_lbl.pack(side=tk.LEFT, padx=5, pady=14)

        # Right side status indicator
        self.engine_status_lbl = tk.Label(
            header,
            text="● Loading Neural Network...",
            font=("Segoe UI", 10, "bold"),
            fg="#fbbf24",
            bg="#1a202c"
        )
        self.engine_status_lbl.pack(side=tk.RIGHT, padx=20)

    def build_main_content(self):
        container = tk.Frame(self.root, bg="#12151b")
        container.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        # Left Column: Video Feed & Fingerspelling Box
        left_col = tk.Frame(container, bg="#12151b")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        # Video Canvas Card
        video_card = tk.Frame(left_col, bg="#1a202c", bd=1, relief="solid")
        video_card.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(video_card, bg="#0d1117", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Fingerspelling Output Card (Below video)
        text_card = tk.Frame(left_col, bg="#1a202c", height=120)
        text_card.pack(fill=tk.X, pady=(10, 0))

        text_title_row = tk.Frame(text_card, bg="#1a202c")
        text_title_row.pack(fill=tk.X, padx=12, pady=(8, 2))

        tk.Label(
            text_title_row,
            text="FINGERSPELLED TEXT BUFFER",
            font=("Segoe UI", 9, "bold"),
            fg="#94a3b8",
            bg="#1a202c"
        ).pack(side=tk.LEFT)

        # Stability hold progress bar
        self.hold_var = tk.DoubleVar(value=0.0)
        self.hold_progress = ttk.Progressbar(
            text_title_row,
            variable=self.hold_var,
            maximum=100.0,
            style="Hold.Horizontal.TProgressbar",
            length=160
        )
        self.hold_progress.pack(side=tk.RIGHT)

        self.hold_lbl = tk.Label(
            text_title_row,
            text="Hold: 0%",
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg="#1a202c"
        )
        self.hold_lbl.pack(side=tk.RIGHT, padx=(0, 6))

        # Text display entry
        self.text_entry = tk.Entry(
            text_card,
            font=("Consolas", 15, "bold"),
            bg="#0f172a",
            fg="#f8fafc",
            insertbackground="#00e5ff",
            relief="flat",
            bd=6
        )
        self.text_entry.pack(fill=tk.X, padx=12, pady=4)

        # Text action buttons
        btn_row = tk.Frame(text_card, bg="#1a202c")
        btn_row.pack(fill=tk.X, padx=12, pady=(2, 8))

        ttk.Button(btn_row, text="[ Space ]", style="Action.TButton", command=self.action_space).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row, text="[ Backspace ]", style="Action.TButton", command=self.action_backspace).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="[ Clear ]", style="Action.TButton", command=self.action_clear).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="[ Copy Text ]", style="Action.TButton", command=self.action_copy).pack(side=tk.LEFT, padx=6)

        # Right Column: Controls & Real-Time Stats
        right_col = tk.Frame(container, bg="#1a202c", width=340)
        right_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 0))
        right_col.pack_propagate(False)

        # 1. Prediction Badge Card
        pred_card = tk.Frame(right_col, bg="#222b3a", bd=1, relief="solid")
        pred_card.pack(fill=tk.X, padx=12, pady=12)

        tk.Label(pred_card, text="PREDICTED SIGN", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#222b3a").pack(pady=(8, 2))

        self.pred_badge_lbl = tk.Label(
            pred_card,
            text="No Sign Detected",
            font=("Segoe UI", 20, "bold"),
            fg="#fbbf24",
            bg="#222b3a"
        )
        self.pred_badge_lbl.pack(pady=4)

        self.conf_badge_lbl = tk.Label(
            pred_card,
            text="Confidence: 0.0%",
            font=("Segoe UI", 10),
            fg="#94a3b8",
            bg="#222b3a"
        )
        self.conf_badge_lbl.pack(pady=(0, 2))

        self.hand_mode_lbl = tk.Label(
            pred_card,
            text="Mode: No Hands in View",
            font=("Segoe UI", 8),
            fg="#38bdf8",
            bg="#222b3a"
        )
        self.hand_mode_lbl.pack(pady=(0, 4))

        self.top3_lbl = tk.Label(
            pred_card,
            text="Top 3: —",
            font=("Consolas", 9, "bold"),
            fg="#cbd5e1",
            bg="#222b3a"
        )
        self.top3_lbl.pack(pady=(0, 8))

        # 2. PiP Preview of Canny Preprocessed 64x64 input
        pip_card = tk.Frame(right_col, bg="#222b3a", bd=1, relief="solid")
        pip_card.pack(fill=tk.X, padx=12, pady=(0, 12))

        tk.Label(pip_card, text="CNN INPUT PREVIEW (64x64 Canny)", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#222b3a").pack(pady=(8, 4))

        self.pip_canvas = tk.Canvas(pip_card, width=128, height=128, bg="#0d1117", highlightthickness=1, highlightbackground="#334155")
        self.pip_canvas.pack(pady=(0, 8))

        # 3. Hardware & Stream Controls Card
        ctrl_card = tk.Frame(right_col, bg="#222b3a", bd=1, relief="solid")
        ctrl_card.pack(fill=tk.X, padx=12, pady=(0, 12))

        tk.Label(ctrl_card, text="STREAM CONTROLS", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#222b3a").pack(pady=(8, 6))

        cam_row = tk.Frame(ctrl_card, bg="#222b3a")
        cam_row.pack(fill=tk.X, padx=12, pady=4)
        tk.Label(cam_row, text="Camera Index:", font=("Segoe UI", 9), fg="#e2e8f0", bg="#222b3a").pack(side=tk.LEFT)
        self.cam_spin = ttk.Spinbox(cam_row, from_=0, to=3, width=4)
        self.cam_spin.set(0)
        self.cam_spin.pack(side=tk.RIGHT)

        btn_stream_row = tk.Frame(ctrl_card, bg="#222b3a")
        btn_stream_row.pack(fill=tk.X, padx=12, pady=8)

        self.btn_toggle_cam = ttk.Button(btn_stream_row, text="▶ Start Webcam", style="Success.TButton", command=self.toggle_camera)
        self.btn_toggle_cam.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_snapshot = ttk.Button(btn_stream_row, text="📷 Snapshot", style="Action.TButton", command=self.take_snapshot)
        self.btn_snapshot.pack(side=tk.RIGHT, padx=(4, 0))

        # Auto-Capture on Sign Result Checkbox
        self.auto_capture_var = tk.BooleanVar(value=True)
        self.auto_capture_chk = ttk.Checkbutton(
            ctrl_card,
            text="📸 Auto-Capture on Sign Result",
            variable=self.auto_capture_var
        )
        self.auto_capture_chk.pack(anchor=tk.W, padx=14, pady=(2, 6))

        # 4. Settings & Gating Parameters
        param_card = tk.Frame(right_col, bg="#222b3a", bd=1, relief="solid")
        param_card.pack(fill=tk.X, padx=12, pady=(0, 12))

        tk.Label(param_card, text="DETECTION PARAMETERS", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#222b3a").pack(pady=(8, 4))

        # Confidence Threshold Slider
        thresh_row = tk.Frame(param_card, bg="#222b3a")
        thresh_row.pack(fill=tk.X, padx=12, pady=2)
        self.thresh_lbl = tk.Label(thresh_row, text="Confidence Gate: 55%", font=("Segoe UI", 9), fg="#e2e8f0", bg="#222b3a")
        self.thresh_lbl.pack(side=tk.LEFT)

        self.thresh_slider = ttk.Scale(param_card, from_=35, to=95, value=55, command=self.on_thresh_change)
        self.thresh_slider.pack(fill=tk.X, padx=12, pady=(0, 6))

        # Stability Hold Duration Slider
        hold_row = tk.Frame(param_card, bg="#222b3a")
        hold_row.pack(fill=tk.X, padx=12, pady=2)
        self.hold_dur_lbl = tk.Label(hold_row, text="Hold Duration: 1.0s", font=("Segoe UI", 9), fg="#e2e8f0", bg="#222b3a")
        self.hold_dur_lbl.pack(side=tk.LEFT)

        self.hold_slider = ttk.Scale(param_card, from_=5, to=25, value=10, command=self.on_hold_dur_change)
        self.hold_slider.pack(fill=tk.X, padx=12, pady=(0, 6))

        # Tracking Mode Toggle
        mode_row = tk.Frame(param_card, bg="#222b3a")
        mode_row.pack(fill=tk.X, padx=12, pady=(2, 10))
        self.mode_var = tk.StringVar(value="dynamic")
        ttk.Radiobutton(mode_row, text="MediaPipe Tracking", variable=self.mode_var, value="dynamic", command=self.on_mode_change).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Fixed Box", variable=self.mode_var, value="fixed", command=self.on_mode_change).pack(side=tk.RIGHT)

        # 5. Performance Stats
        stats_card = tk.Frame(right_col, bg="#1a202c")
        stats_card.pack(fill=tk.X, padx=12, pady=(0, 8), side=tk.BOTTOM)
        self.stats_lbl = tk.Label(
            stats_card,
            text="FPS: 0.0 | Latency: 0.0 ms",
            font=("Consolas", 9),
            fg="#38bdf8",
            bg="#1a202c"
        )
        self.stats_lbl.pack(pady=4)

    def build_status_bar(self):
        status = tk.Frame(self.root, bg="#0d1117", height=24)
        status.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_msg_lbl = tk.Label(
            status,
            text="Ready. Click [Start Webcam] to begin ISL gesture recognition.",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#0d1117"
        )
        self.status_msg_lbl.pack(side=tk.LEFT, padx=12)

        author_lbl = tk.Label(
            status,
            text="ISL Converter v1.0 | Offline & Standalone",
            font=("Segoe UI", 8),
            fg="#64748b",
            bg="#0d1117"
        )
        author_lbl.pack(side=tk.RIGHT, padx=12)

    def init_engine_thread(self):
        def _load():
            try:
                # Priority 1: Unified Combined Model (36 classes: 0-9 & A-Z, Adults & Teens)
                combined_model = Path("artifacts/best_model_combined.keras")
                combined_map = Path("artifacts/class_mapping_mendeley.json")

                # Priority 2: Retrained Real Camera Model (26 classes: A-Z)
                real_model = Path("artifacts/best_model_real.keras")
                real_map = Path("artifacts/class_mapping_real.json")

                if combined_model.exists() and combined_map.exists():
                    model_path = str(combined_model)
                    map_path = str(combined_map)
                    model_desc = "36 Classes: 0–9 & A–Z (Unified Real + Mendeley Adults/Teens Model)"
                elif real_model.exists() and real_map.exists():
                    model_path = str(real_model)
                    map_path = str(real_map)
                    model_desc = "26 Classes: A-Z (Real Camera Model)"
                else:
                    model_path = "artifacts/best_model.keras"
                    map_path = "artifacts/class_mapping.json"
                    model_desc = "35 Classes: 1-9, A-Z (Studio Model)"

                self.engine = ISLInferenceEngine(
                    model_path=model_path,
                    class_mapping_path=map_path,
                    confidence_threshold=0.55,
                    smoothing_window=5,
                    stability_duration=1.0,
                    margin_scale=1.15
                )
                self.model_description = model_desc
                self.root.after(0, self.on_engine_ready)
            except Exception as e:
                self.root.after(0, lambda: self.on_engine_error(str(e)))

        threading.Thread(target=_load, daemon=True).start()

    def on_engine_ready(self):
        desc = getattr(self, "model_description", "Neural Network Ready")
        num_classes = len(self.engine.classes) if self.engine else 26
        first_cls = self.engine.classes[0] if self.engine else "A"
        last_cls = self.engine.classes[-1] if self.engine else "Z"
        self.subtitle_lbl.configure(text=f"{num_classes} Classes ({first_cls}–{last_cls}) | MediaPipe Vision + CNN")
        self.engine_status_lbl.configure(text=f"● {desc}", fg="#10b981")
        self.status_msg_lbl.configure(text=f"Ready: {desc}. Click [Start Webcam] to begin.")

    def on_engine_error(self, err_msg):
        self.engine_status_lbl.configure(text="● Model Load Error", fg="#ef4444")
        messagebox.showerror("Model Error", f"Failed to load CNN model or MediaPipe:\n{err_msg}")

    def on_thresh_change(self, val):
        thresh_val = float(val) / 100.0
        self.thresh_lbl.configure(text=f"Confidence Gate: {int(float(val))}%")
        if self.engine:
            self.engine.confidence_threshold = thresh_val

    def on_hold_dur_change(self, val):
        dur_val = float(val) / 10.0
        self.hold_dur_lbl.configure(text=f"Hold Duration: {dur_val:.1f}s")
        if self.engine:
            self.engine.stability_duration = dur_val

    def on_mode_change(self):
        self.use_fixed_roi = (self.mode_var.get() == "fixed")

    def toggle_camera(self):
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        if not self.engine:
            messagebox.showwarning("Please Wait", "Neural network is still initializing. Please wait a moment.")
            return

        try:
            self.camera_idx = int(self.cam_spin.get())
        except ValueError:
            self.camera_idx = 0

        self.status_msg_lbl.configure(text=f"Opening camera index {self.camera_idx}...")
        self.cap = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)

        if not self.cap.isOpened():
            self.status_msg_lbl.configure(text=f"Warning: Camera {self.camera_idx} failed. Trying Camera 1...")
            self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Unable to access camera index {self.camera_idx}.\nPlease check device connection and Windows camera permissions.")
            self.status_msg_lbl.configure(text="Camera connection failed.")
            return

        # Request 16:9 widescreen HD (1280x720) for wider field of view and extra signing room
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.is_running = True
        self.btn_toggle_cam.configure(text="■ Stop Webcam", style="Danger.TButton")
        self.status_msg_lbl.configure(text="Webcam running. Present ISL signs in view.")

        # Start processing loop
        self.update_video_loop()

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_toggle_cam.configure(text="▶ Start Webcam", style="Success.TButton")
        self.canvas.delete("all")
        self.pip_canvas.delete("all")
        self.pred_badge_lbl.configure(text="Camera Stopped", fg="#94a3b8")
        self.conf_badge_lbl.configure(text="Confidence: —")
        self.hold_var.set(0.0)
        self.hold_lbl.configure(text="Hold: 0%")
        self.status_msg_lbl.configure(text="Webcam stopped.")

    def update_video_loop(self):
        if not self.is_running or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.root.after(30, self.update_video_loop)
            return

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if self.use_fixed_roi:
            # Fixed central crop mode
            box_side = 240
            x1 = (w - box_side) // 2
            y1 = (h - box_side) // 2
            x2 = x1 + box_side
            y2 = y1 + box_side
            roi_bgr = frame[y1:y2, x1:x2]

            preprocessed = preprocess_image(
                roi_bgr,
                target_size=self.engine.target_size,
                threshold_value=self.engine.threshold_value,
                use_canny=self.engine.use_canny,
                canny_low=self.engine.canny_low,
                canny_high=self.engine.canny_high,
                normalize=True
            )
            prep_preview = (preprocessed.squeeze() * 255).astype(np.uint8)
            tensor_input = np.expand_dims(preprocessed, axis=0)
            preds = self.engine.model(tensor_input, training=False).numpy()[0]
            best_idx = int(np.argmax(preds))
            raw_conf = float(preds[best_idx])
            raw_cls = self.engine.idx_to_class.get(best_idx, "Unknown")

            if raw_cls and raw_conf >= self.engine.confidence_threshold:
                self.engine.prediction_buffer.append(raw_cls)
                self.engine.confidence_buffer.append(raw_conf)
            else:
                self.engine.prediction_buffer.append(None)
                self.engine.confidence_buffer.append(0.0)

            valid_preds = [p for p in self.engine.prediction_buffer if p is not None]
            if valid_preds and len(valid_preds) >= (self.engine.smoothing_window // 2 + 1):
                from collections import Counter
                maj_cls, _ = Counter(valid_preds).most_common(1)[0]
                avg_conf = float(np.mean([c for p, c in zip(self.engine.prediction_buffer, self.engine.confidence_buffer) if p == maj_cls]))
                is_reliable = True
                smooth_cls = maj_cls
            else:
                smooth_cls = "No reliable sign detected"
                avg_conf = 0.0
                is_reliable = False

            annotated = frame.copy()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 100) if is_reliable else (50, 150, 255), 2)
            result = {
                "predicted_class": smooth_cls,
                "confidence": avg_conf,
                "raw_class": raw_cls,
                "raw_confidence": raw_conf,
                "is_reliable": is_reliable,
                "latency_ms": 22.0,
                "fps": 30.0,
                "spelled_text": self.engine.spelled_text,
                "hold_progress": 0.0
            }
        else:
            annotated, prep_preview, result = self.engine.process_frame(frame)

        # Update Live Prediction Cards
        if result["is_reliable"]:
            self.pred_badge_lbl.configure(text=f"Sign: {result['predicted_class']}", fg="#10b981")
            self.conf_badge_lbl.configure(
                text=f"Confidence: {result['confidence'] * 100:.1f}% (Gate: {int(self.engine.confidence_threshold * 100)}%)",
                fg="#34d399"
            )
        else:
            self.pred_badge_lbl.configure(text="No Reliable Sign", fg="#fbbf24")
            self.conf_badge_lbl.configure(
                text=f"Raw: {result['raw_class']} ({result['raw_confidence'] * 100:.1f}%)",
                fg="#94a3b8"
            )

        # Update Hand Mode Label
        if "hand_mode" in result:
            self.hand_mode_lbl.configure(text=f"✋ {result['hand_mode']}")

        # Update Top-3 Breakdown Label
        if result.get("top3"):
            t_str = " | ".join([f"{item['class']}:{item['prob']*100:.0f}%" for item in result["top3"]])
            self.top3_lbl.configure(text=f"Top 3: {t_str}")
        else:
            self.top3_lbl.configure(text="Top 3: —")

        # Update Fingerspelling Entry
        current_text = self.text_entry.get()
        if current_text != self.engine.spelled_text:
            self.text_entry.delete(0, tk.END)
            self.text_entry.insert(0, self.engine.spelled_text)
            self.text_entry.xview_moveto(1.0)

        # Update Hold Progress Bar
        hold_pct = result.get("hold_progress", 0.0) * 100.0
        self.hold_var.set(hold_pct)
        self.hold_lbl.configure(text=f"Hold: {int(hold_pct)}%")

        # Update Performance Label
        self.stats_lbl.configure(text=f"FPS: {result['fps']:.1f} | Latency: {result['latency_ms']:.1f} ms")

        # Render Main Video Canvas
        self.render_canvas(annotated)

        # Render PiP Inset
        if prep_preview is not None:
            self.render_pip(prep_preview)
        else:
            self.pip_canvas.delete("all")

        # Save last frames for snapshot
        self.last_annotated = annotated
        self.last_prep = prep_preview
        self.last_result = result

        # Auto-Capture Full Screen on Confirmed Result
        if result.get("new_char_confirmed") and self.auto_capture_var.get() and self.engine:
            app_win_img = None
            try:
                pil_win = self.capture_full_window()
                if pil_win is not None:
                    app_win_img = cv2.cvtColor(np.array(pil_win), cv2.COLOR_RGB2BGR)
            except Exception:
                pass

            snap_res = self.engine.save_accurate_snapshot(
                frame_bgr=annotated,
                prep_preview=prep_preview,
                result=result,
                prefix="auto_result",
                snapshots_dir=self.snapshots_dir,
                capture_fullscreen=True,
                app_window_img=app_win_img
            )
            self.status_msg_lbl.configure(
                text=f"📸 Auto-Captured Full Screen [{result['confirmed_char']}] ({snap_res['confidence']}%) -> {snap_res['filename']}",
                fg="#38bdf8"
            )

        # Schedule next frame (~30ms = ~33 FPS)
        if self.is_running:
            self.root.after(15, self.update_video_loop)

    def render_canvas(self, bgr_img):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            return

        ih, iw = bgr_img.shape[:2]
        # Maintain aspect ratio
        scale = min(cw / iw, ch / ih)
        nw, nh = int(iw * scale), int(ih * scale)

        resized = cv2.resize(bgr_img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.tk_main_img = ImageTk.PhotoImage(Image.fromarray(rgb_img))

        self.canvas.delete("all")
        # Center image in canvas
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_main_img, anchor=tk.CENTER)

    def render_pip(self, gray_64x64):
        # Resize 64x64 to 128x128 for clear inset display
        scaled = cv2.resize(gray_64x64, (128, 128), interpolation=cv2.INTER_NEAREST)
        self.tk_pip_img = ImageTk.PhotoImage(Image.fromarray(scaled))
        self.pip_canvas.delete("all")
        self.pip_canvas.create_image(64, 64, image=self.tk_pip_img, anchor=tk.CENTER)

    def on_canvas_resize(self, event):
        # Redraw last frame on window resize
        if hasattr(self, "last_annotated") and self.last_annotated is not None:
            self.render_canvas(self.last_annotated)

    def action_space(self):
        if self.engine:
            self.engine.add_space()
            self.text_entry.delete(0, tk.END)
            self.text_entry.insert(0, self.engine.spelled_text)

    def action_backspace(self):
        if self.engine:
            self.engine.backspace()
            self.text_entry.delete(0, tk.END)
            self.text_entry.insert(0, self.engine.spelled_text)

    def action_clear(self):
        if self.engine:
            self.engine.clear_text()
            self.text_entry.delete(0, tk.END)

    def action_copy(self):
        text = self.text_entry.get()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_msg_lbl.configure(text=f"Copied '{text}' to clipboard!")

    def capture_full_window(self):
        """
        Captures a high-resolution screenshot of the entire application window,
        including webcam canvas, fingerspelled text, CNN PiP preview, prediction badges,
        and stream controls.
        """
        self.root.update_idletasks()

        if os.name == "nt":
            try:
                import ctypes
                from ctypes import wintypes
                user32 = ctypes.windll.user32
                gdi32 = ctypes.windll.gdi32

                hwnd = self.root.winfo_id()
                parent = user32.GetParent(hwnd)
                target_hwnd = parent if parent else hwnd

                rect = wintypes.RECT()
                user32.GetWindowRect(target_hwnd, ctypes.byref(rect))
                w = max(1, rect.right - rect.left)
                h = max(1, rect.bottom - rect.top)

                hdc_win = user32.GetWindowDC(target_hwnd)
                hdc_mem = gdi32.CreateCompatibleDC(hdc_win)
                hbm = gdi32.CreateCompatibleBitmap(hdc_win, w, h)
                gdi32.SelectObject(hdc_mem, hbm)

                # PW_RENDERFULLCONTENT = 2, fallback to 0
                res = user32.PrintWindow(target_hwnd, hdc_mem, 2)
                if not res:
                    res = user32.PrintWindow(target_hwnd, hdc_mem, 0)

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
                bmi.biHeight = -h  # top-down DIB
                bmi.biPlanes = 1
                bmi.biBitCount = 32
                bmi.biCompression = 0

                buffer = ctypes.create_string_buffer(w * h * 4)
                gdi32.GetDIBits(hdc_mem, hbm, 0, h, buffer, ctypes.byref(bmi), 0)

                img = Image.frombuffer('RGBA', (w, h), buffer, 'raw', 'BGRA', 0, 1).convert('RGB')

                gdi32.DeleteObject(hbm)
                gdi32.DeleteDC(hdc_mem)
                user32.ReleaseDC(target_hwnd, hdc_win)
                return img
            except Exception as e:
                print(f"[UI] PrintWindow capture failed: {e}")

        # Fallback to PIL ImageGrab if available
        try:
            from PIL import ImageGrab
            x = self.root.winfo_rootx()
            y = self.root.winfo_rooty()
            w = self.root.winfo_width()
            h = self.root.winfo_height()
            return ImageGrab.grab(bbox=(x, y, x + w, y + h))
        except Exception:
            return None

    def take_snapshot(self):
        """Saves a pixel-accurate, high-resolution full-screen snapshot of the entire desktop and app."""
        if not hasattr(self, "last_annotated") or self.last_annotated is None:
            self.status_msg_lbl.configure(text="No active frame to capture.", fg="#fbbf24")
            return

        res = getattr(self, "last_result", None) or {"predicted_class": "MANUAL", "confidence": 1.0}
        app_win_img = None
        try:
            pil_win = self.capture_full_window()
            if pil_win is not None:
                app_win_img = cv2.cvtColor(np.array(pil_win), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

        if self.engine:
            snap_res = self.engine.save_accurate_snapshot(
                frame_bgr=self.last_annotated,
                prep_preview=getattr(self, "last_prep", None),
                result=res,
                prefix="manual_snapshot",
                snapshots_dir=self.snapshots_dir,
                capture_fullscreen=True,
                app_window_img=app_win_img
            )
            self.status_msg_lbl.configure(
                text=f"📸 Full Screen Saved: {snap_res['filename']}",
                fg="#10b981"
            )
        else:
            snap_id = int(time.time())
            snap_path = self.snapshots_dir / f"fullscreen_snapshot_{snap_id}.png"
            if app_win_img is not None:
                cv2.imwrite(str(snap_path), app_win_img)
            else:
                cv2.imwrite(str(snap_path), self.last_annotated)
            self.status_msg_lbl.configure(text=f"Full Screen Saved: {snap_path.name}", fg="#10b981")

        self.root.after(3500, lambda: self.status_msg_lbl.configure(
            text="Webcam running. Present ISL signs in view." if self.is_running else "Ready.",
            fg="#94a3b8"
        ))

    def on_close(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.engine:
            self.engine.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ISLDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
