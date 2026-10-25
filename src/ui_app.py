"""
src/ui_app.py - Desktop GUI Application for Indian Sign Language (ISL) Recognition

Features:
  - Unified Dual-Engine: 126D Landmark MLP (Primary) & 2D CNN (Legacy)
  - Tkinter-based interactive desktop graphical interface
  - Live webcam canvas with MediaPipe skeleton & dynamic ROI overlays
  - Dual Picture-in-Picture (PiP) insets: Source Hand ROI & 64x64 CNN / 126D Vector
  - Multilingual Translation to Indian Regional Languages (Hindi, Punjabi, Tamil, etc.)
  - Non-blocking Text-to-Speech (gTTS + pygame.mixer)
  - Intelligent Fingerspelling Buffer (TextBuilder) with Groq LLM & offline spell corrector
  - Dedicated controls: Start/Stop, Camera selector, Space, Backspace, Clear, Copy, Speak, Snapshot
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
from src.translator_engine import SUPPORTED_LANGUAGES


class ISLDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Indian Sign Language (ISL) Recognition & Translation Bridge")
        self.root.geometry("1160x800")
        self.root.minsize(1020, 720)
        self.root.configure(bg="#12151b")

        # Core State
        self.cap = None
        self.is_running = False
        self.camera_idx = 0
        self.engine = None
        self.snapshots_dir = Path("reports/snapshots")
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Performance Tracking
        self.last_frame_time = time.time()
        self.fps = 0.0

        # Language selection mappings
        self.lang_display_names = list(SUPPORTED_LANGUAGES.keys())
        self.current_lang_code = "hi"

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
            foreground="#f8fafc",
            font=("Segoe UI", 9),
            borderwidth=0,
            padding=5
        )
        self.style.map("Action.TButton", background=[("active", "#475569")])

        self.style.configure(
            "TProgressbar",
            thickness=6,
            troughcolor="#1e293b",
            background="#00e5ff"
        )

    def build_header(self):
        header = tk.Frame(self.root, bg="#1a202c", height=60)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        title_frame = tk.Frame(header, bg="#1a202c")
        title_frame.pack(side=tk.LEFT, padx=16, pady=8)

        title_lbl = tk.Label(
            title_frame,
            text="INDIAN SIGN LANGUAGE BRIDGE",
            font=("Segoe UI", 15, "bold"),
            fg="#f8fafc",
            bg="#1a202c"
        )
        title_lbl.pack(anchor=tk.W)

        subtitle_lbl = tk.Label(
            title_frame,
            text="Landmark Neural Translation & Multilingual Regional Speech",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1a202c"
        )
        subtitle_lbl.pack(anchor=tk.W)

        self.engine_status_lbl = tk.Label(
            header,
            text="● Initializing Engine...",
            font=("Segoe UI", 9, "bold"),
            fg="#fbbf24",
            bg="#1a202c"
        )
        self.engine_status_lbl.pack(side=tk.RIGHT, padx=20)

    def build_main_content(self):
        container = tk.Frame(self.root, bg="#12151b")
        container.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # Left Column: Video & Fingerspelling Text Area
        left_col = tk.Frame(container, bg="#12151b")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # Video Canvas Card
        video_card = tk.Frame(left_col, bg="#1a202c", bd=1, relief="solid")
        video_card.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(video_card, bg="#0d1117", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Bottom Area: English & Translated Text Buffers
        text_card = tk.Frame(left_col, bg="#1a202c", bd=1, relief="solid")
        text_card.pack(fill=tk.X, pady=(10, 0))

        # Row 1: English Sentences
        text_title_row = tk.Frame(text_card, bg="#1a202c")
        text_title_row.pack(fill=tk.X, padx=12, pady=(6, 2))

        tk.Label(text_title_row, text="ENGLISH SENTENCE BUFFER:", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#1a202c").pack(side=tk.LEFT)

        self.hold_var = tk.DoubleVar(value=0.0)
        self.hold_bar = ttk.Progressbar(text_title_row, variable=self.hold_var, maximum=100, length=100)
        self.hold_bar.pack(side=tk.RIGHT, padx=(4, 0))

        self.hold_lbl = tk.Label(text_title_row, text="Hold: 0%", font=("Segoe UI", 8), fg="#94a3b8", bg="#1a202c")
        self.hold_lbl.pack(side=tk.RIGHT, padx=(0, 6))

        self.text_entry = tk.Entry(
            text_card,
            font=("Consolas", 14, "bold"),
            bg="#0f172a",
            fg="#f8fafc",
            insertbackground="#00e5ff",
            relief="flat",
            bd=5
        )
        self.text_entry.pack(fill=tk.X, padx=12, pady=2)

        # Row 2: Regional Translation
        trans_title_row = tk.Frame(text_card, bg="#1a202c")
        trans_title_row.pack(fill=tk.X, padx=12, pady=(4, 2))

        self.trans_title_lbl = tk.Label(
            trans_title_row,
            text="REGIONAL TRANSLATION (HINDI):",
            font=("Segoe UI", 9, "bold"),
            fg="#38bdf8",
            bg="#1a202c"
        )
        self.trans_title_lbl.pack(side=tk.LEFT)

        self.trans_entry = tk.Entry(
            text_card,
            font=("Segoe UI", 13, "bold"),
            bg="#0f172a",
            fg="#38bdf8",
            insertbackground="#38bdf8",
            relief="flat",
            bd=5
        )
        self.trans_entry.pack(fill=tk.X, padx=12, pady=2)

        # Action Buttons Row
        btn_row = tk.Frame(text_card, bg="#1a202c")
        btn_row.pack(fill=tk.X, padx=12, pady=(4, 8))

        ttk.Button(btn_row, text="[ Space ]", style="Action.TButton", command=self.action_space).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_row, text="[ Backspace ]", style="Action.TButton", command=self.action_backspace).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="[ Clear ]", style="Action.TButton", command=self.action_clear).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="[ Copy Text ]", style="Action.TButton", command=self.action_copy).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="🔊 Speak Aloud", style="Primary.TButton", command=self.action_speak).pack(side=tk.RIGHT, padx=(5, 0))

        # Right Column: Controls, Insets & Settings
        right_col = tk.Frame(container, bg="#1a202c", width=340)
        right_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 0))
        right_col.pack_propagate(False)

        # 1. Prediction Badge Card
        pred_card = tk.Frame(right_col, bg="#222b3a", bd=1, relief="solid")
        pred_card.pack(fill=tk.X, padx=12, pady=10)

        tk.Label(pred_card, text="CURRENT SIGN", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#222b3a").pack(pady=(6, 2))

        self.pred_badge_lbl = tk.Label(
            pred_card,
            text="No Sign Detected",
            font=("Segoe UI", 18, "bold"),
            fg="#fbbf24",
            bg="#222b3a"
        )
        self.pred_badge_lbl.pack(pady=2)

        self.conf_badge_lbl = tk.Label(
            pred_card,
            text="Confidence: 0.0%",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#222b3a"
        )
        self.conf_badge_lbl.pack(pady=(0, 6))

        # 2. Dual Debug Canvases (Source ROI & 126D / CNN Input)
        pip_card = tk.Frame(right_col, bg="#222b3a", bd=1, relief="solid")
        pip_card.pack(fill=tk.X, padx=12, pady=(0, 10))

        tk.Label(pip_card, text="LIVE SENSOR INSETS", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#222b3a").pack(pady=(5, 3))

        pip_flex = tk.Frame(pip_card, bg="#222b3a")
        pip_flex.pack(pady=(0, 6))

        # Inset 1: Source Hand ROI
        roi_sub = tk.Frame(pip_flex, bg="#222b3a")
        roi_sub.pack(side=tk.LEFT, padx=5)
        tk.Label(roi_sub, text="Source Hand ROI", font=("Segoe UI", 8), fg="#cbd5e1", bg="#222b3a").pack()
        self.roi_canvas = tk.Canvas(roi_sub, width=105, height=105, bg="#0d1117", highlightthickness=1, highlightbackground="#334155")
        self.roi_canvas.pack(pady=2)

        # Inset 2: 126D Heatmap or 64x64 Canny
        pip_sub = tk.Frame(pip_flex, bg="#222b3a")
        pip_sub.pack(side=tk.LEFT, padx=5)
        self.pip_title_lbl = tk.Label(pip_sub, text="126D Vector", font=("Segoe UI", 8), fg="#cbd5e1", bg="#222b3a")
        self.pip_title_lbl.pack()
        self.pip_canvas = tk.Canvas(pip_sub, width=105, height=105, bg="#0d1117", highlightthickness=1, highlightbackground="#334155")
        self.pip_canvas.pack(pady=2)

        # 3. Model Engine & Translation Controls
        mod_card = tk.Frame(right_col, bg="#222b3a", bd=1, relief="solid")
        mod_card.pack(fill=tk.X, padx=12, pady=(0, 10))

        tk.Label(mod_card, text="ENGINE & TRANSLATION", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#222b3a").pack(pady=(6, 4))

        # Engine Radiobuttons
        engine_row = tk.Frame(mod_card, bg="#222b3a")
        engine_row.pack(fill=tk.X, padx=10, pady=2)
        self.engine_var = tk.StringVar(value="landmark_mlp")
        ttk.Radiobutton(engine_row, text="Landmark MLP (<1ms)", variable=self.engine_var, value="landmark_mlp", command=self.on_engine_change).pack(side=tk.LEFT)
        ttk.Radiobutton(engine_row, text="Legacy CNN", variable=self.engine_var, value="cnn_legacy", command=self.on_engine_change).pack(side=tk.RIGHT)

        # Language Selector Combobox
        lang_row = tk.Frame(mod_card, bg="#222b3a")
        lang_row.pack(fill=tk.X, padx=10, pady=4)
        tk.Label(lang_row, text="Target Language:", font=("Segoe UI", 8), fg="#e2e8f0", bg="#222b3a").pack(side=tk.LEFT)
        self.lang_combo = ttk.Combobox(lang_row, values=self.lang_display_names, state="readonly", width=14)
        self.lang_combo.set("Hindi (हिंदी)")
        self.lang_combo.bind("<<ComboboxSelected>>", self.on_language_selected)
        self.lang_combo.pack(side=tk.RIGHT)

        # Speech (TTS) Toggle Checkbox
        tts_row = tk.Frame(mod_card, bg="#222b3a")
        tts_row.pack(fill=tk.X, padx=10, pady=(2, 6))
        self.tts_var = tk.BooleanVar(value=False)
        self.tts_check = tk.Checkbutton(
            tts_row,
            text="Automatic Voice Speech (TTS)",
            variable=self.tts_var,
            command=self.on_tts_toggle,
            fg="#e2e8f0",
            bg="#222b3a",
            selectcolor="#0f172a",
            activebackground="#222b3a",
            activeforeground="#00e5ff",
            font=("Segoe UI", 8)
        )
        self.tts_check.pack(side=tk.LEFT)

        # 4. Stream Controls Card
        ctrl_card = tk.Frame(right_col, bg="#222b3a", bd=1, relief="solid")
        ctrl_card.pack(fill=tk.X, padx=12, pady=(0, 10))

        tk.Label(ctrl_card, text="STREAM CONTROLS", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#222b3a").pack(pady=(6, 4))

        cam_row = tk.Frame(ctrl_card, bg="#222b3a")
        cam_row.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(cam_row, text="Camera Index:", font=("Segoe UI", 8), fg="#e2e8f0", bg="#222b3a").pack(side=tk.LEFT)
        self.cam_spin = ttk.Spinbox(cam_row, from_=0, to=3, width=4)
        self.cam_spin.set(0)
        self.cam_spin.pack(side=tk.RIGHT)

        btn_stream_row = tk.Frame(ctrl_card, bg="#222b3a")
        btn_stream_row.pack(fill=tk.X, padx=10, pady=6)

        self.btn_toggle_cam = ttk.Button(btn_stream_row, text="▶ Start Webcam", style="Success.TButton", command=self.toggle_camera)
        self.btn_toggle_cam.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_snapshot = ttk.Button(btn_stream_row, text="📷 Snapshot", style="Action.TButton", command=self.take_snapshot)
        self.btn_snapshot.pack(side=tk.RIGHT, padx=(4, 0))

        # 5. Performance Stats
        stats_card = tk.Frame(right_col, bg="#1a202c")
        stats_card.pack(fill=tk.X, padx=12, pady=(0, 6), side=tk.BOTTOM)
        self.stats_lbl = tk.Label(
            stats_card,
            text="FPS: 0.0 | Latency: 0.0 ms",
            font=("Consolas", 9),
            fg="#38bdf8",
            bg="#1a202c"
        )
        self.stats_lbl.pack(pady=2)

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
            text="ISL Neural Bridge v2.0 | Dual-Engine & Multilingual",
            font=("Segoe UI", 8),
            fg="#64748b",
            bg="#0d1117"
        )
        author_lbl.pack(side=tk.RIGHT, padx=12)

    def init_engine_thread(self):
        def _load():
            try:
                self.engine = ISLInferenceEngine(
                    active_engine="landmark_mlp",
                    target_language=self.current_lang_code,
                    tts_enabled=False
                )
                self.root.after(0, self.on_engine_ready)
            except Exception as e:
                self.root.after(0, lambda: self.on_engine_error(str(e)))

        threading.Thread(target=_load, daemon=True).start()

    def on_engine_ready(self):
        self.engine_status_lbl.configure(text="● Dual-Engine Ready", fg="#10b981")
        self.status_msg_lbl.configure(text="Landmark MLP & CNN ready. Click [Start Webcam] to begin.")

    def on_engine_error(self, err_msg):
        self.engine_status_lbl.configure(text="● Engine Load Error", fg="#ef4444")
        messagebox.showerror("Model Error", f"Failed to initialize neural network engine:\n{err_msg}")

    def on_engine_change(self):
        mode = self.engine_var.get()
        if self.engine:
            self.engine.set_engine_mode(mode)
            self.pip_title_lbl.configure(text="126D Vector" if mode == "landmark_mlp" else "64x64 Canny")
            self.status_msg_lbl.configure(text=f"Engine switched to: {mode}")

    def on_language_selected(self, event=None):
        disp_name = self.lang_combo.get()
        lang_code = SUPPORTED_LANGUAGES.get(disp_name, "hi")
        self.current_lang_code = lang_code
        self.trans_title_lbl.configure(text=f"REGIONAL TRANSLATION ({disp_name.split()[0].upper()}):")
        if self.engine:
            self.engine.set_target_language(lang_code)
            self.status_msg_lbl.configure(text=f"Translation target language set to: {disp_name}")

    def on_tts_toggle(self):
        if self.engine:
            self.engine.tts_enabled = self.tts_var.get()

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
            self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Unable to access camera index {self.camera_idx}.\nPlease check device permissions.")
            self.status_msg_lbl.configure(text="Camera connection failed.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.is_running = True
        self.btn_toggle_cam.configure(text="■ Stop Webcam", style="Danger.TButton")
        self.status_msg_lbl.configure(text="Webcam running. Hold closed fist for SPACE word boundary.")

        self.update_video_loop()

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_toggle_cam.configure(text="▶ Start Webcam", style="Success.TButton")
        self.canvas.delete("all")
        self.roi_canvas.delete("all")
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

        frame = cv2.flip(frame, 1)
        annotated, source_roi, prep_preview, result = self.engine.process_frame(frame)

        # Update Badge
        if result.get("fist_detected"):
            self.pred_badge_lbl.configure(text="Fist (SPACE)", fg="#00e5ff")
            self.conf_badge_lbl.configure(text="Word Boundary Triggered", fg="#38bdf8")
        elif result["is_reliable"]:
            self.pred_badge_lbl.configure(text=f"Sign: {result['predicted_class']}", fg="#10b981")
            self.conf_badge_lbl.configure(
                text=f"Confidence: {result['confidence'] * 100:.1f}%",
                fg="#34d399"
            )
        else:
            self.pred_badge_lbl.configure(text="No Reliable Sign", fg="#fbbf24")
            self.conf_badge_lbl.configure(
                text=f"Raw: {result['raw_class']} ({result['raw_confidence'] * 100:.1f}%)",
                fg="#94a3b8"
            )

        # Update Text Entries
        current_sentence = result.get("sentence", "")
        current_raw = result.get("spelled_text", "")
        display_en = f"{current_sentence} [{current_raw}]" if current_raw else current_sentence

        self.text_entry.delete(0, tk.END)
        self.text_entry.insert(0, display_en)

        trans_str = result.get("translated_sentence", "") or result.get("translated_word", "")
        self.trans_entry.delete(0, tk.END)
        self.trans_entry.insert(0, trans_str)

        # Update Performance Label
        self.stats_lbl.configure(text=f"FPS: {result['fps']:.1f} | Latency: {result['latency_ms']:.1f} ms | Engine: {result['active_engine']}")

        # Render Main Video Canvas
        self.render_canvas(annotated)

        # Render Source ROI
        if source_roi is not None and source_roi.size > 0:
            self.render_roi(source_roi)
        else:
            self.roi_canvas.delete("all")

        # Render PiP Inset (126D heatmap or 64x64 Canny)
        if result.get("active_engine") == "landmark_mlp":
            vec_img = np.zeros((105, 105, 3), dtype=np.uint8)
            vec_img[:, :] = (20, 26, 35)
            if result.get("is_reliable"):
                cv2.putText(vec_img, "126D", (24, 45), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 150), 1, cv2.LINE_AA)
                cv2.putText(vec_img, "MLP ACTIVE", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 255, 200), 1, cv2.LINE_AA)
            else:
                cv2.putText(vec_img, "WAITING", (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 140, 160), 1, cv2.LINE_AA)
            self.render_pip_bgr(vec_img)
        elif prep_preview is not None:
            self.render_pip_gray(prep_preview)
        else:
            self.pip_canvas.delete("all")

        # Save for snapshot
        self.last_annotated = annotated
        self.last_source_roi = source_roi
        self.last_prep = prep_preview

        if self.is_running:
            self.root.after(15, self.update_video_loop)

    def render_canvas(self, bgr_img):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            return

        ih, iw = bgr_img.shape[:2]
        scale = min(cw / iw, ch / ih)
        nw, nh = int(iw * scale), int(ih * scale)

        resized = cv2.resize(bgr_img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.tk_main_img = ImageTk.PhotoImage(Image.fromarray(rgb_img))

        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_main_img, anchor=tk.CENTER)

    def render_roi(self, bgr_roi):
        rh, rw = bgr_roi.shape[:2]
        s = min(105 / rw, 105 / rh)
        nw, nh = max(1, int(round(rw * s))), max(1, int(round(rh * s)))
        resized = cv2.resize(bgr_roi, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb_roi = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.tk_roi_img = ImageTk.PhotoImage(Image.fromarray(rgb_roi))
        self.roi_canvas.delete("all")
        self.roi_canvas.create_image(52, 52, image=self.tk_roi_img, anchor=tk.CENTER)

    def render_pip_gray(self, gray_64x64):
        scaled = cv2.resize(gray_64x64, (105, 105), interpolation=cv2.INTER_NEAREST)
        self.tk_pip_img = ImageTk.PhotoImage(Image.fromarray(scaled))
        self.pip_canvas.delete("all")
        self.pip_canvas.create_image(52, 52, image=self.tk_pip_img, anchor=tk.CENTER)

    def render_pip_bgr(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        self.tk_pip_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.pip_canvas.delete("all")
        self.pip_canvas.create_image(52, 52, image=self.tk_pip_img, anchor=tk.CENTER)

    def on_canvas_resize(self, event):
        if hasattr(self, "last_annotated") and self.last_annotated is not None:
            self.render_canvas(self.last_annotated)

    def action_space(self):
        if self.engine:
            self.engine.add_space()

    def action_backspace(self):
        if self.engine:
            self.engine.backspace()

    def action_clear(self):
        if self.engine:
            self.engine.clear_text()
            self.text_entry.delete(0, tk.END)
            self.trans_entry.delete(0, tk.END)

    def action_copy(self):
        text = self.trans_entry.get() or self.text_entry.get()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_msg_lbl.configure(text=f"Copied to clipboard: '{text}'")

    def action_speak(self):
        if self.engine:
            text = self.engine.translated_sentence or self.engine.translated_word or self.text_entry.get()
            if text:
                self.status_msg_lbl.configure(text=f"Speaking aloud: '{text}' ({self.current_lang_code})")
                self.engine.translator.speak_async(text, self.current_lang_code)

    def take_snapshot(self):
        if not hasattr(self, "last_annotated") or self.last_annotated is None:
            messagebox.showinfo("Snapshot", "Start webcam to capture snapshots.")
            return

        snap_id = int(time.time())
        hud_path = self.snapshots_dir / f"gui_snapshot_{snap_id}.png"
        cv2.imwrite(str(hud_path), self.last_annotated)
        self.status_msg_lbl.configure(text=f"Snapshot saved to: {hud_path.name}")

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
