"""
src/ui_app.py - Multi-Modal Indian Sign Language (ISL) Recognition Platform

Features:
  - Obsidian Slate & Electric Cyan modern professional design system
  - 3 Dedicated Modes:
      1. 📹 Live Webcam (Real-time live streaming & hold-stability fingerspelling)
      2. 📸 Photo Mode (One-shot letter/digit classification with MediaPipe skeleton & Canny ROI)
      3. 🎥 Video Mode (Video file playback, timeline scrubbing, and continuous word assembly)
  - Picture-in-Picture (PiP) 64x64 Canny edge inset
  - Live prediction card with confidence meter, latency, and Top-3 distribution
  - Shared fingerspelling text buffer with Space, Backspace, Clear, and Copy controls
  - Full-screen snapshot and auto-capture verification
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.inference import ISLInferenceEngine
from src.preprocessing import preprocess_image
from src.reverse_translator import ReverseISLTranslator


# Visual Design System Theme Tokens
THEME = {
    "bg_dark": "#0B0F19",
    "bg_card": "#131B2E",
    "bg_inner": "#1B253B",
    "border": "#23304B",
    "cyan": "#00E5FF",
    "indigo": "#6366F1",
    "emerald": "#10B981",
    "amber": "#F59E0B",
    "rose": "#F43F5E",
    "text_white": "#F8FAFC",
    "text_muted": "#94A3B8"
}


class ISLMultiModalApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Indian Sign Language (ISL) Multi-Modal AI Recognition Platform")
        self.root.geometry("1200x820")
        self.root.minsize(1040, 720)
        self.root.configure(bg=THEME["bg_dark"])

        # Core State
        self.engine = None
        self.current_mode = "webcam"  # 'webcam' | 'photo' | 'video' | 'reverse'
        self.use_fixed_roi = False
        self.snapshots_dir = Path("reports/snapshots")
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Reverse Translation State (Speech & Text to Sign)
        self.reverse_translator = ReverseISLTranslator()
        self.reverse_sequence = []
        self.reverse_idx = 0
        self.reverse_is_playing = False
        self.reverse_anim_job = None
        self.reverse_speed_ms = 1200
        self.reverse_auto_speech = tk.BooleanVar(value=True)

        # Webcam State
        self.cap = None
        self.is_webcam_running = False
        self.camera_idx = 0

        # Video File State
        self.video_cap = None
        self.is_video_playing = False
        self.video_path = None
        self.video_total_frames = 0
        self.video_fps = 30.0
        self.video_cur_frame = 0
        self.video_scrubbing = False

        # Photo State
        self.photo_path = None
        self.photo_original_bgr = None

        # Performance & Render State
        self.fps = 0.0
        self.last_frame_time = time.time()
        self.last_annotated = None
        self.last_prep = None
        self.last_result = None

        # Setup Styling & Layout
        self.setup_styles()
        self.build_header()
        self.build_mode_selector()
        self.build_main_content()
        self.build_status_bar()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initialize Inference Engine asynchronously
        self.init_engine_thread()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Global base styles
        self.style.configure(".", background=THEME["bg_dark"], foreground=THEME["text_white"], font=("Segoe UI", 10))
        self.style.configure("TFrame", background=THEME["bg_dark"])
        self.style.configure("Card.TFrame", background=THEME["bg_card"])
        self.style.configure("InnerCard.TFrame", background=THEME["bg_inner"])

        # Buttons
        self.style.configure(
            "Primary.TButton",
            background=THEME["cyan"],
            foreground="#000000",
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
            padding=6
        )
        self.style.map("Primary.TButton", background=[("active", "#70efff")])

        self.style.configure(
            "Indigo.TButton",
            background=THEME["indigo"],
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
            padding=6
        )
        self.style.map("Indigo.TButton", background=[("active", "#818cf8")])

        self.style.configure(
            "Success.TButton",
            background=THEME["emerald"],
            foreground="#000000",
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
            padding=6
        )
        self.style.map("Success.TButton", background=[("active", "#34d399")])

        self.style.configure(
            "Danger.TButton",
            background=THEME["rose"],
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
            padding=6
        )
        self.style.map("Danger.TButton", background=[("active", "#fb7185")])

        self.style.configure(
            "Action.TButton",
            background=THEME["bg_inner"],
            foreground=THEME["text_white"],
            font=("Segoe UI", 9, "bold"),
            borderwidth=1,
            relief="solid",
            padding=5
        )
        self.style.map("Action.TButton", background=[("active", "#2a3754")])

        # Progress bars and scales
        self.style.configure("Horizontal.TScale", background=THEME["bg_card"], troughcolor=THEME["border"])
        self.style.configure("Hold.Horizontal.TProgressbar", background=THEME["cyan"], troughcolor=THEME["border"])
        self.style.configure("Video.Horizontal.TScale", background=THEME["bg_card"], troughcolor=THEME["border"])

    def build_header(self):
        header = tk.Frame(self.root, bg=THEME["bg_card"], height=64)
        header.pack(fill=tk.X, side=tk.TOP)

        title_box = tk.Frame(header, bg=THEME["bg_card"])
        title_box.pack(side=tk.LEFT, padx=20, pady=10)

        title_lbl = tk.Label(
            title_box,
            text="INDIAN SIGN LANGUAGE (ISL) MULTI-MODAL RECOGNITION",
            font=("Segoe UI Variable Display", 13, "bold"),
            fg=THEME["cyan"],
            bg=THEME["bg_card"]
        )
        title_lbl.pack(anchor=tk.W)

        self.subtitle_lbl = tk.Label(
            title_box,
            text="Unified Live Webcam • High-Res Photo • Video Fingerspelling",
            font=("Segoe UI", 9),
            fg=THEME["text_muted"],
            bg=THEME["bg_card"]
        )
        self.subtitle_lbl.pack(anchor=tk.W)

        # Right side status indicator
        self.engine_status_lbl = tk.Label(
            header,
            text="● Initializing Deep Neural Network...",
            font=("Segoe UI", 10, "bold"),
            fg=THEME["amber"],
            bg=THEME["bg_card"]
        )
        self.engine_status_lbl.pack(side=tk.RIGHT, padx=20)

    def build_mode_selector(self):
        """Builds the top segmented switcher for Webcam, Photo, Video, and Reverse modes."""
        selector_bar = tk.Frame(self.root, bg=THEME["bg_dark"], height=48)
        selector_bar.pack(fill=tk.X, padx=16, pady=(8, 4))

        tk.Label(
            selector_bar,
            text="INPUT MODE:",
            font=("Segoe UI", 9, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["bg_dark"]
        ).pack(side=tk.LEFT, padx=(4, 12))

        # Button container with pill border look
        pill_frame = tk.Frame(selector_bar, bg=THEME["bg_card"], bd=1, relief="solid")
        pill_frame.pack(side=tk.LEFT)

        self.btn_mode_webcam = tk.Button(
            pill_frame,
            text="📹  Live Webcam",
            font=("Segoe UI", 10, "bold"),
            bg=THEME["cyan"],
            fg="#000000",
            activebackground=THEME["cyan"],
            activeforeground="#000000",
            relief="flat",
            bd=0,
            padx=16,
            pady=6,
            cursor="hand2",
            command=lambda: self.switch_mode("webcam")
        )
        self.btn_mode_webcam.pack(side=tk.LEFT)

        self.btn_mode_photo = tk.Button(
            pill_frame,
            text="📸  Photo Mode",
            font=("Segoe UI", 10),
            bg=THEME["bg_card"],
            fg=THEME["text_muted"],
            activebackground=THEME["bg_inner"],
            activeforeground=THEME["text_white"],
            relief="flat",
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2",
            command=lambda: self.switch_mode("photo")
        )
        self.btn_mode_photo.pack(side=tk.LEFT)

        self.btn_mode_video = tk.Button(
            pill_frame,
            text="🎥  Video Mode",
            font=("Segoe UI", 10),
            bg=THEME["bg_card"],
            fg=THEME["text_muted"],
            activebackground=THEME["bg_inner"],
            activeforeground=THEME["text_white"],
            relief="flat",
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2",
            command=lambda: self.switch_mode("video")
        )
        self.btn_mode_video.pack(side=tk.LEFT)

        self.btn_mode_reverse = tk.Button(
            pill_frame,
            text="🗣️  Speech & Text → Sign",
            font=("Segoe UI", 10),
            bg=THEME["bg_card"],
            fg=THEME["text_muted"],
            activebackground=THEME["bg_inner"],
            activeforeground=THEME["text_white"],
            relief="flat",
            bd=0,
            padx=16,
            pady=6,
            cursor="hand2",
            command=lambda: self.switch_mode("reverse")
        )
        self.btn_mode_reverse.pack(side=tk.LEFT)

        # Mode explanation tip
        self.mode_tip_lbl = tk.Label(
            selector_bar,
            text="Continuous real-time fingerspelling recognition with stability timer.",
            font=("Segoe UI", 9, "italic"),
            fg=THEME["text_muted"],
            bg=THEME["bg_dark"]
        )
        self.mode_tip_lbl.pack(side=tk.LEFT, padx=16)

    def switch_mode(self, new_mode: str):
        """Switches UI view between Webcam, Photo, Video, and Reverse modes."""
        if self.current_mode == new_mode:
            return

        # Stop any active streaming
        if self.is_webcam_running:
            self.stop_webcam()
        if self.is_video_playing:
            self.pause_video()
        if hasattr(self, "reverse_is_playing") and self.reverse_is_playing:
            self.stop_reverse_animation()

        self.current_mode = new_mode

        # Update button visual styling
        for mode, btn in [
            ("webcam", self.btn_mode_webcam),
            ("photo", self.btn_mode_photo),
            ("video", self.btn_mode_video),
            ("reverse", self.btn_mode_reverse)
        ]:
            if mode == new_mode:
                btn.configure(bg=THEME["cyan"], fg="#000000", font=("Segoe UI", 10, "bold"))
            else:
                btn.configure(bg=THEME["bg_card"], fg=THEME["text_muted"], font=("Segoe UI", 10))

        # Update Mode Tip & Control Layout
        if new_mode == "webcam":
            self.mode_tip_lbl.configure(text="Live camera feed: Hold sign ~1.0s to accumulate fingerspelled letters.")
            self.webcam_ctrl_frame.pack(fill=tk.X, pady=(0, 10))
            self.photo_ctrl_frame.pack_forget()
            self.video_ctrl_frame.pack_forget()
            if hasattr(self, "reverse_ctrl_frame"):
                self.reverse_ctrl_frame.pack_forget()
            self.video_playback_bar.pack_forget()
            if hasattr(self, "reverse_playback_bar"):
                self.reverse_playback_bar.pack_forget()
            if hasattr(self, "pip_card"):
                self.pip_card.pack(fill=tk.X, padx=12, pady=(0, 10))
            if hasattr(self, "param_card"):
                self.param_card.pack(fill=tk.X, padx=12, pady=(0, 8))
            if hasattr(self, "btn_convert_text"):
                self.btn_convert_text.pack_forget()
            if hasattr(self, "hold_progress"):
                self.hold_progress.pack(side=tk.RIGHT)
            if hasattr(self, "hold_lbl"):
                self.hold_lbl.pack(side=tk.RIGHT, padx=(0, 8))
            if hasattr(self, "text_title_lbl"):
                self.text_title_lbl.configure(text="ASSEMBLED WORD / FINGERSPELLED TEXT BUFFER", fg=THEME["text_muted"])
        elif new_mode == "photo":
            self.mode_tip_lbl.configure(text="Upload static image (.jpg, .png): Instant offline classification.")
            self.webcam_ctrl_frame.pack_forget()
            self.photo_ctrl_frame.pack(fill=tk.X, pady=(0, 10))
            self.video_ctrl_frame.pack_forget()
            if hasattr(self, "reverse_ctrl_frame"):
                self.reverse_ctrl_frame.pack_forget()
            self.video_playback_bar.pack_forget()
            if hasattr(self, "reverse_playback_bar"):
                self.reverse_playback_bar.pack_forget()
            if hasattr(self, "pip_card"):
                self.pip_card.pack(fill=tk.X, padx=12, pady=(0, 10))
            if hasattr(self, "param_card"):
                self.param_card.pack(fill=tk.X, padx=12, pady=(0, 8))
            if hasattr(self, "btn_convert_text"):
                self.btn_convert_text.pack_forget()
            if hasattr(self, "hold_progress"):
                self.hold_progress.pack(side=tk.RIGHT)
            if hasattr(self, "hold_lbl"):
                self.hold_lbl.pack(side=tk.RIGHT, padx=(0, 8))
            if hasattr(self, "text_title_lbl"):
                self.text_title_lbl.configure(text="ASSEMBLED WORD / FINGERSPELLED TEXT BUFFER", fg=THEME["text_muted"])
        elif new_mode == "video":
            self.mode_tip_lbl.configure(text="Upload video recording (.mp4, .avi): Track sign sequence and construct full words.")
            self.webcam_ctrl_frame.pack_forget()
            self.photo_ctrl_frame.pack_forget()
            self.video_ctrl_frame.pack(fill=tk.X, pady=(0, 10))
            if hasattr(self, "reverse_ctrl_frame"):
                self.reverse_ctrl_frame.pack_forget()
            self.video_playback_bar.pack(fill=tk.X, pady=(6, 0))
            if hasattr(self, "reverse_playback_bar"):
                self.reverse_playback_bar.pack_forget()
            if hasattr(self, "pip_card"):
                self.pip_card.pack(fill=tk.X, padx=12, pady=(0, 10))
            if hasattr(self, "param_card"):
                self.param_card.pack(fill=tk.X, padx=12, pady=(0, 8))
            if hasattr(self, "btn_convert_text"):
                self.btn_convert_text.pack_forget()
            if hasattr(self, "hold_progress"):
                self.hold_progress.pack(side=tk.RIGHT)
            if hasattr(self, "hold_lbl"):
                self.hold_lbl.pack(side=tk.RIGHT, padx=(0, 8))
            if hasattr(self, "text_title_lbl"):
                self.text_title_lbl.configure(text="ASSEMBLED WORD / FINGERSPELLED TEXT BUFFER", fg=THEME["text_muted"])
        elif new_mode == "reverse":
            self.mode_tip_lbl.configure(text="Speak into mic or type English letters/words in the box below to render ISL signs.")
            self.webcam_ctrl_frame.pack_forget()
            self.photo_ctrl_frame.pack_forget()
            self.video_ctrl_frame.pack_forget()
            self.video_playback_bar.pack_forget()
            if hasattr(self, "pip_card"):
                self.pip_card.pack_forget()
            if hasattr(self, "param_card"):
                self.param_card.pack_forget()
            if hasattr(self, "reverse_ctrl_frame"):
                self.reverse_ctrl_frame.pack(fill=tk.X, pady=(0, 10))
            if hasattr(self, "reverse_playback_bar"):
                self.reverse_playback_bar.pack(fill=tk.X, pady=(6, 0))
            if hasattr(self, "hold_progress"):
                self.hold_progress.pack_forget()
            if hasattr(self, "hold_lbl"):
                self.hold_lbl.pack_forget()
            if hasattr(self, "btn_convert_text"):
                self.btn_convert_text.pack(side=tk.RIGHT, padx=6)
            if hasattr(self, "text_title_lbl"):
                self.text_title_lbl.configure(
                    text="✍️ TYPE ENGLISH LETTER OR WORD HERE (A-Z, 1-9) & PRESS ENTER (OR CLICK [SHOW ISL SIGN]):",
                    fg=THEME["cyan"]
                )
            if hasattr(self, "text_entry"):
                self.text_entry.focus_set()
                self.text_entry.select_range(0, tk.END)
            # Display default sign A on entry
            self.translate_text_to_signs("A")

        # Clear canvas if not reverse mode
        if new_mode != "reverse":
            self.canvas.delete("all")
            self.pip_canvas.delete("all")
            self.pred_badge_lbl.configure(text="Ready in " + new_mode.capitalize() + " Mode", fg=THEME["text_muted"])
            self.conf_badge_lbl.configure(text="Confidence: —")
            self.status_msg_lbl.configure(text=f"Switched to {new_mode.upper()} mode. Select input or start feed.")

    def build_main_content(self):
        container = tk.Frame(self.root, bg=THEME["bg_dark"])
        container.pack(fill=tk.BOTH, expand=True, padx=16, pady=(4, 10))

        # ----------------- LEFT COLUMN: Canvas & Text Accumulator -----------------
        left_col = tk.Frame(container, bg=THEME["bg_dark"])
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        # Video / Photo Canvas Card
        viewport_card = tk.Frame(left_col, bg=THEME["bg_card"], bd=1, relief="solid")
        viewport_card.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(viewport_card, bg="#080C14", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Video Playback Bar (Hidden unless in Video mode)
        self.video_playback_bar = tk.Frame(viewport_card, bg=THEME["bg_card"])
        # (Will be packed when in video mode)

        self.video_slider_var = tk.DoubleVar(value=0.0)
        self.video_slider = ttk.Scale(
            self.video_playback_bar,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.video_slider_var,
            style="Video.Horizontal.TScale",
            command=self.on_video_seek
        )
        self.video_slider.pack(fill=tk.X, padx=12, pady=(4, 2))

        vid_btn_row = tk.Frame(self.video_playback_bar, bg=THEME["bg_card"])
        vid_btn_row.pack(fill=tk.X, padx=12, pady=(2, 6))

        self.btn_vid_rewind = ttk.Button(vid_btn_row, text="⏮ Rewind", style="Action.TButton", command=self.rewind_video)
        self.btn_vid_rewind.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_vid_play = ttk.Button(vid_btn_row, text="▶ Play", style="Success.TButton", command=self.toggle_video_playback)
        self.btn_vid_play.pack(side=tk.LEFT, padx=4)

        self.btn_vid_stop = ttk.Button(vid_btn_row, text="⏹ Stop", style="Action.TButton", command=self.stop_video)
        self.btn_vid_stop.pack(side=tk.LEFT, padx=4)

        self.vid_time_lbl = tk.Label(
            vid_btn_row,
            text="00:00 / 00:00  (Frame 0/0)",
            font=("Consolas", 9),
            fg=THEME["text_muted"],
            bg=THEME["bg_card"]
        )
        self.vid_time_lbl.pack(side=tk.RIGHT, padx=6)

        # Reverse Playback Bar (Hidden unless in Reverse mode)
        self.reverse_playback_bar = tk.Frame(viewport_card, bg=THEME["bg_card"])
        rev_btn_row = tk.Frame(self.reverse_playback_bar, bg=THEME["bg_card"])
        rev_btn_row.pack(fill=tk.X, padx=12, pady=(4, 6))

        self.btn_rev_prev = ttk.Button(rev_btn_row, text="⏮ Prev Sign", style="Action.TButton", command=self.prev_reverse_sign)
        self.btn_rev_prev.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_rev_play = ttk.Button(rev_btn_row, text="▶ Play Fingerspelling", style="Success.TButton", command=self.toggle_reverse_animation)
        self.btn_rev_play.pack(side=tk.LEFT, padx=4)

        self.btn_rev_next = ttk.Button(rev_btn_row, text="Next Sign ⏭", style="Action.TButton", command=self.next_reverse_sign)
        self.btn_rev_next.pack(side=tk.LEFT, padx=4)

        self.btn_rev_speak = ttk.Button(rev_btn_row, text="🔊 Speak Aloud", style="Indigo.TButton", command=self.pronounce_current_sign)
        self.btn_rev_speak.pack(side=tk.LEFT, padx=8)

        self.reverse_step_lbl = tk.Label(
            rev_btn_row,
            text="Sign: 1 / 1  ('A')",
            font=("Consolas", 10, "bold"),
            fg=THEME["cyan"],
            bg=THEME["bg_card"]
        )
        self.reverse_step_lbl.pack(side=tk.RIGHT, padx=6)

        # Fingerspelling Output Card (Below Canvas)
        text_card = tk.Frame(left_col, bg=THEME["bg_card"], height=125, bd=1, relief="solid")
        text_card.pack(fill=tk.X, pady=(10, 0))

        text_title_row = tk.Frame(text_card, bg=THEME["bg_card"])
        text_title_row.pack(fill=tk.X, padx=14, pady=(8, 2))

        self.text_title_lbl = tk.Label(
            text_title_row,
            text="ASSEMBLED WORD / FINGERSPELLED TEXT BUFFER",
            font=("Segoe UI", 9, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["bg_card"]
        )
        self.text_title_lbl.pack(side=tk.LEFT)

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
            font=("Consolas", 8),
            fg=THEME["text_muted"],
            bg=THEME["bg_card"]
        )
        self.hold_lbl.pack(side=tk.RIGHT, padx=(0, 8))

        # Text display entry (Used for both Webcam assembly and direct typing in Reverse mode)
        self.text_entry = tk.Entry(
            text_card,
            font=("Consolas", 16, "bold"),
            bg="#080C14",
            fg=THEME["text_white"],
            insertbackground=THEME["cyan"],
            relief="flat",
            bd=6
        )
        self.text_entry.pack(fill=tk.X, padx=14, pady=4)
        self.text_entry.bind("<Return>", lambda e: self.on_main_text_submit())

        # Text action buttons
        btn_row = tk.Frame(text_card, bg=THEME["bg_card"])
        btn_row.pack(fill=tk.X, padx=14, pady=(2, 8))

        ttk.Button(btn_row, text="[ ␣ Space ]", style="Action.TButton", command=self.action_space).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row, text="[ ⌫ Backspace ]", style="Action.TButton", command=self.action_backspace).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="[ 🗑 Clear Buffer ]", style="Action.TButton", command=self.action_clear).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="[ 📋 Copy Text ]", style="Action.TButton", command=self.action_copy).pack(side=tk.LEFT, padx=6)

        # Dedicated Convert Button for Reverse Mode
        self.btn_convert_text = ttk.Button(
            btn_row,
            text="[ 🔍 Show ISL Sign ]",
            style="Success.TButton",
            command=self.on_main_text_submit
        )

        # ----------------- RIGHT COLUMN: Cards & Controls -----------------
        right_col = tk.Frame(container, bg=THEME["bg_card"], width=350, bd=1, relief="solid")
        right_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 0))
        right_col.pack_propagate(False)

        # 1. Prediction Badge Card
        pred_card = tk.Frame(right_col, bg=THEME["bg_inner"], bd=1, relief="solid")
        pred_card.pack(fill=tk.X, padx=12, pady=12)

        tk.Label(pred_card, text="PREDICTED SIGN / CHARACTER", font=("Segoe UI", 9, "bold"), fg=THEME["text_muted"], bg=THEME["bg_inner"]).pack(pady=(8, 2))

        self.pred_badge_lbl = tk.Label(
            pred_card,
            text="No Sign Detected",
            font=("Segoe UI Variable Display", 22, "bold"),
            fg=THEME["amber"],
            bg=THEME["bg_inner"]
        )
        self.pred_badge_lbl.pack(pady=4)

        self.conf_badge_lbl = tk.Label(
            pred_card,
            text="Confidence: 0.0%",
            font=("Segoe UI", 10),
            fg=THEME["text_muted"],
            bg=THEME["bg_inner"]
        )
        self.conf_badge_lbl.pack(pady=(0, 2))

        self.hand_mode_lbl = tk.Label(
            pred_card,
            text="Mode: No Hands in View",
            font=("Segoe UI", 8),
            fg=THEME["cyan"],
            bg=THEME["bg_inner"]
        )
        self.hand_mode_lbl.pack(pady=(0, 4))

        self.top3_lbl = tk.Label(
            pred_card,
            text="Top 3: —",
            font=("Consolas", 9, "bold"),
            fg=THEME["text_white"],
            bg=THEME["bg_inner"]
        )
        self.top3_lbl.pack(pady=(0, 8))

        # 2. PiP Preview of Canny Preprocessed 64x64 input
        self.pip_card = tk.Frame(right_col, bg=THEME["bg_inner"], bd=1, relief="solid")
        self.pip_card.pack(fill=tk.X, padx=12, pady=(0, 10))

        tk.Label(self.pip_card, text="CNN INPUT RADAR (64x64 Canny Edge)", font=("Segoe UI", 9, "bold"), fg=THEME["text_muted"], bg=THEME["bg_inner"]).pack(pady=(8, 4))

        self.pip_canvas = tk.Canvas(self.pip_card, width=128, height=128, bg="#080C14", highlightthickness=1, highlightbackground=THEME["border"])
        self.pip_canvas.pack(pady=(0, 8))

        # 3. Dynamic Mode Controls Container
        self.controls_container = tk.Frame(right_col, bg=THEME["bg_card"])
        self.controls_container.pack(fill=tk.X, padx=12, pady=(0, 8))

        # -- A. WEBCAM CONTROLS FRAME --
        self.webcam_ctrl_frame = tk.Frame(self.controls_container, bg=THEME["bg_inner"], bd=1, relief="solid")
        self.webcam_ctrl_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(self.webcam_ctrl_frame, text="LIVE WEBCAM CONTROLS", font=("Segoe UI", 9, "bold"), fg=THEME["cyan"], bg=THEME["bg_inner"]).pack(pady=(8, 4))

        cam_row = tk.Frame(self.webcam_ctrl_frame, bg=THEME["bg_inner"])
        cam_row.pack(fill=tk.X, padx=12, pady=4)
        tk.Label(cam_row, text="Camera Index:", font=("Segoe UI", 9), fg=THEME["text_white"], bg=THEME["bg_inner"]).pack(side=tk.LEFT)
        self.cam_spin = ttk.Spinbox(cam_row, from_=0, to=3, width=4)
        self.cam_spin.set(0)
        self.cam_spin.pack(side=tk.RIGHT)

        btn_stream_row = tk.Frame(self.webcam_ctrl_frame, bg=THEME["bg_inner"])
        btn_stream_row.pack(fill=tk.X, padx=12, pady=8)

        self.btn_toggle_cam = ttk.Button(btn_stream_row, text="▶ Start Webcam", style="Success.TButton", command=self.toggle_webcam)
        self.btn_toggle_cam.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_snapshot = ttk.Button(btn_stream_row, text="📷 Snapshot", style="Action.TButton", command=self.take_snapshot)
        self.btn_snapshot.pack(side=tk.RIGHT, padx=(4, 0))

        self.auto_capture_var = tk.BooleanVar(value=True)
        self.auto_capture_chk = ttk.Checkbutton(
            self.webcam_ctrl_frame,
            text="📸 Auto-Capture on Sign Result",
            variable=self.auto_capture_var
        )
        self.auto_capture_chk.pack(anchor=tk.W, padx=14, pady=(2, 6))

        # -- B. PHOTO CONTROLS FRAME (Initially unmapped) --
        self.photo_ctrl_frame = tk.Frame(self.controls_container, bg=THEME["bg_inner"], bd=1, relief="solid")

        tk.Label(self.photo_ctrl_frame, text="PHOTO RECOGNITION (OFFLINE)", font=("Segoe UI", 9, "bold"), fg=THEME["cyan"], bg=THEME["bg_inner"]).pack(pady=(8, 4))

        self.photo_file_lbl = tk.Label(
            self.photo_ctrl_frame,
            text="No image loaded",
            font=("Segoe UI", 8),
            fg=THEME["text_muted"],
            bg=THEME["bg_inner"],
            wraplength=300
        )
        self.photo_file_lbl.pack(padx=12, pady=(0, 6))

        btn_photo_row = tk.Frame(self.photo_ctrl_frame, bg=THEME["bg_inner"])
        btn_photo_row.pack(fill=tk.X, padx=12, pady=4)

        ttk.Button(btn_photo_row, text="📂 Choose Photo...", style="Primary.TButton", command=self.browse_photo).pack(fill=tk.X, pady=2)
        ttk.Button(btn_photo_row, text="➕ Append Sign to Buffer", style="Action.TButton", command=self.append_photo_sign).pack(fill=tk.X, pady=(4, 8))

        # -- C. VIDEO CONTROLS FRAME (Initially unmapped) --
        self.video_ctrl_frame = tk.Frame(self.controls_container, bg=THEME["bg_inner"], bd=1, relief="solid")

        tk.Label(self.video_ctrl_frame, text="VIDEO STREAM CONTROLS", font=("Segoe UI", 9, "bold"), fg=THEME["cyan"], bg=THEME["bg_inner"]).pack(pady=(8, 4))

        self.video_file_lbl = tk.Label(
            self.video_ctrl_frame,
            text="No video loaded",
            font=("Segoe UI", 8),
            fg=THEME["text_muted"],
            bg=THEME["bg_inner"],
            wraplength=300
        )
        self.video_file_lbl.pack(padx=12, pady=(0, 6))

        ttk.Button(self.video_ctrl_frame, text="📂 Choose Video File...", style="Indigo.TButton", command=self.browse_video).pack(fill=tk.X, padx=12, pady=(2, 6))

        # -- D. REVERSE CONTROLS FRAME (Speech & Text to ISL) --
        self.reverse_ctrl_frame = tk.Frame(self.controls_container, bg=THEME["bg_inner"], bd=1, relief="solid")

        tk.Label(self.reverse_ctrl_frame, text="ENGLISH → ISL CONVERTER", font=("Segoe UI", 9, "bold"), fg=THEME["cyan"], bg=THEME["bg_inner"]).pack(pady=(6, 3))

        # 1. Voice Speech Input
        voice_box = tk.Frame(self.reverse_ctrl_frame, bg=THEME["bg_card"], bd=1, relief="solid")
        voice_box.pack(fill=tk.X, padx=8, pady=(2, 4))

        tk.Label(voice_box, text="🎙️ VOICE INPUT (SPEAK ENGLISH)", font=("Segoe UI", 8, "bold"), fg=THEME["cyan"], bg=THEME["bg_card"]).pack(anchor=tk.W, padx=6, pady=(3, 1))

        self.btn_voice_input = tk.Button(
            voice_box,
            text="🎙️  Speak Letter / Word",
            font=("Segoe UI", 9, "bold"),
            bg=THEME["cyan"],
            fg="#000000",
            activebackground="#70efff",
            relief="flat",
            bd=0,
            pady=4,
            cursor="hand2",
            command=self.start_voice_recognition
        )
        self.btn_voice_input.pack(fill=tk.X, padx=6, pady=3)

        self.voice_status_lbl = tk.Label(
            voice_box,
            text="Mic ready. Click and speak clearly.",
            font=("Segoe UI", 8, "italic"),
            fg=THEME["text_muted"],
            bg=THEME["bg_card"],
            wraplength=300
        )
        self.voice_status_lbl.pack(padx=6, pady=(1, 4))

        # 2. Text Input Field
        text_box = tk.Frame(self.reverse_ctrl_frame, bg=THEME["bg_card"], bd=1, relief="solid")
        text_box.pack(fill=tk.X, padx=8, pady=(2, 4))

        tk.Label(text_box, text="✍️ WRITE / TYPE ENGLISH TEXT", font=("Segoe UI", 8, "bold"), fg=THEME["cyan"], bg=THEME["bg_card"]).pack(anchor=tk.W, padx=6, pady=(3, 1))

        text_input_row = tk.Frame(text_box, bg=THEME["bg_card"])
        text_input_row.pack(fill=tk.X, padx=6, pady=3)

        self.rev_input_entry = tk.Entry(
            text_input_row,
            font=("Consolas", 11, "bold"),
            bg="#080C14",
            fg=THEME["text_white"],
            insertbackground=THEME["cyan"],
            relief="flat",
            bd=3
        )
        self.rev_input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.rev_input_entry.bind("<Return>", lambda e: self.on_reverse_text_submit())

        btn_show = tk.Button(
            text_input_row,
            text="🔍 Show",
            font=("Segoe UI", 9, "bold"),
            bg=THEME["indigo"],
            fg="#ffffff",
            activebackground="#818cf8",
            relief="flat",
            bd=0,
            padx=8,
            pady=2,
            cursor="hand2",
            command=self.on_reverse_text_submit
        )
        btn_show.pack(side=tk.RIGHT)

        # 3. Quick Selector Keypad
        keypad_box = tk.Frame(self.reverse_ctrl_frame, bg=THEME["bg_card"], bd=1, relief="solid")
        keypad_box.pack(fill=tk.X, padx=8, pady=(2, 4))

        tk.Label(keypad_box, text="⚡ QUICK SELECTOR (TAP ANY SIGN)", font=("Segoe UI", 8, "bold"), fg=THEME["text_muted"], bg=THEME["bg_card"]).pack(anchor=tk.W, padx=6, pady=(3, 1))

        grid_frame = tk.Frame(keypad_box, bg=THEME["bg_card"])
        grid_frame.pack(fill=tk.X, padx=4, pady=2)

        for i, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            r = i // 7
            c = i % 7
            btn = tk.Button(
                grid_frame,
                text=char,
                font=("Consolas", 8, "bold"),
                bg=THEME["bg_inner"],
                fg=THEME["text_white"],
                activebackground=THEME["cyan"],
                activeforeground="#000000",
                relief="flat",
                bd=0,
                padx=1,
                pady=1,
                width=3,
                cursor="hand2",
                command=lambda ch=char: self.on_quick_key_click(ch)
            )
            btn.grid(row=r, column=c, padx=1, pady=1)

        digits_row = tk.Frame(keypad_box, bg=THEME["bg_card"])
        digits_row.pack(fill=tk.X, padx=4, pady=(2, 4))
        for num in "123456789":
            btn = tk.Button(
                digits_row,
                text=num,
                font=("Consolas", 8, "bold"),
                bg=THEME["bg_inner"],
                fg=THEME["amber"],
                activebackground=THEME["amber"],
                activeforeground="#000000",
                relief="flat",
                bd=0,
                padx=1,
                pady=1,
                width=2,
                cursor="hand2",
                command=lambda n=num: self.on_quick_key_click(n)
            )
            btn.pack(side=tk.LEFT, expand=True, padx=1)

        # 4. Audio Feedback Toggle Checkbutton
        chk_row = tk.Frame(self.reverse_ctrl_frame, bg=THEME["bg_inner"])
        chk_row.pack(fill=tk.X, padx=8, pady=(2, 4))
        self.auto_speak_chk = ttk.Checkbutton(
            chk_row,
            text="🔊 Auto-Pronounce Sign (TTS)",
            variable=self.reverse_auto_speech
        )
        self.auto_speak_chk.pack(anchor=tk.W)

        # 4. Settings & Gating Parameters
        self.param_card = tk.Frame(right_col, bg=THEME["bg_inner"], bd=1, relief="solid")
        self.param_card.pack(fill=tk.X, padx=12, pady=(0, 8))

        tk.Label(self.param_card, text="DETECTION TUNING", font=("Segoe UI", 9, "bold"), fg=THEME["text_muted"], bg=THEME["bg_inner"]).pack(pady=(8, 4))

        thresh_row = tk.Frame(self.param_card, bg=THEME["bg_inner"])
        thresh_row.pack(fill=tk.X, padx=12, pady=2)
        self.thresh_lbl = tk.Label(thresh_row, text="Confidence Gate: 50%", font=("Segoe UI", 8), fg=THEME["text_white"], bg=THEME["bg_inner"])
        self.thresh_lbl.pack(side=tk.LEFT)

        self.thresh_slider = ttk.Scale(self.param_card, from_=25, to=90, value=50, command=self.on_thresh_change)
        self.thresh_slider.pack(fill=tk.X, padx=12, pady=(0, 6))

        hold_row = tk.Frame(self.param_card, bg=THEME["bg_inner"])
        hold_row.pack(fill=tk.X, padx=12, pady=2)
        self.hold_dur_lbl = tk.Label(hold_row, text="Hold Duration: 0.5s", font=("Segoe UI", 8), fg=THEME["text_white"], bg=THEME["bg_inner"])
        self.hold_dur_lbl.pack(side=tk.LEFT)

        self.hold_slider = ttk.Scale(self.param_card, from_=2, to=20, value=5, command=self.on_hold_dur_change)
        self.hold_slider.pack(fill=tk.X, padx=12, pady=(0, 6))

        # Tracking Mode Toggle
        mode_row = tk.Frame(self.param_card, bg=THEME["bg_inner"])
        mode_row.pack(fill=tk.X, padx=12, pady=(2, 8))
        self.mode_var = tk.StringVar(value="dynamic")
        ttk.Radiobutton(mode_row, text="MediaPipe Auto", variable=self.mode_var, value="dynamic", command=self.on_tracking_mode_change).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Fixed ROI Box", variable=self.mode_var, value="fixed", command=self.on_tracking_mode_change).pack(side=tk.RIGHT)

        # 5. Performance Stats
        stats_card = tk.Frame(right_col, bg=THEME["bg_card"])
        stats_card.pack(fill=tk.X, padx=12, pady=(0, 6), side=tk.BOTTOM)

        self.stats_lbl = tk.Label(
            stats_card,
            text="FPS: 0.0 | Latency: 0.0 ms",
            font=("Consolas", 9),
            fg=THEME["cyan"],
            bg=THEME["bg_card"]
        )
        self.stats_lbl.pack(pady=4)

    def build_status_bar(self):
        status = tk.Frame(self.root, bg="#060910", height=26)
        status.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_msg_lbl = tk.Label(
            status,
            text="Ready. Select Webcam, Photo, or Video mode to begin.",
            font=("Segoe UI", 9),
            fg=THEME["text_muted"],
            bg="#060910"
        )
        self.status_msg_lbl.pack(side=tk.LEFT, padx=14)

        author_lbl = tk.Label(
            status,
            text="ISL Multi-Modal AI Platform • Local Offline Engine",
            font=("Segoe UI", 8),
            fg="#475569",
            bg="#060910"
        )
        author_lbl.pack(side=tk.RIGHT, padx=14)

    # ----------------- ENGINE INITIALIZATION -----------------
    def init_engine_thread(self):
        def _load():
            try:
                combined_model = Path("artifacts/best_model_combined.keras")
                combined_map = Path("artifacts/class_mapping_mendeley.json")
                real_model = Path("artifacts/best_model_real.keras")
                real_map = Path("artifacts/class_mapping_real.json")

                if combined_model.exists() and combined_map.exists():
                    model_path = str(combined_model)
                    map_path = str(combined_map)
                    model_desc = "36 Classes: 0–9 & A–Z (Unified Real + Mendeley Model)"
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
                    confidence_threshold=0.50,
                    smoothing_window=5,
                    stability_duration=0.5,
                    margin_scale=1.15,
                    adaptive_canny=True,
                    use_ensemble=True
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
        self.subtitle_lbl.configure(text=f"{num_classes} Classes ({first_cls}–{last_cls}) • Unified CNN & MediaPipe")
        self.engine_status_lbl.configure(text=f"● {desc}", fg=THEME["emerald"])
        self.status_msg_lbl.configure(text=f"Ready: {desc}. Select an input mode above.")

    def on_engine_error(self, err_msg):
        self.engine_status_lbl.configure(text="● Model Load Error", fg=THEME["rose"])
        messagebox.showerror("Model Error", f"Failed to initialize CNN model:\n{err_msg}")

    # ----------------- PARAMETER HANDLERS -----------------
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

    def on_tracking_mode_change(self):
        self.use_fixed_roi = (self.mode_var.get() == "fixed")

    # ----------------- WEBCAM MODE LOGIC -----------------
    def toggle_webcam(self):
        if not self.is_webcam_running:
            self.start_webcam()
        else:
            self.stop_webcam()

    def start_webcam(self):
        if not self.engine:
            messagebox.showwarning("Please Wait", "Deep learning engine is initializing. Please wait a moment.")
            return

        try:
            self.camera_idx = int(self.cam_spin.get())
        except ValueError:
            self.camera_idx = 0

        self.status_msg_lbl.configure(text=f"Connecting to camera {self.camera_idx}...")
        self.cap = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Unable to access camera index {self.camera_idx}.\nPlease check connection and Windows permissions.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.is_webcam_running = True
        self.btn_toggle_cam.configure(text="■ Stop Webcam", style="Danger.TButton")
        self.status_msg_lbl.configure(text="Webcam active. Present ISL signs in view.")

        self.update_webcam_loop()

    def stop_webcam(self):
        self.is_webcam_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_toggle_cam.configure(text="▶ Start Webcam", style="Success.TButton")
        self.canvas.delete("all")
        self.pip_canvas.delete("all")
        self.pred_badge_lbl.configure(text="Camera Stopped", fg=THEME["text_muted"])
        self.conf_badge_lbl.configure(text="Confidence: —")
        self.hold_var.set(0.0)
        self.hold_lbl.configure(text="Hold: 0%")

    def update_webcam_loop(self):
        if not self.is_webcam_running or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.root.after(25, self.update_webcam_loop)
            return

        frame = cv2.flip(frame, 1)
        self.process_and_render_frame(frame, is_stream=True)

        if self.is_webcam_running:
            self.root.after(15, self.update_webcam_loop)

    # ----------------- PHOTO MODE LOGIC -----------------
    def browse_photo(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select ISL Gesture Photo", filetypes=filetypes)
        if not path:
            return

        self.photo_path = path
        self.photo_file_lbl.configure(text=Path(path).name)

        # Windows safe load (supports non-ASCII paths)
        try:
            raw_bytes = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image data.")
        except Exception as e:
            messagebox.showerror("Image Load Error", f"Could not load image:\n{e}")
            return

        self.photo_original_bgr = img
        self.status_msg_lbl.configure(text=f"Analyzing photo: {Path(path).name}...")

        # Run offline prediction using static MediaPipe mode
        if not self.engine:
            messagebox.showwarning("Please Wait", "Engine is still initializing.")
            return

        annotated, prep_preview, result = self.engine.predict_image(img)

        # Update displays
        self.update_prediction_ui(result, prep_preview)
        self.render_canvas(annotated)
        if prep_preview is not None:
            self.render_pip(prep_preview)
        else:
            self.pip_canvas.delete("all")

        self.last_annotated = annotated
        self.last_prep = prep_preview
        self.last_result = result

        sign = result.get("sign", "?")
        conf = result.get("confidence", 0.0) * 100.0
        self.status_msg_lbl.configure(text=f"Photo Analysis Complete: Identified [{sign}] ({conf:.1f}%) in {result.get('latency_ms', 0):.1f}ms")

    def append_photo_sign(self):
        """Appends the recognized photo sign to the fingerspelled text buffer."""
        if hasattr(self, "last_result") and self.last_result:
            sign = self.last_result.get("sign")
            if sign and sign != "Unknown" and "failed" not in sign.lower():
                if self.engine:
                    self.engine.spelled_text += sign
                self.text_entry.delete(0, tk.END)
                self.text_entry.insert(0, self.engine.spelled_text if self.engine else sign)
                self.status_msg_lbl.configure(text=f"Appended '{sign}' to text buffer.")

    # ----------------- VIDEO MODE LOGIC -----------------
    def browse_video(self):
        filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select ISL Video File", filetypes=filetypes)
        if not path:
            return

        self.load_video_file(path)

    def load_video_file(self, path: str):
        if self.video_cap:
            self.video_cap.release()

        self.video_path = path
        self.video_file_lbl.configure(text=Path(path).name)

        self.video_cap = cv2.VideoCapture(path)
        if not self.video_cap.isOpened():
            messagebox.showerror("Video Error", f"Unable to open video file:\n{path}")
            return

        self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_cur_frame = 0

        self.video_slider.configure(to=max(1, self.video_total_frames - 1))
        self.video_slider_var.set(0.0)

        # Clear engine smoothing for fresh video stream
        if self.engine:
            self.engine.reset_smoothing()

        # Read and display the first frame
        ret, frame = self.video_cap.read()
        if ret and frame is not None:
            self.process_and_render_frame(frame, is_stream=False)
            self.update_video_time_label()

        self.status_msg_lbl.configure(text=f"Loaded video '{Path(path).name}' ({self.video_total_frames} frames @ {self.video_fps:.1f} FPS). Click Play.")

    def toggle_video_playback(self):
        if not self.is_video_playing:
            self.play_video()
        else:
            self.pause_video()

    def play_video(self):
        if not self.video_cap:
            messagebox.showinfo("No Video", "Please choose a video file first.")
            return

        self.is_video_playing = True
        self.btn_vid_play.configure(text="⏸ Pause", style="Indigo.TButton")
        self.status_msg_lbl.configure(text="Playing video and recognizing sign sequences...")
        self.update_video_playback_loop()

    def pause_video(self):
        self.is_video_playing = False
        self.btn_vid_play.configure(text="▶ Play", style="Success.TButton")
        self.status_msg_lbl.configure(text="Video paused.")

    def stop_video(self):
        self.is_video_playing = False
        self.btn_vid_play.configure(text="▶ Play", style="Success.TButton")
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_cur_frame = 0
            self.video_slider_var.set(0.0)
            ret, frame = self.video_cap.read()
            if ret and frame is not None:
                self.process_and_render_frame(frame, is_stream=False)
                self.update_video_time_label()
        if self.engine:
            self.engine.reset_smoothing()
        self.status_msg_lbl.configure(text="Video stopped and reset to start.")

    def rewind_video(self):
        if not self.video_cap:
            return
        new_frame = max(0, self.video_cur_frame - int(self.video_fps * 5))
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.video_cur_frame = new_frame
        self.video_slider_var.set(new_frame)
        ret, frame = self.video_cap.read()
        if ret and frame is not None:
            self.process_and_render_frame(frame, is_stream=False)
            self.update_video_time_label()

    def on_video_seek(self, val):
        if not self.video_cap or self.video_scrubbing:
            return
        target_frame = int(float(val))
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        self.video_cur_frame = target_frame
        ret, frame = self.video_cap.read()
        if ret and frame is not None:
            self.process_and_render_frame(frame, is_stream=False)
            self.update_video_time_label()

    def update_video_time_label(self):
        cur_sec = int(self.video_cur_frame / max(1.0, self.video_fps))
        tot_sec = int(self.video_total_frames / max(1.0, self.video_fps))
        cur_str = f"{cur_sec // 60:02d}:{cur_sec % 60:02d}"
        tot_str = f"{tot_sec // 60:02d}:{tot_sec % 60:02d}"
        self.vid_time_lbl.configure(text=f"{cur_str} / {tot_str}  (Frame {self.video_cur_frame}/{self.video_total_frames})")

    def update_video_playback_loop(self):
        if not self.is_video_playing or not self.video_cap:
            return

        ret, frame = self.video_cap.read()
        if not ret or frame is None:
            # End of video reached
            self.is_video_playing = False
            self.btn_vid_play.configure(text="▶ Replay", style="Success.TButton")
            self.status_msg_lbl.configure(text="End of video reached. Fully processed.")
            return

        self.video_cur_frame += 1
        self.video_slider_var.set(self.video_cur_frame)
        self.update_video_time_label()

        self.process_and_render_frame(frame, is_stream=True)

        # Dynamic sleep timing to honor video source FPS
        delay_ms = max(10, int(1000.0 / max(10.0, self.video_fps)))
        if self.is_video_playing:
            self.root.after(delay_ms, self.update_video_playback_loop)

    # ----------------- CORE PIPELINE DISPATCHER -----------------
    def process_and_render_frame(self, frame_bgr, is_stream=True):
        """Processes frame through CNN & MediaPipe, updates badges, PiP, and text buffer."""
        if not self.engine:
            return

        h, w = frame_bgr.shape[:2]

        if self.use_fixed_roi:
            box_side = min(h, w, 240)
            x1 = (w - box_side) // 2
            y1 = (h - box_side) // 2
            roi_bgr = frame_bgr[y1:y1 + box_side, x1:x1 + box_side]

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
            conf = float(preds[best_idx])
            pred_cls = self.engine.idx_to_class.get(best_idx, "Unknown")
            is_reliable = (conf >= self.engine.confidence_threshold)

            annotated = frame_bgr.copy()
            box_color = (0, 255, 100) if is_reliable else (50, 150, 255)
            cv2.rectangle(annotated, (x1, y1), (x1 + box_side, y1 + box_side), box_color, 2)

            result = {
                "predicted_class": pred_cls if is_reliable else "No reliable sign detected",
                "sign": pred_cls,
                "confidence": conf,
                "raw_class": pred_cls,
                "raw_confidence": conf,
                "is_reliable": is_reliable,
                "latency_ms": 20.0,
                "fps": 30.0,
                "spelled_text": self.engine.spelled_text,
                "hold_progress": 0.0,
                "top3": [{"class": pred_cls, "prob": conf}],
                "num_hands": 1,
                "hand_mode": "Fixed Central ROI Mode"
            }
        else:
            annotated, prep_preview, result = self.engine.process_frame(frame_bgr)

        # Update UI Controls
        self.update_prediction_ui(result, prep_preview)
        self.render_canvas(annotated)

        if prep_preview is not None:
            self.render_pip(prep_preview)
        else:
            self.pip_canvas.delete("all")

        self.last_annotated = annotated
        self.last_prep = prep_preview
        self.last_result = result

        # Auto-Capture Full Screen on Confirmed Result (Webcam Mode only)
        if self.current_mode == "webcam" and result.get("new_char_confirmed") and self.auto_capture_var.get() and self.engine:
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
                text=f"📸 Auto-Captured [{result['confirmed_char']}] ({snap_res['confidence']}%) -> {snap_res['filename']}",
                fg=THEME["cyan"]
            )

    def update_prediction_ui(self, result: dict, prep_preview):
        # 1. Prediction Badge
        is_reliable = result.get("is_reliable", False)
        pred_cls = result.get("predicted_class", "No Sign Detected")
        conf = result.get("confidence", 0.0)

        if is_reliable and pred_cls != "No reliable sign detected":
            self.pred_badge_lbl.configure(text=f"Sign: {pred_cls}", fg=THEME["emerald"])
            self.conf_badge_lbl.configure(
                text=f"Confidence: {conf * 100:.1f}% (Gate: {int(self.engine.confidence_threshold * 100)}%)",
                fg=THEME["emerald"]
            )
        else:
            raw_c = result.get("raw_class", "—")
            raw_p = result.get("raw_confidence", 0.0) * 100.0
            self.pred_badge_lbl.configure(text=pred_cls if "Low Conf" in pred_cls else "No Reliable Sign", fg=THEME["amber"])
            self.conf_badge_lbl.configure(text=f"Raw: {raw_c} ({raw_p:.1f}%)" if raw_c != "—" else "Confidence: —", fg=THEME["text_muted"])

        # 2. Hand Mode
        if "hand_mode" in result:
            self.hand_mode_lbl.configure(text=f"✋ {result['hand_mode']}")

        # 3. Top 3
        if result.get("top3"):
            t_str = " | ".join([f"{item['class']}:{item['prob']*100:.0f}%" for item in result["top3"]])
            self.top3_lbl.configure(text=f"Top 3: {t_str}")
        else:
            self.top3_lbl.configure(text="Top 3: —")

        # 4. Spelled Text Buffer
        spelled = self.engine.spelled_text if self.engine else ""
        current_text = self.text_entry.get()
        if current_text != spelled:
            self.text_entry.delete(0, tk.END)
            self.text_entry.insert(0, spelled)
            self.text_entry.xview_moveto(1.0)

        # 5. Hold Progress
        hold_pct = result.get("hold_progress", 0.0) * 100.0
        self.hold_var.set(hold_pct)
        self.hold_lbl.configure(text=f"Hold: {int(hold_pct)}%")

        # 6. Performance
        fps = result.get("fps", 0.0)
        lat = result.get("latency_ms", 0.0)
        self.stats_lbl.configure(text=f"FPS: {fps:.1f} | Latency: {lat:.1f} ms")

    def render_canvas(self, bgr_img):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            return

        ih, iw = bgr_img.shape[:2]
        scale = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))

        resized = cv2.resize(bgr_img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.tk_main_img = ImageTk.PhotoImage(Image.fromarray(rgb_img))

        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_main_img, anchor=tk.CENTER)

    def render_pip(self, gray_64x64):
        scaled = cv2.resize(gray_64x64, (128, 128), interpolation=cv2.INTER_NEAREST)
        self.tk_pip_img = ImageTk.PhotoImage(Image.fromarray(scaled))
        self.pip_canvas.delete("all")
        self.pip_canvas.create_image(64, 64, image=self.tk_pip_img, anchor=tk.CENTER)

    def on_canvas_resize(self, event):
        if hasattr(self, "last_annotated") and self.last_annotated is not None:
            self.render_canvas(self.last_annotated)

    # ----------------- TEXT ACTIONS -----------------
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

    # ----------------- REVERSE TRANSLATION (SPEECH & TEXT TO SIGN) -----------------
    def start_voice_recognition(self):
        """Spawns background worker thread to listen to user speech and transcribe."""
        if hasattr(self, "reverse_translator") and self.reverse_translator.is_listening:
            return

        self.btn_voice_input.configure(bg=THEME["amber"], text="🎙️ Listening...", state="disabled")
        self.voice_status_lbl.configure(text="Listening to microphone... Speak a letter or word.", fg=THEME["cyan"])

        def _voice_worker():
            def _progress(msg):
                self.root.after(0, lambda: self.voice_status_lbl.configure(text=msg))

            success, result_text = self.reverse_translator.listen_and_transcribe(status_callback=_progress)

            def _on_finish():
                self.btn_voice_input.configure(bg=THEME["cyan"], text="🎙️ Speak Letter / Word", state="normal")
                if success:
                    clean = result_text.strip().upper()
                    # Clean up phrases like "LETTER A" or "THE LETTER B"
                    if clean.startswith("LETTER ") and len(clean.split()) == 2:
                        clean = clean.split()[1]
                    self.rev_input_entry.delete(0, tk.END)
                    self.rev_input_entry.insert(0, clean)
                    self.voice_status_lbl.configure(text=f"Heard: '{clean}'", fg=THEME["emerald"])
                    self.translate_text_to_signs(clean)
                else:
                    self.voice_status_lbl.configure(text=f"Notice: {result_text}", fg=THEME["amber"])

            self.root.after(0, _on_finish)

        threading.Thread(target=_voice_worker, daemon=True).start()

    def on_reverse_text_submit(self):
        text = self.rev_input_entry.get().strip().upper()
        if not text:
            text = self.text_entry.get().strip().upper()
        if not text:
            self.voice_status_lbl.configure(text="Please type or speak a letter (A-Z) or word first.", fg=THEME["amber"])
            return
        self.translate_text_to_signs(text)

    def on_main_text_submit(self):
        text = self.text_entry.get().strip().upper()
        if not text:
            self.status_msg_lbl.configure(text="Please type a letter (A-Z) or word first.", fg=THEME["amber"])
            return
        if hasattr(self, "rev_input_entry"):
            self.rev_input_entry.delete(0, tk.END)
            self.rev_input_entry.insert(0, text)
        self.translate_text_to_signs(text)

    def on_quick_key_click(self, char: str):
        self.rev_input_entry.delete(0, tk.END)
        self.rev_input_entry.insert(0, char)
        self.translate_text_to_signs(char)

    def translate_text_to_signs(self, text: str):
        """Translates the input text into a sequence of ISL signs and shows the first."""
        self.stop_reverse_animation()
        self.reverse_sequence = self.reverse_translator.text_to_sign_sequence(text)
        if not self.reverse_sequence:
            self.status_msg_lbl.configure(text="No valid signs found for input.", fg=THEME["amber"])
            return

        # Mirror in spelled text entry
        self.text_entry.delete(0, tk.END)
        self.text_entry.insert(0, text)

        self.reverse_idx = 0
        self.show_reverse_sign_at_index(0)

        # TTS Pronunciation
        if self.reverse_auto_speech.get():
            if len(text) == 1:
                self.reverse_translator.speak_async(f"Sign {text}")
            else:
                self.reverse_translator.speak_async(text)

    def show_reverse_sign_at_index(self, idx: int):
        if not self.reverse_sequence:
            return
        idx = max(0, min(idx, len(self.reverse_sequence) - 1))
        self.reverse_idx = idx

        symbol, path, desc = self.reverse_sequence[idx]
        seq_text = f"Sign {idx + 1} of {len(self.reverse_sequence)}"
        card_bgr, pip_64 = self.reverse_translator.create_sign_display_card(
            symbol=symbol,
            sequence_info=seq_text
        )

        self.last_annotated = card_bgr
        self.last_prep = pip_64
        self.render_canvas(card_bgr)
        self.render_pip(pip_64)

        # Update telemetry badges
        self.pred_badge_lbl.configure(text=f"Sign '{symbol}'", fg=THEME["cyan"])
        self.conf_badge_lbl.configure(text="Reference Standard: 100%", fg=THEME["emerald"])
        self.hand_mode_lbl.configure(text=f"{desc[:36]}..." if len(desc) > 36 else desc)
        self.top3_lbl.configure(text=f"Target: '{symbol}' ({seq_text})")
        self.reverse_step_lbl.configure(text=f"{seq_text}  ('{symbol}')")
        self.status_msg_lbl.configure(text=f"Showing ISL Sign '{symbol}': {desc}", fg=THEME["text_white"])

    def prev_reverse_sign(self):
        if self.reverse_sequence and self.reverse_idx > 0:
            self.stop_reverse_animation()
            self.show_reverse_sign_at_index(self.reverse_idx - 1)

    def next_reverse_sign(self):
        if self.reverse_sequence and self.reverse_idx < len(self.reverse_sequence) - 1:
            self.stop_reverse_animation()
            self.show_reverse_sign_at_index(self.reverse_idx + 1)

    def toggle_reverse_animation(self):
        if self.reverse_is_playing:
            self.stop_reverse_animation()
        else:
            if not self.reverse_sequence:
                text = self.rev_input_entry.get().strip().upper()
                if text:
                    self.translate_text_to_signs(text)
                else:
                    return

            if self.reverse_idx >= len(self.reverse_sequence) - 1:
                self.reverse_idx = 0
                self.show_reverse_sign_at_index(0)

            self.reverse_is_playing = True
            self.btn_rev_play.configure(text="⏸ Pause", style="Indigo.TButton")
            self._schedule_next_reverse_frame()

    def _schedule_next_reverse_frame(self):
        if not self.reverse_is_playing:
            return
        self.reverse_anim_job = self.root.after(self.reverse_speed_ms, self._step_reverse_animation)

    def _step_reverse_animation(self):
        if not self.reverse_is_playing:
            return

        if self.reverse_idx < len(self.reverse_sequence) - 1:
            self.reverse_idx += 1
            self.show_reverse_sign_at_index(self.reverse_idx)
            sym = self.reverse_sequence[self.reverse_idx][0]
            if self.reverse_auto_speech.get() and sym != "SPACE":
                self.reverse_translator.speak_async(sym)
            self._schedule_next_reverse_frame()
        else:
            self.stop_reverse_animation()

    def stop_reverse_animation(self):
        self.reverse_is_playing = False
        if self.reverse_anim_job:
            self.root.after_cancel(self.reverse_anim_job)
            self.reverse_anim_job = None
        self.btn_rev_play.configure(text="▶ Play Fingerspelling", style="Success.TButton")

    def pronounce_current_sign(self):
        if self.reverse_sequence:
            sym = self.reverse_sequence[self.reverse_idx][0]
            if sym == "SPACE":
                self.reverse_translator.speak_async("Space")
            else:
                self.reverse_translator.speak_async(f"Letter {sym}")
        else:
            text = self.rev_input_entry.get().strip()
            if text:
                self.reverse_translator.speak_async(text)

    # ----------------- SNAPSHOT & EXPORT -----------------
    def capture_full_window(self):
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
                bmi.biHeight = -h
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
            except Exception:
                pass

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
        if not hasattr(self, "last_annotated") or self.last_annotated is None:
            self.status_msg_lbl.configure(text="No active frame to capture.", fg=THEME["amber"])
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
                text=f"📸 Snapshot Saved: {snap_res['filename']}",
                fg=THEME["emerald"]
            )

    def on_close(self):
        self.is_webcam_running = False
        self.is_video_playing = False
        if hasattr(self, "reverse_is_playing") and self.reverse_is_playing:
            self.stop_reverse_animation()
        if self.cap:
            self.cap.release()
        if self.video_cap:
            self.video_cap.release()
        if self.engine:
            self.engine.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ISLMultiModalApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
