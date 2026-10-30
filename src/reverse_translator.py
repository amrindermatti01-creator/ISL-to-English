"""
src/reverse_translator.py - Reverse Translation Engine (English Text/Speech to ISL Signs)

Provides:
  - Sign Image Retrieval for letters (A-Z) and numerals (1-9)
  - Text-to-Sign sequence parsing and word fingerspelling
  - Speech-to-Text transcription via SpeechRecognition (Google Web Speech API)
  - Text-to-Speech audio feedback via pyttsx3
"""

import os
import sys
import time
import queue
import logging
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

# Configure module logger
logger = logging.getLogger("ISL.ReverseTranslator")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [ReverseISL]: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Sign descriptions for pedagogical guidance
SIGN_DESCRIPTIONS: Dict[str, str] = {
    "A": "Closed fist with thumb upright against the side of the index finger.",
    "B": "Open flat palm facing outward, 4 fingers upright, thumb tucked inward.",
    "C": "Hand curved into a 'C' shape with thumb and fingers forming an arc.",
    "D": "Index finger pointing up, remaining fingers curled with thumb touching middle finger.",
    "E": "Fingers curled inward towards palm, thumb folded across underneath fingertips.",
    "F": "Index finger and thumb touching tips to form an 'O', remaining 3 fingers spread upward.",
    "G": "Index finger and thumb extended horizontally parallel to each other.",
    "H": "Index and middle fingers extended together horizontally.",
    "I": "Pinky finger extended straight up, other fingers closed into fist with thumb across.",
    "J": "Pinky finger extended upward then drawn down in a sweeping 'J' curve in the air.",
    "K": "Index finger straight up, middle finger angled slightly forward, thumb resting between.",
    "L": "Index finger up and thumb extended outward at right angle forming an 'L'.",
    "M": "Thumb tucked under the first three fingers (index, middle, ring).",
    "N": "Thumb tucked under the first two fingers (index and middle).",
    "O": "All fingertips touch the thumb tip to form an 'O' circle.",
    "P": "Downward pointing 'K' handshape: index pointing forward, middle pointing down.",
    "Q": "Downward pointing 'G' handshape: index and thumb pointing downwards.",
    "R": "Index and middle fingers crossed over each other.",
    "S": "Tight fist with thumb folded across over the fingers.",
    "T": "Thumb inserted between index and middle fingers of a closed fist.",
    "U": "Index and middle fingers extended together straight up, thumb folded over ring finger.",
    "V": "Index and middle fingers spread in a 'V' shape (peace sign).",
    "W": "Index, middle, and ring fingers extended upward in a 'W' shape.",
    "X": "Index finger hooked into a curved shape, remaining fingers curled in fist.",
    "Y": "Thumb and pinky extended outwards (shaka sign), middle three fingers folded.",
    "Z": "Index finger traces the shape of a 'Z' in the air.",
    "1": "Index finger pointing up, others closed.",
    "2": "Index and middle fingers pointing up (V shape).",
    "3": "Thumb, index, and middle fingers extended.",
    "4": "Four fingers upright, thumb folded in palm.",
    "5": "Open palm with all five fingers spread wide.",
    "6": "Three middle fingers up, thumb and pinky touching tips.",
    "7": "Thumb and ring finger touching, other three fingers up.",
    "8": "Thumb and middle finger touching, other three fingers up.",
    "9": "Thumb and index finger touching, other three fingers up."
}


class ReverseISLTranslator:
    """
    Engine for translating English text and spoken voice into Indian Sign Language (ISL) visuals.
    """

    def __init__(self, signs_dir: Optional[Path] = None):
        self.project_root = Path(__file__).resolve().parent.parent
        self.signs_dir = Path(signs_dir) if signs_dir else (self.project_root / "assets" / "reference_signs")
        
        # Fallback search paths if an image is missing
        self.fallback_dirs = [
            Path(r"C:\Users\hp\Real ISL Camera dataset\Letters"),
            Path(r"C:\Users\hp\Indian")
        ]

        # Verify reference signs cache
        self.available_signs = self._scan_available_signs()
        logger.info(f"Reverse ISL Translator initialized. Available signs: {len(self.available_signs)} ({', '.join(sorted(self.available_signs.keys()))})")

        # Speech recognition lock and state
        self._sr_lock = threading.Lock()
        self.is_listening = False

        # Text-to-speech engine queue
        self._tts_queue = queue.Queue()
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

    def _scan_available_signs(self) -> Dict[str, Path]:
        """Scans the signs directory for valid image files (A-Z, 1-9)."""
        signs = {}
        if self.signs_dir.exists():
            for p in self.signs_dir.glob("*.*"):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    stem = p.stem.upper()
                    signs[stem] = p

        # Check fallback directories if some letters are missing
        all_expected = [chr(c) for c in range(ord('A'), ord('Z') + 1)] + [str(d) for d in range(1, 10)]
        missing = [c for c in all_expected if c not in signs]

        if missing:
            real_letters_dir = Path(r"C:\Users\hp\Real ISL Camera dataset\Letters")
            if real_letters_dir.exists():
                for c in missing:
                    cand = real_letters_dir / f"{c}.jpg"
                    if cand.exists():
                        signs[c] = cand

            indian_dir = Path(r"C:\Users\hp\Indian")
            if indian_dir.exists():
                for c in [m for m in all_expected if m not in signs]:
                    cand = indian_dir / c / "0.jpg"
                    if cand.exists():
                        signs[c] = cand

        return signs

    def get_sign_path(self, symbol: str) -> Optional[Path]:
        """Returns the file path for a symbol's reference image."""
        clean = symbol.strip().upper()
        return self.available_signs.get(clean, None)

    def get_sign_image_bgr(self, symbol: str, target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """Reads and optionally resizes the sign image in BGR format."""
        path = self.get_sign_path(symbol)
        if not path or not path.exists():
            return None

        img = cv2.imread(str(path))
        if img is None:
            return None

        if target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        return img

    def get_description(self, symbol: str) -> str:
        """Returns the instructional description for a sign."""
        clean = symbol.strip().upper()
        return SIGN_DESCRIPTIONS.get(clean, f"Indian Sign Language gesture for '{clean}'.")

    def text_to_sign_sequence(self, text: str) -> List[Tuple[str, Optional[Path], str]]:
        """
        Parses an input English string into a sequence of ISL signs.
        
        Returns:
            List of tuples: (symbol, image_path, description)
            Spaces are represented as ('SPACE', None, 'Pause / Word separator')
        """
        sequence = []
        cleaned_text = text.strip()
        
        for char in cleaned_text:
            if char.isspace():
                if sequence and sequence[-1][0] != "SPACE":
                    sequence.append(("SPACE", None, "Pause between words"))
            else:
                symbol = char.upper()
                if symbol in self.available_signs:
                    sequence.append((symbol, self.available_signs[symbol], self.get_description(symbol)))
                else:
                    # Unsupported character (punctuation, etc.)
                    sequence.append((symbol, None, f"Character '{symbol}' has no direct ISL fingerspelling representation."))

        return sequence

    def listen_and_transcribe(
        self,
        timeout: float = 6.0,
        phrase_time_limit: float = 6.0,
        status_callback: Optional[callable] = None
    ) -> Tuple[bool, str]:
        """
        Captures speech from the default microphone and transcribes it using Google Web Speech API.
        
        Args:
            timeout: Maximum seconds to wait for speech to begin.
            phrase_time_limit: Maximum seconds for the phrase duration.
            status_callback: Optional callable for live progress status updates.
            
        Returns:
            (success: bool, transcript_or_error_message: str)
        """
        try:
            import speech_recognition as sr
        except ImportError:
            return False, "speech_recognition library is not installed in the environment."

        with self._sr_lock:
            self.is_listening = True
            r = sr.Recognizer()
            r.dynamic_energy_threshold = True

            try:
                if status_callback:
                    status_callback("Opening microphone...")

                with sr.Microphone() as source:
                    if status_callback:
                        status_callback("Calibrating background noise...")
                    r.adjust_for_ambient_noise(source, duration=0.6)

                    if status_callback:
                        status_callback("🎙️ Listening... Speak a letter or word clearly.")

                    audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

                if status_callback:
                    status_callback("Transcribing speech with Google API...")

                transcript = r.recognize_google(audio)
                logger.info(f"Speech successfully transcribed: '{transcript}'")
                return True, transcript.upper()

            except sr.WaitTimeoutError:
                return False, "Listening timed out. No speech was detected."
            except sr.UnknownValueError:
                return False, "Could not understand audio. Please speak clearly into the microphone."
            except sr.RequestError as e:
                return False, f"Could not request results from Google Speech Recognition service: {e}"
            except Exception as e:
                logger.error(f"Microphone recording error: {e}", exc_info=True)
                return False, f"Microphone error: {str(e)}"
            finally:
                self.is_listening = False

    def speak_async(self, text: str):
        """Asynchronously synthesizes speech using pyttsx3."""
        if text.strip():
            self._tts_queue.put(text.strip())

    def _tts_worker(self):
        """Background daemon thread to process text-to-speech requests safely."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.setProperty("volume", 0.95)
        except Exception as e:
            logger.warning(f"pyttsx3 initialization failed in background worker: {e}")
            return

        while True:
            text = self._tts_queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS synthesis error: {e}")
            finally:
                self._tts_queue.task_done()

    def create_sign_display_card(
        self,
        symbol: str,
        sequence_info: Optional[str] = None,
        canvas_w: int = 700,
        canvas_h: int = 520
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates an Obsidian Slate & Cyan reference card with the authentic ISL gesture photo,
        header banner, and pedagogical instruction text.
        
        Returns:
            (display_canvas_bgr, canny_pip_64x64)
        """
        clean_sym = symbol.strip().upper()
        canvas = np.full((canvas_h, canvas_w, 3), (11, 15, 25), dtype=np.uint8)

        # Top Header Bar
        cv2.rectangle(canvas, (0, 0), (canvas_w, 55), (19, 27, 46), -1)
        cv2.line(canvas, (0, 55), (canvas_w, 55), (35, 48, 75), 1)

        title_text = f"ISL REFERENCE: '{clean_sym}'"
        cv2.putText(canvas, title_text, (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 229, 255), 2, cv2.LINE_AA)

        if sequence_info:
            cv2.putText(canvas, sequence_info, (canvas_w - 280, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1, cv2.LINE_AA)

        # Retrieve sign image
        raw_img = self.get_sign_image_bgr(clean_sym)
        canny_64 = np.zeros((64, 64), dtype=np.uint8)

        if raw_img is not None:
            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(gray, 50, 150)
            canny_64 = cv2.resize(canny, (64, 64), interpolation=cv2.INTER_AREA)

            avail_h = canvas_h - 150
            avail_w = canvas_w - 60
            ih, iw = raw_img.shape[:2]
            scale = min(avail_w / iw, avail_h / ih, 1.4)
            nw, nh = int(iw * scale), int(ih * scale)
            resized = cv2.resize(raw_img, (nw, nh), interpolation=cv2.INTER_LINEAR)

            ox = (canvas_w - nw) // 2
            oy = 65 + (avail_h - nh) // 2

            cv2.rectangle(canvas, (ox - 4, oy - 4), (ox + nw + 4, oy + nh + 4), (0, 229, 255), 2)
            canvas[oy:oy + nh, ox:ox + nw] = resized
        else:
            if clean_sym == "SPACE":
                cv2.putText(canvas, "[ WORD SEPARATOR / SPACE ]", (canvas_w // 2 - 160, canvas_h // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 229, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, f"Sign image for '{clean_sym}' not available", (canvas_w // 2 - 170, canvas_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (244, 63, 94), 1, cv2.LINE_AA)

        # Bottom Guidance Banner
        cv2.rectangle(canvas, (0, canvas_h - 75), (canvas_w, canvas_h), (19, 27, 46), -1)
        cv2.line(canvas, (0, canvas_h - 75), (canvas_w, canvas_h - 75), (35, 48, 75), 1)

        desc = self.get_description(clean_sym)
        cv2.putText(canvas, "GESTURE INSTRUCTION:", (20, canvas_h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1, cv2.LINE_AA)
        cv2.putText(canvas, desc, (20, canvas_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (248, 250, 252), 1, cv2.LINE_AA)

        return canvas, canny_64


