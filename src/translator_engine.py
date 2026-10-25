"""
src/translator_engine.py
Multilingual Translation, Non-Blocking Text-to-Speech (TTS),
and Regional Unicode Font Rendering for Indian Sign Language.
"""

import os
import io
import logging
import threading
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("ISLTranslatorEngine")

# Try initializing pygame mixer safely
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AUDIO_AVAILABLE = True
except Exception as e:
    logger.warning(f"Pygame mixer initialization skipped: {e}")
    PYGAME_AUDIO_AVAILABLE = False


SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi (हिंदी)": "hi",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Bengali (বাংলা)": "bn",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Marathi (मराठी)": "mr",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Urdu (اردو)": "ur",
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS_DIR = os.path.join(PROJECT_ROOT, "assets", "fonts")

FONT_MAP = {
    "hi": os.path.join(FONTS_DIR, "NotoSansDevanagari-Regular.ttf"),
    "mr": os.path.join(FONTS_DIR, "NotoSansDevanagari-Regular.ttf"),
    "pa": os.path.join(FONTS_DIR, "NotoSansGurmukhi-Regular.ttf"),
    "bn": os.path.join(FONTS_DIR, "NotoSansBengali-Regular.ttf"),
    "ta": os.path.join(FONTS_DIR, "NotoSansTamil-Regular.ttf"),
    "te": os.path.join(FONTS_DIR, "NotoSansTelugu-Regular.ttf"),
    "kn": os.path.join(FONTS_DIR, "NotoSansKannada-Regular.ttf"),
    "ur": os.path.join(FONTS_DIR, "NotoSansArabic-Regular.ttf"),
    "ar": os.path.join(FONTS_DIR, "NotoSansArabic-Regular.ttf"),
    "en": os.path.join(FONTS_DIR, "NotoSans-Regular.ttf"),
}
DEFAULT_FONT = os.path.join(FONTS_DIR, "NotoSans-Regular.ttf")


class TranslatorEngine:
    def __init__(self):
        self._translation_cache: Dict[Tuple[str, str], str] = {}
        self._is_speaking = False
        self._font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

    def translate(self, text: str, target_lang: str = "hi") -> str:
        """
        Translates text from English to target regional language using deep-translator.
        Cached in memory to prevent duplicate requests.
        """
        clean_text = text.strip()
        if not clean_text or target_lang == "en":
            return clean_text

        cache_key = (clean_text, target_lang)
        if cache_key in self._translation_cache:
            return self._translation_cache[cache_key]

        # Common sign vocabulary local offline dictionary for zero-latency instant translation
        OFFLINE_SIGNS = {
            ("HELLO", "hi"): "नमस्ते",
            ("HELLO", "pa"): "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ",
            ("HELLO", "ta"): "வணக்கம்",
            ("HELLO", "bn"): "নমস্কার",
            ("HELLO", "te"): "నమస్కారం",
            ("HELP", "hi"): "मदद",
            ("HELP", "pa"): "ਮਦਦ",
            ("HELP", "ta"): "உதவி",
            ("HELP", "bn"): "সাহায্য",
            ("THANKS", "hi"): "धन्यवाद",
            ("THANKS", "pa"): "ਧੰਨਵਾਦ",
            ("THANKS", "ta"): "நன்றி",
            ("THANKS", "bn"): "ধন্যবাদ",
            ("PLEASE", "hi"): "कृपया",
            ("PLEASE", "pa"): "ਕਿਰਪਾ ਕਰਕੇ",
            ("YES", "hi"): "हाँ",
            ("NO", "hi"): "नहीं",
            ("GOOD", "hi"): "अच्छा",
            ("BAD", "hi"): "बुरा"
        }
        upper_pair = (clean_text.upper(), target_lang)
        if upper_pair in OFFLINE_SIGNS:
            self._translation_cache[cache_key] = OFFLINE_SIGNS[upper_pair]
            return OFFLINE_SIGNS[upper_pair]

        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source="en", target=target_lang)
            translated = translator.translate(clean_text)
            self._translation_cache[cache_key] = translated
            return translated
        except Exception as e:
            # Fallback to MyMemoryTranslator if Google rate-limits
            try:
                from deep_translator import MyMemoryTranslator
                trans_mem = MyMemoryTranslator(source=f"en-US", target=f"{target_lang}-IN")
                translated = trans_mem.translate(clean_text)
                self._translation_cache[cache_key] = translated
                return translated
            except Exception as e2:
                logger.warning(f"Translation failed for '{clean_text}' to '{target_lang}': {e} / {e2}")
                return clean_text

    def speak_async(self, text: str, lang_code: str = "en") -> None:
        """
        Synthesizes spoken audio using gTTS and plays non-blockingly via pygame.mixer.
        """
        if not PYGAME_AUDIO_AVAILABLE or not text.strip():
            return

        def _play_worker():
            try:
                from gtts import gTTS
                # Map Urdu/Arabic gTTS code if needed
                tts_lang = lang_code if lang_code != "pa" else "hi"  # fallback for Punjabi if unsupported in gTTS
                tts = gTTS(text=text, lang=tts_lang, slow=False)
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)

                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()

                pygame.mixer.music.load(audio_buffer)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(50)
            except Exception as e:
                logger.warning(f"TTS audio playback failed: {e}")

        thread = threading.Thread(target=_play_worker, daemon=True)
        thread.start()

    def get_font(self, lang_code: str, size: int = 24) -> ImageFont.FreeTypeFont:
        """Loads and caches Pillow TrueType fonts for the specified language."""
        font_path = FONT_MAP.get(lang_code, DEFAULT_FONT)
        if not os.path.exists(font_path):
            font_path = DEFAULT_FONT

        key = (font_path, size)
        if key not in self._font_cache:
            try:
                self._font_cache[key] = ImageFont.truetype(font_path, size)
            except Exception:
                self._font_cache[key] = ImageFont.load_default()
        return self._font_cache[key]

    def render_unicode_text(
        self,
        frame_bgr: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        lang_code: str = "en",
        font_size: int = 24,
        color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Optional[Tuple[int, int, int]] = (20, 20, 20)
    ) -> np.ndarray:
        """
        Renders regional Unicode text (Devanagari, Gurmukhi, Tamil, etc.)
        onto an OpenCV BGR frame using Pillow with antialiasing.
        """
        if not text:
            return frame_bgr

        # Convert to RGB PIL Image
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        font = self.get_font(lang_code, font_size)

        x, y = pos
        # Optional background highlight box
        if bg_color is not None:
            bbox = draw.textbbox((x, y), text, font=font)
            pad = 4
            draw.rectangle(
                [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                fill=bg_color
            )

        # Draw antialiased text
        draw.text((x, y), text, font=font, fill=color)

        # Convert back to BGR NumPy array
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
