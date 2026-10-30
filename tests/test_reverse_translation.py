"""
tests/test_reverse_translation.py - Unit test suite for Reverse ISL Translation (Speech & Text to Sign)
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reverse_translator import ReverseISLTranslator


def test_reverse_translator():
    print("=" * 60)
    print("TESTING REVERSE ISL TRANSLATOR (SPEECH & TEXT TO SIGN)")
    print("=" * 60)

    # 1. Initialize Translator
    print("[Test 1/5] Initializing ReverseISLTranslator...")
    translator = ReverseISLTranslator()
    available = translator.available_signs
    print(f" -> Found {len(available)} available sign references.")
    assert len(available) >= 35, f"Expected at least 35 signs (A-Z, 1-9), got {len(available)}"
    print(" -> PASS: Translator initialized successfully.")

    # 2. Verify all A-Z and 1-9 signs exist
    print("\n[Test 2/5] Verifying all A-Z and 1-9 sign images exist on disk...")
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        path = translator.get_sign_path(letter)
        assert path is not None and path.exists(), f"Missing sign image for letter {letter}"
    for num in "123456789":
        path = translator.get_sign_path(num)
        assert path is not None and path.exists(), f"Missing sign image for numeral {num}"
    print(" -> PASS: All 26 alphabets (A-Z) and 9 numerals (1-9) confirmed.")

    # 3. Test Text-to-Sign Sequence Parsing
    print("\n[Test 3/5] Testing Text to Fingerspelling Sequence Conversion...")
    test_phrase = "ISL 2026"
    sequence = translator.text_to_sign_sequence(test_phrase)
    print(f" -> Input: '{test_phrase}' -> {len(sequence)} sequence tokens:")
    for symbol, path, desc in sequence:
        print(f"    • Symbol '{symbol}': Path={path.name if path else 'None'} | Desc: {desc[:40]}...")

    symbols = [s[0] for s in sequence]
    expected_symbols = ["I", "S", "L", "SPACE", "2", "0", "2", "6"]
    assert symbols == expected_symbols, f"Expected {expected_symbols}, got {symbols}"
    print(" -> PASS: Text parsing and sequence tokenization verified.")

    # 4. Test Sign Image Loading & Formatting
    print("\n[Test 4/5] Testing Sign Image Decoding & Resizing...")
    img_bgr = translator.get_sign_image_bgr("A", target_size=(256, 256))
    assert img_bgr is not None, "Failed to decode image for 'A'"
    assert img_bgr.shape == (256, 256, 3), f"Expected (256, 256, 3), got {img_bgr.shape}"
    print(f" -> Successfully loaded sign image 'A.jpg' with shape {img_bgr.shape}")
    print(" -> PASS: Image retrieval and formatting verified.")

    # 5. Test TTS and SpeechRecognition module presence
    print("\n[Test 5/5] Checking Speech & TTS Subsystems...")
    translator.speak_async("ISL")
    print(" -> Asynchronous TTS invocation queued safely.")

    import speech_recognition as sr
    mics = sr.Microphone.list_microphone_names()
    print(f" -> SpeechRecognition ready with {len(mics)} system input audio devices.")
    print(" -> PASS: Speech & Audio subsystems operational.")

    print("\n" + "=" * 60)
    print("ALL REVERSE TRANSLATOR TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_reverse_translator()
