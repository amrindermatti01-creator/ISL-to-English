"""
tests/test_multimodal_modes.py - Multi-Modal Verification Test Suite
Tests:
  1. Static Photo Mode (predict_image)
  2. Video Stream Mode (temporal frame sequence & word accumulation)
  3. Spelled buffer manipulation (space, backspace, clear)
"""

import sys
import os
import time
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import ISLInferenceEngine

def run_tests():
    print("=" * 60)
    print("RUNNING MULTI-MODAL VERIFICATION TESTS")
    print("=" * 60)

    # 1. Initialize Engine
    print("[Test 1/3] Initializing ISLInferenceEngine...")
    engine = ISLInferenceEngine(confidence_threshold=0.50)
    assert hasattr(engine, "predict_image"), "Engine must have predict_image method"
    assert hasattr(engine, "static_hands"), "Engine must have static_hands MediaPipe instance"
    print(" -> Engine initialized successfully.")

    # 2. Test Photo Mode (predict_image)
    print("\n[Test 2/3] Testing Photo Mode Inference (predict_image)...")
    # Create test synthetic frame
    test_frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.circle(test_frame, (320, 240), 90, (200, 180, 160), -1)
    # Add finger-like protrusions
    cv2.rectangle(test_frame, (300, 110), (340, 200), (200, 180, 160), -1)

    annotated, prep_preview, result = engine.predict_image(test_frame)
    assert "predicted_class" in result, "Result must contain predicted_class"
    assert "confidence" in result, "Result must contain confidence"
    assert "top3" in result, "Result must contain top3"
    assert "latency_ms" in result, "Result must contain latency_ms"
    assert annotated.shape == test_frame.shape, "Annotated frame must match input shape"
    assert prep_preview is not None and prep_preview.shape == (64, 64), "PiP preview must be 64x64"

    print(f" -> Photo Mode Output: Class={result['predicted_class']}, Conf={result['confidence']*100:.1f}%, Latency={result['latency_ms']}ms")
    print(" -> Photo Mode PASSED.")

    # 3. Test Video Mode Simulation (temporal stream & word assembly)
    print("\n[Test 3/3] Testing Video Mode Frame Processing & Word Accumulation...")
    engine.clear_text()
    assert engine.spelled_text == "", "Spelled text should be reset"

    # Simulate feeding sequential video frames
    for i in range(10):
        ann, prep, res = engine.process_frame(test_frame)

    # Test buffer operations
    engine.spelled_text = "HELLO"
    engine.add_space()
    assert engine.spelled_text == "HELLO ", f"Expected 'HELLO ', got '{engine.spelled_text}'"
    engine.spelled_text += "WORLD"
    assert engine.spelled_text == "HELLO WORLD"
    engine.backspace()
    assert engine.spelled_text == "HELLO WORL"
    engine.clear_text()
    assert engine.spelled_text == ""

    print(" -> Video & Word Accumulation Operations PASSED.")

    engine.release()
    print("\n" + "=" * 60)
    print("ALL MULTI-MODAL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
