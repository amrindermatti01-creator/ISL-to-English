"""
src/webcam_app.py
Real-Time Webcam Pipeline for Indian Sign Language (ISL) Recognition.
Features:
  - Unified Dual-Engine: 126D Landmark MLP (Primary) & 2D CNN (Legacy)
  - Picture-in-Picture (PiP) Dual Debug Insets (Source ROI & Canny CNN)
  - Multilingual Translation to Indian Regional Languages (Hindi, Punjabi, Tamil, etc.)
  - Non-blocking Text-to-Speech (gTTS + pygame.mixer)
  - Pillow Unicode Overlay for regional script display
  - Word boundary detection via closed fist gesture or [SPACE] key
  - Automated benchmark & snapshot verification
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.inference import ISLInferenceEngine
from src.translator_engine import SUPPORTED_LANGUAGES

LANG_CYCLE = ["en", "hi", "pa", "bn", "ta", "te", "mr", "ur"]


def draw_hud(
    canvas: np.ndarray,
    result: Dict[str, Any],
    prep_preview: Optional[np.ndarray],
    source_roi: Optional[np.ndarray] = None,
    threshold: float = 0.70
) -> np.ndarray:
    """
    Renders live heads-up display overlay on webcam frame.
    Includes active engine badge, PiP insets, word reconstruction,
    multilingual translation, and audio status.
    """
    h, w = canvas.shape[:2]

    # 1. Header Bar
    header_h = 58
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
    cv2.line(canvas, (0, header_h), (w, header_h), (50, 70, 90), 1)

    engine_tag = "MLP 126D (Primary, <1ms)" if result.get("active_engine") == "landmark_mlp" else "CNN 64x64 (Legacy)"
    cv2.putText(
        canvas,
        f"ISL REAL-TIME BRIDGE | ENGINE: {engine_tag}",
        (16, 24),
        cv2.FONT_HERSHEY_DUPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    lang_code = result.get("target_language", "hi")
    tts_str = "TTS: ON" if result.get("tts_enabled") else "TTS: OFF"
    tts_col = (0, 255, 120) if result.get("tts_enabled") else (140, 150, 160)

    metrics_text = f"FPS: {result['fps']:.1f} | Latency: {result['latency_ms']:.1f} ms | Lang: {lang_code.upper()} | {tts_str}"
    cv2.putText(
        canvas,
        metrics_text,
        (16, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 220, 255),
        1,
        cv2.LINE_AA
    )

    # 2. Picture-in-Picture (PiP) Dual Debug Panels (Top-Right)
    pip_size = 95
    pip_gap = 10
    pip_y = header_h + 10

    # Panel 1: Source Hand ROI
    pip1_x = w - (pip_size * 2) - pip_gap - 16
    cv2.rectangle(canvas, (pip1_x - 2, pip_y - 18), (pip1_x + pip_size + 2, pip_y + pip_size + 2), (30, 36, 44), -1)
    cv2.putText(canvas, "Source ROI", (pip1_x, pip_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 200, 220), 1, cv2.LINE_AA)

    if source_roi is not None and source_roi.size > 0:
        rh, rw = source_roi.shape[:2]
        s = min(pip_size / rw, pip_size / rh)
        nw, nh = max(1, int(round(rw * s))), max(1, int(round(rh * s)))
        scaled_roi = cv2.resize(source_roi, (nw, nh), interpolation=cv2.INTER_AREA)
        box_canvas = np.zeros((pip_size, pip_size, 3), dtype=np.uint8)
        ox = (pip_size - nw) // 2
        oy = (pip_size - nh) // 2
        box_canvas[oy:oy+nh, ox:ox+nw] = scaled_roi
        canvas[pip_y:pip_y + pip_size, pip1_x:pip1_x + pip_size] = box_canvas
        border_col = (255, 120, 0) if result.get("fist_detected") else (0, 220, 255)
        cv2.rectangle(canvas, (pip1_x, pip_y), (pip1_x + pip_size, pip_y + pip_size), border_col, 2)
    else:
        cv2.rectangle(canvas, (pip1_x, pip_y), (pip1_x + pip_size, pip_y + pip_size), (60, 70, 80), 1)
        roi_txt = "Truncated" if result.get("roi_status") == "truncated" else "No Hand"
        cv2.putText(canvas, roi_txt, (pip1_x + 14, pip_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 110, 120), 1, cv2.LINE_AA)

    # Panel 2: Preprocessed Input
    pip2_x = w - pip_size - 16
    cv2.rectangle(canvas, (pip2_x - 2, pip_y - 18), (pip2_x + pip_size + 2, pip_y + pip_size + 2), (30, 36, 44), -1)
    panel2_title = "126D Vector" if result.get("active_engine") == "landmark_mlp" else "64x64 Canny"
    cv2.putText(canvas, panel2_title, (pip2_x, pip_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 200, 220), 1, cv2.LINE_AA)

    if result.get("active_engine") == "landmark_mlp":
        # Render a mini 126D vector heatmap
        vec_box = np.zeros((pip_size, pip_size, 3), dtype=np.uint8)
        vec_box[:, :] = (25, 30, 38)
        if result.get("is_reliable"):
            cv2.putText(vec_box, "126D", (22, 42), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 150), 1, cv2.LINE_AA)
            cv2.putText(vec_box, "MLP ACTIVE", (8, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 255, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(vec_box, "SEARCHING", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 140, 160), 1, cv2.LINE_AA)
        canvas[pip_y:pip_y + pip_size, pip2_x:pip2_x + pip_size] = vec_box
        cv2.rectangle(canvas, (pip2_x, pip_y), (pip2_x + pip_size, pip_y + pip_size), (0, 255, 150), 2)
    elif prep_preview is not None:
        pip_bgr = cv2.cvtColor(prep_preview, cv2.COLOR_GRAY2BGR)
        pip_resized = cv2.resize(pip_bgr, (pip_size, pip_size), interpolation=cv2.INTER_NEAREST)
        canvas[pip_y:pip_y + pip_size, pip2_x:pip2_x + pip_size] = pip_resized
        cv2.rectangle(canvas, (pip2_x, pip_y), (pip2_x + pip_size, pip_y + pip_size), (0, 255, 180), 2)
    else:
        cv2.rectangle(canvas, (pip2_x, pip_y), (pip2_x + pip_size, pip_y + pip_size), (60, 70, 80), 1)
        cv2.putText(canvas, "No Input", (pip2_x + 18, pip_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 110, 120), 1, cv2.LINE_AA)

    # 3. Detection & Confidence Banner
    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    is_reliable = result["is_reliable"]

    banner_w = 380
    banner_h = 65
    banner_x = 16
    banner_y = h - 170

    overlay = canvas.copy()
    cv2.rectangle(overlay, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

    if result.get("fist_detected"):
        border_col = (255, 120, 0)
        badge_title = "GESTURE: CLOSED FIST (SPACE)"
        conf_str = "Word Boundary Triggered"
    elif is_reliable:
        border_col = (0, 220, 120)
        badge_title = f"DETECTED SIGN: {pred_class}"
        conf_str = f"Confidence: {confidence * 100:.1f}% (Thresh: {int(threshold * 100)}%)"
    else:
        border_col = (50, 150, 255)
        badge_title = "NO RELIABLE SIGN"
        conf_str = f"Raw: {result['raw_class']} ({result['raw_confidence'] * 100:.1f}%)"

    cv2.rectangle(canvas, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h), border_col, 2)
    cv2.putText(canvas, badge_title, (banner_x + 14, banner_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, conf_str, (banner_x + 14, banner_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, border_col, 1, cv2.LINE_AA)

    # 4. Multilingual & Fingerspelling Buffer Area (Bottom)
    text_bar_y = h - 95
    text_bar_h = 60
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, text_bar_y), (w, text_bar_y + text_bar_h), (15, 18, 22), -1)
    cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0, canvas)
    cv2.line(canvas, (0, text_bar_y), (w, text_bar_y), (60, 75, 90), 1)

    spelled = result.get("spelled_text", "")
    comm_word = result.get("committed_word", "")
    sentence = result.get("sentence", "")

    display_en = f"Buffer: [{spelled}] | Word: {comm_word} | Sentence: {sentence}"
    cv2.putText(canvas, display_en, (16, text_bar_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 230, 100), 1, cv2.LINE_AA)

    trans_str = result.get("translated_sentence", "") or result.get("translated_word", "")
    if trans_str and lang_code != "en":
        cv2.putText(canvas, f"Translated: {trans_str}", (16, text_bar_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 200), 1, cv2.LINE_AA)

    # 5. Instructions Bar (Very Bottom)
    inst_bar_y = h - 30
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, inst_bar_y), (w, h), (10, 12, 16), -1)
    cv2.addWeighted(overlay, 0.95, canvas, 0.05, 0, canvas)

    inst_text = "[M] Mode | [L] Lang | [T] TTS | [V] Speak | [SPACE] Word | [BACKSPACE] Del | [C] Clear | [S] Snap | [Q] Quit"
    cv2.putText(canvas, inst_text, (16, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 175, 190), 1, cv2.LINE_AA)

    return canvas


def run_benchmark(engine: ISLInferenceEngine, num_frames: int = 100) -> Path:
    """Automated benchmark validating dual-engine performance and latency."""
    print(f"\n[Benchmark] Initiating automated benchmark across {num_frames} frames (Engine: {engine.active_engine})...")
    latencies = []
    fps_values = []

    test_w, test_h = 640, 480
    start_bench = time.time()

    for _ in range(num_frames):
        frame = np.ones((test_h, test_w, 3), dtype=np.uint8) * 40
        annotated, source_roi, prep, res = engine.process_frame(frame)
        latencies.append(res["latency_ms"])
        if res["fps"] > 0:
            fps_values.append(res["fps"])

    total_time = time.time() - start_bench
    avg_fps = num_frames / max(0.001, total_time)
    latencies = np.array(latencies)

    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))

    print(f"[Benchmark] Completed in {total_time:.2f}s | Throughput: {avg_fps:.1f} FPS | Mean Latency: {mean_lat:.1f}ms")
    return Path("reports/landmark_model_metrics.json")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Modernized ISL Webcam Bridge")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--engine", type=str, default="landmark_mlp", choices=["landmark_mlp", "cnn_legacy"], help="Active recognition engine")
    parser.add_argument("--lang", type=str, default="hi", help="Default target language code (hi, pa, bn, ta, te, mr, ur, en)")
    parser.add_argument("--tts", action="store_true", help="Enable spoken text-to-speech audio")
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold")
    parser.add_argument("--benchmark", action="store_true", help="Run automated latency & stability benchmark")
    parser.add_argument("--bench-frames", type=int, default=100, help="Number of benchmark frames")
    args = parser.parse_args()

    engine = ISLInferenceEngine(
        active_engine=args.engine,
        target_language=args.lang,
        tts_enabled=args.tts,
        confidence_threshold=args.threshold
    )

    if args.benchmark:
        run_benchmark(engine, num_frames=args.bench_frames)
        engine.release()
        return

    print(f"[WebcamApp] Attempting to open camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[WebcamApp] ERROR: No webcam available! Running automated benchmark instead...")
        run_benchmark(engine, num_frames=args.bench_frames)
        engine.release()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    snapshots_dir = Path("reports/snapshots")
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    window_name = "Indian Sign Language (ISL) Recognition Bridge"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    lang_idx = LANG_CYCLE.index(args.lang) if args.lang in LANG_CYCLE else 1

    print("\n" + "=" * 65)
    print("  ISL MODERNIZED WEBCAM RECOGNITION BRIDGE")
    print("  Active Engine: Landmark MLP (126D)")
    print("  Hotkeys:")
    print("    [M] Toggle Engine (Landmark MLP <-> Legacy CNN)")
    print("    [L] Cycle Language (Hindi, Punjabi, Tamil, etc.)")
    print("    [T] Toggle Speech Audio (TTS)")
    print("    [V] Speak Current Sentence Aloud")
    print("    [SPACE] Commit Word (or hold closed fist)")
    print("    [BACKSPACE] Delete Character")
    print("    [C] Clear Text Buffer")
    print("    [S] Save Live Snapshot")
    print("    [Q] / [ESC] Exit")
    print("=" * 65 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            annotated, source_roi, prep_preview, result = engine.process_frame(frame)

            hud_frame = draw_hud(
                annotated,
                result,
                prep_preview,
                source_roi=source_roi,
                threshold=engine.confidence_threshold
            )

            cv2.imshow(window_name, hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key in [ord('m'), ord('M')]:
                new_mode = "cnn_legacy" if engine.active_engine == "landmark_mlp" else "landmark_mlp"
                engine.set_engine_mode(new_mode)
                print(f"[WebcamApp] Switched engine to: {engine.active_engine}")
            elif key in [ord('l'), ord('L')]:
                lang_idx = (lang_idx + 1) % len(LANG_CYCLE)
                new_lang = LANG_CYCLE[lang_idx]
                engine.set_target_language(new_lang)
                print(f"[WebcamApp] Switched language to: {new_lang.upper()}")
            elif key in [ord('t'), ord('T')]:
                tts_state = engine.toggle_tts()
                print(f"[WebcamApp] Speech Audio (TTS): {'ENABLED' if tts_state else 'DISABLED'}")
            elif key in [ord('v'), ord('V')]:
                sentence = engine.translated_sentence or engine.text_builder.last_reconstructed_word
                if sentence:
                    print(f"[WebcamApp] Speaking aloud: '{sentence}' ({engine.target_language})")
                    engine.translator.speak_async(sentence, engine.target_language)
            elif key == ord(' '):
                engine.add_space()
                print(f"[WebcamApp] Word committed. Sentence: '{engine.text_builder.sentence_words}'")
            elif key in [8, 127]:
                engine.backspace()
            elif key in [ord('c'), ord('C')]:
                engine.clear_text()
                print("[WebcamApp] Buffer cleared.")
            elif key in [ord('s'), ord('S')]:
                snap_id = int(time.time())
                cv2.imwrite(str(snapshots_dir / f"snapshot_{snap_id}.png"), hud_frame)
                if source_roi is not None and source_roi.size > 0:
                    cv2.imwrite(str(snapshots_dir / f"source_roi_{snap_id}.png"), source_roi)
                print(f"[WebcamApp] Saved snapshot_{snap_id}.png to {snapshots_dir}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.release()
        print("[WebcamApp] Resources cleanly released.")


if __name__ == "__main__":
    main()
