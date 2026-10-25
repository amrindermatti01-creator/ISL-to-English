import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.inference import ISLInferenceEngine, capture_fullscreen_desktop

def test_fullscreen_auto_capture():
    print("=" * 65)
    print("  TESTING FULL-SCREEN AUTO-CAPTURE & SNAPSHOT PIPELINE")
    print("=" * 65)

    # 1. Test standalone desktop full-screen capture
    print("\n[Step 1] Testing capture_fullscreen_desktop()...")
    desktop_img = capture_fullscreen_desktop()
    if desktop_img is not None:
        print(f"  -> Desktop full-screen capture SUCCESS: shape={desktop_img.shape}, dtype={desktop_img.dtype}")
        print(f"  -> Screen Mean: {desktop_img.mean():.2f}, Min: {desktop_img.min()}, Max: {desktop_img.max()}")
        assert desktop_img.shape[0] >= 480 and desktop_img.shape[1] >= 640, "Captured image too small to be a desktop"
    else:
        print("  -> Desktop full-screen capture returned None (running in non-graphical context).")

    # 2. Test ISLInferenceEngine initialization
    print("\n[Step 2] Initializing ISLInferenceEngine...")
    engine = ISLInferenceEngine(
        model_path="artifacts/best_model_real.keras",
        class_mapping_path="artifacts/class_mapping_real.json",
        confidence_threshold=0.55
    )

    # 3. Load reference sign image
    test_img_path = r"C:\Users\hp\Real ISL Camera dataset\Letters\A.jpg"
    img = cv2.imread(test_img_path)
    assert img is not None, f"Could not load image: {test_img_path}"
    print(f"  -> Loaded test image: {img.shape}")

    # 4. Run inference
    annotated, prep_preview, result = engine.process_frame(img)
    pred_class = result["predicted_class"]
    conf = result["confidence"]
    print(f"  -> Inference result: [{pred_class}] ({conf * 100:.1f}%)")

    # 5. Simulate confirmed sign recognition event
    result["new_char_confirmed"] = True
    result["confirmed_char"] = pred_class

    # Create a synthetic full-application HUD frame (1280x720) to simulate live app window
    dummy_hud = np.zeros((720, 1280, 3), dtype=np.uint8)
    dummy_hud[:] = (24, 30, 38)
    cv2.putText(dummy_hud, "Indian Sign Language Recognizer - Full Window HUD", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 180), 2)
    # Inset the annotated camera frame
    ih, iw = annotated.shape[:2]
    dummy_hud[100:100 + min(ih, 500), 30:30 + min(iw, 600)] = annotated[:min(ih, 500), :min(iw, 600)]

    # 6. Execute save_accurate_snapshot with full-screen capture
    print("\n[Step 3] Executing save_accurate_snapshot(capture_fullscreen=True)...")
    snap_info = engine.save_accurate_snapshot(
        frame_bgr=annotated,
        prep_preview=prep_preview,
        result=result,
        prefix="auto_result",
        snapshots_dir="reports/snapshots",
        capture_fullscreen=True,
        app_window_img=dummy_hud
    )

    print("  -> Snapshot metadata output:")
    for k, v in snap_info.items():
        print(f"     * {k}: {v}")

    # 7. Verification of saved files
    print("\n[Step 4] Verifying generated screenshot files on disk...")
    primary_file = Path(snap_info["snapshot_path"])
    assert primary_file.exists() and primary_file.stat().st_size > 0, f"Primary file missing: {primary_file}"
    primary_img = cv2.imread(str(primary_file))
    print(f"  -> Primary full-screen snapshot verified: shape={primary_img.shape}, size={primary_file.stat().st_size:,} bytes")
    # Verify it has full-screen / full-window scale (at least 720p or 1080p, not just small 64x64 or 200x200 crop)
    assert primary_img.shape[0] >= 700 and primary_img.shape[1] >= 1200, f"Expected full-screen/full-window dimensions, got {primary_img.shape}"

    if snap_info["fullscreen_path"]:
        fs_file = Path(snap_info["fullscreen_path"])
        assert fs_file.exists() and fs_file.stat().st_size > 0, "Fullscreen snapshot missing or empty"
        print(f"  -> Verified FULL SCREEN screenshot: {fs_file.name} ({fs_file.stat().st_size:,} bytes)")

    if snap_info["window_path"]:
        win_file = Path(snap_info["window_path"])
        assert win_file.exists() and win_file.stat().st_size > 0, "Window snapshot missing or empty"
        print(f"  -> Verified APP WINDOW HUD screenshot: {win_file.name} ({win_file.stat().st_size:,} bytes)")

    if snap_info["result_path"]:
        res_file = Path(snap_info["result_path"])
        assert res_file.exists() and res_file.stat().st_size > 0, "Result snapshot missing or empty"
        print(f"  -> Verified CAMERA RESULT crop screenshot: {res_file.name} ({res_file.stat().st_size:,} bytes)")

    if snap_info["canny_path"]:
        canny_file = Path(snap_info["canny_path"])
        assert canny_file.exists() and canny_file.stat().st_size > 0, "Canny edge map missing or empty"
        print(f"  -> Verified CNN CANNY INPUT preview: {canny_file.name} ({canny_file.stat().st_size:,} bytes)")

    print("\n" + "=" * 65)
    print("  [SUCCESS] FULL-SCREEN SCREENSHOT CAPTURE FULLY VERIFIED!")
    print("=" * 65)
    engine.release()

if __name__ == "__main__":
    test_fullscreen_auto_capture()
