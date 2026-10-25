import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp
from src.preprocessing import preprocess_image
from src.inference import ISLInferenceEngine

model = tf.keras.models.load_model('artifacts/best_model.keras')
with open('artifacts/class_mapping.json') as f:
    mapping = json.load(f)
idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}

engine = ISLInferenceEngine(
    model_path='artifacts/best_model.keras',
    class_mapping_path='artifacts/class_mapping.json',
    confidence_threshold=0.0
)

root = Path(r'C:\Users\hp\Real ISL Camera dataset\Letters')
print('='*75)
print(f'{"Letter":<6} | {"Direct Pred":<12} | {"Direct Conf":<11} | {"MediaPipe Pred":<14} | {"MP Conf":<8}')
print('='*75)

direct_correct = 0
mp_correct = 0
total = 0

for f in sorted(root.glob('*.jpg')):
    true_label = f.stem
    img = cv2.imread(str(f))
    if img is None:
        continue
    total += 1

    # 1. Direct Preprocessing
    x_dir = preprocess_image(img)
    preds_dir = model.predict(np.expand_dims(x_dir, 0), verbose=0)[0]
    pdir_idx = int(np.argmax(preds_dir))
    pdir_class = idx_to_class[pdir_idx]
    pdir_conf = float(preds_dir[pdir_idx])
    if pdir_class == true_label:
        direct_correct += 1

    # 2. MediaPipe Pipeline
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_res = engine.hands.process(img_rgb)
    if mp_res.multi_hand_landmarks:
        roi_sq, bbox = engine.extract_hand_roi(img, mp_res.multi_hand_landmarks)
        if roi_sq is not None:
            x_mp = preprocess_image(roi_sq)
            preds_mp = model.predict(np.expand_dims(x_mp, 0), verbose=0)[0]
            pmp_idx = int(np.argmax(preds_mp))
            pmp_class = idx_to_class[pmp_idx]
            pmp_conf = float(preds_mp[pmp_idx])
        else:
            pmp_class = "NO_ROI"
            pmp_conf = 0.0
    else:
        pmp_class = "NO_HAND"
        pmp_conf = 0.0

    if pmp_class == true_label:
        mp_correct += 1

    dir_mark = "OK" if pdir_class == true_label else "FAIL"
    mp_mark = "OK" if pmp_class == true_label else "FAIL"

    print(f'{true_label:<6} | {pdir_class:<2} ({dir_mark:<4})   | {pdir_conf*100:5.1f}%     | {pmp_class:<4} ({mp_mark:<4})     | {pmp_conf*100:5.1f}%')

print('='*75)
print(f'Direct Pipeline Accuracy:    {direct_correct}/{total} ({direct_correct/total*100:.1f}%)')
print(f'MediaPipe Pipeline Accuracy: {mp_correct}/{total} ({mp_correct/total*100:.1f}%)')
