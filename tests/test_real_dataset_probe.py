import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import tensorflow as tf
import json
from src.preprocessing import preprocess_image

model = tf.keras.models.load_model('artifacts/best_model.keras')
with open('artifacts/class_mapping.json') as f:
    mapping = json.load(f)
idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}

print('--- Testing on C:\\Users\\hp\\Indian (Training dataset samples) ---')
train_root = Path(r'C:\Users\hp\Indian')
for label in ['A', 'B', 'C', 'D', 'E', 'Z']:
    p = train_root / label
    if p.exists():
        imgs = list(p.glob('*.*'))[:2]
        for img_path in imgs:
            x = preprocess_image(img_path)
            x_batch = np.expand_dims(x, axis=0)
            preds = model.predict(x_batch, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            top_class = idx_to_class[top_idx]
            conf = float(preds[top_idx])
            print(f'Train {label} ({img_path.name}) -> Pred: {top_class} ({conf*100:.1f}%)')

print('\n--- Testing directly on C:\\Users\\hp\\Real ISL Camera dataset\\Letters (Direct Preprocessing) ---')
real_root = Path(r'C:\Users\hp\Real ISL Camera dataset')
for label in ['A', 'B', 'C', 'D', 'E', 'Z']:
    img_path = real_root / 'Letters' / f'{label}.jpg'
    if img_path.exists():
        x = preprocess_image(img_path)
        x_batch = np.expand_dims(x, axis=0)
        preds = model.predict(x_batch, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        top_class = idx_to_class[top_idx]
        conf = float(preds[top_idx])
        print(f'Real Letter {label} -> Direct Pred: {top_class} ({conf*100:.1f}%)')
