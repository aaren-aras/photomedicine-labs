from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

import paths # utils/paths.py
from model import dice_coefficient, DiceMetric

MODEL_PATH = Path('benchmark') / 'segmentation_best.keras'

IMG_VALID_DIR = Path('data') / 'processed' / 'images' / 'valid'
MASK_VALID_DIR = Path('data') / 'processed' / 'masks' / 'valid'

OUTPUT_DIR = Path('images')
OUTPUT_DIR.mkdir(exist_ok=True)

model = load_model(MODEL_PATH, custom_objects={
    'dice_coefficient': dice_coefficient, 'DiceMetric': DiceMetric
    }, safe_mode=False) # deserialize custom code structs without worry

# Load 1 validation sample
img = np.load(sorted(IMG_VALID_DIR.iterdir())[0]).astype(np.float32) / 255.0
mask = np.load(sorted(MASK_VALID_DIR.iterdir())[0]).astype(np.float32)
img_input = np.expand_dims(np.expand_dims(img, axis=-1), axis=0) # (1, 400, 400, 1)

# Predict
pred = model.predict(img_input)[0, :, :, 0]
print(f"Prediction stats:")
print(f"  Min: {pred.min():.4f}, Max: {pred.max():.4f}, Mean: {pred.mean():.4f}")
print(f"  Values > 0.5: {(pred > 0.5).sum()} pixels ({(pred > 0.5).mean()*100:.1f}%)")
print(f"Ground truth vessel pixels: {mask.sum()} ({mask.mean()*100:.1f}%)")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Input Image')
axes[1].imshow(mask, cmap='gray')
axes[1].set_title(f'Ground Truth ({mask.mean()*100:.1f}% vessels)')
axes[2].imshow(pred, cmap='gray')
axes[2].set_title(f'Prediction ({pred.mean()*100:.1f}% confidence)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'prediction_check.png', dpi=150)
print("*SAVED: prediction_check.png")
