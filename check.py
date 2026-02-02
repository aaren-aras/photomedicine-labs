# diagnostic.py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_DIR = Path('data/OCTA-500_processed')
IMG_TRAIN_DIR = OUTPUT_DIR / 'images' / 'train'
MASK_TRAIN_DIR = OUTPUT_DIR / 'masks' / 'train'
IMG_VALID_DIR = OUTPUT_DIR / 'images' / 'valid'
MASK_VALID_DIR = OUTPUT_DIR / 'masks' / 'valid'

print("=" * 60)
print("CHECKING RAW DATA")
print("=" * 60)

# Check 3 training samples
for i, (img_file, mask_file) in enumerate(zip(sorted(IMG_TRAIN_DIR.iterdir())[:3], 
                                                sorted(MASK_TRAIN_DIR.iterdir())[:3])):
    img = np.load(img_file)
    mask = np.load(mask_file)
    print(f"\nTrain sample {i}:")
    print(f"  Image: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")
    print(f"  Mask:  shape={mask.shape}, dtype={mask.dtype}, range=[{mask.min()}, {mask.max()}]")
    print(f"  Mask vessel ratio: {mask.mean():.4f}")

# Check 3 validation samples
for i, (img_file, mask_file) in enumerate(zip(sorted(IMG_VALID_DIR.iterdir())[:3], 
                                                sorted(MASK_VALID_DIR.iterdir())[:3])):
    img = np.load(img_file)
    mask = np.load(mask_file)
    print(f"\nValid sample {i}:")
    print(f"  Image: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")
    print(f"  Mask:  shape={mask.shape}, dtype={mask.dtype}, range=[{mask.min()}, {mask.max()}]")
    print(f"  Mask vessel ratio: {mask.mean():.4f}")

# Visualize one pair from each
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

train_img = np.load(sorted(IMG_TRAIN_DIR.iterdir())[0])
train_mask = np.load(sorted(MASK_TRAIN_DIR.iterdir())[0])
axes[0, 0].imshow(train_img, cmap='gray')
axes[0, 0].set_title('Train Image')
axes[0, 1].imshow(train_mask, cmap='gray')
axes[0, 1].set_title(f'Train Mask (ratio: {train_mask.mean():.3f})')

val_img = np.load(sorted(IMG_VALID_DIR.iterdir())[0])
val_mask = np.load(sorted(MASK_VALID_DIR.iterdir())[0])
axes[1, 0].imshow(val_img, cmap='gray')
axes[1, 0].set_title('Val Image')
axes[1, 1].imshow(val_mask, cmap='gray')
axes[1, 1].set_title(f'Val Mask (ratio: {val_mask.mean():.3f})')

plt.tight_layout()
plt.savefig('raw_data_check.png', dpi=150)
print("\n" + "=" * 60)
print("Saved raw_data_check.png")
print("=" * 60)