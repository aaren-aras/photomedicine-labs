"""
data.py
=======
Preprocesses OCTA-500 dataset into .npy arrays for model training.

OCTA-500 structure expected at data/OCTA-500/:
    OCTA(ILM_OPL)/       ← en face projection images (.bmp)
    Label/
        GT_LargeVessel/  ← binary vessel masks (.bmp)
        GT_Artery/
        GT_Vein/
        GT_FAZ/

Output written to data/OCTA-500_processed/:
    images/{train,valid,test}/*.npy   ← uint8 [0,255] CLAHE-enhanced projections
    masks/{train,valid,test}/*.npy    ← uint8 {0,1} binary vessel masks
    metadata/{train,valid,test}/*.json

Why ILM_OPL slab?
    The ILM (Inner Limiting Membrane) to OPL (Outer Plexiform Layer) slab
    captures both the superficial and deep retinal capillary plexuses —
    the two main vascular networks targeted by GT_LargeVessel labels.
    Other slabs (OPL_BM) capture the avascular outer retina and
    choriocapillaris, which are not relevant for large vessel segmentation.

Why CLAHE preprocessing?
    OCTA images suffer from uneven illumination and low local contrast in
    vessel regions. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    enhances local contrast in small tiles, making vessel edges more distinct
    without amplifying noise globally (unlike standard histogram equalization).
    clipLimit=2.0 and tileGridSize=(8,8) are standard for retinal/OCTA imaging
    and were validated in Giarratano et al. (Trans. Vis. Sci. Tech., 2020).
"""

from pathlib import Path
from typing import Generator
import json

from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split

from config import LABEL, BINARIZE_MASK, RANDOM_STATE

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_DIR  = (SCRIPT_DIR / 'data/OCTA-500').resolve()
OUTPUT_DIR = (SCRIPT_DIR / 'data/OCTA-500_processed').resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create output directory structure for all splits
for split in ['train', 'valid', 'test']:
    (OUTPUT_DIR / 'images'   / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'masks'    / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'metadata' / split).mkdir(parents=True, exist_ok=True)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance
    vessel contrast in OCTA projections.

    How it works:
        The image is divided into small tiles (tileGridSize). Within each tile,
        histogram equalization is applied independently, enhancing local contrast.
        The 'clip limit' prevents over-amplification of noise in nearly-uniform
        regions (e.g., background) by redistributing histogram bins that exceed
        the limit across all intensity levels.

    Args:
        img: uint8 grayscale image [0, 255]

    Returns:
        uint8 CLAHE-enhanced image [0, 255]
    """
    clahe = cv2.createCLAHE(
        clipLimit=2.0,      # max contrast amplification per tile — 2.0 prevents noise boost
        tileGridSize=(8, 8) # divide 400x400 image into 8x8 = 64 tiles of 50x50px each
    )
    return clahe.apply(img)


def process_sample(
    id: str,
    label: str,
    binarize_mask: bool = True
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Load one OCTA-500 sample: en face projection + vessel mask.

    Processing steps:
        1. Load ILM_OPL projection as uint8 grayscale
        2. Apply CLAHE to enhance vessel contrast
        3. Load corresponding ground truth mask
        4. Optionally binarize mask: pixel values {0,255} → {0,1}

    Args:
        id:             filename (e.g., '10001.bmp')
        label:          mask subdirectory (e.g., 'GT_LargeVessel')
        binarize_mask:  if True, maps mask to binary {0,1}

    Returns:
        (img, mask, id) where img is uint8 [0,255] and mask is uint8 {0,1}
    """
    # Load en face OCTA projection and convert to grayscale uint8
    img_path = INPUT_DIR / 'OCTA(ILM_OPL)' / id
    img = np.array(Image.open(img_path).convert('L')).astype(np.uint8)

    # Enhance vessel contrast with CLAHE before saving
    # This is done here (not in the generator) so it only runs once,
    # not every epoch
    img = apply_clahe(img)

    # Load binary vessel mask
    mask_path = INPUT_DIR / 'Label' / label / id
    mask = np.array(Image.open(mask_path).convert('L'))

    # Binarize: OCTA-500 masks have pixel values {0, 255}
    # We map to {0, 1} because:
    #   - BinaryCrossentropy expects targets in [0,1]
    #   - Dice coefficient computation uses element-wise multiplication
    #     where values must be 0 or 1, not 0 or 255
    if binarize_mask:
        mask = (mask > 0).astype(np.uint8)

    return img, mask, id


def save_sample(
    img: np.ndarray,
    mask: np.ndarray,
    id: str,
    split: str
) -> None:
    """
    Save processed image and mask as .npy arrays with metadata.

    Why .npy instead of .png?
        - np.load() is ~3-5× faster than PIL Image.open() for training loops
        - Preserves exact array values without JPEG compression artifacts
        - Easier to load with consistent dtypes (no conversion needed)

    Args:
        img:   uint8 array [0,255] — CLAHE-enhanced OCTA projection
        mask:  uint8 array {0,1}  — binary vessel mask
        id:    original filename (e.g., '10001.bmp')
        split: 'train', 'valid', or 'test'
    """
    # Save image — kept as uint8 here, normalized to float32 [0,1] in generator
    np.save(OUTPUT_DIR / 'images' / split / f'{id}.npy', img)

    # Save mask — uint8 {0,1}, cast to float32 in generator (no /255 needed)
    np.save(OUTPUT_DIR / 'masks' / split / f'{id}.npy', mask)

    # Save metadata for dataset analysis and debugging
    metadata = {
        'id': id,
        'split': split,
        'image_shape': list(img.shape),
        'vessel_pixels': int(mask.sum()),
        'total_pixels': int(mask.size),
        # vessel_ratio tells us class imbalance — typically ~0.10-0.15 for retinal OCTA
        # A very low ratio (<0.05) means sparse vessels → model may predict all background
        'vessel_ratio': float(mask.mean()),
    }
    metadata_path = OUTPUT_DIR / 'metadata' / split / f'{id}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def prepare_data() -> None:
    """
    Full preprocessing pipeline for OCTA-500.

    Split rationale (80/10/10):
        - 400 training samples: enough to learn vessel patterns with augmentation
        - 50 validation samples: enough for stable val_loss/val_dice estimates
        - 50 test samples: held out entirely until final evaluation
        - random_state=28 ensures reproducible splits across runs

    This split is standard for datasets of this size in medical imaging literature
    (e.g., Giarratano et al. 2020 use similar proportions on retinal OCTA datasets).
    """
    all_samples = []
    proj_ids = sorted([p.name for p in (INPUT_DIR / 'OCTA(ILM_OPL)').iterdir()])

    for id in tqdm(proj_ids, desc='Processing OCTA-500 samples'):
        sample = process_sample(id, label=LABEL, binarize_mask=BINARIZE_MASK)
        all_samples.append(sample)

    # 80% train, 10% valid, 10% test
    train_data, temp   = train_test_split(all_samples, test_size=0.2,  random_state=RANDOM_STATE)
    valid_data, test_data = train_test_split(temp,     test_size=0.5,  random_state=RANDOM_STATE)

    splits = {'train': train_data, 'valid': valid_data, 'test': test_data}
    for split_name, data in splits.items():
        for img, mask, id in tqdm(data, desc=f"Saving '{split_name}' split"):
            save_sample(img, mask, id, split_name)

    # Print class imbalance summary — important for understanding loss behavior
    all_masks = [s[1] for s in all_samples]
    ratios = [m.mean() for m in all_masks]
    print(f'\n*COMPLETE: {len(all_samples)} samples processed')
    print(f'  Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}')
    print(f'  Vessel ratio — mean: {np.mean(ratios):.3f}, '
          f'min: {np.min(ratios):.3f}, max: {np.max(ratios):.3f}')
    print(f'  → Class imbalance ~{1/np.mean(ratios):.0f}:1 (background:vessel)')
    print(f'  → Dice loss handles this naturally; no class weighting needed')


if __name__ == '__main__':
    prepare_data()
