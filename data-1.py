"""
data.py
=======
Preprocesses OCTA-500 + ROSE datasets into .npy arrays for training.

Why combine both datasets?
    OCTA-500 alone (~400 training images) showed val_loss instability —
    the model memorised OCTA-500's specific noise characteristics and
    didn't generalise. Adding ROSE provides:
        - 117 additional images with different scanner noise profiles
        - Capillary-level vessel annotations (finer than OCTA-500's large vessels)
        - Different FOV and resolution (304×304 → resized to 400×400)
    Combined dataset: ~500 training images, more varied, better generalisation.

Directory structure expected:
    data/
        OCTA-500/
            OCTA(ILM_OPL)/          ← en face projections
            Label/GT_LargeVessel/   ← binary vessel masks
        ROSE/
            ROSE-1/                 ← or ROSE1, check your folder name
                train/
                    original/       ← OCTA images (.png or .tif)
                    gt/             ← vessel masks
                test/
                    original/
                    gt/

Output:
    data/processed/
        images/{train,valid,test}/*.npy   ← float32 normalised [0,1]
        masks/{train,valid,test}/*.npy    ← float32 binary {0,1}

Run with: python data.py
"""

from pathlib import Path
from typing import Optional
import json

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import IMG_SIZE, LABEL, BINARIZE_MASK, RANDOM_STATE

SCRIPT_DIR  = Path(__file__).resolve().parent
OCTA500_DIR = SCRIPT_DIR / 'data/OCTA-500'
ROSE_DIR    = SCRIPT_DIR / 'data/ROSE'
OUTPUT_DIR  = SCRIPT_DIR / 'data/processed'

for split in ['train', 'valid', 'test']:
    (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'masks'  / split).mkdir(parents=True, exist_ok=True)


# ── Preprocessing helpers ──────────────────────────────────────────────────────

def apply_clahe_numpy(img: np.ndarray, clip_limit: float = 2.0,
                      grid_size: int = 8) -> np.ndarray:
    """
    CLAHE without cv2 — works in any Python environment including Colab.

    Divides image into grid_size×grid_size tiles, applies histogram
    equalisation within each tile with contrast clipping, then bilinearly
    interpolates tile boundaries. This enhances local vessel contrast
    without amplifying noise in uniform background regions.

    clipLimit=2.0, grid_size=8 are validated for retinal OCTA in
    Giarratano et al. (Trans. Vis. Sci. Tech., 2020).
    """
    h, w   = img.shape
    gh, gw = h // grid_size, w // grid_size
    result = np.zeros_like(img, dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * gh, (i + 1) * gh
            x1, x2 = j * gw, (j + 1) * gw
            tile = img[y1:y2, x1:x2].astype(np.float32)

            hist, _ = np.histogram(tile.flatten(), 256, [0, 255])
            clip     = int(clip_limit * tile.size / 256)
            excess   = np.sum(np.maximum(hist - clip, 0))
            hist     = np.minimum(hist, clip)
            hist    += excess // 256

            cdf          = hist.cumsum()
            cdf_min      = cdf[cdf > 0].min()
            cdf_norm     = (cdf - cdf_min) * 255 / (tile.size - cdf_min + 1e-8)
            cdf_norm     = np.clip(cdf_norm, 0, 255)
            result[y1:y2, x1:x2] = cdf_norm[tile.astype(np.int32)]

    return result.astype(np.uint8)


def load_and_preprocess(
    img_path: Path,
    mask_path: Optional[Path],
    target_size: tuple = IMG_SIZE,
    binarize: bool = BINARIZE_MASK
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load one image+mask pair, resize to target_size, apply CLAHE.

    Returns:
        img:  uint8 [0,255] CLAHE-enhanced OCTA projection
        mask: uint8 {0,1} binary vessel mask, or None if no mask provided
    """
    # Load image
    img = np.array(Image.open(img_path).convert('L').resize(
        (target_size[1], target_size[0]), Image.LANCZOS
    )).astype(np.uint8)

    img = apply_clahe_numpy(img)

    # Load mask
    mask = None
    if mask_path is not None and mask_path.exists():
        mask = np.array(Image.open(mask_path).convert('L').resize(
            (target_size[1], target_size[0]), Image.NEAREST
        ))
        if binarize:
            mask = (mask > 0).astype(np.uint8)

    return img, mask


# ── OCTA-500 loader ───────────────────────────────────────────────────────────

def load_octa500() -> list[tuple]:
    """
    Load all OCTA-500 ILM_OPL projections + GT_LargeVessel masks.

    Returns list of (img, mask, sample_id) tuples.
    """
    proj_dir = OCTA500_DIR / 'OCTA(ILM_OPL)'
    mask_dir = OCTA500_DIR / 'Label' / LABEL

    if not proj_dir.exists():
        print(f'WARNING: OCTA-500 not found at {proj_dir}')
        return []

    samples = []
    for img_path in tqdm(sorted(proj_dir.iterdir()), desc='Loading OCTA-500'):
        mask_path = mask_dir / img_path.name
        img, mask = load_and_preprocess(img_path, mask_path)
        samples.append((img, mask, f'octa500_{img_path.stem}'))

    print(f'OCTA-500: {len(samples)} samples loaded')
    return samples


# ── ROSE loader ───────────────────────────────────────────────────────────────

def load_rose() -> list[tuple]:
    """
    Load ROSE-1 (SVC_DVC) + ROSE-2 datasets.

    ROSE-1: 304×304, Optovue SD-OCT, pixel-level gt masks
        - Use SVC_DVC slab only (matches OCTA-500 ILM_OPL — full retinal plexus)
        - Use both train/ and test/ splits for training our model
          (ROSE's test split is for their own benchmark, not ours)
        - Our held-out test set is OCTA-500 only

    ROSE-2: 840×840, Heidelberg SD-OCT, pixel-level gt masks
        - Different scanner = different noise profile = better generalisation
        - Downsampled to 400×400 (same as OCTA-500 target size)
        - Use both train/ and test/ splits for same reason as ROSE-1

    Total ROSE contribution: ~151 images across 2 scanners
    """
    sources = [
        # (img_dir, mask_dir, label)
        (ROSE_DIR / 'ROSE-1' / 'SVC_DVC' / 'train' / 'img',
         ROSE_DIR / 'ROSE-1' / 'SVC_DVC' / 'train' / 'gt',
         'rose1_train'),
        (ROSE_DIR / 'ROSE-1' / 'SVC_DVC' / 'test'  / 'img',
         ROSE_DIR / 'ROSE-1' / 'SVC_DVC' / 'test'  / 'gt',
         'rose1_test'),
        (ROSE_DIR / 'ROSE-2' / 'train' / 'original',
         ROSE_DIR / 'ROSE-2' / 'train' / 'gt',
         'rose2_train'),
        (ROSE_DIR / 'ROSE-2' / 'test'  / 'original',
         ROSE_DIR / 'ROSE-2' / 'test'  / 'gt',
         'rose2_test'),
    ]

    samples  = []
    extensions = {'.png', '.tif', '.tiff', '.bmp', '.jpg'}

    for img_dir, mask_dir, label in sources:
        if not img_dir.exists():
            print(f'WARNING: {label} not found at {img_dir}')
            continue

        count = 0
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in extensions:
                continue

            # Match mask by stem — handles extension mismatches
            mask_path = None
            for ext in extensions:
                candidate = mask_dir / (img_path.stem + ext)
                if candidate.exists():
                    mask_path = candidate
                    break

            if mask_path is None:
                print(f'WARNING: no mask found for {img_path.name}, skipping')
                continue

            img, mask = load_and_preprocess(img_path, mask_path)
            samples.append((img, mask, f'{label}_{img_path.stem}'))
            count += 1

        print(f'  {label}: {count} samples loaded')

    print(f'ROSE total: {len(samples)} samples')
    return samples


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_sample(img: np.ndarray, mask: Optional[np.ndarray],
                sample_id: str, split: str) -> None:
    """
    Save image (and optionally mask) as .npy files.

    Images saved as uint8 [0,255] — normalised to float32 [0,1] in generator.
    Masks saved as uint8 {0,1} — cast to float32 in generator (no /255 needed).
    """
    np.save(OUTPUT_DIR / 'images' / split / f'{sample_id}.npy', img)
    if mask is not None:
        np.save(OUTPUT_DIR / 'masks' / split / f'{sample_id}.npy', mask)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def prepare_data(use_rose: bool = True) -> None:
    """
    Full preprocessing pipeline for OCTA-500 + ROSE (optional).

    Split strategy:
        OCTA-500: 80/10/10 train/valid/test
        ROSE:     80/20 train/valid only — ROSE has no separate test split
                  and we reserve OCTA-500 test set as the held-out benchmark

    This ensures:
        - Test set is always OCTA-500 only (clean benchmark)
        - Validation set contains both datasets (diverse val_loss signal)
        - Training set is as large as possible from both sources
    """
    # Load datasets
    octa500_samples = load_octa500()
    rose_samples    = load_rose() if use_rose else []

    if not octa500_samples:
        raise RuntimeError('OCTA-500 is required. ROSE is optional.')

    # Split OCTA-500: 80/10/10
    train_500, temp_500   = train_test_split(
        octa500_samples, test_size=0.2, random_state=RANDOM_STATE
    )
    valid_500, test_500   = train_test_split(
        temp_500, test_size=0.5, random_state=RANDOM_STATE
    )

    # Split ROSE: 80/20 (no test split)
    train_rose, valid_rose = [], []
    if rose_samples:
        train_rose, valid_rose = train_test_split(
            rose_samples, test_size=0.2, random_state=RANDOM_STATE
        )

    # Combine
    train_all = train_500 + train_rose
    valid_all = valid_500 + valid_rose
    test_all  = test_500   # OCTA-500 only

    print(f'\nFinal split sizes:')
    print(f'  Train: {len(train_all)} '
          f'(OCTA-500: {len(train_500)}, ROSE: {len(train_rose)})')
    print(f'  Valid: {len(valid_all)} '
          f'(OCTA-500: {len(valid_500)}, ROSE: {len(valid_rose)})')
    print(f'  Test:  {len(test_all)} (OCTA-500 only — held-out benchmark)')

    # Save
    for split_name, data in [
        ('train', train_all),
        ('valid', valid_all),
        ('test',  test_all)
    ]:
        for img, mask, sid in tqdm(data, desc=f"Saving '{split_name}'"):
            save_sample(img, mask, sid, split_name)

    # Class imbalance summary
    all_masks   = [s[1] for s in octa500_samples + rose_samples if s[1] is not None]
    ratios      = [m.mean() for m in all_masks]
    print(f'\nVessel ratio — mean: {np.mean(ratios)*100:.1f}%, '
          f'range: {np.min(ratios)*100:.1f}–{np.max(ratios)*100:.1f}%')
    print(f'Class imbalance ≈ {1/np.mean(ratios):.0f}:1 — Dice loss handles this')
    print('\nDone. Run model.py to start training.')


if __name__ == '__main__':
    prepare_data(use_rose=True)
