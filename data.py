"""
Preprocesses OCTA-500 and ROSE datasets into uint8 numpy arrays,
with the following directory structure expected:

data/
    OCTA-500/
        OCTA(ILM_OPL)/          
        Label/GT_LargeVessel/   
    ROSE/
        ROSE-1/                 
            train/
                original/       
                gt/             
            test/
                original/
                gt/
"""

from pathlib import Path
import json

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import (
    CLIP_LIMIT, TILE_GRID_SIZE, IMG_SIZE, PROJECTIONS, 
    OCTA500_LABEL, BINARIZE_MASK, RANDOM_STATE
)

ROOT_DIR = Path(__file__).resolve().parent
ROSE_DIR = ROOT_DIR / 'data' / 'ROSE'
OUTPUT_DIR = ROOT_DIR / 'data' / 'processed2'

for split in ['train', 'valid', 'test']:
    (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'masks'  / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'metadata' / split).mkdir(parents=True, exist_ok=True)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Applies (C)ontrast (L)imited (A)daptive (H)istogram (E)qualization 
    to enhance local vessel contrast in OCTA projections.
    """
    # Divides images into tiles, then computes intensity histograms (256 bins 
    # for 0-255 intensity values) and flattens/stretches distribution for each.
    clahe = cv2.createCLAHE(
        clipLimit=CLIP_LIMIT, # separate process from histogram equalization (see config)
        tileGridSize=TILE_GRID_SIZE # looks at neighbouring pixels for blending tile boundaries
    )
    return clahe.apply(img)


def load_and_preprocess(img_dir: Path, mask_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Loads, resizes, and normalizes image-mask pairs."""
    img = np.array(Image.open(img_dir).convert('L').resize( # converts to greyscale
        # Lanczos resampling is high-fidelity, and preserves vessel textures well
        (IMG_SIZE[1], IMG_SIZE[0]), Image.LANCZOS 
    )).astype(np.uint8) # uint8 [0, 255] -> uint8 [0, 1]

    # Runs CLAHE once, before entering the generators in model.py
    img = apply_clahe(img) 

    mask = np.array(Image.open(mask_dir).convert('L').resize(
        # Nearest-neighbour interpolation here to ensure mask remains binary
        (IMG_SIZE[1], IMG_SIZE[0]), Image.NEAREST 
    ))
    if BINARIZE_MASK: # for BinaryCrossentropy and Dice in model.py
        mask = (mask > 0).astype(np.uint8) # uint8 [0, 255] -> uint8 [0, 1]

    return img, mask


def load_octa500() -> list[tuple]:
    """
    Loads all OCTA-500 ILM_OPL projections and GT_LargeVessel 
    labels, and returns a list of (img, mask, sample_id) tuples.
    """
    octa500_dir = ROOT_DIR / 'data' / 'OCTA-500'
    proj_dir = octa500_dir / PROJECTIONS['octa500']
    label_dir = octa500_dir / 'Label' / OCTA500_LABEL

    if not proj_dir.exists():
        raise FileNotFoundError(
            f"OCTA-500 '{PROJECTIONS['octa500']}' projections missing at '{proj_dir}'... " 
            "Pipeline cannot proceed."
        )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"OCTA-500 '{OCTA500_LABEL}' labels missing at '{label_dir}'... "
            "Pipeline cannot proceed."
        )

    samples = []
    for proj_path in tqdm(sorted(proj_dir.iterdir()), desc='Processing OCTA-500 samples'):
        label_path = label_dir / proj_path.name
        img, mask = load_and_preprocess(proj_path, label_path)
        samples.append((img, mask, f'octa500_{proj_path.stem}'))

    print(f'*COMPLETE: {len(samples)} OCTA-500 samples loaded.')
    return samples


def load_rose() -> list[tuple]:
    """
    Loads all ROSE SVC_SVC projections and labels, and 
    returns a list of (img, mask, sample_id) tuples.
    """   
    sources = []
    # Dynamically pieces together paths
    for split in ['train', 'test']:
        r1_base_path = ROSE_DIR / 'ROSE-1' / PROJECTIONS['rose1'] / split
        # (proj_dir, label_dir, label_id, label_ext)
        sources.append((r1_base_path / 'img', r1_base_path / 'gt', f'rose1_{split}', '.tif'))

        r2_base_path = ROSE_DIR / 'ROSE-2' / split
        sources.append((r2_base_path / 'original', r2_base_path / 'gt', f'rose2_{split}', '.png'))

    samples = []
    for proj_dir, label_dir, label_id, label_ext in sources:
        if not proj_dir.exists() or not label_dir.exists():
            print(f"*WARNING: Directories missing for '{label_id}' at:\n"
                  f'   Proj: {proj_dir}\n   Label: {label_dir}\n Skipping split...')
            continue

        for proj_path in tqdm(sorted(proj_dir.iterdir()), desc=f"Processing '{label_id}'"):
            label_path = label_dir / f'{proj_path.stem}{label_ext}'
            img, mask = load_and_preprocess(proj_path, label_path)
            samples.append((img, mask, f'{label_id}_{proj_path.stem}'))

    print(f'*COMPLETE: {len(samples)} ROSE samples loaded.')
    return samples


def save_sample(img: np.ndarray, mask: np.ndarray, sample_id: str, split: str) -> None:
    """
    Saves image-mask pairs as numpy arrays for faster loading 
    during training, and metadata for analysis and debugging.
    """
    np.save(OUTPUT_DIR / 'images' / split / f'{sample_id}.npy', img)
    np.save(OUTPUT_DIR / 'masks' / split / f'{sample_id}.npy', mask)
    
    metadata = {
        'id': sample_id,
        'split': split,
        'image_shape': list(img.shape),
        'vessel_pixels': int(mask.sum()),
        'total_pixels': int(mask.size),
        # Tells us class imbalance (~0.10-0.15 for retinal OCTA)
        'vessel_ratio': float(mask.mean()), # model might predict all bkgd for <0.05 
    }
    metadata_path = OUTPUT_DIR / 'metadata' / split / f'{sample_id}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2) # force line breaks


def prepare_data(use_rose: bool = True) -> None:
    """
    Processes projections and corresponding labels for model training,
    reserving OCTA-500 as the held-out benchmark for testing with splits 
    of similar proportions to literature (Giarratano et al. 2020).
    """
    octa500_samples = load_octa500()
    rose_samples    = load_rose() if use_rose else []

    # Splits OCTA-500 into 80/10/10
    octa500_train, octa500_temp = train_test_split(
        octa500_samples, test_size=0.2, random_state=RANDOM_STATE
    )
    octa500_valid, octa500_test = train_test_split(
        octa500_temp, test_size=0.5, random_state=RANDOM_STATE
    )

    # Splits ROSE into 80/20 (no test split)
    rose_train, rose_valid = [], []
    if rose_samples:
        rose_train, rose_valid = train_test_split(
            rose_samples, test_size=0.2, random_state=RANDOM_STATE
        )

    all_train = octa500_train + rose_train
    all_valid = octa500_valid + rose_valid
    all_test  = octa500_test # OCTA-500 only

    print(f'Final split sizes:')
    print(f'  Train: {len(all_train)} '
          f'(OCTA-500: {len(octa500_train)}, ROSE: {len(rose_train)})')
    print(f'  Valid: {len(all_valid)} '
          f'(OCTA-500: {len(octa500_valid)}, ROSE: {len(rose_valid)})')
    print(f'  Test:  {len(all_test)}')

    for split_name, data in [
        ('train', all_train),
        ('valid', all_valid),
        ('test',  all_test)
    ]:
        for img, mask, sid in tqdm(data, desc=f"Saving '{split_name}'"):
            save_sample(img, mask, sid, split_name)

    all_masks = [s[1] for s in octa500_samples + rose_samples if s[1] is not None]
    ratios = [m.mean() for m in all_masks]
    print(f'*COMPLETE: {(len(octa500_samples) + len(rose_samples))} samples processed.')
    print(f'  Train: {len(all_train)}, Valid: {len(all_valid)}, Test: {len(all_test)}')
    print(f'Vessel ratio: mean of {np.mean(ratios)*100:.1f}%, ' # rounded to 1 decimal
        f'range of {np.min(ratios)*100:.1f}–{np.max(ratios)*100:.1f}%')
    print(f'Class imbalance: {1/np.mean(ratios):.0f}:1' # rounded to nearest whole
        f'background:vessel pixel ratio (correctable with Dice loss)') 


if __name__ == '__main__':
    prepare_data(use_rose=True)