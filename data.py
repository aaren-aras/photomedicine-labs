"""
Preprocesses OCTA-500 and ROSE datasets into .npy arrays for model training.

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
        images/{train,valid,test}/*.npy   ← float32 normalised [0,1], CLAHE-enhanced projections
        masks/{train,valid,test}/*.npy    ← float32 binary {0,1}

"""

from pathlib import Path
import json

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import (
    CLIP_LIMIT, TILE_GRID_SIZE, IMG_SIZE, PROJECTIONS, 
    OCTA500_LABEL, BINARIZE_MASK, RANDOM_STATE
)

ROOT_DIR = Path(__file__).resolve().parent
ROSE_DIR = ROOT_DIR / 'data' / 'ROSE'
OUTPUT_DIR = ROOT_DIR / 'data' / 'processed'

for split in ['train', 'valid', 'test']:
    (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'masks'  / split).mkdir(parents=True, exist_ok=True)


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
    """Loads and preprocesses image-mask pair."""
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
            f"*ERROR: OCTA-500 '{PROJECTIONS['octa500']}' projections missing at '{proj_dir}'... " 
            "Pipeline cannot proceed."
        )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"*ERROR: OCTA-500 '{OCTA500_LABEL}' labels missing at '{label_dir}'... "
            "Pipeline cannot proceed."
        )

    samples = []
    for proj_path in tqdm(sorted(proj_dir.iterdir()), desc='Processing OCTA-500 samples'):
        label_path = label_dir / proj_path.name
        img, mask = load_and_preprocess(proj_path, label_path)
        samples.append((img, mask, f'octa500_{proj_path.stem}'))

    print(f'*SUCCESS: {len(samples)} OCTA-500 samples loaded.')
    return samples


def load_rose() -> list[tuple]:
    """
    Loads all ROSE SVC_SVC projections and labels, and 
    returns a list of (img, mask, sample_id) tuples.
    """
    # Defines folder and file structure for each subset
    dataset_configs = {
        'ROSE-1': {
            'proj_subdir': PROJECTIONS['rose1'], 
            'img_subdir': 'img', 
            'label_ext': '.tif'
        },
        'ROSE-2': {
            'proj_subdir': '', 
            'img_subdir': 'original', 
            'label_ext': '.png'
        } 
    }

    # Dynamically pieces together paths
    sources = []
    for subset, config in dataset_configs.items():
        for split in ['train', 'test']:
            base_path = ROSE_DIR / subset / config['proj_subdir'] / split
            proj_dir = base_path / config['img_subdir']
            label_dir = base_path / 'gt'
            label_id = f"{subset.lower().replace('-', '')}_{split}" # e.g., rose1_train
            sources.append((proj_dir, label_id_dir, label_id, config['label_ext']))

    samples = []
    for proj_dir, label_dir, label_id, label_ext in sources:
        if not proj_dir.exists():
            print(f"*WARNING: ROSE projections missing at '{proj_dir}'... "
                f"   Skipping '{label_id}' split.")
            continue
        
        if not label_dir.exists():
            print(f"*WARNING: ROSE labels missing at '{label_dir}'... "
                f"   Skipping '{label_id}' split.")
            continue

        for proj_path in tqdm(sorted(proj_dir.iterdir()), desc=f"Processing '{label_id}'"):        
            if proj_path.name.startswith('.'): 
                    continue

            label_path = label_dir / f'{proj_path.stem}{label_ext}'
            
            if not label_path.exists():
                print(f'   --> Skip: No matching label file for {proj_path.name}')
                continue

            img, mask = load_and_preprocess(proj_path, label_path)
            samples.append((img, mask, f'{label_id}_{proj_path.stem}'))
        
    print(f'*SUCCESS: {len(samples)} ROSE samples loaded.')
    return samples


def save_sample(img: np.ndarray, mask: np.ndarray, sample_id: str, split: str) -> None:
    """
    Saves image-mask pairs as .npy files with metadata.

    Why .npy instead of .png?
    - np.load() is ~3-5× faster than PIL Image.open() for training loops
    - Preserves exact array values without JPEG compression artifacts
    - Easier to load with consistent dtypes (no conversion needed)

    Images saved as uint8 [0,255] — normalised to float32 [0,1] in generator.
    Masks saved as uint8 {0,1} — cast to float32 in generator (no /255 needed).
    """

    # Faster to load NPYs vs. PNGs or JPGs 
    np.save(OUTPUT_DIR / 'images' / split / f'{sample_id}.npy', img)
    np.save(OUTPUT_DIR / 'masks' / split / f'{sample_id}.npy', mask)
    
    # Save metadata for dataset analysis and debugging
    metadata = {
        'id': sample_id,
        'split': split,
        'image_shape': list(img.shape),
        'vessel_pixels': int(mask.sum()),
        'total_pixels': int(mask.size),
        # vessel_ratio tells us class imbalance — typically ~0.10-0.15 for retinal OCTA
        # A very low ratio (<0.05) means sparse vessels → model may predict all background
        'vessel_ratio': float(mask.mean()),
    }
    metadata_path = OUTPUT_DIR / 'metadata' / split / f'{sample_id}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def prepare_data(use_rose: bool = True) -> None:
    """
    Process OCTA-500 and ROSE projection maps (ILM–OPL slab) and labels for model training.

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
    
    # Load datasets
    octa500_samples = load_octa500()
    rose_samples    = load_rose() if use_rose else []

    if not octa500_samples:
        raise RuntimeError('OCTA-500 is required. ROSE is optional.')

    # Split OCTA-500: 80/10/10
    train_500, temp_500 = train_test_split(octa500_samples, test_size=0.2, random_state=RANDOM_STATE)
    valid_500, test_500 = train_test_split(temp_500, test_size=0.5, random_state=RANDOM_STATE)

    # Split ROSE: 80/20 (no test split)
    train_rose, valid_rose = [], []
    if rose_samples:
        train_rose, valid_rose = train_test_split(rose_samples, test_size=0.2, random_state=RANDOM_STATE)

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

    # splits = {'train': train_data, 'valid': valid_data, 'test': test_data}
    # for split_name, data in splits.items():
    #     for img, mask, id in tqdm(data, desc=f"Saving '{split_name}' split"):
    #         save_sample(img, mask, id, split_name)

    # Class imbalance summary
    all_masks   = [s[1] for s in octa500_samples + rose_samples if s[1] is not None]
    ratios      = [m.mean() for m in all_masks]
    print(f'\n*COMPLETE: {len(all_samples)} samples processed.')
    print(f'  Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}')
    print(f'\nVessel ratio — mean: {np.mean(ratios)*100:.1f}%, '
          f'range: {np.min(ratios)*100:.1f}–{np.max(ratios)*100:.1f}%')
    print(f'Class imbalance ≈ {1/np.mean(ratios):.0f}:1 — Dice loss handles this')
    print('\nDone. Run model.py to start training.')


if __name__ == '__main__':
    prepare_data(use_rose=True)
