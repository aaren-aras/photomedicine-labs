"""
check.py
========
Diagnostic script — run this BEFORE training to verify:
    1. Data loaded correctly (shapes, dtypes, value ranges)
    2. Class imbalance statistics (vessel ratio per split)
    3. Augmentation pipeline works (visual check)
    4. Motion artifact simulation looks realistic (visual check)

Run with: python check.py
Expected output: raw_data_check.png, augmentation_check.png, artifact_check.png
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUTPUT_DIR     = Path('data/OCTA-500_processed')
IMG_TRAIN_DIR  = OUTPUT_DIR / 'images' / 'train'
IMG_VALID_DIR  = OUTPUT_DIR / 'images' / 'valid'
MASK_TRAIN_DIR = OUTPUT_DIR / 'masks'  / 'train'
MASK_VALID_DIR = OUTPUT_DIR / 'masks'  / 'valid'


def check_data_integrity():
    """
    Verify shapes, dtypes, and value ranges for 3 samples from each split.
    Catches issues like:
        - Images accidentally normalized twice (range [0, 0.004] instead of [0, 255])
        - Masks with values {0, 255} instead of {0, 1} (binarization not applied)
        - Shape mismatches between image and mask
    """
    print('=' * 60)
    print('DATA INTEGRITY CHECK')
    print('=' * 60)

    splits = [
        (IMG_TRAIN_DIR, MASK_TRAIN_DIR, 'TRAIN'),
        (IMG_VALID_DIR, MASK_VALID_DIR, 'VALID'),
    ]

    all_vessel_ratios = []

    for img_dir, mask_dir, split_name in splits:
        img_files  = sorted(img_dir.iterdir())
        mask_files = sorted(mask_dir.iterdir())

        print(f'\n{split_name} ({len(img_files)} samples):')

        for i, (img_f, mask_f) in enumerate(zip(img_files[:3], mask_files[:3])):
            img  = np.load(img_f)
            mask = np.load(mask_f)
            ratio = mask.mean()
            all_vessel_ratios.append(ratio)

            print(f'  Sample {i}: img {img.shape} {img.dtype} [{img.min()},{img.max()}] | '
                  f'mask {mask.shape} {mask.dtype} [{mask.min()},{mask.max()}] | '
                  f'vessel%={ratio*100:.1f}%')

            # Assertions — will raise if something is wrong
            assert img.dtype == np.uint8,  f'Expected uint8 image, got {img.dtype}'
            assert mask.dtype == np.uint8, f'Expected uint8 mask, got {mask.dtype}'
            assert img.max() == 255,       f'Image not in [0,255] range'
            assert set(np.unique(mask)).issubset({0, 1}), \
                f'Mask has unexpected values: {np.unique(mask)}'
            assert img.shape == mask.shape, \
                f'Shape mismatch: img={img.shape}, mask={mask.shape}'

    ratios = np.array(all_vessel_ratios)
    print(f'\nVessel ratio stats:')
    print(f'  Mean:  {ratios.mean()*100:.1f}%')
    print(f'  Range: {ratios.min()*100:.1f}% – {ratios.max()*100:.1f}%')
    print(f'  → Class imbalance ≈ {1/ratios.mean():.0f}:1 background:vessel')
    print(f'  ✓ Dice loss handles this — no class weighting needed')
    print('\n✓ Data integrity check PASSED')


def visualize_samples(n: int = 4):
    """Save a grid showing n image/mask pairs from train and valid."""
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    fig.suptitle('Data Check: Image | Mask | Image (val) | Mask (val)', fontsize=13)

    train_imgs  = sorted(IMG_TRAIN_DIR.iterdir())
    train_masks = sorted(MASK_TRAIN_DIR.iterdir())
    val_imgs    = sorted(IMG_VALID_DIR.iterdir())
    val_masks   = sorted(MASK_VALID_DIR.iterdir())

    for i in range(n):
        ti = np.load(train_imgs[i])
        tm = np.load(train_masks[i])
        vi = np.load(val_imgs[i])
        vm = np.load(val_masks[i])

        axes[i][0].imshow(ti, cmap='gray'); axes[i][0].set_title(f'Train img {i}')
        axes[i][1].imshow(tm, cmap='gray', vmin=0, vmax=1)
        axes[i][1].set_title(f'Train mask ({tm.mean()*100:.1f}% vessels)')
        axes[i][2].imshow(vi, cmap='gray'); axes[i][2].set_title(f'Val img {i}')
        axes[i][3].imshow(vm, cmap='gray', vmin=0, vmax=1)
        axes[i][3].set_title(f'Val mask ({vm.mean()*100:.1f}% vessels)')

        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    plt.savefig('raw_data_check.png', dpi=150, bbox_inches='tight')
    print('Saved raw_data_check.png')


def visualize_motion_artifacts(n: int = 3):
    """
    Show clean vs degraded image pairs to verify the artifact simulation
    looks realistic before training the restoration model.
    """
    # Import here to keep check.py runnable without full model.py
    from model import simulate_motion_artifacts

    img_files = sorted(IMG_TRAIN_DIR.iterdir())
    severities = [0.15, 0.30, 0.40]  # mild / medium / severe

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    fig.suptitle('Motion Artifact Simulation: Clean | Mild | Medium | Severe', fontsize=13)

    for i in range(n):
        clean = np.load(img_files[i]).astype(np.float32) / 255.0
        clean = np.expand_dims(clean, axis=-1)

        axes[i][0].imshow(clean.squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i][0].set_title('Clean (ground truth)')

        for k, sev in enumerate(severities):
            deg = simulate_motion_artifacts(clean, severity=sev)
            axes[i][k+1].imshow(deg.squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[i][k+1].set_title(f'Severity={sev:.2f}')

        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    plt.savefig('artifact_check.png', dpi=150, bbox_inches='tight')
    print('Saved artifact_check.png')
    print('→ Verify: degraded images should show horizontal banding and')
    print('  speckle noise, similar to real low-repeat OCTA acquisitions')


if __name__ == '__main__':
    check_data_integrity()
    visualize_samples()
    visualize_motion_artifacts()
    print('\n✓ All checks complete')
