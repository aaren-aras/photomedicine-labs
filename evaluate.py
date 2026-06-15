"""
Benchmarks trained pipeline using the 'test' split and makes 
metric tables, distribution plots, and visual comparisons.

Stage 1 — Restoration:
    - Takes clean test images, simulates degradation, runs through restoration model
    - Shows: Degraded Input | Restored Output | Clean Ground Truth | Difference Map
    - Metrics: PSNR, SSIM (targets: PSNR > 23 dB, SSIM > 0.55 per Liao et al.)

Stage 2 — Segmentation:
    - Runs segmentation model on clean test images
    - Shows: Input | Ground Truth | Prediction | TP/FP/FN Overlay
    - Metrics: Dice, IoU, Precision, Recall, Specificity, clDice

Usage:
    python evaluate.py                    # both stages
    python evaluate.py --stage 1          # restoration only
    python evaluate.py --stage 2          # segmentation only
    python evaluate.py --single <path>    # single image inference
"""
import importlib
import sys
from pathlib import Path
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg') # don't let Matplotlib open windows inside Docker 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf

from config import EPSILON, THRESHOLD

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / 'data' / 'processed'
MODELS_DIR = ROOT_DIR / 'benchmark'
RESULTS_DIR = ROOT_DIR / 'images'
RESULTS_DIR.mkdir(exist_ok=True)

# Imports stuff from 'model.py'
mod = importlib.import_module('model')
dice_coefficient = mod.dice_coefficient
DiceMetric = mod.DiceMetric
IoUMetric = mod.IoUMetric
cl_dice_loss = mod.cl_dice_loss
simulate_motion_artifacts = mod.simulate_motion_artifacts
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    # Allocate memory incrementally instead of all at once
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f'*GPUs FOUND: {gpus[0].name}')
else:
    print('*WARNING: No GPU found. Testing will begin on CPU...')

# ── Stage 1: Restoration ──

def compute_restoration_metrics(clean: np.ndarray, restored: np.ndarray) -> dict[str, float]:
    """Calculates PSNR and SSIM (Liao et al. targets: PSNR > 23 dB, SSIM > 0.55)."""

    """
    PSNR (Peak Signal-to-Noise Ratio, dB):
        10 × log₁₀(MAX² / MSE)
        Measures pixel-level fidelity. Higher=better.
        Typical good OCTA restoration: >23 dB (Liao et al. baseline: 24.2 dB)

    SSIM (Structural Similarity Index, [0,1]):
        Compares luminance, contrast, and structure between images.
        More perceptually meaningful than PSNR — a blurry image can have
        decent PSNR but low SSIM. Higher=better.
        Typical good OCTA restoration: >0.55 (Liao et al. baseline: 0.59)
     - perceptual quality to measure similarity
    """
    c = tf.constant(clean[np.newaxis].astype(np.float32))
    r = tf.constant(restored[np.newaxis].astype(np.float32))

    # Peak signal-to-noise ratio:
    psnr = float(tf.image.psnr(c, r, max_val=1.0).numpy()[0])
    
    # Structural similarity index:  
    ssim = float(tf.image.ssim(c, r, max_val=1.0).numpy()[0])
    return {'psnr': psnr, 'ssim': ssim}


def evaluate_restoration(n_visual: int = 6):
    """
    Evaluate restoration model on test set.

    For each test image:
        1. Load clean image
        2. Simulate degradation (same as training — shot noise, dropout, speckle)
        3. Run through restoration model
        4. Compute PSNR and SSIM vs clean ground truth

    Saves:
        results/restoration_results.png  — visual grid
        results/restoration_metrics.png  — PSNR/SSIM distributions
    """
    rest_path = MODELS_DIR / 'restoration_best.keras'
    if not rest_path.exists():
        print('WARNING: restoration_best.keras not found, skipping Stage 1')
        return

    print('\n=== Evaluating Stage 1: Restoration ===')

    # Load with compile=False — we don't need to recompile for inference
    # custom_objects not needed for restoration model (no custom metrics)
    rest_model = tf.keras.models.load_model(str(rest_path), compile=False)
    print('Restoration model loaded.')

    img_files = sorted(IMG_TEST.iterdir())
    all_metrics = []

    # Test at multiple severity levels to show model generalises
    severities = [0.25, 0.40, 0.70]  # mild, medium, severe (was 0.15, 0.25, 0.4)

    for img_f in img_files:
        clean = np.load(img_f).astype(np.float32) / 255.0
        if clean.ndim == 2:
            clean = np.expand_dims(clean, axis=-1)

        # Use medium severity (0.25) as the standard evaluation point
        degraded  = simulate_motion_artifacts(clean, severity=0.25)
        restored  = rest_model.predict(degraded[np.newaxis], verbose=0)[0]

        metrics = compute_restoration_metrics(clean, restored)
        metrics['file'] = img_f.name
        all_metrics.append((clean, degraded, restored, metrics))

    # Print summary
    psnr_vals = [m['psnr'] for _, _, _, m in all_metrics]
    ssim_vals = [m['ssim'] for _, _, _, m in all_metrics]
    print(f'\n{"="*50}')
    print('RESTORATION RESULTS (test set, severity=0.25)')
    print(f'{"="*50}')
    print(f'  PSNR: {np.mean(psnr_vals):.2f} ± {np.std(psnr_vals):.2f} dB')
    print(f'  SSIM: {np.mean(ssim_vals):.3f} ± {np.std(ssim_vals):.3f}')
    print(f'\nLiao et al. baseline (2-repeat input):')
    print(f'  PSNR: 15.70 dB  →  IRU-Net: 24.23 dB')
    print(f'  SSIM: 0.28      →  IRU-Net: 0.59')
    print(f'\nYour model:')
    print(f'  PSNR: {np.mean(psnr_vals):.2f} dB  (target: > 23 dB)')
    print(f'  SSIM: {np.mean(ssim_vals):.3f}      (target: > 0.55)')

    # Save visual results
    _save_restoration_visuals(all_metrics, severities, rest_model, n_visual)
    _save_restoration_metrics(all_metrics)


def _save_restoration_visuals(all_metrics, severities, model, n: int):
    """
    Grid showing degradation → restoration → ground truth for n test images.
    Also shows difference map (|restored - clean|) to highlight errors.
    Columns: Degraded | Restored | Clean GT | Difference
    """
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    fig.suptitle(
        'Stage 1: Restoration Results (severity=0.25)\n'
        'Col1: Degraded Input | Col2: Restored Output | '
        'Col3: Clean Ground Truth | Col4: Difference Map\n'
        'Difference map: brighter = larger error (amplified ×5 for visibility)',
        fontsize=10
    )

    for i, (clean, degraded, restored, metrics) in enumerate(all_metrics[:n]):
        diff = np.abs(restored - clean).squeeze() * 5  # amplify for visibility

        axes[i][0].imshow(degraded.squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i][0].set_title(f'Degraded\n{metrics["file"][:12]}')

        axes[i][1].imshow(restored.squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i][1].set_title(
            f'Restored\nPSNR={metrics["psnr"]:.1f}dB | SSIM={metrics["ssim"]:.3f}'
        )

        axes[i][2].imshow(clean.squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i][2].set_title('Clean Ground Truth')

        im = axes[i][3].imshow(diff, cmap='hot', vmin=0, vmax=1)
        axes[i][3].set_title('Difference (×5)')
        plt.colorbar(im, ax=axes[i][3], fraction=0.046, pad=0.04)

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    out = RESULTS_DIR / 'restoration_results.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')

    # Also save a severity comparison grid for the first image
    _save_severity_comparison(all_metrics[0][0], severities, model)


def _save_severity_comparison(clean: np.ndarray, severities: list, model):
    """
    Show restoration quality at different degradation severities for one image.
    Demonstrates model generalises across mild → severe degradation.
    """
    n_sev = len(severities)
    fig, axes = plt.subplots(n_sev, 4, figsize=(16, 4 * n_sev))
    fig.suptitle(
        'Restoration at Different Degradation Severities\n'
        'Severity 0.15=mild (few repeats) | 0.25=medium | 0.40=severe (2 repeats)',
        fontsize=11
    )

    for i, sev in enumerate(severities):
        degraded = simulate_motion_artifacts(clean, severity=sev)
        restored = model.predict(degraded[np.newaxis], verbose=0)[0]
        metrics  = compute_restoration_metrics(clean, restored)
        diff     = np.abs(restored - clean).squeeze() * 5

        axes[i][0].imshow(degraded.squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i][0].set_title(f'Degraded (severity={sev})')

        axes[i][1].imshow(restored.squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i][1].set_title(
            f'Restored\nPSNR={metrics["psnr"]:.1f}dB SSIM={metrics["ssim"]:.3f}'
        )

        axes[i][2].imshow(clean.squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i][2].set_title('Clean Ground Truth')

        axes[i][3].imshow(diff, cmap='hot', vmin=0, vmax=1)
        axes[i][3].set_title('Difference (×5)')

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    out = RESULTS_DIR / 'restoration_severity_comparison.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'*SAVED: {out}')


def _save_restoration_metrics(all_metrics):
    """PSNR and SSIM distribution plots across test set."""
    psnr_vals = [m['psnr'] for _, _, _, m in all_metrics]
    ssim_vals = [m['ssim'] for _, _, _, m in all_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Stage 1: Restoration Metric Distributions (Test Set)', fontsize=13)

    for ax, vals, name, target, unit in [
        (ax1, psnr_vals, 'PSNR', 23.0, 'dB'),
        (ax2, ssim_vals, 'SSIM', 0.55, ''),
    ]:
        ax.boxplot(vals, patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.7))
        ax.axhline(np.mean(vals), color='red', linestyle='--',
                   label=f'mean={np.mean(vals):.3f}{unit}')
        ax.axhline(target, color='green', linestyle=':',
                   label=f'Liao target={target}{unit}')
        ax.set_title(f'{name} ({unit})')
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = RESULTS_DIR / 'restoration_metrics.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


# ── Stage 2: Segmentation ──
def compute_segmentation_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict[str, float]:
    """Calculates Dice, IoU, Precision, Recall, Specificity, clDice."""
    y_true = y_true.squeeze().astype(np.float32) # ground truth (binary mask)
    y_pred = y_pred_prob.squeeze() # model's predictions (probability map)
    y_bin = (y_pred > THRESHOLD).astype(np.float32) # prob map -> binary choice

    tp = np.sum(y_true * y_bin)
    fp = np.sum((1 - y_true) * y_bin)
    fn = np.sum(y_true * (1 - y_bin))
    tn = np.sum((1 - y_true) * (1 - y_bin))

    # How well do predicted and ground truth vessels overlap?
    dice = (2 * tp + EPSILON) / (2 * tp + fp + fn + EPSILON)
    
    # How strictly do predicted and ground truth vessel boundaries intersect?
    iou = (tp + EPSILON) / (tp + fp + fn + EPSILON)
    
    # How many predicted vessels were true?
    precision = (tp + EPSILON) / (tp + fp + EPSILON)
    
    # How many true vessels were found?
    recall = (tp + EPSILON) / (tp + fn + EPSILON)
    
    # How many bkgd pixels were rejected?
    specificity = (tn + EPSILON) / (tn + fp + EPSILON)

    # How well do predicted vessels preserve structural connectivity?
    yt_tf = tf.constant(y_true[np.newaxis, :, :, np.newaxis]) # (H, W) -> (1, H, W)
    yp_tf = tf.constant(y_bin[np.newaxis,  :, :, np.newaxis]) # (1, H, W) -> (1, H, W, 1)
    cldice = float(1.0 - cl_dice_loss(yt_tf, yp_tf).numpy())

    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'cldice': cldice,
    }


def evaluate_segmentation(n_visual: int = 6):
    """
    Evaluate segmentation model on held-out test set.
    Saves visual grid and metric distribution plots.
    """
    seg_path = MODELS_DIR / 'segmentation_best.keras'
    if not seg_path.exists():
        print('WARNING: segmentation_best.keras not found, skipping Stage 2')
        return

    print('\n=== Evaluating Stage 2: Segmentation ===')

    # Load with custom objects so Keras knows how to deserialize them
    custom_objects = {'DiceMetric': DiceMetric, 'IoUMetric': IoUMetric}
    seg_model = tf.keras.models.load_model(
        str(seg_path), custom_objects=custom_objects, compile=False
    )
    print('Segmentation model loaded.')

    img_files  = sorted(IMG_TEST.iterdir())
    mask_files = sorted(MASK_TEST.iterdir())
    all_metrics = []

    for img_f, mask_f in zip(img_files, mask_files):
        img  = np.load(img_f).astype(np.float32) / 255.0
        mask = np.load(mask_f).astype(np.float32)

        pred = seg_model.predict(
            img[np.newaxis, :, :, np.newaxis], verbose=0
        )[0, :, :, 0]

        metrics = compute_segmentation_metrics(mask, pred)
        metrics['file'] = img_f.name
        all_metrics.append((img, mask, pred, metrics))

    # Print summary
    keys = ['dice', 'iou', 'precision', 'recall', 'specificity', 'cldice']
    print(f'\n{"="*50}')
    print('SEGMENTATION RESULTS (test set)')
    print(f'{"="*50}')
    for k in keys:
        vals = [m[k] for _, _, _, m in all_metrics]
        print(f'  {k:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')
    print(f'\nGiarratano 2020 benchmark: Dice ≈ 0.80–0.85')

    print(f'\nReference (OCTA-500 state of art, Giarratano 2020):')
    print(f'  Dice ≈ 0.80-0.85 for large vessel segmentation')
    print(f'  If your Dice > 0.75, the model is working well.')
    print(f'  If Dice < 0.60, check: mask binarization, loss function, learning rate.')

    _save_segmentation_visuals(all_metrics, n_visual)
    _save_segmentation_metrics(all_metrics, keys)


def _save_segmentation_visuals(all_metrics, n: int):
    """Grid: Input | Ground Truth | Prediction probability | TP/FP/FN overlay."""
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    fig.suptitle(
        'Segmentation Results\n'
        'Col1: Input | Col2: Ground Truth | Col3: Prediction | Col4: Overlay\n'
        'Overlay: Green=TP (correct vessel), Red=FP (spurious vessel), '
        'Blue=FN (missed vessel)',
        fontsize=10
    )

    for i, (img, mask, pred, metrics) in enumerate(all_metrics[:n]):
        pred_bin = (pred > 0.5).astype(np.float32)
        # Colour overlay: TP=green, FP=red (false alarm), FN=blue (missed vessel)
        overlay  = np.zeros((*img.shape, 3))
        overlay[:, :, 1] = mask * pred_bin         # Green: TP
        overlay[:, :, 0] = (1 - mask) * pred_bin   # Red:   FP
        overlay[:, :, 2] = mask * (1 - pred_bin)   # Blue:  FN

        axes[i][0].imshow(img, cmap='gray')
        axes[i][0].set_title(f'Input\n{metrics["file"][:12]}')

        axes[i][1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[i][1].set_title(
            f'Ground Truth\n({mask.mean()*100:.1f}% vessels)'
        )

        axes[i][2].imshow(pred, cmap='hot', vmin=0, vmax=1)
        axes[i][2].set_title(f'Prediction\nDice={metrics["dice"]:.3f}')

        axes[i][3].imshow(overlay)
        axes[i][3].set_title(
            f'IoU={metrics["iou"]:.3f} | clDice={metrics["cldice"]:.3f}'
        )

        for ax in axes[i]:
            ax.axis('off')

    legend = [
        mpatches.Patch(color='green', label='True Positive (correct vessel)'),
        mpatches.Patch(color='red',   label='False Positive (spurious vessel)'),
        mpatches.Patch(color='blue',  label='False Negative (missed vessel)'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    out = RESULTS_DIR / 'segmentation_results.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'*SAVED: {out}')


def _save_segmentation_metrics(all_metrics, keys):
    """Box plots for all segmentation metrics across test set."""
    fig, axes = plt.subplots(1, len(keys), figsize=(3 * len(keys), 5))
    fig.suptitle('Test Set Metric Distributions', fontsize=13)

    for ax, k in zip(axes, keys):
        vals = [m[k] for _, _, _, m in all_metrics]
        ax.boxplot(vals, patch_artist=True,
            boxprops=dict(facecolor='steelblue', alpha=0.7))
        ax.set_title(k)
        ax.set_ylim(0, 1)
        ax.axhline(np.mean(vals), color='red', linestyle='--',
            label=f'mean={np.mean(vals):.3f}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / 'metric_distributions.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'*SAVED: {out}')


def run_pipeline(img_path: str):
    """
    Run the complete two-stage pipeline on a single image.
    Useful for running on real degraded OCTA data.

    Input:  any .npy or image file (degraded OCTA)
    Output: results/pipeline_<filename>.png
            showing: Degraded → Restored → Vessel Mask
    """
    from PIL import Image as PILImage

    path = Path(img_path)
    if path.suffix == '.npy':
        img = np.load(path).astype(np.float32) / 255.0
    else:
        img = np.array(PILImage.open(path).convert('L')).astype(np.float32) / 255.0

    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    # Stage 1: restore
    rest_model_path = MODELS_DIR / 'restoration_best.keras'
    if rest_model_path.exists():
        rest_model = tf.keras.models.load_model(str(rest_model_path), compile=False)
        restored = rest_model.predict(img[np.newaxis], verbose=0)[0]
        rest_metrics = compute_restoration_metrics(img, restored)
        print(f'Stage 1 done — PSNR={rest_metrics["psnr"]:.1f}dB, '
              f'SSIM={rest_metrics["ssim"]:.3f}')
    else:
        restored = img
        print('No restoration model found — skipping Stage 1')

    # Stage 2: segment
    custom_objects = {'DiceMetric': DiceMetric, 'IoUMetric': IoUMetric}
    seg_model = tf.keras.models.load_model(
        str(MODELS_DIR / 'segmentation_best.keras'),
        custom_objects=custom_objects, compile=False
    )
    pred = seg_model.predict(restored[np.newaxis], verbose=0)[0, :, :, 0]
    pred_bin = (pred > 0.5).astype(np.float32)
    print(f'Stage 2 done — vessel pixels: '
          f'{pred_bin.sum():.0f} ({pred_bin.mean()*100:.1f}%)')

    # Save
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Full Pipeline: {path.name}', fontsize=12)

    axes[0].imshow(img.squeeze(), cmap='gray'); axes[0].set_title('Input (Degraded)')
    axes[1].imshow(restored.squeeze(), cmap='gray'); axes[1].set_title('Restored')
    axes[2].imshow(pred, cmap='hot'); axes[2].set_title('Vessel Probability')
    axes[3].imshow(pred_bin, cmap='gray'); axes[3].set_title('Vessel Mask (0.5 threshold)')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    out = RESULTS_DIR / f'pipeline_{path.stem}.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',  type=int, default=0,
                        help='1=restoration only, 2=segmentation only, 0=both')
    parser.add_argument('--single', type=str, default=None,
                        help='Path to single image for full pipeline inference')
    args = parser.parse_args()

    if args.single:
        run_pipeline(args.single)
    elif args.stage == 1:
        evaluate_restoration()
    elif args.stage == 2:
        evaluate_segmentation()
    else:
        evaluate_restoration()
        evaluate_segmentation()