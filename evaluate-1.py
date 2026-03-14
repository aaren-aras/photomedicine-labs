"""
evaluate.py
===========
Full evaluation pipeline — runs AFTER training.

Produces:
    1. Quantitative metrics table (Dice, IoU, clDice, PSNR, SSIM) on test set
    2. Visual comparison: Input | Restored | Ground Truth | Prediction | Overlay
    3. Per-sample metric distribution plots
    4. Single-image inference demo

Usage:
    python evaluate.py                    # full test set evaluation
    python evaluate.py --single <path>    # inference on one image
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from tensorflow.keras.models import load_model

from model import (
    dice_coefficient, DiceMetric, IoUMetric,
    simulate_motion_artifacts, cl_dice_loss
)

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR     = Path('data/OCTA-500_processed')
IMG_TEST_DIR   = OUTPUT_DIR / 'images' / 'test'
MASK_TEST_DIR  = OUTPUT_DIR / 'masks'  / 'test'
MODELS_DIR     = Path('models')
RESULTS_DIR    = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)


# ── Metric Functions ───────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """
    Compute the full set of segmentation metrics for one image.

    Metrics explained:
        Dice (F1):    2×TP / (2×TP + FP + FN)
                      Harmonic mean of precision and recall. The primary metric
                      for medical image segmentation. Range [0,1], higher=better.

        IoU (Jaccard): TP / (TP + FP + FN)
                      Stricter than Dice (denominator is larger).
                      Same ordering as Dice but lower absolute values.
                      Range [0,1], higher=better.

        Precision:    TP / (TP + FP)
                      Of all predicted vessel pixels, how many are actually vessels?
                      Low precision = over-segmentation (predicting too much).

        Recall:       TP / (TP + FN)
                      Of all true vessel pixels, how many did we find?
                      Low recall = under-segmentation (missing vessels).

        clDice:       Centerline Dice — topology-preserving metric.
                      Penalises broken vessel centerlines more than pixel errors.
                      Critical for OCTA: a disconnected vessel map is clinically
                      misleading even if Dice looks reasonable.

        Specificity:  TN / (TN + FP)
                      Of all background pixels, how many did we correctly reject?
                      Usually very high in vessel segmentation (>0.95) because
                      background dominates and is easy to classify.

    Args:
        y_true:      binary mask {0,1}, shape (H,W) or (H,W,1)
        y_pred_prob: probability map [0,1], same shape
        threshold:   classification threshold (0.5 is standard)

    Returns:
        dict of metric name → float value
    """
    y_true = y_true.squeeze().astype(np.float32)
    y_pred = y_pred_prob.squeeze()
    y_bin  = (y_pred > threshold).astype(np.float32)

    # Confusion matrix components
    TP = np.sum(y_true * y_bin)
    FP = np.sum((1 - y_true) * y_bin)
    FN = np.sum(y_true * (1 - y_bin))
    TN = np.sum((1 - y_true) * (1 - y_bin))

    eps = 1e-8  # prevent division by zero

    dice      = (2 * TP + eps) / (2 * TP + FP + FN + eps)
    iou       = (TP + eps) / (TP + FP + FN + eps)
    precision = (TP + eps) / (TP + FP + eps)
    recall    = (TP + eps) / (TP + FN + eps)
    specificity = (TN + eps) / (TN + FP + eps)

    # clDice via TF (uses soft skeleton approximation)
    yt_tf = tf.constant(y_true[np.newaxis, :, :, np.newaxis])
    yp_tf = tf.constant(y_bin[np.newaxis,  :, :, np.newaxis])
    cldice_val = float(1.0 - cl_dice_loss(yt_tf, yp_tf).numpy())

    return {
        'dice':        float(dice),
        'iou':         float(iou),
        'precision':   float(precision),
        'recall':      float(recall),
        'specificity': float(specificity),
        'cldice':      cldice_val,
        'vessel_ratio_true': float(y_true.mean()),
        'vessel_ratio_pred': float(y_bin.mean()),
    }


def compute_restoration_metrics(clean: np.ndarray, restored: np.ndarray) -> dict:
    """
    Compute image quality metrics for restoration model evaluation.

    PSNR (Peak Signal-to-Noise Ratio, dB):
        10 × log₁₀(MAX² / MSE)
        Measures pixel-level fidelity. Higher=better.
        Typical good OCTA restoration: >23 dB (Liao et al. baseline: 24.2 dB)

    SSIM (Structural Similarity Index, [0,1]):
        Compares luminance, contrast, and structure between images.
        More perceptually meaningful than PSNR — a blurry image can have
        decent PSNR but low SSIM. Higher=better.
        Typical good OCTA restoration: >0.55 (Liao et al. baseline: 0.59)
    """
    c = tf.constant(clean[np.newaxis].astype(np.float32))
    r = tf.constant(restored[np.newaxis].astype(np.float32))
    psnr = float(tf.image.psnr(c, r, max_val=1.0).numpy()[0])
    ssim = float(tf.image.ssim(c, r, max_val=1.0).numpy()[0])
    return {'psnr': psnr, 'ssim': ssim}


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_segmentation_model(n_visual: int = 6):
    """
    Run segmentation model on entire test set, compute metrics, save visuals.
    """
    # Load with custom objects so Keras knows how to deserialize them
    seg_model = load_model(
        MODELS_DIR / 'segmentation_best.keras',
        custom_objects={'DiceMetric': DiceMetric, 'IoUMetric': IoUMetric},
        compile=False
    )
    print('Segmentation model loaded.')

    img_files  = sorted(IMG_TEST_DIR.iterdir())
    mask_files = sorted(MASK_TEST_DIR.iterdir())

    all_metrics = []

    for img_f, mask_f in zip(img_files, mask_files):
        img  = np.load(img_f).astype(np.float32) / 255.0
        mask = np.load(mask_f).astype(np.float32)

        pred = seg_model.predict(
            img[np.newaxis, :, :, np.newaxis], verbose=0
        )[0, :, :, 0]

        metrics = compute_all_metrics(mask, pred)
        metrics['file'] = img_f.name
        all_metrics.append(metrics)

    # Aggregate
    keys = ['dice', 'iou', 'precision', 'recall', 'specificity', 'cldice']
    print('\n' + '=' * 60)
    print('SEGMENTATION RESULTS (test set)')
    print('=' * 60)
    for k in keys:
        vals = [m[k] for m in all_metrics]
        print(f'  {k:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')

    print(f'\nReference (OCTA-500 state of art, Giarratano 2020):')
    print(f'  Dice ≈ 0.80-0.85 for large vessel segmentation')
    print(f'  If your Dice > 0.75, the model is working well.')
    print(f'  If Dice < 0.60, check: mask binarization, loss function, learning rate.')

    # Save visual results
    _save_segmentation_visuals(seg_model, img_files, mask_files, n_visual)

    # Save metric distribution plots
    _save_metric_distributions(all_metrics, keys)

    return all_metrics


def _save_segmentation_visuals(model, img_files, mask_files, n: int):
    """Save grid: Input | Ground Truth | Prediction | Overlay (TP/FP/FN)."""
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    fig.suptitle(
        'Segmentation Results\n'
        'Col1: Input | Col2: Ground Truth | Col3: Prediction | Col4: Overlay\n'
        'Overlay: Green=TP, Red=FP, Blue=FN',
        fontsize=11
    )

    for i in range(n):
        img  = np.load(img_files[i]).astype(np.float32) / 255.0
        mask = np.load(mask_files[i]).astype(np.float32)
        pred_prob = model.predict(img[np.newaxis, :, :, np.newaxis], verbose=0)[0, :, :, 0]
        pred_bin  = (pred_prob > 0.5).astype(np.float32)

        metrics = compute_all_metrics(mask, pred_prob)

        # Colour overlay: TP=green, FP=red (false alarm), FN=blue (missed vessel)
        overlay = np.zeros((*img.shape, 3))
        overlay[:, :, 1] = mask * pred_bin        # Green: TP
        overlay[:, :, 0] = (1-mask) * pred_bin    # Red: FP
        overlay[:, :, 2] = mask * (1-pred_bin)    # Blue: FN

        axes[i][0].imshow(img, cmap='gray')
        axes[i][0].set_title(f'Input\n{img_files[i].name[:10]}')
        axes[i][1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[i][1].set_title(f'Ground Truth\n({mask.mean()*100:.1f}% vessels)')
        axes[i][2].imshow(pred_prob, cmap='hot', vmin=0, vmax=1)
        axes[i][2].set_title(f'Prediction\nDice={metrics["dice"]:.3f}')
        axes[i][3].imshow(overlay)
        axes[i][3].set_title(f'IoU={metrics["iou"]:.3f} | clDice={metrics["cldice"]:.3f}')

        for ax in axes[i]: ax.axis('off')

    # Legend for overlay
    legend = [
        mpatches.Patch(color='green', label='True Positive (correct vessel)'),
        mpatches.Patch(color='red',   label='False Positive (spurious vessel)'),
        mpatches.Patch(color='blue',  label='False Negative (missed vessel)'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(RESULTS_DIR / 'segmentation_results.png', dpi=150, bbox_inches='tight')
    print(f'Saved {RESULTS_DIR}/segmentation_results.png')


def _save_metric_distributions(all_metrics: list, keys: list):
    """Box plots showing per-sample metric distributions across test set."""
    fig, axes = plt.subplots(1, len(keys), figsize=(3 * len(keys), 5))
    fig.suptitle('Test Set Metric Distributions', fontsize=13)

    for ax, k in zip(axes, keys):
        vals = [m[k] for m in all_metrics]
        ax.boxplot(vals, patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.7))
        ax.set_title(k)
        ax.set_ylim(0, 1)
        ax.axhline(np.mean(vals), color='red', linestyle='--',
                   label=f'mean={np.mean(vals):.3f}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'metric_distributions.png', dpi=150, bbox_inches='tight')
    print(f'Saved {RESULTS_DIR}/metric_distributions.png')


def run_single_inference(img_path: str):
    """
    Run the full pipeline (restore → segment) on a single image.
    Works on any .npy or .png OCTA image, including Andrei's chicken embryo data.
    """
    from PIL import Image

    # Load image
    path = Path(img_path)
    if path.suffix == '.npy':
        img = np.load(path).astype(np.float32) / 255.0
    else:
        img = np.array(Image.open(path).convert('L')).astype(np.float32) / 255.0

    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    # Load models
    custom_objects = {'DiceMetric': DiceMetric, 'IoUMetric': IoUMetric}
    seg_model = load_model(MODELS_DIR / 'segmentation_best.keras',
                           custom_objects=custom_objects, compile=False)

    # Stage 1: restore (if restoration model exists)
    rest_path = MODELS_DIR / 'restoration_best.keras'
    if rest_path.exists():
        rest_model = load_model(rest_path, compile=False)
        restored = rest_model.predict(img[np.newaxis], verbose=0)[0]
        print('✓ Stage 1: image restored')
    else:
        restored = img
        print('⚠ No restoration model found — skipping Stage 1')

    # Stage 2: segment
    pred_prob = seg_model.predict(restored[np.newaxis], verbose=0)[0, :, :, 0]
    pred_bin  = (pred_prob > 0.5).astype(np.uint8)
    print(f'✓ Stage 2: segmentation complete')
    print(f'  Predicted vessel pixels: {pred_bin.sum()} ({pred_bin.mean()*100:.1f}%)')

    # Save output
    out_path = RESULTS_DIR / f'{path.stem}_segmentation.png'
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img.squeeze(),      cmap='gray'); axes[0].set_title('Input')
    axes[1].imshow(restored.squeeze(), cmap='gray'); axes[1].set_title('Restored')
    axes[2].imshow(pred_prob,          cmap='hot');  axes[2].set_title('Vessel Probability')
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'✓ Saved {out_path}')
    return pred_bin


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', type=str, default=None,
                        help='Path to single image for inference')
    args = parser.parse_args()

    if args.single:
        run_single_inference(args.single)
    else:
        evaluate_segmentation_model()
