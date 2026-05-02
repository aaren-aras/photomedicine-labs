# segmentation.py
# ===============
# Vessel segmentation inference — loads the saved .keras model and runs
# prediction on a single image. No Docker, no training dependencies needed.
#
# The .keras file is self-contained (weights + architecture). CPU inference
# on a single 400×400 image takes ~1 second. GPU is used automatically if
# available but is NOT required.
#
# Usage:
#   from segmentation import SegmentationModel
#   model = SegmentationModel()          # loads model once
#   mask, overlay = model.predict(img)   # img: HxW uint8 numpy array

import numpy as np
from pathlib import Path
from constants import SEG_MODEL_PATH, MODEL_INPUT_SIZE


class SegmentationModel:
    """
    Wrapper around the trained ResU-Net segmentation model.
    Handles model loading, preprocessing, inference, and overlay generation.

    Lazy-loads on first predict() call so GUI startup is fast even if
    TensorFlow takes a few seconds to initialise.
    """

    def __init__(self, model_path: Path = SEG_MODEL_PATH):
        self.model_path = model_path
        self._model = None   # loaded lazily

    def is_available(self) -> bool:
        """Check if the model file exists before attempting to load."""
        return self.model_path.exists()
    
    def _load(self):
        import os
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        
        import tensorflow as tf
        self._model = tf.keras.models.load_model(
            str(self.model_path),
            compile=False
        )

    def predict(self, img: np.ndarray,
                threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """
        Run vessel segmentation on a single OCTA image.

        Preprocessing pipeline (matches training exactly):
            1. Convert to grayscale if RGB/RGBA
            2. Resize to MODEL_INPUT_SIZE × MODEL_INPUT_SIZE (400×400)
            3. Normalise to float32 [0, 1]
            4. Add batch + channel dimensions → (1, 400, 400, 1)

        Postprocessing:
            1. Threshold probability map at 0.5 → binary mask
            2. Resize binary mask back to original image dimensions
            3. Build RGBA overlay for display

        Args:
            img:       HxW uint8 grayscale, or HxWx3/HxWx4 — any format
            threshold: vessel probability threshold (default 0.5)

        Returns:
            mask:    HxW uint8 {0,1} binary vessel mask (original resolution)
            overlay: HxWx4 uint8 RGBA array for semi-transparent display
                     Vessel pixels: cyan (#39d0d8) at 70% opacity
                     Background:    fully transparent
        """
        if self._model is None:
            self._load()

        import cv2

        orig_h, orig_w = img.shape[:2]

        # ── Preprocess ────────────────────────────────────────────────────────
        # Grayscale conversion
        if img.ndim == 3 and img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.squeeze()

        # Resize to model input size (400×400 — must match training)
        resized = cv2.resize(
            gray, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalise and add batch + channel dims
        inp = resized.astype(np.float32) / 255.0
        inp = inp[np.newaxis, :, :, np.newaxis]   # (1, 400, 400, 1)

        # ── Inference ─────────────────────────────────────────────────────────
        prob_map = self._model.predict(inp, verbose=0)[0, :, :, 0]
        # prob_map shape: (400, 400), values in [0, 1]

        # ── Postprocess ───────────────────────────────────────────────────────
        # Threshold → binary mask at model resolution
        mask_model = (prob_map > threshold).astype(np.uint8)

        # Resize back to original image dimensions
        mask = cv2.resize(
            mask_model, (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST   # nearest: preserves binary values
        )

        # ── RGBA overlay ──────────────────────────────────────────────────────
        # Cyan (#39d0d8) at alpha=180/255 ≈ 70% opacity on vessel pixels
        overlay = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
        overlay[mask == 1] = [57, 208, 216, 180]   # R, G, B, A

        return mask, overlay

    def predict_with_metrics(self, img: np.ndarray,
                             gt_mask: np.ndarray = None,
                             threshold: float = 0.5) -> dict:
        """
        Run prediction and optionally compute metrics against a ground truth.

        Args:
            img:      input image (any format, see predict())
            gt_mask:  optional HxW uint8 {0,1} ground truth mask
            threshold: probability threshold

        Returns:
            dict with keys:
                'mask':    binary prediction mask
                'overlay': RGBA overlay
                'vessel_ratio': fraction of pixels predicted as vessel
                'dice':    Dice coefficient (only if gt_mask provided)
                'iou':     IoU (only if gt_mask provided)
        """
        mask, overlay = self.predict(img, threshold)
        result = {
            'mask':         mask,
            'overlay':      overlay,
            'vessel_ratio': float(mask.mean()),
        }

        if gt_mask is not None:
            eps = 1e-8
            TP = np.sum(gt_mask * mask)
            FP = np.sum((1 - gt_mask) * mask)
            FN = np.sum(gt_mask * (1 - mask))
            result['dice'] = float((2 * TP + eps) / (2 * TP + FP + FN + eps))
            result['iou']  = float((TP + eps) / (TP + FP + FN + eps))

        return result
