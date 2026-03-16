"""
model.py
========
Two-stage OCTA vessel analysis pipeline (Option C):

    Stage 1 — Restoration:   Degraded OCTA  →  Clean OCTA
    Stage 2 — Segmentation:  Clean OCTA     →  Vessel mask

Why two stages instead of one?
    Training a segmentation model directly on degraded images forces it to
    simultaneously learn (a) what an artifact looks like and (b) where vessels
    are — two very different tasks. Separating them allows each model to
    specialize, and crucially, Stage 2 can be trained on clean OCTA-500 images
    while Stage 1 handles the real-world degradation from Andrei's in-lab setup.

    This matches the IRU-Net paper approach (Liao et al. 2023): restore first,
    then run downstream analysis on the restored image.

Architectures:
    Both stages use a ResU-Net (Residual U-Net):
    - U-Net: encoder-decoder with skip connections (Ronneberger et al. 2015)
    - Residual blocks: allow gradient flow through deep networks without vanishing
      (He et al. 2016 — same idea as ResNet)
    The skip connections are especially important for vessels: they pass fine
    spatial detail (thin vessel edges) directly from encoder to decoder,
    bypassing the lossy bottleneck compression.
"""

from pathlib import Path
from typing import Generator
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l2

from config import (
    IMG_SIZE, MODALITIES, NUM_CLASSES, FILTERS, KERNEL_SIZE, SCALE_FACTOR,
    DROPOUT_RATE, LEARNING_RATE, EPSILON, BATCH_SIZE, BCE_WEIGHT,
    AUG_CONFIG, MOTION_AUG, EPOCHS
)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (SCRIPT_DIR / 'data/OCTA-500_processed').resolve()
MODELS_DIR = SCRIPT_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# ── GPU Setup ─────────────────────────────────────────────────────────────────
physical_devices = tf.config.list_physical_devices('GPU')
print(f'*AVAILABLE GPUs: {physical_devices}')
if physical_devices:
    # Incremental memory allocation prevents TF from claiming all VRAM upfront,
    # leaving headroom for the VGG19 feature extractor in Stage 1 loss
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED: RESIDUAL BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

def residual_block(x: tf.Tensor, filters: int, kernel_size: int = KERNEL_SIZE) -> tf.Tensor:
    """
    Residual block: two conv layers with a skip connection around them.

    Why residual connections?
        In deep networks, gradients shrink as they're backpropagated through
        many layers (vanishing gradient problem). The skip connection provides
        a direct gradient path: ∂loss/∂input flows through BOTH the conv
        path AND the shortcut, preventing vanishing.

        Mathematically: output = F(x) + x
        The network only needs to learn the *residual* F(x) = output - x,
        which is easier than learning the full mapping from scratch.

    Why BatchNorm before ReLU (not after)?
        "Pre-activation" ordering (BN → ReLU → Conv) improves gradient flow
        and training stability compared to the original "post-activation" order
        (Conv → BN → ReLU). Both work; pre-activation is slightly preferred
        for deep networks.

    Args:
        x:           input tensor
        filters:     number of output feature maps
        kernel_size: convolution kernel size (3×3 by default)

    Returns:
        output tensor with same spatial dimensions as input
    """
    shortcut = x

    # First conv + normalize + activate
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=l2(1e-4), activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second conv + normalize (activation applied AFTER adding shortcut)
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=l2(1e-4), activation=None)(x)
    x = layers.BatchNormalization()(x)

    # Projection shortcut: if filter counts differ, use 1×1 conv to match dims
    # before adding. 1×1 conv changes channel depth without touching spatial dims.
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, padding='same')(shortcut)

    # Add shortcut BEFORE final ReLU — this is the key residual connection
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: RESTORATION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def build_restoration_model(
    input_shape: tuple = (*IMG_SIZE, 1)
) -> Model:
    """
    ResU-Net for OCTA image restoration.
    Input:  degraded OCTA image (simulated motion artifacts, sparse B-scans)
    Output: restored clean OCTA image

    Loss: L2 + 0.01 × VGG19 content loss
        - L2 (MSE): stabilizes training, suppresses noise
        - Content loss: ensures restored images are perceptually similar to
          clean ground truth by comparing deep VGG19 features, not just pixels.
          Pure L2 produces blurry outputs; content loss preserves fine vessel
          texture and edge sharpness.
        - Weight 0.01 on content loss found optimal by Liao et al. (2023)
          through ablation study across weights {0.001, 0.01, 0.1, 1.0}

    Metrics: PSNR and SSIM
        - PSNR (Peak Signal-to-Noise Ratio, dB): pixel-level fidelity
        - SSIM (Structural Similarity Index): perceptual quality, 0→1
        Both are standard in image restoration literature (Liao et al.,
        Das et al.) and directly comparable to their reported results.
    """
    from tensorflow.keras.applications import VGG19

    # Build VGG19 feature extractor once — frozen, not trained
    # Using 'block5_conv4' output layer — found optimal for OCTA in Liao et al.
    # Deep features capture structural/textural similarity better than pixel MSE
    vgg_base = VGG19(include_top=False, weights='imagenet',
                     input_shape=(*IMG_SIZE, 3))
    vgg_base.trainable = False
    vgg_extractor = Model(
        inputs=vgg_base.input,
        outputs=vgg_base.get_layer('block5_conv4').output,
        name='vgg_feature_extractor'
    )

    def restoration_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """L2 + 0.01 * VGG19 content loss (Liao et al. 2023)."""
        l2_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        # Tile grayscale (H,W,1) → (H,W,3) for VGG input; scale to [0,255]
        y_true_rgb = tf.repeat(y_true, 3, axis=-1) * 255.0
        y_pred_rgb = tf.repeat(y_pred, 3, axis=-1) * 255.0
        content = tf.reduce_mean(tf.square(
            vgg_extractor(y_true_rgb, training=False) -
            vgg_extractor(y_pred_rgb, training=False)
        ))
        return l2_loss + 0.01 * content

    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    c1 = residual_block(layers.Conv2D(FILTERS[0], KERNEL_SIZE, padding='same')(inputs), FILTERS[0])
    p1 = layers.MaxPooling2D(SCALE_FACTOR)(c1)   # 400→200
    c2 = residual_block(layers.Conv2D(FILTERS[1], KERNEL_SIZE, padding='same')(p1), FILTERS[1])
    p2 = layers.MaxPooling2D(SCALE_FACTOR)(c2)   # 200→100
    c3 = residual_block(layers.Conv2D(FILTERS[2], KERNEL_SIZE, padding='same')(p2), FILTERS[2])
    p3 = layers.MaxPooling2D(SCALE_FACTOR)(c3)   # 100→50

    # Bottleneck
    b = residual_block(layers.Conv2D(FILTERS[3], KERNEL_SIZE, padding='same')(p3), FILTERS[3])
    b = layers.Dropout(DROPOUT_RATE)(b)

    # Decoder — skip connections concatenate encoder feature maps
    u3 = layers.UpSampling2D(SCALE_FACTOR)(b)    # 50→100
    u3 = layers.Dropout(0.2)(u3)
    c4 = residual_block(layers.Concatenate()([u3, c3]), FILTERS[2])
    u2 = layers.UpSampling2D(SCALE_FACTOR)(c4)   # 100→200
    u2 = layers.Dropout(0.2)(u2)
    c5 = residual_block(layers.Concatenate()([u2, c2]), FILTERS[1])
    u1 = layers.UpSampling2D(SCALE_FACTOR)(c5)   # 200→400
    u1 = layers.Dropout(0.2)(u1)
    c6 = residual_block(layers.Concatenate()([u1, c1]), FILTERS[0])

    # Output: 1-channel sigmoid → pixel intensities in [0,1]
    output = layers.Conv2D(1, kernel_size=1, activation='sigmoid',
                           name='restoration')(c6)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, epsilon=EPSILON),
        loss=restoration_loss,
        metrics=[
            tf.keras.metrics.MeanSquaredError(name='mse'),
        ]
    )
    return model, vgg_extractor  # return extractor so it can be reused


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: SEGMENTATION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """
    Dice coefficient = 2 × |intersection| / (|A| + |B|)
    Ranges from 0 (no overlap) to 1 (perfect match).

    Why Dice and not accuracy?
        Vessel pixels are ~10-15% of total pixels. A model predicting ALL
        background achieves ~87% accuracy but Dice = 0. Dice penalises
        this by requiring the model to correctly identify both vessel AND
        background pixels proportionally.

    The smooth=1.0 term prevents division by zero when both masks are empty
    (i.e., an image with no vessels) and slightly stabilises gradients
    in early training when predictions are near-zero everywhere.
    """
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """1 - Dice. Minimizing this maximizes spatial overlap with ground truth."""
    return 1.0 - dice_coefficient(y_true, y_pred)


def cl_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """
    centerline Dice (clDice) — specifically penalises broken vessel centerlines.

    Standard Dice treats all vessel pixels equally. clDice additionally penalises
    topological errors: a thin break in a vessel centerline costs as much as
    losing many pixels on a wide vessel. This is critical for OCTA because
    clinically important information (FAZ boundary, capillary dropout) depends
    on vessel connectivity, not just pixel count.

    Approximation using max pooling as a differentiable skeletonization proxy.
    Full morphological skeletonization is not differentiable, so we use a
    multi-scale max pooling approach to approximate the centerline response.

    Reference: Shit et al., "clDice - a Novel Topology-Preserving Loss Function
    for Tubular Structure Segmentation", CVPR 2021.
    """
    def soft_skeleton(x: tf.Tensor, iters: int = 5) -> tf.Tensor:
        """Approximate skeleton via iterated morphological erosion proxy."""
        s = x
        for _ in range(iters):
            # Max pool then upsample approximates morphological dilation
            dilated = tf.nn.max_pool2d(s, ksize=3, strides=1, padding='SAME')
            # Subtract to keep only centerline-like pixels
            s = tf.maximum(s - dilated + s, 0)
        return s

    skel_pred = soft_skeleton(y_pred)
    skel_true = soft_skeleton(tf.cast(y_true, tf.float32))

    # Topology precision: how much of predicted skeleton overlaps true skeleton
    tprec = (tf.reduce_sum(skel_pred * tf.cast(y_true, tf.float32)) + smooth) / \
            (tf.reduce_sum(skel_pred) + smooth)
    # Topology sensitivity: how much of true skeleton is captured
    tsens = (tf.reduce_sum(skel_true * y_pred) + smooth) / \
            (tf.reduce_sum(skel_true) + smooth)

    cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens + 1e-8)
    return cl_dice


class DiceMetric(tf.keras.metrics.Metric):
    """Tracks average Dice coefficient across batches during training."""
    def __init__(self, name='dice', **kwargs):
        super().__init__(name=name, **kwargs)
        self.dice_sum = self.add_weight(name='dice_sum', initializer='zeros')
        self.count    = self.add_weight(name='count',    initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Threshold at 0.5 to get binary prediction before computing Dice
        # (Dice on soft probabilities is not meaningful as a metric)
        y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
        self.dice_sum.assign_add(dice_coefficient(y_true, y_pred_bin))
        self.count.assign_add(1.0)

    def result(self):
        return self.dice_sum / self.count

    def reset_states(self):
        self.dice_sum.assign(0.)
        self.count.assign(0.)


class IoUMetric(tf.keras.metrics.Metric):
    """
    Intersection over Union (Jaccard Index) — another standard segmentation metric.
    IoU = |intersection| / |union| = Dice / (2 - Dice)
    Slightly stricter than Dice; both should be reported for completeness.
    """
    def __init__(self, name='iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iou_sum = self.add_weight(name='iou_sum', initializer='zeros')
        self.count   = self.add_weight(name='count',   initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
        y_true_f   = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred_f   = tf.reshape(y_pred_bin, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union        = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
        iou = (intersection + 1.0) / (union + 1.0)
        self.iou_sum.assign_add(iou)
        self.count.assign_add(1.0)

    def result(self):
        return self.iou_sum / self.count

    def reset_states(self):
        self.iou_sum.assign(0.)
        self.count.assign(0.)


def build_segmentation_model(
    input_shape: tuple = (*IMG_SIZE, len(MODALITIES)),
    num_classes: int = NUM_CLASSES
) -> Model:
    """
    ResU-Net for binary OCTA vessel segmentation.
    Input:  clean OCTA projection [0,1] (after Stage 1 restoration)
    Output: vessel probability map [0,1], thresholded at 0.5 for binary mask

    Loss: Dice + BCE_WEIGHT × BinaryCrossentropy
        - Dice loss: handles class imbalance (vessels ≈ 10-15% of pixels),
          directly optimizes the metric we care about
        - BinaryCrossentropy (BCE): pixel-wise probability calibration,
          provides well-defined gradients even when Dice is near 0 or 1
        - BCE_WEIGHT=0.3: lets Dice dominate while BCE prevents overconfident
          predictions and provides gradient signal at flat regions
        Previous code used weight=0.5 which over-penalised individual pixel
        errors and produced conservative, blurry vessel boundaries.

    Metrics: Dice, IoU, and clDice (centerline Dice)
        - Dice: standard overlap metric, main optimization target
        - IoU (Jaccard): slightly stricter than Dice, commonly reported
        - clDice: topology-preserving metric, penalises broken vessel
          centerlines. Critical for OCTA where connectivity matters clinically.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder: hierarchical feature extraction, resolution halved each stage
    c1 = residual_block(layers.Conv2D(FILTERS[0], KERNEL_SIZE, padding='same')(inputs), FILTERS[0])
    p1 = layers.MaxPooling2D(SCALE_FACTOR)(c1)   # 400→200

    c2 = residual_block(layers.Conv2D(FILTERS[1], KERNEL_SIZE, padding='same')(p1), FILTERS[1])
    p2 = layers.MaxPooling2D(SCALE_FACTOR)(c2)   # 200→100

    c3 = residual_block(layers.Conv2D(FILTERS[2], KERNEL_SIZE, padding='same')(p2), FILTERS[2])
    p3 = layers.MaxPooling2D(SCALE_FACTOR)(c3)   # 100→50

    # Bottleneck: richest feature representation at lowest resolution
    b = residual_block(layers.Conv2D(FILTERS[3], KERNEL_SIZE, padding='same')(p3), FILTERS[3])
    b = layers.Dropout(DROPOUT_RATE)(b)

    # Decoder: upsample + concatenate skip connections from encoder
    # Skip connections re-introduce spatial detail lost during downsampling
    u3 = layers.UpSampling2D(SCALE_FACTOR)(b)    # 50→100
    u3 = layers.Dropout(0.2)(u3)
    c4 = residual_block(layers.Concatenate()([u3, c3]), FILTERS[2])

    u2 = layers.UpSampling2D(SCALE_FACTOR)(c4)   # 100→200
    u2 = layers.Dropout(0.2)(u2)
    c5 = residual_block(layers.Concatenate()([u2, c2]), FILTERS[1])

    u1 = layers.UpSampling2D(SCALE_FACTOR)(c5)   # 200→400
    u1 = layers.Dropout(0.2)(u1)
    c6 = residual_block(layers.Concatenate()([u1, c1]), FILTERS[0])

    # Output: sigmoid on 1 channel → vessel probability per pixel
    # sigmoid (not softmax) because this is binary, not multi-class
    output = layers.Conv2D(num_classes, kernel_size=1, activation='sigmoid',
                           name='segmentation')(c6)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, epsilon=EPSILON),
        loss=lambda y_true, y_pred: (
            dice_loss(y_true, y_pred) +
            BCE_WEIGHT * tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
        ),
        metrics=[DiceMetric(), IoUMetric()]
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 DATA GENERATOR: on-the-fly motion artifact simulation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_motion_artifacts(clean: np.ndarray, severity: float = 0.3) -> np.ndarray:
    """
    Simulate OCTA motion artifacts for restoration model training.

    Physical basis:
        Real OCTA acquires multiple repeated B-scans at each location and
        computes the decorrelation signal between them to detect moving RBCs.
        With only 2 repeats (vs. standard 8-12), SNR drops by √(NR) factor.
        Meanwhile, eye movements between repeats create:

        1. Shot noise: random photon arrival fluctuations, proportional to √intensity
           (Poisson statistics). Dominant noise source at low repeat counts.

        2. B-scan line dropout: eye saccades between repeat acquisitions corrupt
           entire horizontal scan lines — visible as white/dark horizontal stripes.
           This is the most clinically recognizable motion artifact in OCTA and
           matches what's visible in Andrei's B-scan images (the banding pattern).

        3. Speckle noise: coherent interference between scattered light creates
           a granular noise pattern inherent to all OCT systems. Described by
           Rayleigh distribution (fully developed speckle statistics).

    Args:
        clean:    float32 array [0,1], shape (H, W, 1) — clean OCTA projection
        severity: float in [0.15, 0.40] — overall degradation strength
                  0.15 = mild (few repeated B-scans), 0.40 = severe (2 repeats)

    Returns:
        degraded float32 array [0,1], same shape
    """
    img = clean.squeeze()  # (H, W)
    H, W = img.shape

    # 1. Shot noise — Gaussian approximation of Poisson noise
    # Variance scales with signal intensity: σ² ∝ √intensity
    shot_std = severity * MOTION_AUG['shot_noise_scale'] * np.sqrt(img + 1e-6)
    noisy = img + np.random.normal(0, 1, img.shape).astype(np.float32) * shot_std

    # 2. B-scan line dropout — horizontal stripe artifacts from eye saccades
    # Each dropped line is interpolated from its neighbour (not zeroed) to
    # simulate realistic partial corruption rather than complete data loss
    n_dropped = int(H * severity * MOTION_AUG['bscan_dropout_rate'])
    dropped_lines = np.random.choice(H, n_dropped, replace=False)
    for line in dropped_lines:
        neighbour = max(0, line - 1)
        noisy[line, :] = noisy[neighbour, :] * 0.4  # partial signal, not total loss

    # 3. Speckle noise — multiplicative Rayleigh-distributed noise
    # Multiplicative (not additive) because speckle scales with local intensity
    speckle = np.random.rayleigh(
        severity * MOTION_AUG['speckle_scale'], img.shape
    ).astype(np.float32)
    noisy = noisy * (1.0 + speckle)

    return np.expand_dims(np.clip(noisy, 0.0, 1.0), axis=-1)


def restoration_data_generator(
    img_dir: Path,
    augment: bool = False,
    batch_size: int = BATCH_SIZE,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yields (degraded, clean) pairs for supervised restoration training.
    Degradation is re-sampled every epoch so the model sees different noise
    realisations — this acts as implicit regularisation.
    """
    img_files = sorted(img_dir.iterdir())
    n = len(img_files)

    while True:
        indices = list(range(n))
        if augment:
            random.shuffle(indices)  # shuffle each epoch to prevent ordering bias

        for i in range(0, n, batch_size):
            batch_clean, batch_deg = [], []
            for j in indices[i:min(i + batch_size, n)]:
                raw = np.load(img_files[j])
                clean = raw.astype(np.float32) / 255.0  # uint8 [0,255] → float32 [0,1]
                if clean.ndim == 2:
                    clean = np.expand_dims(clean, axis=-1)  # (H,W) → (H,W,1)

                if augment:
                    # Geometric augmentation applied to clean image only
                    # (restoration is translation-invariant)
                    if np.random.rand() > 0.5: clean = np.fliplr(clean)
                    if np.random.rand() > 0.5: clean = np.flipud(clean)

                # Sample severity uniformly — curriculum from mild to severe
                severity = np.random.uniform(
                    MOTION_AUG['severity_min'],
                    MOTION_AUG['severity_max']
                )
                batch_clean.append(clean)
                batch_deg.append(simulate_motion_artifacts(clean, severity))

            # Input=degraded, Target=clean (model learns to reverse the degradation)
            yield np.array(batch_deg), np.array(batch_clean)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 DATA GENERATOR: image + mask pairs with augmentation
# ═══════════════════════════════════════════════════════════════════════════════

def segmentation_data_generator(
    img_dir: Path,
    mask_dir: Path,
    augment: bool = False,
    batch_size: int = BATCH_SIZE,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yields (image, mask) pairs for segmentation training.

    BUG FIXES from previous version:
        1. img_files already contains full Path objects from .iterdir() —
           previously doing np.load(img_dir / img_files[j]) prepended the
           directory twice, causing FileNotFoundError or loading wrong files.
        2. SegmentationMapsOnImage shape argument now uses img.shape (H,W,C)
           not img.shape[:2] (H,W) — imgaug requires the full shape tuple.
        3. Added epoch-level shuffling (augment=True) to prevent the model
           from learning sample ordering patterns.
    """
    img_files  = sorted(img_dir.iterdir())   # full Path objects
    mask_files = sorted(mask_dir.iterdir())  # full Path objects
    n = len(img_files)

    # Import here to avoid dependency if only using Stage 1
    import imgaug.augmenters as iaa
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage

    seq = iaa.Sequential([
        iaa.Fliplr(AUG_CONFIG['flip_lr']),
        iaa.Flipud(AUG_CONFIG['flip_ud']),
        iaa.Sometimes(0.5, iaa.Affine(
            rotate=AUG_CONFIG['rotate_range'],
            shear=AUG_CONFIG['shear_range'],
            scale=AUG_CONFIG['scale_range'],
            translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}
        )),
        iaa.ElasticTransformation(
            alpha=AUG_CONFIG['elastic_alpha'],
            sigma=AUG_CONFIG['elastic_sigma']
        )
    ])

    while True:
        indices = list(range(n))
        if augment:
            random.shuffle(indices)  # FIX 3: shuffle order each epoch

        for i in range(0, n, batch_size):
            batch_imgs, batch_masks = [], []

            for j in indices[i:min(i + batch_size, n)]:
                # FIX 1: use img_files[j] directly (already a full Path)
                img  = np.load(img_files[j]).astype(np.float32) / 255.0
                mask = np.load(mask_files[j]).astype(np.float32)
                # mask is already {0.0, 1.0} — no /255 needed (saved as uint8 {0,1})

                img  = np.expand_dims(img,  axis=-1)  # (H,W) → (H,W,1)
                mask = np.expand_dims(mask, axis=-1)  # (H,W) → (H,W,1)

                if augment:
                    uint8_img = (img * 255).astype(np.uint8)
                    mask_binary = (mask > 0.5).astype(np.uint8)

                    # FIX 2: pass img.shape (H,W,C), not img.shape[:2] (H,W)
                    segmap = SegmentationMapsOnImage(
                        mask_binary.squeeze(), shape=img.shape
                    )
                    aug_img, aug_segmap = seq(
                        image=uint8_img, segmentation_maps=segmap
                    )
                    img  = aug_img.astype(np.float32) / 255.0
                    mask = np.expand_dims(aug_segmap.get_arr(), axis=-1).astype(np.float32)
                    mask = (mask > 0.5).astype(np.float32)

                batch_imgs.append(img)
                batch_masks.append(mask)

            yield np.array(batch_imgs), np.array(batch_masks)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def train_restoration_model() -> None:
    """
    Stage 1: Train the restoration model on clean OCTA-500 images
    with on-the-fly motion artifact simulation.
    Saves: models/restoration_best.keras, models/restoration_final.keras
    """
    IMG_TRAIN_DIR = OUTPUT_DIR / 'images' / 'train'
    IMG_VALID_DIR = OUTPUT_DIR / 'images' / 'valid'

    model, _ = build_restoration_model()
    model.summary()

    train_gen   = restoration_data_generator(IMG_TRAIN_DIR, augment=True)
    valid_gen   = restoration_data_generator(IMG_VALID_DIR, augment=False)
    train_steps = len(list(IMG_TRAIN_DIR.iterdir())) // BATCH_SIZE
    valid_steps = len(list(IMG_VALID_DIR.iterdir())) // BATCH_SIZE

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODELS_DIR / 'restoration_best.keras',
                        monitor='val_loss', save_best_only=True, verbose=1),
        TensorBoard(log_dir=MODELS_DIR / 'logs' / 'restoration')
    ]

    model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=valid_gen,
        validation_steps=valid_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    model.save(MODELS_DIR / 'restoration_final.keras')
    print('*Stage 1 complete: restoration model saved')


def train_segmentation_model() -> None:
    """
    Stage 2: Train the segmentation model on clean OCTA-500 images.
    In the full pipeline, images would first be passed through the
    restoration model, but for OCTA-500 (clean dataset) we train directly.
    Saves: models/segmentation_best.keras, models/segmentation_final.keras
    """
    IMG_TRAIN_DIR  = OUTPUT_DIR / 'images' / 'train'
    IMG_VALID_DIR  = OUTPUT_DIR / 'images' / 'valid'
    MASK_TRAIN_DIR = OUTPUT_DIR / 'masks'  / 'train'
    MASK_VALID_DIR = OUTPUT_DIR / 'masks'  / 'valid'

    model = build_segmentation_model()
    model.summary()

    train_gen   = segmentation_data_generator(IMG_TRAIN_DIR, MASK_TRAIN_DIR, augment=True)
    valid_gen   = segmentation_data_generator(IMG_VALID_DIR, MASK_VALID_DIR, augment=False)
    train_steps = len(list(IMG_TRAIN_DIR.iterdir())) // BATCH_SIZE
    valid_steps = len(list(IMG_VALID_DIR.iterdir())) // BATCH_SIZE

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODELS_DIR / 'segmentation_best.keras',
                        monitor='val_dice', mode='max',
                        save_best_only=True, verbose=1),
        TensorBoard(log_dir=MODELS_DIR / 'logs' / 'segmentation')
    ]

    model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=valid_gen,
        validation_steps=valid_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    model.save(MODELS_DIR / 'segmentation_final.keras')
    print('*Stage 2 complete: segmentation model saved')


if __name__ == '__main__':
    # Run Stage 1 first, then Stage 2
    print('=== STAGE 1: Restoration ===')
    train_restoration_model()
    print('\n=== STAGE 2: Segmentation ===')
    train_segmentation_model()
