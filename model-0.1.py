"""
OCTA Image Restoration Model
==============================
Architecture: ResU-Net with dense skip connections
Loss: L2 + 0.01 * VGG19 Content Loss  (from Liao et al., Biomed. Opt. Express 2023)
Task: Reconstruct high-quality OCTA from sparsely/noisily acquired images
      Input  -> degraded image (simulated 2-repeat sparse acquisition)
      Output -> clean image    (ground truth dense acquisition)

Key changes from segmentation model:
  - Output head: 1-channel sigmoid reconstruction (not multi-class)
  - Loss: L2 + perceptual content loss (not Dice + BCE)
  - Metrics: PSNR + SSIM (not Dice)
  - Generator: (degraded, clean) pairs synthesized on-the-fly
"""

from pathlib import Path
from typing import Generator

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l2

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE      = (304, 304)   # adjust to your actual image size
FILTERS       = [32, 64, 128, 256]
KERNEL_SIZE   = 3
SCALE_FACTOR  = 2
DROPOUT_RATE  = 0.3
LEARNING_RATE = 1e-4
BATCH_SIZE    = 4
EPOCHS        = 100
CONTENT_WEIGHT = 0.01        # β from Liao et al. — sweet spot found empirically

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (SCRIPT_DIR / 'data/OCTA-500_processed').resolve()
MODELS_DIR = SCRIPT_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# ── GPU setup ─────────────────────────────────────────────────────────────────
physical_devices = tf.config.list_physical_devices('GPU')
print(f'*AVAILABLE GPUs: {physical_devices}')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# ── Metrics ───────────────────────────────────────────────────────────────────

class PSNRMetric(tf.keras.metrics.Metric):
    """Peak Signal-to-Noise Ratio — standard restoration quality metric."""
    def __init__(self, name='psnr', **kwargs):
        super().__init__(name=name, **kwargs)
        self.psnr_sum = self.add_weight(name='psnr_sum', initializer='zeros')
        self.count    = self.add_weight(name='count',    initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        psnr = tf.image.psnr(y_true, y_pred, max_val=1.0)
        self.psnr_sum.assign_add(tf.reduce_sum(psnr))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.psnr_sum / self.count

    def reset_states(self):
        self.psnr_sum.assign(0.)
        self.count.assign(0.)


class SSIMMetric(tf.keras.metrics.Metric):
    """Structural Similarity Index — captures perceptual image quality."""
    def __init__(self, name='ssim', **kwargs):
        super().__init__(name=name, **kwargs)
        self.ssim_sum = self.add_weight(name='ssim_sum', initializer='zeros')
        self.count    = self.add_weight(name='count',    initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
        self.ssim_sum.assign_add(tf.reduce_sum(ssim))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.ssim_sum / self.count

    def reset_states(self):
        self.ssim_sum.assign(0.)
        self.count.assign(0.)


# ── Loss ──────────────────────────────────────────────────────────────────────

def build_vgg_feature_extractor() -> Model:
    """
    Extract deep features from VGG19 'block5_conv4' layer.
    This is the optimal layer found by Liao et al. for OCTA content loss.
    Input must be 3-channel — we tile grayscale OCTA images to RGB.
    """
    vgg = VGG19(include_top=False, weights='imagenet',
                input_shape=(*IMG_SIZE, 3))
    vgg.trainable = False
    return Model(inputs=vgg.input,
                 outputs=vgg.get_layer('block5_conv4').output,
                 name='vgg_feature_extractor')


# Build once at module level so it isn't rebuilt every forward pass
_vgg_extractor = None

def _get_vgg():
    global _vgg_extractor
    if _vgg_extractor is None:
        _vgg_extractor = build_vgg_feature_extractor()
    return _vgg_extractor


def content_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    VGG19 perceptual content loss.
    Computes MSE between deep feature maps of ground truth and prediction.
    Preserves high-frequency vascular texture detail (Liao et al., 2023).
    """
    vgg = _get_vgg()
    # Tile grayscale (H,W,1) -> (H,W,3) for VGG input
    y_true_rgb = tf.repeat(y_true, 3, axis=-1)
    y_pred_rgb = tf.repeat(y_pred, 3, axis=-1)
    # Scale to VGG's expected [0, 255] range
    y_true_rgb = y_true_rgb * 255.0
    y_pred_rgb = y_pred_rgb * 255.0

    true_features = vgg(y_true_rgb, training=False)
    pred_features = vgg(y_pred_rgb, training=False)
    return tf.reduce_mean(tf.square(true_features - pred_features))


def restoration_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Combined loss from Liao et al.:
        L = L2 + 0.01 * L_content
    L2 stabilises training and reduces noise.
    Content loss preserves perceptual vascular texture details.
    """
    l2_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    c_loss  = content_loss(y_true, y_pred)
    return l2_loss + CONTENT_WEIGHT * c_loss


# ── Architecture ──────────────────────────────────────────────────────────────

def residual_dense_block(x: tf.Tensor, filters: int,
                          kernel_size: int = KERNEL_SIZE) -> tf.Tensor:
    """
    Residual block with dense feature concatenation.
    Removes batch norm (Liao et al. found it hurts contrast in reconstruction).
    Uses LeakyReLU for smoother gradients on restoration tasks.
    """
    shortcut = x

    h1 = layers.Conv2D(filters, kernel_size, padding='same',
                        kernel_regularizer=l2(1e-4))(x)
    h1 = layers.LeakyReLU(0.2)(h1)

    # Dense connection: concatenate input with h1 features
    h2_input = layers.Concatenate()([x, h1]) if x.shape[-1] == filters else h1
    h2 = layers.Conv2D(filters, kernel_size, padding='same',
                        kernel_regularizer=l2(1e-4))(h2_input)
    h2 = layers.LeakyReLU(0.2)(h2)

    # Residual skip
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, padding='same')(shortcut)

    out = layers.Add()([shortcut, h2])
    return out


def build_restoration_model(
    input_shape: tuple = (*IMG_SIZE, 1)
) -> Model:
    """
    ResU-Net for OCTA image restoration.

    Encoder: extracts hierarchical features, downsamples x8
    Bottleneck: richest feature representation + dropout
    Decoder: upsamples with skip connections to recover spatial detail
    Output: single-channel [0,1] reconstructed image

    Architecture grounded in:
    - Liao et al. (IRU-Net, Biomed. Opt. Express 2023) for loss + dense blocks
    - Das et al. (RRTGAN, npj AI 2025) for skip connection rationale
    """
    inputs = tf.keras.Input(shape=input_shape)

    # ── Encoder ──
    c1 = residual_dense_block(
        layers.Conv2D(FILTERS[0], KERNEL_SIZE, padding='same')(inputs), FILTERS[0])
    p1 = layers.MaxPooling2D(SCALE_FACTOR)(c1)

    c2 = residual_dense_block(
        layers.Conv2D(FILTERS[1], KERNEL_SIZE, padding='same')(p1), FILTERS[1])
    p2 = layers.MaxPooling2D(SCALE_FACTOR)(c2)

    c3 = residual_dense_block(
        layers.Conv2D(FILTERS[2], KERNEL_SIZE, padding='same')(p2), FILTERS[2])
    p3 = layers.MaxPooling2D(SCALE_FACTOR)(c3)

    # ── Bottleneck ──
    b = residual_dense_block(
        layers.Conv2D(FILTERS[3], KERNEL_SIZE, padding='same')(p3), FILTERS[3])
    b = layers.Dropout(DROPOUT_RATE)(b)

    # ── Decoder ──
    # Each decoder stage: upsample -> concat skip connection -> refine
    u3 = layers.UpSampling2D(SCALE_FACTOR)(b)
    u3 = layers.Dropout(0.15)(u3)
    c4 = residual_dense_block(layers.Concatenate()([u3, c3]), FILTERS[2])

    u2 = layers.UpSampling2D(SCALE_FACTOR)(c4)
    u2 = layers.Dropout(0.15)(u2)
    c5 = residual_dense_block(layers.Concatenate()([u2, c2]), FILTERS[1])

    u1 = layers.UpSampling2D(SCALE_FACTOR)(c5)
    u1 = layers.Dropout(0.15)(u1)
    c6 = residual_dense_block(layers.Concatenate()([u1, c1]), FILTERS[0])

    # ── Output ──
    # sigmoid keeps output in [0,1] to match normalised input
    output = layers.Conv2D(1, kernel_size=1, activation='sigmoid',
                           name='restoration')(c6)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, beta_1=0.8, beta_2=0.999),
        loss=restoration_loss,
        metrics=[PSNRMetric(), SSIMMetric()]
    )
    return model


# ── Data Generator ────────────────────────────────────────────────────────────

def simulate_sparse_acquisition(clean: np.ndarray,
                                 severity: float = 0.3) -> np.ndarray:
    """
    Simulate low-quality OCTA from sparse B-scan acquisition.

    Mimics what happens physically when you use only 2 repeats instead of 12:
    1. Shot noise  — photon counting noise, scales with sqrt(intensity)
    2. B-scan dropout — random horizontal lines lost (motion between repeats)
    3. Speckle noise — coherent interference artefact inherent to OCT

    This is consistent with the degradation described in:
    - Liao et al. 2023 (2-repeat vs 12-repeat acquisition)
    - Das et al. 2025 (1/4 sparse vs dense pixel sampling)

    Args:
        clean: float32 array in [0,1], shape (H, W, 1)
        severity: controls overall degradation strength [0.1 – 0.5]

    Returns:
        degraded: float32 array in [0,1], same shape
    """
    img = clean.squeeze()  # (H, W)
    H, W = img.shape

    # 1. Shot noise (Poisson-like via Gaussian approximation)
    shot_std = severity * 0.3 * np.sqrt(img + 1e-6)
    noisy = img + np.random.normal(0, 1, img.shape).astype(np.float32) * shot_std

    # 2. B-scan line dropout — simulates motion between repeat acquisitions
    n_dropped = int(H * severity * 0.4)
    dropped_lines = np.random.choice(H, n_dropped, replace=False)
    for line in dropped_lines:
        # Interpolate from neighbours rather than zeroing (more realistic)
        neighbour = max(0, line - 1)
        noisy[line, :] = noisy[neighbour, :] * 0.4

    # 3. Multiplicative speckle noise (coherent noise pattern)
    speckle = np.random.rayleigh(severity * 0.2, img.shape).astype(np.float32)
    noisy = noisy * (1.0 + speckle)

    # Clip and re-normalise to [0,1]
    noisy = np.clip(noisy, 0.0, 1.0)
    return np.expand_dims(noisy, axis=-1)


def restoration_data_generator(
    img_dir: Path,
    augment: bool = False,
    batch_size: int = BATCH_SIZE,
    severity_range: tuple = (0.15, 0.40)
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yields (degraded, clean) pairs for supervised restoration training.

    The degradation is applied on-the-fly so the model sees slightly
    different noise realisations each epoch — acts as implicit regularisation.

    Args:
        img_dir: directory of clean .npy images (float32 or uint8)
        augment: apply random flips for data augmentation
        batch_size: number of pairs per batch
        severity_range: (min, max) degradation severity sampled uniformly
    """
    img_files  = sorted(img_dir.iterdir())
    num_samples = len(img_files)

    while True:
        indices = np.arange(num_samples)
        if augment:
            np.random.shuffle(indices)  # shuffle each epoch when training

        for i in range(0, num_samples, batch_size):
            batch_clean    = []
            batch_degraded = []

            for j in indices[i : min(i + batch_size, num_samples)]:
                # Load + normalise
                raw = np.load(img_files[j])
                if raw.dtype == np.uint8:
                    clean = raw.astype(np.float32) / 255.0
                else:
                    clean = raw.astype(np.float32)
                    if clean.max() > 1.0:
                        clean /= 255.0

                # Ensure (H, W, 1)
                if clean.ndim == 2:
                    clean = np.expand_dims(clean, axis=-1)

                # Optional geometric augmentation applied to BOTH
                if augment:
                    if np.random.rand() > 0.5:
                        clean = np.fliplr(clean)
                    if np.random.rand() > 0.5:
                        clean = np.flipud(clean)

                # Synthesize degraded version on-the-fly
                severity = np.random.uniform(*severity_range)
                degraded = simulate_sparse_acquisition(clean, severity=severity)

                batch_clean.append(clean)
                batch_degraded.append(degraded)

            # Input = degraded, Target = clean
            yield np.array(batch_degraded), np.array(batch_clean)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model() -> None:
    """
    Train OCTA restoration model and save checkpoints.
    Expects preprocessed clean .npy images in:
        data/OCTA-500_processed/images/train/
        data/OCTA-500_processed/images/valid/
    """
    IMG_TRAIN_DIR = OUTPUT_DIR / 'images' / 'train'
    IMG_VALID_DIR = OUTPUT_DIR / 'images' / 'valid'

    model = build_restoration_model()
    model.summary()

    train_gen   = restoration_data_generator(IMG_TRAIN_DIR, augment=True)
    valid_gen   = restoration_data_generator(IMG_VALID_DIR, augment=False)
    train_steps = len(list(IMG_TRAIN_DIR.iterdir())) // BATCH_SIZE
    valid_steps = len(list(IMG_VALID_DIR.iterdir())) // BATCH_SIZE

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODELS_DIR / 'octa_restoration_best.keras',
                        monitor='val_psnr', mode='max',
                        save_best_only=True, verbose=1),
        TensorBoard(log_dir=MODELS_DIR / 'logs')
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=valid_gen,
        validation_steps=valid_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    model.save(MODELS_DIR / 'octa_restoration_final.keras')
    print('*COMPLETE: restoration model saved.')
    return history


if __name__ == '__main__':
    train_model()
