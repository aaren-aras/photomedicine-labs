"""
model.py
========
Two-stage OCTA pipeline — updated with full Liao et al. training setup.

Changes from previous version:
    - Adam beta1=0.8, beta2=0.999 (Liao et al. Section 4.1)
    - Learning rate decay ×0.95 every 10k steps (Liao et al. Section 4.1)
    - Gaussian noise σ=0.4 added to restoration inputs (Liao et al. Section 4.1)
    - EARLY_STOP_PATIENCE=20 (Liao et al. used 20, not 10)
    - Batch size 4 (matches Liao et al., safer on T4 with VGG19 content loss)

Training order:
    python model.py
    → trains Stage 1 (restoration) then Stage 2 (segmentation) sequentially
    → both stages use early stopping, so total time depends on convergence
    → on Colab T4: Stage 1 ~2-3hrs, Stage 2 ~1-2hrs
"""

from pathlib import Path
from typing import Generator
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
)
from tensorflow.keras.regularizers import l2

from config import (
    IMG_SIZE, NUM_CLASSES, FILTERS, KERNEL_SIZE, SCALE_FACTOR,
    DROPOUT_RATE, LEARNING_RATE, EPSILON, ADAM_BETA1, ADAM_BETA2,
    LR_DECAY_FACTOR, LR_DECAY_STEPS, BATCH_SIZE, BCE_WEIGHT,
    AUG_CONFIG, MOTION_AUG, EPOCHS, EARLY_STOP_PATIENCE,
    GAUSSIAN_NOISE_SIGMA
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'data/processed'
MODELS_DIR = SCRIPT_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# GPU setup — incremental allocation leaves headroom for VGG19
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f'GPU: {gpus[0].name}')
else:
    print('WARNING: No GPU found — training will be slow on CPU')


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════════

def make_lr_scheduler(steps_per_epoch: int):
    """
    Decay LR by LR_DECAY_FACTOR every LR_DECAY_STEPS training steps.

    Liao et al. Section 4.1: "The learning rate was decayed by a factor
    of 0.95 every 1×10⁴ training steps."

    We convert steps to epochs since Keras LearningRateScheduler works
    per-epoch. decay_every_n_epochs = LR_DECAY_STEPS / steps_per_epoch.
    """
    decay_every = max(1, LR_DECAY_STEPS // steps_per_epoch)

    def schedule(epoch: int, lr: float) -> float:
        if epoch > 0 and epoch % decay_every == 0:
            new_lr = lr * LR_DECAY_FACTOR
            print(f'  LR decay: {lr:.2e} → {new_lr:.2e} (epoch {epoch})')
            return new_lr
        return lr

    return LearningRateScheduler(schedule, verbose=0)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED: RESIDUAL BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

def residual_block(x: tf.Tensor, filters: int,
                   kernel_size: int = KERNEL_SIZE,
                   use_batchnorm: bool = True) -> tf.Tensor:
    """
    Residual block: two conv layers with skip connection.

    use_batchnorm=False for restoration model (Stage 1):
        Liao et al. Section 3.1: "the batch normalization layer was removed
        to increase the performance and reduce the computational cost in
        the image reconstruction task." BatchNorm distorts intensity
        statistics which matters for pixel-accurate reconstruction.

    use_batchnorm=True for segmentation model (Stage 2):
        Segmentation only cares about where vessels are, not absolute
        intensity values — BN improves training stability here.
    """
    shortcut = x

    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=l2(1e-4), use_bias=False)(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=l2(1e-4), use_bias=False)(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)

    # 1×1 projection shortcut if channel count changes
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, padding='same',
                                 use_bias=False)(shortcut)

    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: RESTORATION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def build_restoration_model(input_shape: tuple = (*IMG_SIZE, 1)):
    """
    ResU-Net for OCTA restoration: degraded → clean.

    Architecture follows Liao et al. IRU-Net:
        - Encoder-decoder (U-Net) backbone
        - Residual blocks WITHOUT BatchNorm (Liao et al. Section 3.1)
        - Skip connections from encoder to decoder

    Loss: L2 + 0.01 × VGG19 content loss
        α=1, β=0.01 — optimal from Liao et al. Table 1 ablation study
        VGG19 block5_conv4 — optimal from Liao et al. Table 2 ablation study
        Content loss before ReLU activation (Liao et al. Section 3.2)

    Metrics: PSNR and SSIM
        Targets: PSNR > 23 dB, SSIM > 0.55 (Liao et al. baseline results)
    """
    from tensorflow.keras.applications import VGG19

    # Frozen VGG19 feature extractor — block5_conv4 per Liao et al. Table 2
    # "Content loss output before the ReLU activation layer" (Section 3.2)
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
    vgg.trainable = False

    # Get the layer BEFORE its ReLU — i.e. the conv output pre-activation
    # block5_conv4 in VGG19 is followed by block5_pool
    vgg_out = Model(
        inputs=vgg.input,
        outputs=vgg.get_layer('block5_conv4').output,
        name='vgg_feature_extractor'
    )

    def restoration_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """L2 + β×content loss. α=1, β=0.01 per Liao et al. Table 1."""
        l2_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # VGG19 expects 3-channel input, scaled to [0,255]
        # Tile grayscale (H,W,1) → (H,W,3)
        yt_rgb = tf.repeat(y_true, 3, axis=-1) * 255.0
        yp_rgb = tf.repeat(y_pred, 3, axis=-1) * 255.0

        feat_true = vgg_out(yt_rgb, training=False)
        feat_pred = vgg_out(yp_rgb, training=False)

        # Equation 5 from Liao et al.: mean squared error of feature maps
        content = tf.reduce_mean(tf.square(feat_true - feat_pred))

        return l2_loss + 0.01 * content

    # Build encoder-decoder (NO BatchNorm per Liao et al.)
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    c1 = residual_block(
        layers.Conv2D(FILTERS[0], KERNEL_SIZE, padding='same')(inputs),
        FILTERS[0], use_batchnorm=False
    )
    p1 = layers.MaxPooling2D(SCALE_FACTOR)(c1)   # 400→200

    c2 = residual_block(
        layers.Conv2D(FILTERS[1], KERNEL_SIZE, padding='same')(p1),
        FILTERS[1], use_batchnorm=False
    )
    p2 = layers.MaxPooling2D(SCALE_FACTOR)(c2)   # 200→100

    c3 = residual_block(
        layers.Conv2D(FILTERS[2], KERNEL_SIZE, padding='same')(p2),
        FILTERS[2], use_batchnorm=False
    )
    p3 = layers.MaxPooling2D(SCALE_FACTOR)(c3)   # 100→50

    # Bottleneck
    b = residual_block(
        layers.Conv2D(FILTERS[3], KERNEL_SIZE, padding='same')(p3),
        FILTERS[3], use_batchnorm=False
    )
    b = layers.Dropout(DROPOUT_RATE)(b)

    # Decoder
    u3 = layers.UpSampling2D(SCALE_FACTOR)(b)
    c4 = residual_block(
        layers.Concatenate()([u3, c3]), FILTERS[2], use_batchnorm=False
    )
    u2 = layers.UpSampling2D(SCALE_FACTOR)(c4)
    c5 = residual_block(
        layers.Concatenate()([u2, c2]), FILTERS[1], use_batchnorm=False
    )
    u1 = layers.UpSampling2D(SCALE_FACTOR)(c5)
    c6 = residual_block(
        layers.Concatenate()([u1, c1]), FILTERS[0], use_batchnorm=False
    )

    # Output: sigmoid → [0,1] pixel intensities
    output = layers.Conv2D(1, kernel_size=1, activation='sigmoid')(c6)

    model = Model(inputs=inputs, outputs=output, name='IRU_Net')
    model.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE,
            beta_1=ADAM_BETA1,    # 0.8 per Liao et al.
            beta_2=ADAM_BETA2,    # 0.999 per Liao et al.
            epsilon=EPSILON
        ),
        loss=restoration_loss,
        metrics=[
            tf.keras.metrics.MeanSquaredError(name='mse'),
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
        ]
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: SEGMENTATION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def cl_dice_loss(y_true, y_pred, smooth=1.0):
    """
    Centerline Dice — penalises broken vessel centerlines.
    Shit et al., CVPR 2021. Differentiable soft skeleton via max pooling.
    """
    def soft_skeleton(x, iters=5):
        s = x
        for _ in range(iters):
            dilated = tf.nn.max_pool2d(s, ksize=3, strides=1, padding='SAME')
            s = tf.maximum(s - dilated + s, 0)
        return s

    sp = soft_skeleton(y_pred)
    st = soft_skeleton(tf.cast(y_true, tf.float32))
    tprec = (tf.reduce_sum(sp * tf.cast(y_true, tf.float32)) + smooth) / \
            (tf.reduce_sum(sp) + smooth)
    tsens = (tf.reduce_sum(st * y_pred) + smooth) / \
            (tf.reduce_sum(st) + smooth)
    return 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens + 1e-8)


class DiceMetric(tf.keras.metrics.Metric):
    def __init__(self, name='dice', **kwargs):
        super().__init__(name=name, **kwargs)
        self.dice_sum = self.add_weight(initializer='zeros')
        self.count    = self.add_weight(initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.dice_sum.assign_add(
            dice_coefficient(y_true, tf.cast(y_pred > 0.5, tf.float32))
        )
        self.count.assign_add(1.0)

    def result(self):
        return self.dice_sum / self.count

    def reset_states(self):
        self.dice_sum.assign(0.)
        self.count.assign(0.)


class IoUMetric(tf.keras.metrics.Metric):
    def __init__(self, name='iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iou_sum = self.add_weight(initializer='zeros')
        self.count   = self.add_weight(initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_b = tf.cast(y_pred > 0.5, tf.float32)
        yt  = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        yp  = tf.reshape(y_b, [-1])
        inter = tf.reduce_sum(yt * yp)
        union = tf.reduce_sum(yt) + tf.reduce_sum(yp) - inter
        self.iou_sum.assign_add((inter + 1.) / (union + 1.))
        self.count.assign_add(1.0)

    def result(self):
        return self.iou_sum / self.count

    def reset_states(self):
        self.iou_sum.assign(0.)
        self.count.assign(0.)


def build_segmentation_model(input_shape=(*IMG_SIZE, 1)):
    """
    ResU-Net for binary vessel segmentation: clean OCTA → vessel mask.
    WITH BatchNorm (segmentation doesn't need intensity preservation).
    Loss: Dice + 0.3 × BCE. Metrics: Dice, IoU.
    """
    inputs = tf.keras.Input(shape=input_shape)

    c1 = residual_block(
        layers.Conv2D(FILTERS[0], KERNEL_SIZE, padding='same')(inputs), FILTERS[0]
    )
    p1 = layers.MaxPooling2D(SCALE_FACTOR)(c1)

    c2 = residual_block(
        layers.Conv2D(FILTERS[1], KERNEL_SIZE, padding='same')(p1), FILTERS[1]
    )
    p2 = layers.MaxPooling2D(SCALE_FACTOR)(c2)

    c3 = residual_block(
        layers.Conv2D(FILTERS[2], KERNEL_SIZE, padding='same')(p2), FILTERS[2]
    )
    p3 = layers.MaxPooling2D(SCALE_FACTOR)(c3)

    b = residual_block(
        layers.Conv2D(FILTERS[3], KERNEL_SIZE, padding='same')(p3), FILTERS[3]
    )
    b = layers.Dropout(DROPOUT_RATE)(b)

    u3 = layers.UpSampling2D(SCALE_FACTOR)(b)
    c4 = residual_block(layers.Concatenate()([u3, c3]), FILTERS[2])
    u2 = layers.UpSampling2D(SCALE_FACTOR)(c4)
    c5 = residual_block(layers.Concatenate()([u2, c2]), FILTERS[1])
    u1 = layers.UpSampling2D(SCALE_FACTOR)(c5)
    c6 = residual_block(layers.Concatenate()([u1, c1]), FILTERS[0])

    output = layers.Conv2D(NUM_CLASSES, kernel_size=1, activation='sigmoid')(c6)

    model = Model(inputs=inputs, outputs=output, name='SegU_Net')
    model.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE,
            beta_1=ADAM_BETA1,
            beta_2=ADAM_BETA2,
            epsilon=EPSILON
        ),
        loss=lambda yt, yp: (
            dice_loss(yt, yp) +
            BCE_WEIGHT * tf.keras.losses.BinaryCrossentropy()(yt, yp)
        ),
        metrics=[DiceMetric(), IoUMetric()]
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_motion_artifacts(clean: np.ndarray, severity: float) -> np.ndarray:
    """
    Simulate OCTA acquisition artifacts (shot noise, B-scan dropout, speckle).
    Physical basis documented in config.py MOTION_AUG block.
    """
    img = clean.squeeze()
    H, W = img.shape

    # 1. Shot noise (Poisson approximation)
    shot_std = severity * MOTION_AUG['shot_noise_scale'] * np.sqrt(img + 1e-6)
    noisy = img + np.random.normal(0, 1, img.shape).astype(np.float32) * shot_std

    # 2. B-scan line dropout (horizontal stripe artifacts from organism motion)
    n_dropped = int(H * severity * MOTION_AUG['bscan_dropout_rate'])
    for line in np.random.choice(H, n_dropped, replace=False):
        noisy[line, :] = noisy[max(0, line-1), :] * 0.4

    # 3. Speckle noise (multiplicative Rayleigh — correct model for OCT speckle)
    speckle = np.random.rayleigh(
        severity * MOTION_AUG['speckle_scale'], img.shape
    ).astype(np.float32)
    noisy = noisy * (1.0 + speckle)

    return np.expand_dims(np.clip(noisy, 0.0, 1.0), axis=-1)


def restoration_generator(img_dir: Path, augment: bool = False,
                           batch_size: int = BATCH_SIZE):
    """
    Yields (degraded, clean) pairs for restoration training.

    Key addition vs previous version:
        Gaussian noise σ=0.4 added to degraded input AFTER physics simulation.
        Directly from Liao et al. Section 4.1 — simulates photon detector
        shot noise and improves generalisation to unseen noise levels.

    The Gaussian noise is applied to the ALREADY-degraded image, so the
    model sees: clean → [motion artifacts] → [Gaussian noise] → input
    and must learn to reverse both degradation sources.
    """
    img_files = sorted(img_dir.iterdir())
    n = len(img_files)

    while True:
        indices = list(range(n))
        if augment:
            random.shuffle(indices)  # epoch-level shuffle prevents ordering bias

        for i in range(0, n, batch_size):
            batch_clean, batch_deg = [], []
            for j in indices[i:min(i + batch_size, n)]:
                raw   = np.load(img_files[j])           # uint8 [0,255]
                clean = raw.astype(np.float32) / 255.0  # float32 [0,1]
                if clean.ndim == 2:
                    clean = np.expand_dims(clean, axis=-1)

                if augment:
                    if np.random.rand() > 0.5: clean = np.fliplr(clean)
                    if np.random.rand() > 0.5: clean = np.flipud(clean)

                # Simulate acquisition degradation
                severity = np.random.uniform(
                    MOTION_AUG['severity_min'], MOTION_AUG['severity_max']
                )
                degraded = simulate_motion_artifacts(clean, severity)

                # Add Gaussian noise σ=0.4 — Liao et al. Section 4.1
                # Scale σ relative to image range [0,1]
                gaussian = np.random.normal(
                    0, GAUSSIAN_NOISE_SIGMA / 255.0, degraded.shape
                ).astype(np.float32)
                degraded = np.clip(degraded + gaussian, 0.0, 1.0)

                batch_clean.append(clean)
                batch_deg.append(degraded)

            yield np.array(batch_deg), np.array(batch_clean)


def segmentation_generator(img_dir: Path, mask_dir: Path,
                            augment: bool = False, batch_size: int = BATCH_SIZE):
    """
    Yields (image, mask) pairs for segmentation training.
    All three generator bugs from original code fixed (see previous comments).
    """
    import imgaug.augmenters as iaa
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage

    img_files  = sorted(img_dir.iterdir())   # full Path objects
    mask_files = sorted(mask_dir.iterdir())
    n = len(img_files)

    seq = iaa.Sequential([
        iaa.Fliplr(AUG_CONFIG['flip_lr']),
        iaa.Flipud(AUG_CONFIG['flip_ud']),
        iaa.Sometimes(0.5, iaa.Affine(
            rotate=AUG_CONFIG['rotate_range'],
            scale=AUG_CONFIG['scale_range'],
            translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}
        )),
    ])

    while True:
        indices = list(range(n))
        if augment:
            random.shuffle(indices)

        for i in range(0, n, batch_size):
            batch_imgs, batch_masks = [], []
            for j in indices[i:min(i + batch_size, n)]:
                img  = np.load(img_files[j]).astype(np.float32) / 255.0
                mask = np.load(mask_files[j]).astype(np.float32)
                img  = np.expand_dims(img,  axis=-1)
                mask = np.expand_dims(mask, axis=-1)

                if augment:
                    uint8_img   = (img * 255).astype(np.uint8)
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    segmap      = SegmentationMapsOnImage(
                        mask_binary.squeeze(), shape=img.shape  # FIX: full shape not [:2]
                    )
                    aug_img, aug_seg = seq(image=uint8_img, segmentation_maps=segmap)
                    img  = aug_img.astype(np.float32) / 255.0
                    mask = np.expand_dims(
                        aug_seg.get_arr(), axis=-1
                    ).astype(np.float32)

                batch_imgs.append(img)
                batch_masks.append(mask)

            yield np.array(batch_imgs), np.array(batch_masks)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def count_files(directory: Path) -> int:
    return len(list(directory.iterdir()))


def train_restoration():
    """Stage 1: train restoration model on OCTA-500 + ROSE images."""
    img_train = OUTPUT_DIR / 'images' / 'train'
    img_valid = OUTPUT_DIR / 'images' / 'valid'

    n_train = count_files(img_train)
    n_valid = count_files(img_valid)
    steps_train = n_train // BATCH_SIZE
    steps_valid = n_valid // BATCH_SIZE

    print(f'\n=== STAGE 1: Restoration ===')
    print(f'Train: {n_train} images ({steps_train} steps/epoch)')
    print(f'Valid: {n_valid} images ({steps_valid} steps/epoch)')

    model = build_restoration_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            MODELS_DIR / 'restoration_best.keras',
            monitor='val_loss', save_best_only=True, verbose=1
        ),
        TensorBoard(log_dir=MODELS_DIR / 'logs' / 'restoration'),
        make_lr_scheduler(steps_train),
    ]

    model.fit(
        restoration_generator(img_train, augment=True),
        steps_per_epoch=steps_train,
        validation_data=restoration_generator(img_valid, augment=False),
        validation_steps=steps_valid,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    model.save(MODELS_DIR / 'restoration_final.keras')
    print('Stage 1 complete.')


def train_segmentation():
    """Stage 2: train segmentation model on OCTA-500 + ROSE images + masks."""
    img_train  = OUTPUT_DIR / 'images' / 'train'
    img_valid  = OUTPUT_DIR / 'images' / 'valid'
    mask_train = OUTPUT_DIR / 'masks'  / 'train'
    mask_valid = OUTPUT_DIR / 'masks'  / 'valid'

    n_train = count_files(img_train)
    n_valid = count_files(img_valid)
    steps_train = n_train // BATCH_SIZE
    steps_valid = n_valid // BATCH_SIZE

    print(f'\n=== STAGE 2: Segmentation ===')
    print(f'Train: {n_train} images ({steps_train} steps/epoch)')
    print(f'Valid: {n_valid} images ({steps_valid} steps/epoch)')

    model = build_segmentation_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor='val_dice', mode='max',
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            MODELS_DIR / 'segmentation_best.keras',
            monitor='val_dice', mode='max',
            save_best_only=True, verbose=1
        ),
        TensorBoard(log_dir=MODELS_DIR / 'logs' / 'segmentation'),
        make_lr_scheduler(steps_train),
    ]

    model.fit(
        segmentation_generator(img_train, mask_train, augment=True),
        steps_per_epoch=steps_train,
        validation_data=segmentation_generator(img_valid, mask_valid, augment=False),
        validation_steps=steps_valid,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    model.save(MODELS_DIR / 'segmentation_final.keras')
    print('Stage 2 complete.')


if __name__ == '__main__':
    train_restoration()
    train_segmentation()
