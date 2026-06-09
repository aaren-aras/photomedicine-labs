"""
2-stage ResU-Net CNN pipeline for OCTA image restoration followed by vessel segmentation,
trained on augmented OCTA-500 and ROSE data, based loosely on Liao et al. setup (see docs). 
"""

from pathlib import Path
from typing import Generator
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    LearningRateScheduler, EarlyStopping, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.regularizers import l2

from config import (
    IMG_SIZE, NUM_CLASSES, LOSS_ALPHA, LOSS_BETA, SMOOTH, SKELETON_ITERS,
    FILTERS, KERNEL_SIZE, SCALE_FACTOR, DROPOUT_RATE, DROPOUT_SEVERITY, 
    LEARNING_RATE, EPSILON, ADAM_BETA1, ADAM_BETA2, LR_DECAY_FACTOR, 
    LR_DECAY_STEPS, BATCH_SIZE, BCE_WEIGHT, GEO_AUG, MOTION_AUG, 
    EPOCHS, EARLY_STOP_PATIENCE, GAUSSIAN_NOISE_SIGMA
)

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / 'data' / 'processed'
MODELS_DIR = ROOT_DIR / 'benchmark'
MODELS_DIR.mkdir(exist_ok=True)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Allocate memory incrementally instead of all at once
    tf.config.experimental.set_memory_growth(gpus[0], True)

    build_info = tf.sysconfig.get_build_info()
    cuda_version = build_info.get('cuda_version', 'Unknown')
    cudnn_version = build_info.get('cudnn_version', 'Unknown')
    cuda_compute_capabilities = tf_build_info.build_info.get('cuda_compute_capabilities')

    print(f'*CUDA TOOLKIT VERSION: {cuda_version}')
    print(f'*cuDNN VERSION: {cudnn_version}')
    print(f'*TF CUDA COMPUTE CAPABILITIES: {cuda_compute_capabilities}')
    print(f'*GPUs FOUND: {gpus[0].name}')
else:
    print('*WARNING: No GPU found. Training will begin on CPU, but it\'ll be slower...')


def count_files(directory: Path) -> int:
    '''Counts the total number of files and subdirectories within a given path.'''
    return len(list(directory.iterdir()))


def make_lr_scheduler(steps_per_epoch: int) -> LearningRateScheduler:
    """
    Reduces learning rate for rapid learning early on but fine-tuned vessel detail later. 
    Steps are converted to epochs cause Keras LearningRateScheduler works per-epoch.
    """
    decay_every = max(1, LR_DECAY_STEPS // steps_per_epoch)

    def schedule(epoch: int, lr: float) -> float:
        if epoch > 0 and epoch % decay_every == 0:
            new_lr = lr * LR_DECAY_FACTOR
            print(f'  LR decay: {lr:.2e} -> {new_lr:.2e} (epoch {epoch})')
            return new_lr
        return lr

    return LearningRateScheduler(schedule, verbose=0)


def residual_block(
    x: tf.Tensor, # input
    filters: int,
    use_batchnorm: bool
) -> tf.Tensor:
    """
    Residual block of 2 convolutional layers with a skip connection for re-introducing
    spatial detail lost during downsampling (and avoiding vanishing gradients). CNNs 
    learn localized residuals (output - input) instead of entire mappings from scratch.
    """
    shortcut = x

    x = layers.Conv2D(
        filters, KERNEL_SIZE, padding='same', kernel_regularizer=l2(1e-4), use_bias=False
    )(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    # Keeps layers from collapsing into 1 (*TO DO: use layers.LeakyReLU instead?)
    x = layers.Activation('relu')(x) 

    x = layers.Conv2D(
        filters, KERNEL_SIZE, padding='same', kernel_regularizer=l2(1e-4), use_bias=False
    )(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)

    # If filter counts don't match, use 1x1 conv to match dims without distorting image 
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, kernel_size=1, padding='same', use_bias=False
        )(shortcut)

    # Combines original image and modifications before cleaning signal with another ReLU
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x


# ── Stage 1: Restoration ──
def simulate_motion_artifacts(clean: np.ndarray, severity: float) -> np.ndarray:
    """
    Degrades clean OCTA projections by simulating common OCTA acquisition 
    artifacts (shot noise, B-scan dropout, speckle) for model training.
    """
    img = clean.squeeze()
    H, W = img.shape

    # (1) Shot noise: fine, sharp static (random photon arrival fluctuations shaped like Poisson dist.)
    std_dev = severity * MOTION_AUG['shot_noise_scale'] * np.sqrt(img + 1e-6) # scales with sqrt(intensity)
    noisy = img + np.random.normal(0, 1, img.shape).astype(np.float32) * std_dev

    # (2) B-scan line dropout: smearing effect (horizontal bands/stripes from eye movement)
    n_dropped = int(H * severity * MOTION_AUG['bscan_dropout_rate'])
    for line in np.random.choice(H, n_dropped, replace=False): # copy from one row above and dim
        noisy[line, :] = noisy[max(0, line-1), :] * MOTION_AUG['bscan_dropout_severity'] 

    # (3) Speckle noise: grainy texture (follows multiplicative Rayleigh dist.)
    speckle = np.random.rayleigh(
        severity * MOTION_AUG['speckle_scale'], img.shape
    ).astype(np.float32)
    noisy = noisy * (1.0 + speckle) # scales with local (element-wise) intensity

    # Clips and re-normalizes to [0, 1]
    return np.expand_dims(np.clip(noisy, 0.0, 1.0), axis=-1)


def restoration_generator(
    img_dir: Path, 
    augment: bool
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yields degraded-clean image pairs for model training. Degradation is re-sampled
    every epoch so the model sees different noise patterns, which prevents overfitting!
    """
    img_files = sorted(img_dir.iterdir())
    n = len(img_files)

    while True:
        indices = list(range(n))
        if augment:
            random.shuffle(indices) # shuffles order every epoch

        # Process image data individually, then load into batches
        for i in range(0, n, BATCH_SIZE):
            batch_clean, batch_degraded = [], []
            for j in indices[i:min(i + BATCH_SIZE, n)]:
                raw = np.load(img_files[j]) # uint8 [0, 255]
                clean = raw.astype(np.float32) / 255.0 # float32 [0,1]
                if clean.ndim == 2: # ensure (H, W, 1)
                    clean = np.expand_dims(clean, axis=-1) # (H, W) -> (H, W, 1)

                if augment:
                    if np.random.rand() > 0.5: clean = np.fliplr(clean)
                    if np.random.rand() > 0.5: clean = np.flipud(clean)
  
                severity = np.random.uniform(
                    MOTION_AUG['severity_min'], MOTION_AUG['severity_max']
                )
                # Augmentations carried over from clean image
                degraded = simulate_motion_artifacts(clean, severity)

                # Scales Gaussian noise relative to image range [0, 1]
                gaussian = np.random.normal(
                    0, GAUSSIAN_NOISE_SIGMA / 255.0, degraded.shape
                ).astype(np.float32)
                degraded = np.clip(degraded + gaussian, 0.0, 1.0)

                batch_clean.append(clean) # (H, W, 1) -> (4, H, W, 1)
                batch_degraded.append(degraded)

            yield np.array(batch_degraded), np.array(batch_clean)


def build_restoration_model() -> Model:
    """
    ResU-Net for restoring degraded OCTA images with loss computed as 
    α x L2 + β x VGG19 content loss, where α=1, β=0.01 (Liao et al.).
    """
    # Heavy but popular pre-trained CNN from (V)isual (G)eometry (G)roup, Oxford
    from tensorflow.keras.applications import VGG19

    # Builds once then freezes weights
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
    vgg.trainable = False

    # *TO DO: instantiate VGG outside this func to load model elsewhere with compile=True?
    vgg_out = Model(
        inputs=vgg.input,
        # Liao et al. found this layer optimal for OCTA 
        outputs=vgg.get_layer('block5_conv4').output,
        name='vgg_feature_extractor'
    )

    def restoration_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes a combined loss of L2/mean squared error (pixel-level) and
        VGG19 perceptual (high-level) loss to check for similarities in both
        intensity and finer vessel texture between prediction and ground truth. 
        """
        l2_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # VGG19 expects 3-channel input (H, W, 3), scaled to [0, 255]
        y_true_rgb = tf.repeat(y_true, 3, axis=-1) * 255.0 
        y_pred_rgb = tf.repeat(y_pred, 3, axis=-1) * 255.0

        feat_true = vgg_out(y_true_rgb, training=False)
        feat_pred = vgg_out(y_pred_rgb, training=False)

        # Perceptual loss: computes mean squared error on feature maps
        content = tf.reduce_mean(tf.square(feat_true - feat_pred))

        return LOSS_ALPHA * l2_loss + LOSS_BETA * content

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 1))

    # Encoder: hierarchical feature extraction
    c1 = residual_block(
        layers.Conv2D(FILTERS[0], KERNEL_SIZE, padding='same')(inputs),
        # No BatchNorm when reconstructing images pixel-by-pixel (avoids washed out look)
        FILTERS[0], use_batchnorm=False 
    )
    p1 = layers.MaxPooling2D(SCALE_FACTOR)(c1) # 400->200

    c2 = residual_block(
        layers.Conv2D(FILTERS[1], KERNEL_SIZE, padding='same')(p1),
        FILTERS[1], use_batchnorm=False
    )
    p2 = layers.MaxPooling2D(SCALE_FACTOR)(c2) # 200->100

    c3 = residual_block(
        layers.Conv2D(FILTERS[2], KERNEL_SIZE, padding='same')(p2),
        FILTERS[2], use_batchnorm=False
    )
    p3 = layers.MaxPooling2D(SCALE_FACTOR)(c3) # 100->50

    # Bottleneck: richest feature representation at lowest resolution
    b = residual_block(
        layers.Conv2D(FILTERS[3], KERNEL_SIZE, padding='same')(p3),
        FILTERS[3], use_batchnorm=False
    )
    b = layers.Dropout(DROPOUT_RATE)(b)

    # Decoder: upsamples and concatenates skip connections from encoder feature maps
    u3 = layers.UpSampling2D(SCALE_FACTOR)(b) # 50->100
    c4 = residual_block(
        layers.Concatenate()([u3, c3]), FILTERS[2], use_batchnorm=False
    )
    
    u2 = layers.UpSampling2D(SCALE_FACTOR)(c4) # 100>200
    c5 = residual_block(
        layers.Concatenate()([u2, c2]), FILTERS[1], use_batchnorm=False
    )
    
    u1 = layers.UpSampling2D(SCALE_FACTOR)(c5) # 200->400
    c6 = residual_block(
        layers.Concatenate()([u1, c1]), FILTERS[0], use_batchnorm=False # (4, H, W, 16)
    )

    # 1-channel sigmoid (S-shaped curve) since pixel intensities already within [0, 1]
    output = layers.Conv2D(1, kernel_size=1, activation='sigmoid')(c6) # collapse 16 channels->1

    model = Model(inputs=inputs, outputs=output, name='restoration')
    model.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE,
            beta_1=ADAM_BETA1,   
            beta_2=ADAM_BETA2,  
            epsilon=EPSILON
        ),
        # Keras binds vgg_out to model's compiled loss wrapper
        loss=restoration_loss,
        metrics=[
            tf.keras.metrics.MeanSquaredError(name='mse'), # penalizes big errors only
            tf.keras.metrics.MeanAbsoluteError(name='mae'), # penalizes all errors proportionally
        ]
    )
    return model


def train_restoration_model() -> None:
    """Uses above functions to train restoration model."""
    img_train = OUTPUT_DIR / 'images' / 'train'
    img_valid = OUTPUT_DIR / 'images' / 'valid'

    n_train = count_files(img_train)
    n_valid = count_files(img_valid)

    steps_train = n_train // BATCH_SIZE
    steps_valid = n_valid // BATCH_SIZE

    print(f'\n=== STAGE 1: RESTORATION ===')
    print(f'Train: {n_train} images, ({steps_train} steps/epoch)')
    print(f'Valid: {n_valid} images, ({steps_valid} steps/epoch)')

    model = build_restoration_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            # Keep training as this ↓
            monitor='val_loss', mode='min', 
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            str(MODELS_DIR / 'restoration_best.keras'),
            # Overwrite checkpoint when it hits a new min
            monitor='val_loss', mode='min', 
            save_best_only=True, verbose=1
        ),
        TensorBoard(log_dir=str(MODELS_DIR / 'logs' / 'restoration')),
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
    print('*STAGE 1 COMPLETE: restoration model saved.')


# ── Stage 2: Segmentation ── 
def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Measures the spatial overlap between prediction and ground 
    truth, ranging from 0 (no overlap) to 1 (perfect match).
    """
    # Flattens (4, H, W, 1) into 1D columns
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / ( # formula
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + SMOOTH
    )


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Redefines Dice for optimizer (minimize vs. maximize) and 
    handles class imbalance by ignoring true negatives.
    """
    return 1.0 - dice_coefficient(y_true, y_pred)


def cl_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Measures spatial overlap of (c)enter(l)ines for evaluating 
    vessel continuity and connectivity (Shit et al. 2021).
    """
    def soft_skeleton(x: tf.Tensor) -> tf.Tensor:
        """
        Approximates skeletonization (reduces vessel thickness to 1-pixel-wide 
        lines) with max pooling, subtraction, and ReLU cause full skeletonization 
        is non-continuous/non-differentiable and breaks the optimizer.
        """
        s = x
        for _ in range(SKELETON_ITERS):
            # Shaves off a pixel from left and right vessel borders per iteration
            dilated = tf.nn.max_pool2d(s, ksize=KERNEL_SIZE, strides=1, padding='SAME')
            # Doubling original image "shields" vessel interior from getting scrapped
            s = tf.maximum(s - dilated + s, 0) 
        return s

    s_pred = soft_skeleton(y_pred)
    s_true = soft_skeleton(tf.cast(y_true, tf.float32))
    
    # Topology precision: how much do predicted and true skeletons overlap?
    t_prec = (tf.reduce_sum(s_pred * tf.cast(y_true, tf.float32)) + SMOOTH) / \
        (tf.reduce_sum(s_pred) + SMOOTH)
    
    # Topology sensitivity: how much of the true skeleton's captured?
    t_sens = (tf.reduce_sum(s_true * y_pred) + SMOOTH) / \
        (tf.reduce_sum(s_true) + SMOOTH)
    
    # clDice loss = 1 - clDice, like with regular Dice.
    return 1.0 - 2.0 * (t_prec * t_sens) / (t_prec + t_sens + EPSILON)


class DiceMetric(tf.keras.metrics.Metric):
    """Tracks per-epoch average Dice coefficient across batches per-epoch."""
    def __init__(self, name: str='dice', **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.dice_sum = self.add_weight(name='dice_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> None:
        # Flips pixel probabilities >50% to vessels (1.0)
        self.dice_sum.assign_add(
            dice_coefficient(y_true, tf.cast(y_pred > 0.5, tf.float32))
        )
        self.count.assign_add(1.0)

    def result(self) -> tf.Tensor:
        return self.dice_sum / self.count # mean

    def reset_states(self) -> None:
        self.dice_sum.assign(0.)
        self.count.assign(0.)

class IoUMetric(tf.keras.metrics.Metric):
    """
    Measures spatial overlap slightly more strictly by weighing 
    true positives, false positives, and false negatives equally.
    """
    def __init__(self, name: str='iou', **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.iou_sum = self.add_weight(name='iou_sum', initializer='zeros')
        self.count   = self.add_weight(name='iou_count', initializer='zeros')

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> None:
        # Flips pixel probabilities >50% to vessels (1.0)
        y_binarized = tf.cast(y_pred > 0.5, tf.float32)

        # Flattens (4, H, W, 1) into 1D columns
        yt = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        yp = tf.reshape(y_binarized, [-1])

        # Formula
        inter = tf.reduce_sum(yt * yp) # true positives
        union = tf.reduce_sum(yt) + tf.reduce_sum(yp) - inter
        self.iou_sum.assign_add((inter + SMOOTH) / (union + SMOOTH))
        self.count.assign_add(1.0)

    def result(self) -> tf.Tensor:
        return self.iou_sum / self.count # mean

    def reset_states(self) -> None:
        self.iou_sum.assign(0.)
        self.count.assign(0.)


def build_segmentation_model() -> Model:
    """
    ResU-Net for binary vessel segmentation, with loss computed as 
    Dice + ω x BCE, where ω=0.3.
    """
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 1))

    # Encoder: hierarchical feature extraction
    c1 = residual_block(
        layers.Conv2D(FILTERS[0], KERNEL_SIZE, padding='same')(inputs), 
        FILTERS[0], use_batchnorm=True # don't need exact pixel values, so use here 
    )
    p1 = layers.MaxPooling2D(SCALE_FACTOR)(c1) # 400->200

    c2 = residual_block(
        layers.Conv2D(FILTERS[1], KERNEL_SIZE, padding='same')(p1), 
        FILTERS[1], use_batchnorm=True
    )
    p2 = layers.MaxPooling2D(SCALE_FACTOR)(c2) # 200->100

    c3 = residual_block(
        layers.Conv2D(FILTERS[2], KERNEL_SIZE, padding='same')(p2), 
        FILTERS[2], use_batchnorm=True
    )
    p3 = layers.MaxPooling2D(SCALE_FACTOR)(c3) # 100->50

    # Bottleneck: richest feature representation at lowest resolution
    b = residual_block(
        layers.Conv2D(FILTERS[3], KERNEL_SIZE, padding='same')(p3), 
        FILTERS[3], use_batchnorm=True
    )
    b = layers.Dropout(DROPOUT_RATE)(b)

    # Decoder: upsamples and concatenates skip connections from encoder feature maps
    u3 = layers.UpSampling2D(SCALE_FACTOR)(b) # 50->100
    c4 = residual_block(
        layers.Concatenate()([u3, c3]), FILTERS[2], use_batchnorm=True
    )
    
    u2 = layers.UpSampling2D(SCALE_FACTOR)(c4) # 100->200
    c5 = residual_block(
        layers.Concatenate()([u2, c2]), FILTERS[1], use_batchnorm=True
    )
    
    u1 = layers.UpSampling2D(SCALE_FACTOR)(c5) # 200->400
    c6 = residual_block(
        layers.Concatenate()([u1, c1]), FILTERS[0], use_batchnorm=True
    )

    # 1-channel sigmoid (S-shaped curve) since binary
    output = layers.Conv2D(NUM_CLASSES, kernel_size=1, activation='sigmoid')(c6)

    model = Model(inputs=inputs, outputs=output, name='segmentation')
    model.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE,
            beta_1=ADAM_BETA1,
            beta_2=ADAM_BETA2,
            epsilon=EPSILON
        ),
        loss=lambda yt, yp: (
            # Lets Dice dominate while BCE prevents overconfident predictions and stabilizes gradients
            # without over-penalizing individual pixel errors and blurring vessel boundaries
            dice_loss(yt, yp) + BCE_WEIGHT * tf.keras.losses.BinaryCrossentropy()(yt, yp)
        ),
        metrics=[DiceMetric(), IoUMetric()]
    )
    return model

def segmentation_generator(
    img_dir: Path, 
    mask_dir: Path,
    augment: bool, 
    restoration_model: Model=None
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Yields image-mask pairs for model training. Images are from Stage 1."""
    # Only using these more complex transformations for Stage 2
    import imgaug.augmenters as iaa
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage

    img_files = sorted(img_dir.iterdir()) 
    mask_files = sorted(mask_dir.iterdir())
    n = len(img_files)

    seq = iaa.Sequential([
        iaa.Fliplr(GEO_AUG['flip_lr']),
        iaa.Flipud(GEO_AUG['flip_ud']),
        iaa.Sometimes(0.5, iaa.Affine(
            rotate=GEO_AUG['rotate_range'],
            scale=GEO_AUG['scale_range'],
            translate_percent=GEO_AUG['translate_range'],
            translate_percent={
                'x': GEO_AUG['translate_range'],
                'y': GEO_AUG['translate_range']
            }
        ))
    ])

    while True:
        indices = list(range(n))
        if augment:
            random.shuffle(indices) # shuffles order every epoch

        # Process image and mask data individually, then load into batches
        for i in range(0, n, BATCH_SIZE):
            batch_imgs, batch_masks = [], []
            for j in indices[i:min(i + BATCH_SIZE, n)]:
                img = np.load(img_files[j]).astype(np.float32) / 255.0 # uint8 [0, 255] -> float32 [0, 1]
                mask = np.load(mask_files[j]).astype(np.float32) # uint8 [0, 1] -> float32 [0, 1]

                img = np.expand_dims(img,  axis=-1) # (H, W) -> (H, W, 1)
                mask = np.expand_dims(mask, axis=-1)

                # Teach Stage 2 model how to segment images coming out of Stage 1
                if restoration_model is not None:
                    severity = np.random.uniform(
                        MOTION_AUG['severity_min'], MOTION_AUG['severity_max']
                    )
                    degraded = simulate_motion_artifacts(img, severity)
                    img = restoration_model.predict(
                        degraded[np.newaxis], verbose=0
                    )[0] # degraded -> restored

                if augment:
                    img_uint8 = (img * 255).astype(np.uint8) # imgaug needs uint8 [0, 255]
                    mask_binary = (mask > 0.5).astype(np.uint8) # flips pixel prob > 50% to vessels (1.0)

                    # Ensure image and mask get same transformations per pair
                    segmap = SegmentationMapsOnImage(mask_binary.squeeze(), shape=img.shape) # (H, W, 1) -> (H, W)
                    aug_img, aug_seg = seq(image=img_uint8, segmentation_maps=segmap)
                    img = aug_img.astype(np.float32) / 255.0 # ok, bring back to float32 [0, 1] now
                    mask = np.expand_dims(aug_seg.get_arr(), axis=-1).astype(np.float32)

                batch_imgs.append(img)
                batch_masks.append(mask)

            yield np.array(batch_imgs), np.array(batch_masks)


def train_segmentation_model() -> None:
    """Uses above functions to train segmentation model."""

    img_train  = OUTPUT_DIR / 'images' / 'train'
    img_valid  = OUTPUT_DIR / 'images' / 'valid'
    mask_train = OUTPUT_DIR / 'masks'  / 'train'
    mask_valid = OUTPUT_DIR / 'masks'  / 'valid'

    n_train = count_files(img_train)
    n_valid = count_files(img_valid)

    steps_train = n_train // BATCH_SIZE
    steps_valid = n_valid // BATCH_SIZE

    print(f'\n=== STAGE 2: SEGMENTATION ===')
    print(f'Train: {n_train} images, ({steps_train} steps/epoch)')
    print(f'Valid: {n_valid} images, ({steps_valid} steps/epoch)')

    # Load restoration model if connecting pipeline
    rest_model = None
    rest_model_path = MODELS_DIR / 'restoration_best.keras'
    if rest_model_path.exists():
        rest_model = tf.keras.models.load_model(
            str(rest_model_path), compile=False
        )
        rest_model.trainable = False  # frozen: inference only
        print('*PIPELINE CONNECTED: training segmentation on restored images.')
    else:
        print('*WARNING: \'restoration_best.keras\' not found...'
            'Training segmentation on clean images directly.')

    model = build_segmentation_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            # Keep training as this ↑ 
            monitor='val_dice', mode='max', 
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            str(MODELS_DIR / 'segmentation_best.keras'),
            # Overwrite checkpoint when it hits a new max
            monitor='val_dice', mode='max',
            save_best_only=True, verbose=1
        ),
        TensorBoard(log_dir=str(MODELS_DIR / 'logs' / 'segmentation')),
        make_lr_scheduler(steps_train),
    ]

    model.fit(
        segmentation_generator(
            img_train, mask_train, augment=True, 
            restoration_model=rest_model
        ),
        steps_per_epoch=steps_train,
        validation_data=segmentation_generator(
            img_valid, mask_valid, augment=False, 
            restoration_model=rest_model
        ),
        validation_steps=steps_valid,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    model.save(MODELS_DIR / 'segmentation_final.keras')
    print('*STAGE 2 COMPLETE: segmentation model saved.')


if __name__ == '__main__':
    train_restoration_model()
    train_segmentation_model()
    pass