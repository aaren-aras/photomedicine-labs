from pathlib import Path
from typing import Generator

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.python.platform import build_info as tf_build_info
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
import imgaug.augmenters as iaa 
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from config import (
    IMG_SIZE, MODALITIES, NUM_CLASSES, FILTERS, KERNEL_SIZE, SCALE_FACTOR, 
    DROPOUT_RATE, LEARNING_RATE, EPSILON, BATCH_SIZE, AUG_CONFIG, EPOCHS
)

# Use GPU for faster training (if available)
physical_devices = tf.config.list_physical_devices('GPU')
print('*AVAILABLE GPUs:', physical_devices)
if physical_devices: 
    tf.config.experimental.set_memory_growth(physical_devices[0], True) # allocate memory incrementally, instead of all at once
    
    # mixed_precision.set_global_policy('mixed_float16') # faster training, lower memory usage
    # print('*MIXED PRECISION POLICY:', mixed_precision.global_policy())

    build_info = tf.sysconfig.get_build_info()
    cuda_version = build_info.get('cuda_version', 'Unknown')
    cudnn_version = build_info.get('cudnn_version', 'Unknown')
    cuda_compute_capabilities = tf_build_info.build_info.get('cuda_compute_capabilities')

    print(f'*CUDA TOOLKIT VERSION: {cuda_version}')
    print(f'*cuDNN VERSION: {cudnn_version}')
    print(f'*TF CUDA COMPUTE CAPABILITIES:, {cuda_compute_capabilities}')

    if cuda_version == 'Unknown' or cudnn_version == 'Unknown':
        print('*WARNING: CUDA or cuDNN version information unavailable. Verify your TensorFlow GPU setup...')


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float=EPSILON*100) -> tf.Tensor: 
    """
    Compute the Dice coefficient between two tensors (ground truth and prediction masks), ranging from 0 (no overlap) to 1 (perfect match).
    """
    y_true_f = tf.reshape(y_true, [-1]) # (H, W, 4) -> 1D, i.e., (240, 240, 4) = 240 x 240 x 4 = [2300400] elements
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f) # sum of element-wise product (common 1s between masks)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth) # Dice formula


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute the Dice loss (1 - Dice coefficient) for training purposes."""
    return 1 - dice_coefficient(y_true, y_pred)


class DiceMetric(tf.keras.metrics.Metric):
    """
    Custom Keras abstract base class for tracking average Dice coefficient across batches.
    """
    def __init__(self, name='dice', **kwargs):
        super(DiceMetric, self).__init__(name=name, **kwargs)
        self.dice = self.add_weight(name='dice', initializer='zeros') # cumulative Dice score
        self.count = self.add_weight(name='count', initializer='zeros') # cumulative num batches

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor=None):
        """
        Update the state with current batch's Dice score.
        """
        dice = dice_coefficient(y_true, y_pred)
        self.dice.assign_add(dice)
        self.count.assign_add(1.0)

    def result(self):
        """
        Return the average Dice score so far.
        """
        return self.dice / self.count

    def reset_states(self):
        """
        Reset all states for fresh tracking at the start of each epoch.
        """
        self.dice.assign(0)
        self.count.assign(0)


def residual_block(x: tf.Tensor, filters: int, kernel_size: int=KERNEL_SIZE) -> tf.Tensor:
    """
    Create a residual block comprising 2 convolutional layers plus a skip connection, allowing gradients to 
    flow backwards (reuse earlier layers) without vanishing (shrinking) during backpropogation.
    """
    shortcut = x

    x = layers.Conv2D(filters, kernel_size, padding='same', activation=None)(x) # capture spatial features with sliding window 
    x = layers.BatchNormalization()(x) # normalize feature maps to speed up training
    x = layers.Activation('relu')(x) # (Re)ctified (L)inear (U)nit: -ve values -> 0 (bend at x=0 induces non-linearity)

    x = layers.Conv2D(filters, kernel_size, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters: # ensure tensor dims (num channels) match before adding
        shortcut = layers.Conv2D(filters=filters, kernel_size=1, padding='same')(shortcut)

    x = layers.Add()([shortcut, x]) # skip connection: add original input (shortcut) to processed output (x) BEFORE ReLU
    x = layers.Activation('relu')(x)
    return x


def build_segmentation_model(input_shape: tuple[int, int, int]=(*IMG_SIZE, len(MODALITIES)), num_seg_classes: int=NUM_CLASSES) -> Model:
    """
    Build a ResU-Net for multi-class image segmentation with softmax output (4 classes).
    """
    inputs = tf.keras.Input(shape=input_shape)

    # (1) Encoder: extract hierarchical features (low-level (edges, corners) -> high-level (tumour outlines, organ boundaries)) and downsample resolution by 1/2
    c1 = residual_block(layers.Conv2D(FILTERS[0], KERNEL_SIZE, padding='same')(inputs), FILTERS[0]) # preserve spatial res. with padding='same': (H, W) -> (H, W)
    p1 = layers.MaxPooling2D(SCALE_FACTOR)(c1) # (H, W) -> (H/2, W/2)

    c2 = residual_block(layers.Conv2D(FILTERS[1], KERNEL_SIZE, padding='same')(p1), FILTERS[1])
    p2 = layers.MaxPooling2D(SCALE_FACTOR)(c2)

    c3 = residual_block(layers.Conv2D(FILTERS[2], KERNEL_SIZE, padding='same')(p2), FILTERS[2])
    p3 = layers.MaxPooling2D(SCALE_FACTOR)(c3)

    # (2) Bottleneck: extract richest (high-level) features, then zero out some to reduce overfitting
    b = residual_block(layers.Conv2D(FILTERS[3], KERNEL_SIZE, padding='same')(p3), FILTERS[3])
    b = layers.Dropout(DROPOUT_RATE)(b)

    # (3) Decoder: upsample resolution back by 2 and link skip connections to refine earlier features (high-level + low-level)
    u3 = layers.UpSampling2D(SCALE_FACTOR)(b) # (H, W) -> (H*2, W*2)
    c4 = residual_block(layers.Concatenate()([u3, c3]), FILTERS[2])

    u2 = layers.UpSampling2D(SCALE_FACTOR)(c4)
    c5 = residual_block(layers.Concatenate()([u2, c2]), FILTERS[1])

    u1 = layers.UpSampling2D(SCALE_FACTOR)(c5)
    c6 = residual_block(layers.Concatenate()([u1, c1]), FILTERS[0])

    # Compute class probabilities for each pixel (e.g., [2.5, 0.3, 1.1, 3.2] -> [0.28, 0.03, 0.07, 0.62], i.e., 62% chance it's ET > 28% chance its bkgd > ...)
    seg_output = layers.Conv2D(num_seg_classes, kernel_size=1, activation='sigmoid', name='segmentation')(c6) # 'softmax': raw output scores (logits) -> probability dist
    model = Model(inputs=inputs, outputs=seg_output) # define (trainable) Keras model obj from computation graph: input->output tensor flow (hehe)

    # Use gradients (of error func w.r.t. weights) from backpropogation to adjust each weight based on its contribution to prediction error
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE), # how to learn (adaptive learning rate + fast convergence (stable accuracy))
        loss=lambda y_true, y_pred: dice_loss(y_true, y_pred) + tf.keras.losses.BinaryCrossentropy()(y_true, y_pred), # how to measure error (spatial overlap + class prob error)
        metrics=[DiceMetric()] # what to watch (segmentation performance)
    )

    return model


def data_generator(
    img_dir: str, 
    mask_dir: str, 
    augment: bool = False, 
    batch_size: int=BATCH_SIZE, 
    num_classes: int=NUM_CLASSES
    # Generator[yield_type, send_type, return_type] where send_type = type(values received mid-iteration), return_type = type(values received once terminated)
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]: 
    """
    Custom on-the-fly generator for multi-class MRI segmentation training that:
    - Loads .npy image and mask files saved in 'data.py',
    - Applies mask-aware (synchronized) augmentation on training data,
    - One-hot encodes masks for categorical_crossentropy,
    - Yields batches of image-mask pairs for model.fit().
    """
    img_files = sorted(img_dir.iterdir())
    mask_files = sorted(mask_dir.iterdir())
    num_samples = len(img_files)

    # Define augmentation pipeline + probabilities (e.g., 0.5 = 50% chance to apply)
    seq = iaa.Sequential([
        iaa.Fliplr(AUG_CONFIG['flip_lr']), # horizontal flip
        iaa.Flipud(AUG_CONFIG['flip_ud']), # vertical flip
        iaa.Sometimes(0.5, iaa.Affine(
            rotate=(AUG_CONFIG['rotate_range']), 
            shear=(AUG_CONFIG['shear_range']), 
            scale=(AUG_CONFIG['scale_range']), 
            translate_percent={'x': (AUG_CONFIG['translate_range']), 'y': (AUG_CONFIG['translate_range'])} 
        )),
        # Simulate deformations (↑ alpha ↑ distortion strength, ↑ sigma ↑ smoothness)
        iaa.ElasticTransformation(alpha=AUG_CONFIG['elastic_alpha'], sigma=AUG_CONFIG['elastic_sigma']) 
    ])

    while True: # stops when model.fit() stops it (via steps_per_epoch)
        for i in range(0, num_samples, batch_size):
            batch_imgs = []
            batch_masks = []

            for j in range(i, min(i + batch_size, num_samples)): # avoid going over end of dataset 
                img = np.load(img_dir / img_files[j]).astype(np.float32) / 255.0 # uint8 [0, 255] -> float32 [0,1]
                mask = np.load(mask_dir / mask_files[j]) # uint8 [0, 255]

                img = np.expand_dims(img, axis=-1)    # (H, W, 1)
                mask = np.expand_dims(mask, axis=-1)  # (H, W, 1)

                if augment:
                    uint8_img = (img * 255).astype(np.uint8) # float32 [0, 1] -> uint8 [0, 255] (for imgaug)
                    
                    # Convert mask to segmentation map to match augmentation with corresponding img
                    segmap = SegmentationMapsOnImage(mask.squeeze(), shape=img.shape[:2]) # (H, W, 4) img -> (H, W) img ~ (H, W) mask

                    # Apply augmentation pipeline to both image and mask
                    aug_img, aug_segmap = seq(image=uint8_img, segmentation_maps=segmap)
                    img = aug_img.astype(np.float32) / 255.0 # uint8 [0, 255] -> float32 [0,1] (convert back for model)
                    mask = np.expand_dims(aug_segmap.get_arr(), axis=-1)

                    
                    # mask_to_encode = aug_segmap.get_arr() # SegmentationMapsOnImage obj -> arr
                    # mask_to_encode = np.clip(mask_to_encode, 0, num_classes=-1)
                
                batch_imgs.append(img)
                batch_masks.append(mask)

            yield np.array(batch_imgs), np.array(batch_masks)


def train_model() -> None:
    """
    Train a 2D U-Net segmentation model on BraTS data and save it for later inference.
    """
    model = build_segmentation_model()
    MODELS_DIR = Path(__file__).resolve().parent.parent / 'models' # oncer/api/models
    MODELS_DIR.mkdir(exist_ok=True)

    # Define subsubdirectories from 'data.py'
    IMG_TRAIN_DIR = OUTPUT_DIR / 'images' / 'train'
    IMG_VALID_DIR = OUTPUT_DIR / 'valid'
    MASK_TRAIN_DIR = OUTPUT_DIR / 'masks' / 'train'
    MASK_VALID_DIR = OUTPUT_DIR / 'masks' / 'valid'

    # Initialize data generators for one-hot encoding and augmentation during model.fit()
    train_gen = data_generator(IMG_TRAIN_DIR, MASK_TRAIN_DIR, augment=True)
    valid_gen = data_generator(IMG_VALID_DIR, MASK_VALID_DIR, augment=False)

    # Set num batches to process per epoch
    train_steps = len(list(IMG_TRAIN_DIR.iterdir())) // BATCH_SIZE # generators don't have lengths
    valid_steps = len(list(IMG_VALID_DIR.iterdir())) // BATCH_SIZE

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(Path(MODELS_DIR / 'oncer_model_checkpoint.keras'), monitor='val_loss', save_best_only=True, verbose=1),
        TensorBoard(log_dir=Path(MODELS_DIR / 'logs'))
    ]

    # Train model on BraTS dataset ((1) Forward Pass -> (2) Loss Calculation -> (3) Backward Pass/Backpropogation -> (4) Weight Update)
    model.fit(
        train_gen, # unpacks to (x, y) = (np.array(batch_imgs), np.array(batch_masks))
        steps_per_epoch=train_steps, # calls next(train_gen) under the hood steps_per_epoch num times
        validation_data=valid_gen,
        validation_steps=valid_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    model.save(Path(MODELS_DIR / 'octa_model.keras'))
    print(f"*COMPLETE: model has been trained and saved to 'models' folder")


if __name__ == '__main__':
    train_model()