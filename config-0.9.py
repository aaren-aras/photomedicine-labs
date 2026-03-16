"""
config.py
=========
Central configuration for the OCTA vessel segmentation + restoration pipeline.

Pipeline overview (Option C — restore THEN segment):
    Raw OCTA image
        │
        ▼
    [Stage 1] ResU-Net Restoration
        Degraded OCTA → Clean OCTA  (PSNR / SSIM metrics)
        Trained on: clean images + on-the-fly simulated motion artifacts
        │
        ▼
    [Stage 2] ResU-Net Segmentation  
        Clean OCTA → Binary vessel mask  (Dice / IoU / clDice metrics)
        Trained on: OCTA-500 ILM_OPL projections + GT_LargeVessel labels
        │
        ▼
    Eventually: fine-tune on Andrei's chicken embryo OCTA data

Dataset: OCTA-500 (https://ieee-dataport.org/open-access/octa-500)
  - 500 retinal OCTA volumes from 500 subjects
  - ILM_OPL slab: en face projection from Inner Limiting Membrane to
    Outer Plexiform Layer — this is the standard slab for large vessel
    visualization as it captures the superficial + deep retinal plexuses
  - GT_LargeVessel: binary ground truth masks for large vessel segmentation
"""

# ── Dataset / Label ───────────────────────────────────────────────────────────

# Which OCTA-500 ground truth label to train on.
# Options: 'GT_LargeVessel', 'GT_Artery', 'GT_Vein', 'GT_FAZ'
# We use LargeVessel because:
#   (a) it's the most clinically relevant for vascular assessment
#   (b) it's the most transferable to Andrei's chicken embryo data where
#       we also expect to segment large, prominent vessels
LABEL = 'GT_LargeVessel'

# Convert mask pixel values to binary 0/1.
# OCTA-500 masks contain values {0, 255} — binarizing maps these to {0, 1}
# which is required for BinaryCrossentropy and Dice loss to work correctly.
BINARIZE_MASK = True

# Seed for reproducible train/valid/test splits across runs
RANDOM_STATE = 28


# ── Image ─────────────────────────────────────────────────────────────────────

# OCTA-500 images are 400×400 pixels (en face projections)
# This must match the actual image dimensions — changing this requires
# re-running data.py to regenerate the .npy files at the new size
IMG_SIZE = (400, 400)

# Single-channel grayscale OCTA projections (no colour, no multi-modal stacking)
# In the future this could include multiple OCTA slabs (e.g., ILM_OPL + OPL_BM)
# as separate channels, similar to how BraTS uses T1/T2/FLAIR/T1ce
MODALITIES = ('OCTA',)

# Binary segmentation: vessel (1) vs background (0)
# Not multi-class — OCTA-500 large vessel masks are binary
NUM_CLASSES = 1


# ── Model Architecture ────────────────────────────────────────────────────────

# Filter counts double with each encoder stage (standard U-Net convention).
# Starting at 16 keeps parameter count manageable for our dataset size (~400 images).
# Doubling: 16 → 32 → 64 → 128 (bottleneck)
# Rule of thumb: start small, increase if val_dice plateaus and you're not overfitting.
FILTERS = (16, 32, 64, 128)

# 3×3 is the standard conv kernel for image segmentation:
#   - Large enough to capture local texture (vessel edges, bifurcations)
#   - Small enough to stack efficiently (two 3×3 convs ≈ one 5×5 receptive field)
KERNEL_SIZE = 3

# Pooling/upsampling factor — halves spatial dimensions each encoder stage.
# 2× is standard for U-Net. Using 4× would lose too much spatial detail
# for thin vessel structures.
SCALE_FACTOR = 2

# Dropout applied at bottleneck and decoder stages to prevent overfitting.
# 0.3 = 30% of neurons randomly zeroed per forward pass during training.
# Higher values (0.5) are common for larger datasets; 0.3 is appropriate
# for our ~320 training images.
DROPOUT_RATE = 0.3


# ── Training ──────────────────────────────────────────────────────────────────

# Adam optimizer learning rate.
# 1e-4 is the standard starting point for medical image segmentation with Adam.
# 5e-4 (previous value) caused unstable loss oscillations — too large for the
# small gradient steps needed when fine-tuning on sparse vessel structures.
# Reference: Liao et al. 2023 use 1e-4 for OCTA reconstruction with Adam.
LEARNING_RATE = 1e-4

# Adam epsilon — prevents division by zero in parameter updates.
# Default 1e-7 can cause instability with float32 on small gradients.
# 1e-8 is a safer, commonly used value for medical imaging tasks.
EPSILON = 1e-8

# Batch size: number of image+mask pairs processed before a weight update.
# 8 is the maximum safe batch size for 400×400 images on Colab T4 (16GB VRAM).
# At batch=16 (previous value) with float32: 16 × 400 × 400 × 1 × 4 bytes ≈ 1GB
# just for images, but feature maps in the bottleneck multiply this ~8×, causing OOM.
# If training locally with more VRAM, can increase to 16.
BATCH_SIZE = 8

# Total training epochs. EarlyStopping (patience=10) will terminate before
# this limit if val_loss stops improving, so this is effectively a ceiling.
EPOCHS = 100

# Weight of BinaryCrossentropy term in the combined segmentation loss:
#   Total loss = Dice loss + BCE_WEIGHT × BinaryCrossentropy
# Dice loss handles class imbalance well (vessels are ~10-15% of pixels).
# BCE provides pixel-wise probability calibration.
# 0.3 downweights BCE to let Dice dominate — this balance was found optimal
# for binary vessel segmentation tasks in the literature.
# Previous value of 0.5 over-penalised individual pixel errors and caused
# the model to predict conservative, blurry masks.
BCE_WEIGHT = 0.3


# ── Motion Artifact Augmentation (Stage 1 — Restoration) ─────────────────────
# These parameters control the on-the-fly degradation applied to clean OCTA
# images during restoration model training, simulating real-world acquisition
# limitations (fewer B-scan repeats, eye motion, blinking).
#
# Physical basis (from Liao et al. 2023, Das et al. 2025):
#   - Real OCTA requires 4-12 repeated B-scans per location; using only 2
#     repeats introduces shot noise and reduces SNR by factor √(NR)
#   - Eye movement between repeat acquisitions causes horizontal banding
#     (the white stripe artifacts visible in Andrei's B-scan images)
#   - Speckle noise is an inherent coherent noise pattern in all OCT systems
MOTION_AUG = {
    'severity_min': 0.15,   # minimum degradation severity (mild artifacts)
    'severity_max': 0.40,   # maximum degradation severity (severe artifacts)
    'shot_noise_scale': 0.3,  # shot noise amplitude relative to severity
    'bscan_dropout_rate': 0.4,  # fraction of B-scan lines affected by motion
    'speckle_scale': 0.2,   # speckle noise amplitude relative to severity
}


# ── Geometric Augmentation (Stage 2 — Segmentation) ──────────────────────────
# Applied to both image AND mask synchronously during segmentation training.
# Purpose: artificially expand the ~320-image training set and improve
# generalization to Andrei's chicken embryo data (different vessel geometries).
AUG_CONFIG = {
    # Horizontal flip: 50% chance. Retinal vessel patterns have no meaningful
    # left/right asymmetry, so this doubles effective dataset size for free.
    'flip_lr': 0.5,

    # Vertical flip: 50% chance. Same reasoning as horizontal flip.
    'flip_ud': 0.5,

    # Rotation range in degrees. ±10° simulates variability in scan orientation.
    # Larger rotations (±30°) risk rotating vessels out of the image boundary.
    'rotate_range': (-10, 10),

    # Shear: disabled (0). Shear can distort vessel widths unrealistically.
    'shear_range': (0, 0),

    # Scale: ±5% zoom. Simulates slight variability in imaging distance/FOV.
    'scale_range': (0.95, 1.05),

    # Translation: ±10% shift. Simulates variability in scan centering.
    'translate_range': (0.1, -0.1),

    # Elastic deformation: disabled for now.
    # When enabled, simulates tissue deformation (useful for histology, less so
    # for OCTA where vessels maintain rigid topology). Can re-enable with
    # alpha=50, sigma=5 if the model struggles with curved vessel segments.
    'elastic_alpha': 0,
    'elastic_sigma': 0,
}
