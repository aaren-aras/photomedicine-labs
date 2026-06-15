"""WHAT DOES INCREASING/DECREASING EACH OF THESE DO"""
""" [*] values borrowed from Liao et al. Section 4.1 (see docs). """

# ── Data ──
CLIP_LIMIT = 2.0 # limits noise over-amp. by redist. excess pixels from saturated histogram bins (e.g., bkgd)
TILE_GRID_SIZE = (8, 8) # divides 400x400 image into 64 tiles (8x8) of 50x50 px each
PROJECTIONS = {
    'octa500': 'OCTA(ILM_OPL)', # (I)nner (L)imiting (M)embrane to (O)uter (P)lexiform (L)ayer slab
    'rose1': 'SVC_DVC' # (S)uperficial (V)ascular (C)omplex and (D)eep (V)ascular (C)omplex
}
OCTA500_LABEL = 'GT_LargeVessel' # (G)round (T)ruth label (data > OCTA-500 > Label)
BINARIZE_MASK = True # [0, 255] -> [0, 1] for BCE and Dice loss
RANDOM_STATE = 28 # reproducible train/valid/test split
IMG_SIZE = (400, 400) # (H, W): OCTA-500 maintained (400x400), ROSE upscaled (304x304/840x840 -> 400x400)
MODALITIES = ('OCTA',) # 1-channel greyscale OCTA projections (*TO DO: modality per OCTA slab?)
NUM_CLASSES = 1 # binary segmentation for BinaryCrossentropy() and dice_loss(): vessel (1) vs. bkgd (0)

# ── Augmentation ──
GAUSSIAN_NOISE_SIGMA = 0.4 # on top of simulated degradation to help model generalize shot noise [*]
MOTION_AUG = { # stage 1: restoration
    'severity_min': 0.30, # ~few-repeat acquisition 
    'severity_max': 0.70, # ~2-repeat acquisition 
    'shot_noise_scale': 0.3, 
    'bscan_dropout_rate': 0.4, 
    'bscan_dropout_severity': 0.4, # % signal loss (1 for solid black line)
    'speckle_scale': 0.2, 
}
GEO_AUG = { # stage 2: segmentation
    'flip_lr': 0.5, # % chance to occur
    'flip_ud': 0.5, # % chance to occur
    'rotate_range': (-10, 10), # deg
    'shear_range': (0, 0), # deg
    'scale_range': (0.95, 1.05), # %
    'translate_range': (-0.1, 0.1), # %
}

# ── Model ── 
LOSS_ALPHA = 1 # L2/pixel loss weight [*]
LOSS_BETA = 0.01 # VGG19/perceptual loss weight [*]
SMOOTH = 1 # avoid dividing by 0 for Dice classes (namely for earlier gradients)
THRESHOLD = 0.5 # cutoff probability above which a px is classified as a vessel
SKELETON_ITERS = 5 # ~OCTA-500 vessel half-width in px
FILTERS = (16, 32, 64, 128) # counts per encoder stage (↑ this ⇒ ↑ vessel detail, ↑ overfitting)
KERNEL_SIZE = 3 # filter size for skeleton erosion and square matrix convolution in px 
SCALE_FACTOR = 2 # pool/upsample per encoder and decoder stage (↑ this ⇒ ↓ vessel detail)
DROPOUT_RATE = 0.3 # % neurons zeroed at bottleneck and decoder stages to prevent overfitting

# ── Optimizer ── 
LEARNING_RATE = 1e-4 # small gradient steps for fine-tuning on sparser vessel structures [*]
EPSILON = 1e-8 # avoid dividing by 0 for clDice, float32 on small gradients, test metrics
ADAM_BETA1 = 0.8 # momentum: forget gradient history faster [*]
ADAM_BETA2 = 0.999 # step size confidence: go slower on bigger gradients [*]
LR_DECAY_FACTOR = 0.95 # prevent oscillation in late training for near-min loss [*]
LR_DECAY_STEPS  = 10000 # [*]

# ── Training ── 
BATCH_SIZE = 4 # num samples (image + mask) per step before updating weights; keep low to avoid OOM [*]
EPOCHS = 400 # full passes through entire dataset, let early stopping end training
EARLY_STOP_PATIENCE = 20 # [*]
BCE_WEIGHT = 0.3 # empirical adjustment to balance out BinaryCrossentropy and Dice



# Sigma for Gaussian noise added to degraded inputs during restoration training
# Applied ON TOP of the physics-based motion artifact simulation
# different noise characteristics than OCTA-500

# adam_beta1 Useful for medical image data where gradient direction changes more rapidly
# adam_beta2 # Controls second moment (variance) estimate decay rate

"""
Central configuration for the OCTA restoration + segmentation pipeline.

Pipeline (Option C — restore then segment):
    Degraded OCTA → [Stage 1 ResU-Net] → Clean OCTA → [Stage 2 ResU-Net] → Vessel mask

Dataset: OCTA-500 + ROSE combined
    OCTA-500: 500 retinal OCTA, 400×400, ILM_OPL slab + GT_LargeVessel masks
    ROSE:     117 retinal OCTA, 304×304, capillary-level vessel annotations
    Combined: more varied noise profiles, finer vessel detail, better generalisation
"""


"""

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


Dataset: OCTA-500 (https://ieee-dataport.org/open-access/octa-500)
  - 500 retinal OCTA volumes from 500 subjects
  - ILM_OPL slab: en face projection from Inner Limiting Membrane to
    Outer Plexiform Layer — this is the standard slab for large vessel
    visualization as it captures the superficial + deep retinal plexuses
  - GT_LargeVessel: binary ground truth masks for large vessel segmentation
"""

