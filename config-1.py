"""
config.py
=========
Central configuration for the OCTA restoration + segmentation pipeline.

Pipeline (Option C — restore then segment):
    Degraded OCTA → [Stage 1 ResU-Net] → Clean OCTA → [Stage 2 ResU-Net] → Vessel mask

Dataset: OCTA-500 + ROSE combined
    OCTA-500: 500 retinal OCTA, 400×400, ILM_OPL slab + GT_LargeVessel masks
    ROSE:     117 retinal OCTA, 304×304, capillary-level vessel annotations
    Combined: more varied noise profiles, finer vessel detail, better generalisation

Changes from previous version (post Liao et al. deep read):
    - ADAM_BETA1: 0.9 → 0.8  (Liao et al. Section 4.1, explicitly cited)
    - ADAM_BETA2: default → 0.999 (Liao et al. Section 4.1)
    - LR_DECAY_FACTOR: added 0.95 every 10k steps (Liao et al. Section 4.1)
    - GAUSSIAN_NOISE_SIGMA: added 0.4 on top of simulated degradation
      (Liao et al. Section 4.1 — "Gaussian noise σ=0.4 applied to inputs
       to enhance generalisation and simulate shot noise from photon detector")
    - BATCH_SIZE: 8 → 4 (matching Liao et al.; also safer on Colab T4)
    - EPOCHS: 100 → 400 (Liao et al. used 400; early stopping handles termination)
    - EARLY_STOP_PATIENCE: 10 → 20 (Liao et al. used 20)
"""


# ── Optimiser ─────────────────────────────────────────────────────────────────


# Adam beta1: 0.8 — directly from Liao et al. Section 4.1
# Lower than default 0.9: "forgets" gradient history faster
# Useful for medical image data where gradient direction changes more rapidly
# than in natural image tasks
ADAM_BETA1 = 0.8

# Adam beta2: 0.999 — directly from Liao et al. Section 4.1
# Controls second moment (variance) estimate decay rate
ADAM_BETA2 = 0.999

# Learning rate decay: multiply by 0.95 every 10,000 training steps
# Directly from Liao et al. Section 4.1
# Prevents oscillation in late training when loss is near minimum
# Implemented via LearningRateScheduler callback in model.py
LR_DECAY_FACTOR = 0.95
LR_DECAY_STEPS  = 10_000


# ── Training ──────────────────────────────────────────────────────────────────

# Batch size: 4 matches Liao et al. exactly; safe on Colab T4 at 400×400
# Previous value of 8 risks OOM when VGG19 feature extractor is loaded
# for content loss computation during Stage 1
BATCH_SIZE = 4

# Max epochs: 400 matches Liao et al.
# EarlyStopping (patience=20) will terminate before this in practice
EPOCHS = 400

# Early stopping patience: 20 matches Liao et al. exactly
# Previous value of 10 was too aggressive —
# natural loss oscillations can cause 10-epoch flat stretches
EARLY_STOP_PATIENCE = 20

# BCE weight in combined segmentation loss:
#   Total loss = Dice + BCE_WEIGHT × BinaryCrossentropy
# 0.3: Dice dominates (handles class imbalance), BCE stabilises gradients
# Previous value 0.5 over-penalised individual pixel errors → blurry masks
BCE_WEIGHT = 0.3


# ── Gaussian Noise Augmentation (Stage 1 inputs only) ────────────────────────

# Sigma for Gaussian noise added to degraded inputs during restoration training
# Directly from Liao et al. Section 4.1:
#   "Gaussian noise with σ=0.4 was applied to the input images in training
#    datasets to enhance the generalisation of the trained network and
#    simulate the shot noise generated from the balance photon detector"
# Applied ON TOP of the physics-based motion artifact simulation
# Helps the model generalise to noise levels slightly outside the training
# distribution — critical since Andrei's microfluidic data will have
# different noise characteristics than OCTA-500
GAUSSIAN_NOISE_SIGMA = 0.4


# ── Motion Artifact Simulation (Stage 1 degradation) ─────────────────────────

# Physical basis: real OCTA acquires 4-12 repeated B-scans per position
# Using only 2 repeats reduces SNR by sqrt(NR) factor and introduces:
#   1. Shot noise: photon arrival fluctuations, scales with sqrt(intensity)
#   2. B-scan dropout: horizontal stripes from eye/organism movement
#   3. Speckle noise: coherent interference, multiplicative Rayleigh noise
MOTION_AUG = {
    'severity_min':      0.15,  # mild: few-repeat acquisition
    'severity_max':      0.40,  # severe: 2-repeat acquisition
    'shot_noise_scale':  0.3,
    'bscan_dropout_rate': 0.4,
    'speckle_scale':     0.2,
}


# ── Geometric Augmentation (Stage 2 segmentation) ────────────────────────────

AUG_CONFIG = {
    'flip_lr':       0.5,
    'flip_ud':       0.5,
    'rotate_range':  (-10, 10),
    'shear_range':   (0, 0),
    'scale_range':   (0.95, 1.05),
    'translate_range': (0.1, -0.1),
    'elastic_alpha': 0,
    'elastic_sigma': 0,
}
