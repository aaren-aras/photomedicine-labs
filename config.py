''' D A T A S E T '''
# Choose OCTA-500 (G)round (T)ruth label
LABEL = 'GT_LargeVessel'
# Do [0, 255] -> [0, 1] 
BINARIZE_MASK = True 
# Hold seed for data split
RANDOM_STATE = 28 
# Fix OCTA-500 400x400 and upscale ROSE 304x304->400x400
IMG_SIZE = (400, 400) 
# Keep 1-channel greyscale OCTA projections (*TO DO: modality per OCTA slab? ILM_OPL vs. OPL_BM)
MODALITIES = ('OCTA',) 
# Do binary segmentation for BinaryCrossentropy() and dice_loss(): vessel (1) vs. bkgd (0)
NUM_CLASSES = 1 

''' M O D E L '''
# Progress U-net encoders
FILTERS = (16, 32, 64, 128) 
# Select conv. filter matrix size
KERNEL_SIZE = 3 
# Pool/Upsample per stage
SCALE_FACTOR = 2 
# Deactivate some neurons @ bottleneck: prevent overfitting or underfitting
DROPOUT_RATE = 0.3 

''' O P T I M I Z E R '''
# Use small gradient steps for sparser vessel structures (Liao et al. Section 4.1)
LEARNING_RATE = 1e-4
# Avoid dividing by 0 
EPSILON = 1e-8 
# Forget gradient history faster (Liao et al. Section 4.1)
ADAM_BETA1 = 0.8 
# Go slower on bigger gradients (Liao et al. Section 4.1)
ADAM_BETA2 = 0.999 
# Prevent oscillation in late training for near-min loss
LR_DECAY_FACTOR = 0.95 
#
LR_DECAY_STEPS  = 10_000


''' T R A I N I N G '''

BATCH_SIZE = 16 # num samples (image + mask) per training step (before updating weights)
EPOCHS = 50 # full passes through entire dataset
AUG_CONFIG = {
    'flip_lr': 0.5, # % chance to occur
    'flip_ud': 0.5, # % chance to occur
    'rotate_range': (-10, 10), # deg
    'shear_range': (0, 0), # deg
    'scale_range': (0.95, 1.05), # %
    'translate_range': (0.1, -0.1), # %
    'elastic_alpha': 0,
    'elastic_sigma': 0,
}


"""
preprocessing.py
 - ...
"""
PRIORITY_ORDER = ('t1ce', 'flair', 't2', 't1')
EXTENSIONS = ('.png', '.jpg', '.jpeg', '.nii', '.nii.gz', '.dcm')


"""
segmentation.py
 - Match RGB with $accent-3 from 'app/assets/scss/_palette.scss'
 - Define opacities (A) based on clinical importance (ET > ED > NCR)
"""
BASE_COLOR = (82, 113, 255)
ALPHA_NCR = 128 # Necrotic tumour core (NCR) 
ALPHA_ED = 180 # Peritumoural edematous/invaded tissue (ED)
ALPHA_ET = 255 # Gadolinium-enhancing tumour (ET)