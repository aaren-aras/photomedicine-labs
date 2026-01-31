"""
data.py
 - Define small constant ε to avoid division-by-0 cases
"""
LABEL_MAP = { 0: 'background', 1: 'NCR', 2: 'ED', 3: 'ET' } # 4 -> 3
MODALITIES = ('OCTA') # CHANGED
EPSILON = 1e-8


"""
model.py
 - Match image size with default x-y resolution for BraTS scans 
 - Define scale factor for symmetric downsampling (max pooling) and upsampling
"""
IMG_SIZE = (224, 224)
NUM_CLASSES = 1 # CHANGED
FILTERS = (32, 64, 128, 256)
KERNEL_SIZE = 3
SCALE_FACTOR = 2
DROPOUT_RATE = 0.5
LEARNING_RATE = 1e-4
BATCH_SIZE = 24 # num samples (image + mask) per training step (before updating weights)
EPOCHS = 30 # full passes through entire dataset
AUG_CONFIG = {
    'flip_lr': 0.5, # % chance to occur
    'flip_ud': 0.2, # % chance to occur
    'rotate_range': (-15, 15), # deg
    'shear_range': (-10, 10), # deg
    'scale_range': (0.9, 1.1), # %
    'translate_range': (0.1, -0.1), # %
    'elastic_alpha': 50,
    'elastic_sigma': 5,
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