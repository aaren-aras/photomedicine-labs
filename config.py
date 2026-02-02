

LABEL = 'GT_LargeVessel'
BINARIZE_MASK = True
RANDOM_STATE = 28



LABEL_MAP = { 0: 'background', 1: 'NCR', 2: 'ED', 3: 'ET' } # 4 -> 3
EPSILON = 1e-8 # avoid dividing by 0


IMG_SIZE = (400, 400)
MODALITIES = ('OCTA',) 
NUM_CLASSES = 1 # bkgd vs. vessel


FILTERS = (16, 32, 64, 128)
KERNEL_SIZE = 3
SCALE_FACTOR = 2
DROPOUT_RATE = 0.3
LEARNING_RATE = 5e-4
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