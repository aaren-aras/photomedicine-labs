from pathlib import Path
import json

from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# from config import MODALITIES, EPSILON, LABEL_MAP

SCRIPT_DIR = Path(__file__).resolve().parent # root dir
OCTA_500_DIR = (SCRIPT_DIR / 'data/OCTA_500_Training/OCTA(ILM_OPL)').resolve() 

INPUT_DIR = (SCRIPT_DIR / 'data/OCTA_500_Training').resolve() 

OUTPUT_DIR = (SCRIPT_DIR / 'data/OCTA_500_Processed').resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_DIR = OUTPUT_DIR / 'images'
MASK_DIR = OUTPUT_DIR / 'masks'

# Make directories and subdirectories in _Processed folder
for split in ['train', 'valid', 'test']:
    (IMG_DIR / split).mkdir(parents=True, exist_ok=True)
    (MASK_DIR / split).mkdir(parents=True, exist_ok=True)


def normalize_modality(img: np.ndarray) -> np.ndarray:
    """
    Normalize OCTA projection to 8-bit greyscale to standardize pixel intensities and highlight structure over brightness.
    """
    # img = np.nan_to_num(img) # handle any corrupted/missing values
    # img = np.clip(img, 0, np.percentile(img, 99)) # cut off extremes (99th %tile)
    # img = (img - np.min(img)) / (np.max(img) - np.min(img) + EPSILON) # scale to [0, 1]
    # return (img * 255).astype(np.uint8) # scale to [0, 255] (2e8)
    return img.astype(np.float32) / 255.0



def process_pair(img_path: Path) -> list[tuple[np.ndarray, np.ndarray, str, int]]:
    """
    Given a OCTA-500 projection (10xxx.bmp):
    - Stack all 4 MRI modalities (T1, T1CE, T2, FLAIR) into a 4-channel 3D image volume,
    - Extract MRI image slices from both the image volume and corresponding 3D segmentation mask.
    """
    id = img_path.name
    img = np.array(Image.open(INPUT_DIR / 'OCTA(ILM_OPL)' / id))
    normalized_img = normalize_modality(img)

    mask = np.array(Image.open(INPUT_DIR / 'Label' / 'GT_LargeVessel' / id))

    sample = [(normalized_img, mask, id)]
    return sample

    # for m in MODALITIES:
    #     m_nii = nib.load(subject_path / f'{subject_id}_{m}.nii.gz')
    #     data = normalize_modality(m_nii.get_fdata()) # Nifti1Image obj -> np arr (raw voxels as float) -> np arr (raw voxels normalized to [0, 255])
    #     imgs.append(data)
    # stacked = np.stack(imgs, axis=-1) # shape (H, W, D) x 4 -> (H, W, D, 4) (append new dim at the end)

    # seg_nii = nib.load(subject_path / f'{subject_id}_seg.nii.gz')
    # mask = seg_nii.get_fdata().astype(np.uint8) # Nifti1Image obj -> np arr (raw voxels as float) -> np arr (raw voxels as uint8)  
    # mask[mask == 4] = 3 # remap label 4 -> 3 for convenience

    # slices = []
    # for i in range(stacked.shape[2]): # per axial slice (z-resolution)
    #     img_slice = stacked[:, :, i, :] # shape (H, W, 4)
    #     mask_slice = mask[:, :, i] # shape (H, W)
    #     slices.append(img_slice, mask_slice, subject_id)

    # return slices


def save_sample(img: np.ndarray, mask: np.ndarray, id: str, split: str) -> None:
    """
    Save OCTA projection and corresponding segmentation label to their respective subsubdirectories.
    """
    img_path = IMG_DIR / split / f'{id}.npy' 
    mask_path = MASK_DIR / split / f'{id}_mask.npy'

    np.save(img_path, img) # 4-channel (T1, T1CE, T2, FLAIR): shape (H, W, 4), uint8 (0-255 greyscale)
    np.save(mask_path, mask) # 1-channel (class labels): shape (H, W), uint8 (0: background, 1: NCR, 2: ED, 4->3: ET)
    
    # total_pixels = mask_slice.size # total = H * W
    # unique, counts = np.unique(mask_slice, return_counts=True) # num pixels belonging to each class
    # class_pixel_counts = {int(u): int(c) for u, c in zip(unique, counts)}

    # Append per-slice stats to metadata JSON
    # stats = {}
    # for label_id, label_name in LABEL_MAP.items():
    #     count = class_pixel_counts.get(label_id, 0)
    #     stats[label_name] = {
    #         'pixel_count': count,
    #         'percent': round((count / total_pixels * 100), 2)
    #     }
    
    # metadata = {
    #     'subject_id': subject_id,
    #     'segmentation_stats': stats
    # }

    # with open(metadata_path, 'w') as file:
    #     json.dump(metadata, file, default=str) # ->str if unserializable


def prepare_data() -> None:
    """
    Process 2D OCTA-500 projection maps and labels (GT_LargeVessel) for model training.
    """
    
    '''
    prepare_data -> process/normalize -> prepare_data -> save 


    for each image of the 300 in octa(ilm_opl), 
    - normalize image
    - save image and label

    grab the corresopnding label in gt_largevessel and save them
    both as np to OCTA_500_Processed folder


    split into train, valid, test 
    '''
    all_samples = []
    
    imgs = sorted([p for p in (INPUT_DIR / 'OCTA(ILM_OPL)').iterdir()])
    # img_paths = INPUT_DIR / 'OCTA(ILM_OPL)'

    for img_path in tqdm(imgs, desc='Processing OCTA-500 data'):
        sample = process_pair(img_path)
        all_samples.extend(sample)
    print(f'Total 2D projection maps: {len(all_samples)}')
    
    # Split OCTA projections into training, validation, and test sets (70-15-15)
    train, temp = train_test_split(all_samples, test_size=0.3, random_state=2026) # 70% train, 30% temp
    valid, test = train_test_split(temp, test_size=0.5, random_state=2025) # of 30% temp: 50% valid, 50% test

    splits = [(train, 'train'), (valid, 'valid'), (test, 'test')]
    for split_data, split_name in splits:
        for img, mask, id in tqdm(split_data, desc=f'Saving \'{split_name}\' samples'):
            save_sample(img, mask, id, split_name)

    print('*COMPLETE: images have been distributed across training, validation, and test sets')


if __name__ == '__main__':
    prepare_data()
