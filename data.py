from pathlib import Path
import json

from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from config import LABEL, BINARIZE_MASK, RANDOM_STATE

SCRIPT_DIR = Path(__file__).resolve().parent # root dir

INPUT_DIR = (SCRIPT_DIR / 'data/OCTA-500').resolve() 
OUTPUT_DIR = (SCRIPT_DIR / 'data/OCTA-500_processed').resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Make directories and subdirectories in _processed folder
for split in ['train', 'valid', 'test']:
    (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'masks' / split).mkdir(parents=True, exist_ok=True)


def process_sample(id: str, label: str, binarize_mask=bool) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Normalize projection maps and binarize segmentation masks. 
    """
    img = np.array(Image.open(INPUT_DIR / 'OCTA(ILM_OPL)' / id).convert('L')) / 255.0 # .bmp -> np arr
    mask = np.array(Image.open(INPUT_DIR / 'Label' / label / id).convert('L')) # .bmp -> np arr
    mask = (mask > 0).astype(np.uint8) if binarize_mask else mask
    return img, mask, id


def save_sample(img: np.ndarray, mask: np.ndarray, id: str, split: str) -> None:
    """
    Save projection maps and segmentation masks to their respective directories.
    """
    img_path = OUTPUT_DIR / 'images' / split / f'{id}.npy' 
    np.save(img_path, img) # float32

    mask_path = OUTPUT_DIR / 'masks' / split / f'{id}_mask.npy'
    np.save(mask_path, mask) # uint8 
    
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
    
    metadata = {
        'vessel_pixels': int(mask.sum()),
        'vessel_ratio': float(mask.mean())
    }

    # with open(metadata_path, 'w') as file:
    #     json.dump(metadata, file, default=str) # ->str if unserializable


def prepare_data() -> None:
    """
    Process 2D OCTA-500 projection maps (ILM–OPL slab) and labels for model training.
    """
    all_samples = []
    proj_ids = sorted([p.name for p in (INPUT_DIR / 'OCTA(ILM_OPL)').iterdir()])

    for id in tqdm(proj_ids, desc='Processing OCTA-500 samples'):
        sample = process_sample(id, label=LABEL, binarize_mask=BINARIZE_MASK)
        all_samples.append(sample)
    print(f'Total projection maps: {len(all_samples)}')
    
    # 70% train, 15% valid, 15% train
    train, temp = train_test_split(all_samples, test_size=0.3, random_state=RANDOM_STATE) 
    valid, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_STATE) 

    splits = { 'train': train, 'valid': valid, 'test': test }
    for name, data in splits.items():
        for img, mask, id in tqdm(data, desc=f"Saving '{name}' samples"):
            save_sample(img, mask, id, name)

    print('*COMPLETE: samples have been distributed across training, validation, and test sets')


if __name__ == '__main__':
    prepare_data()
