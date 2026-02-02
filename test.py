import numpy as np
from pathlib import Path
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# ----------------
# Utilities
# ----------------
# def normalize(slice):
#     """Normalize slice to [0,255] uint8."""
#     slice = slice.astype(np.float32)
#     slice = (slice - np.min(slice)) / (np.max(slice) - np.min(slice) + 1e-6)
#     return (slice * 255).astype(np.uint8)

def save_mask(mask: np.ndarray, output_path: Path, colorize=True):
    output_path.mkdir(exist_ok=True)

    # Save clean binary mask
    Image.fromarray((mask * 255).astype(np.uint8)).save(
        output_path / f"{output_path.stem}_mask.png"
    )

    # Optional visualization
    plt.imsave(
        output_path / f"{output_path.stem}_mask_vis.png",
        mask,
        cmap='gray',
        vmin=0,
        vmax=1
    )


# ----------------
# Inference
# ----------------
SCRIPT_DIR = Path(__file__).resolve().parent 

MODEL_PATH = SCRIPT_DIR / 'models' / 'octa_model.keras'

INPUT_DIR = SCRIPT_DIR / 'data' / 'OCTA-500_processed' / 'images' / 'test' / '10008.bmp.npy'
OUTPUT_DIR = SCRIPT_DIR / 'previews'
OUTPUT_DIR.mkdir(exist_ok=True)

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded.")

# Load projection (H, W)
img = np.load(INPUT_DIR).astype(np.float32) / 255.0
img = np.expand_dims(img, axis=-1)  # (H, W, 1)
img = np.expand_dims(img, axis=0)   # (1, H, W, 1) for batch

# Run inference
pred_mask = model.predict(img)[0, :, :, 0]  # (1, H, W, 1) -> (H, W), values 0-1
binary_mask = (pred_mask > 0.5).astype(np.uint8) # thresholdl for binary mask

# Save output
save_mask(binary_mask, OUTPUT_DIR)
print(f"*SAVE: predicted mask saved to {OUTPUT_DIR}")
