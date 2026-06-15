import sys
from pathlib import Path


"""HOW TO GENERATE config.json and metadata.json as was done in earlier file version?"""

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path: sys.path.append(str(ROOT_DIR))

import zipfile, h5py, numpy as np
import tensorflow as tf
from model import build_segmentation_model

KERAS_PATH  = 'models/segmentation_best.keras'
WEIGHTS_H5  = 'models/_extracted/model.weights.h5'
H5_OUT      = 'models/segmentation_best.h5'

# ── Step 1: Extract weights ───────────────────────────────────────────────────
with zipfile.ZipFile(KERAS_PATH, 'r') as z:
    z.extract('model.weights.h5', 'models/_extracted/')
    z.extract('config.json', EXTRACT_DIR)
    z.extract('metadata.json', EXTRACT_DIR)
    print(f"│   ├── Saved layout specs → {CONFIG_JSON}")
    print(f"│   └── Saved training run records → {METADATA_JSON}")

# ── Step 2: Rebuild architecture (same code that trained it) ──────────────────
model = build_segmentation_model()

# ── Step 3: Map h5 layer names → model layers, inject weights ─────────────────
with h5py.File(WEIGHTS_H5, 'r') as f:
    layers_group = f['layers']
    
    for layer in model.layers:
        layer_name = layer.name  # e.g. 'conv2d', 'batch_normalization_3'
        
        if layer_name not in layers_group:
            continue  # input, pooling, activation, etc. — no weights
        
        vars_group = layers_group[layer_name]['vars']
        if len(vars_group) == 0:
            continue
        
        # Read weights in index order: 0, 1, 2, ...
        saved_weights = [
            vars_group[str(i)][:]
            for i in range(len(vars_group))
        ]
        
        try:
            layer.set_weights(saved_weights)
            print(f"  ✓  {layer_name:40s} {[w.shape for w in saved_weights]}")
        except ValueError as e:
            print(f"  ✗  {layer_name}: {e}")

# ── Step 4: Save as legacy .h5 ────────────────────────────────────────────────
model.save(H5_OUT)
print(f"\nDone → {H5_OUT}")