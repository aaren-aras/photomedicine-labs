# constants.py
# ============
# All shared constants: theme colours, galvo parameters, image defaults.
# Import from here everywhere — never hardcode these values elsewhere.

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent   # photomedicine-labs/
MODELS_DIR = ROOT_DIR / 'models'
SEG_MODEL_PATH = MODELS_DIR / 'segmentation_best.h5'

# ── Galvo / scan parameters ───────────────────────────────────────────────────
GALVO_V_MIN     = 0.0    # volts — start of full sweep
GALVO_V_MAX     = 2.0    # volts — end of full sweep
OCTA_PIXEL_SIZE = 500    # OCTA-500 native image resolution (px)
MODEL_INPUT_SIZE = 400   # segmentation model input size (must match training)

# ── Dark scientific theme ─────────────────────────────────────────────────────
BG_DARK      = "#0d1117"
BG_PANEL     = "#161b22"
BG_CARD      = "#1c2128"
ACCENT_CYAN  = "#39d0d8"
ACCENT_GREEN = "#3fb950"
ACCENT_AMBER = "#e3b341"
ACCENT_RED   = "#f85149"
TEXT_PRIMARY = "#e6edf3"
TEXT_DIM     = "#7d8590"
BORDER       = "#30363d"
