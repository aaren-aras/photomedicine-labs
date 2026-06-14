# waveforms.py
# ============
# Pure functions for galvo mirror waveform generation and pixel↔voltage mapping.
# No Qt imports — fully testable in isolation.
#
# Physical context:
#   X axis = fast axis → sawtooth: sweeps left→right on every scan line
#   Y axis = slow axis → stair-step: steps down one line at a time
#   Full scan: X ∈ [0, 2V], Y ∈ [0, 2V] (OCTA-500 assumption)
#   ROI selection maps a pixel sub-region proportionally to a voltage sub-range

import numpy as np
from constants import GALVO_V_MIN, GALVO_V_MAX


def px_to_volts(px: int, img_size: int,
                v_min: float = GALVO_V_MIN,
                v_max: float = GALVO_V_MAX) -> float:
    """
    Convert a pixel coordinate to a galvo voltage.
    Linear/proportional: pixel 0 → v_min, pixel img_size → v_max.

    Args:
        px:       pixel coordinate along one axis
        img_size: total image dimension in pixels
        v_min:    voltage at pixel 0    (default 0.0 V)
        v_max:    voltage at pixel W-1  (default 2.0 V)

    Returns:
        Galvo voltage in volts
    """
    return v_min + (px / img_size) * (v_max - v_min)


def volts_to_px(v: float, img_size: int,
                v_min: float = GALVO_V_MIN,
                v_max: float = GALVO_V_MAX) -> int:
    """Inverse of px_to_volts — convert voltage back to pixel coordinate."""
    return int((v - v_min) / (v_max - v_min) * img_size)


def generate_sawtooth(v_start: float, v_end: float,
                      n_lines: int, samples_per_line: int) -> np.ndarray:
    """
    Generate X (fast axis) sawtooth waveform.

    Each scan line: linear ramp from v_start → v_end over samples_per_line
    samples, then instant fly-back (not included — DAQ handles retrace).
    The waveform is tiled n_lines times.

    Returns:
        1D float64 array of length n_lines * samples_per_line
    """
    line = np.linspace(v_start, v_end, samples_per_line)
    return np.tile(line, n_lines)


def generate_stair_step(v_start: float, v_end: float,
                        n_lines: int, samples_per_line: int) -> np.ndarray:
    """
    Generate Y (slow axis) stair-step waveform.

    One voltage step per line, held constant for samples_per_line samples.
    Steps linearly from v_start to v_end across n_lines steps.

    Returns:
        1D float64 array of length n_lines * samples_per_line
    """
    steps = np.linspace(v_start, v_end, n_lines)
    return np.repeat(steps, samples_per_line)


def roi_to_voltages(x0: int, y0: int, x1: int, y1: int,
                    img_w: int, img_h: int,
                    vx_max: float = GALVO_V_MAX,
                    vy_max: float = GALVO_V_MAX) -> dict:
    """
    Convert ROI pixel coordinates to galvo voltage ranges.

    Args:
        x0, y0: top-left pixel of ROI
        x1, y1: bottom-right pixel of ROI
        img_w:  image width in pixels
        img_h:  image height in pixels
        vx_max: full-scale X voltage (default 2.0 V)
        vy_max: full-scale Y voltage (default 2.0 V)

    Returns:
        dict with keys: vx0, vx1, vy0, vy1 (all floats in volts)
    """
    return {
        'vx0': px_to_volts(x0, img_w, 0.0, vx_max),
        'vx1': px_to_volts(x1, img_w, 0.0, vx_max),
        'vy0': px_to_volts(y0, img_h, 0.0, vy_max),
        'vy1': px_to_volts(y1, img_h, 0.0, vy_max),
    }
