# waveform_canvas.py
# ==================
# WaveformCanvas: Matplotlib-backed Qt widget showing live sawtooth + stair-step
# waveform previews. Updates automatically when the ROI changes.

import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel, QSizePolicy
from constants import (
    BG_DARK, BG_PANEL, BORDER, ACCENT_CYAN, ACCENT_AMBER,
    ACCENT_GREEN, ACCENT_RED, TEXT_DIM, TEXT_PRIMARY
)

try:
    import matplotlib
    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


if HAS_MPL:
    class WaveformCanvas(FigureCanvas):
        """
        Two-panel Matplotlib canvas:
            Top:    X fast axis — sawtooth waveform
            Bottom: Y slow axis — stair-step waveform

        Call update_waveforms() whenever the ROI or scan parameters change.
        The canvas updates in real-time as the user drags the ROI rectangle.
        """

        def __init__(self, parent=None):
            self.fig = Figure(figsize=(5, 3.5), facecolor=BG_DARK)
            super().__init__(self.fig)
            self.setParent(parent)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._setup_axes()

        def _setup_axes(self):
            self.fig.clear()
            self.ax_x = self.fig.add_subplot(211)
            self.ax_y = self.fig.add_subplot(212)
            for ax in (self.ax_x, self.ax_y):
                ax.set_facecolor(BG_PANEL)
                ax.tick_params(colors=TEXT_DIM, labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor(BORDER)
            self.ax_x.set_title("X  —  Fast Axis  (Sawtooth)",
                                 color=ACCENT_CYAN, fontsize=8)
            self.ax_y.set_title("Y  —  Slow Axis  (Stair-Step)",
                                 color=ACCENT_AMBER, fontsize=8)
            for ax in (self.ax_x, self.ax_y):
                ax.set_ylabel("Voltage (V)", color=TEXT_DIM, fontsize=7)
            self.ax_y.set_xlabel("Samples", color=TEXT_DIM, fontsize=7)
            self.fig.tight_layout(pad=1.8)

        def update_waveforms(self, x_start: float, x_end: float,
                             y_start: float, y_end: float,
                             n_lines: int = 50,
                             samples_per_line: int = 500):
            """
            Replot both waveforms for the given voltage ranges.
            Draws only the first 50 lines for speed — full waveform is
            generated at export time.
            """
            from waveforms import generate_sawtooth, generate_stair_step

            # Cap preview lines to 50 for speed (full export uses spin_nlines)
            preview_lines = min(n_lines, 50)
            spl = samples_per_line

            x_wave = generate_sawtooth(x_start, x_end, preview_lines, spl)
            y_wave = generate_stair_step(y_start, y_end, preview_lines, spl)
            t = np.arange(len(x_wave))

            self._setup_axes()

            self.ax_x.plot(t, x_wave, color=ACCENT_CYAN, lw=0.8)
            self.ax_y.plot(t, y_wave, color=ACCENT_AMBER, lw=0.8)

            for ax, v0, v1 in [
                (self.ax_x, x_start, x_end),
                (self.ax_y, y_start, y_end),
            ]:
                ax.axhline(v0, color=ACCENT_GREEN, lw=0.8, ls="--",
                           label=f"start  {v0:.3f} V")
                ax.axhline(v1, color=ACCENT_RED,   lw=0.8, ls="--",
                           label=f"end    {v1:.3f} V")
                ax.legend(fontsize=6, facecolor=BG_PANEL,
                          edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

            self.fig.tight_layout(pad=1.8)
            self.draw()

else:
    # Fallback: plain label if matplotlib not installed
    class WaveformCanvas(QLabel):
        def __init__(self, parent=None):
            super().__init__(
                "Install matplotlib for waveform preview\n"
                "  pip install matplotlib",
                parent
            )

        def update_waveforms(self, *args, **kwargs):
            pass
