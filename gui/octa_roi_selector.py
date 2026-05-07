#!/usr/bin/env python3

import sys
import os
import argparse

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from main_window import OCTROISelector


def main():
    """
    Entrypoint for the OCTA ROI GUI, standalone executable (#!/usr/bin/env python3).

    Usage:
        python octa_roi_selector.py
        python octa_roi_selector.py --image path/to/image.png

    """
    parser = argparse.ArgumentParser(
        description="OCTA ROI Selector->Galvo Waveform Generator"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to OCT image to load on startup"
    )
    args = parser.parse_args()

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)

    app = QApplication(sys.argv)
    app.setApplicationName("OCTA ROI Selector")

    window = OCTROISelector()
    window.show()

    if args.image and os.path.isfile(args.image):
        window.canvas.load_image(args.image)
        window.lbl_img_info.setText(
            f"{os.path.basename(args.image)}"
        )

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
