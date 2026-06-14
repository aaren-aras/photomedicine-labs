import sys
from pathlib import Path

# Run utils/ files as self-contained scripts you can run from anywhere
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path: sys.path.insert(0, str(ROOT_DIR))