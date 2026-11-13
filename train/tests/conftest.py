"""Make training logic and the Python simulator importable during discovery."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python-simulation" / "simulation"))
sys.path.insert(0, str(ROOT / "train" / "logic"))
