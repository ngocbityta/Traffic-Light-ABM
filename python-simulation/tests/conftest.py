"""Make the standalone simulation package importable during test discovery."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python-simulation" / "simulation"))
