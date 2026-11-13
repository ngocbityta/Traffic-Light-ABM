"""Shared import paths for repository-level script tests."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python-simulation" / "simulation"))
sys.path.insert(0, str(ROOT / "train" / "logic"))
