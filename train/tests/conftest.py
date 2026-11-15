"""Make the training package, the Python simulator and the repo scripts importable."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
# ``train`` is imported as a package (``train.logic.*``). Do not put
# ``train/logic`` itself on the path: its ``train.py`` would shadow the package.
for entry in (ROOT / "scripts", ROOT / "python-simulation" / "simulation", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
