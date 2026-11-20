"""Select an optional native build made from these exact simulator sources."""
from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import sys
import sysconfig


ROOT = Path(__file__).resolve().parent
MODULES = ("vehicle_geometry", "junction_negotiation", "python_sim")
CACHE = ROOT / "__pycache__" / "native"
SUFFIX = sysconfig.get_config_var("EXT_SUFFIX")


@lru_cache(maxsize=1)
def build_path():
    digest = hashlib.sha256()
    for name in (*MODULES, "_native"):
        digest.update((ROOT / (name + ".py")).read_bytes())
    digest.update((ROOT / "python_sim.pxd").read_bytes())
    abi = sysconfig.get_config_var("SOABI") or sys.implementation.cache_tag
    return CACHE / f"{abi}-{platform.machine()}-{digest.hexdigest()[:24]}"


def load_if_built(name, source):
    if not source.endswith(".py") or not CACHE.is_dir() or os.getenv("LVL_PURE_PYTHON") == "1":
        return
    binary = build_path() / (name.rsplit(".", 1)[-1] + SUFFIX)
    if not binary.is_file():
        return
    spec = importlib.util.spec_from_file_location(name, binary)
    module = importlib.util.module_from_spec(spec)
    original = sys.modules[name]
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules[name] = original
        raise


class _PythonAnnotations:
    compiled = False

    @staticmethod
    def annotation_typing(enabled):
        return lambda function: function


cython = _PythonAnnotations()
