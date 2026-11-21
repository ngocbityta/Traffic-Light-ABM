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


def profile_requested():
    """``LVL_NATIVE_PROFILE=1``: build/load the cProfile-visible extensions.
    Cython's profiling hooks cost ~9 % per tick, so the default is without."""
    return os.getenv("LVL_NATIVE_PROFILE") == "1"


@lru_cache(maxsize=2)
def build_path(profile=None):
    if profile is None:
        profile = profile_requested()
    digest = hashlib.sha256()
    for name in (*MODULES, "_native"):
        digest.update((ROOT / (name + ".py")).read_bytes())
    for header in ("python_sim.pxd", "junction_negotiation.pxd"):
        digest.update((ROOT / header).read_bytes())
    if profile:
        digest.update(b"cython-profile")
    abi = sysconfig.get_config_var("SOABI") or sys.implementation.cache_tag
    return CACHE / f"{abi}-{platform.machine()}-{digest.hexdigest()[:24]}"


def load_if_built(name, source):
    if not source.endswith(".py") or not CACHE.is_dir() or os.getenv("LVL_PURE_PYTHON") == "1":
        return
    filename = name.rsplit(".", 1)[-1] + SUFFIX
    binary = build_path() / filename
    if not binary.is_file():
        # The other variant (with or without profiling hooks) of the same sources.
        binary = build_path(not profile_requested()) / filename
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
