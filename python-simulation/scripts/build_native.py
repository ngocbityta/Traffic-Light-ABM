"""Build the optional, cProfile-visible simulator extensions for this Python."""
import os
from pathlib import Path
import platform
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation"))
from python_simulation._native import CACHE, MODULES, ROOT, SUFFIX, build_path


def main():
    destination = build_path()
    if all((destination / (name + SUFFIX)).is_file() for name in MODULES):
        print(f"Native simulator is ready in {destination}")
        return
    if sys.implementation.name != "cpython" or sys.version_info[:2] == (3, 12):
        sys.exit("This build requires CPython with Cython profiling support, e.g. CPython 3.14.")
    try:
        from Cython.Build import cythonize
        from setuptools import Extension, setup
    except ImportError:
        sys.exit('Install the build tools with: python3 -m pip install "Cython==3.3.1" setuptools')

    if sys.platform == "darwin":
        os.environ.setdefault("ARCHFLAGS", "-arch " + platform.machine())
    flags = ["/fp:strict"] if os.name == "nt" else ["-ffp-contract=off"]
    CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=CACHE) as temp:
        stage = Path(temp)
        extensions = [Extension("python_simulation." + name, [str(ROOT / (name + ".py"))],
                                extra_compile_args=flags) for name in MODULES]
        extensions = cythonize(extensions, build_dir=str(stage / "cython"), compiler_directives={
            "language_level": 3, "profile": True, "annotation_typing": False, "infer_types": False,
            "nonecheck": True,
        })
        setup(name="lvl-native", packages=[], ext_modules=extensions, script_args=[
            "build_ext", "--build-temp", str(stage / "objects"),
            "--build-lib", str(stage / "lib"), "--parallel", str(len(MODULES)),
        ])
        build_path.cache_clear()
        if build_path() != destination:
            sys.exit("Simulator sources changed during the build. Run the build again.")
        # Publish all three together. Another successful build may finish first.
        try:
            (stage / "lib" / "python_simulation").rename(destination)
        except OSError:
            if not all((destination / (name + SUFFIX)).is_file() for name in MODULES):
                raise
    print(f"Native simulator is ready in {destination}")


if __name__ == "__main__":
    main()
