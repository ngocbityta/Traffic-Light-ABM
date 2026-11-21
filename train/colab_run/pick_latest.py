"""Choose the most advanced copy of a training run and install it.

Candidates are run directories or tar files (from ``sync_run.py``). The one
whose ``training_state.pt`` holds the most learner decisions wins; with
``--install DIR`` it replaces DIR (the old DIR is kept as DIR.before_<time>).

    python pick_latest.py --install train/results/RUN CANDIDATE [CANDIDATE ...]
"""
import argparse
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import torch


def decisions_in(directory):
    state = Path(directory) / "training_state.pt"
    if not state.is_file():
        return None
    try:
        snapshot = torch.load(state, map_location="cpu", weights_only=False)
        return int(snapshot["agent"]["decisions"]), int(snapshot["episode"])
    except Exception as error:
        print(f"[pick] unreadable checkpoint in {directory}: {error!r}")
        return None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", required=True)
    parser.add_argument("candidates", nargs="*")
    args = parser.parse_args(argv)
    target = Path(args.install)
    work = Path(tempfile.mkdtemp(prefix="pick_", dir=target.parent if target.parent.exists() else None))
    try:
        options = []
        for index, candidate in enumerate([str(target)] + args.candidates):
            path = Path(candidate).expanduser()
            if path.is_file() and tarfile.is_tarfile(path):
                unpacked = work / f"c{index}"
                unpacked.mkdir()
                with tarfile.open(path) as archive:
                    try:
                        archive.extractall(unpacked, filter="data")
                    except TypeError:  # Python without extraction filters
                        archive.extractall(unpacked)
                directory = unpacked
            elif path.is_dir():
                directory = path
            else:
                continue
            found = decisions_in(directory)
            print(f"[pick] {candidate}: " + (f"{found[0]} decisions, {found[1]} rollouts"
                                             if found else "no checkpoint"))
            if found:
                options.append((found, index, path, directory))
        if not options:
            print("[pick] no checkpoint found")
            return 2
        options.sort(key=lambda item: (item[0], -item[1]))
        found, index, path, directory = options[-1]
        if index == 0:
            print(f"[pick] keeping {target} ({found[0]} decisions)")
            return 0
        if target.exists():
            backup = target.with_name(target.name + time.strftime(".before_%Y%m%d_%H%M%S"))
            target.rename(backup)
            print(f"[pick] previous copy kept as {backup}")
        target.mkdir(parents=True)
        for item in Path(directory).iterdir():
            if item.is_file() and not item.name.endswith((".copying", ".part", ".tmp")):
                shutil.copy2(item, target / item.name)
        print(f"[pick] installed {path} ({found[0]} decisions) into {target}")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
