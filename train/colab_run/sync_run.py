"""Back up a training run directory so it can be resumed elsewhere.

``training_state.pt`` (the resumable checkpoint, written atomically by the
trainer) is copied first, the logs and weights after it: a
``training_log.csv`` copied later never has fewer rows than the checkpoint
needs, which ``--resume-dir`` checks. ``demand/`` is skipped: the simulators
of unfinished episodes carry their demand inside the checkpoint and new
episodes regenerate theirs.

Each copy is written as ``<name>.copying`` and renamed, so a backup that is
interrupted (runtime stopped, out of compute units) keeps the previous
complete file.

    python sync_run.py --src RUN [--dst DIR ...] [--tar FILE]
                       [--every SECONDS --until FLAG_FILE]
"""
import argparse
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

FIRST = ("training_state.pt",)
SKIP_SUFFIXES = (".tmp", ".copying", ".part")


def run_files(src):
    files = [path for path in Path(src).iterdir()
             if path.is_file() and not path.name.endswith(SKIP_SUFFIXES)]
    return sorted(files, key=lambda path: (path.name not in FIRST, path.name))


def copy_atomic(src, dst):
    temporary = dst.with_name(dst.name + ".copying")
    shutil.copy2(src, temporary)
    os.replace(temporary, dst)


def sync_once(src, destinations, tar_path=None):
    """Copy the run to every destination (in order) and optionally into one
    uncompressed tar; return the number of files copied."""
    files = run_files(src)
    for destination in destinations:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for path in files:
            try:
                copy_atomic(path, destination / path.name)
            except FileNotFoundError:  # replaced between listing and copying
                pass
    if tar_path is not None:
        tar_path = Path(tar_path)
        temporary = tar_path.with_name(tar_path.name + ".part")
        with tarfile.open(temporary, "w") as archive:
            for path in files:
                try:
                    archive.add(path, arcname=path.name)
                except FileNotFoundError:
                    pass
        os.replace(temporary, tar_path)
    return len(files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", action="append", default=[],
                        help="backup directory (repeatable); skipped when its parent is missing")
    parser.add_argument("--tar", default=None, help="also write the run into this tar")
    parser.add_argument("--every", type=float, default=0.0, help="seconds between backups")
    parser.add_argument("--until", default=None, help="loop until this file exists, then sync once more")
    args = parser.parse_args(argv)

    def once():
        destinations = [d for d in args.dst if Path(d).parent.exists()]
        started = time.monotonic()
        try:
            count = sync_once(args.src, destinations, args.tar)
            print(f"[sync] {time.strftime('%H:%M:%S')} {count} files -> "
                  f"{', '.join(destinations) or '-'}{' + ' + args.tar if args.tar else ''} "
                  f"({time.monotonic() - started:.1f}s)", flush=True)
        except Exception as error:  # keep the loop alive; the next round retries
            print(f"[sync] {time.strftime('%H:%M:%S')} failed: {error!r}", flush=True)

    if not args.every:
        once()
        return 0
    while True:
        deadline = time.monotonic() + args.every
        while time.monotonic() < deadline:
            if args.until and Path(args.until).exists():
                once()
                return 0
            time.sleep(min(5.0, args.every))
        if Path(args.src).exists():
            once()


if __name__ == "__main__":
    sys.exit(main())
