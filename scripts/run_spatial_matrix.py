#!/usr/bin/env python3
"""Run spatial experiments on identical demand and validation seeds."""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "dqn_py" / "train.py"
ALGORITHMS = ("rainbow", "hybrid")
LEGACY_REWARDS = ("r1", "r2", "r3", "r4")
LVL_REWARDS = ("r1", "r2", "r4")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--max-train-seconds", type=float)
    parser.add_argument("--episode-horizon", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--port", type=int, default=6868)
    parser.add_argument("--scenario", choices=("lvl", "legacy"), default="lvl")
    parser.add_argument("--backend", choices=("gama", "python"), default="python")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rollout-decisions", type=int, default=16)
    parser.add_argument("--fast-simulation", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--validation-seed-base", type=int, default=10_000_000)
    parser.add_argument("--r1-ref", type=float, default=50.0)
    parser.add_argument("--r2-ref", type=float, default=50.0)
    parser.add_argument("--b-ref", type=float, default=50.0)
    parser.add_argument("--r2-lambda", type=float, default=1.0)
    parser.add_argument("--r3-beta", type=float, default=1.0)
    parser.add_argument("--pressure-lambda", type=float, default=.5)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--warmup", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.total_steps <= 0 or args.episode_horizon <= 100 or args.workers <= 0 or
            args.rollout_decisions <= 0 or
            (args.max_train_seconds is not None and args.max_train_seconds <= 0) or
            min(args.validate_every, args.checkpoint_every) <= 0):
        parser.error("Steps/workers must be positive and horizon must exceed 100 cycles")
    return args


def run_matrix(args):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "matrix_manifest.json"
    common = {key: str(getattr(args, key)) for key in
                           ("scenario", "backend", "total_steps", "max_train_seconds", "episode_horizon", "episodes", "workers",
                            "rollout_decisions", "fast_simulation",
                            "seed_base", "validation_seed_base", "r1_ref", "r2_ref", "b_ref",
                            "r2_lambda", "r3_beta", "pressure_lambda", "validate_every",
                            "checkpoint_every", "buffer_size", "warmup", "batch_size")}
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"{manifest_path} exists; use --resume")
        manifest = json.loads(manifest_path.read_text())
        if manifest["common"] != common:
            raise ValueError("Matrix resume requires identical common settings")
    else:
        manifest = {"common": common, "cases": {}}
    for algorithm in ALGORITHMS:
        for reward in (LVL_REWARDS if args.scenario == "lvl" else LEGACY_REWARDS):
            name = f"{algorithm}_{reward}"
            destination = root / name
            snapshot = destination / "training_state.pt"
            previous = manifest["cases"].get(name)
            if (args.resume and previous and previous.get("exit_code") == 0 and
                    (destination / "dqn_weights.pt").exists() and
                    (args.max_train_seconds is None or
                     previous.get("train_seconds", 0) >= args.max_train_seconds - 5)):
                print(f"[MATRIX] Skip completed {name}", flush=True)
                continue
            if args.resume and destination.exists() and not snapshot.exists():
                archived = root / f"{name}_interrupted_{int(time.time())}"
                destination.rename(archived)
                print(f"[MATRIX] Archived incomplete {name} to {archived.name}", flush=True)
            if destination.exists() and not (args.resume and snapshot.exists()):
                raise FileExistsError(f"{destination} exists; choose another directory or --resume")
            command = [sys.executable, "-u", str(TRAIN),
                       "--algorithm", algorithm, "--scenario", args.scenario,
                       "--backend", args.backend,
                       "--reward-mode", reward,
                       "--total-steps", str(args.total_steps),
                       "--episode-horizon", str(args.episode_horizon),
                       "--port", str(args.port), "--workers", str(args.workers),
                       "--rollout-decisions", str(args.rollout_decisions),
                       "--fast-simulation" if args.fast_simulation else "--no-fast-simulation",
                       "--seed-base", str(args.seed_base),
                       "--validation-seed-base", str(args.validation_seed_base),
                       "--r1-ref", str(args.r1_ref), "--r2-ref", str(args.r2_ref),
                       "--b-ref", str(args.b_ref), "--r2-lambda", str(args.r2_lambda),
                       "--r3-beta", str(args.r3_beta),
                       "--pressure-lambda", str(args.pressure_lambda),
                       "--buffer-size", str(args.buffer_size),
                       "--warmup", str(args.warmup), "--batch-size", str(args.batch_size),
                       "--validate-every", str(args.validate_every),
                       "--checkpoint-every", str(args.checkpoint_every)]
            if args.backend == "gama":
                command.append("--start-server")
            if args.max_train_seconds is not None:
                command += ["--max-train-seconds", str(args.max_train_seconds)]
            if args.episodes:
                command += ["--episodes", str(args.episodes)]
            command += ["--resume-dir" if args.resume and snapshot.exists() else "--output-dir",
                        str(destination)]
            print(f"\n[MATRIX] {name}: {' '.join(command)}", flush=True)
            start = time.monotonic()
            result = subprocess.run(command, cwd=ROOT, check=False)
            server_errors = []
            for server_log in destination.glob("server_*.log"):
                contents = server_log.read_text(errors="replace")
                server_errors.extend(re.findall(
                    r"(?m)^(?:java\.lang\.[\w.]+(?:Exception|Error):|Java error:|"
                    r"SimulationError|RuntimeError)",
                    contents))
            metrics = {}
            training_log = destination / "training_log.csv"
            if training_log.exists():
                with training_log.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if rows:
                    train_seconds = sum(float(row["wall_seconds"]) for row in rows)
                    decisions = sum(int(row["decisions"]) for row in rows)
                    sim_seconds = sum(int(row.get("simulated_cycles", row["final_cycle"]))
                                      for row in rows) * (.2 if args.scenario == "lvl" else .1)
                    updates = int(rows[-1]["updates"])
                    metrics = {"episodes": len(rows), "decisions": decisions,
                               "updates": updates,
                               "train_seconds": round(train_seconds, 1),
                               "sim_seconds": round(sim_seconds, 1),
                               "sim_seconds_per_wall_second": round(sim_seconds / train_seconds, 3)
                               if train_seconds > 0 else None,
                               "decisions_per_second": round(decisions / train_seconds, 4)
                               if train_seconds > 0 else None,
                               "last_rollout_reward": float(rows[-1].get(
                                   "rollout_reward", rows[-1].get("episode_reward", 0))),
                               "last_avg_loss": float(rows[-1]["avg_loss"])}
            summary_path = destination / "training_summary.json"
            if summary_path.exists():
                training_summary = json.loads(summary_path.read_text())
                metrics.update({
                    "episodes": training_summary["episodes"],
                    "decisions": training_summary["decisions"],
                    "updates": training_summary["updates"],
                    "train_seconds": training_summary["train_wall_seconds"],
                    "sim_seconds": training_summary["aggregate_simulated_seconds"],
                    "decisions_per_second": training_summary["decisions_per_second"],
                    "sim_seconds_per_wall_second": training_summary["simulated_seconds_per_wall_second"],
                })
            manifest["cases"][name] = {"algorithm": algorithm, "reward": reward,
                                       "output_dir": str(destination),
                                       "exit_code": result.returncode,
                                       "server_error_count": len(server_errors),
                                       "wall_seconds": round(time.monotonic() - start, 1),
                                       "final_weights": str(destination / "dqn_weights.pt"),
                                       "best_weights": str(destination / "best_weights.pt"),
                                       **metrics}
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(manifest_path)
            if result.returncode:
                raise SystemExit(f"{name} failed with exit code {result.returncode}; see {destination}")
            if server_errors:
                print(f"[MATRIX] Warning: {name} server log has {len(server_errors)} Java/runtime errors", flush=True)
    print(f"[MATRIX] {len(manifest['cases'])} cases complete: {manifest_path}")


if __name__ == "__main__":
    run_matrix(parse_args())
