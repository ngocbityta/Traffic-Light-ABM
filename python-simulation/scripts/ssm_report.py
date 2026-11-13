#!/usr/bin/env python3
"""Run the Le Van Luong simulation, record trajectories and report SSMs.

Example:
    python3 python-simulation/scripts/ssm_report.py --seed 1 --ticks 3000 \
        --save comparison_results/ssm_seed1.npz

Prints a JSON summary (rear-end TTC/MTTC/DRAC/time gap, TET/TIT, PET at
lane-level conflict points, junction disc TTC, hard braking). See
``python_simulation/ssm.py`` for definitions and caveats.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1] / "simulation"
sys.path.insert(0, str(ROOT))

from python_simulation import ssm, trajectory  # noqa: E402
from python_simulation.python_sim import LeVanLuongSimulator  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ticks", type=int, default=3000)
    parser.add_argument("--flow-per-approach", type=float, default=2400.0)
    parser.add_argument("--yellow", type=float, default=4.0)
    parser.add_argument("--ttc-threshold", type=float, default=ssm.DEFAULT_TTC_THRESHOLD)
    parser.add_argument("--no-pet", action="store_true",
                        help="skip conflict-point geometry and PET (faster)")
    parser.add_argument("--save", type=Path, default=None,
                        help="write the trajectory columns to this .npz")
    args = parser.parse_args()

    sim = LeVanLuongSimulator(seed=args.seed, flow_per_approach=args.flow_per_approach,
                              yellow=args.yellow)
    start = time.perf_counter()
    recorder = trajectory.record_run(sim, args.ticks)
    data = recorder.arrays()
    points = None if args.no_pet else trajectory.movement_conflict_points(sim, recorder)
    summary = ssm.summarize(data, points, ttc_threshold=args.ttc_threshold)
    summary.update(seed=args.seed, ticks=args.ticks, passed=sim.passed,
                   entries_during_yellow=sim.entries_during_yellow,
                   entries_during_red=sim.entries_during_red,
                   seconds=round(time.perf_counter() - start, 1))
    if args.save is not None:
        recorder.save(args.save)
        summary["saved"] = str(args.save)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
