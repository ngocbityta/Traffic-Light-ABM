#!/usr/bin/env python3
"""Measure or calibrate queue discharge (saturation flow, start-up lost time).

Measure the current model:
    python3 python-simulation/scripts/calibrate_discharge.py --measure-only

Calibrate against your own field values (JSON with keys returned by
``measure_discharge``, e.g. {"saturation_flow_per_m": 650, "lost_time": 2.0}):
    python3 python-simulation/scripts/calibrate_discharge.py \
        --targets targets.json --params moto_headway moto_gap reaction_delay \
        --budget 40 --output calibrated.json

Vehicle mix from a count file (Nguyen Trai - Khuat Duy Tien format):
    python3 python-simulation/scripts/calibrate_discharge.py --counts \
        train/benchmarks/gama/data/traffic_data_NT_KDT.csv
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] / "simulation"
sys.path.insert(0, str(ROOT))

from python_simulation import calibration  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measure-only", action="store_true")
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--params", nargs="+",
                        default=["moto_headway", "moto_gap", "reaction_delay",
                                 "discharge_headway"],
                        choices=sorted(calibration.PARAMETER_SPACE))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--arm", default="SE", choices=["NE", "SE", "SW", "NW"])
    parser.add_argument("--junction-model", default="reservation",
                        choices=["reservation", "negotiation"])
    parser.add_argument("--counts", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.counts:
        rows = calibration.demand_from_counts(args.counts)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    base = {"junction_model": args.junction_model, "arm": args.arm}
    if args.measure_only or not args.targets:
        results = [calibration.measure_discharge(seed=s, **base) for s in args.seeds]
        print(json.dumps(results, indent=2))
        return
    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    result = calibration.calibrate(
        targets, names=args.params, seeds=args.seeds, budget=args.budget,
        base_kwargs=base,
        log=lambda h: print(json.dumps({"loss": round(h["loss"], 4), **h["params"],
                                        **h["measured"]}), flush=True))
    print(json.dumps({"best": result["best"], "simulator_kwargs": result["kwargs"]},
                     indent=2))
    if args.output:
        calibration.save(result, args.output)


if __name__ == "__main__":
    main()
