"""Layer 2: the simulator against closed-form traffic-flow results (no field data).

    python3 validation/layer2_theory.py discharge --out validation/results/layer2_discharge.json
    python3 validation/layer2_theory.py delay --flows 300 600 --out validation/results/layer2_delay.json

discharge  Saturated discharge of each approach (calibration.measure_discharge,
           first green after a full red) for (a) cars only, compared with the
           maximum equilibrium flow of the car IDM and with the HCM base
           saturation flow (1900 pc/h/lane), and (b) the default Hanoi mix.
delay      Mean time in network of through and left-turning vehicles at
           several demand levels (fixed plan 50/40 s, 94 s cycle). The increase
           over the lowest demand is compared with Webster's delay formula,
           using the saturation flow and lost time measured in 'discharge'.
           Results are appended to the output JSON (one entry per flow), so
           the command can be split over several calls.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python-simulation" / "simulation"))

from python_simulation import calibration  # noqa: E402
from python_simulation.python_sim import (ARMS, DEFAULT_VEHICLE_PARAMS, DT,  # noqa: E402
                                          LeVanLuongSimulator)

LANE_WIDTH = 3.375  # m, physical lane width used to count lanes (13.5 m approach = 4 lanes)


def idm_qmax(p, delta=4.0):
    """Maximum equilibrium flow (veh/h per column) of the IDM with parameters p."""
    v0 = p["speed_mean"] / 3.6
    v = np.linspace(0.05, 0.999 * v0, 4000)
    s_e = (p["gap"] + v * p["headway"]) / np.sqrt(1 - (v / v0) ** delta)
    q = v / (s_e + p["length"]) * 3600.0
    i = int(np.argmax(q))
    return float(q[i]), float(v[i])


def discharge(seeds, arms=ARMS, out=None):
    probe = LeVanLuongSimulator(seed=1, flow_per_approach=0)
    out = out or {}
    out.setdefault("car_only", {})
    out.setdefault("hanoi_mix", {})
    for label, ratios in (("car_only", (0.0, 0.0)), ("hanoi_mix", (0.85, None))):
        for arm in arms:
            kwargs = {"motorbike_ratio": ratios[0], "arm": arm}
            if ratios[1] is not None:
                kwargs["heavy_ratio"] = ratios[1]
            runs = [calibration.measure_discharge(seed=s, **kwargs) for s in seeds]
            width = probe.roads[f"{arm}_in"]["num_lanes"] * 1.05
            out[label][arm] = {
                "approach_width_m": width,
                "lanes": width / LANE_WIDTH,
                "saturation_flow": float(np.mean([r["saturation_flow"] for r in runs])),
                "saturation_flow_sd": float(np.std([r["saturation_flow"] for r in runs])),
                "lost_time": float(np.mean([r["lost_time"] for r in runs])),
                "first_10s": float(np.mean([r["first_10s"] for r in runs])),
                "seeds": list(seeds)}
    q, v = idm_qmax(DEFAULT_VEHICLE_PARAMS["car"])
    car = out["car_only"]
    per_lane = [a["saturation_flow"] / a["lanes"] for a in car.values()]
    per_column = [a["saturation_flow"] / (a["approach_width_m"] // 2.1) for a in car.values()]
    out["car_checks"] = {
        "idm_qmax_per_column": q, "idm_speed_at_qmax": v,
        "sim_per_physical_lane_mean": float(np.mean(per_lane)),
        "sim_per_car_column_mean": float(np.mean(per_column)),
        "hcm_base_per_lane": 1900.0,
        "note": "car columns = floor(width / 2.1 m): two 1.05 m sublanes per car"}
    return out


def mean_time_in_network(flow, seed, ticks, warmup):
    sim = LeVanLuongSimulator(seed=seed, flow_per_approach=flow)
    times = {arm: [] for arm in ARMS}
    alive = {}
    for k in range(warmup + ticks):
        sim.step()
        now = {v.creation_id: v for v in sim.vehicles}
        if k >= warmup:
            for vid, v in alive.items():
                if vid not in now and v.maneuver != "right":
                    times[v.arm].append((sim.cycle - v.created_cycle) * DT)
        alive = now
    return {arm: (float(np.mean(t)) if t else float("nan"), len(t)) for arm, t in times.items()}


def webster(q, s, g_eff, cycle):
    """Webster (1958) mean delay per vehicle (s); q, s in veh/h."""
    lam = g_eff / cycle
    x = q / (s * lam)
    if x >= 1:
        return float("nan"), x
    qs = q / 3600.0
    d = (cycle * (1 - lam) ** 2 / (2 * (1 - lam * x)) + x ** 2 / (2 * qs * (1 - x))
         - 0.65 * (cycle / qs ** 2) ** (1 / 3) * x ** (2 + 5 * lam))
    return float(d), float(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["discharge", "delay"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--flows", nargs="+", type=float, default=[300, 600, 900, 1200, 1500])
    parser.add_argument("--ticks", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.what == "discharge":
        old = json.loads(args.out.read_text()) if args.out.exists() else None
        args.out.write_text(json.dumps(discharge(args.seeds, args.arms, old), indent=1))
        return
    data = json.loads(args.out.read_text()) if args.out.exists() else {}
    for flow in args.flows:
        for seed in args.seeds:
            data[f"{flow:g}_{seed}"] = {"flow": flow, "seed": seed,
                                        "time_in_network": mean_time_in_network(
                                            flow, seed, args.ticks, args.warmup)}
            args.out.write_text(json.dumps(data, indent=1))
            print(flow, seed, flush=True)


if __name__ == "__main__":
    main()
