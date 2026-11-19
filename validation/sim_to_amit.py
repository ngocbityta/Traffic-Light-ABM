"""Run the Python simulator and write its output as AMIT recordings.

    python3 validation/sim_to_amit.py --out validation/sim_runs/hanoi_default --seeds 1 2 3 \
        --ticks 4500 [--sim-config cfg.json]

Each seed becomes one recording '<name>-<seed>' with '_TRJ.csv' (10 Hz,
linear interpolation of the 5 Hz simulator ticks), '_linemarking.csv' (stop
lines from areas.geojson, exit lines at the start of each exit road) and
'_signal.csv' (fixed plan of the run). Approaches: NE -> A, SE -> B, SW -> C,
NW -> D (clockwise, as AMIT). Vehicle classes: motorbike -> m, car -> c,
heavy -> t. The first 'warmup' seconds are dropped and the clock restarts at
the end of the warm-up, so the network is loaded when the recording starts.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SIM = REPO / "python-simulation" / "simulation"
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM / "python_simulation"))

from python_simulation.python_sim import ARMS, DT, LeVanLuongSimulator  # noqa: E402
from python_simulation.trajectory import TrajectoryRecorder, MANEUVERS  # noqa: E402

LETTER = {arm: "ABCD"[i] for i, arm in enumerate(ARMS)}
CLASS = {0: "m", 1: "c", 2: "t"}
TURN_OFFSET = {"left": 1, "through": 2, "right": 3}
INCLUDES = SIM / "python_simulation" / "includes"


def line_marking(roads_path=INCLUDES / "roads.geojson", areas_path=INCLUDES / "areas.geojson"):
    roads = {f["properties"]["id"]: f for f in json.loads(Path(roads_path).read_text())["features"]}
    areas = json.loads(Path(areas_path).read_text())["features"]
    stop = {}
    for f in areas:
        if f["properties"].get("kind") == "stopline":
            (x1, y1), (x2, y2) = f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][-1]
            stop[f["properties"]["arm"]] = (x1, y1, x2, y2)
    exit_ = {}
    for arm in ARMS:
        f = roads[f"{arm}_out"]
        (x0, y0), (x1, y1) = f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1]
        ux, uy = x1 - x0, y1 - y0
        n = math.hypot(ux, uy)
        ux, uy = ux / n, uy / n
        vx, vy = -uy, ux                      # left of the outward direction (inbound side)
        w = f["properties"]["width"]
        exit_[arm] = (x0, y0, x0 - vx * w, y0 - vy * w)
    lines = ["stopline,,,,", "Direction,X1,Y1,X2,Y2"]
    lines += [f"{LETTER[a]},{s[0]:.3f},{s[1]:.3f},{s[2]:.3f},{s[3]:.3f}" for a, s in stop.items()]
    lines += [",,,,", "exitline,,,,", "Direction,X1,Y1,X2,Y2"]
    lines += [f"{LETTER[a]},{s[0]:.3f},{s[1]:.3f},{s[2]:.3f},{s[3]:.3f}" for a, s in exit_.items()]
    return "\r\n".join(lines) + "\r\n"


def signal_table(tick_cycle, tick_phase, tick_yellow, t0):
    """AMIT signal intervals from the recorded phase per tick (PHASES = LVL, HDT, none)."""
    green = {0: ("NE", "SW"), 1: ("SE", "NW")}
    rows, last = [], None
    for cycle, phase, yellow in zip(tick_cycle, tick_phase, tick_yellow):
        state = {}
        for arm in ARMS:
            if phase in green and arm in green[phase]:
                s = "g"
            elif yellow in green and arm in green[yellow]:
                s = "y"
            else:
                s = "r"
            for turn in "LTR":
                state[LETTER[arm] + turn] = s
        key = tuple(state[c] for c in sorted(state))
        t = (cycle - 1) * DT - t0
        if key != last:
            rows.append({"Timestep": round(max(t, 0.0), 2), **state})
            last = key
    sig = pd.DataFrame(rows)
    end = (tick_cycle[-1]) * DT - t0
    sig.insert(1, "Duration", np.round(np.diff(np.append(sig["Timestep"].to_numpy(), end)), 2))
    cols = ["Timestep", "Duration"] + [l + t for l in "ABCD" for t in "LTR"]
    return sig[cols]


def trj_table(arr, t0):
    """10 Hz AMIT trajectory table from TrajectoryRecorder arrays."""
    df = pd.DataFrame({k: arr[k] for k in ("cycle", "vid", "x", "y", "heading", "speed", "accel",
                                           "length", "half_width", "kind", "arm", "maneuver")})
    df["t"] = (df["cycle"] - 1) * DT - t0
    out = []
    for vid, g in df.groupby("vid", sort=False):
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        grid = np.round(np.arange(math.ceil(t[0] * 10) / 10, t[-1] + 1e-9, 0.1), 1)
        grid = grid[grid > 0]
        if len(grid) < 2:
            continue
        x = np.interp(grid, t, g["x"].to_numpy())
        y = np.interp(grid, t, g["y"].to_numpy())
        h = np.interp(grid, t, np.unwrap(g["heading"].to_numpy()))
        v = np.interp(grid, t, g["speed"].to_numpy())
        a = np.interp(grid, t, g["accel"].to_numpy())
        L = float(g["length"].iloc[0])
        W = 2 * float(g["half_width"].iloc[0])
        arm = ARMS[int(g["arm"].iloc[0])]
        turn = MANEUVERS[int(g["maneuver"].iloc[0])]
        out_arm = ARMS[(ARMS.index(arm) + TURN_OFFSET[turn]) % 4]
        out.append(pd.DataFrame({
            "Timestep": grid, "Frame": np.round(grid * 10).astype(int), "Vehicle ID": int(vid),
            "Front X": x + 0.5 * L * np.cos(h), "Front Y": y + 0.5 * L * np.sin(h),
            "Rear X": x - 0.5 * L * np.cos(h), "Rear Y": y - 0.5 * L * np.sin(h),
            "Length": L, "Width": W, "Speed": v, "Acceleration": a,
            "intersection in": LETTER[arm] + "I", "intersection out": LETTER[out_arm] + "O",
            "class": CLASS[int(g["kind"].iloc[0])]}))
    trj = pd.concat(out, ignore_index=True).sort_values(["Frame", "Vehicle ID"])
    return trj


def heading_check(arr):
    """Median angle between recorded heading and displacement of moving vehicles (rad)."""
    df = pd.DataFrame({k: arr[k] for k in ("vid", "cycle", "x", "y", "heading", "speed")})
    df = df.sort_values(["vid", "cycle"])
    dx = df.groupby("vid")["x"].diff().to_numpy()
    dy = df.groupby("vid")["y"].diff().to_numpy()
    ok = np.hypot(dx, dy) > 0.5
    diff = np.angle(np.exp(1j * (np.arctan2(dy[ok], dx[ok]) - df["heading"].to_numpy()[ok])))
    return float(np.median(np.abs(diff)))


def run(seed, ticks, warmup, sim_kwargs):
    sim = LeVanLuongSimulator(seed=seed, **sim_kwargs)
    for _ in range(int(round(warmup / DT))):
        sim.step()
    rec = TrajectoryRecorder(sim)
    for _ in range(ticks):
        sim.step()
        rec.record()
    arr = rec.arrays()
    t0 = (sim.cycle - ticks) * DT
    return arr, t0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--ticks", type=int, default=4500)
    parser.add_argument("--warmup", type=float, default=120.0)
    parser.add_argument("--sim-config", type=Path)
    args = parser.parse_args()
    kwargs = json.loads(args.sim_config.read_text()) if args.sim_config else {}
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.out.name
    lm = line_marking(kwargs.get("roads_path", INCLUDES / "roads.geojson"))
    for seed in args.seeds:
        arr, t0 = run(seed, args.ticks, args.warmup, kwargs)
        check = heading_check(arr)
        if check > 0.3:
            raise SystemExit(f"heading does not follow motion (median {check:.2f} rad)")
        rec = f"{name}-{seed:02d}"
        trj_table(arr, t0).to_csv(args.out / f"{rec}_TRJ.csv", index=False, float_format="%.3f")
        (args.out / f"{rec}_linemarking.csv").write_text(lm)
        signal_table(arr["tick_cycle"], arr["tick_phase"], arr["tick_yellow"], t0).to_csv(
            args.out / f"{rec}_signal.csv", index=False)
        print(rec, "heading check", round(check, 3), flush=True)
    (args.out / "sim_config.json").write_text(json.dumps(
        {"seeds": args.seeds, "ticks": args.ticks, "warmup": args.warmup, "kwargs": kwargs},
        indent=1, default=str))


if __name__ == "__main__":
    main()
