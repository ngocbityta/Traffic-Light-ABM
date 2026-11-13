#!/usr/bin/env python3
"""Pair Le Van Luong vehicles by CSV row and inspect GAMA driving parameters."""

import argparse
import asyncio
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON_SIM_ROOT = ROOT / "python-simulation" / "simulation"
sys.path.insert(0, str(PYTHON_SIM_ROOT))
sys.path.insert(0, str(ROOT / "train" / "logic"))

from train.logic.gama_env import GamaTrafficEnv
from train.logic.server_manager import GamaServerManager
from python_simulation.python_sim import LeVanLuongSimulator
from python_simulation.demand import generate_demand, demand_parameters


SNAPSHOT_EXPR = ('"" + (vehicle collect ("" + each.demand_row_id + ":" + '
                 'each.speed + ":" + each.lowest_lane + ":" + '
                 '(each.current_road = nil ? "nil" : road(each.current_road).rid) + ":" + '
                 'each.distance_to_current_target + ":" + each.max_deceleration + ":" + '
                 'each.time_headway + ":" + each.max_acceleration + ":" + '
                 'each.delta_idm + ":" + each.speed_coeff + ":" + '
                 'each.acc_bias + ":" + each.max_safe_deceleration + ":" + '
                 'each.lane_change_cooldown + ":" + each.time_since_lane_change))')
PATTERN = re.compile(r'(-?\d+):([\d.Ee+-]+):(-?\d+):([A-Za-z0-9_]+):'
                     r'([\d.Ee+-]+):([\d.Ee+-]+):([\d.Ee+-]+):'
                     r'([\d.Ee+-]+):([\d.Ee+-]+):([\d.Ee+-]+):'
                     r'([\d.Ee+-]+):([\d.Ee+-]+):([\d.Ee+-]+):([\d.Ee+-]+)')


async def main(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lvl_micro_") as directory:
        demand = Path(directory) / "demand.csv"
        digest = generate_demand(demand, args.seed, args.horizon)
        python = LeVanLuongSimulator(seed=args.seed, demand_file=demand)
        manager = GamaServerManager(port=args.port, workdir=str(Path(directory) / "gama_ws"),
                                    log_file=str(output.with_suffix(".gama.log")))
        gama = None
        try:
            manager.start(reuse_running=False)
            gama = GamaTrafficEnv(port=args.port,
                                  model_path=str(ROOT / "gama-simulation/simulation/models/LeVanLuong.gaml"),
                                  experiment="python_parity", verbose=False)
            await gama.setup()
            await gama.load_experiment(parameters=[
                *demand_parameters(demand),
                {"name": "traffic_seed", "type": "float", "value": str(float(args.seed))}])
            with output.open("w") as stream:
                for tick in range(0, args.horizon + 1, args.sample_every):
                    raw = str(await gama._exec_expression(SNAPSHOT_EXPR))
                    reference = {}
                    for match in PATTERN.finditer(raw):
                        (rid, speed, lane, road, remaining, decel, headway, accel,
                         delta, coeff, bias, safe_decel, cooldown, since_lc) = match.groups()
                        reference[int(rid)] = dict(speed=float(speed), lane=int(lane), road=road,
                                                   remaining=float(remaining), decel=float(decel),
                                                   headway=float(headway), accel=float(accel),
                                                   delta=float(delta), coeff=float(coeff),
                                                   bias=float(bias), safe_decel=float(safe_decel),
                                                   cooldown=float(cooldown), since_lc=float(since_lc))
                    observed = {v.demand_row_id: v for v in python.vehicles}
                    for rid in sorted(reference.keys() | observed.keys()):
                        v = observed.get(rid)
                        ga = reference.get(rid)
                        stream.write(json.dumps(dict(tick=tick, row_id=rid,
                            python_position=v.position if v else None,
                            python_speed=v.speed if v else None,
                            python_lane=v.lane if v else None,
                            python_road=v.road if v else None,
                            gama=ga)) + "\n")
                    if tick < args.horizon:
                        python.step(args.sample_every)
                        await gama.step_cycles(args.sample_every)
        finally:
            manager.stop(force=True)
            if gama is not None:
                try:
                    gama.disconnect()
                except Exception:
                    pass
    manifest = dict(output=str(output), seed=args.seed, horizon=args.horizon,
                    sample_every=args.sample_every, demand_sha256=digest,
                    python_sim_sha256=hashlib.sha256(
                        (ROOT / "python-simulation/simulation/python_simulation/python_sim.py").read_bytes()).hexdigest(),
                    demand_source_sha256=hashlib.sha256(
                        (ROOT / "python-simulation/simulation/python_simulation/demand.py").read_bytes()).hexdigest(),
                    gama_model_sha256=hashlib.sha256(
                        (ROOT / "gama-simulation/simulation/models/LeVanLuong.gaml").read_bytes()).hexdigest())
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=50)
    parser.add_argument("--port", type=int, default=6991)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/lvl_micro.jsonl"))
    options = parser.parse_args()
    if options.horizon <= 0 or options.sample_every <= 0 or options.horizon % options.sample_every:
        parser.error("Horizon must be positive and divisible by sample-every")
    asyncio.run(main(options))
