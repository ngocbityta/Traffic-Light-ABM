#!/usr/bin/env python3
"""Inspect GAMA driving state for paired Le Van Luong vehicles at one tick."""

import argparse
import asyncio
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
from python_simulation.demand import demand_parameters, generate_demand


async def main(args):
    with tempfile.TemporaryDirectory(prefix="lvl_mobil_probe_") as directory:
        root = Path(directory)
        demand = root / "demand.csv"
        generate_demand(demand, args.seed, args.tick)
        manager = GamaServerManager(port=args.port, workdir=str(root / "gama_ws"),
                                    log_file=str(root / "gama.log"))
        env = None
        try:
            manager.start(reuse_running=False)
            env = GamaTrafficEnv(
                port=args.port,
                model_path=str(ROOT / "gama-simulation/simulation/models/LeVanLuong.gaml"),
                experiment="python_parity", verbose=False)
            await env.setup()
            await env.load_experiment(parameters=[
                *demand_parameters(demand),
                {"name": "traffic_seed", "type": "float", "value": str(float(args.seed))}])
            await env.step_cycles(args.tick)
            print("population_order:", await env._exec_expression(
                '"" + (vehicle collect each.demand_row_id)'))
            if args.road_id:
                road = f'(road first_with (each.rid = "{args.road_id}"))'
                for label, expression in (
                    ("source_node", f'{road}.source_node'),
                    ("target_node", f'{road}.target_node'),
                    ("roads_in_source", f'intersection({road}.source_node).roads_in'),
                ):
                    try:
                        value = await env._exec_expression(f'"" + {expression}')
                    except Exception as exc:
                        value = f"ERROR {exc}"
                    print(f"road={args.road_id} {label}={value}")
                for lane in range(args.lanes):
                    try:
                        ordering = await env._exec_expression(
                            f'"" + {road}.vehicle_ordering[{lane}]')
                    except Exception as exc:
                        ordering = f"ERROR {exc}"
                    print(f"road={args.road_id} lane={lane} ordering={ordering}")
            for row_id in args.row_ids:
                vehicle = f"(vehicle first_with (each.demand_row_id = {row_id}))"
                print(f"row_id={row_id}")
                for field in ("lowest_lane", "allowed_lanes", "right_side_driving",
                              "leading_distance", "leading_vehicle", "follower",
                              "current_road", "next_road", "current_target",
                              "time_since_lane_change", "lane_change_cooldown",
                              "acc_gain_threshold", "acc_bias"):
                    try:
                        value = await env._exec_expression(f'"" + {vehicle}.{field}')
                    except Exception as exc:
                        value = f"ERROR {exc}"
                    print(f"  {field}: {value}")
        finally:
            if env is not None:
                try:
                    env.disconnect()
                except Exception:
                    pass
            manager.stop(force=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tick", type=int, default=85)
    parser.add_argument("--row-ids", type=int, nargs="+", default=[4, 6, 10])
    parser.add_argument("--road-id", choices=("NE_in", "SE_in", "SW_in", "NW_in",
                                              "SW_NW", "NE_SW", "SE_NW", "NW_SE"))
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--port", type=int, default=6994)
    options = parser.parse_args()
    if options.tick < 1 or min(options.row_ids) < 0 or options.lanes < 1:
        parser.error("tick must be positive and row IDs non-negative")
    asyncio.run(main(options))
