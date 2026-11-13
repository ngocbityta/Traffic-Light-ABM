#!/usr/bin/env python3
"""Compare the same demand row's vehicle in Python and GAMA by simulation tick."""

import argparse
import asyncio
import csv
import json
import math
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train" / "logic"))

from train.logic.demand import demand_parameters, generate_demand
from train.logic.gama_env import GamaTrafficEnv
from train.logic.python_env import PythonTrafficEnv
from train.logic.server_manager import GamaServerManager


SNAPSHOT_EXPR = ('"" + (active_vehicles collect ("" + each.demand_row_id + ":" + '
                 'each.location.x + ":" + each.location.y + ":" + '
                 'each.speed + ":" + each.lowest_lane + ":" + '
                 '(each.current_road = nil ? "nil" : road(each.current_road).road_key)))')
PENDING_EXPR = '"" + pending_by_origin'
PORTAL_OWNERS_EXPR = ('"" + ((network_nodes["F_NS1"].merge_owners + '
                      'network_nodes["F_SN1"].merge_owners) collect each.demand_row_id)')
ORIGIN_COORDS = ((998.0, 10.0), (1002.0, 1990.0),
                 (1990.0, 998.0), (10.0, 1002.0))


def parse_snapshot(raw):
    vehicles = {}
    pattern = r'(-?\d+):([-\d.Ee]+):([-\d.Ee]+):([-\d.Ee]+):(-?\d+):([A-Za-z0-9_]+)'
    for match in re.finditer(pattern, str(raw)):
        request_id, x, y, speed, lane, road_key = match.groups()
        vehicles[int(request_id)] = (float(x), float(y), float(speed), int(lane), road_key)
    return vehicles


async def step_to(env, target, controller):
    if controller == "fixed":
        current = (env._cycle if isinstance(env, PythonTrafficEnv) else
                   int(float(await env._exec_expression("cycle"))))
        await env.step_cycles(target - current)
        return
    state = await env.read_state()
    while state["cycle"] < target:
        if state["dqn_waiting_action"]:
            await env.set_action(state["decision_count"] % 4)
            await env.step_cycles(1)
        else:
            await env.step_cycles(min(target - state["cycle"],
                                      state["next_decision_in_cycles"]))
        state = await env.read_state()


async def main(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="micro_parity_") as td:
        temp = Path(td)
        demand = temp / "demand.csv"
        digest = generate_demand(demand, args.seed, args.demand_horizon or args.horizon)
        requests = list(csv.reader(demand.open()))
        python = PythonTrafficEnv(controller_mode=args.controller, state_mode="spatial",
                                  reward_mode="r1", verbose=False)
        await python.reset(seed=args.seed, parameters=demand_parameters(demand))
        manager = GamaServerManager(port=args.port, workdir=str(temp / "gama_ws"),
                                    log_file=str(output.with_suffix(".gama.log")))
        gama = None
        try:
            manager.start(reuse_running=False)
            gama = GamaTrafficEnv(port=args.port,
                                  model_path=str(ROOT / "train/benchmarks/gama/models" /
                                      ("intersection-fixed-time.gaml" if args.controller == "fixed"
                                       else "intersection-dqn-python.gaml")),
                                  experiment=("python_parity" if args.controller == "fixed"
                                              else "dqn_server_experiment"),
                                  state_mode="spatial", reward_mode="r1",
                                  fast_training=True, verbose=False)
            await gama.setup()
            if args.controller == "fixed":
                await gama.load_experiment(parameters=[
                    *demand_parameters(demand),
                    {"name": "traffic_seed", "type": "float", "value": str(float(args.seed))},
                    {"name": "fast_training", "type": "bool", "value": "true"}])
            else:
                await gama.reset(seed=args.seed, parameters=demand_parameters(demand))
            geometry = {}
            for road_name in ("N_left", "N_right", "N_uturn", "N_through"):
                geometry[road_name] = float(await gama._exec_expression(
                    f'road(network_roads["{road_name}"]).shape.perimeter'))
            output.with_suffix(".geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
            with (output.open("w") as stream,
                  output.with_suffix(".raw.txt").open("w") as raw_stream,
                  output.with_suffix(".state.jsonl").open("w") as state_stream):
                ticks = sorted(set([0, *range(args.first_sample or 0,
                                              args.horizon + 1, args.sample_every)]))
                for tick in ticks:
                    if tick > python._cycle:
                        await step_to(python, tick, args.controller)
                        await step_to(gama, tick, args.controller)
                    assert python._cycle == tick
                    assert int(float(await gama._exec_expression("cycle"))) == tick
                    raw = await gama._exec_expression(SNAPSHOT_EXPR)
                    raw_stream.write(f"{tick}: {raw}\n")
                    ga = parse_snapshot(raw)
                    blockers = [dict(request_id=v.demand_row_id, position=v.position,
                                     maneuver=v.maneuver, yellow_commit=v.yellow_commit)
                                for v in python.vehicles if v.signal and (
                                    949.5 <= v.position < v.connector_end + 5 + v.length / 2 or
                                    (v.yellow_commit and v.position < 949.5 and
                                     949.5-v.position <= 4.5+v.length/2+2))]
                    state_stream.write(json.dumps(dict(
                        tick=tick, python_phase=python._phase,
                        python_phase_ticks=python._phase_ticks,
                        gama_phase=int(float(await gama._exec_expression("global_phase"))),
                        gama_phase_timer=float(await gama._exec_expression(
                            "global_counter" if args.controller == "fixed" else "phase_timer")),
                        python_pending_by_origin=[len(q) for q in python._pending],
                        gama_pending_by_origin=[int(x) for x in re.findall(
                            r'\d+', str(await gama._exec_expression(PENDING_EXPR)))],
                        python_portal_owners={name: (owner.demand_row_id if owner else None)
                                              for name, owner in python._flyover_portal_owner.items()},
                        gama_portal_owner_ids=[int(x) for x in re.findall(
                            r'\d+', str(await gama._exec_expression(PORTAL_OWNERS_EXPR)))],
                        python_stopped_integral=python._integrals["stopped_time_integral"],
                        gama_stopped_integral=float(await gama._exec_expression(
                            "stopped_time_integral")),
                        gama_clear=str(await gama._exec_expression(
                            "signal_junction_clear()")),
                        python_clear=python._junction_clear(), blockers=blockers)) + "\n")
                    py = {v.demand_row_id: v for v in python.vehicles}
                    if args.probe_id is not None:
                        probe = py.get(args.probe_id)
                        probe_expr = ('"" + (active_vehicles where '
                                      f'(each.demand_row_id = {args.probe_id}) collect '
                                      '("" + each.speed + ":" + each.max_speed + ":" + '
                                      'each.measured_speed + ":" + '
                                      'each.distance_to_current_target + ":" + '
                                      'each.leading_distance))')
                        state_stream.write(json.dumps(dict(
                            tick=tick, probe_id=args.probe_id,
                            python_probe=(dict(position=probe.position,
                                               speed=probe.speed) if probe else None),
                            gama_probe=str(await gama._exec_expression(probe_expr)))) + "\n")
                    for request_id in sorted(py.keys() | ga.keys()):
                        request = requests[request_id]
                        origin = int(request[1])
                        vehicle = py.get(request_id)
                        snapshot = ga.get(request_id)
                        gx, gy, gs, gl, gr = snapshot if snapshot else (None,) * 5
                        px = vehicle.position if vehicle else None
                        gps = (math.hypot(gx - ORIGIN_COORDS[origin][0],
                                          gy - ORIGIN_COORDS[origin][1])
                               if snapshot and origin < 4 else None)
                        record = dict(tick=tick, request_id=request_id, origin=origin,
                                      python_present=vehicle is not None,
                                      gama_present=snapshot is not None,
                                      python_position=px, gama_distance_from_source=gps,
                                      position_error=abs(px-gps) if px is not None and gps is not None else None,
                                      python_speed=vehicle.speed if vehicle else None,
                                      gama_speed=gs, python_lane=vehicle.lane if vehicle else None,
                                      gama_lane=gl, gama_road=gr)
                        stream.write(json.dumps(record) + "\n")
        finally:
            manager.stop(force=True)
            if gama is not None:
                try:
                    gama.disconnect()
                except Exception:
                    pass
    print(json.dumps(dict(output=str(output), seed=args.seed, horizon=args.horizon,
                          demand_sha256=digest)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1002)
    parser.add_argument("--controller", choices=("fixed", "dqn"), default="fixed")
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--demand-horizon", type=int, default=None,
                        help="Generate the full benchmark's demand while inspecting an earlier prefix")
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--first-sample", type=int, default=0)
    parser.add_argument("--probe-id", type=int, default=None)
    parser.add_argument("--port", type=int, default=6980)
    parser.add_argument("--output", type=Path, default=Path("reports/python_parity_20261109/micro_fixed.jsonl"))
    options = parser.parse_args()
    if (options.horizon <= 0 or options.sample_every <= 0 or
            options.first_sample < 0 or options.first_sample > options.horizon or
            (options.horizon - options.first_sample) % options.sample_every or
            (options.demand_horizon is not None and options.demand_horizon < options.horizon)):
        parser.error("horizon must be positive and divisible by sample-every")
    asyncio.run(main(options))
