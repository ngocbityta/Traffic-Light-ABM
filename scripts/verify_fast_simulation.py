#!/usr/bin/env python3
"""Check that R1 fast mode preserves traffic and Rainbow observations."""

import argparse
import asyncio
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.logic.demand import demand_parameters, generate_demand
from train.logic.gama_env import GamaTrafficEnv, featurize
from train.logic.server_manager import GamaServerManager


async def rollout(env, seed, horizon, demand):
    await env.setup()
    records = []
    state = await env.reset(seed=seed, parameters=demand_parameters(demand))
    await env.step_cycles(state["next_decision_in_cycles"])
    state = await env.read_state()
    actions = (0, 2, 1, 3)
    while state["cycle"] < horizon:
        if not state["dqn_waiting_action"]:
            raise RuntimeError("Expected a decision state")
        features = featurize(state)
        action = actions[len(records) % len(actions)]
        next_state, reward, _, info = await env.step(
            action, horizon=horizon, current_state=state)
        positions = str(await env._exec_expression(
            "string(active_vehicles collect [each.name, each.location, "
            "each.measured_speed, each.lowest_lane, each.current_road])"))
        records.append({
            "cycle": state["cycle"], "action": action,
            "mask": state["action_mask"].astype(int).tolist(),
            "grid_sha256": hashlib.sha256(features["grid"].tobytes()).hexdigest(),
            "aux": features["aux"].tolist(),
            "pressure": features["pressure"].tolist(),
            "reward": reward, "next_cycle": next_state["cycle"],
            "bootstrap_valid": info["bootstrap_valid"],
            "positions_sha256": hashlib.sha256(positions.encode()).hexdigest(),
            "passed": next_state["total_vehicles_passed"],
            "created": next_state["total_vehicles_created"],
            "queue": next_state["current_queue_length"],
            "system_time_integral": next_state["system_time_integral"],
        })
        state = next_state
        if not info["bootstrap_valid"]:
            break
    return records


async def compare_pair(port, seed, horizon, demand, managers):
    envs = [GamaTrafficEnv(port=port + i, verbose=False, reward_mode="r1",
                           state_mode="spatial", fast_training=bool(i),
                           reset_mode="fresh") for i in range(2)]
    try:
        return await asyncio.gather(*(rollout(env, seed, horizon, demand)
                                      for env in envs))
    finally:
        await asyncio.gather(*(asyncio.to_thread(manager.stop, force=True)
                               for manager in managers))
        for env in envs:
            env.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=6940)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="gama_fast_verify_") as directory:
        root = Path(directory)
        demand = root / "demand.csv"
        generate_demand(demand, args.seed, args.horizon)
        managers = [GamaServerManager(port=args.port + i,
                                      workdir=str(root / f"ws_{i}"),
                                      log_file=str(root / f"server_{i}.log"))
                    for i in range(2)]
        try:
            for manager in managers:
                manager.start(reuse_running=False)
            ordinary, fast = asyncio.run(compare_pair(
                args.port, args.seed, args.horizon, demand, managers))
        finally:
            for manager in managers:
                manager.stop(force=True)
        errors = []
        for index in range(2):
            contents = (root / f"server_{index}.log").read_text(errors="replace")
            errors.extend(re.findall(
                r"(?m)^(?:java\.lang\.[\w.]+Exception:|Java error:|SimulationError|RuntimeError)",
                contents))
        if errors:
            raise AssertionError(f"GAMA logs contain {len(errors)} Java/runtime errors")
    if len(ordinary) != len(fast):
        raise AssertionError(f"Decision counts differ: {len(ordinary)} vs {len(fast)}")
    for i, (before, after) in enumerate(zip(ordinary, fast)):
        for key in before:
            if key in ("aux", "pressure"):
                if not np.allclose(before[key], after[key], rtol=0, atol=1e-6):
                    raise AssertionError(f"Decision {i}: {key} changed")
            elif key in ("reward", "system_time_integral"):
                if not np.isclose(before[key], after[key], rtol=0, atol=1e-6):
                    raise AssertionError(f"Decision {i}: {key} changed")
            elif before[key] != after[key]:
                raise AssertionError(f"Decision {i}: {key} changed")
    result = {"seed": args.seed, "horizon": args.horizon,
              "decisions": len(ordinary), "traffic_and_r1_identical": True,
              "last_cycle": ordinary[-1]["next_cycle"] if ordinary else 0,
              "server_error_count": 0}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
