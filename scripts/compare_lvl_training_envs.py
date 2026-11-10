#!/usr/bin/env python3
"""Replay one Le Van Luong DQN action sequence through GAMA and Python."""

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dqn_py"))
from lvl_env import LVLGamaEnv, LVLPythonEnv
from gama_env import featurize
from hybrid_agent import SpatialHybridAgent, ARCHITECTURE as HYBRID_ARCHITECTURE
from server_manager import GamaServerManager
from LeVanLuongIntersection.demand import demand_parameters, generate_demand


async def main(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    demand = output.with_name(output.stem + "_demand.csv")
    digest = generate_demand(demand, args.seed, args.horizon)
    manager = GamaServerManager(port=args.port,
                                workdir=str(output.parent / "lvl_train_ws"),
                                log_file=str(output.with_suffix(".gama.log")))
    gama = LVLGamaEnv(port=args.port, reward_mode=args.reward_mode, state_mode="spatial",
                      verbose=False)
    python = LVLPythonEnv(reward_mode=args.reward_mode, state_mode="spatial", verbose=False)
    rows = []
    gama_wall = python_wall = 0.0
    agent = None
    if args.weights:
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != HYBRID_ARCHITECTURE:
            raise ValueError("--weights currently supports a hybrid checkpoint")
        agent = SpatialHybridAgent(buffer_size=1, warmup=1, batch_size=1,
                                   pressure_lambda=checkpoint["pressure_lambda"],
                                   reward_mode=checkpoint["reward_mode"],
                                   reward_settings=checkpoint["reward_settings"])
        agent.load(args.weights)
        agent.eps = 0.0
    try:
        manager.start(reuse_running=False)
        await gama.setup()
        await python.setup()
        params = demand_parameters(demand)
        gs = await gama.reset(seed=args.seed, parameters=params)
        ps = await python.reset(seed=args.seed, parameters=params)
        started = time.perf_counter()
        await gama.step_cycles(50)
        gama_wall += time.perf_counter() - started
        started = time.perf_counter()
        await python.step_cycles(50)
        python_wall += time.perf_counter() - started
        gs = await gama.read_state()
        ps = await python.read_state()
        for decision in range(args.horizon // 50 - 1):
            action = (0, 2, 2, 0)[decision % 4]
            if agent:
                gama_action = agent.act(featurize(gs), greedy=True,
                                        action_mask=gs["action_mask"])
                python_action = agent.act(featurize(ps), greedy=True,
                                          action_mask=ps["action_mask"])
            elif args.policy == "queue":
                gama_action = 0 if gs["queue_ns"] >= gs["queue_ew"] else 2
                python_action = 0 if ps["queue_ns"] >= ps["queue_ew"] else 2
            else:
                gama_action = python_action = action
            if not gs["dqn_waiting_action"] or not ps["dqn_waiting_action"]:
                mode = await gama._exec_expression("train_mode")
                raise RuntimeError(f"Missing decision at {gs['cycle']}/{ps['cycle']}: "
                                   f"GAMA={gs['dqn_waiting_action']}, Python={ps['dqn_waiting_action']}, "
                                   f"GAMA train_mode={mode}")
            if decision == 0 and args.debug_action:
                await gama.set_action(gama_action)
                value = await gama._exec_expression('"" + dqn_action + "|" + dqn_waiting_action')
                raise RuntimeError(f"After GAMA set_action({gama_action}): {value}")
            started = time.perf_counter()
            gs, gr, _, _ = await gama.step(gama_action, horizon=args.horizon,
                                           current_state=gs)
            gama_wall += time.perf_counter() - started
            started = time.perf_counter()
            ps, pr, _, _ = await python.step(python_action, horizon=args.horizon,
                                             current_state=ps)
            python_wall += time.perf_counter() - started
            if gs["cycle"] != ps["cycle"]:
                raise RuntimeError(f"Decision cycles differ: GAMA {gs['cycle']} "
                                   f"waiting={gs['dqn_waiting_action']} next={gs['next_decision_in_cycles']}; "
                                   f"Python {ps['cycle']} waiting={ps['dqn_waiting_action']} "
                                   f"next={ps['next_decision_in_cycles']}")
            rows.append(dict(cycle=gs["cycle"], gama_action=gama_action,
                             python_action=python_action,
                             same_action=int(gama_action == python_action),
                             gama_phase=gs["global_phase"], python_phase=ps["global_phase"],
                             gama_created=gs["total_vehicles_created"],
                             python_created=ps["total_vehicles_created"],
                             gama_passed=gs["total_vehicles_passed"],
                             python_passed=ps["total_vehicles_passed"],
                             gama_queue=gs["current_queue_length"],
                             python_queue=ps["current_queue_length"],
                             gama_reward=gr, python_reward=pr,
                             spatial_mae=float(abs(gs["spatial"] - ps["spatial"]).mean())))
    finally:
        try:
            gama.disconnect()
        except Exception:
            pass
        manager.stop(force=True)
        await python.close()
    if rows:
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(dict(output=str(output), seed=args.seed,
                          horizon=args.horizon, demand_sha256=digest,
                          decisions=len(rows),
                          gama_step_wall_seconds=round(gama_wall, 3),
                          python_step_wall_seconds=round(python_wall, 3),
                          step_speedup=round(gama_wall / python_wall, 2)
                          if python_wall else None,
                          final=rows[-1] if rows else None), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--port", type=int, default=6992)
    parser.add_argument("--reward-mode", choices=("r1", "r2", "r4"), default="r1")
    parser.add_argument("--debug-action", action="store_true")
    parser.add_argument("--weights", type=Path,
                        help="Evaluate a trained hybrid policy independently in both backends")
    parser.add_argument("--policy", choices=("sequence", "queue"), default="sequence",
                        help="Action sequence or independent queue-responsive control")
    parser.add_argument("--output", type=Path,
                        default=Path("reports/lvl_training_parity.csv"))
    options = parser.parse_args()
    if options.horizon < 100 or options.horizon % 50:
        parser.error("Horizon must be a multiple of 50 and at least 100")
    asyncio.run(main(options))
