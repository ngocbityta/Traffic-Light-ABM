#!/usr/bin/env python3
"""
==============================================================================
EVALUATE GAMA AGENTS - Danh gia chinh sach da hoc (greedy policy)
==============================================================================
Chay N episode voi weights da train, moi 10 cycle doc metrics 1 lan
(giong hang "METRIC_CSV" cua 3 model kia) va xuat CSV cung schema voi
compare_traffic_control.py:

    cycle, time_sec, throughput, avg_waiting_time, queue_length,
    avg_speed, avg_travel_time, max_waiting_time, episode

Ket qua: comparison_results/dqn_metrics.csv
-> co the ghep vao pipeline so sanh cac thuat toan.

Vi du:
    venv/bin/python evaluate.py --weights run/train_YYYYMMDD_HHMMSS/dqn_weights.pt
==============================================================================
"""

import argparse
import asyncio
import csv
import json
import math
import os
import time

import torch

from .three_dqn_agent import ThreeDQNAgent, FlatDQNAgent, ARCHITECTURE as THREE_ARCHITECTURE
from .rainbow_agent import (RainbowAgent, ARCHITECTURE as RAINBOW_ARCHITECTURE,
                           LEGACY_ARCHITECTURE, PREVIOUS_ARCHITECTURE)
from .hybrid_agent import SpatialHybridAgent, ARCHITECTURE as HYBRID_ARCHITECTURE
from .tabular_agent import TabularQAgent, ARCHITECTURE as TABULAR_ARCHITECTURE
from .gama_env import GamaTrafficEnv, featurize
from .python_env import PythonTrafficEnv
from .lvl_env import LVLGamaEnv, LVLPythonEnv
from .server_manager import GamaServerManager
from .demand import generate_demand, demand_parameters
from python_simulation.demand import generate_demand as generate_lvl_demand
from python_simulation.demand import demand_parameters as lvl_demand_parameters

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPARISON_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "comparison_results")


def write_sample(writer, state, episode, horizon, step_seconds=.1):
    if state["cycle"] % 10 != 0 or state["cycle"] >= horizon:
        return False
    writer.writerow({
        "episode": episode, "cycle": state["cycle"],
        "time_sec": round(state["cycle"] * step_seconds, 1),
        "throughput": state["total_vehicles_passed"],
        "avg_waiting_time": state["average_waiting_time"],
        "queue_length": state["current_queue_length"],
        "avg_speed": state["average_speed"],
        "avg_travel_time": state["average_travel_time"],
        "max_waiting_time": state["max_waiting_time"],
    })
    return True


async def evaluate_episode(env, agent, seed, horizon, writer, episode,
                           collect_samples=True, parameters=None):
    state = await env.reset(seed=seed, parameters=parameters)
    pending = None
    reward = 0.0
    decisions = rows = 0
    while state["cycle"] < horizon:
        if state["dqn_waiting_action"]:
            if pending is not None:
                reward += env.transition_reward(pending, state)
            action = agent.act(featurize(state), greedy=True, action_mask=state["action_mask"])
            pending = state
            await env.set_action(action)
            await env.step_cycles(1)
            decisions += 1
        else:
            count = min(state["next_decision_in_cycles"], horizon - state["cycle"])
            if collect_samples:
                count = min(count, 10 - state["cycle"] % 10)
            if count <= 0:
                raise RuntimeError("Invalid DQN evaluation schedule")
            await env.step_cycles(count)
        # Always use GAMA's current schedule, including extensions of all-red.
        state = await env.read_state()
        if collect_samples:
            rows += int(write_sample(writer, state, episode, horizon,
                                     getattr(env, "step_seconds", .1)))
    if pending is not None:
        if "dtse" not in state and "spatial" not in state:
            state = await env.read_state(force_dtse=True)
        reward += env.transition_reward(pending, state)
    return state, reward, decisions, rows


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained DQN traffic controller")
    p.add_argument("--weights", type=str, required=True,
                   help="File dqn_weights.pt / best_weights.pt tu train.py")
    p.add_argument("--scenario", choices=("lvl", "legacy"), default="lvl")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--episode-horizon", type=int, default=8000)
    p.add_argument("--seed-base", type=int, default=20_000_000,
                   help="Seed danh gia (khac voi seed train de tranh overfit)")
    p.add_argument("--demand-mode", choices=("paired", "model"), default="paired")
    p.add_argument("--port", type=int, default=6868)
    p.add_argument("--backend", choices=("gama", "python"), default="python",
                   help="Simulation engine for the selected scenario")
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--start-server", action="store_true")
    p.add_argument("--keep-server", action="store_true")
    p.add_argument("--experiment", type=str, default="dqn_server_experiment",
                   help="Ten experiment trong file GAML")
    p.add_argument("--output", type=str,
                   default=os.path.join(COMPARISON_DIR, "dqn_metrics.csv"))
    args = p.parse_args()
    if args.backend == "python" and args.start_server:
        p.error("--start-server is only valid with --backend gama")
    if args.backend == "python" and args.demand_mode != "paired":
        p.error("Python backend requires paired CSV demand; GAMA's live RNG is different")
    if args.scenario == "lvl" and args.demand_mode != "paired":
        p.error("Le Van Luong evaluation requires paired demand")
    return args


async def amain(args):
    output_dir = os.path.dirname(os.path.abspath(args.output))
    manager = (GamaServerManager(
        port=args.port,
        workdir=os.path.join(output_dir, "server_ws", str(args.port)),
        log_file=os.path.join(output_dir, f"server_{args.port}.log"))
        if args.start_server else None)
    env = None
    try:
        if manager is not None:
            manager.start()
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
        run_summary = os.path.join(os.path.dirname(os.path.abspath(args.weights)),
                                   "training_summary.json")
        if os.path.exists(run_summary):
            with open(run_summary) as summary_file:
                trained_scenario = json.load(summary_file).get("scenario", "legacy")
            if trained_scenario != args.scenario:
                raise ValueError(f"Checkpoint belongs to scenario {trained_scenario}, "
                                 f"but evaluation requested {args.scenario}")
        architecture = checkpoint.get("architecture")
        is_spatial = architecture in (RAINBOW_ARCHITECTURE, LEGACY_ARCHITECTURE,
                                      PREVIOUS_ARCHITECTURE,
                                      HYBRID_ARCHITECTURE)
        reward_options = dict(checkpoint.get("reward_settings", {}))
        if is_spatial and "r1_ref" not in reward_options:
            reward_options["r1_ref"] = checkpoint.get("r1_ref", 50.0)
        if args.scenario == "lvl":
            env_type = LVLPythonEnv if args.backend == "python" else LVLGamaEnv
        else:
            env_type = PythonTrafficEnv if args.backend == "python" else GamaTrafficEnv
        env = env_type(host=args.host, port=args.port,
                             experiment=args.experiment,
                             reward_mode=checkpoint.get("reward_mode", "reference"),
                             state_mode="spatial" if is_spatial else "dtse",
                             **reward_options)
        await env.setup()
        if args.backend == "python":
            print("[Eval] Python movement is approximate; compare metrics with GAMA before scientific use")
        agent_class = ({THREE_ARCHITECTURE: ThreeDQNAgent,
                        "gama-four-phase-flat-dqn-v1": FlatDQNAgent,
                        TABULAR_ARCHITECTURE: TabularQAgent,
                        RAINBOW_ARCHITECTURE: RainbowAgent,
                        LEGACY_ARCHITECTURE: RainbowAgent,
                        PREVIOUS_ARCHITECTURE: RainbowAgent,
                        HYBRID_ARCHITECTURE: SpatialHybridAgent}).get(architecture)
        if agent_class is None:
            raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
        if architecture in (RAINBOW_ARCHITECTURE, LEGACY_ARCHITECTURE,
                            PREVIOUS_ARCHITECTURE):
            agent = agent_class(buffer_size=1, warmup=1, batch_size=1,
                                atoms=checkpoint["atoms"], v_min=checkpoint["v_min"],
                                v_max=checkpoint["v_max"], r1_ref=reward_options["r1_ref"],
                                reward_mode=checkpoint["reward_mode"],
                                reward_settings=reward_options,
                                aux_size=17 if architecture == RAINBOW_ARCHITECTURE else 13)
        elif architecture == HYBRID_ARCHITECTURE:
            agent = agent_class(buffer_size=1, warmup=1, batch_size=1,
                                pressure_lambda=checkpoint["pressure_lambda"],
                                reward_mode=checkpoint["reward_mode"],
                                reward_settings=reward_options)
        else:
            agent = agent_class(buffer_size=1, warmup=1)
        agent.load(args.weights)
        env.reward_mode = agent.reward_mode
        agent.eps = 0.0
        print(f"[Eval] Da load weights: {args.weights} | policy = GREEDY (eps=0)")

        os.makedirs(output_dir, exist_ok=True)
        fieldnames = ["episode", "cycle", "time_sec", "throughput", "avg_waiting_time",
                      "queue_length", "avg_speed", "avg_travel_time", "max_waiting_time"]
        n_rows = 0
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for ep in range(1, args.episodes + 1):
                seed = args.seed_base + ep
                t0 = time.time()
                parameters = None
                if args.demand_mode == "paired":
                    demand_path = os.path.join(os.path.dirname(os.path.abspath(args.output)),
                                               "demand", f"seed_{seed}_h{args.episode_horizon}.csv")
                    if args.scenario == "lvl":
                        generate_lvl_demand(demand_path, seed, args.episode_horizon)
                        parameters = lvl_demand_parameters(demand_path)
                    else:
                        generate_demand(demand_path, seed, args.episode_horizon)
                        parameters = demand_parameters(demand_path)
                state, ep_reward, decisions, count = await evaluate_episode(
                    env, agent, seed, args.episode_horizon, writer, ep,
                    parameters=parameters)
                n_rows += count

                def measured(value, unit="s"):
                    return f"{value:.2f}{unit}" if math.isfinite(value) else "n/a"
                print(f"[Eval] Ep {ep} | seed={seed} | reward={ep_reward:.1f} "
                      f"| decisions={decisions} | passed={state['total_vehicles_passed']} "
                      f"| avg_wait={measured(state['average_waiting_time'])} "
                      f"| max_wait={measured(state['max_waiting_time'])} "
                      f"| avg_speed={state['average_speed']:.2f}km/h "
                      f"| avg_travel={measured(state['average_travel_time'])} "
                      f"| {time.time()-t0:.1f}s")

        print(f"\n[Eval] Da ghi {n_rows} dong -> {args.output}")
        if args.scenario == "legacy":
            print("[Eval] Ghep voi 3 model khac: python3 compare_traffic_control.py "
                  "(hoac doc README.md)")
    finally:
        try:
            if env is not None:
                await env.close()
        finally:
            if manager is not None and not args.keep_server:
                manager.stop()


if __name__ == "__main__":
    asyncio.run(amain(parse_args()))
