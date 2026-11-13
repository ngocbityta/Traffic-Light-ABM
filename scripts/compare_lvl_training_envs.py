#!/usr/bin/env python3
"""Replay one Le Van Luong DQN action sequence through GAMA and Python."""

import argparse
import asyncio
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train" / "logic"))
from train.logic.lvl_env import LVLGamaEnv, LVLPythonEnv
from train.logic.gama_env import featurize
from train.logic.hybrid_agent import SpatialHybridAgent, ARCHITECTURE as HYBRID_ARCHITECTURE
from train.logic.rainbow_agent import RainbowAgent, ARCHITECTURE as RAINBOW_ARCHITECTURE
from train.logic.server_manager import GamaServerManager
from python_simulation.demand import demand_parameters, generate_demand


def validate_thresholds(args):
    limits = (args.max_created_delta, args.max_passed_delta,
              args.max_queue_delta, args.max_reward_relative_error,
              args.max_integral_relative_error, args.max_spatial_mae,
              args.max_aux_mae, args.max_phase_mismatches,
              args.max_action_mismatches)
    if any(not math.isfinite(float(value)) or value < 0 for value in limits):
        raise ValueError("Parity thresholds must be finite and non-negative")


async def main(args):
    validate_thresholds(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    demand = output.with_name(output.stem + "_demand.csv")
    digest = generate_demand(demand, args.seed, args.horizon)
    source_files = (ROOT / "train/logic/lvl_env.py", ROOT / "train/logic/train.py",
                    ROOT / "python-simulation/simulation/python_simulation/python_sim.py",
                    ROOT / "python-simulation/simulation/python_simulation/demand.py",
                    ROOT / "gama-simulation/simulation/models/LeVanLuong.gaml",
                    ROOT / "python-simulation/simulation/python_simulation/includes/roads.geojson",
                    Path(__file__).resolve())
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_files}
    checkpoint_sha256 = (hashlib.sha256(args.weights.read_bytes()).hexdigest()
                         if args.weights else None)
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
        architecture = checkpoint.get("architecture")
        if architecture == HYBRID_ARCHITECTURE:
            agent = SpatialHybridAgent(buffer_size=1, warmup=1, batch_size=1,
                                       pressure_lambda=checkpoint["pressure_lambda"],
                                       reward_mode=checkpoint["reward_mode"],
                                       reward_settings=checkpoint["reward_settings"])
        elif architecture == RAINBOW_ARCHITECTURE:
            agent = RainbowAgent(buffer_size=1, warmup=1, batch_size=1,
                                 atoms=checkpoint["atoms"],
                                 v_min=checkpoint["v_min"],
                                 v_max=checkpoint["v_max"],
                                 reward_mode=checkpoint["reward_mode"],
                                 reward_settings=checkpoint["reward_settings"])
        else:
            raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
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
            g_features = featurize(gs)
            p_features = featurize(ps)
            reward_error = abs(gr - pr) / max(1.0, abs(gr), abs(pr))
            queue_integral_error = abs(gs["queue_time_integral"] - ps["queue_time_integral"]) / max(
                1.0, abs(gs["queue_time_integral"]), abs(ps["queue_time_integral"]))
            system_integral_error = abs(gs["system_time_integral"] - ps["system_time_integral"]) / max(
                1.0, abs(gs["system_time_integral"]), abs(ps["system_time_integral"]))
            weighted_integral_error = abs(gs["weighted_delay_integral"] - ps["weighted_delay_integral"]) / max(
                1.0, abs(gs["weighted_delay_integral"]), abs(ps["weighted_delay_integral"]))
            integral_error = max(queue_integral_error, system_integral_error,
                                 weighted_integral_error)
            spatial_error = float(abs(gs["spatial"] - ps["spatial"]).mean())
            aux_error = float(abs(g_features["aux"] - p_features["aux"]).mean())
            created_error = abs(gs["total_vehicles_created"] - ps["total_vehicles_created"])
            passed_error = abs(gs["total_vehicles_passed"] - ps["total_vehicles_passed"])
            queue_error = abs(gs["current_queue_length"] - ps["current_queue_length"])
            phase_error = int(gs["global_phase"] != ps["global_phase"])
            action_error = int(gama_action != python_action)
            failures = []
            for label, actual, limit in (
                    ("created_abs_error", created_error, args.max_created_delta),
                    ("passed_abs_error", passed_error, args.max_passed_delta),
                    ("queue_abs_error", queue_error, args.max_queue_delta),
                    ("reward_relative_error", reward_error, args.max_reward_relative_error),
                    ("integral_relative_error", integral_error,
                     args.max_integral_relative_error),
                    ("spatial_mae", spatial_error, args.max_spatial_mae),
                    ("aux_mae", aux_error, args.max_aux_mae),
                    ("phase_mismatch", phase_error, args.max_phase_mismatches),
                    ("action_mismatch", action_error, args.max_action_mismatches)):
                if actual > limit:
                    failures.append(label)
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
                             gama_green_cycles=gs["green_cycles"],
                             python_green_cycles=ps["green_cycles"],
                             gama_reward=gr, python_reward=pr,
                             created_abs_error=created_error, passed_abs_error=passed_error,
                             queue_abs_error=queue_error,
                             reward_relative_error=reward_error,
                             queue_integral_relative_error=queue_integral_error,
                             system_integral_relative_error=system_integral_error,
                             weighted_integral_relative_error=weighted_integral_error,
                             spatial_mae=spatial_error, aux_mae=aux_error,
                             phase_mismatch=phase_error, action_mismatch=action_error,
                             gate_pass=int(not failures), gate_failures=";".join(failures)))
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
    failed_rows = [row for row in rows if not row["gate_pass"]]
    summary = dict(output=str(output), seed=args.seed,
                          horizon=args.horizon, demand_sha256=digest,
                          source_sha256=source_hashes,
                          checkpoint=str(args.weights.resolve()) if args.weights else None,
                          checkpoint_sha256=checkpoint_sha256,
                          decisions=len(rows),
                          gate_pass=bool(rows) and not failed_rows,
                          gate_error="no decision checkpoints were evaluated" if not rows else None,
                          failed_checkpoints=[row["cycle"] for row in failed_rows],
                          thresholds={"max_created_delta": args.max_created_delta,
                                      "max_passed_delta": args.max_passed_delta,
                                      "max_queue_delta": args.max_queue_delta,
                                      "max_reward_relative_error": args.max_reward_relative_error,
                                      "max_integral_relative_error": args.max_integral_relative_error,
                                      "max_spatial_mae": args.max_spatial_mae,
                                      "max_aux_mae": args.max_aux_mae,
                                      "max_phase_mismatches": args.max_phase_mismatches,
                                      "max_action_mismatches": args.max_action_mismatches},
                          gama_step_wall_seconds=round(gama_wall, 3),
                          python_step_wall_seconds=round(python_wall, 3),
                          step_speedup=round(gama_wall / python_wall, 2)
                          if python_wall else None,
                          final=rows[-1] if rows else None)
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failed_rows or not rows:
        raise RuntimeError(f"Parity gate failed at {len(failed_rows)} checkpoints; see {output} and {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--port", type=int, default=6992)
    parser.add_argument("--reward-mode", choices=("r1", "r2", "r4"), default="r1")
    parser.add_argument("--debug-action", action="store_true")
    parser.add_argument("--weights", type=Path,
                        help="Evaluate a trained hybrid or Rainbow policy independently in both backends")
    parser.add_argument("--policy", choices=("sequence", "queue"), default="sequence",
                        help="Action sequence or independent queue-responsive control")
    parser.add_argument("--output", type=Path,
                        default=Path("reports/lvl_training_parity.csv"))
    parser.add_argument("--max-created-delta", type=int, default=3)
    parser.add_argument("--max-passed-delta", type=int, default=5)
    parser.add_argument("--max-queue-delta", type=float, default=10.0)
    parser.add_argument("--max-reward-relative-error", type=float, default=0.10)
    parser.add_argument("--max-integral-relative-error", type=float, default=0.10)
    parser.add_argument("--max-spatial-mae", type=float, default=0.05)
    parser.add_argument("--max-aux-mae", type=float, default=0.02)
    parser.add_argument("--max-phase-mismatches", type=int, default=0)
    parser.add_argument("--max-action-mismatches", type=int, default=0)
    options = parser.parse_args()
    if options.horizon < 100 or options.horizon % 50:
        parser.error("Horizon must be a multiple of 50 and at least 100")
    asyncio.run(main(options))
