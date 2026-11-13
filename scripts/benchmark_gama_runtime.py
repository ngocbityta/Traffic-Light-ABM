#!/usr/bin/env python3
"""Measure GAMA simulation throughput with surrogate-safety sensing on/off."""

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.logic.gama_env import GamaTrafficEnv
from train.logic.server_manager import GamaServerManager


async def measure(port, horizon, seed, repetitions):
    env = GamaTrafficEnv(port=port, verbose=False)
    await env.setup()
    rows = []
    try:
        # Alternate order to reduce JVM warmup bias.
        modes = [mode for index in range(repetitions)
                 for mode in ((True, False) if index % 2 == 0 else (False, True))]
        for enabled in modes:
            await env.reset(seed=seed, parameters=[
                {"name": "enable_surrogate_safety", "type": "bool",
                 "value": str(enabled).lower()}])
            start = time.perf_counter()
            await env.step_cycles(horizon)
            elapsed = time.perf_counter() - start
            state = await env.read_state()
            positions = str(await env._exec_expression(
                "string(active_vehicles collect [each.name,each.location,each.current_road])"))
            wait_integrals = str(await env._exec_expression(
                '"" + queue_time_integral + "|" + external_wait_integral'))
            rows.append(dict(safety_metrics=enabled, wall_seconds=elapsed,
                             cycles_per_second=horizon / elapsed,
                             cycle=state["cycle"],
                             passed=state["total_vehicles_passed"],
                             created=state["total_vehicles_created"],
                             wait_integrals=wait_integrals,
                             positions_sha256=hashlib.sha256(positions.encode()).hexdigest()))
        if any(row["cycle"] != horizon for row in rows):
            raise RuntimeError("GAMA stopped before the requested benchmark horizon")
        identical = len({(row["passed"], row["created"], row["wait_integrals"],
                          row["positions_sha256"])
                         for row in rows}) == 1
        median_on = statistics.median(r["cycles_per_second"] for r in rows if r["safety_metrics"])
        median_off = statistics.median(r["cycles_per_second"] for r in rows if not r["safety_metrics"])
        return dict(horizon=horizon, seed=seed, rows=rows,
                    median_on_cycles_per_second=median_on,
                    median_off_cycles_per_second=median_off,
                    speedup=median_off / median_on,
                    traffic_trajectory_identical=identical)
    finally:
        await env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=6917)
    parser.add_argument("--horizon", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=5001)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.horizon < 100 or args.repetitions < 1:
        parser.error("horizon >= 100 and repetitions >= 1 are required")
    with tempfile.TemporaryDirectory(prefix="gama_speed_") as work:
        manager = GamaServerManager(port=args.port, workdir=work,
                                    log_file=str(Path(work) / "server.log"))
        manager.start(reuse_running=False)
        try:
            result = asyncio.run(measure(args.port, args.horizon, args.seed,
                                         args.repetitions))
        finally:
            manager.stop()
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
