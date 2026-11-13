#!/usr/bin/env python3
"""Live DoAn/3DQN integration check; requires a fresh GAMA server on --port."""
import argparse
import asyncio
import json
import math
import tempfile
import time
from pathlib import Path

if __package__:
    from .gama_env import GamaTrafficEnv, STATE_FIELDS, featurize
    from .server_manager import GamaServerManager
else:
    from gama_env import GamaTrafficEnv, STATE_FIELDS, featurize
    from server_manager import GamaServerManager


async def run(port, horizon):
    env = GamaTrafficEnv(port=port, verbose=False)
    await env.setup()
    try:
        snapshots = []
        for seed in (5001, 5001, 5002):
            state = await env.reset(seed=seed)
            assert state['cycle'] == 0
            await env.step_cycles(100)
            state = await env.read_state()
            assert state['dqn_waiting_action']
            snapshots.append(await env._exec_expression(
                'string(active_vehicles collect [each.vehicle_type,each.original_max_speed,each.maneuver,each.location])'))
        assert snapshots[0] == snapshots[1], 'Same seed must reproduce vehicles'
        assert snapshots[1] != snapshots[2], 'Changing seed must change vehicles'
        state = await env.reset(seed=5001)
        await env.step_cycles(100)
        state = await env.read_state()
        decisions = []
        while state['cycle'] < horizon:
            action = len(decisions) % 4
            if not state['action_mask'][action]:
                action = next(a for a in range(4) if state['action_mask'][a])
            before = state['cycle']
            try:
                state, reward, done, info = await env.step(action)
            except Exception:
                diagnostics = {
                    'last_decision_cycle': before,
                    'state': await env.read_state(),
                }
                for key, expression in {
                    'yellow_cycles': 'yellow_cycles',
                    'all_red_cycles': 'all_red_cycles',
                    'phase_timer': 'phase_timer',
                    'connector_vehicles': 'string(active_vehicles where '
                        '(each.current_road != nil and road(each.current_road).is_signal_connector) '
                        'collect [each.name, road(each.current_road).road_key, each.location, '
                        'each.speed, each.max_speed, each.lowest_lane, each.distance_to_current_target, '
                        'road(each.next_road).road_key])',
                    'south_exit_vehicles': 'string(active_vehicles where '
                        '(each.current_road != nil and distance_to(each.location, {998,1040}) < 70) '
                        'collect [each.name, road(each.current_road).road_key, each.location, '
                        'each.speed, each.max_speed, each.lowest_lane, each.distance_to_current_target])',
                    'south_uturn_owners': 'string(network_nodes["S_U_OUT"].merge_owners collect each.name)',
                    'south_right_owners': 'string(network_nodes["S_MERGE"].merge_owners collect each.name)',
                }.items():
                    try:
                        diagnostics[key] = await env._exec_expression(expression)
                    except Exception as error:
                        diagnostics[key] = str(error)
                print('DQN_FAILURE_DIAGNOSTICS,' + json.dumps(diagnostics, default=str), flush=True)
                raise
            assert all(field in state for field in STATE_FIELDS)
            assert state['dqn_waiting_action'] and state['cycle'] > before
            dtse = featurize(state)
            assert dtse.shape == (3, 8, 20) and all(math.isfinite(float(x)) for x in dtse.flat)
            assert math.isfinite(state['delay_sq']) and state['delay_sq'] >= 0
            assert math.isfinite(reward) and not done
            decisions.append(dict(action=action, cycle=state['cycle'], phase=state['global_phase'], reward=reward))
        ground = int(float(await env._exec_expression('total_vehicles_passed')))
        bypass = int(float(await env._exec_expression('bypass_vehicles_passed')))
        # A short protocol smoke run can reach many decisions before any
        # vehicle traverses the full network; trip completion needs a long run.
        if horizon >= 4000:
            assert ground > 0 and bypass > 0, 'Both ground and bypass traffic must complete trips'
        assert not [m for m in env._server_messages if isinstance(m, dict) and m.get('type') in
                    ('SimulationError', 'RuntimeError', 'GamaServerError', 'SimulationErrorDialog')]
        return dict(passed=True, reproducible_seed=True, changed_seed_changes_traffic=True,
                    ground_passed=ground, bypass_passed=bypass, decisions=decisions)
    finally:
        await env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=6871)
    p.add_argument('--horizon', type=int, default=4000)
    p.add_argument('--output', type=Path)
    p.add_argument('--start-server', action='store_true')
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix='doan_live_server_') as work:
        manager = None
        if args.start_server:
            log = args.output.with_suffix('.server.log') if args.output else Path(work) / 'server.log'
            manager = GamaServerManager(port=args.port, workdir=work, log_file=str(log))
            manager.start(reuse_running=False)
        try:
            result = asyncio.run(run(args.port, args.horizon))
        finally:
            if manager:
                manager.stop()
                for _ in range(50):
                    if not manager.is_port_open(timeout=0.2):
                        break
                    time.sleep(0.2)
                assert not manager.is_port_open(), 'Owned GAMA server did not release its port'
    output = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(output)
    print(output)
