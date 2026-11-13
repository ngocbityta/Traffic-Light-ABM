"""Behavioral checks for the standalone four-phase traffic environment."""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demand import demand_parameters, generate_demand
from python_env import PythonTrafficEnv, STOP_DISTANCE, Vehicle


class PythonTrafficEnvTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.demand = Path(self.tmp.name) / "demand.csv"
        generate_demand(self.demand, 101, 1200)

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_phase_schedule_and_reward_integrals(self):
        env = PythonTrafficEnv(verbose=False, reward_mode="total_wait")
        state = await env.reset(seed=101, parameters=demand_parameters(self.demand))
        self.assertEqual(state["cycle"], 0)
        self.assertEqual(state["next_decision_in_cycles"], 100)
        self.assertEqual(state["total_vehicles_created"], 0)
        await env.step_cycles(100)
        state = await env.read_state()
        self.assertTrue(state["dqn_waiting_action"])
        self.assertEqual(state["dtse"].shape, (3, 8, 20))
        next_state, reward, truncated, info = await env.step(0, current_state=state)
        self.assertEqual((next_state["cycle"], next_state["global_phase"]), (200, 0))
        self.assertFalse(truncated)
        self.assertTrue(info["bootstrap_valid"])
        self.assertLessEqual(reward, 0)
        next_state, reward, _, _ = await env.step(1, current_state=next_state)
        self.assertEqual((next_state["cycle"], next_state["global_phase"]), (351, 2))
        self.assertLessEqual(reward, 0)
        self.assertGreater(next_state["queue_time_integral"], state["queue_time_integral"])

    async def test_seeded_replay_is_deterministic(self):
        async def rollout():
            env = PythonTrafficEnv(verbose=False, reward_mode="r1", state_mode="spatial")
            state = await env.reset(seed=101, parameters=demand_parameters(self.demand))
            await env.step_cycles(100)
            state = await env.read_state()
            self.assertEqual(state["spatial"].shape, (4, 6, 20, 16))
            self.assertTrue((state["spatial"][:, 5] == 1).all())
            out = []
            while state["cycle"] < 1200:
                action = state["decision_count"] % 4
                state, reward, _, _ = await env.step(action, horizon=1200, current_state=state)
                out.append((state["cycle"], state["total_vehicles_created"],
                            state["total_vehicles_passed"], reward))
            return out
        self.assertEqual(await rollout(), await rollout())

    async def test_autonomous_baseline_phase_durations_without_traffic(self):
        empty = Path(self.tmp.name) / "empty.csv"
        empty.write_text("")
        params = demand_parameters(empty)
        fixed = PythonTrafficEnv(controller_mode="fixed", verbose=False)
        await fixed.reset(seed=1, parameters=params)
        await fixed.step_cycles(300)
        self.assertEqual((await fixed.read_state())["global_phase"], 1)
        await fixed.step_cycles(50)
        self.assertEqual((await fixed.read_state())["global_phase"], 2)
        await fixed.step_cycles(120)
        self.assertEqual((await fixed.read_state())["global_phase"], 3)
        maxpressure = PythonTrafficEnv(controller_mode="maxpressure", verbose=False)
        await maxpressure.reset(seed=1, parameters=params)
        await maxpressure.step_cycles(600)
        self.assertEqual((await maxpressure.read_state())["global_phase"], 1)
        await maxpressure.step_cycles(50)
        self.assertEqual((await maxpressure.read_state())["global_phase"], 2)

    async def test_following_uses_nearest_overlapping_lane_leader(self):
        env = PythonTrafficEnv(verbose=False)
        vehicles = [Vehicle(0, 1, 12.0, "straight", 4, 5, 7, 4.6, 1.9, 0,
                            position=position, speed=speed)
                    for position, speed in ((100.0, 10.0), (50.0, 5.0), (0.0, 8.0))]
        env.vehicles = vehicles
        gap, leader_speed = env._leader_gaps()[id(vehicles[-1])]
        self.assertAlmostEqual(gap, 45.4)
        self.assertEqual(leader_speed, 5.0)

    async def test_shared_flyover_and_outgoing_edges_use_physical_progress(self):
        env = PythonTrafficEnv(verbose=False)
        through = Vehicle(4, 1, 10, "straight", 9, -1, -1, 4.6, 1.9, 0,
                          position=700, speed=5)
        ramp = Vehicle(6, 1, 10, "straight", 9, -1, -1, 4.6, 1.9, 0,
                       position=680, speed=6)
        env.vehicles = [through, ramp]
        gap, speed = env._leader_gaps()[id(ramp)]
        self.assertAlmostEqual(gap, 15.4)
        self.assertEqual(speed, 5)
        ground_feeder = Vehicle(0, 1, 10, "straight", 9, 5, 7, 4.6, 1.9, 0,
                                position=100, speed=5)
        on_ramp = Vehicle(6, 1, 10, "straight", 9, -1, -1, 4.6, 1.9, 0,
                          position=80, speed=6)
        env.vehicles = [ground_feeder, on_ramp]
        gap, _ = env._leader_gaps()[id(on_ramp)]
        self.assertAlmostEqual(gap, 15.4)
        straight = Vehicle(0, 1, 10, "straight", 8, 5, 7, 4.6, 1.9, 0, speed=5)
        left = Vehicle(2, 1, 10, "left", 8, 4, 7, 4.6, 1.9, 0, speed=6)
        straight.position = straight.connector_end + 20
        left.position = left.connector_end + 10
        env.vehicles = [straight, left]
        gap, _ = env._leader_gaps()[id(left)]
        self.assertAlmostEqual(gap, 5.4)

    async def test_turn_connectors_match_gaml_node_distances(self):
        def make(turn):
            return Vehicle(0, 0, 8.0, turn, 4, 5, 6, 2.0, 0.8, 0)

        self.assertEqual(make("right").connector_start, 910.0)
        self.assertEqual(make("uturn").connector_start, 920.0)
        self.assertAlmostEqual(make("left").connector_end, 1016.9420611565265)
        self.assertAlmostEqual(make("right").connector_end, 1033.3823215440631)

    async def test_outgoing_ground_vehicle_blocks_opposite_source(self):
        env = PythonTrafficEnv(verbose=False)
        empty = Path(self.tmp.name) / "empty_source.csv"
        empty.write_text("")
        await env.reset(seed=1, parameters=demand_parameters(empty))
        outgoing = Vehicle(2, 1, 10.0, "left", 12, 4, 5, 4.6, 1.9, 0,
                           position=0.0, speed=0.0)
        outgoing.position = outgoing.route_end - 5.0
        env.vehicles.append(outgoing)
        env._pending[0].append((0, 0, 1, 35, 0, 8))
        env._cycle = 1
        env._admit()
        self.assertEqual(len(env._pending[0]), 1)
        outgoing.position = outgoing.route_end - 20.0
        env._admit()
        self.assertEqual(len(env._pending[0]), 0)
        off_ramp = Vehicle(8, 1, 10.0, "straight", 9, -1, -1, 4.6, 1.9, 0,
                           position=1975.0)
        env.vehicles = [off_ramp]
        env._pending[6].append((0, 6, 1, 35, 0, 8))
        env._admit()
        self.assertEqual(len(env._pending[6]), 1)

    async def test_terminal_idm_reaches_target(self):
        env = PythonTrafficEnv(verbose=False)
        vehicle = Vehicle(0, 1, 10.0, "straight", 8, 5, 7, 4.6, 1.9, 0,
                          speed=10.0)
        vehicle.position = vehicle.route_end - 20.0
        env.vehicles = [vehicle]
        for _ in range(300):
            env._advance_vehicles()
            if not env.vehicles:
                break
        self.assertFalse(env.vehicles)
        self.assertEqual(env._count_passed, 1)

    async def test_spatial_lateral_channel_tracks_change_since_capture(self):
        env = PythonTrafficEnv(state_mode="spatial", reward_mode="r1", verbose=False)
        empty = Path(self.tmp.name) / "empty_spatial.csv"
        empty.write_text("")
        await env.reset(seed=1, parameters=demand_parameters(empty))
        vehicle = Vehicle(0, 1, 10.0, "straight", 8, 5, 7, 4.6, 1.9, 0,
                          position=STOP_DISTANCE - 40, speed=5.0,
                          lateral_peak=2.8)
        env.vehicles = [vehicle]
        first = await env.read_state(force_dtse=True)
        second = await env.read_state(force_dtse=True)
        self.assertLess(first["spatial"][1, 4].min(), 0.0)
        self.assertEqual(second["spatial"][1, 4].min(), 0.0)

    async def test_ramp_priority_uses_head_of_lane(self):
        env = PythonTrafficEnv(verbose=False)
        through = Vehicle(4, 1, 10, "straight", 9, -1, -1, 4.6, 1.9, 0,
                          position=570)
        ramp_head = Vehicle(6, 1, 10, "straight", 9, -1, -1, 4.6, 1.9, 0,
                            position=560, ramp_stop_ticks=50)
        ramp_follower = Vehicle(6, 1, 10, "straight", 9, -1, -1, 4.6, 1.9, 0,
                                position=540, ramp_stop_ticks=100)
        env.vehicles = [through, ramp_head, ramp_follower]
        env._reserve_flyover_portals()
        self.assertIs(env._flyover_portal_owner["NS"], ramp_head)


if __name__ == "__main__":
    unittest.main()
