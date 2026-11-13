"""Tests for the opt-in GAMA 2025.6.4 simulator profile."""
import unittest
from types import MethodType

from python_simulation.python_sim import LeVanLuongSimulator
from python_simulation.gama_2025_idm_mobil import MobilResult


class GamaParityModeTests(unittest.TestCase):
    def test_safe_remains_the_default_and_invalid_modes_are_rejected(self):
        simulator = LeVanLuongSimulator(flow_per_approach=0)
        self.assertEqual(simulator.mode, "safe")
        with self.assertRaisesRegex(ValueError, "mode must be"):
            LeVanLuongSimulator(flow_per_approach=0, mode="parity")

    def test_gama_spawn_does_not_apply_python_safe_spawn_spacing(self):
        safe = LeVanLuongSimulator(flow_per_approach=0, mode="safe")
        parity = LeVanLuongSimulator(flow_per_approach=0, mode="gama_2025_6_4")
        for simulator in (safe, parity):
            simulator._create_vehicle("NE", "through", False, 0, 40.0)
            simulator._create_vehicle("NE", "through", False, 0, 40.0)
        self.assertLess(safe.vehicles[1].position, 0.0)
        self.assertEqual(parity.vehicles[1].position, 0.0)

    def test_zero_new_speed_keeps_gama_vehicle_unregistered_at_source(self):
        simulator = LeVanLuongSimulator(flow_per_approach=0,
                                        mode="gama_2025_6_4")
        lane = simulator.roads["NE_in"]["num_lanes"] - 2
        simulator._create_vehicle("NE", "through", True, lane, 40.0)
        simulator._create_vehicle("NE", "through", True, lane, 40.0)
        simulator._move()
        leader, blocked = simulator.vehicles
        self.assertEqual(leader.road, "NE_in")
        self.assertTrue(leader.gama_entered)
        self.assertEqual(blocked.road, "")
        self.assertFalse(blocked.gama_entered)
        self.assertEqual((blocked.position, blocked.speed), (0.0, 0.0))

    def test_zero_speed_gate_does_not_commit_provisional_mobil_lane(self):
        simulator = LeVanLuongSimulator(flow_per_approach=0,
                                        mode="gama_2025_6_4")
        simulator._create_vehicle("SW", "right", True, 2, 40.0)
        vehicle = simulator.vehicles[0]
        lane = vehicle.lane

        def provisional_change(self, agent, lane_maps):
            return MobilResult(agent.lane + 1, -1.0, 0.0)

        simulator._choose_lane = MethodType(provisional_change, simulator)
        simulator._move()
        self.assertEqual(vehicle.lane, lane)
        self.assertEqual(vehicle.road, "")
        self.assertEqual(vehicle.time_since_lane_change, 0.0)

    def test_red_approach_does_not_apply_intersection_gate_with_vehicle_leader(self):
        simulator = LeVanLuongSimulator(flow_per_approach=0,
                                        mode="gama_2025_6_4")
        simulator.phase = "LVL"  # NW is red in this phase.
        simulator._create_vehicle("NW", "through", True, 9, 52.933159248652885)
        vehicle = simulator.vehicles[0]
        vehicle.road = "NW_in"
        vehicle.gama_entered = True
        vehicle.speed = 10.033610835048695
        vehicle.time_since_lane_change = 4.400000000000001
        simulator._neighbor_road_state = {id(vehicle): ("NW_in", 23.40087909996772)}
        leader = type("Leader", (), {"speed": 9.934052910674382})()

        def neighbors(self, agent, lane, lane_maps):
            if lane == 8:
                return None, 1e6, None, 1e6
            return leader, 29.80725943035282, None, 1e6

        simulator._neighbors = MethodType(neighbors, simulator)
        lane_maps = {"NW_in": [[] for _ in range(simulator.roads["NW_in"]["num_lanes"])]}
        decision = simulator._choose_lane_gama(vehicle, lane_maps)
        self.assertEqual(decision.lane, 8)

    def test_red_approach_virtual_node_does_not_trigger_intersection_skill_gate(self):
        simulator = LeVanLuongSimulator(flow_per_approach=0,
                                        mode="gama_2025_6_4")
        simulator.phase = "LVL"  # NW is red in this phase.
        simulator._create_vehicle("NW", "through", True, 9, 52.933159248652885)
        vehicle = simulator.vehicles[0]
        vehicle.road = "NW_in"
        vehicle.gama_entered = True
        vehicle.speed = 10.0
        vehicle.time_since_lane_change = 10.0
        simulator._neighbor_road_state = {id(vehicle): ("NW_in", 23.4)}

        def neighbors(self, agent, lane, lane_maps):
            # The red stop line is represented by the synthetic node leader in
            # this lane. GAMA 2025.6.4's exact skill-name check does not classify
            # that agent as implementing "intersection" in this model.
            if lane == 9:
                return None, 3.0, None, 1e6
            return None, 1e6, None, 1e6

        simulator._neighbors = MethodType(neighbors, simulator)
        lane_maps = {"NW_in": [[] for _ in range(simulator.roads["NW_in"]["num_lanes"])]}
        decision = simulator._choose_lane_gama(vehicle, lane_maps)
        self.assertEqual(decision.lane, 8)

    def test_gama_final_exit_uses_virtual_target_node_as_leader(self):
        simulator = LeVanLuongSimulator(flow_per_approach=0,
                                        mode="gama_2025_6_4")
        simulator._create_vehicle("NE", "through", True, 2, 49.0)
        vehicle = simulator.vehicles[0]
        vehicle.road = "SW_out"
        vehicle.position = 4.0
        simulator._neighbor_road_state = {id(vehicle): ("SW_out", 4.0)}
        lane_maps = {"SW_out": [[] for _ in range(simulator.roads["SW_out"]["num_lanes"])]}

        leader, front_gap, follower, back_gap = simulator._neighbors(
            vehicle, vehicle.lane, lane_maps)

        self.assertIsNone(leader)
        self.assertIsNone(follower)
        self.assertAlmostEqual(front_gap,
                               simulator.roads["SW_out"]["length"] - 4.0 + vehicle.gap)
        self.assertEqual(back_gap, float("inf"))


if __name__ == "__main__":
    unittest.main()
