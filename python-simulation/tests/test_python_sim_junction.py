import unittest

from python_simulation.junction_conflicts import ConflictMap
from python_simulation.python_sim import LeVanLuongSimulator


class JunctionIntegrationTests(unittest.TestCase):
    def make_sim(self, **kwargs):
        return LeVanLuongSimulator(seed=1, flow_per_approach=0, **kwargs)

    @staticmethod
    def create(sim, arm, turn, lane=0, speed=18):
        is_car = True
        sim._create_vehicle(arm, turn, is_car, lane, speed)
        return sim.vehicles[-1]

    def test_real_opposing_pairs_and_same_origin_split_are_conflicts(self):
        sim = self.make_sim()
        conflict_map = sim._conflict_map
        for a, b in (("NE_SW", "SW_NW"), ("SW_NE", "NE_SE")):
            with self.subTest(a=a, b=b):
                self.assertTrue(conflict_map.intervals(a, b))

    def test_right_turn_sublanes_continue_curb_lanes(self):
        """Right-turn connectors are built from the curb: sublane k of the
        approach, the connector and the exit line up, so a right-turner keeps
        its lateral position and heading at both seams."""
        import math
        sim = self.make_sim()
        for arm, to in (("NE", "NW"), ("SE", "NE"), ("SW", "SE"), ("NW", "SW")):
            conn = f"{arm}_{to}"
            vehicle = self.create(sim, arm, "right")
            n = sim.roads[conn]["num_lanes"]
            for old, new in ((f"{arm}_in", conn), (conn, f"{to}_out")):
                for lane in range(n):
                    with self.subTest(seam=f"{old}->{new}", lane=lane):
                        self.assertEqual(sim._mapped_lane(vehicle, lane, old, new),
                                         min(lane, sim.roads[new]["num_lanes"] - vehicle.occupied))
                        shift = lane * 1.05
                        along, new_shift = sim._continuous_seam_pose(old, new, shift)
                        self.assertAlmostEqual(new_shift, shift, delta=0.05)
                        self.assertAlmostEqual(along, 0.0, delta=0.05)
                        a = sim._geometry_roads[old].pose(sim._geometry_roads[old].length, shift)
                        b = sim._geometry_roads[new].pose(0.0, shift)
                        dh = (a.heading - b.heading + math.pi) % (2 * math.pi) - math.pi
                        self.assertLess(abs(math.degrees(dh)), 3.0)

    def test_committed_connector_blocks_conflicting_live_approach(self):
        sim = self.make_sim()
        sim.phase = "LVL"
        inside = self.create(sim, "SW", "through")
        inside.road = "SW_NE"
        inside.position = inside.approach_end + 8
        approaching = self.create(sim, "NE", "left")
        approaching.position = sim._stop_position(approaching) - 10
        sim._reserve_junction()
        self.assertNotIn(approaching.creation_id, sim._junction_grants)
        self.assertIn(str(inside.creation_id), sim._conflict_engine.reservations)

    def test_exit_room_checks_full_vehicle_body(self):
        sim = self.make_sim()
        entering = self.create(sim, "SW", "through")  # SW_NE exits to NE.
        blocker = self.create(sim, "SE", "right")     # SE_NE also exits to NE.
        blocker.road = "NE_out"
        blocker.position = blocker.connector_end + 2.0
        blocker.current_lateral_shift = entering.current_lateral_shift
        self.assertFalse(sim._destination_has_room(entering))

    def test_same_connector_following_uses_idm_and_stable_creation_ids(self):
        sim = self.make_sim()
        leader = self.create(sim, "NE", "through")
        follower = self.create(sim, "NE", "through")
        self.assertNotEqual(leader.creation_id, follower.creation_id)
        self.assertEqual(leader.road, follower.road)
        self.assertTrue(sim._conflict_map.intervals("NE_SW", "NE_SW"))

    def test_train_action_switch_keeps_committed_connector_reserved_on_all_red(self):
        sim = self.make_sim(train_mode=True, junction_model="reservation")
        inside = self.create(sim, "NE", "through")
        inside.road = "NE_SW"
        inside.position = inside.approach_end + 5.0
        sim.cycle = 50
        sim.dqn_waiting_action = True
        sim.set_action(2)
        before = inside.position
        sim.step()
        self.assertEqual(sim.phase, "none")
        self.assertIn(str(inside.creation_id), sim._conflict_engine.reservations)
        self.assertGreaterEqual(inside.position, before)

    def test_entry_token_expires_when_arm_leaves_green_before_crossing(self):
        # A token granted at the very end of green used to survive all-red and
        # the next phase, leaving a motionless "committed" phantom on the stop
        # line that blocked every conflicting movement.
        sim = self.make_sim()
        sim.phase = "LVL"
        waiting = self.create(sim, "NE", "through")
        waiting.position = sim._stop_position(waiting)
        waiting.junction_entry_token = True
        crossed = self.create(sim, "SW", "through", lane=6)
        crossed.position = sim._stop_position(crossed) + 1.0
        crossed.junction_entry_token = True
        sim.phase = "none"
        sim._reserve_junction()
        self.assertFalse(waiting.junction_entry_token)
        self.assertNotIn(str(waiting.creation_id), sim._conflict_engine.reservations)
        # A vehicle whose front already crossed keeps its commitment.
        self.assertTrue(crossed.junction_entry_token)

    def test_no_stop_line_entry_token_outside_green_over_full_cycles(self):
        # Seed 4 at 2400 veh/h/approach produced the stale token at tick 250.
        sim = LeVanLuongSimulator(seed=4, flow_per_approach=2400)
        for _ in range(340):
            sim.step()
            for v in sim.vehicles:
                if (v.road == f"{v.arm}_in" and v.junction_entry_token and
                        v.position <= sim._stop_position(v) + 1e-8):
                    self.assertTrue(sim._green_for(v),
                                    (sim.cycle, sim.phase, v.arm, v.creation_id))


if __name__ == "__main__":
    unittest.main()


class JunctionLivenessTests(unittest.TestCase):
    """Regression for the 13/11 gridlock: a through vehicle outside its
    corridor kept an entry token and a committed reservation it could never
    use, which blocked opposing lefts and, through starvation priority, the
    whole LVL phase."""

    def make_sim(self):
        sim = LeVanLuongSimulator(seed=1, flow_per_approach=0)
        sim.cycle = 10
        sim.phase = "LVL"
        return sim

    def wrong_lane_through(self, sim, arm="NE"):
        sim._create_vehicle(arm, "through", False, 0, 40)
        vehicle = sim.vehicles[-1]
        self.assertNotEqual(vehicle.lane, sim._route_target_lane(vehicle))
        vehicle.position = sim._stop_position(vehicle)
        vehicle.speed = 0.0
        return vehicle

    def test_wrong_lane_vehicle_loses_token_and_requests_nothing(self):
        sim = self.make_sim()
        stuck = self.wrong_lane_through(sim)
        stuck.junction_entry_token = True
        sim._reserve_junction()
        self.assertFalse(stuck.junction_entry_token)
        self.assertNotIn(stuck.creation_id, sim._junction_grants)
        self.assertNotIn(str(stuck.creation_id), sim._conflict_engine.reservations)

    def test_wrong_lane_vehicle_does_not_block_opposing_left(self):
        sim = self.make_sim()
        stuck = self.wrong_lane_through(sim)
        stuck.junction_entry_token = True
        sim._create_vehicle("SW", "left", False, sim.roads["SW_in"]["num_lanes"] - 1, 30)
        left = sim.vehicles[-1]
        left.position = sim._stop_position(left)
        left.speed = 0.0
        sim._reserve_junction()
        self.assertIn(left.creation_id, sim._junction_grants)

    def test_stalled_token_is_withdrawn_with_cooldown(self):
        from python_simulation.python_sim import DT, TOKEN_STALL_LIMIT
        sim = self.make_sim()
        sim._create_vehicle("NE", "through", False, 5, 40)
        vehicle = sim.vehicles[-1]
        self.assertEqual(vehicle.lane, sim._route_target_lane(vehicle))
        vehicle.position = sim._stop_position(vehicle)
        vehicle.speed = 0.0
        vehicle.junction_entry_token = True
        for _ in range(int(round(TOKEN_STALL_LIMIT / DT))):
            sim._reserve_junction()
        self.assertFalse(vehicle.junction_entry_token)
        self.assertGreater(vehicle.token_cooldown_until, sim.cycle * DT)
        self.assertNotIn(vehicle.creation_id, sim._junction_grants)

    def test_merge_hold_never_moves_vehicle_backwards(self):
        sim = LeVanLuongSimulator(seed=1, flow_per_approach=0)
        sim._create_vehicle("NE", "through", False, 0, 40)
        vehicle = sim.vehicles[-1]
        vehicle.position = sim._stop_position(vehicle) - 12
        last = vehicle.position
        for _ in range(60):
            sim.step()
            if vehicle not in sim.vehicles:
                break
            self.assertGreaterEqual(vehicle.position, last - 1e-9)
            last = vehicle.position

    def test_mid_change_body_moving_away_is_not_blocked(self):
        sim = LeVanLuongSimulator(seed=1, flow_per_approach=0)
        self.assertTrue(sim._moving_away(9.45, 8.40, 9.78, 10.50))
        self.assertTrue(sim._moving_away(10.07, 10.50, 9.45, 9.45))
        self.assertFalse(sim._moving_away(9.45, 10.50, 9.78, 10.50))

    def test_through_vehicle_merges_beside_a_jammed_corridor(self):
        sim = LeVanLuongSimulator(seed=1, flow_per_approach=0)
        lanes = sim.roads["NE_in"]["num_lanes"]
        first = __import__("python_simulation.python_sim", fromlist=["x"]).turn_lane_bounds(
            lanes, 1, "through")[0]
        sim._create_vehicle("NE", "through", False, first - 1, 30)
        merger = sim.vehicles[-1]
        stop = sim._stop_position(merger)
        merger.position = stop - 40
        for k in range(6):  # a jam-spaced column in the corridor sublane
            sim._create_vehicle("NE", "through", False, first, 30)
            sim.vehicles[-1].position = stop - 37 - 2.2 * k
        for _ in range(600):
            sim.step()
            if merger not in sim.vehicles or merger.lane == sim._route_target_lane(merger):
                break
        self.assertTrue(merger not in sim.vehicles or
                        merger.lane == sim._route_target_lane(merger))
