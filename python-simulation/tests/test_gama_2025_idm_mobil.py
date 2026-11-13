"""Deterministic checks for the installed GAMA 2025.6.4 IDM/MOBIL port."""
import math
import unittest

from python_simulation.gama_2025_idm_mobil import (
    Candidate, IDMParams, MobilParams, choose_lane, idm_acceleration,
)


class IDMTests(unittest.TestCase):
    def test_idm_matches_tag_equation_and_road_speed_cap(self):
        params = IDMParams(time_headway=1.2, max_acceleration=2.5,
                           max_deceleration=3.0, desired_speed=20,
                           min_safety_distance=1.0, delta=4.0)
        v, leader, gap = 10.0, 8.0, 20.0
        s_star = 1.0 + max(0.0, v * 1.2 + v * (v - leader) / (2 * math.sqrt(2.5 * 3)))
        expected = 2.5 * (1 - (10 / 12) ** 4 - (s_star / gap) ** 2)
        actual = idm_acceleration(speed=v, leading_speed=leader, gap=gap,
                                  params=params, road_max_speed=12)
        self.assertAlmostEqual(actual, expected)

    def test_no_leader_is_free_road_idm(self):
        p = IDMParams(max_acceleration=2, desired_speed=15)
        self.assertAlmostEqual(idm_acceleration(speed=10, leading_speed=0,
                                                gap=math.inf, params=p),
                               2 * (1 - (10 / 15) ** 4))


class MOBILTests(unittest.TestCase):
    def setUp(self):
        self.vehicle = IDMParams(time_headway=1.2, max_acceleration=2.5,
                                 max_deceleration=3.0, desired_speed=20,
                                 min_safety_distance=1.0, delta=4.0)

    def run_choice(self, *, candidates, mobil=MobilParams(), **kw):
        base = dict(current_lane=1, lane_count=3, lanes_occupied=1,
                    speed=10, vehicle_length=4.5, current_gap=10,
                    current_leader_speed=0, current_follower_id=None,
                    current_time_since_lane_change=10, timestep=.2,
                    vehicle_params=self.vehicle, candidates=candidates,
                    mobil=mobil)
        base.update(kw)
        return choose_lane(**base)

    def test_lane_range_is_ascending_and_exact_tie_keeps_lowest_index(self):
        # Equal front conditions and disabled bias produce equal incentive.
        candidates = {0: Candidate(front_gap=30), 2: Candidate(front_gap=30)}
        result = self.run_choice(candidates=candidates,
                                 mobil=MobilParams(acceleration_bias=0))
        self.assertEqual(result.lane, 0)

    def test_right_side_bias_prefers_lower_lane_on_equal_acceleration(self):
        candidates = {0: Candidate(front_gap=30), 2: Candidate(front_gap=30)}
        result = self.run_choice(candidates=candidates)
        self.assertEqual(result.lane, 0)

    def test_engine_has_no_python_extra_required_gap_filter(self):
        # At 10 m/s Python's extra front requirement would be 13 m for these
        # GAML parameters, but GAMA MOBIL accepts by IDM incentive/safety only.
        result = self.run_choice(candidates={0: Candidate(front_gap=2)}, current_gap=1)
        self.assertEqual(result.lane, 0)

    def test_follower_safety_threshold_rejects_at_or_below_minus_bsave(self):
        # A very short back gap makes the induced follower braking unsafe.
        candidate = Candidate(front_gap=20, follower_id="back", back_gap=.1,
                              follower_speed=20, follower_leading_distance=math.inf,
                              follower_params=self.vehicle)
        result = self.run_choice(candidates={0: candidate})
        self.assertEqual(result.lane, 1)

    def test_cooldown_and_intersection_gate_increment_clock(self):
        no_candidates = {}
        result = self.run_choice(candidates=no_candidates, current_time_since_lane_change=3.9)
        self.assertEqual(result.lane, 1)
        self.assertAlmostEqual(result.time_since_lane_change, 4.1)
        result = self.run_choice(candidates={0: Candidate(front_gap=30)},
                                 leader_is_intersection=True,
                                 current_time_since_lane_change=10)
        self.assertEqual(result.lane, 1)

    def test_current_lane_overlap_bypasses_cooldown_gate(self):
        result = self.run_choice(candidates={0: Candidate(front_gap=30)},
                                 current_gap=-.1,
                                 current_time_since_lane_change=0)
        # Engine assigns -Double.MAX_VALUE to stay acceleration and still scans.
        self.assertEqual(result.lane, 0)
        self.assertEqual(result.time_since_lane_change, 0.0)

    def test_engine_default_threshold_and_gaml_override_are_distinct(self):
        self.assertEqual(MobilParams().acceleration_threshold, .2)
        # LeVanLuong.gaml sets acc_gain_threshold=.5 and politeness=.1.
        model = MobilParams(acceleration_threshold=.5, politeness=.1)
        self.assertEqual((model.acceleration_threshold, model.politeness), (.5, .1))


if __name__ == "__main__":
    unittest.main()
