import unittest
from pathlib import Path

from python_simulation.junction_conflicts import (ConflictEngine, ConflictMap, Movement,
                                 MovementRequest, _cumulative)


def movement(mid, points, turn="through", arm="A", to_arm="B"):
    return Movement(mid, arm, to_arm, turn, tuple(points), tuple(_cumulative(points)))


def crossing_map():
    # Metric, deliberately small paths make each test's conflict geometry clear.
    return ConflictMap({
        "straight": movement("straight", [(-10, 0), (10, 0)], "through"),
        "left": movement("left", [(0, -10), (0, 10)], "left", "C", "D"),
        "opposing": movement("opposing", [(0, 10), (0, -10)], "through", "D", "C"),
        "parallel": movement("parallel", [(-10, 5), (10, 5)], "through", "E", "F"),
        "following": movement("following", [(-10, 0), (10, 0)], "through", "A", "B"),
    }, clearance=0.5)


def near_miss_map():
    return ConflictMap({
        "lower": movement("lower", [(-10, 0), (10, 0)]),
        "near": movement("near", [(-10, 1.8), (10, 1.8)], arm="C", to_arm="D"),
        "clear": movement("clear", [(-10, 4), (10, 4)], arm="E", to_arm="F"),
    }, clearance=0.5, near_miss_distance=2.2)


class ConflictEngineTests(unittest.TestCase):
    def test_real_geojson_opposing_through_and_left_conflicts(self):
        source = Path(__file__).resolve().parents[1] / "simulation" / "python_simulation" / "includes" / "roads.geojson"
        conflict_map = ConflictMap.from_geojson(source)
        # Opposing arms, rather than two movements that both originate at NE.
        for through_id, left_id in (("NE_SW", "SW_NW"), ("SW_NE", "NE_SE")):
            self.assertTrue(conflict_map.intervals(through_id, left_id))
            engine = ConflictEngine(conflict_map)
            decisions = engine.decide_batch([
                MovementRequest("left", left_id, arrival_time=1.0),
                MovementRequest("through", through_id, arrival_time=1.0),
            ])
            self.assertTrue(decisions["through"].allowed, (through_id, left_id))
            self.assertFalse(decisions["left"].allowed, (through_id, left_id))

    def test_straight_has_priority_over_left_at_crossing(self):
        engine = ConflictEngine(crossing_map())
        decisions = engine.decide_batch([
            MovementRequest("left-car", "left", speed=5),
            MovementRequest("straight-car", "straight", speed=5),
        ])
        self.assertTrue(decisions["straight-car"].allowed)
        self.assertFalse(decisions["left-car"].allowed)
        self.assertEqual(decisions["left-car"].blockers, ("straight-car",))

    def test_near_left_is_not_held_by_distant_through_vehicle(self):
        engine = ConflictEngine(crossing_map())
        decisions = engine.decide_batch([
            MovementRequest("distant-through", "straight", arrival_time=1.5),
            MovementRequest("near-left", "left", arrival_time=1.0),
        ])
        self.assertTrue(decisions["near-left"].allowed)
        self.assertFalse(decisions["distant-through"].allowed)

    def test_equal_arrival_through_priority_remains_deterministic(self):
        engine = ConflictEngine(crossing_map())
        decisions = engine.decide_batch([
            MovementRequest("left", "left", arrival_time=2.0),
            MovementRequest("through", "straight", arrival_time=2.0),
        ])
        self.assertTrue(decisions["through"].allowed)
        self.assertFalse(decisions["left"].allowed)

    def test_geometrically_nonconflicting_movements_run_together(self):
        engine = ConflictEngine(crossing_map())
        decisions = engine.decide_batch([
            MovementRequest("one", "straight", speed=5),
            MovementRequest("two", "parallel", speed=5),
        ])
        self.assertTrue(decisions["one"].allowed)
        self.assertTrue(decisions["two"].allowed)

    def test_earlier_opposing_arrival_wins_and_stable_id_breaks_ties(self):
        engine = ConflictEngine(crossing_map())
        decisions = engine.decide_batch([
            MovementRequest("z-car", "straight", arrival_time=2),
            MovementRequest("a-car", "opposing", arrival_time=1),
        ])
        self.assertTrue(decisions["a-car"].allowed)
        self.assertFalse(decisions["z-car"].allowed)
        engine = ConflictEngine(crossing_map())
        decisions = engine.decide_batch([
            MovementRequest("z-car", "straight", arrival_time=1),
            MovementRequest("a-car", "opposing", arrival_time=1),
        ])
        self.assertTrue(decisions["a-car"].allowed)
        self.assertFalse(decisions["z-car"].allowed)

    def test_explicit_arrival_time_is_absolute_even_when_now_is_nonzero(self):
        engine = ConflictEngine(crossing_map())
        req = MovementRequest("future", "straight", speed=10, arrival_time=11.0)
        windows, _ = engine._windows(req, now=10.0)
        start = windows["left"][0][0]
        self.assertGreater(start, 11.0)
        self.assertLess(start, 13.0)

    def test_each_batch_rebuilds_approach_windows_from_latest_eta(self):
        engine = ConflictEngine(crossing_map())
        engine.decide_batch([MovementRequest(
            "approach", "straight", speed=5, arrival_time=10.0)], now=0.0)
        before = engine.reservations["approach"].conflict_windows["left"][0][0]
        result = engine.decide_batch([MovementRequest(
            "approach", "straight", speed=2, arrival_time=20.0)], now=5.0)
        after = engine.reservations["approach"].conflict_windows["left"][0][0]
        self.assertTrue(result["approach"].allowed)
        self.assertGreater(after, before + 9.0)

    def test_vehicle_already_committed_gets_first_reservation(self):
        engine = ConflictEngine(crossing_map())
        decisions = engine.decide_batch([
            MovementRequest("approaching-through", "straight", speed=5),
            MovementRequest("committed-left", "left", speed=5, committed=True,
                            connector_s=3.0),
        ])
        self.assertTrue(decisions["committed-left"].allowed)
        self.assertFalse(decisions["approaching-through"].allowed)

    def test_waiting_left_turn_ages_ahead_of_through_but_below_committed(self):
        engine = ConflictEngine(crossing_map(), left_wait_priority_after=10.0)
        aged_left = MovementRequest("patient-left", "left", waiting_since=0.0)
        fresh_through = MovementRequest("new-through", "straight")
        committed_through = MovementRequest("inside", "straight", committed=True,
                                             connector_s=3.0)
        self.assertLess(engine._rank(committed_through, 20.0), engine._rank(aged_left, 20.0))
        decisions = engine.decide_batch([fresh_through, aged_left], now=20.0)
        self.assertTrue(decisions["patient-left"].allowed)
        self.assertFalse(decisions["new-through"].allowed)

    def test_following_platoon_can_enter_with_headway(self):
        engine = ConflictEngine(crossing_map(), following_headway=0.8)
        first = engine.decide_batch([MovementRequest(
            "leader", "straight", speed=10, arrival_time=0.0)])["leader"]
        second = engine.decide_batch([
            MovementRequest("leader", "straight", speed=10, arrival_time=0.0),
            MovementRequest("follower", "straight", speed=10, arrival_time=1.2),
        ])["follower"]
        too_close = engine.decide_batch([
            MovementRequest("leader", "straight", speed=10, arrival_time=0.0),
            MovementRequest("follower", "straight", speed=10, arrival_time=1.2),
            MovementRequest("tailgater", "straight", speed=10, arrival_time=1.5),
        ])["tailgater"]
        self.assertTrue(first.allowed)
        self.assertTrue(second.allowed)
        self.assertFalse(too_close.allowed)
        self.assertEqual(too_close.reason, "conflict_reserved")

    def test_live_connector_reservation_refreshes_after_progress_or_slowdown(self):
        engine = ConflictEngine(crossing_map())
        engine.decide_batch([MovementRequest(
            "live", "straight", speed=10, connector_s=0.0, committed=True)], now=0.0)
        before = engine.reservations["live"].clears_at
        self.assertTrue(engine.refresh(MovementRequest(
            "live", "straight", speed=2, connector_s=5.0, committed=True), now=1.0))
        after = engine.reservations["live"].clears_at
        self.assertGreater(after, before)

    def test_downstream_occupied_exit_blocks_entry(self):
        engine = ConflictEngine(crossing_map())
        request = MovementRequest("blocked", "straight", downstream_clear=False)
        decision = engine.decide_batch([request])["blocked"]
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "downstream_occupied")

    def test_prior_approach_grant_does_not_override_new_downstream_block(self):
        engine = ConflictEngine(crossing_map())
        engine.decide_batch([MovementRequest(
            "approach", "straight", arrival_time=1.0, downstream_clear=True)])
        decision = engine.decide_batch([MovementRequest(
            "approach", "straight", arrival_time=1.0, downstream_clear=False)], now=0.5)["approach"]
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "downstream_occupied")
        self.assertNotIn("approach", engine.reservations)  # blocked grants are released

    def test_near_eta_request_beats_prior_tick_approach_grant(self):
        engine = ConflictEngine(crossing_map())
        engine.decide_batch([
            MovementRequest("through", "straight", arrival_time=0.0),
            MovementRequest("left", "left", arrival_time=10.0),
        ])
        decision = engine.decide_batch([
            MovementRequest("through", "straight", arrival_time=10.5),
            MovementRequest("left", "left", arrival_time=10.0),
        ], now=1.0)["through"]
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "conflict_reserved")
        self.assertEqual(decision.blockers, ("left",))
        self.assertNotIn("through", engine.reservations)

    def test_unsubmitted_approach_reservation_is_released(self):
        engine = ConflictEngine(crossing_map())
        engine.decide_batch([MovementRequest("stale", "straight")])
        engine.decide_batch([])
        self.assertNotIn("stale", engine.reservations)

    def test_vehicle_on_connector_can_clear_when_exit_becomes_blocked(self):
        engine = ConflictEngine(crossing_map())
        engine.decide_batch([MovementRequest(
            "inside", "straight", connector_s=2.0, committed=True)])
        decision = engine.decide_batch([MovementRequest(
            "inside", "straight", connector_s=3.0, committed=True,
            downstream_clear=False)], now=0.5)["inside"]
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, "already_reserved")

    def test_newly_reported_connector_vehicle_gets_clearance_over_approach_reservation(self):
        engine = ConflictEngine(crossing_map())
        engine.decide_batch([MovementRequest(
            "approach", "left", arrival_time=0.0)])
        decision = engine.decide_batch([MovementRequest(
            "inside", "straight", connector_s=2.0, committed=True,
            downstream_clear=False)], now=0.1)["inside"]
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, "committed_clearance")

    def test_near_parallel_path_is_conservatively_conflicting_and_marked(self):
        conflict_map = near_miss_map()
        key = ("lower", "near")
        self.assertIn(key, conflict_map.near_miss_pairs)
        self.assertTrue(conflict_map.intervals("lower", "near"))
        self.assertFalse(conflict_map.intervals("lower", "clear"))


class GreenStallRegressionTests(unittest.TestCase):
    """Cause-level regressions for vehicles stopping at a green light."""

    def test_committed_approach_is_never_denied_and_stays_visible(self):
        # A vehicle holding an entry token will cross whatever the arbiter
        # says, so a denial would hide its window and let a conflicting
        # vehicle be granted onto a crossing path.
        engine = ConflictEngine(crossing_map())
        decisions = engine.decide_batch([
            MovementRequest("straight-on-connector", "straight",
                            connector_s=9.0, committed=True, speed=5.0),
            MovementRequest("left-with-token", "left", arrival_time=0.3,
                            committed=True, speed=5.0),
            # "parallel" crosses only the left movement, so only the
            # token holder can be what blocks it.
            MovementRequest("crossing-new", "parallel", arrival_time=0.3,
                            speed=5.0),
        ])
        self.assertTrue(decisions["left-with-token"].allowed)
        self.assertEqual(decisions["left-with-token"].reason, "committed_clearance")
        self.assertFalse(decisions["crossing-new"].allowed)
        self.assertEqual(decisions["crossing-new"].blockers, ("left-with-token",))

    def test_vehicle_that_already_passed_the_conflict_does_not_block(self):
        # The occupant is still on its connector but its rear has left the
        # crossing zone and the newcomer only arrives well afterwards. The old
        # "hold until the whole connector clears" rule blocked this.
        engine = ConflictEngine(crossing_map())
        decisions = engine.decide_batch([
            MovementRequest("past-crossing", "straight", connector_s=19.0,
                            committed=True, speed=5.0),
            MovementRequest("late-left", "left", arrival_time=3.0, speed=5.0),
        ])
        self.assertTrue(decisions["late-left"].allowed, decisions["late-left"])

    def test_overlapping_windows_still_block(self):
        engine = ConflictEngine(crossing_map())
        decisions = engine.decide_batch([
            MovementRequest("straight-first", "straight", arrival_time=1.0,
                            speed=5.0),
            MovementRequest("left-same-time", "left", arrival_time=1.0,
                            speed=5.0),
        ])
        self.assertTrue(decisions["straight-first"].allowed)
        self.assertFalse(decisions["left-same-time"].allowed)

    def test_kinematic_travel_time_accounts_for_acceleration(self):
        travel = ConflictEngine._travel_time
        # From rest at 2.5 m/s^2: 20 m takes sqrt(2*20/2.5) = 4.0 s, quicker
        # than the 4.44 s a constant 4.5 m/s assumption predicts.
        self.assertAlmostEqual(travel(20.0, 0.0, 2.5, 11.0), 4.0, places=6)
        self.assertLess(travel(20.0, 0.0, 2.5, 11.0), 20.0 / 4.5)
        # Speed is capped at the desired speed.
        t = travel(100.0, 0.0, 2.5, 10.0)
        self.assertAlmostEqual(t, 4.0 + (100.0 - 20.0) / 10.0, places=6)
        # Already at or above the cap: constant speed.
        self.assertAlmostEqual(travel(30.0, 12.0, 2.5, 10.0), 30.0 / 12.0, places=6)
        self.assertEqual(travel(0.0, 3.0, 2.5, 10.0), 0.0)

    def test_kinematic_window_starts_no_later_than_constant_speed_window(self):
        engine = ConflictEngine(crossing_map())
        base = dict(movement_id="straight", arrival_time=1.0, speed=4.5,
                    vehicle_length=4.5)
        legacy, _ = engine._windows(MovementRequest("legacy", **base), 0.0)
        kinematic, _ = engine._windows(MovementRequest(
            "kin", accel=2.5, desired_speed=11.0, speed0=0.0,
            entry_distance=1.0, **base), 0.0)
        # The newcomer must not be predicted later than it can really arrive.
        self.assertLessEqual(kinematic["left"][0][0], legacy["left"][0][0] + 1e-9)
        # The clearing time is still bounded (latest arrival, half acceleration).
        self.assertGreater(kinematic["left"][0][1], kinematic["left"][0][0])


if __name__ == "__main__":
    unittest.main()
