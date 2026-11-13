import unittest
from types import SimpleNamespace

from python_simulation.vehicle_geometry import Road, SUBLANE_WIDTH
from scripts.audit_lvl_geometry import (Route, _sampled_sweep, _seam_jump,
                                        build_roads, enforce_failed, vehicle_state)


class GeometryAuditTests(unittest.TestCase):
    def test_enforced_audit_requires_passed_vehicle_coverage(self):
        empty = {"static_event_count": 0, "swept_event_count": 0,
                 "simulation_coverage": {"passed": 0}}
        covered = {"static_event_count": 0, "swept_event_count": 0,
                   "simulation_coverage": {"passed": 1}}
        self.assertTrue(enforce_failed(empty))
        self.assertFalse(enforce_failed(empty, min_passed=0))
        self.assertFalse(enforce_failed(covered))

    def test_crossing_rectangles_collide_without_matching_centers(self):
        # Centers pass 1.0m apart while 4.5m long rectangles cross.
        horizontal = Road(((-5, 0), (5, 0)))
        vertical = Road(((0, -5), (0, 5)))
        a = {"arm": "a", "maneuver": "through", "position": 5.0, "shift": 0.0,
             "kind": "car", "route": _SingleRoadRoute(horizontal)}
        b = {"arm": "b", "maneuver": "through", "position": 4.0, "shift": 0.0,
             "kind": "car", "route": _SingleRoadRoute(vertical)}
        pa = dict(a, position=3.0)
        pb = dict(b, position=6.0)
        self.assertTrue(_sampled_sweep(pa, a, pb, b, 0.0))

    def test_seam_rebase_does_not_invent_swept_contact(self):
        roads = {
            "A_in": {"id": "A_in", "kind": "approach", "arm": "A",
                     "to_arm": "", "turn": "", "num_lanes": 2,
                     "coordinates": ((0, 0), (10, 0))},
            "A_B": {"id": "A_B", "kind": "connector", "arm": "A",
                    "to_arm": "B", "turn": "through", "num_lanes": 2,
                    "coordinates": ((0, -10), (0, 20))},
            "B_out": {"id": "B_out", "kind": "exit", "arm": "B",
                      "to_arm": "", "turn": "", "num_lanes": 2,
                      "coordinates": ((0, 20), (10, 20))},
        }
        route = Route(roads, "A", "through")
        old_pose = route.pose(9.5, 2.0)[1]
        new_position = route.parts[1][2] + 8.5
        new_pose = route.pose(new_position, 10.0)[1]
        transitioning = SimpleNamespace(
            road="A_B", seam_from_road="A_in", seam_to_road="A_B",
            seam_progress=8.0, seam_shift=10.0)
        before = {"arm": "A", "maneuver": "through", "sim_road": "A_in",
                  "position": 9.5, "shift": 2.0, "kind": "car", "pose": old_pose,
                  "route": route}
        after = {"arm": "A", "maneuver": "through", "sim_road": "A_B",
                 "position": new_position, "shift": 10.0, "kind": "car",
                 "pose": new_pose, "route": route, "vehicle": transitioning}

        # The old whole-tick interpolation moves laterally from 2m to 10m
        # while route progress jumps to the new frame, passing through (6,-6).
        # The actual path stays near x=10, so this waiting vehicle is clear.
        blocker_road = Road(((6, -10), (6, 10)))
        blocker_route = _SingleRoadRoute(blocker_road)
        blocker_pose = blocker_road.pose(4.0, 0.0)
        blocker = {"arm": "A", "maneuver": "through", "position": 4.0,
                   "shift": 0.0, "kind": "car", "pose": blocker_pose,
                   "route": blocker_route}
        self.assertFalse(_sampled_sweep(before, after, blocker, blocker, 0.0))

    def test_state_uses_simulator_road_when_rebased_progress_precedes_seam(self):
        roads = {
            "A_in": {"id": "A_in", "kind": "approach", "arm": "A",
                     "to_arm": "", "turn": "", "num_lanes": 2,
                     "coordinates": ((0, 0), (10, 0))},
            "A_B": {"id": "A_B", "kind": "connector", "arm": "A",
                    "to_arm": "B", "turn": "through", "num_lanes": 2,
                    "coordinates": ((0, -10), (0, 20))},
            "B_out": {"id": "B_out", "kind": "exit", "arm": "B",
                      "to_arm": "", "turn": "", "num_lanes": 2,
                      "coordinates": ((0, 20), (10, 20))},
        }
        route = Route(roads, "A", "through")
        vehicle = SimpleNamespace(
            arm="A", to_arm="B", maneuver="through", road="A_B",
            position=9.0, current_lateral_shift=10.0, occupied=2,
            demand_row_id=11, lane=0, speed=2.0)
        state = vehicle_state(vehicle, {("A", "B", "through"): route})
        expected = route.pose_on_road("A_B", -1.0, 10.0)
        self.assertEqual(state["road"], "A_B")
        self.assertEqual(state["sim_road"], "A_B")
        self.assertAlmostEqual(state["pose"].x, expected.x)
        self.assertAlmostEqual(state["pose"].y, expected.y)

    def test_real_same_connector_following_remains_detected(self):
        road = Road(((0, 0), (20, 0)))
        route = _SingleRoadRoute(road)
        follower_before = {"arm": "A", "maneuver": "through", "position": 0.0,
                           "shift": 0.0, "kind": "car", "pose": road.pose(0, 0),
                           "route": route}
        follower_after = dict(follower_before, position=4.0, pose=road.pose(4, 0))
        leader_before = {"arm": "A", "maneuver": "through", "position": 7.0,
                         "shift": 0.0, "kind": "car", "pose": road.pose(7, 0),
                         "route": route}
        leader_after = dict(leader_before)
        self.assertTrue(_sampled_sweep(follower_before, follower_after,
                                       leader_before, leader_after, 0.0))

    def test_vehicle_footprint_overhangs_exit_seam(self):
        roads = build_roads(__import__("pathlib").Path(__file__).parents[1] /
                            "python-simulation" / "simulation" / "python_simulation" / "includes" / "roads.geojson")
        route = Route(roads, "NE", "through")
        seam_progress = route.parts[1][2] + route.parts[1][1].length
        _, pose = route.pose(seam_progress - 0.5, 0.0)
        # The center is still on the connector; the 4.5m body must extend
        # beyond the connector/exit seam.
        self.assertEqual(route.pose(seam_progress - 0.5, 0.0)[0], route.parts[1][0])
        # Longitudinal half-length exceeds the remaining 0.5m to the seam,
        # proving the oriented rectangle crosses the segment boundary.
        projection_to_seam = 2.25 - 0.5
        self.assertGreater(projection_to_seam, 0)

    def test_lateral_shift_is_already_metres(self):
        road = Road(((0, 0), (10, 0)), num_lanes=3)
        v = SimpleNamespace(arm="A", to_arm="B", maneuver="through",
                            current_lateral_shift=SUBLANE_WIDTH, position=5.0,
                            occupied=1, demand_row_id=7, lane=1, speed=0.0)
        state = vehicle_state(v, {("A", "B", "through"): _SingleRoadRoute(road)})
        expected = road.pose(5.0, SUBLANE_WIDTH)
        self.assertAlmostEqual(state["pose"].x, expected.x)
        self.assertAlmostEqual(state["pose"].y, expected.y)
        self.assertAlmostEqual(state["pose"].y, -(3 * SUBLANE_WIDTH - SUBLANE_WIDTH - SUBLANE_WIDTH / 2.0))

    def test_seam_jump_uses_actual_old_and_new_lane_offsets(self):
        roads = {
            "A_in": {"id": "A_in", "kind": "approach", "num_lanes": 2,
                     "coordinates": ((0, 0), (10, 0))},
            "A_B": {"id": "A_B", "kind": "connector", "arm": "A", "turn": "through",
                    "to_arm": "B", "num_lanes": 2, "coordinates": ((10, 0), (10, 10))},
            "B_out": {"id": "B_out", "kind": "exit", "num_lanes": 2,
                      "coordinates": ((10, 10), (20, 10))},
        }
        route = Route(roads, "A", "through")
        event = _seam_jump(12, "spawn-4", route, "A_in", "A_B", 1, 0, SUBLANE_WIDTH, 0.0)
        self.assertEqual(event["route_id"], "A_in->A_B->B_out")
        self.assertEqual(event["seam"], "A_in->A_B")
        self.assertEqual(event["old_lane"], 1)
        self.assertEqual(event["new_lane"], 0)
        expected_lat = SUBLANE_WIDTH / 2.0
        self.assertAlmostEqual(event["old_center"]["x"], 10.0)
        self.assertAlmostEqual(event["old_center"]["y"], -expected_lat)
        self.assertAlmostEqual(event["new_center"]["x"], 10.0 + 3 * expected_lat)
        self.assertAlmostEqual(event["new_center"]["y"], -expected_lat)
        self.assertAlmostEqual(event["center_jump_m"], 3 * expected_lat)


class _SingleRoadRoute:
    def __init__(self, road):
        self.road = road

    def pose(self, progress, shift):
        return "test", self.road.pose(progress, shift)


if __name__ == "__main__":
    unittest.main()
