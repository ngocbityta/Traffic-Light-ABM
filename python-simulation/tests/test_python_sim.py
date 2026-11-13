"""Checks the GeoJSON-backed Python layout and GAML signal schedule."""

import unittest
import csv
import tempfile
from pathlib import Path
import math

from python_simulation.python_sim import (DT, LATERAL_CLEARANCE,
                                               LeVanLuongSimulator, road_lengths)
from python_simulation.vehicle_geometry import SUBLANE_WIDTH
from python_simulation.demand import generate_demand


class LeVanLuongTests(unittest.TestCase):
    def test_non_finite_simulation_parameters_are_rejected(self):
        for name in ("flow_per_approach", "motorbike_ratio", "p_left",
                     "p_right", "green_lvl", "green_hdt", "all_red"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(parameter=name, value=value):
                    with self.assertRaises(ValueError):
                        LeVanLuongSimulator(**{name: value})

    def test_geometry_and_empty_network_phase(self):
        roads = road_lengths()
        self.assertEqual(len(roads), 20)
        self.assertEqual({roads[f"{arm}_in"]["kind"] for arm in ("NE", "SE", "SW", "NW")},
                         {"approach"})
        # Legacy fixed plan without a yellow interval.
        sim = LeVanLuongSimulator(flow_per_approach=0, yellow=0.0)
        self.assertEqual(sim.phase, "LVL")
        self.assertEqual(sim.step(250)["phase"], "LVL")
        self.assertEqual(sim.step(1)["phase"], "none")
        self.assertEqual(sim.step(15)["phase"], "HDT")
        self.assertEqual(sim.step(200)["phase"], "none")
        self.assertEqual(sim.step(15)["phase"], "LVL")
        self.assertEqual(sim.created, 0)
        self.assertEqual(sim.queue_time_integral, 0)

    def test_seeded_arrivals_repeat(self):
        a = LeVanLuongSimulator(seed=12)
        b = LeVanLuongSimulator(seed=12)
        self.assertEqual(a.step(500), b.step(500))

    def test_paired_csv_creates_exact_requested_vehicles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demand.csv"
            digest = generate_demand(path, seed=12, horizon=200)
            self.assertEqual(digest, generate_demand(path, seed=12, horizon=200))
            with path.open(newline="") as stream:
                requests = list(csv.reader(stream))
            sim = LeVanLuongSimulator(seed=999, demand_file=path)
            sim.step(200)
            self.assertEqual(sim.created, len(requests))
            empty = Path(directory) / "empty.csv"
            empty.write_text("")
            self.assertEqual(LeVanLuongSimulator(demand_file=empty).step(200)["created"], 0)

    def test_replay_rejects_invalid_rows(self):
        invalid_rows = ("0,0,0,40,2,0\n", "1,0,0,nan,2,0\n",
                        "1,0,0,40,2,999\n", "1,4,0,40,2,0\n",
                        "2,0,0,40,2,0\n1,1,0,40,2,0\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            for content in invalid_rows:
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaises(ValueError):
                        LeVanLuongSimulator(demand_file=path)

    def test_spawn_spacing_and_turn_lanes(self):
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NE", "right", True, 0, 40)
        sim._create_vehicle("NE", "right", True, 0, 40)
        front, rear = sim.vehicles
        self.assertGreaterEqual(front.position - rear.position,
                                (front.length + rear.length) / 2 + rear.gap)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demand.csv"
            generate_demand(path, seed=1, horizon=1000)
            sim2 = LeVanLuongSimulator(flow_per_approach=0)
            with path.open(newline="") as stream:
                for _, origin, car, _, turn, lane in csv.reader(stream):
                    if int(turn) not in (0, 1):
                        continue
                    arm = ("NE", "SE", "SW", "NW")[int(origin)]
                    sim2.vehicles.clear()
                    direction = "left" if int(turn) == 0 else "right"
                    sim2._create_vehicle(arm, direction, bool(int(car)), int(lane), 40)
                    self.assertEqual(sim2._route_target_lane(sim2.vehicles[0]), int(lane))
                    if direction == "left":
                        self.assertGreaterEqual(int(lane),
                            sim2.roads[f"{arm}_in"]["num_lanes"] - (2 if int(car) else 1) - 2)

    def test_adjacent_car_and_bike_use_physical_side_gap(self):
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NE", "through", True, 2, 30)
        sim._create_vehicle("NE", "through", False, 1, 30)
        car, bike = sim.vehicles
        car.position = bike.position = 20.0
        # Place bike laterally close enough (e.g. mid-lane-change) to intrude into car's clearance envelope.
        bike.current_lateral_shift = 2.5 * SUBLANE_WIDTH - 1.1
        lane_maps = {"NE_in": [[] for _ in range(sim.roads["NE_in"]["num_lanes"])]}
        sim._neighbor_road_state = {id(car): ("NE_in", 20.0),
                                    id(bike): ("NE_in", 20.0)}
        for item in sim.vehicles:
            for sub in range(item.lane, item.lane + item.occupied):
                lane_maps["NE_in"][sub].append((20.0, id(item), item))
            lane_maps["NE_in"] = [sorted(entries) for entries in lane_maps["NE_in"]]

        leader, gap, _, _ = sim._neighbors(car, 2, lane_maps)

        self.assertIs(leader, bike)
        self.assertLess(gap, 0.0)

    def test_road_links_keep_body_clearance(self):
        sim = LeVanLuongSimulator(seed=1)
        for _ in range(1000):
            sim.step()
            by_road = {}
            for vehicle in sim.vehicles:
                road, progress = sim._road_state(vehicle)
                if road.endswith(("_in", "_out")):
                    by_road.setdefault(road, []).append((progress, vehicle))
            for entries in by_road.values():
                entries.sort(key=lambda item: item[0])
                for index, (behind_progress, behind) in enumerate(entries):
                    for ahead_progress, ahead in entries[index + 1:]:
                        if ahead_progress - behind_progress > 6.0:
                            break
                        widths = sim._half_width(behind) + sim._half_width(ahead) + LATERAL_CLEARANCE
                        if widths - abs(behind.current_lateral_shift -
                                        ahead.current_lateral_shift) <= 0:
                            continue
                        gap = ahead_progress - behind_progress - (
                            behind.length + ahead.length) / 2
                        self.assertGreaterEqual(gap + 1e-8,
                                                max(1.0, behind.gap, ahead.gap))

    def test_red_light_keeps_whole_vehicle_before_stop_line(self):
        sim = LeVanLuongSimulator(seed=1)
        for _ in range(1000):
            sim.step()
            for vehicle in sim.vehicles:
                road, _ = sim._road_state(vehicle)
                if not road.endswith("_in"):
                    continue
                # Green, or a "go" decision taken at the yellow onset.
                green = sim._green_for(vehicle)
                if not green:
                    if vehicle.position > sim._stop_position(vehicle) + 1e-8:
                        # Entered on yellow (go decision) and still clearing
                        # into the next phase: it must keep moving.
                        self.assertTrue(vehicle.junction_entry_token)
                        self.assertGreater(vehicle.speed, 0.0)
                        continue
                    self.assertLessEqual(vehicle.position + vehicle.length / 2 + 0.5,
                                         vehicle.approach_end + 1e-8)

    def test_lateral_sweep_does_not_intersect_another_body(self):
        def inside(a, slope, radius):
            if abs(slope) < 1e-12:
                return (0.0, 1.0) if abs(a) < radius else None
            first, last = sorted(((-radius - a) / slope, (radius - a) / slope))
            first, last = max(0.0, first), min(1.0, last)
            return (first, last) if first + 1e-8 < last else None

        sim = LeVanLuongSimulator(seed=1)
        for _ in range(1000):
            previous = {id(v): (*sim._road_state(v), v.current_lateral_shift)
                        for v in sim.vehicles}
            sim.step()
            by_road = {}
            for vehicle in sim.vehicles:
                road, progress = sim._road_state(vehicle)
                before = previous.get(id(vehicle))
                if road.endswith(("_in", "_out")) and before and before[0] == road:
                    by_road.setdefault(road, []).append(
                        (vehicle, before[1], before[2], progress,
                         vehicle.current_lateral_shift))
            for entries in by_road.values():
                for index, (a, ap0, ay0, ap1, ay1) in enumerate(entries):
                    for b, bp0, by0, bp1, by1 in entries[index + 1:]:
                        if min(abs(ap0 - bp0), abs(ap1 - bp1)) > 9:
                            continue
                        longitudinal = inside(ap0 - bp0,
                            (ap1 - ap0) - (bp1 - bp0), (a.length + b.length) / 2 - 0.05)
                        if longitudinal is None:
                            continue
                        lateral = inside(ay0 - by0,
                            (ay1 - ay0) - (by1 - by0),
                            sim._half_width(a) + sim._half_width(b) - 0.05)
                        self.assertFalse(lateral and
                            max(longitudinal[0], lateral[0]) <
                            min(longitudinal[1], lateral[1]))

    def test_vehicle_exits_without_braking_at_destination(self):
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NE", "through", False, 0, 40)
        vehicle = sim.vehicles[0]
        vehicle.road = f"{vehicle.to_arm}_out"
        vehicle.position = vehicle.route_end - 1.0
        vehicle.speed = 10.0
        sim.step()
        self.assertEqual(sim.passed, 1)
        self.assertEqual(len(sim.vehicles), 0)

    def test_exit_queue_holds_vehicle_on_connector_until_clearance_exists(self):
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("SW", "through", False, 0, 40)
        sim._create_vehicle("NW", "left", False, 0, 40)
        rear, ahead = sim.vehicles
        for vehicle, progress in ((rear, 0.5), (ahead, 2.0)):
            vehicle.road = "NE_out"
            vehicle.position = vehicle.connector_end + progress
            vehicle.lane = 0
            vehicle.current_lateral_shift = SUBLANE_WIDTH / 2.0

        sim._resolve_road_conflicts()

        road, _ = sim._road_state(rear)
        self.assertEqual(road, f"{rear.arm}_{rear.to_arm}")
        ahead_progress = ahead.position - ahead.connector_end
        rear_projected_exit_progress = rear.position - rear.connector_end
        actual_gap = (ahead_progress - rear_projected_exit_progress -
                      (rear.length + ahead.length) / 2)
        self.assertGreaterEqual(actual_gap + 1e-9,
                                max(1.0, rear.gap, ahead.gap))

    def test_exit_queue_return_maps_lane_back_into_connector(self):
        sim = LeVanLuongSimulator(flow_per_approach=0)
        nw_max = sim.roads["NW_in"]["num_lanes"] - 1
        ne_max = sim.roads["NE_out"]["num_lanes"] - 1
        sim._create_vehicle("NW", "left", False, nw_max, 40)
        sim._create_vehicle("SW", "through", False, ne_max, 40)
        rear, ahead = sim.vehicles
        for vehicle, progress in ((rear, 0.5), (ahead, 2.0)):
            vehicle.road = "NE_out"
            vehicle.position = vehicle.connector_end + progress
            vehicle.lane = ne_max
            vehicle.current_lateral_shift = ne_max * SUBLANE_WIDTH

        sim._resolve_road_conflicts()

        road, _ = sim._road_state(rear)
        self.assertEqual(road, "NW_NE")
        self.assertLess(rear.lane + rear.occupied,
                        sim.roads[road]["num_lanes"] + 1)
        self.assertAlmostEqual(rear.current_lateral_shift,
                               rear.lane * SUBLANE_WIDTH)

    def test_boundary_crossing_integrates_new_road_acceleration_for_remainder(self):
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NE", "through", False, 3, 40)
        vehicle = sim.vehicles[0]
        vehicle.position = vehicle.approach_end - 0.2
        vehicle.speed = 8.0
        sim.phase = "LVL"
        old_acceleration, new_acceleration = 1.0, 2.0

        def controlled_idm(v, gap, leader_speed):
            return (new_acceleration if v.position >= v.approach_end
                    else old_acceleration)

        sim._idm = controlled_idm
        sim.step()

        crossing_speed = 8.0 + old_acceleration * DT
        remaining = DT - 0.2 / crossing_speed
        self.assertAlmostEqual(vehicle.speed,
                               crossing_speed + new_acceleration * remaining)
        self.assertAlmostEqual(vehicle.position,
                               vehicle.approach_end + vehicle.seam_progress +
                               crossing_speed * remaining +
                               0.5 * new_acceleration * remaining ** 2)

    def test_blocked_follower_speed_matches_capped_displacement(self):
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NE", "through", False, 5, 40)
        sim._create_vehicle("NE", "through", False, 5, 40)
        front, rear = sim.vehicles
        front.position, front.speed = 20.0, 0.0
        rear.position, rear.speed = 17.0, 2.0
        sim._idm = lambda *_: 0.0
        initial_rear_position = rear.position

        sim.step()

        self.assertLessEqual(rear.position - initial_rear_position, 0.1 + 1e-8)
        self.assertEqual(rear.speed, 0.0)

    def test_stopped_vehicle_does_not_finish_lane_change_at_red_light(self):
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NW", "through", False, 5, 40)
        vehicle = sim.vehicles[0]
        vehicle.position = sim._stop_position(vehicle)
        vehicle.speed = 0.0
        vehicle.lane_change_from = 4
        vehicle.lane_change_progress = 0.25
        vehicle.start_lateral_shift = 3.2
        vehicle.current_lateral_shift = 3.325
        vehicle.target_lateral_shift = 4.0
        sim.step(10)
        self.assertEqual(vehicle.lane, 5)
        self.assertEqual(vehicle.lane_change_progress, 0.25)
        self.assertEqual(vehicle.current_lateral_shift, 3.325)
        self.assertEqual(vehicle.position, sim._stop_position(vehicle))

    def test_signal_change_to_red_does_not_rewind_committed_vehicle(self):
        sim = LeVanLuongSimulator(flow_per_approach=0)
        sim._create_vehicle("NW", "through", False, 0, 40)
        vehicle = sim.vehicles[0]
        vehicle.position = sim._stop_position(vehicle) + 0.4
        vehicle.speed = 5.0
        sim.phase = "none"
        prior_position = vehicle.position

        sim._move()

        self.assertGreaterEqual(vehicle.position, prior_position)
        self.assertGreaterEqual(vehicle.speed, 0.0)

    def test_high_flow_red_phase_holds_uncommitted_vehicles_and_clears_committed(self):
        sim = LeVanLuongSimulator(seed=42, flow_per_approach=10000)
        committed_on_green = set()
        for _ in range(500):
            sim.step()
            for vehicle in sim.vehicles:
                road, _ = sim._road_state(vehicle)
                if not road.endswith("_in"):
                    continue
                # Green, or a "go" decision taken at the yellow onset.
                green = sim._green_for(vehicle)
                if not green:
                    committed = vehicle.creation_id in committed_on_green
                    if committed:
                        # Green commitment is allowed to clear the approach
                        # during all-red. It must keep moving until it leaves.
                        self.assertGreater(vehicle.speed, 0.0)
                    else:
                        self.assertLessEqual(
                            vehicle.position + vehicle.length / 2 + 0.5,
                            vehicle.approach_end + 1e-8)
                    if vehicle.speed <= 0.1:
                        self.assertLessEqual(
                            vehicle.position,
                            sim._stop_position(vehicle) + 1e-8)
                elif vehicle.position > sim._stop_position(vehicle) + 1e-8:
                    committed_on_green.add(vehicle.creation_id)

    def test_training_action_switch_keeps_front_before_red_stop_line(self):
        sim = LeVanLuongSimulator(seed=1, train_mode=True)
        for _ in range(150):
            if sim.dqn_waiting_action:
                sim.set_action(2 if sim.phase == "LVL" else 0)
            sim.step()
            for vehicle in sim.vehicles:
                road, _ = sim._road_state(vehicle)
                if not road.endswith("_in"):
                    continue
                # Green, or a "go" decision taken at the yellow onset.
                green = sim._green_for(vehicle)
                if not green:
                    self.assertLessEqual(
                        vehicle.position + vehicle.length / 2 + 0.5,
                        vehicle.approach_end + 1e-8)

    def test_vehicle_committed_on_green_clears_approach_during_all_red(self):
        sim = LeVanLuongSimulator(flow_per_approach=0, train_mode=True)
        sim._create_vehicle("NW", "through", True, 5, 40)
        vehicle = sim.vehicles[0]
        vehicle.position = sim._stop_position(vehicle) + 0.5
        vehicle.speed = 0.0
        sim.phase = "none"

        sim.step(10)

        self.assertNotEqual(vehicle.road, "NW_in")
        self.assertGreaterEqual(vehicle.position, vehicle.approach_end)

    def test_training_clock_and_mobil_cooldown(self):
        # Legacy transition (all-red only); the yellow variant is tested below.
        sim = LeVanLuongSimulator(flow_per_approach=0, train_mode=True, yellow=0.0)
        sim._create_vehicle("NE", "through", False, 10, 40)
        sim.step(19)
        self.assertEqual(sim._route_target_lane(sim.vehicles[0]), sim.vehicles[0].lane)
        sim.step(1)
        self.assertEqual(sim._route_target_lane(sim.vehicles[0]), sim.vehicles[0].lane)
        sim.step(1)
        self.assertEqual(sim._route_target_lane(sim.vehicles[0]), sim.vehicles[0].lane)
        sim.step(29)
        self.assertTrue(sim.dqn_waiting_action)
        self.assertEqual(sim.cycle, 50)
        sim.set_action(2)
        sim.step(14)
        self.assertEqual(sim.phase, "none")
        sim.step(1)
        self.assertEqual(sim.phase, "HDT")
        sim.step(35)
        self.assertTrue(sim.dqn_waiting_action)
        self.assertEqual(sim.cycle, 100)

    def test_training_switch_shows_yellow_then_all_red(self):
        sim = LeVanLuongSimulator(flow_per_approach=0, train_mode=True, yellow=3.0)
        sim.step(50)
        self.assertTrue(sim.dqn_waiting_action)
        sim.set_action(2)
        sim.step(1)
        self.assertEqual((sim.phase, sim.yellow_phase), ("none", "LVL"))
        sim.step(14)
        self.assertEqual((sim.phase, sim.yellow_phase), ("none", "LVL"))
        sim.step(1)
        self.assertEqual((sim.phase, sim.yellow_phase), ("none", None))
        sim.step(14)
        self.assertEqual(sim.phase, "HDT")

    def test_fixed_plan_yellow_is_carved_from_green(self):
        sim = LeVanLuongSimulator(flow_per_approach=0, yellow=3.0)
        # LVL green 50 s = 47 s green + 3 s yellow; onsets are unchanged.
        self.assertEqual(sim._signal(1 + int(46.8 / DT)), ("LVL", None))
        self.assertEqual(sim._signal(1 + int(47.2 / DT)), ("none", "LVL"))
        self.assertEqual(sim._signal(1 + int(50.4 / DT)), ("none", None))
        self.assertEqual(sim._signal(1 + int(53.2 / DT)), ("HDT", None))
        self.assertEqual(sim._signal(1 + int(90.2 / DT)), ("none", "HDT"))
        legacy = LeVanLuongSimulator(flow_per_approach=0, yellow=0.0)
        for cycle in range(1, 481):
            self.assertIsNone(legacy._signal(cycle)[1])

    def test_stop_or_go_logit_matches_paper_coefficients(self):
        sim = LeVanLuongSimulator(flow_per_approach=0)
        lanes = sim.roads["NE_in"]["num_lanes"]
        sim._create_vehicle("NE", "through", False, 3, 40)  # kerb-side third
        moto = sim.vehicles[-1]
        moto.speed = 10.0
        moto.position = moto.approach_end - moto.length / 2 - 30.0
        u = 4.518165 + 0.106 * 10.0 - 0.083 * 30.0
        self.assertLess(moto.current_lateral_shift / (lanes * SUBLANE_WIDTH), 1 / 3)
        self.assertAlmostEqual(sim.stop_or_go_probability(moto), 1 / (1 + math.exp(-u)))
        moto.position = moto.approach_end - moto.length / 2 - 140.0
        self.assertLess(sim.stop_or_go_probability(moto), 0.01)

    def test_go_decision_persists_into_all_red_and_lapses_at_next_green(self):
        sim = LeVanLuongSimulator(flow_per_approach=0, yellow=3.0)
        sim._create_vehicle("NE", "through", False, 4, 40)
        vehicle = sim.vehicles[-1]
        sim.cycle = 1 + int(47.2 / DT)
        sim.phase, sim.yellow_phase = sim._signal()
        sim._yellow_epoch += 1
        sim._update_last_yellow()
        vehicle.position = sim._stop_position(vehicle) - 5.0
        vehicle.speed = 9.0  # 81/(2*5) > STOP_GO_MAX_DECEL: cannot stop
        self.assertTrue(sim._green_for(vehicle))
        sim.cycle = 1 + int(51.0 / DT)
        sim.phase, sim.yellow_phase = sim._signal()
        sim._update_last_yellow()
        self.assertEqual((sim.phase, sim.yellow_phase), ("none", None))
        self.assertTrue(sim._green_for(vehicle))
        sim.cycle = 1 + int(54.0 / DT)
        sim.phase, sim.yellow_phase = sim._signal()
        sim._update_last_yellow()
        self.assertFalse(sim._green_for(vehicle))

    def test_mobil_resolves_spawn_overlap_before_cooldown(self):
        sim = LeVanLuongSimulator(flow_per_approach=0)
        lane = sim.roads["NE_in"]["num_lanes"] - 2
        sim._create_vehicle("NE", "through", False, lane, 40)
        sim._create_vehicle("NE", "through", False, lane, 40)
        initial_lane = sim.vehicles[0].lane
        sim.step(1)
        self.assertTrue(all(abs(vehicle.lane - initial_lane) <= 1 for vehicle in sim.vehicles))
        self.assertTrue(all(vehicle.speed > 0 for vehicle in sim.vehicles))

    def test_through_vehicle_stays_out_of_right_turn_queue(self):
        for is_car in (False, True):
            sim = LeVanLuongSimulator(flow_per_approach=0)
            sim._create_vehicle("NW", "through", is_car, 0, 40)
            vehicle = sim.vehicles[0]
            self.assertEqual(vehicle.lane, 0)
            sim.step(250)
            self.assertGreaterEqual(vehicle.lane, sim.roads["NW_in"]["num_lanes"] // 4)
            self.assertEqual(sim._route_target_lane(vehicle), vehicle.lane)
            if not is_car:
                sim.step(150)
                self.assertGreater(sim.passed, 0)

    def test_red_queue_staggers_stops_and_motorcycle_filters(self):
        sim = LeVanLuongSimulator(seed=8, flow_per_approach=0)
        for lane in (4, 5, 6, 7, 8, 9):
            sim._create_vehicle("NW", "through", False, lane, 40)
        offsets = {round(v.queue_stop_offset, 2) for v in sim.vehicles}
        self.assertGreater(len(offsets), 1)
        self.assertTrue(all(sim._stop_position(v) < v.approach_end - v.length / 2
                            for v in sim.vehicles))
        sim.vehicles.clear()
        sim._create_vehicle("NW", "through", True, 5, 40)
        sim._create_vehicle("NW", "through", False, 5, 40)
        car, bike = sim.vehicles
        car.position = car.approach_end - 15.0
        bike.position = bike.approach_end - 25.0
        bike.speed = 3.0
        bike.time_since_lane_change = 2.0
        sim.step(1)
        self.assertNotEqual(bike.lane, 5)

    def test_lane_changes_are_adjacent(self):
        with tempfile.TemporaryDirectory() as directory:
            demand = Path(directory) / "seed1.csv"
            generate_demand(demand, seed=1, horizon=50)
            sim = LeVanLuongSimulator(demand_file=demand)
            sim.step(36)
            row8 = next(v for v in sim.vehicles if v.demand_row_id == 8)
            before = row8.lane
            sim.step()
            self.assertLessEqual(abs(row8.lane - before), 1)

    def test_seed1_200_tick_replay_keeps_sequential_lane_map_consistent(self):
        sim = LeVanLuongSimulator(seed=1)
        result = sim.step(200)
        self.assertGreater(result["created"], 0)
        self.assertEqual(result["created"], result["passed"] + result["active"])

    def test_seed1_idm_speed_remains_finite_at_tick_68(self):
        with tempfile.TemporaryDirectory() as directory:
            demand = Path(directory) / "seed1.csv"
            generate_demand(demand, seed=1, horizon=200)
            sim = LeVanLuongSimulator(demand_file=demand)
            sim.step(68)
            row5 = next(v for v in sim.vehicles if v.demand_row_id == 5)
            self.assertGreaterEqual(row5.speed, 0)
            self.assertLess(row5.speed, row5.desired + 1)

    def test_vehicles_remain_on_their_route(self):
        with tempfile.TemporaryDirectory() as directory:
            demand = Path(directory) / "seed1.csv"
            generate_demand(demand, seed=1, horizon=200)
            sim = LeVanLuongSimulator(demand_file=demand)
            sim.step(74)
            row5 = next(v for v in sim.vehicles if v.demand_row_id == 5)
            road = sim._road_state(row5)[0]
            self.assertIn(road, (f"{row5.arm}_in", f"{row5.arm}_{row5.to_arm}",
                                 f"{row5.to_arm}_out"))

    def test_lanes_stay_within_road_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            demand = Path(directory) / "seed1.csv"
            generate_demand(demand, seed=1, horizon=200)
            sim = LeVanLuongSimulator(demand_file=demand)
            sim.step(86)
            vehicles = {v.demand_row_id: v for v in sim.vehicles}
            for vehicle in vehicles.values():
                road = sim._road_state(vehicle)[0]
                self.assertGreaterEqual(vehicle.lane, 0)
                self.assertLessEqual(vehicle.lane + vehicle.occupied,
                                     sim.roads[road]["num_lanes"])

if __name__ == "__main__":
    unittest.main()


class SpatialIndexEquivalenceTests(unittest.TestCase):
    """The road index is only a candidate filter: every neighbour query must
    return exactly what the original full scan over all vehicles returns."""

    def test_indexed_neighbors_match_full_scan(self):
        sim = LeVanLuongSimulator(seed=3)
        original = LeVanLuongSimulator._neighbors
        checked = [0]

        def checked_neighbors(self, vehicle, lane, lane_maps, shift=None):
            fast = original(self, vehicle, lane, lane_maps, shift)
            saved = self._road_index
            self._road_index = None
            try:
                slow = original(self, vehicle, lane, lane_maps, shift)
            finally:
                self._road_index = saved
            checked[0] += 1
            assert fast[0] is slow[0] and fast[2] is slow[2], (fast, slow)
            assert fast[1] == slow[1] and fast[3] == slow[3], (fast, slow)
            return fast

        sim._neighbors = checked_neighbors.__get__(sim)
        sim.step(400)
        self.assertGreater(checked[0], 1000)


class ResolverIsOnlyAGuardTests(unittest.TestCase):
    """After unifying the neighbour predicate (gap-based leader, body
    position, no double-counted leader advance) the post-move resolver must
    practically never fire; before 13/11 it moved ~10 vehicles per tick."""

    def test_resolver_corrections_are_negligible(self):
        sim = LeVanLuongSimulator(seed=2)
        sim.step(1200)
        self.assertLess(sim.max_road_safety_correction, 0.01)
        self.assertLess(sim.road_safety_corrections, 20)

    def test_vehicles_never_move_backwards(self):
        sim = LeVanLuongSimulator(seed=1)
        for _ in range(800):
            before = {id(v): v.position for v in sim.vehicles}
            sim.step()
            for v in sim.vehicles:
                if id(v) in before:
                    self.assertGreaterEqual(v.position, before[id(v)] - 1e-9)
