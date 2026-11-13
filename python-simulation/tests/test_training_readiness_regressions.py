"""Regressions for the Python training simulator's operational contract."""
import unittest

from python_simulation.python_sim import LeVanLuongSimulator, DT
from python_simulation.sim_config import load_sim_config
from python_simulation.vehicle_geometry import footprint, polygons_overlap


class TrainingReadinessRegressions(unittest.TestCase):
    def test_training_switch_observes_every_clearance_tick(self):
        for yellow, all_red in ((4.0, 2.0), (4.0, 0.0), (0.0, 0.0)):
            with self.subTest(yellow=yellow, all_red=all_red):
                sim = LeVanLuongSimulator(flow_per_approach=0, train_mode=True,
                                          yellow=yellow, all_red=all_red)
                sim.step(50)
                sim.set_action(2)
                observed = []
                for _ in range(int(round((yellow + all_red) / DT)) + 1):
                    sim.step()
                    observed.append((sim.phase, sim.yellow_phase))
                self.assertEqual(sum(p == "none" and y == "LVL" for p, y in observed),
                                 round(yellow / DT))
                self.assertEqual(sum(p == "none" and y is None for p, y in observed),
                                 round(all_red / DT))
                self.assertEqual(observed[-1][0], "HDT")

    def test_shared_default_and_calibrated_parameters(self):
        defaults = load_sim_config()
        self.assertEqual(defaults["junction_model"], "negotiation")
        sim = LeVanLuongSimulator(flow_per_approach=0, **defaults)
        self.assertEqual(sim.junction_model, "negotiation")
        configured = load_sim_config(config={"kwargs": {
            "vehicle_params": {"motorbike": {"headway": 0.75}},
            "startup_params": {"reaction_delay": 0.7}}})
        sim = LeVanLuongSimulator(flow_per_approach=0, **configured)
        self.assertEqual(sim.vehicle_params["motorbike"]["headway"], 0.75)
        self.assertEqual(sim.startup_params["reaction_delay"], 0.7)
        self.assertEqual(load_sim_config(config={
            "python_sim_mode": "safe", "simulation_config": configured}), configured)

    def test_connector_bodies_do_not_overlap_at_previous_failure(self):
        sim = LeVanLuongSimulator(seed=1)
        sim.step(323)
        by_connector = {}
        for vehicle in sim.vehicles:
            if not vehicle.road.endswith(("_in", "_out")):
                by_connector.setdefault(vehicle.road, []).append(vehicle)
        for connector, vehicles in by_connector.items():
            road = sim._geometry_roads[connector]
            bodies = [(vehicle, footprint(
                road.pose(vehicle.position - vehicle.approach_end,
                          vehicle.current_lateral_shift),
                length=vehicle.length, width=2 * sim._half_width(vehicle)))
                for vehicle in vehicles]
            for index, (a, body_a) in enumerate(bodies):
                for b, body_b in bodies[index + 1:]:
                    with self.subTest(connector=connector,
                                      pair=(a.creation_id, b.creation_id)):
                        self.assertFalse(polygons_overlap(body_a, body_b))

    def test_exit_seam_preserves_center_position(self):
        sim = LeVanLuongSimulator(flow_per_approach=0)
        old, new = sim._geometry_roads["NW_SE"], sim._geometry_roads["SE_out"]
        for shift in (0.0, 1.05, 3.15, 7.35):
            progress, new_shift = sim._continuous_seam_pose("NW_SE", "SE_out", shift)
            before = old.pose(old.length, shift)
            after = new.pose(progress, new_shift)
            self.assertLess(((before.x-after.x)**2 + (before.y-after.y)**2)**0.5,
                            0.1)

    def test_side_by_side_motorcycles_cross_exit_seam_without_false_leader(self):
        sim = LeVanLuongSimulator(seed=1, flow_per_approach=0)
        for shift, distance in ((-0.885, 0.625), (0.0, 0.595)):
            sim._create_vehicle("NW", "through", False, 0, 40)
            vehicle = sim.vehicles[-1]
            vehicle.road = vehicle.conn_road
            vehicle.position = vehicle.connector_end - distance
            vehicle.speed = 10.5
            vehicle.current_lateral_shift = shift
            vehicle.start_lateral_shift = shift
            vehicle.target_lateral_shift = shift
        sim.step()
        self.assertTrue(all(v.road == v.out_road and v.speed > 8.0
                            for v in sim.vehicles))

    def test_distant_vehicle_stops_for_yellow_it_cannot_clear(self):
        sim = LeVanLuongSimulator(seed=1, flow_per_approach=0)
        sim._create_vehicle("SW", "through", False, 0, 35)
        vehicle = sim.vehicles[-1]
        vehicle.position = vehicle.approach_end - 68.0
        vehicle.speed = 10.0
        vehicle.desired = 10.0
        self.assertFalse(sim._decide_stop_or_go(vehicle))

    def test_red_right_turn_enters_only_when_through_exit_is_clear(self):
        for model in ("negotiation", "reservation"):
            with self.subTest(model=model):
                sim = LeVanLuongSimulator(seed=2, flow_per_approach=0,
                                          junction_model=model)
                sim._create_vehicle("SE", "right", False, 0, 35)
                right = sim.vehicles[-1]
                right.position = right.approach_end - 2
                right.speed = 3
                self.assertFalse(sim._arm_is_green("SE"))
                self.assertTrue(sim._green_for(right))

                sim._create_vehicle("SW", "through", False, 3, 40)
                through = sim.vehicles[-1]
                # The through vehicle has merged to the curb sublane of the
                # common exit; a red right turn must yield to its full body.
                through.road = through.out_road
                through.position = through.connector_end + 0.5
                through.lane = 0
                through.current_lateral_shift = 0.0
                through.start_lateral_shift = 0.0
                through.target_lateral_shift = 0.0
                through.speed = 0.0
                self.assertFalse(sim._destination_has_room(right))
                sim.step()
                self.assertEqual(right.road, right.in_road)
                self.assertNotIn(right.creation_id, sim._junction_grants)

                free = LeVanLuongSimulator(seed=2, flow_per_approach=0,
                                           junction_model=model)
                free._create_vehicle("SE", "right", False, 0, 35)
                right = free.vehicles[-1]
                right.position = right.approach_end - 2
                right.speed = 3
                for _ in range(20):
                    free.step()
                    if right.road == right.conn_road:
                        break
                self.assertEqual(right.road, right.conn_road)
                self.assertEqual(free.phase, "LVL")
                self.assertEqual(free.entries_during_red, 0)
                self.assertEqual(free.permitted_right_on_red_entries, 1)

    def test_red_right_turn_yields_to_through_vehicle_at_merge(self):
        sim = LeVanLuongSimulator(seed=2, flow_per_approach=0,
                                  junction_model="negotiation")
        sim._create_vehicle("SE", "right", False, 0, 35)
        right = sim.vehicles[-1]
        right.position = right.approach_end - 2
        right.speed = 3
        sim._create_vehicle("SW", "through", False, 0, 40)
        through = sim.vehicles[-1]
        through.road = through.conn_road
        through.position = through.approach_end + 25
        through.lane = 0
        through.current_lateral_shift = 0.0
        through.start_lateral_shift = 0.0
        through.target_lateral_shift = 0.0
        through.speed = 4

        def body(vehicle):
            road, progress = sim._road_state(vehicle)
            return footprint(sim._geometry_roads[road].pose(
                progress, vehicle.current_lateral_shift), length=vehicle.length,
                width=2 * sim._half_width(vehicle))

        for tick in range(35):
            sim.step()
            if tick == 15:
                # The through vehicle is already in the merge zone. The
                # right turn must not force it into an instant stop.
                self.assertGreater(through.speed, 8.0)
                self.assertLess(right.position - right.approach_end, 15.0)
            self.assertFalse(polygons_overlap(body(right), body(through)))
        self.assertEqual(right.road, right.out_road)


if __name__ == "__main__":
    unittest.main()
