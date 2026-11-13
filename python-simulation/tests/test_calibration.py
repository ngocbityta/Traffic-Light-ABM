import math
import tempfile
import unittest
from pathlib import Path

from python_simulation import calibration
from python_simulation.python_sim import DEFAULT_VEHICLE_PARAMS, LeVanLuongSimulator


class CalibrationTests(unittest.TestCase):
    def test_params_map_to_simulator_kwargs(self):
        kwargs = calibration.params_to_kwargs(
            {"moto_headway": 0.9, "car_gap": 1.5, "reaction_delay": 1.0})
        self.assertEqual(kwargs["vehicle_params"],
                         {"motorbike": {"headway": 0.9}, "car": {"gap": 1.5}})
        self.assertEqual(kwargs["startup_params"], {"reaction_delay": 1.0})
        sim = LeVanLuongSimulator(flow_per_approach=0, **kwargs)
        sim._create_vehicle("NE", "through", False, 5, 40)
        self.assertEqual(sim.vehicles[0].headway, 0.9)
        self.assertEqual(sim.vehicles[0].gap, DEFAULT_VEHICLE_PARAMS["motorbike"]["gap"])
        self.assertEqual(sim._startup_schedulers["NE"].reaction_delay, 1.0)

    def test_rejects_unknown_parameters(self):
        with self.assertRaises(ValueError):
            LeVanLuongSimulator(vehicle_params={"truck": {"gap": 2.0}})
        with self.assertRaises(ValueError):
            LeVanLuongSimulator(vehicle_params={"car": {"tau": 2.0}})

    def test_measure_discharge_reports_finite_saturation_flow(self):
        result = calibration.measure_discharge(seed=1)
        self.assertGreater(result["crossings"], 20)
        self.assertTrue(math.isfinite(result["saturation_flow"]))
        self.assertGreater(result["saturation_flow"], 0)

    def test_demand_from_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "counts.csv"
            path.write_text(
                "time_slot,station_id,station_name,direction,heading,motorbike,car,"
                "small_truck_bus,heavy_truck_bus,trailer,total_flow\n"
                "07h00-08h00,1,A,E,270,750,200,30,20,0,1000\n", encoding="utf-8")
            rows = calibration.demand_from_counts(path)
        self.assertEqual(rows[0]["flow_veh_h"], 1000)
        self.assertAlmostEqual(rows[0]["motorbike_ratio"], 0.75)
