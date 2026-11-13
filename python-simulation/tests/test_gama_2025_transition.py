import unittest

from python_simulation.gama_2025_transition import transition_at_node


class Gama2025TransitionTests(unittest.TestCase):
    def test_node_crossing_keeps_old_speed_for_remaining_distance(self):
        result = transition_at_node(
            speed=10.0,
            old_road_acceleration=2.0,
            distance_to_node=6.0,
            remaining_time=1.0,
            dt=1.0,
            new_road_acceleration=1.0,
            current_lane=4,
            num_lanes_occupied=2,
            new_road_num_lanes=3,
        )

        # Old-road predicted speed is 12; GAMA estimates 0.5 s remains.
        # At the node it has not stored 12, so the new road starts from 10:
        # distance=10*.5 + .5*1*.5^2=5.125. Stored speed uses full DT: 11.
        self.assertTrue(result.crossed)
        self.assertAlmostEqual(result.remaining_time, 0.5)
        self.assertAlmostEqual(result.new_road_distance, 5.125)
        self.assertAlmostEqual(result.stored_speed, 11.0)
        self.assertEqual(result.new_lane, 1)

        # The current Python crossing-speed path instead starts from predicted
        # old-road speed (12), integrates its acceleration over remaining time,
        # and stores 12 + 1*.5 = 12.5.
        python_distance = 12.0 * 0.5 + 0.5 * 1.0 * 0.5**2
        python_speed = 12.0 + 1.0 * 0.5
        self.assertAlmostEqual(python_distance, 6.125)
        self.assertAlmostEqual(python_speed, 12.5)
        self.assertNotEqual(result.new_road_distance, python_distance)
        self.assertNotEqual(result.stored_speed, python_speed)

    def test_red_rejection_stops_at_node_without_lane_change(self):
        result = transition_at_node(
            speed=10.0,
            old_road_acceleration=0.0,
            distance_to_node=2.0,
            remaining_time=1.0,
            dt=1.0,
            new_road_acceleration=3.0,
            current_lane=3,
            num_lanes_occupied=2,
            new_road_num_lanes=1,
            red_blocked=True,
        )
        self.assertFalse(result.crossed)
        self.assertEqual(result.remaining_time, 1.0)
        self.assertEqual(result.new_road_distance, 0.0)
        self.assertEqual(result.stored_speed, 0.0)
        self.assertEqual(result.new_lane, 3)

    def test_zero_predicted_new_road_speed_does_not_enter_road(self):
        result = transition_at_node(
            speed=2.0,
            old_road_acceleration=0.0,
            distance_to_node=1.0,
            remaining_time=1.0,
            dt=1.0,
            new_road_acceleration=-2.0,
            current_lane=2,
            num_lanes_occupied=1,
            new_road_num_lanes=4,
        )
        self.assertFalse(result.crossed)
        self.assertEqual(result.remaining_time, 1.0)
        self.assertEqual(result.new_road_distance, 0.0)
        self.assertEqual(result.stored_speed, 0.0)
        self.assertEqual(result.new_lane, 2)


if __name__ == "__main__":
    unittest.main()
