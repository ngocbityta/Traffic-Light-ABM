import unittest

from python_simulation.safe_startup import SafeStartupScheduler, StartupVehicle


def vehicle(identifier, road="R", lane=0, position=90.0, *, speed=0.0,
            line=100.0, committed=False, occupied=1):
    return StartupVehicle(identifier, road, lane, position, speed, line,
                          committed, occupied)


class SafeStartupTests(unittest.TestCase):
    def scheduler(self):
        return SafeStartupScheduler(seed=17, reaction_delay=0.5,
                                    discharge_headway=0.8)

    def test_repeatable_release_times(self):
        queue = [vehicle("a", position=99), vehicle("b", position=95),
                 vehicle("c", position=91)]
        first, second = self.scheduler(), self.scheduler()
        first.update(10, True, queue)
        second.update(10, True, queue)
        self.assertEqual([first.release_time(v.vehicle_id) for v in queue],
                         [second.release_time(v.vehicle_id) for v in queue])

    def test_lanes_schedule_independently_and_followers_keep_headway(self):
        queue = [vehicle("near", lane=0, position=99),
                 vehicle("far", lane=0, position=94),
                 vehicle("other-lane", lane=1, position=96)]
        scheduler = self.scheduler()
        scheduler.update(0, True, queue)
        near = scheduler.release_time("near")
        far = scheduler.release_time("far")
        other = scheduler.release_time("other-lane")
        self.assertGreaterEqual(far, near + 0.8)
        self.assertLess(other, far)

    def test_car_footprint_delays_bike_queue_in_overlapping_sublane(self):
        # The car spans sublanes 0 and 1, so a bike behind it in sublane 1
        # follows its discharge even though their nominal lane keys differ.
        queue = [vehicle("car", lane=0, position=99, occupied=2),
                 vehicle("bike", lane=1, position=95)]
        scheduler = self.scheduler()
        scheduler.update(0, True, queue)
        self.assertGreaterEqual(scheduler.release_time("bike"),
                                scheduler.release_time("car") + 0.8)

    def test_unscheduled_vehicles_remain_allowed_during_green(self):
        scheduler = self.scheduler()
        queue = [vehicle("stopped", position=99)]
        self.assertEqual(scheduler.update(0, True, queue), frozenset())
        # A moving vehicle joining later was not in the green-onset snapshot.
        joined = queue + [vehicle("joining", position=80, speed=1.0)]
        released = scheduler.update(0.000001, True, joined)
        self.assertIn("joining", released)
        self.assertNotIn("stopped", released)

    def test_red_resets_queue_for_next_green(self):
        scheduler = self.scheduler()
        queue = [vehicle("a", position=99), vehicle("b", position=95)]
        scheduler.update(0, True, queue)
        self.assertEqual(scheduler.update(1, False, queue), frozenset())
        self.assertIsNone(scheduler.release_time("a"))
        scheduler.update(5, True, queue)
        self.assertAlmostEqual(scheduler.release_time("a"),
                               5 + self.scheduler()._reaction("a"))

    def test_committed_vehicle_is_exempt_from_startup_delay(self):
        scheduler = self.scheduler()
        queue = [vehicle("stopped", position=99),
                 vehicle("inside", speed=2, position=102, committed=True)]
        self.assertEqual(scheduler.update(0, True, queue), frozenset({"inside"}))
        self.assertIn("inside", scheduler.update(0.01, True, queue))

    def test_red_position_constraint_never_crosses_stop_line(self):
        self.assertEqual(SafeStartupScheduler.constrain_position(
            101.2, 100, green=False), 100)
        self.assertEqual(SafeStartupScheduler.constrain_position(
            99.2, 100, green=False), 99.2)
        self.assertEqual(SafeStartupScheduler.constrain_position(
            101.2, 100, green=False, committed=True), 101.2)


if __name__ == "__main__":
    unittest.main()
