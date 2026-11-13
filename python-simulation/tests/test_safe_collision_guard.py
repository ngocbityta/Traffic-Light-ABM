import unittest

from python_simulation.vehicle_geometry import RouteGeometry, Road
from python_simulation.safe_collision_guard import Motion, first_contact, first_contact_fraction, max_safe_fraction


def route(points, lanes=1, name="r"):
    return RouteGeometry([(name, Road(points, lanes))])


class SafeCollisionGuardTests(unittest.TestCase):
    def test_distinct_centers_with_overlapping_vehicle_bodies(self):
        r=route([(0, 0), (20, 0)])
        moving=Motion("a",r,2,2)
        parked=Motion("b",r,6,6)
        hit=first_contact(moving,[parked],clearance=0)
        self.assertIsNotNone(hit)
        self.assertTrue(hit.start_overlap)
        self.assertEqual(hit.fraction,0)

    def test_clearance_contact_is_not_reported_as_start_overlap(self):
        r=route([(0, 0), (20, 0)])
        moving=Motion("a",r,2,2)
        nearby=Motion("b",r,6.6,6.6)
        hit=first_contact(moving,[nearby],clearance=0.2)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.fraction,0)
        self.assertFalse(hit.start_overlap)

    def test_bike_lane_change_with_clear_endpoints(self):
        r=route([(0, 0), (20, 0)])
        changer=Motion("changing",r,5,5,0,-2,length=1.9,width=0.7)
        neighbor=Motion("neighbor",r,5,5,-1,-1,length=1.9,width=0.7)
        # Each endpoint is a full metre from the neighbor: both are clear, but
        # the lateral sweep crosses its occupied lane.
        self.assertIsNone(first_contact(Motion("start",r,5,5,0,0,length=1.9,width=0.7),
                                        [neighbor],clearance=0))
        self.assertIsNone(first_contact(Motion("end",r,5,5,-2,-2,length=1.9,width=0.7),
                                        [neighbor],clearance=0))
        hit=first_contact(changer,[neighbor])
        self.assertIsNotNone(hit)
        self.assertGreater(hit.fraction,0)
        self.assertLess(hit.fraction,1)
        self.assertFalse(hit.start_overlap)

    def test_crossing_connector_with_clear_endpoints(self):
        east=route([(0, 0),(10, 0)],name="east")
        north=route([(5,-5),(5,5)],name="north")
        moving=Motion("eastbound",east,0,10)
        stationary=Motion("northbound",north,5,5)
        hit=first_contact(moving,[stationary])
        self.assertIsNotNone(hit)
        self.assertGreater(hit.fraction,0)
        self.assertLess(hit.fraction,1)
        self.assertLess(max_safe_fraction(moving,[stationary]),1)

    def test_route_seam_is_sampled(self):
        r=RouteGeometry([("one",Road([(0,0),(5,0)])),
                         ("two",Road([(5,0),(10,0)]))])
        crossing=Motion("moving",r,0,10)
        parked=Motion("parked",route([(0,0),(10,0)],name="park"),5,5)
        hit=first_contact(crossing,[parked],clearance=0)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.fraction,0.05,delta=0.01)

    def test_clear_parallel_paths(self):
        a=route([(0,0),(10,0)],name="a")
        b=route([(0,3),(10,3)],name="b")
        self.assertIsNone(first_contact(Motion("a",a,0,10),[Motion("b",b,0,10)]))

    def test_backwards_motion_rejected(self):
        r=route([(0,0),(10,0)])
        with self.assertRaises(ValueError):
            Motion("reverse",r,5,4)


if __name__ == "__main__":
    unittest.main()
