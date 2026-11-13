import unittest

from python_simulation.vehicle_geometry import Pose, Road, RouteGeometry, SUBLANE_WIDTH, footprint, polygons_overlap, vehicles_overlap, swept_overlap


def pose(x, y, heading=0):
    import math
    return Pose(x, y, math.cos(heading), math.sin(heading), heading, 0)


class VehicleGeometryTests(unittest.TestCase):
    def test_distinct_centers_but_rectangles_overlap(self):
        self.assertGreater(2.0, 0)
        self.assertTrue(vehicles_overlap(pose(0, 0), pose(3.0, 0)))

    def test_curved_connector_arc_length_and_lane_offset(self):
        road=Road([(0,0),(5,0),(5,5)], num_lanes=2)
        p=road.pose(7, lateral_shift=0)
        expected_lat = 2 * SUBLANE_WIDTH - SUBLANE_WIDTH / 2.0
        self.assertAlmostEqual(p.x, 5.0 + expected_lat)  # right normal of northbound final segment
        self.assertAlmostEqual(p.y, 2)
        self.assertAlmostEqual(p.heading, 1.57079632679)
        self.assertAlmostEqual(road.pose(-1).x, -1)
        self.assertAlmostEqual(road.pose(-1).y, -expected_lat)
        self.assertAlmostEqual(road.pose(11).y, 6)

    def test_touching_and_near_clearance(self):
        a=footprint(pose(0,0), length=2, width=2)
        b=footprint(pose(0,2), length=2, width=2)
        self.assertTrue(polygons_overlap(a,b))
        c=footprint(pose(0,2.05), length=2, width=2)
        self.assertFalse(polygons_overlap(a,c))
        self.assertTrue(polygons_overlap(a,c,clearance=0.03))

    def test_swept_crossing_detected_despite_clear_endpoints(self):
        horizontal=Road([(-5,0),(5,0)])
        vertical=Road([(0,-5),(0,5)])
        # The two vehicles exchange sides during the tick; neither endpoint overlaps.
        self.assertFalse(vehicles_overlap(horizontal.pose(1), vertical.pose(9)))
        self.assertFalse(vehicles_overlap(horizontal.pose(9), vertical.pose(1)))
        self.assertTrue(swept_overlap(horizontal,1,9,vertical,9,1,max_step=.1))

    def test_lateral_lane_change_sweeps_into_occupied_lane(self):
        road=Road([(0,0),(20,0)], num_lanes=3)
        # Bike A moves across the stationary bike's adjacent lane at the same progress.
        self.assertFalse(vehicles_overlap(road.pose(10,0), road.pose(10,SUBLANE_WIDTH),"motorbike","motorbike"))
        self.assertFalse(vehicles_overlap(road.pose(10,2*SUBLANE_WIDTH), road.pose(10,SUBLANE_WIDTH),"motorbike","motorbike"))
        self.assertTrue(swept_overlap(road,10,10,road,10,10,"motorbike","motorbike",
                                     shift_a=0, shift_b=SUBLANE_WIDTH, end_shift_a=2*SUBLANE_WIDTH))

    def test_route_seam_continuity_and_reported_lane_offset_jump(self):
        a=Road([(0,0),(5,0)],1); b=Road([(5,0),(10,0)],1)
        route=RouteGeometry([("approach",a),("exit",b)])
        mid=SUBLANE_WIDTH/2.0
        left=route.pose(route.seams[0]-1e-6,mid)[1]
        right=route.pose(route.seams[0],mid)[1]
        self.assertAlmostEqual(left.x,right.x,places=5)
        self.assertAlmostEqual(left.y,right.y,places=5)
        # Raw shared endpoint but different lane counts yield a real lateral jump.
        changed=RouteGeometry([("wide",Road([(0,0),(5,0)],3)),("narrow",Road([(5,0),(10,0)],1))])
        dx,dy=changed.seam_discontinuities(mid)[0]
        self.assertAlmostEqual(dx,0)
        self.assertAlmostEqual(dy,2*SUBLANE_WIDTH)

    def test_footprint_overhangs_across_route_seam(self):
        route=RouteGeometry([("a",Road([(0,0),(5,0)],1)),("b",Road([(5,0),(10,0)],1))])
        p=route.pose(route.seams[0]-.1,.4)[1]
        self.assertGreater(max(x for x,y in footprint(p)),5.0)

    def test_route_sweep_crosses_seam_with_clear_endpoints(self):
        route=RouteGeometry([("in",Road([(0,0),(5,0)],1)),("turn",Road([(5,0),(5,5)],1))])
        obstacle=Road([(5,-5),(5,5)],1)
        start=route.pose(0,.4)[1]; end=route.pose(10,.4)[1]; parked=obstacle.pose(5,.4)
        self.assertFalse(vehicles_overlap(start,parked))
        self.assertFalse(vehicles_overlap(end,parked))
        self.assertTrue(swept_overlap(route,0,10,obstacle,5,5,shift_a=.4,shift_b=.4))


if __name__ == "__main__": unittest.main()
