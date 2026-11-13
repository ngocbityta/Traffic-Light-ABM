"""Small reference for GAMA 2025.06.4's road-node transition behavior.

This isolates the temporal and lane state at a node. It deliberately does not
model connector geometry, neighbor queries, or vehicle conflicts.

Source: ``gama/gama.extension.traffic/src/gama/extension/traffic/driving/DrivingSkill.java``
from tag ``2025.06.4`` (commit ``7cde6fac892138a161d75c7d02d112a981cbfcb6``):

* ``readyToCross`` checks node stop lists and rejects crossing at lines 1311-1342.
* ``moveAcrossRoads`` sets speed to zero on rejection (1793-1813), computes a
  new-road speed for the zero-speed gate (1825-1839), registers the new road,
  and calls segment movement with remaining time (1876-1886).
* ``moveAcrossSegments`` computes distance using its ``time`` argument, but
  ``computeSpeed`` uses the full clock step; reaching a node returns leftover
  time without committing that speed (1924-1985, especially 1940-1964).
* ``computeSpeed`` always uses the full simulation step (1998-2003).
* Default lane selection retains/clamps the current index (1619-1633).
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NodeTransition:
    """Result for a vehicle that reaches a node during one simulation step."""

    crossed: bool
    remaining_time: float
    new_road_distance: float
    stored_speed: float
    new_lane: int


def transition_at_node(
    *,
    speed: float,
    old_road_acceleration: float,
    distance_to_node: float,
    remaining_time: float,
    dt: float,
    new_road_acceleration: float,
    current_lane: int,
    num_lanes_occupied: int,
    new_road_num_lanes: int,
    red_blocked: bool = False,
    linked_lane_limit: int = 0,
) -> NodeTransition:
    """Apply the GAMA 2025.06.4 transition at one road node.

    Callers must invoke this only after the vehicle has reached the current
    road's target node. ``distance_to_node`` is the distance used by GAMA to
    estimate how much time remains after reaching that node. The old-road
    predicted speed is used for that estimate exactly as ``moveAcrossSegments``
    does. At node arrival, GAMA has not stored that predicted speed: the next
    road therefore starts from ``speed``. Its distance uses leftover time,
    while its stored speed uses the full ``dt``. This models a new-road segment that does
    not itself reach another node; on a second node arrival GAMA again returns
    without committing the calculated speed.

    A red rejection corresponds to ``readyToCross == false``: GAMA sets speed
    to zero and returns before choosing a lane or moving onto the new road. A
    non-red transition whose new-road predicted speed is zero hits the separate
    ``newSpeed == 0`` gate and also returns before entering that road.
    """
    values = (speed, old_road_acceleration, distance_to_node, remaining_time, dt,
              new_road_acceleration)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("transition values must be finite")
    if remaining_time < 0 or dt < 0 or distance_to_node < 0:
        raise ValueError("times and distance_to_node must be nonnegative")
    if num_lanes_occupied <= 0 or new_road_num_lanes <= 0:
        raise ValueError("lane counts must be positive")

    if red_blocked:
        return NodeTransition(False, remaining_time, 0.0, 0.0, current_lane)

    new_road_speed = max(0.0, speed + new_road_acceleration * dt)
    if new_road_speed == 0.0:
        return NodeTransition(False, remaining_time, 0.0, 0.0, current_lane)

    old_road_speed = max(0.0, speed + old_road_acceleration * dt)
    if old_road_speed == 0.0:
        raise ValueError("cannot estimate node arrival time with zero predicted speed")
    time_after_node = max(0.0, remaining_time - distance_to_node / old_road_speed)

    # At the node, moveAcrossSegments returns before setSpeed(newSpeed), so
    # new-road distance starts from the pre-step stored speed.
    new_road_distance = max(
        0.0,
        speed * time_after_node
        + 0.5 * new_road_acceleration * time_after_node * time_after_node,
    )
    stored_speed = new_road_speed
    if num_lanes_occupied > new_road_num_lanes + linked_lane_limit:
        raise ValueError("vehicle occupies more lanes than available on new road")
    highest_lane = new_road_num_lanes + linked_lane_limit - num_lanes_occupied
    new_lane = min(current_lane, highest_lane)
    return NodeTransition(True, time_after_node, new_road_distance, stored_speed, new_lane)
