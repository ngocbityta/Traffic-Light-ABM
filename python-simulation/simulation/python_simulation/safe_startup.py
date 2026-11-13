"""Deterministic safe-mode release timing for queued vehicles.

This standalone helper is intended for an optional Python simulation mode. It
does not alter GAMA parity behavior. Positions are metres along each road,
increasing toward the stop line; lane keys distinguish road and sublane.

Defaults are synthetic engineering values for exercising staggered release,
not calibrated measurements for the Le Van Luong intersection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Hashable, Iterable


@dataclass(frozen=True)
class StartupVehicle:
    vehicle_id: Hashable
    road_id: Hashable
    sublane: Hashable
    position: float
    speed: float
    stop_line_position: float
    committed: bool = False
    occupied: int = 1


class SafeStartupScheduler:
    """Schedule stopped queues after green with deterministic reaction delays.

    Call :meth:`update` once per simulation tick with a coherent snapshot of
    vehicles. On green onset, stopped vehicles are ordered front-to-back in
    each road and receive release times separated by ``discharge_headway``
    from any earlier vehicle with an overlapping occupied sublane footprint.
    Sublane identifiers must be integer indices. ``release_ids`` means
    permission to resume motion;
    normal car-following and stop-line checks must still constrain movement.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        reaction_delay: float = 0.6,
        discharge_headway: float = 0.8,
        stopped_speed: float = 0.1,
    ) -> None:
        if reaction_delay < 0 or discharge_headway < 0 or stopped_speed < 0:
            raise ValueError("delay, headway, and stopped_speed must be nonnegative")
        self.seed = int(seed)
        self.reaction_delay = float(reaction_delay)
        self.discharge_headway = float(discharge_headway)
        self.stopped_speed = float(stopped_speed)
        self._green = False
        self._green_since: float | None = None
        self._release_at: dict[Hashable, float] = {}

    def reset(self) -> None:
        """Clear all signal-cycle state; stable ID delays remain reproducible."""
        self._green = False
        self._green_since = None
        self._release_at.clear()

    def update(
        self,
        now: float,
        green: bool,
        vehicles: Iterable[StartupVehicle],
    ) -> frozenset[Hashable]:
        """Return IDs allowed to resume this tick; red always returns empty.

        A new green schedules the snapshot once. Vehicles already committed to
        the junction are exempt from startup delay and are returned immediately
        while green. A red tick clears the previous schedule so the next green
        gets a fresh queue ordering and start time.
        """
        now = float(now)
        if not green:
            self._green = False
            self._green_since = None
            self._release_at.clear()
            return frozenset()

        items = tuple(vehicles)
        if any(v.occupied < 1 for v in items):
            raise ValueError("occupied must be at least 1 sublane")
        if not self._green:
            self._green = True
            self._green_since = now
            self._release_at.clear()
            queues: dict[Hashable, list[StartupVehicle]] = {}
            for vehicle in items:
                if vehicle.committed or vehicle.speed > self.stopped_speed:
                    continue
                if vehicle.position > vehicle.stop_line_position:
                    continue
                queues.setdefault(vehicle.road_id, []).append(vehicle)
            for queue in queues.values():
                # Largest station is the first vehicle nearest the stop line.
                queue.sort(key=lambda v: (-v.position, str(v.vehicle_id)))
                scheduled: list[tuple[StartupVehicle, float, set[int]]] = []
                for vehicle in queue:
                    reaction = self._reaction(vehicle.vehicle_id)
                    release = now + reaction
                    footprint = set(range(int(vehicle.sublane),
                                          int(vehicle.sublane) + vehicle.occupied))
                    for predecessor, predecessor_release, other_footprint in scheduled:
                        if footprint & other_footprint:
                            release = max(release,
                                          predecessor_release + self.discharge_headway)
                    self._release_at[vehicle.vehicle_id] = release
                    scheduled.append((vehicle, release, footprint))

        return frozenset(
            v.vehicle_id
            for v in items
            if v.committed or v.vehicle_id not in self._release_at
            or self._release_at[v.vehicle_id] <= now
        )

    def release_time(self, vehicle_id: Hashable) -> float | None:
        """Scheduled release time, or ``None`` if this ID is not queued."""
        return self._release_at.get(vehicle_id)

    @staticmethod
    def constrain_position(
        proposed_position: float,
        stop_line_position: float,
        *,
        green: bool,
        committed: bool = False,
    ) -> float:
        """Clamp red-light motion at the line (red positions cannot exceed it)."""
        proposed_position = float(proposed_position)
        if green or committed:
            return proposed_position
        return min(proposed_position, float(stop_line_position))

    def _reaction(self, vehicle_id: Hashable) -> float:
        if self.reaction_delay == 0:
            return 0.0
        raw = f"{self.seed}\0{vehicle_id}".encode("utf-8")
        integer = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")
        # Map to [0, reaction_delay], with deterministic, platform-stable math.
        unit = integer / ((1 << 64) - 1)
        return unit * self.reaction_delay

