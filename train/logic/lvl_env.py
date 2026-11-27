"""Common Le Van Luong observations for Python and the canonical GAMA model."""

import asyncio
import re
import sys
from pathlib import Path

import numpy as np

from .gama_env import GamaEnvError, GamaTrafficEnv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python-simulation" / "simulation"))
from python_simulation.python_sim import ARMS, DT, LeVanLuongSimulator, road_lengths
from python_simulation.sim_config import load_sim_config

MODEL = ROOT / "gama-simulation/simulation/models/LeVanLuong.gaml"
ROADS = road_lengths()
VEHICLES_EXPR = ('"" + (vehicle collect ("" + each.speed + ":" + '
                 'each.max_speed + ":" + each.lowest_lane + ":" + '
                 'each.num_lanes_occupied + ":" + '
                 '(each.current_road = nil ? "nil" : road(each.current_road).rid) + ":" + '
                 'each.distance_to_current_target))')
VEHICLE_PATTERN = re.compile(
    r'([\d.Ee+-]+):([\d.Ee+-]+):(-?\d+):(\d+):([A-Za-z0-9_]+):([\d.Ee+-]+)')
STATE_EXPR = ('"" + cycle + "|" + phase + "|" + dqn_waiting_action + "|" + '
              'next_decision_in_cycles + "|" + total_vehicles_created + "|" + '
              'total_vehicles_passed + "|" + queue_time_integral + "|" + '
              'system_time_integral + "|" + weighted_delay_integral + "|" + '
              'green_ticks')
MASK = np.asarray([True, False, True, False])
# A phase must have shown green this long before the policy may leave it
# (10 s = 50 cycles of 0.2 s). The minimum applies to actual green time.
MIN_GREEN_CYCLES = 50


def action_mask(phase, green_cycles, min_green_cycles=MIN_GREEN_CYCLES):
    """Legal actions (LVL, -, HDT, -): staying is always legal, leaving the
    running phase only after its minimum green."""
    if phase not in ("LVL", "HDT"):
        return MASK.copy()
    may_switch = green_cycles is None or int(green_cycles) >= int(min_green_cycles)
    return np.asarray([phase == "LVL" or may_switch, False,
                       phase == "HDT" or may_switch, False])


# ---------------------------------------------------------------------------
# State v5 (27/11/2026): what a camera covering 50 m before each stop line
# sees, plus the signal controller's own clock. Built only by the Python
# backend (``state_version="v5"``); the simulator keeps the zone counters
# (``observation_zone=50``).
#
# grid[arm, k, row, col]: arms in ARMS order (NE, SE, SW, NW), row j covers
# [5j, 5j + 5) m before the stop line (reference point of the vehicle),
# col = sublane 0..11 counted from the kerb. Channels:
#   0 any vehicle, 1 motorbike, 2 car, 3 heavy vehicle, 4 stopped vehicle
#   (< 1 km/h), 5 highest speed in the cell / 15 m/s, 6 sublane exists.
# aux (30):
#   0      axis with green (LVL 0, HDT 1)
#   1      green shown so far / max_green
#   2      last green of the waiting axis / max_green
#   3      room left to extend: (max_green - planned green) / (max_green - min_green)
#   4-7    stopped vehicles in the zone, log1p(n) / log1p(200)
#   8-11   moving vehicles that reach the line within 5 s, log1p(n) / log1p(100)
#   12-15  occupied share of the 40-50 m band (queue reaching the edge of view)
#   16-19  arrivals into the zone, EWMA over 60 s, veh/s / 1.5
#   20-23  longest stopped time of a vehicle in the zone / 120 s
#   24-27  vehicles on the first 30 m of each exit road, log1p(n) / log1p(100)
#   28-29  stop-line crossings since the last decision, LVL / HDT, veh/s / 3
# Every value is clipped to [0, 1].
# ---------------------------------------------------------------------------
V5_GRID_SHAPE = (4, 7, 10, 12)
V5_AUX_SIZE = 30
V5_ZONE = 50.0
V5_CELL = 5.0
V5_EFFECTIVE_SECONDS = 5.0
V5_SPEED_REF = 15.0
V5_ARRIVAL_TAU = 60.0
V5_EXIT_ZONE = 30.0
STATE_VERSIONS = ("v4", "v5")
_LOG200 = float(np.log1p(200.0))
_LOG100 = float(np.log1p(100.0))
_KIND_CHANNEL = {"motorbike": 1, "car": 2, "heavy": 3}


class ZoneTracker:
    """Rates between two decision snapshots of the simulator's zone counters."""

    def __init__(self):
        self.cycle = None
        self.arrivals = None
        self.crossings = None
        self.arrival_rate = np.zeros(4, dtype=np.float64)
        self.discharge = np.zeros(2, dtype=np.float64)

    def update(self, sim):
        arrivals = np.asarray([sim.zone_arrivals[arm] for arm in ARMS], dtype=np.float64)
        crossings = np.asarray([sim.stopline_crossings[arm] for arm in ARMS], dtype=np.float64)
        if self.cycle is not None and sim.cycle > self.cycle:
            dt = (sim.cycle - self.cycle) * DT
            alpha = 1.0 - float(np.exp(-dt / V5_ARRIVAL_TAU))
            self.arrival_rate += alpha * ((arrivals - self.arrivals) / dt - self.arrival_rate)
            crossed = (crossings - self.crossings) / dt
            self.discharge = np.asarray([crossed[0] + crossed[2], crossed[1] + crossed[3]])
        elif self.cycle is None or sim.cycle < self.cycle:
            self.arrival_rate[:] = 0.0
            self.discharge[:] = 0.0
        self.cycle, self.arrivals, self.crossings = sim.cycle, arrivals, crossings


def observation_v5(sim, tracker):
    """(grid, aux) of state v5 for the simulator at a decision tick."""
    if abs(sim.observation_zone - V5_ZONE) > 1e-9:
        raise ValueError("state v5 needs the simulator's observation_zone = 50 m")
    grid = np.zeros(V5_GRID_SHAPE, dtype=np.float32)
    lanes = []
    for a, arm in enumerate(ARMS):
        n = min(V5_GRID_SHAPE[3], int(sim.roads[f"{arm}_in"]["num_lanes"]))
        grid[a, 6, :, :n] = 1.0
        lanes.append(n)
    stopped = np.zeros(4)
    effective = np.zeros(4)
    longest = np.zeros(4)
    exit_count = np.zeros(4)
    arm_index = {arm: a for a, arm in enumerate(ARMS)}
    stop_time = sim.zone_stop_time
    for v in sim.vehicles:
        if v.road == v.in_road:
            d = v.approach_end - v.position
            if d >= V5_ZONE:
                continue
            a = arm_index[v.arm]
            row = min(9, max(0, int(max(0.0, d) // V5_CELL)))
            lo, hi = max(0, v.lane), min(lanes[a], v.lane + v.occupied)
            speed = max(0.0, v.speed)
            is_stopped = speed * 3.6 < 1.0
            for col in range(lo, hi):
                cell = grid[a, :, row, col]
                cell[0] = 1.0
                cell[_KIND_CHANNEL[v.kind]] = 1.0
                if is_stopped:
                    cell[4] = 1.0
                cell[5] = max(cell[5], min(1.0, speed / V5_SPEED_REF))
            if is_stopped:
                stopped[a] += 1
                longest[a] = max(longest[a], stop_time.get(v.creation_id, 0.0))
            elif max(0.0, d) <= speed * V5_EFFECTIVE_SECONDS:
                effective[a] += 1
        elif v.road == v.out_road:
            _, progress = sim._road_state(v)
            if progress <= V5_EXIT_ZONE:
                exit_count[arm_index[v.to_arm]] += 1
    tracker.update(sim)
    aux = np.zeros(V5_AUX_SIZE, dtype=np.float32)
    phase = sim.phase if sim.phase in ("LVL", "HDT") else (
        sim.pending_phase if getattr(sim, "_phase_change_pending", False) else "LVL")
    other = "HDT" if phase == "LVL" else "LVL"
    max_green = max(float(getattr(sim, "max_green", 75.0)), 1e-6)
    min_green = float(sim.min_green)
    aux[0] = float(phase == "HDT")
    aux[1] = sim.green_ticks * DT / max_green
    aux[2] = sim.last_green_seconds.get(other, 0.0) / max_green
    if getattr(sim, "signal_control", "phase") == "extend":
        planned = sim.planned_green_ticks * DT
    else:
        planned = sim.green_ticks * DT
    aux[3] = (max_green - planned) / max(max_green - min_green, 1e-6)
    aux[4:8] = np.log1p(stopped) / _LOG200
    aux[8:12] = np.log1p(effective) / _LOG100
    edge = grid[:, 0, 8:10, :]
    aux[12:16] = [edge[a, :, :lanes[a]].mean() if lanes[a] else 0.0 for a in range(4)]
    aux[16:20] = tracker.arrival_rate / 1.5
    aux[20:24] = longest / 120.0
    aux[24:28] = np.log1p(exit_count) / _LOG100
    aux[28:30] = tracker.discharge / 3.0
    np.clip(aux, 0.0, 1.0, out=aux)
    return grid, aux


def _observation(cycle, phase, waiting, next_decision, created, passed,
                 queue_integral, system_integral, weighted_integral, vehicles,
                 state_mode, green_cycles=None, mask=None, transition_of=None,
                 yellow_cycles=0, extra=None):
    grid = np.zeros((4, 6, 20, 16), dtype=np.float32)
    for arm_idx, arm_name in enumerate(ARMS):
        valid_lanes = ROADS[f"{arm_name}_in"]["num_lanes"]
        grid[arm_idx, 5, :, :valid_lanes] = 1.0
    speed_counts = np.zeros((4, 20, 16), dtype=np.float32)
    dtse = np.zeros((3, 8, 20), dtype=np.float32)
    queues = [0, 0, 0, 0]
    approach_counts = [0, 0, 0, 0]
    speed_total = delay_sq = 0.0
    queue_now = 0
    for speed, desired, lane, occupied, road, remaining in vehicles:
        ratio = min(1.0, max(0.0, speed / max(desired, 1e-6)))
        delay_sq += (1 - ratio) ** 2
        speed_total += speed
        queue_now += speed * 3.6 < 1.0
        if not road.endswith("_in"):
            continue
        arm = ARMS.index(road[:2])
        approach_counts[arm] += 1
        if speed * 3.6 < 1.0:
            queues[arm] += 1
        bin_id = min(19, max(0, int(max(0.0, remaining) / 6.0)))
        served = (arm in (0, 2) and phase == "LVL") or (
            arm in (1, 3) and phase == "HDT")
        for sublane in range(max(0, lane), min(16, lane + occupied)):
            grid[arm, 0, bin_id, sublane] = 1.0
            grid[arm, 1, bin_id, sublane] += ratio
            speed_counts[arm, bin_id, sublane] += 1.0
            grid[arm, 2, bin_id, sublane] = float(not served)
            # 1 for cars and heavy vehicles (larger than a motorbike).
            grid[arm, 3, bin_id, sublane] = float(occupied >= 2)
            grid[arm, 4, bin_id, sublane] = (1.0 if arm in (0, 2) else -1.0)
        row = 2 * arm + int(lane >= 8)
        dtse[0, row, bin_id] = 1.0
        dtse[1, row, bin_id] = max(dtse[1, row, bin_id], ratio)
        dtse[2, row, bin_id] = float(served)
    np.divide(grid[:, 1], speed_counts, out=grid[:, 1], where=speed_counts > 0)
    # ``global_phase`` follows the four-phase convention read by featurize():
    # even = green of phase n // 2, odd = its yellow/clearance. During a
    # transition the code is that of the phase being left when it is known
    # (Python backend); it used to be 1 always, i.e. "LVL" for both.
    phase_code = {"LVL": 0, "HDT": 4}
    if phase in phase_code:
        global_phase = phase_code[phase]
    else:
        global_phase = phase_code.get(transition_of, 0) + 1
    state = dict(cycle=cycle, global_phase=global_phase,
                 green_cycles=cycle % 50 if green_cycles is None else int(green_cycles),
                 yellow_cycles=int(yellow_cycles),
                 queue_ns=queues[0] + queues[2], queue_ew=queues[1] + queues[3],
                 approach_ns=approach_counts[0] + approach_counts[2],
                 approach_ew=approach_counts[1] + approach_counts[3],
                 current_queue_length=queue_now,
                 # These are cumulative-integral ratios, not per-vehicle
                 # completed-trip averages. Keep legacy keys for agent/log
                 # compatibility and expose explicit names for reporting.
                 average_waiting_time=queue_integral / max(1, created) if created else 0.0,
                 max_waiting_time=float("nan"),
                 average_speed=3.6 * speed_total / len(vehicles) if vehicles else 0.0,
                 average_travel_time=system_integral / max(1, passed) if passed else 0.0,
                 cumulative_queue_time_per_created_vehicle=(
                     queue_integral / max(1, created) if created else 0.0),
                 cumulative_system_time_per_passed_vehicle=(
                     system_integral / max(1, passed) if passed else 0.0),
                 time_metric_semantics="cumulative_integral_ratio; not completed-vehicle mean",
                 max_waiting_time_semantics="unavailable (NaN)",
                 total_vehicles_passed=passed,
                 total_vehicles_created=created, decision_count=cycle // 50,
                 dqn_waiting_action=waiting, next_decision_in_cycles=next_decision,
                 delay_sq=delay_sq, queue_time_integral=queue_integral,
                 external_wait_integral=0.0,
                 action_mask=(action_mask(phase, green_cycles) if mask is None
                              else np.asarray(mask, dtype=bool)),
                 system_time_integral=system_integral,
                 stopped_time_integral=queue_integral,
                 external_all_time_integral=0.0, outgoing_block_integral=0.0,
                 weighted_delay_integral=weighted_integral,
                 pressure=np.asarray([queues[0] + queues[2], 0,
                                      queues[1] + queues[3], 0], dtype=np.float32),
                 step_seconds=DT)
    state["spatial" if state_mode == "spatial" else "dtse"] = (
        grid if state_mode == "spatial" else dtse)
    if extra:
        state.update(extra)
    return state


class _LVLCommon:
    step_seconds = DT


class LVLGamaEnv(_LVLCommon, GamaTrafficEnv):
    def __init__(self, **kwargs):
        kwargs["model_path"] = kwargs.get("model_path") or str(MODEL)
        kwargs["experiment"] = "dqn_server_experiment"
        super().__init__(**kwargs)

    async def reset(self, seed=None, parameters=None):
        params = list(parameters or [])
        params.append({"name": "train_mode", "type": "bool", "value": "true"})
        if seed is not None:
            params.append({"name": "traffic_seed", "type": "float",
                           "value": str(float(int(seed)))})
        if self.exp_id is None:
            await self.load_experiment(parameters=params)
        else:
            await self.pause_until_stable()
            if self.reset_mode == "fresh":
                await self.stop_experiment()
                await self.load_experiment(parameters=params)
            else:
                response = await self.client.reload_awaitable(
                    self.exp_id, parameters=params, timeout=self.timeout)
                if response.get("type") != "CommandExecutedSuccessfully":
                    raise GamaEnvError(f"Le Van Luong reload failed: {response}")
                await self.pause_until_stable()
        state = await self.read_state()
        if state["cycle"] != 0:
            raise GamaEnvError(f"Reset returned cycle {state['cycle']}")
        return state

    async def read_state(self, force_dtse=False):
        fields = str(await self._exec_expression(STATE_EXPR)).split("|")
        if len(fields) != 10:
            raise GamaEnvError(f"Invalid Le Van Luong state: {fields}")
        raw = str(await self._exec_expression(VEHICLES_EXPR))
        vehicles = [(float(a), float(b), int(c), int(d), e, float(f))
                    for a, b, c, d, e, f in VEHICLE_PATTERN.findall(raw)]
        if len(vehicles) != int(float(fields[4])) - int(float(fields[5])):
            raise GamaEnvError("Le Van Luong vehicle snapshot is incomplete")
        return _observation(int(float(fields[0])), fields[1], fields[2].lower() == "true",
                            int(float(fields[3])), int(float(fields[4])),
                            int(float(fields[5])), *map(float, fields[6:9]),
                            vehicles, self.state_mode,
                            green_cycles=int(float(fields[9])))


class LVLPythonEnv(_LVLCommon, GamaTrafficEnv):
    PYTHON_SIM_MODES = ("safe", "gama_2025_6_4")

    def __init__(self, python_sim_mode="safe", sim_config=None, **kwargs):
        self._zone_tracker = ZoneTracker()
        if python_sim_mode not in self.PYTHON_SIM_MODES:
            raise ValueError(
                f"Unknown Python simulator mode: {python_sim_mode!r}; "
                f"expected one of {self.PYTHON_SIM_MODES}")
        kwargs["model_path"] = kwargs.get("model_path") or str(MODEL)
        super().__init__(**kwargs)
        self.python_sim_mode = python_sim_mode
        self.sim_config = load_sim_config(config=sim_config)
        self.sim = None

    async def setup(self):
        pass

    async def close(self, stop_experiment=True):
        self.sim = None

    async def reset(self, seed=None, parameters=None):
        demand = next((row["value"] for row in parameters or []
                       if row["name"] == "demand_file_path"), None)
        if not demand:
            raise ValueError("Le Van Luong training requires a paired demand CSV")
        config = dict(self.sim_config)
        if getattr(self, "state_version", "v4") == "v5":
            config.setdefault("observation_zone", V5_ZONE)
        self.sim = LeVanLuongSimulator(seed=1 if seed is None else seed, demand_file=demand,
                                      train_mode=True, mode=self.python_sim_mode,
                                      **config)
        self._zone_tracker = ZoneTracker()
        return await self.read_state()

    async def read_state(self, force_dtse=False):
        sim = self.sim
        vehicles = []
        for vehicle in sim.vehicles:
            if sim.mode == "gama_2025_6_4" and not vehicle.gama_entered:
                road, remaining = "nil", 0.0
            else:
                road, progress = sim._road_state(vehicle)
                remaining = max(0.0, sim.roads[road]["length"] - progress)
            vehicles.append((vehicle.speed, vehicle.desired, vehicle.lane,
                             vehicle.occupied, road, remaining))
        yellow_total = int(round(sim._yellow / DT))
        return _observation(sim.cycle, sim.phase, sim.dqn_waiting_action,
                            sim.next_decision_in_cycles, sim.created, sim.passed,
                            sim.queue_time_integral, sim.system_time_integral,
                            sim.weighted_delay_integral, vehicles, self.state_mode,
                            green_cycles=getattr(sim, "green_ticks", None),
                            mask=sim.action_mask(),
                            transition_of=(sim.yellow_phase or sim._last_yellow),
                            yellow_cycles=(yellow_total - sim.yellow_ticks_left
                                           if sim.yellow_phase else 0),
                            extra=self._extra(sim))

    def _extra(self, sim):
        extra = {"jam_removed": sim.jam_removed,
                 "hard_brake_events": sim.hard_brake_events}
        if getattr(self, "state_version", "v4") == "v5":
            if not hasattr(self, "_zone_tracker"):
                self._zone_tracker = ZoneTracker()
            grid, aux = observation_v5(sim, self._zone_tracker)
            extra.update(spatial=grid, aux_v5=aux)
        return extra

    async def set_action(self, action):
        self.sim.set_action(action)

    async def step_cycles(self, n):
        self.sim.step(int(n))

    async def get_next_decision_in(self):
        return self.sim.next_decision_in_cycles
