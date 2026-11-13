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


def _observation(cycle, phase, waiting, next_decision, created, passed,
                 queue_integral, system_integral, weighted_integral, vehicles,
                 state_mode, green_cycles=None):
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
            grid[arm, 3, bin_id, sublane] = float(occupied == 2)
            grid[arm, 4, bin_id, sublane] = (1.0 if arm in (0, 2) else -1.0)
        row = 2 * arm + int(lane >= 8)
        dtse[0, row, bin_id] = 1.0
        dtse[1, row, bin_id] = max(dtse[1, row, bin_id], ratio)
        dtse[2, row, bin_id] = float(served)
    np.divide(grid[:, 1], speed_counts, out=grid[:, 1], where=speed_counts > 0)
    state = dict(cycle=cycle, global_phase={"LVL": 0, "HDT": 4, "none": 1}[phase],
                 green_cycles=cycle % 50 if green_cycles is None else int(green_cycles),
                 yellow_cycles=0,
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
                 external_wait_integral=0.0, action_mask=MASK.copy(),
                 system_time_integral=system_integral,
                 stopped_time_integral=queue_integral,
                 external_all_time_integral=0.0, outgoing_block_integral=0.0,
                 weighted_delay_integral=weighted_integral,
                 pressure=np.asarray([queues[0] + queues[2], 0,
                                      queues[1] + queues[3], 0], dtype=np.float32),
                 step_seconds=DT)
    state["spatial" if state_mode == "spatial" else "dtse"] = (
        grid if state_mode == "spatial" else dtse)
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
        self.sim = LeVanLuongSimulator(seed=1 if seed is None else seed, demand_file=demand,
                                      train_mode=True, mode=self.python_sim_mode,
                                      **self.sim_config)
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
        return _observation(sim.cycle, sim.phase, sim.dqn_waiting_action,
                            sim.next_decision_in_cycles, sim.created, sim.passed,
                            sim.queue_time_integral, sim.system_time_integral,
                            sim.weighted_delay_integral, vehicles, self.state_mode,
                            green_cycles=getattr(sim, "green_ticks", None))

    async def set_action(self, action):
        self.sim.set_action(action)

    async def step_cycles(self, n):
        self.sim.step(int(n))

    async def get_next_decision_in(self):
        return self.sim.next_decision_in_cycles
