#!/usr/bin/env python3
"""
==============================================================================
GAMA TRAFFIC ENV - WebSocket bridge for four-phase DTSE and spatial agents
==============================================================================
Giao tiep voi gama-server qua thu vien chinh thuc `gama-client` (WebSocket).

CHU Y: gama-client 2.x yeu cau event loop dang chay khi tao client va khi goi
lenh -> moi method o day la ASYNC, dung ben trong `asyncio.run(main())`:

    Python                                    GAMA (intersection-dqn-python.gaml)
    ------                                    -----------------------------------
    await load_awaitable(...)     ->  exp_id
    await expression_awaitable(GET state) -> metrics (18 fields)
    await ask_awaitable(capture_dtse) -> 3x8x20 DTSE and squared delay
    await expression_awaitable("dqn_action <- a")
    await step_awaitable(exp_id, 1, sync=True)  ->  controller ap dung action
    await expression_awaitable(GET next_decision_in_cycles)
    await step_awaitable(exp_id, nb, sync=True) ->  chay dung den tick ke tiep

Protocol chi tiet xem header cua train/benchmarks/gama/models/intersection-dqn-python.gaml
==============================================================================
"""

import asyncio
import json
import os
from pathlib import Path
import re

import numpy as np

from gama_client.sync_client import GamaSyncClient

# Duong dan mac dinh
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = str(REPO_DIR / "train" / "benchmarks" / "gama" / "models" / "intersection-dqn-python.gaml")
# doan: mang GAMA cu (Nguyen Trai - Khuat Duy Tien); lvl: nut Le Van Luong x Hoang Dao Thuy (chi state spatial)
SITE_MODELS = {
    "doan": DEFAULT_MODEL,
    "lvl": str(REPO_DIR / "gama-simulation" / "simulation" / "models" / "LeVanLuong.gaml"),
}

# Bieu thuc GAML: dong goi 18 bien thanh 1 chuoi "|" de 1 request lay het state
STATE_PACK_EXPR = (
    '"" + cycle + "|" + global_phase + "|" + green_cycles + "|" + yellow_cycles'
    + ' + "|" + queue_ns + "|" + queue_ew + "|" + approach_ns + "|" + approach_ew'
    + ' + "|" + current_queue_length'
    + ' + "|" + (round(average_waiting_time*100.0)/100.0)'
    + ' + "|" + (round(max_waiting_time*100.0)/100.0)'
    + ' + "|" + (round(average_speed*100.0)/100.0)'
    + ' + "|" + (round(average_travel_time*100.0)/100.0)'
    + ' + "|" + total_vehicles_passed + "|" + total_vehicles_created'
    + ' + "|" + decision_count + "|" + dqn_waiting_action'
    + ' + "|" + next_decision_in_cycles'
)

# One expression after capture_dtse replaces four WebSocket round trips at
# every decision. The DTSE list never contains the delimiter.
DTSE_PACK_EXPR = ('"" + dtse_pack + "|" + dtse_delay_sq + "|" + '
                  'dtse_delay_sq_near + "|" + queue_time_integral + "|" + '
                  'external_wait_integral + "|" + dqn_action_mask')
SPATIAL_PACK_EXPR = ('"" + spatial_pack + "|" + system_time_integral + "|" + '
                     'stopped_time_integral + "|" + external_all_time_integral + "|" + '
                     'outgoing_block_integral + "|" + weighted_delay_integral + "|" + '
                     'spatial_pressure + "|" + dqn_action_mask')

STATE_FIELDS = [
    "cycle", "global_phase", "green_cycles", "yellow_cycles",
    "queue_ns", "queue_ew", "approach_ns", "approach_ew",
    "current_queue_length", "average_waiting_time", "max_waiting_time",
    "average_speed", "average_travel_time",
    "total_vehicles_passed", "total_vehicles_created",
    "decision_count", "dqn_waiting_action", "next_decision_in_cycles",
]

INT_FIELDS = {
    "cycle", "global_phase", "green_cycles", "yellow_cycles",
    "queue_ns", "queue_ew", "approach_ns", "approach_ew",
    "current_queue_length", "total_vehicles_passed",
    "total_vehicles_created", "decision_count", "next_decision_in_cycles",
}

# DTSE shape and normalization are defined by capture_dtse in the GAML model.
class GamaEnvError(Exception):
    pass


class GamaTrafficEnv:
    """
    Moi truong giao thong GAMA (async, giao dien gym-ish).

    Cach dung (ben trong asyncio.run):
        env = GamaTrafficEnv(port=6868)
        await env.setup()          # ket noi; reset se load experiment
        state = await env.reset(seed=42)
        next_state, reward, done, info = await env.step(action)
        await env.close()
    """

    step_seconds = 0.1

    def __init__(self, host="localhost", port=6868,
                 model_path=DEFAULT_MODEL, experiment="dqn_server_experiment",
                 verbose=True, timeout=600.0, reward_scope="all",
                 reward_mode="reference", max_delay_sq=None,
                 state_mode="dtse", r1_ref=50.0, r2_ref=50.0, b_ref=50.0,
                 r2_lambda=1.0, r3_beta=1.0, reset_mode="reload",
                 fast_training=False):
        if reward_scope not in ("all", "near"):
            raise ValueError(f"Unknown reward scope: {reward_scope}")
        if reward_mode not in ("reference", "queue_time", "total_wait", "r1", "r2", "r3", "r4"):
            raise ValueError(f"Unknown reward mode: {reward_mode}")
        if state_mode not in ("dtse", "spatial"):
            raise ValueError(f"Unknown state mode: {state_mode}")
        if reset_mode not in ("reload", "fresh"):
            raise ValueError(f"Unknown reset mode: {reset_mode}")
        if (state_mode == "spatial") != (reward_mode in ("r1", "r2", "r3", "r4")):
            raise ValueError("Spatial state requires R1-R4; DTSE requires a legacy reward")
        if (not all(np.isfinite(x) for x in (r1_ref, r2_ref, b_ref, r2_lambda, r3_beta)) or
                min(r1_ref, r2_ref, b_ref) <= 0 or min(r2_lambda, r3_beta) < 0):
            raise ValueError("Invalid R1-R4 normalization or coefficient")
        self.max_delay_sq = float(max_delay_sq) if max_delay_sq is not None else None
        self.host = host
        self.port = int(port)
        self.model_path = model_path or DEFAULT_MODEL
        self.experiment = experiment
        self.verbose = verbose
        self.timeout = float(timeout)
        self.reward_scope = reward_scope
        self.reward_mode = reward_mode
        self.state_mode = state_mode
        self.reset_mode = reset_mode
        self.fast_training = bool(fast_training)
        self.r1_ref = float(r1_ref)
        self.r2_ref = float(r2_ref)
        self.b_ref = float(b_ref)
        self.r2_lambda = float(r2_lambda)
        self.r3_beta = float(r3_beta)
        self.action_count = 4
        self.exp_id = None
        self.client = None
        self._server_messages = []
        self._worst_delay_sq = 0.0

    # --------------------------------------------------------- server events
    async def _on_server_message(self, message):
        """Nhan cac message khong phai answer lenh (console, status, error...)."""
        self._server_messages.append(message)
        mtype = message.get("type", "") if isinstance(message, dict) else ""
        if mtype in ("SimulationError", "RuntimeError", "GamaServerError",
                     "SimulationErrorDialog"):
            print(f"[GamaEnv][SERVER-{mtype}] {message}")

    # ------------------------------------------------------------ lifecycle
    async def setup(self):
        """Connect to gama-server; first reset loads the seeded experiment."""
        self.client = GamaSyncClient(
            self.host, self.port,
            other_message_handler=self._on_server_message,
            default_timeout=self.timeout,
        )
        await self.client.connect_awaitable()
        if self.verbose:
            print(f"[GamaEnv] Da ket noi ws://{self.host}:{self.port}")
        # The first reset will load the model with its episode seed. Loading
        # here and immediately reloading it races GAMA's simulation startup.

    async def load_experiment(self, parameters=None):
        resp = await self.client.load_awaitable(
            self.model_path, self.experiment,
            console=False, status=False, dialog=False, runtime=True,
            parameters=parameters,
            timeout=self.timeout,
        )
        rtype = resp.get("type", "")
        if rtype != "CommandExecutedSuccessfully":
            raise GamaEnvError(f"Loi load experiment: {resp}")
        self.exp_id = resp.get("content")
        # gama-server may start the newly loaded GUI experiment immediately.
        # Pause before the first reload/step so its worker cannot mutate the
        # vehicle population while a new simulation is being initialized.
        await self.pause_until_stable()
        if self.verbose:
            print(f"[GamaEnv] Da load '{self.experiment}' -> exp_id={self.exp_id}")
        return self.exp_id

    async def close(self, stop_experiment=True):
        try:
            if self.client is not None and self.exp_id is not None:
                await self.pause_until_stable()
                if stop_experiment:
                    # A standalone server must release its experiment. When
                    # the trainer owns the entire JVM, the manager stops it
                    # directly, avoiding GAMA disposing a busy agent scope.
                    await asyncio.sleep(0.5)
                    await self.stop_experiment()
                    await asyncio.sleep(0.5)
        finally:
            self.disconnect()

    def disconnect(self):
        """Drop the WebSocket after an owned temporary JVM has been stopped."""
        if self.client is not None:
            self.client.close_connection()
            self.client = None
        self.exp_id = None

    async def pause(self):
        resp = await self.client.pause_awaitable(self.exp_id, timeout=self.timeout)
        if resp.get("type") != "CommandExecutedSuccessfully":
            raise GamaEnvError(f"Could not pause GAMA experiment: {resp}")

    async def pause_until_stable(self):
        """Wait until two consecutive reads see the same paused cycle."""
        await self.pause()
        for _ in range(10):
            before = int(float(await self._exec_expression("cycle")))
            await asyncio.sleep(0.2)
            after = int(float(await self._exec_expression("cycle")))
            if before == after:
                return
            await self.pause()
        raise GamaEnvError("GAMA kept stepping after pause; refusing to reload")

    async def stop_experiment(self):
        resp = await self.client.stop_awaitable(self.exp_id, timeout=self.timeout)
        if resp.get("type") != "CommandExecutedSuccessfully":
            raise GamaEnvError(f"Could not stop GAMA experiment: {resp}")
        self.exp_id = None

    # ------------------------------------------------------------ primitives
    async def _exec_expression(self, expr):
        """Chay 1 bieu thuc GAML trong exp_id, tra ve content."""
        resp = await self.client.expression_awaitable(
            self.exp_id, expr, timeout=self.timeout)
        rtype = resp.get("type", "")
        if rtype != "CommandExecutedSuccessfully":
            raise GamaEnvError(f"Expression failed: '{expr}' -> {resp}")
        content = resp.get("content")
        if isinstance(content, list) and content:
            content = content[0]
        return content

    # ------------------------------------------------------------ state I/O
    async def read_state(self, force_dtse=False):
        """Doc 18 bien state cua GAMA trong 1 request."""
        raw = await self._exec_expression(STATE_PACK_EXPR)
        parts = str(raw).split("|")
        if len(parts) != len(STATE_FIELDS):
            raise GamaEnvError(
                f"state_pack sai dinh dang ({len(parts)} truong thay vi "
                f"{len(STATE_FIELDS)}): '{raw}'"
            )
        state = {}
        for name, val in zip(STATE_FIELDS, parts):
            if name == "dqn_waiting_action":
                state[name] = str(val).strip().lower() == "true"
            elif name in INT_FIELDS:
                state[name] = int(float(val))
            else:
                state[name] = float(val)
        if state["dqn_waiting_action"] or force_dtse:
            action_name = "capture_spatial_state" if self.state_mode == "spatial" else "capture_dtse"
            resp = await self.client.ask_awaitable(
                self.exp_id, action_name, "{}", "world.0", timeout=self.timeout)
            if resp.get("type") != "CommandExecutedSuccessfully":
                raise GamaEnvError(f"{action_name} failed: {resp}")
            if self.state_mode == "spatial":
                packed = str(await self._exec_expression(SPATIAL_PACK_EXPR)).rsplit("|", 7)
                if len(packed) != 8:
                    raise GamaEnvError("Invalid spatial state payload")
                (raw_spatial, system_integral, stopped_integral, external_integral,
                 blocked_integral, weighted_integral, raw_pressure, raw_mask) = packed
                values = np.asarray([float(x) for x in re.findall(
                    r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", raw_spatial
                )], dtype=np.float32)
                if values.size != 7680 or not np.isfinite(values).all():
                    raise GamaEnvError(f"Spatial state has {values.size} finite values; expected 7680")
                state["spatial"] = values.reshape(4, 6, 20, 16)
                state["system_time_integral"] = float(system_integral)
                state["stopped_time_integral"] = float(stopped_integral)
                state["external_all_time_integral"] = float(external_integral)
                state["outgoing_block_integral"] = float(blocked_integral)
                state["weighted_delay_integral"] = float(weighted_integral)
                state["pressure"] = np.asarray([float(x) for x in re.findall(
                    r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", raw_pressure
                )], dtype=np.float32)
                state["action_mask"] = np.asarray(
                    [int(x) for x in re.findall(r"\d+", raw_mask)], dtype=np.bool_)
                if (state["pressure"].shape != (4,) or
                        not all(np.isfinite(state[name]) for name in
                                ("system_time_integral", "stopped_time_integral",
                                 "external_all_time_integral", "outgoing_block_integral",
                                 "weighted_delay_integral")) or
                        not np.isfinite(state["pressure"]).all()):
                    raise GamaEnvError("Invalid spatial pressure or R1-R4 integral")
                if state["action_mask"].shape != (4,) or not state["action_mask"].any():
                    raise GamaEnvError(f"Invalid action mask: {raw_mask}")
                return state
            packed = str(await self._exec_expression(DTSE_PACK_EXPR)).rsplit("|", 5)
            if len(packed) != 6:
                raise GamaEnvError("Invalid DTSE metrics payload")
            raw_dtse, delay_all, delay_near, queue_integral, external_integral, raw_mask = packed
            values = np.asarray([float(x) for x in re.findall(
                r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", raw_dtse
            )], dtype=np.float32)
            if values.size != 480:
                raise GamaEnvError(f"DTSE has {values.size} values; expected 480")
            state["dtse"] = values.reshape(3, 8, 20)
            state["delay_sq"] = float(delay_all if self.reward_scope == "all" else delay_near)
            state["queue_time_integral"] = float(queue_integral)
            state["external_wait_integral"] = float(external_integral)
            state["action_mask"] = np.asarray(
                [int(x) for x in re.findall(r"\d+", raw_mask)], dtype=np.bool_)
            if state["action_mask"].shape != (4,) or not state["action_mask"].any():
                raise GamaEnvError(f"Invalid action mask: {raw_mask}")
            if not np.isfinite(values).all() or not all(np.isfinite(state[name]) for name in
                ("delay_sq", "queue_time_integral", "external_wait_integral")):
                raise GamaEnvError("Non-finite DTSE or reward input")
        return state

    async def set_action(self, action):
        """Send selected green phase (0..3) to GAMA through an ask command.

        QUAN TRONG:
          - agent phai la dang tham chieu '<ten>.0' (world agent index 0) vi
            `AgentReference` ben server split theo dau cham; neu khong co duoi
            `.0` thi loi AIOOBE lam server khong tra loi (timeout).
          - args phai la mot JSON STRING (server lam `Json.parse((String) args)`);
            neu gui dict -> ClassCastException -> timeout.
        """
        if action not in range(self.action_count):
            raise GamaEnvError(f"Action khong hop le: {action}")
        args = json.dumps({"a": float(action)})
        resp = await self.client.ask_awaitable(
            self.exp_id, "set_dqn_action", args, "world.0", timeout=self.timeout)
        if resp.get("type") != "CommandExecutedSuccessfully":
            raise GamaEnvError(f"ask set_dqn_action failed: {resp}")

    async def step_cycles(self, n):
        """Run n cycles in bounded synchronous batches and check each response."""
        remaining = int(n)
        if remaining < 0:
            raise ValueError(f"Cycle count must be non-negative: {n}")
        while remaining:
            chunk = min(500, remaining)
            resp = await self.client.step_awaitable(
                self.exp_id, chunk, sync=True, timeout=self.timeout)
            if resp.get("type") != "CommandExecutedSuccessfully":
                raise GamaEnvError(f"GAMA step failed: {resp}")
            remaining -= chunk

    async def get_next_decision_in(self):
        return int(float(await self._exec_expression("next_decision_in_cycles")))

    def reward_from_state(self, state):
        delay_sq = state["delay_sq"]
        if self.max_delay_sq is not None and self.max_delay_sq > 0:
            norm = self.max_delay_sq
        else:
            self._worst_delay_sq = max(self._worst_delay_sq, delay_sq)
            norm = self._worst_delay_sq
        return 0.0 if norm == 0.0 else max(0.0, min(1.0, 1.0 - delay_sq / norm))

    def transition_reward(self, previous, current):
        if self.reward_mode in ("r1", "r2", "r3", "r4"):
            delta = self.reward_cost_from_state(current) - self.reward_cost_from_state(previous)
            if not np.isfinite(delta) or delta < -1e-6:
                raise GamaEnvError(f"Invalid {self.reward_mode} cost increment: {delta}")
            return -max(0.0, delta)
        if self.reward_mode == "queue_time":
            return -(current["queue_time_integral"] - previous["queue_time_integral"]) / 50.0
        if self.reward_mode == "total_wait":
            return -((current["queue_time_integral"] - previous["queue_time_integral"]) +
                     (current["external_wait_integral"] - previous["external_wait_integral"])) / 50.0
        return self.reward_from_state(current)

    def reward_cost_from_state(self, state):
        if self.reward_mode == "r1":
            return state["system_time_integral"] / self.r1_ref
        if self.reward_mode == "r2":
            return (state["stopped_time_integral"] +
                    self.r2_lambda * state["external_all_time_integral"]) / self.r2_ref
        if self.reward_mode == "r3":
            return (state["system_time_integral"] / self.r1_ref +
                    self.r3_beta * state["outgoing_block_integral"] / self.b_ref)
        if self.reward_mode == "r4":
            return state["weighted_delay_integral"] / self.r1_ref
        raise ValueError("Cumulative cost is defined only for R1-R4")

    # -------------------------------------------------------------- gym API
    async def reset(self, seed=None, parameters=None):
        """Load the first seeded episode, then reload for later episodes.

        Luu y: tham so dac biet "seed" cua gama-server reload bi bo qua
        (cac episode trung lap) -> phai truyen qua tham so thuong "rng_seed"
        (model se gan seed <- rng_seed trong init, truoc khi sinh xe).
        """
        params = list(parameters or [])
        self._server_messages.clear()
        if self.fast_training:
            params.extend([
                {"type": "bool", "value": "true", "name": "fast_training"},
                {"type": "string", "value": self.reward_mode if self.reward_mode == "r1"
                 else "all", "name": "training_reward_mode"},
            ])
        if seed is not None:
            params.append({"type": "float", "value": str(float(int(seed))),
                           "name": "rng_seed"})
        if self.exp_id is None:
            await self.load_experiment(parameters=params)
        else:
            await self.pause_until_stable()
            await asyncio.sleep(0.5)
            if self.reset_mode == "fresh":
                await self.stop_experiment()
                await asyncio.sleep(0.5)
                await self.load_experiment(parameters=params)
            else:
                resp = await self.client.reload_awaitable(
                    self.exp_id, parameters=params, timeout=self.timeout)
                if resp.get("type") != "CommandExecutedSuccessfully":
                    raise GamaEnvError(f"Loi reload experiment (seed={seed}): {resp}")
                await self.pause_until_stable()
        state = await self.read_state()
        if state["cycle"] != 0:
            raise GamaEnvError(f"Reset did not start at cycle 0: {state['cycle']}")
        if seed is not None:
            applied_seed = float(await self._exec_expression("world.seed"))
            if applied_seed != int(seed):
                raise GamaEnvError(f"GAMA applied seed {applied_seed}, expected {int(seed)}")
        return state

    async def step(self, action, horizon=None, current_state=None):
        """
        Thuc hien 1 buong quyet dinh DQN (protocol handshake day du).
        Optional horizon truncates exactly, including during junction clearance.
        A truncated state is not a terminal traffic state. Only decision states
        are valid bootstrap states for this action interface.
        """
        current_state = current_state if current_state is not None else await self.read_state()
        if action not in range(self.action_count):
            raise GamaEnvError(f"Action khong hop le: {action}")
        if not current_state["dqn_waiting_action"]:
            raise GamaEnvError(
                f"DQN action sent outside a decision tick (cycle={current_state['cycle']})"
            )
        if horizon is not None and horizon <= current_state["cycle"]:
            raise GamaEnvError("Action cannot be applied at or after the horizon")
        if "action_mask" in current_state and not current_state["action_mask"][action]:
            raise GamaEnvError(f"Action {action} violates the phase service constraints")
        await self.set_action(action)
        await self.step_cycles(1)                      # handshake: GAMA ap dung action
        nb = await self.get_next_decision_in()
        if nb <= 0:
            raise GamaEnvError(f"Invalid next decision interval: {nb}")
        count = nb if horizon is None else min(nb, horizon - current_state["cycle"] - 1)
        await self.step_cycles(count)
        next_state = await self.read_state()
        # Yellow may stay active until vehicles already inside the junction clear.
        # Advance to an actual decision tick instead of returning an intermediate state.
        advanced = count + 1
        while not next_state["dqn_waiting_action"] and (horizon is None or next_state["cycle"] < horizon):
            if advanced >= 1200:
                raise GamaEnvError("No DQN decision within 1200 cycles after action")
            more = max(1, int(next_state["next_decision_in_cycles"]))
            more = min(more, 1200 - advanced)
            if horizon is not None:
                more = min(more, horizon - next_state["cycle"])
            await self.step_cycles(more)
            advanced += more
            next_state = await self.read_state()

        # Ducrocq/Farhi: clipped improvement over the worst squared speed
        # delay observed since this environment was created. The first
        # nonzero delay gives 0.
        truncated = horizon is not None and next_state["cycle"] >= horizon
        observation_key = "spatial" if self.state_mode == "spatial" else "dtse"
        if truncated and observation_key not in next_state:
            next_state = await self.read_state(force_dtse=True)
        reward = self.transition_reward(current_state, next_state)
        info = {"next_decision_in_cycles": advanced - 1,
                "elapsed_seconds": advanced * getattr(self, "step_seconds", 0.1), "truncated": truncated,
                "bootstrap_valid": next_state["dqn_waiting_action"]}
        return next_state, reward, truncated, info


# ---------------------------------------------------------------------------
# Featurization: state dict -> DTSE tensor (occupancy, speed, signal).
# ---------------------------------------------------------------------------
STATE_DIM = (3, 8, 20)


def featurize(state):
    if "spatial" in state:
        phase = state["global_phase"] // 2
        aux = np.zeros(17, dtype=np.float32)
        if 0 <= phase < 4:
            aux[phase] = 1.0
        step_sec = state.get("step_seconds", 0.1)
        aux[4] = min(1.0, (state["green_cycles"] * step_sec) / 60.0)
        aux[5:9] = np.clip([state["queue_ns"], state["queue_ew"],
                            state["approach_ns"], state["approach_ew"]], 0, 100) / 100.0
        pressure = np.asarray(state["pressure"], dtype=np.float32)
        aux[9:13] = np.clip(pressure / 50.0, -1.0, 1.0)
        aux[13:17] = state["action_mask"].astype(np.float32)
        return {"grid": np.asarray(state["spatial"], dtype=np.float32), "aux": aux,
                "pressure": pressure}
    return np.asarray(state["dtse"], dtype=np.float32)
