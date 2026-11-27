#!/usr/bin/env python3
"""Local HTTP observer for the Le Van Luong Python traffic simulator."""

import argparse
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "python-simulation" / "ui"
GEOJSON_DIR = (ROOT / "python-simulation" / "simulation" / "python_simulation" / "includes")
sys.path.insert(0, str(ROOT))
PYTHON_SIM_ROOT = ROOT / "python-simulation" / "simulation"
sys.path.insert(0, str(PYTHON_SIM_ROOT))

from python_simulation.demand import generate_demand
from python_simulation.python_sim import LeVanLuongSimulator
from python_simulation.sim_config import load_sim_config
from python_simulation.vehicle_geometry import SUBLANE_WIDTH
from python_simulation.python_sim import DT


class RainbowController:
    """Greedy Rainbow DQN policy from a training run, acting on the observer's
    simulator exactly as ``train/logic/evaluate.py`` does in the Python env."""

    def __init__(self, weights):
        import torch
        from train.logic.gama_env import featurize
        from train.logic.lvl_env import _observation
        from train.logic.rainbow_agent import (ARCHITECTURE, LEGACY_ARCHITECTURE,
                                               PREVIOUS_ARCHITECTURE, RainbowAgent)
        weights = Path(weights).resolve()
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        architecture = checkpoint.get("architecture")
        if architecture not in (ARCHITECTURE, LEGACY_ARCHITECTURE, PREVIOUS_ARCHITECTURE):
            raise ValueError(f"Not a Rainbow checkpoint: {architecture}")
        settings = dict(checkpoint.get("reward_settings", {}))
        r1_ref = settings.get("r1_ref", checkpoint.get("r1_ref", 50.0))
        settings.setdefault("r1_ref", r1_ref)
        config_path = weights.parent / "training_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        self.queue_feature_scale = (settings.get("queue_feature_scale") or
                                    config.get("queue_feature_scale", "linear"))
        self.agent = RainbowAgent(buffer_size=1, warmup=1, batch_size=1,
                                  atoms=checkpoint["atoms"], v_min=checkpoint["v_min"],
                                  v_max=checkpoint["v_max"], r1_ref=r1_ref,
                                  reward_mode=checkpoint["reward_mode"],
                                  reward_settings=settings, device="cpu",
                                  aux_size=17 if architecture == ARCHITECTURE else 13)
        self.agent.load(str(weights))
        self.agent.eps = 0.0
        self.name = f"Rainbow DQN ({checkpoint['reward_mode'].upper()}) · {weights.parent.name}/{weights.name}"
        self._featurize, self._observation = featurize, _observation
        self.decisions = 0
        self.last_action = None

    def observe(self, sim):
        # Same observation as LVLPythonEnv.read_state (train/logic/lvl_env.py).
        vehicles = []
        for vehicle in sim.vehicles:
            road, progress = sim._road_state(vehicle)
            remaining = max(0.0, sim.roads[road]["length"] - progress)
            vehicles.append((vehicle.speed, vehicle.desired, vehicle.lane,
                             vehicle.occupied, road, remaining))
        yellow_total = int(round(sim._yellow / DT))
        return self._observation(sim.cycle, sim.phase, sim.dqn_waiting_action,
                                 sim.next_decision_in_cycles, sim.created, sim.passed,
                                 sim.queue_time_integral, sim.system_time_integral,
                                 sim.weighted_delay_integral, vehicles, "spatial",
                                 green_cycles=getattr(sim, "green_ticks", None),
                                 mask=sim.action_mask(),
                                 transition_of=(sim.yellow_phase or sim._last_yellow),
                                 yellow_cycles=(yellow_total - sim.yellow_ticks_left
                                                if sim.yellow_phase else 0),
                                 extra={"jam_removed": sim.jam_removed,
                                        "hard_brake_events": sim.hard_brake_events})

    def act(self, sim):
        state = self.observe(sim)
        action = self.agent.act(self._featurize(state, self.queue_feature_scale),
                                greedy=True, action_mask=state["action_mask"])
        sim.set_action(int(action))
        self.decisions += 1
        self.last_action = int(action)
        return action


class Observer:
    def __init__(self, seed=1, mode="fixed", junction_model=None,
                 sim_kwargs=None, flow=3000.0, controller=None):
        self.sim_kwargs = load_sim_config(config=sim_kwargs)
        configured_junction = self.sim_kwargs.pop("junction_model")
        self.junction_model = junction_model or configured_junction
        self.lock = threading.RLock()
        self.flow = float(flow)
        self.controller = controller
        self.seed = seed
        self._demand_path = None
        self.sim = None
        self.mode = "fixed"
        self.reset(seed, mode)

    def reset(self, seed, mode="fixed"):
        if mode not in ("fixed", "train"):
            raise ValueError("mode must be 'fixed' or 'train'")
        if self.controller is not None:
            mode = "train"  # the trained policy drives the signal
        if self._demand_path:
            self._demand_path.unlink(missing_ok=True)
        temp = tempfile.NamedTemporaryFile(prefix="lvl_observer_", suffix=".csv", delete=False)
        temp.close()
        self._demand_path = Path(temp.name)
        generate_demand(self._demand_path, seed=seed, horizon=8000,
                        flow_per_approach=self.flow,
                        heavy_ratio=self.sim_kwargs["heavy_ratio"])
        self.seed = seed
        self.mode = mode
        self.sim = LeVanLuongSimulator(seed=seed, demand_file=self._demand_path,
                                       train_mode=(mode == "train"),
                                       junction_model=self.junction_model,
                                       **self.sim_kwargs)
        if self.controller is not None:
            self.controller.decisions = 0
            self.controller.last_action = None
        self.auto_act()
        return self.state()

    def auto_act(self):
        if self.controller is not None and self.sim.dqn_waiting_action:
            self.controller.act(self.sim)

    def _body(self, vehicle):
        pose = self.sim._rigid_pose(vehicle)
        return [round(pose.x, 4), round(pose.y, 4), round(pose.tx, 6), round(pose.ty, 6)]

    def state(self):
        result = self.sim.snapshot()
        result["mode"] = self.mode
        result["junction_model"] = self.junction_model
        result["flow_per_approach"] = self.flow
        result["controller"] = (self.controller.name if self.controller is not None
                                else "fixed plan" if self.mode == "fixed" else "manual")
        if self.controller is not None:
            result["agent_decisions"] = self.controller.decisions
            result["agent_last_action"] = self.controller.last_action
        result["simulation_config"] = {"junction_model": self.junction_model,
                                       **self.sim_kwargs}
        vehicles = []
        for vehicle in self.sim.vehicles:
            road, progress = self.sim._road_state(vehicle)
            is_changing = bool(vehicle.lane_change_progress < 1.0)
            target_shift = getattr(vehicle, "target_lateral_shift", (vehicle.lane + (vehicle.occupied - 1) / 2.0) * SUBLANE_WIDTH)
            curr_shift = getattr(vehicle, "current_lateral_shift", target_shift)
            start_shift = getattr(vehicle, "start_lateral_shift", curr_shift)
            turn_signal = "none"
            if is_changing:
                if target_shift > start_shift + 0.05:
                    turn_signal = "left"
                elif target_shift < start_shift - 0.05:
                    turn_signal = "right"
            vehicles.append({
                "id": vehicle.demand_row_id,
                "road": road,
                "progress": round(progress, 4),
                "lane": vehicle.lane,
                "occupied": vehicle.occupied,
                "speed": round(vehicle.speed, 4),
                "speed_kmh": round(vehicle.speed * 3.6, 2),
                "is_car": vehicle.occupied > 1,
                "kind": vehicle.kind,
                "width": round(2 * vehicle.half_width, 2),
                # Rigid two-axle body (heavy vehicles): centre and heading
                # in map coordinates, placed by its axles in the simulator.
                "body": (self._body(vehicle) if vehicle.wheelbase > 0.0 else None),
                "wheelbase": round(vehicle.wheelbase, 2),
                "front_overhang": round(vehicle.front_overhang, 2),
                "arm": vehicle.arm,
                "to_arm": vehicle.to_arm,
                "turn": vehicle.maneuver,
                "length": round(vehicle.length, 2),
                "lateral_shift": round(curr_shift, 4),
                "start_lateral_shift": round(start_shift, 4),
                "target_lateral_shift": round(target_shift, 4),
                "lane_change_progress": round(getattr(vehicle, "lane_change_progress", 1.0), 4),
                "lateral_duration": round(getattr(vehicle, "lateral_duration", 0.8), 4),
                "is_changing_lane": is_changing,
                "turn_signal": turn_signal,
            })
        result["vehicles"] = vehicles
        return result


def make_handler(observer):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, data, status=200):
            body = json.dumps(data, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ValueError("Invalid Content-Length")
            if length <= 0 or length > 65536:
                raise ValueError("Request body must be between 1 and 65536 bytes")
            try:
                data = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ValueError("Body must be valid JSON")
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def do_GET(self):
            if self.path == "/":
                path = UI_DIR / "observer.html"
                try:
                    body = path.read_bytes()
                except OSError:
                    return self._json({"error": "observer.html not found"}, 404)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/roads":
                try:
                    body = (GEOJSON_DIR / "roads.geojson").read_bytes()
                except OSError:
                    return self._json({"error": "roads.geojson not found"}, 500)
                self.send_response(200)
                self.send_header("Content-Type", "application/geo+json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/areas":
                try:
                    body = (GEOJSON_DIR / "areas.geojson").read_bytes()
                except OSError:
                    return self._json({"error": "areas.geojson not found"}, 500)
                self.send_response(200)
                self.send_header("Content-Type", "application/geo+json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/state":
                with observer.lock:
                    return self._json(observer.state())
            return self._json({"error": "Not found"}, 404)

        def do_POST(self):
            try:
                data = self._body()
                if self.path == "/api/step":
                    cycles = data.get("cycles")
                    if type(cycles) is not int or not 1 <= cycles <= 100:
                        raise ValueError("cycles must be an integer from 1 to 100")
                    with observer.lock:
                        if observer.controller is not None:
                            for _ in range(cycles):
                                observer.auto_act()
                                observer.sim.step(1)
                            observer.auto_act()
                            result = observer.state()
                            result["step_status"] = "advanced"
                            result["cycles_advanced"] = cycles
                            return self._json(result)
                        if observer.mode == "train":
                            advanced = 0
                            while advanced < cycles and not observer.sim.dqn_waiting_action:
                                observer.sim.step(1)
                                advanced += 1
                            result = observer.state()
                            result["step_status"] = ("waiting_for_action"
                                if observer.sim.dqn_waiting_action else "advanced")
                            result["cycles_advanced"] = advanced
                            return self._json(result)
                        observer.sim.step(cycles)
                        result = observer.state()
                        result["step_status"] = "advanced"
                        result["cycles_advanced"] = cycles
                        return self._json(result)
                if self.path == "/api/action":
                    action = data.get("action")
                    if type(action) is not int or action not in (0, 2):
                        raise ValueError("action must be the integer 0 or 2")
                    with observer.lock:
                        if observer.controller is not None:
                            return self._json({"error": "The trained policy controls the signal"}, 409)
                        if observer.mode != "train":
                            return self._json({"error": "Actions are only available in train mode"}, 409)
                        if not observer.sim.dqn_waiting_action:
                            return self._json({"error": "Simulator is not waiting for a decision action"}, 409)
                        observer.sim.set_action(action)
                        return self._json(observer.state())
                if self.path == "/api/reset":
                    seed = data.get("seed")
                    if type(seed) is not int:
                        raise ValueError("seed must be an integer")
                    mode = data.get("mode", "fixed")
                    if mode not in ("fixed", "train"):
                        raise ValueError("mode must be 'fixed' or 'train'")
                    with observer.lock:
                        return self._json(observer.reset(seed, mode))
                return self._json({"error": "Not found"}, 404)
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)

        def log_message(self, fmt, *args):
            print("[%s] %s" % (self.log_date_time_string(), fmt % args), file=sys.stderr)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--junction-model", choices=("negotiation", "reservation"),
                        default=None,
                        help="negotiation: vehicles enter, creep and weave (default); "
                             "reservation: the older ConflictEngine time-window arbiter")
    parser.add_argument("--startup-reaction", type=float, default=None,
                        help="max reaction delay (s) of a stopped queue at green onset")
    parser.add_argument("--startup-headway", type=float, default=None,
                        help="release headway (s) between queued vehicles in one sublane")
    parser.add_argument("--sim-config", type=Path,
                        help="Shared simulator JSON or calibrate_discharge output")
    parser.add_argument("--heavy-ratio", type=float, default=None,
                        help="share of heavy vehicles (trucks, buses) in the demand")
    parser.add_argument("--flow", type=float, default=3000.0,
                        help="demand per approach in vehicles/h (default 3000)")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Rainbow DQN weights (.pt) of a training run: the greedy "
                             "policy drives the signal, e.g. "
                             "train/results/rainbow_r1_5m/best_weights.pt")
    args = parser.parse_args()
    simulation = load_sim_config(args.sim_config)
    if args.junction_model:
        simulation["junction_model"] = args.junction_model
    if args.heavy_ratio is not None:
        simulation["heavy_ratio"] = args.heavy_ratio
    startup = dict(simulation.get("startup_params", {}))
    if args.startup_reaction is not None:
        startup["reaction_delay"] = args.startup_reaction
    if args.startup_headway is not None:
        startup["discharge_headway"] = args.startup_headway
    simulation["startup_params"] = startup
    junction_model = simulation.pop("junction_model")
    controller = RainbowController(args.weights) if args.weights else None
    observer = Observer(args.seed, junction_model=junction_model,
                        sim_kwargs=simulation, flow=args.flow, controller=controller)
    print(f"Junction model: {junction_model}")
    print(f"Demand: {args.flow:.0f} vehicles/h per approach")
    print(f"Signal: {controller.name if controller else 'fixed plan / manual'}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(observer))
    print(f"LVL observer listening at http://127.0.0.1:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if observer._demand_path:
            observer._demand_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
