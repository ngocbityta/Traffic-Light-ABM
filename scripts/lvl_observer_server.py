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
SIM_DIR = ROOT / "LeVanLuongIntersection"
sys.path.insert(0, str(ROOT))

from LeVanLuongIntersection.demand import generate_demand
from LeVanLuongIntersection.python_sim import LeVanLuongSimulator


class Observer:
    def __init__(self, seed=1, mode="fixed"):
        self.lock = threading.RLock()
        self.seed = seed
        self._demand_path = None
        self.sim = None
        self.mode = "fixed"
        self.reset(seed, mode)

    def reset(self, seed, mode="fixed"):
        if mode not in ("fixed", "train"):
            raise ValueError("mode must be 'fixed' or 'train'")
        if self._demand_path:
            self._demand_path.unlink(missing_ok=True)
        temp = tempfile.NamedTemporaryFile(prefix="lvl_observer_", suffix=".csv", delete=False)
        temp.close()
        self._demand_path = Path(temp.name)
        generate_demand(self._demand_path, seed=seed, horizon=8000)
        self.seed = seed
        self.mode = mode
        self.sim = LeVanLuongSimulator(seed=seed, demand_file=self._demand_path,
                                       train_mode=(mode == "train"))
        return self.state()

    def state(self):
        result = self.sim.snapshot()
        result["mode"] = self.mode
        vehicles = []
        for vehicle in self.sim.vehicles:
            road, progress = self.sim._road_state(vehicle)
            is_changing = bool(vehicle.lane_change_progress < 1.0)
            target_shift = getattr(vehicle, "target_lateral_shift", (vehicle.lane + (vehicle.occupied - 1) / 2.0) * 0.8)
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
                "is_car": vehicle.occupied == 2,
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
                path = SIM_DIR / "observer.html"
                try:
                    body = path.read_bytes()
                except OSError:
                    return self._json({"error": "observer.html not found"}, 404)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/roads":
                try:
                    body = (SIM_DIR / "includes" / "roads.geojson").read_bytes()
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
                    body = (SIM_DIR / "includes" / "areas.geojson").read_bytes()
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
    args = parser.parse_args()
    observer = Observer(args.seed)
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
