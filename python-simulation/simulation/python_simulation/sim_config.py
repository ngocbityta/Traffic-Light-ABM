"""Shared, validated configuration for the Python Le Van Luong simulator."""
import json
from pathlib import Path

DEFAULT_CONFIG = {"junction_model": "negotiation", "yellow": 4.0,
                  "all_red": 2.0, "vehicle_params": {}, "startup_params": {}}
ALLOWED = set(DEFAULT_CONFIG) | {"negotiation_params"}


def load_sim_config(path=None, *, config=None):
    result = {**DEFAULT_CONFIG, "vehicle_params": {}, "startup_params": {}}
    if path is not None:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config is None:
        return result
    if not isinstance(config, dict):
        raise ValueError("Simulation configuration must be a JSON object")
    if "training_config" in config:
        config = config["training_config"]
    if "simulation_config" in config:
        config = config["simulation_config"]
    # calibrate_discharge.py emits {best, kwargs, evaluations, history}.
    if "kwargs" in config:
        config = config["kwargs"]
    if not isinstance(config, dict):
        raise ValueError("Simulation parameters must be a JSON object")
    unknown = set(config) - ALLOWED
    if unknown:
        raise ValueError(f"Unknown simulation configuration fields: {sorted(unknown)}")
    result.update(config)
    return result
