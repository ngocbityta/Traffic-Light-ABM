"""Tabular Q-learning on a compact summary of the four-phase GAMA DTSE."""

import math
import random
from collections import deque

import numpy as np
import torch

try:
    from .three_dqn_agent import N_ACTIONS, STATE_SHAPE, valid_mask
except ImportError:  # direct execution of train.py/evaluate.py
    from .three_dqn_agent import N_ACTIONS, STATE_SHAPE, valid_mask


ARCHITECTURE = "gama-four-phase-tabular-q-v1"
MOVEMENT_ROWS = ((2, 6), (3, 7), (0, 4), (1, 5))


def state_key(state):
    """Coarse stopped/approaching counts per movement plus active green phase."""
    x = np.asarray(state, dtype=np.float32)
    if x.shape != STATE_SHAPE or not np.isfinite(x).all():
        raise ValueError("Invalid DTSE for tabular Q-learning")
    bins = []
    for rows in MOVEMENT_ROWS:
        occupied = x[0, list(rows)] > 0.5
        stopped = occupied & (x[1, list(rows)] < 0.1)
        # Logarithmic bins keep the table small while retaining queue severity.
        for count in (int(stopped.sum()), int(occupied.sum())):
            bins.append(0 if count == 0 else (1 if count < 3 else (2 if count < 6 else 3)))
    green = next((a for a, rows in enumerate(MOVEMENT_ROWS)
                  if x[2, rows[0], 0] > 0.5), 4)
    return tuple(bins) + (green,)


class TabularQAgent:
    def __init__(self, lr=0.1, gamma=0.99, buffer_size=1, warmup=0,
                 batch_size=1, tau=None, eps_start=1.0, eps_min=0.001,
                 eps_decay_steps=2_000_000, seed=None, device=None,
                 reward_mode="reference"):
        if reward_mode not in ("reference", "queue_time", "total_wait"):
            raise ValueError(f"Unknown reward mode: {reward_mode}")
        self.rng = random.Random(seed)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.reward_mode = reward_mode
        self.eps_start = float(eps_start)
        self.eps_min = float(eps_min)
        self.eps_decay_steps = float(eps_decay_steps)
        self.eps = self.eps_start
        self.decisions = 0
        self.update_count = 0
        self.device = "cpu"
        self.table = {}
        self.buffer = deque(maxlen=1)
        self.last_losses = deque(maxlen=100)

    def act(self, state, greedy=False, action_mask=None, force_random=False):
        allowed = np.flatnonzero(valid_mask(action_mask))
        if force_random or (not greedy and self.rng.random() < self.eps):
            return int(self.rng.choice(allowed.tolist()))
        q = self.table.get(state_key(state))
        if q is None:
            return int(allowed[0])
        return int(allowed[np.argmax(q[allowed])])

    def decay_epsilon(self):
        self.decisions += 1
        self.eps = max(self.eps_min,
                       self.eps_start * math.exp(-self.decisions / self.eps_decay_steps))

    def store(self, state, action, reward, next_state, done, seconds=None, next_mask=None):
        if action not in range(N_ACTIONS) or not math.isfinite(reward):
            raise ValueError("Invalid tabular transition")
        if self.reward_mode in ("queue_time", "total_wait"):
            if seconds is None or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("queue_time learning requires elapsed seconds")
            discount = self.gamma ** (seconds / 5.0)
        else:
            discount = self.gamma
        self.buffer.append((state_key(state), action, float(reward), state_key(next_state),
                            bool(done), discount, valid_mask(next_mask)))

    def can_update(self):
        return bool(self.buffer)

    def update(self):
        key, action, reward, next_key, done, discount, next_mask = self.buffer.popleft()
        current = self.table.setdefault(key, np.zeros(N_ACTIONS, dtype=np.float32))
        next_values = self.table.get(next_key)
        bootstrap = 0.0 if done or next_values is None else float(next_values[next_mask].max())
        error = reward + discount * bootstrap - float(current[action])
        current[action] += self.lr * error
        self.update_count += 1
        self.last_losses.append(abs(error))
        return abs(error)

    @property
    def avg_loss(self):
        return float(np.mean(self.last_losses)) if self.last_losses else 0.0

    def save(self, path):
        torch.save({"architecture": ARCHITECTURE, "table": self.table,
                    "eps": self.eps, "decisions": self.decisions,
                    "update_count": self.update_count, "gamma": self.gamma,
                    "reward_mode": self.reward_mode, "lr": self.lr}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != ARCHITECTURE:
            raise ValueError("Checkpoint is not a four-phase tabular Q-learning policy")
        self.table = checkpoint["table"]
        self.eps = checkpoint["eps"]
        self.decisions = checkpoint["decisions"]
        self.update_count = checkpoint["update_count"]
        self.gamma = checkpoint["gamma"]
        self.reward_mode = checkpoint["reward_mode"]
        self.lr = checkpoint["lr"]
        return checkpoint

    def training_state(self):
        return {"architecture": ARCHITECTURE, "table": self.table,
                "eps": self.eps, "decisions": self.decisions,
                "update_count": self.update_count,
                "losses": list(self.last_losses),
                "rng": self.rng.getstate()}

    def restore_training(self, state):
        if state["architecture"] != ARCHITECTURE:
            raise ValueError("Training checkpoint architecture mismatch")
        self.table = state["table"]
        self.eps = state["eps"]
        self.decisions = state["decisions"]
        self.update_count = state["update_count"]
        self.last_losses.clear()
        self.last_losses.extend(state["losses"])
        self.rng.setstate(state["rng"])
