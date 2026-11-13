"""Spatial Dueling Double DQN with Max Pressure in action selection and targets."""

import math
import random
from collections import deque

import numpy as np
import torch
from torch import nn

try:
    from .rainbow_agent import AUX_SIZE, N_ACTIONS, PrioritizedReplay
    from .three_dqn_agent import valid_mask
except ImportError:  # standalone train.py
    from .rainbow_agent import AUX_SIZE, N_ACTIONS, PrioritizedReplay
    from .three_dqn_agent import valid_mask


ARCHITECTURE = "gama-spatial-3dqn-maxpressure-4x6x20x16-v2"


def normalized_scores(values, masks):
    """Z-score each state's legal actions; all-equal scores contribute zero."""
    selected = torch.where(masks, values, torch.zeros_like(values))
    count = masks.sum(dim=1, keepdim=True).clamp(min=1)
    mean = selected.sum(dim=1, keepdim=True) / count
    variance = torch.where(masks, (values - mean).square(), 0.0).sum(dim=1, keepdim=True) / count
    scale = variance.sqrt()
    return torch.where(scale > 1e-6, (values - mean) / scale.clamp(min=1e-6), 0.0)


class HybridReplay(PrioritizedReplay):
    """Compressed spatial replay with uniform sampling and stored pressure."""

    def __init__(self, capacity, seed=None):
        super().__init__(capacity, alpha=0.0, seed=seed)
        self.pressures = np.zeros((capacity, N_ACTIONS), dtype=np.float32)
        self.next_pressures = np.zeros((capacity, N_ACTIONS), dtype=np.float32)

    def push(self, state, action, reward, next_state, done, discount, next_mask):
        p = self.pos
        pressure = np.asarray(state["pressure"], dtype=np.float32)
        next_pressure = np.asarray(next_state["pressure"], dtype=np.float32)
        if (pressure.shape != (N_ACTIONS,) or next_pressure.shape != (N_ACTIONS,) or
                not np.isfinite(pressure).all() or not np.isfinite(next_pressure).all()):
            raise ValueError("Hybrid replay requires four finite phase pressures")
        super().push(state, action, reward, next_state, done, discount, next_mask)
        self.pressures[p] = pressure
        self.next_pressures[p] = next_pressure

    def sample(self, batch_size):
        row = super().sample(batch_size, beta=0.0)
        return (*row, self.pressures[row[0]], self.next_pressures[row[0]])

    def snapshot(self):
        result = super().snapshot()
        result["pressures"] = self.pressures[:self.size].copy()
        result["next_pressures"] = self.next_pressures[:self.size].copy()
        return result

    def restore(self, state):
        super().restore(state)
        self.pressures[:self.size] = state["pressures"]
        self.next_pressures[:self.size] = state["next_pressures"]


class SpatialDuelingQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.arm_encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(), nn.Flatten(),
            nn.Linear(64 * 4 * 3, 128), nn.ReLU())
        self.shared = nn.Sequential(nn.Linear(4 * 128 + AUX_SIZE, 256), nn.ReLU(),
                                    nn.Linear(256, 128), nn.ReLU())
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, N_ACTIONS)

    def forward(self, grid, aux):
        batch = grid.shape[0]
        arms = self.arm_encoder(grid.reshape(batch * 4, 6, 20, 16)).reshape(batch, 4 * 128)
        shared = self.shared(torch.cat((arms, aux), dim=1))
        advantage = self.advantage(shared)
        return self.value(shared) + advantage - advantage.mean(dim=1, keepdim=True)


class SpatialHybridAgent:
    def __init__(self, lr=1e-4, gamma=.97, buffer_size=100_000, warmup=20_000,
                 batch_size=32, tau=1e-3, eps_start=1.0, eps_min=.001,
                 eps_decay_steps=2_000_000, seed=None, device=None, reward_mode="r1",
                 pressure_lambda=.5, reward_settings=None):
        if reward_mode not in ("r1", "r2", "r3", "r4"):
            raise ValueError("Spatial hybrid requires R1-R4")
        if (not all(math.isfinite(x) for x in
                    (lr, gamma, tau, eps_start, eps_min, eps_decay_steps, pressure_lambda)) or
                lr <= 0 or not 0 < gamma <= 1 or not 0 < tau <= 1 or
                not 0 <= eps_min <= eps_start <= 1 or eps_decay_steps <= 0 or
                pressure_lambda < 0 or warmup < 0 or batch_size < 1 or buffer_size < batch_size):
            raise ValueError("Invalid hybrid learning parameters")
        if seed is not None:
            torch.manual_seed(seed)
        self.rng = random.Random(seed)
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.architecture = ARCHITECTURE
        self.reward_mode = reward_mode
        self.reward_settings = dict(reward_settings or {"r1_ref": 50.0})
        self.pressure_lambda = float(pressure_lambda)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.warmup = int(warmup)
        self.batch_size = int(batch_size)
        self.eps_start = float(eps_start)
        self.eps_min = float(eps_min)
        self.eps_decay_steps = float(eps_decay_steps)
        self.eps = self.eps_start
        self.decisions = self.update_count = 0
        self.online_net = SpatialDuelingQNetwork().to(self.device)
        self.target_net = SpatialDuelingQNetwork().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = HybridReplay(buffer_size, seed=seed)
        self.last_losses = deque(maxlen=100)

    def _scores(self, q, pressure, masks):
        return (normalized_scores(q, masks) + self.pressure_lambda *
                normalized_scores(pressure, masks)).masked_fill(~masks, -torch.inf)

    def act(self, state, greedy=False, action_mask=None, force_random=False):
        mask = valid_mask(action_mask)
        if force_random or (not greedy and self.rng.random() < self.eps):
            return int(self.rng.choice(np.flatnonzero(mask).tolist()))
        with torch.no_grad():
            grid = torch.as_tensor(np.asarray(state["grid"])[None], device=self.device)
            aux = torch.as_tensor(np.asarray(state["aux"])[None], device=self.device)
            pressure = torch.as_tensor(np.asarray(state["pressure"])[None], device=self.device)
            legal = torch.as_tensor(mask[None], device=self.device)
            return int(self._scores(self.online_net(grid, aux), pressure, legal).argmax(dim=1).item())

    def decay_epsilon(self):
        self.decisions += 1
        self.eps = max(self.eps_min, self.eps_start * math.exp(-self.decisions / self.eps_decay_steps))

    def store(self, state, action, reward, next_state, done, seconds=None, next_mask=None):
        if seconds is None or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Hybrid learning needs actual positive decision duration")
        self.buffer.push(state, action, reward, next_state, done,
                         self.gamma ** (seconds / 5.0), next_mask)
        return 1

    def can_update(self):
        return len(self.buffer) >= max(self.warmup, self.batch_size)

    def update(self):
        (indices, _, states, aux, actions, rewards, next_states, next_aux,
         dones, discounts, next_masks, _, next_pressures) = self.buffer.sample(self.batch_size)
        device = self.device
        s = torch.as_tensor(states, device=device)
        a = torch.as_tensor(aux, device=device)
        s2 = torch.as_tensor(next_states, device=device)
        a2 = torch.as_tensor(next_aux, device=device)
        action = torch.as_tensor(actions, device=device).unsqueeze(1)
        reward = torch.as_tensor(rewards, device=device)
        done = torch.as_tensor(dones, device=device)
        discount = torch.as_tensor(discounts, device=device)
        mask = torch.as_tensor(next_masks, device=device)
        pressure = torch.as_tensor(next_pressures, device=device)
        q = self.online_net(s, a).gather(1, action).squeeze(1)
        with torch.no_grad():
            online_next = self.online_net(s2, a2)
            selected = self._scores(online_next, pressure, mask).argmax(dim=1, keepdim=True)
            target_q = self.target_net(s2, a2).gather(1, selected).squeeze(1)
            target = reward + discount * (1.0 - done) * target_q
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        with torch.no_grad():
            for target_param, online_param in zip(self.target_net.parameters(),
                                                  self.online_net.parameters()):
                target_param.lerp_(online_param, self.tau)
        self.update_count += 1
        value = float(loss.item())
        self.last_losses.append(value)
        return value

    @property
    def avg_loss(self):
        return float(np.mean(self.last_losses)) if self.last_losses else 0.0

    def save(self, path):
        torch.save({"architecture": ARCHITECTURE, "reward_mode": self.reward_mode,
                    "reward_settings": self.reward_settings,
                    "pressure_lambda": self.pressure_lambda,
                    "online_net": self.online_net.state_dict(),
                    "target_net": self.target_net.state_dict(),
                    "gamma": self.gamma, "eps": self.eps,
                    "decisions": self.decisions, "update_count": self.update_count}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("architecture") != ARCHITECTURE:
            raise ValueError("Spatial hybrid checkpoint architecture mismatch")
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.reward_mode = ckpt["reward_mode"]
        self.reward_settings = dict(ckpt["reward_settings"])
        self.pressure_lambda = ckpt["pressure_lambda"]
        self.gamma = ckpt["gamma"]
        self.eps = ckpt["eps"]
        self.decisions = ckpt["decisions"]
        self.update_count = ckpt["update_count"]
        return ckpt

    def training_state(self):
        return {"architecture": ARCHITECTURE,
                "policy": {"online": self.online_net.state_dict(),
                           "target": self.target_net.state_dict()},
                "optimizer": self.optimizer.state_dict(),
                "eps": self.eps, "decisions": self.decisions,
                "update_count": self.update_count,
                "losses": list(self.last_losses), "action_rng": self.rng.getstate(),
                "replay": self.buffer.snapshot()}

    def restore_training(self, state):
        if state["architecture"] != ARCHITECTURE:
            raise ValueError("Spatial hybrid training checkpoint architecture mismatch")
        self.online_net.load_state_dict(state["policy"]["online"])
        self.target_net.load_state_dict(state["policy"]["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.eps = state["eps"]
        self.decisions, self.update_count = state["decisions"], state["update_count"]
        self.last_losses.clear()
        self.last_losses.extend(state["losses"])
        self.rng.setstate(state["action_rng"])
        self.buffer.restore(state["replay"])
