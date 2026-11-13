"""Dueling Double DQN for the GAMA DTSE observation (3, 8, 20)."""

import math
import random
from collections import deque

import numpy as np
import torch
from torch import nn


STATE_SHAPE = (3, 8, 20)
N_ACTIONS = 4
PACKED_SIZE = 200  # 2 binary planes: 320 bits, plus 160 quantized speeds
ARCHITECTURE = "gama-3dqn-dtse-v3-paper-tsd"


def valid_mask(mask=None):
    result = np.ones(N_ACTIONS, dtype=np.bool_) if mask is None else np.asarray(mask, dtype=np.bool_)
    if result.shape != (N_ACTIONS,) or not result.any():
        raise ValueError("At least one of the four signal actions must be legal")
    return result


def pack_state(state):
    state = np.asarray(state, dtype=np.float32)
    if state.shape != STATE_SHAPE:
        raise ValueError(f"DTSE shape {state.shape}, expected {STATE_SHAPE}")
    if not np.isfinite(state).all():
        raise ValueError("DTSE contains a non-finite value")
    binary = np.packbits(np.concatenate((state[0].ravel() > 0.5,
                                          state[2].ravel() > 0.5)))
    speed = np.rint(np.clip(state[1].ravel(), 0, 1) * 255).astype(np.uint8)
    return np.concatenate((binary, speed))


def unpack_states(packed):
    packed = np.asarray(packed, dtype=np.uint8)
    bits = np.unpackbits(packed[:, :40], axis=1).astype(np.float32)
    result = np.empty((len(packed), *STATE_SHAPE), dtype=np.float32)
    result[:, 0] = bits[:, :160].reshape(-1, 8, 20)
    result[:, 1] = packed[:, 40:].reshape(-1, 8, 20) / 255.0
    result[:, 2] = bits[:, 160:].reshape(-1, 8, 20)
    return result


class ReplayBuffer:
    def __init__(self, capacity=1_000_000, seed=None):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.states = np.empty((capacity, PACKED_SIZE), dtype=np.uint8)
        self.next_states = np.empty((capacity, PACKED_SIZE), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.uint8)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.next_masks = np.empty((capacity, N_ACTIONS), dtype=np.bool_)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done, discount, next_mask=None):
        if action not in range(N_ACTIONS) or not np.isfinite(reward) or not 0 <= discount <= 1:
            raise ValueError("Invalid replay action, reward or discount")
        p = self.pos
        self.states[p] = pack_state(state)
        self.next_states[p] = pack_state(next_state)
        self.actions[p] = action
        self.rewards[p] = reward
        self.dones[p] = done
        self.discounts[p] = discount
        self.next_masks[p] = valid_mask(next_mask)
        self.pos = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(self.size, size=batch_size)
        return (unpack_states(self.states[idx]), self.actions[idx].astype(np.int64),
                self.rewards[idx], unpack_states(self.next_states[idx]),
                self.dones[idx].astype(np.float32), self.discounts[idx], self.next_masks[idx])

    def __len__(self):
        return self.size


class DuelingQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2), nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1), nn.ELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flattened = self.features(torch.zeros(1, *STATE_SHAPE)).shape[1]
        self.shared = nn.Sequential(nn.Linear(flattened, 128), nn.ELU(),
                                    nn.Linear(128, 64), nn.ELU())
        self.value = nn.Linear(64, 1)
        self.advantage = nn.Linear(64, N_ACTIONS)

    def forward(self, x):
        h = self.shared(self.features(x))
        advantage = self.advantage(h)
        return self.value(h) + advantage - advantage.mean(dim=1, keepdim=True)


class FlatQNetwork(nn.Module):
    """Four-action DQN baseline using the same DTSE without convolution/dueling."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(480, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, N_ACTIONS))

    def forward(self, x):
        return self.net(x)


class ThreeDQNAgent:
    def __init__(self, lr=1e-4, gamma=0.99, buffer_size=1_000_000,
                 warmup=100_000, batch_size=32, tau=1e-3,
                 eps_start=1.0, eps_min=0.001, eps_decay_steps=2_000_000,
                 seed=None, device=None, reward_mode="reference",
                 network_factory=DuelingQNetwork, architecture=ARCHITECTURE,
                 double_dqn=True):
        if reward_mode not in ("reference", "queue_time", "total_wait"):
            raise ValueError(f"Unknown reward mode: {reward_mode}")
        if seed is not None:
            torch.manual_seed(seed)
        self.rng = random.Random(seed)
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.gamma = float(gamma)
        self.reward_mode = reward_mode
        self.architecture = architecture
        self.double_dqn = bool(double_dqn)
        self.batch_size = int(batch_size)
        self.warmup = int(warmup)
        self.tau = float(tau)
        self.eps_start = float(eps_start)
        self.eps_min = float(eps_min)
        self.eps_decay_steps = float(eps_decay_steps)
        self.decisions = 0
        self.eps = self.eps_start
        self.online_net = network_factory().to(self.device)
        self.target_net = network_factory().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_size, seed=seed)
        self.update_count = 0
        self.last_losses = deque(maxlen=100)

    def act(self, state, greedy=False, action_mask=None, force_random=False):
        mask = valid_mask(action_mask)
        if force_random or (not greedy and self.rng.random() < self.eps):
            return int(self.rng.choice(np.flatnonzero(mask).tolist()))
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(state)[None], dtype=torch.float32,
                                device=self.device)
            scores = self.online_net(x).masked_fill(
                ~torch.as_tensor(mask[None], device=self.device), -torch.inf)
            return int(scores.argmax(dim=1).item())

    def decay_epsilon(self):
        self.decisions += 1
        self.eps = max(self.eps_min,
                       self.eps_start * math.exp(-self.decisions / self.eps_decay_steps))

    def store(self, state, action, reward, next_state, done, seconds=None, next_mask=None):
        if self.reward_mode in ("queue_time", "total_wait"):
            if seconds is None or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("queue_time learning requires the actual positive transition duration")
            discount = self.gamma ** (seconds / 5.0)
        else:
            discount = self.gamma
        self.buffer.push(state, action, reward, next_state, done, discount, next_mask)

    def can_update(self):
        return len(self.buffer) >= max(self.warmup, self.batch_size)

    def update(self):
        states, actions, rewards, next_states, dones, discounts, next_masks = self.buffer.sample(self.batch_size)
        s = torch.as_tensor(states, device=self.device)
        a = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        r = torch.as_tensor(rewards, device=self.device)
        s2 = torch.as_tensor(next_states, device=self.device)
        d = torch.as_tensor(dones, device=self.device)
        discount = torch.as_tensor(discounts, device=self.device)
        masks = torch.as_tensor(next_masks, device=self.device)
        q = self.online_net(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            if self.double_dqn:
                best_actions = self.online_net(s2).masked_fill(~masks, -torch.inf).argmax(dim=1, keepdim=True)
                next_q = self.target_net(s2).gather(1, best_actions).squeeze(1)
            else:
                next_q = self.target_net(s2).masked_fill(~masks, -torch.inf).max(dim=1).values
            target = r + discount * (1.0 - d) * next_q
        loss = self.loss_fn(q, target)
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
        torch.save({"online_net": self.online_net.state_dict(),
                    "target_net": self.target_net.state_dict(),
                    "eps": self.eps, "decisions": self.decisions,
                    "update_count": self.update_count,
                    "gamma": self.gamma, "reward_mode": self.reward_mode,
                    "architecture": self.architecture,
                    "double_dqn": self.double_dqn}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("architecture") != self.architecture:
            raise ValueError("Checkpoint uses an older DTSE/control contract; retrain with the current GAMA model")
        self.gamma = ckpt["gamma"]
        self.reward_mode = ckpt["reward_mode"]
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.eps = ckpt["eps"]
        self.decisions = ckpt["decisions"]
        self.update_count = ckpt["update_count"]
        return ckpt

    def training_state(self):
        """Full state needed to resume off-policy learning without replay loss."""
        replay = self.buffer
        return {
            "policy": {"online": self.online_net.state_dict(),
                       "target": self.target_net.state_dict()},
            "optimizer": self.optimizer.state_dict(),
            "architecture": self.architecture,
            "eps": self.eps, "decisions": self.decisions,
            "update_count": self.update_count,
            "losses": list(self.last_losses),
            "action_rng": self.rng.getstate(),
            "replay_rng": replay.rng.bit_generator.state,
            "replay": {"capacity": replay.capacity, "size": replay.size,
                       "pos": replay.pos,
                       **{name: getattr(replay, name)[:replay.size]
                          for name in ("states", "next_states", "actions",
                                       "rewards", "dones", "discounts", "next_masks")}},
        }

    def restore_training(self, state):
        if state["architecture"] != self.architecture:
            raise ValueError("Training checkpoint architecture mismatch")
        replay_state = state["replay"]
        if replay_state["capacity"] != self.buffer.capacity:
            raise ValueError("Training checkpoint replay capacity mismatch")
        self.online_net.load_state_dict(state["policy"]["online"])
        self.target_net.load_state_dict(state["policy"]["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.eps = state["eps"]
        self.decisions = state["decisions"]
        self.update_count = state["update_count"]
        self.last_losses.clear()
        self.last_losses.extend(state["losses"])
        self.rng.setstate(state["action_rng"])
        self.buffer.rng.bit_generator.state = state["replay_rng"]
        self.buffer.size = replay_state["size"]
        self.buffer.pos = replay_state["pos"]
        for name in ("states", "next_states", "actions", "rewards",
                     "dones", "discounts", "next_masks"):
            getattr(self.buffer, name)[:self.buffer.size] = replay_state[name]


class FlatDQNAgent(ThreeDQNAgent):
    """Replay/target DQN with the same four legal phases and DTSE input."""

    def __init__(self, **kwargs):
        super().__init__(network_factory=FlatQNetwork,
                         architecture="gama-four-phase-flat-dqn-v1",
                         double_dqn=False, **kwargs)
