#!/usr/bin/env python3
"""
==============================================================================
DQN AGENT (PyTorch) - Deep Q-Network cho dieu khien den giao thong
==============================================================================
Thanh phan DQN day du:
  - QNetwork: MLP (state_dim -> hidden -> n_actions), du doan Q(s, a)
  - Target network: ban sao on dinh cua Q-network, dong bo dinh ky
  - Experience Replay: bo dem chuyen tiep (s, a, r, s', done)
  - Epsilon-greedy exploration: giam dan per quyet dinh (giong Q-learning tabular)
  - Loss: Huber (SmoothL1) tren minibatch ngau nhien tu replay buffer
  - Bellman target: y = r + gamma * max_a' Q_target(s', a') (done -> y = r)

Hypermac dinh canh theo mo hinh Q-learning tabular de so sanh cong bang:
  gamma = 0.9, eps_start = 0.5, eps_decay = 0.995 (per decision), eps_min = 0.05
==============================================================================
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """MLP du doan gia tri Q cho moi action."""

    def __init__(self, state_dim, hidden_dims=(32, 32), n_actions=2):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, n_actions)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Bo dem kinh nghiem (pre-allocated numpy cho nhanh)."""

    def __init__(self, capacity, state_dim):
        self.capacity = int(capacity)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        p = self.pos
        self.states[p] = state
        self.actions[p] = action
        self.rewards[p] = reward
        self.next_states[p] = next_state
        self.dones[p] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, state_dim=8, n_actions=2, hidden_dims=(32, 32),
                 lr=1e-3, gamma=0.9, buffer_size=20000, batch_size=64,
                 target_sync_every=200, eps_start=0.5, eps_decay=0.995,
                 eps_min=0.05, seed=None, device=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_sync_every = int(target_sync_every)

        self.eps = float(eps_start)
        self.eps_decay = float(eps_decay)
        self.eps_min = float(eps_min)

        self.online_net = QNetwork(state_dim, hidden_dims, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, hidden_dims, n_actions).to(self.device)
        self.sync_target()
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=float(lr))
        self.loss_fn = nn.SmoothL1Loss()          # Huber loss - on dinh hon MSE
        self.buffer = ReplayBuffer(buffer_size, state_dim)
        self.update_count = 0
        self.last_losses = deque(maxlen=100)

    # ------------------------------------------------------------ exploration
    def act(self, state_vec, greedy=False):
        """Epsilon-greedy (hoac greedy thuan khi greedy=True / evaluate)."""
        if not greedy and random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            s = torch.as_tensor(state_vec, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            q = self.online_net(s)
            return int(torch.argmax(q, dim=1).item())

    def decay_epsilon(self):
        """Giam epsilon SAU MOI quyet dinh (giong Q-learning tabular cua project)."""
        if self.eps > self.eps_min:
            self.eps = max(self.eps_min, self.eps * self.eps_decay)

    # ------------------------------------------------------------ training
    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def can_update(self):
        return len(self.buffer) >= self.batch_size

    def update(self):
        """1 buot gradient descent tren minibatch tu replay buffer."""
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        d = torch.as_tensor(d, device=self.device)

        q_sa = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_net(s2).max(dim=1).values
            target = r + self.gamma * max_next_q * (1.0 - d)

        loss = self.loss_fn(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clip de tranh no gradient (huan luyen trong mo phong)
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_sync_every == 0:
            self.sync_target()
        loss_val = float(loss.item())
        self.last_losses.append(loss_val)
        return loss_val

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------ persistence
    def save(self, path):
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "eps": self.eps,
            "update_count": self.update_count,
            "config": {
                "state_dim": self.state_dim, "n_actions": self.n_actions,
                "gamma": self.gamma, "batch_size": self.batch_size,
                "target_sync_every": self.target_sync_every,
            },
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.eps = ckpt.get("eps", self.eps_min)
        self.update_count = ckpt.get("update_count", 0)
        return ckpt

    @property
    def avg_loss(self):
        return float(np.mean(self.last_losses)) if self.last_losses else 0.0