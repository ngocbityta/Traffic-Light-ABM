"""Rainbow DQN for the four-arm, six-channel GAMA spatial observation.

The distributional head uses Double DQN action selection, dueling value and
advantage streams, factorized NoisyNet, prioritized replay and n-step returns.
"""

import math
import random
import zlib
from collections import deque

import numpy as np
import torch
from torch import nn

try:
    from .three_dqn_agent import valid_mask
except ImportError:  # train.py also runs as a standalone script
    from .three_dqn_agent import valid_mask


ARCHITECTURE = "gama-rainbow-spatial-4x6x20x16-v4"
LEGACY_ARCHITECTURE = "gama-rainbow-spatial-4x6x20x16-r1-v2"
PREVIOUS_ARCHITECTURE = "gama-rainbow-spatial-4x6x20x16-v3"
GRID_SHAPE = (4, 6, 20, 16)
AUX_SIZE = 17
N_ACTIONS = 4
# State v5 (27/11/2026): 50 m camera grid, 7 channels all in [0, 1], and the
# 30 auxiliary features of train/logic/lvl_env.py (observation_v5).
ARCHITECTURE_V5 = "gama-rainbow-v5-4x7x10x12-aux30"
GRID_SHAPE_V5 = (4, 7, 10, 12)
AUX_SIZE_V5 = 30
STATE_SHAPES = {"v4": (GRID_SHAPE, AUX_SIZE), "v5": (GRID_SHAPE_V5, AUX_SIZE_V5)}


def pack_observation(observation, grid_shape=GRID_SHAPE, aux_size=None):
    grid = np.asarray(observation["grid"], dtype=np.float32)
    aux = np.asarray(observation["aux"], dtype=np.float32)
    grid_shape = tuple(grid_shape)
    if aux_size is None:
        aux_size = AUX_SIZE if grid_shape == GRID_SHAPE else AUX_SIZE_V5
    if grid.shape != grid_shape or aux.shape != (aux_size,):
        raise ValueError(f"Rainbow state must be grid {grid_shape} and aux {aux_size}")
    if not np.isfinite(grid).all() or not np.isfinite(aux).all():
        raise ValueError("Rainbow state contains non-finite values")
    if grid_shape == GRID_SHAPE:
        if (grid[:, :4].min() < -1e-5 or grid[:, :4].max() > 1.00001 or
                grid[:, 5].min() < -1e-5 or grid[:, 5].max() > 1.00001 or
                grid[:, 4].min() < -1.00001 or grid[:, 4].max() > 1.00001):
            raise ValueError("Rainbow state channels are outside their normalized ranges")
        encoded = np.rint(np.clip(grid, 0, 1) * 255).astype(np.uint8)
        encoded[:, 4] = np.rint(np.clip(grid[:, 4], -1, 1) * 127).astype(np.int8).view(np.uint8)
    else:
        if grid.min() < -1e-5 or grid.max() > 1.00001 or aux.min() < -1e-5 or aux.max() > 1.00001:
            raise ValueError("Rainbow v5 state is outside [0, 1]")
        encoded = np.rint(np.clip(grid, 0, 1) * 255).astype(np.uint8)
    blob = zlib.compress(encoded.tobytes(), 1)
    return blob, aux.astype(np.float16)


def unpack_observations(blobs, aux, grid_shape=GRID_SHAPE):
    grid_shape = tuple(grid_shape)
    raw = np.stack([np.frombuffer(zlib.decompress(blob), dtype=np.uint8).reshape(grid_shape)
                    for blob in blobs])
    grids = raw.astype(np.float32) / 255.0
    if grid_shape == GRID_SHAPE:
        grids[:, :, 4] = raw[:, :, 4].view(np.int8).astype(np.float32) / 127.0
    return grids, np.asarray(aux, dtype=np.float32)


class SumMinTree:
    def __init__(self, capacity):
        self.capacity = 1 << (capacity - 1).bit_length()
        self.sum = np.zeros(2 * self.capacity, dtype=np.float64)
        self.minimum = np.full(2 * self.capacity, np.inf, dtype=np.float64)

    def set(self, index, priority):
        node = index + self.capacity
        self.sum[node] = priority
        self.minimum[node] = priority
        while node > 1:
            node //= 2
            self.sum[node] = self.sum[2 * node] + self.sum[2 * node + 1]
            self.minimum[node] = min(self.minimum[2 * node], self.minimum[2 * node + 1])

    def find(self, mass):
        node = 1
        while node < self.capacity:
            left = 2 * node
            if mass < self.sum[left]:
                node = left
            else:
                mass -= self.sum[left]
                node = left + 1
        return node - self.capacity

    def find_many(self, masses):
        """``[self.find(m) for m in masses]`` descending all paths at once
        (same comparisons and subtractions, element by element)."""
        mass = np.array(masses, dtype=np.float64)
        node = np.ones(mass.shape, dtype=np.int64)
        while node[0] < self.capacity:  # every leaf has the same depth
            left = 2 * node
            left_sum = self.sum[left]
            go_left = mass < left_sum
            mass = np.where(go_left, mass, mass - left_sum)
            node = np.where(go_left, left, left + 1)
        return node - self.capacity

    def set_many(self, indices, priorities):
        """Sequential ``set`` for every pair (a repeated index keeps its last
        priority), then each touched ancestor recomputed once from its
        children: the same sums and minima as one ``set`` after another."""
        nodes = np.asarray(indices, dtype=np.int64) + self.capacity
        values = np.asarray(priorities, dtype=np.float64)
        self.sum[nodes] = values      # numpy fancy assignment: last write wins
        self.minimum[nodes] = values
        nodes = np.unique(nodes // 2)
        while nodes.size and nodes[0] >= 1:
            left = 2 * nodes
            self.sum[nodes] = self.sum[left] + self.sum[left + 1]
            self.minimum[nodes] = np.minimum(self.minimum[left], self.minimum[left + 1])
            if nodes[0] == 1:
                break
            nodes = np.unique(nodes // 2)


class PrioritizedReplay:
    def __init__(self, capacity, alpha=.5, seed=None, aux_size=AUX_SIZE, grid_shape=GRID_SHAPE):
        if capacity < 1 or not 0 <= alpha <= 1:
            raise ValueError("Invalid prioritized replay capacity or alpha")
        self.grid_shape = tuple(grid_shape)
        self.aux_size = int(aux_size)
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.rng = np.random.default_rng(seed)
        self.tree = SumMinTree(self.capacity)
        self.states = [None] * self.capacity
        self.next_states = [None] * self.capacity
        self.aux = np.zeros((self.capacity, self.aux_size), dtype=np.float16)
        self.next_aux = np.zeros((self.capacity, self.aux_size), dtype=np.float16)
        self.actions = np.zeros(self.capacity, dtype=np.uint8)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.discounts = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.bool_)
        self.next_masks = np.zeros((self.capacity, N_ACTIONS), dtype=np.bool_)
        self.max_priority = 1.0
        self.pos = self.size = 0

    def push(self, state, action, reward, next_state, done, discount, next_mask):
        if action not in range(N_ACTIONS) or not np.isfinite(reward) or not 0 <= discount <= 1:
            raise ValueError("Invalid Rainbow transition")
        p = self.pos
        # A 13-feature legacy agent stores the 17 features it is given minus
        # the four pressure values, as act() does.
        state, next_state = self._fit(state), self._fit(next_state)
        self.states[p], self.aux[p] = pack_observation(state, self.grid_shape, self.aux_size)
        self.next_states[p], self.next_aux[p] = pack_observation(
            next_state, self.grid_shape, self.aux_size)
        self.actions[p] = action
        self.rewards[p] = reward
        self.dones[p] = done
        self.discounts[p] = discount
        self.next_masks[p] = valid_mask(next_mask)
        self.tree.set(p, self.max_priority ** self.alpha)
        self.pos = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _fit(self, observation):
        aux = np.asarray(observation["aux"])
        if self.aux_size == 13 and aux.shape == (AUX_SIZE,):
            return {**observation, "aux": np.concatenate((aux[:9], aux[13:17]))}
        return observation

    def sample(self, batch_size, beta):
        total = self.tree.sum[1]
        if self.size < batch_size or total <= 0 or not 0 <= beta <= 1:
            raise ValueError("Invalid Rainbow replay sample")
        segment = total / batch_size
        # One draw per segment, in segment order: the same stream as calling
        # ``rng.random()`` batch_size times.
        draws = self.rng.random(batch_size)
        indices = self.tree.find_many((np.arange(batch_size) + draws) * segment)
        probabilities = self.tree.sum[self.tree.capacity + indices] / total
        minimum = self.tree.minimum[1] / total
        weights = (self.size * probabilities) ** (-beta)
        weights /= (self.size * minimum) ** (-beta)
        states, aux = unpack_observations([self.states[i] for i in indices], self.aux[indices],
                                          self.grid_shape)
        next_states, next_aux = unpack_observations(
            [self.next_states[i] for i in indices], self.next_aux[indices], self.grid_shape)
        return (indices, weights.astype(np.float32), states, aux,
                self.actions[indices].astype(np.int64), self.rewards[indices],
                next_states, next_aux, self.dones[indices].astype(np.float32),
                self.discounts[indices], self.next_masks[indices])

    def update_priorities(self, indices, losses):
        priorities = np.abs(np.asarray(losses, dtype=np.float64)) + 1e-6
        if not np.isfinite(priorities).all():
            raise ValueError("Non-finite Rainbow replay priority")
        self.max_priority = max(self.max_priority, float(priorities.max()))
        # Python's float power (libm), as before: numpy's vectorised power
        # can differ in the last bit.
        alpha = self.alpha
        self.tree.set_many(indices, [p ** alpha for p in priorities.tolist()])

    def __len__(self):
        return self.size

    def snapshot(self):
        n = self.size
        return {"capacity": self.capacity, "size": n, "pos": self.pos,
                "max_priority": self.max_priority, "rng": self.rng.bit_generator.state,
                "states": self.states[:n], "next_states": self.next_states[:n],
                "tree_sum": self.tree.sum.copy(), "tree_min": self.tree.minimum.copy(),
                **{name: getattr(self, name)[:n].copy() for name in
                   ("aux", "next_aux", "actions", "rewards", "discounts", "dones", "next_masks")}}

    def restore(self, state):
        if state["capacity"] != self.capacity or not 0 <= state["size"] <= self.capacity:
            raise ValueError("Rainbow replay capacity mismatch")
        self.size, self.pos = state["size"], state["pos"]
        self.max_priority = state["max_priority"]
        self.rng.bit_generator.state = state["rng"]
        self.states[:self.size] = state["states"]
        self.next_states[:self.size] = state["next_states"]
        self.tree.sum[:] = state["tree_sum"]
        self.tree.minimum[:] = state["tree_min"]
        for name in ("aux", "next_aux", "actions", "rewards", "discounts", "dones", "next_masks"):
            getattr(self, name)[:self.size] = state[name]


class NoisyLinear(nn.Module):
    def __init__(self, inputs, outputs, sigma=.5):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(outputs, inputs))
        self.weight_sigma = nn.Parameter(torch.empty(outputs, inputs))
        self.bias_mu = nn.Parameter(torch.empty(outputs))
        self.bias_sigma = nn.Parameter(torch.empty(outputs))
        self.register_buffer("weight_epsilon", torch.zeros(outputs, inputs))
        self.register_buffer("bias_epsilon", torch.zeros(outputs))
        bound = 1 / math.sqrt(inputs)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, sigma / math.sqrt(inputs))
        nn.init.constant_(self.bias_sigma, sigma / math.sqrt(outputs))
        self.reset_noise()

    @staticmethod
    def _noise(n, device):
        x = torch.randn(n, device=device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        x = self._noise(self.weight_mu.shape[1], self.weight_mu.device)
        y = self._noise(self.weight_mu.shape[0], self.weight_mu.device)
        self.weight_epsilon.copy_(y.outer(x))
        self.bias_epsilon.copy_(y)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return nn.functional.linear(x, weight, bias)


class RainbowNetwork(nn.Module):
    def __init__(self, atoms=51, aux_size=AUX_SIZE, grid_shape=GRID_SHAPE):
        super().__init__()
        self.atoms = atoms
        self.grid_shape = tuple(grid_shape)
        channels, rows, cols = self.grid_shape[1:]
        if self.grid_shape == GRID_SHAPE:
            self.arm_encoder = nn.Sequential(
                nn.Conv2d(6, 32, 3, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(), nn.Flatten(),
                nn.Linear(64 * 4 * 3, 128), nn.ReLU())
        else:
            # 10 x 12 cells: a stride-1 layer first keeps the 5 m resolution.
            r1, c1 = rows - 2, cols - 2
            r2, c2 = (r1 - 3) // 2 + 1, (c1 - 3) // 2 + 1
            self.arm_encoder = nn.Sequential(
                nn.Conv2d(channels, 32, 3, stride=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(), nn.Flatten(),
                nn.Linear(64 * r2 * c2, 128), nn.ReLU())
        self.shared = nn.Sequential(nn.Linear(4 * 128 + aux_size, 256), nn.ReLU())
        self.value_hidden = NoisyLinear(256, 128)
        self.value_out = NoisyLinear(128, atoms)
        self.adv_hidden = NoisyLinear(256, 128)
        self.adv_out = NoisyLinear(128, N_ACTIONS * atoms)

    def reset_noise(self):
        for layer in (self.value_hidden, self.value_out, self.adv_hidden, self.adv_out):
            layer.reset_noise()

    def forward(self, grid, aux):
        batch = grid.shape[0]
        arms = self.arm_encoder(grid.reshape(batch * 4, *self.grid_shape[1:])).reshape(batch, 4 * 128)
        shared = self.shared(torch.cat((arms, aux), dim=1))
        value = self.value_out(torch.relu(self.value_hidden(shared))).reshape(batch, 1, self.atoms)
        advantage = self.adv_out(torch.relu(self.adv_hidden(shared))).reshape(batch, N_ACTIONS, self.atoms)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class RainbowAgent:
    def __init__(self, lr=1e-4, gamma=.99, buffer_size=100_000, warmup=20_000,
                 batch_size=32, tau=1e-3, eps_start=1.0, eps_min=.001,
                 eps_decay_steps=2_000_000, seed=None, device=None, reward_mode="r1",
                 n_step=3, per_alpha=.5, per_beta_start=.4, per_beta_steps=1_000_000,
                 atoms=51, v_min=-2000.0, v_max=0.0, target_update=2000,
                 r1_ref=50.0, reward_settings=None, aux_size=None, state_version=None):
        if reward_mode not in ("r1", "r2", "r3", "r4"):
            raise ValueError("Spatial Rainbow requires R1-R4")
        if (not all(math.isfinite(x) for x in (gamma, lr, v_min, v_max, r1_ref)) or
                not 0 < gamma <= 1 or not 0 < lr or n_step < 1 or atoms < 2 or v_min >= v_max):
            raise ValueError("Invalid Rainbow learning parameters")
        if warmup < 0 or batch_size < 1 or buffer_size < batch_size or per_beta_steps < 1:
            raise ValueError("Invalid Rainbow replay parameters")
        if not 0 <= per_beta_start <= 1 or target_update < 1 or r1_ref <= 0:
            raise ValueError("Invalid Rainbow target or R1 parameters")
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
        self.reward_settings = dict(reward_settings or {"r1_ref": r1_ref})
        # The state version travels in reward_settings (like the queue
        # scale), so actors, validation and evaluate.py rebuild the same net.
        if state_version is None:
            state_version = self.reward_settings.get("state_version", "v4")
        if state_version not in STATE_SHAPES:
            raise ValueError(f"Unknown Rainbow state version: {state_version}")
        self.state_version = state_version
        self.grid_shape, default_aux = STATE_SHAPES[state_version]
        if aux_size is None:
            aux_size = default_aux
        if state_version == "v5":
            if aux_size != AUX_SIZE_V5:
                raise ValueError("Rainbow v5 uses 30 auxiliary features")
            self.reward_settings["state_version"] = "v5"
            self.architecture = ARCHITECTURE_V5
        else:
            if aux_size not in (13, AUX_SIZE):
                raise ValueError("Unsupported Rainbow auxiliary state size")
            self.architecture = ARCHITECTURE
        self.aux_size = aux_size
        self.n_step = int(n_step)
        self.batch_size = int(batch_size)
        self.warmup = int(warmup)
        self.per_beta_start = float(per_beta_start)
        self.per_beta_steps = int(per_beta_steps)
        self.target_update = int(target_update)
        self.atoms = int(atoms)
        self.v_min, self.v_max = float(v_min), float(v_max)
        self.r1_ref = float(r1_ref)
        self.support = torch.linspace(v_min, v_max, atoms, device=self.device)
        self.delta_z = (v_max - v_min) / (atoms - 1)
        self.online_net = RainbowNetwork(atoms, aux_size, self.grid_shape).to(self.device)
        self.target_net = RainbowNetwork(atoms, aux_size, self.grid_shape).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = PrioritizedReplay(buffer_size, per_alpha, seed, aux_size, self.grid_shape)
        self.pending = deque()
        self.decisions = self.update_count = 0
        self.eps = 0.0  # NoisyNet replaces epsilon exploration after replay warm-up.
        self.last_losses = deque(maxlen=100)

    def act(self, state, greedy=False, action_mask=None, force_random=False):
        mask = valid_mask(action_mask)
        if force_random:
            return int(self.rng.choice(np.flatnonzero(mask).tolist()))
        was_training = self.online_net.training
        self.online_net.train(not greedy)
        if not greedy:
            self.online_net.reset_noise()
        with torch.no_grad():
            grid = torch.as_tensor(np.asarray(state["grid"])[None], device=self.device)
            aux_values = np.asarray(state["aux"])
            if self.aux_size == 13 and aux_values.shape == (AUX_SIZE,):
                aux_values = np.concatenate((aux_values[:9], aux_values[13:17]))
            aux = torch.as_tensor(aux_values[None], device=self.device)
            probs = self.online_net(grid, aux).softmax(dim=-1)
            values = (probs * self.support).sum(dim=-1)
            values.masked_fill_(~torch.as_tensor(mask[None], device=self.device), -torch.inf)
            action = int(values.argmax(dim=1).item())
        self.online_net.train(was_training)
        return action

    def decay_epsilon(self):
        self.decisions += 1

    def _emit_first(self):
        horizon = min(self.n_step, len(self.pending))
        state, action = self.pending[0][:2]
        cumulative = 0.0
        discount = 1.0
        final = None
        for i in range(horizon):
            item = self.pending[i]
            cumulative += discount * item[2]
            discount *= item[5]
            final = item
            if item[4]:
                break
        self.buffer.push(state, action, cumulative, final[3], final[4], discount, final[6])
        self.pending.popleft()

    def store(self, state, action, reward, next_state, done, seconds=None, next_mask=None):
        if seconds is None or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Rainbow requires the actual positive decision duration")
        self.pending.append((state, action, float(reward), next_state, bool(done),
                             self.gamma ** (seconds / 5.0), valid_mask(next_mask)))
        if done:
            return self.finish_episode()
        elif len(self.pending) >= self.n_step:
            self._emit_first()
            return 1
        return 0

    def finish_episode(self):
        # A rollout quota can end at a valid decision state. Short tail returns
        # bootstrap from that state; a horizon-truncated final action is terminal.
        emitted = 0
        while self.pending:
            self._emit_first()
            emitted += 1
        return emitted

    def can_update(self):
        return len(self.buffer) >= max(self.warmup, self.batch_size)

    def update(self):
        beta = min(1.0, self.per_beta_start +
                   (1.0 - self.per_beta_start) * self.update_count / self.per_beta_steps)
        (indices, weights, states, aux, actions, rewards, next_states,
         next_aux, dones, discounts, masks) = self.buffer.sample(self.batch_size, beta)
        device = self.device
        s = torch.as_tensor(states, device=device)
        a = torch.as_tensor(aux, device=device)
        s2 = torch.as_tensor(next_states, device=device)
        a2 = torch.as_tensor(next_aux, device=device)
        actions = torch.as_tensor(actions, device=device)
        rewards = torch.as_tensor(rewards, device=device)
        dones = torch.as_tensor(dones, device=device)
        discounts = torch.as_tensor(discounts, device=device)
        masks = torch.as_tensor(masks, device=device)
        weights = torch.as_tensor(weights, device=device)
        self.online_net.reset_noise()
        logits = self.online_net(s, a)[torch.arange(self.batch_size, device=device), actions]
        with torch.no_grad():
            online_probs = self.online_net(s2, a2).softmax(dim=-1)
            q_next = (online_probs * self.support).sum(dim=-1).masked_fill(~masks, -torch.inf)
            best = q_next.argmax(dim=1)
            self.target_net.train()
            self.target_net.reset_noise()
            target_probs = self.target_net(s2, a2).softmax(dim=-1)[
                torch.arange(self.batch_size, device=device), best]
            self.target_net.eval()
            tz = (rewards[:, None] + (1 - dones[:, None]) *
                  discounts[:, None] * self.support[None]).clamp(self.v_min, self.v_max)
            b = (tz - self.v_min) / self.delta_z
            lower = b.floor().long().clamp(0, self.atoms - 1)
            upper = b.ceil().long().clamp(0, self.atoms - 1)
            low_weight = upper.float() - b
            low_weight[lower == upper] = 1.0
            projection = torch.zeros_like(target_probs)
            projection.scatter_add_(1, lower, target_probs * low_weight)
            projection.scatter_add_(1, upper, target_probs * (b - lower.float()))
        losses = -(projection * logits.log_softmax(dim=-1)).sum(dim=-1)
        loss = (losses * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        self.buffer.update_priorities(indices, losses.detach().cpu().numpy())
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        value = float(loss.item())
        self.last_losses.append(value)
        return value

    @property
    def avg_loss(self):
        return float(np.mean(self.last_losses)) if self.last_losses else 0.0

    def save(self, path):
        torch.save({"architecture": self.architecture, "reward_mode": self.reward_mode,
                    "state_version": self.state_version,
                    "reward_settings": self.reward_settings,
                    "online_net": self.online_net.state_dict(),
                    "target_net": self.target_net.state_dict(),
                    "decisions": self.decisions, "update_count": self.update_count,
                    "gamma": self.gamma, "n_step": self.n_step,
                    "atoms": self.atoms, "v_min": self.v_min, "v_max": self.v_max,
                    "r1_ref": self.r1_ref}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        architecture = ckpt.get("architecture")
        if architecture == ARCHITECTURE_V5:
            if self.state_version != "v5":
                raise ValueError("Rainbow v5 checkpoint needs state_version='v5'")
        elif architecture in (ARCHITECTURE, LEGACY_ARCHITECTURE, PREVIOUS_ARCHITECTURE):
            if self.state_version != "v4":
                raise ValueError("Rainbow v4 checkpoint loaded into a v5 agent")
            expected_aux = AUX_SIZE if architecture == ARCHITECTURE else 13
            if self.aux_size != expected_aux:
                raise ValueError("Rainbow checkpoint auxiliary state size mismatch")
        else:
            raise ValueError("Rainbow checkpoint architecture mismatch")
        if (ckpt["atoms"], ckpt["v_min"], ckpt["v_max"]) != (self.atoms, self.v_min, self.v_max):
            raise ValueError("Rainbow C51 support mismatch")
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.decisions = ckpt["decisions"]
        self.update_count = ckpt["update_count"]
        self.gamma = ckpt["gamma"]
        self.r1_ref = ckpt["r1_ref"]
        self.reward_mode = ckpt["reward_mode"]
        self.reward_settings = dict(ckpt.get("reward_settings", {"r1_ref": self.r1_ref}))
        return ckpt

    def training_state(self):
        if self.pending:
            raise ValueError("Flush Rainbow n-step tail before checkpointing")
        device_rng = None
        if self.device.type == "mps":
            device_rng = torch.mps.get_rng_state().cpu()
        elif self.device.type == "cuda":
            device_rng = [state.cpu() for state in torch.cuda.get_rng_state_all()]
        return {"architecture": self.architecture,
                "reward_mode": self.reward_mode, "reward_settings": self.reward_settings,
                "policy": {"online": self.online_net.state_dict(),
                           "target": self.target_net.state_dict()},
                "optimizer": self.optimizer.state_dict(),
                "decisions": self.decisions, "update_count": self.update_count,
                "losses": list(self.last_losses), "action_rng": self.rng.getstate(),
                "torch_rng": torch.get_rng_state(), "device_rng": device_rng,
                "replay": self.buffer.snapshot()}

    def restore_training(self, state):
        if state["architecture"] != self.architecture:
            raise ValueError("Rainbow training checkpoint architecture mismatch")
        self.online_net.load_state_dict(state["policy"]["online"])
        self.target_net.load_state_dict(state["policy"]["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.decisions, self.update_count = state["decisions"], state["update_count"]
        self.last_losses.clear()
        self.last_losses.extend(state["losses"])
        self.rng.setstate(state["action_rng"])
        self.reward_mode = state.get("reward_mode", self.reward_mode)
        self.reward_settings = dict(state.get("reward_settings", self.reward_settings))
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "mps" and state.get("device_rng") is not None:
            torch.mps.set_rng_state(state["device_rng"].cpu())
        elif self.device.type == "cuda" and state.get("device_rng") is not None:
            torch.cuda.set_rng_state_all([item.cpu() for item in state["device_rng"]])
        self.buffer.restore(state["replay"])
