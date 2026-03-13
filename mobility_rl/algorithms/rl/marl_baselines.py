# marl_baselines.py
# MARL Baselines: Independent Q-Learning (IQL), Value Decomposition Network (VDN),
# and QMIX for Centralized Training with Decentralized Execution (CTDE).

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from collections import deque


# ======================================================================
# Shared Components
# ======================================================================
class MARLReplayBuffer:
    """Shared replay buffer for multi-agent transitions."""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs_all, actions, reward, next_obs_all, done):
        """
        obs_all: np.array (N, obs_dim)
        actions: list of int (N,)
        reward: float (shared cooperative reward)
        next_obs_all: np.array (N, obs_dim)
        done: bool
        """
        self.buffer.append((obs_all, actions, reward, next_obs_all, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, acts, rews, next_obs, dones = zip(*batch)
        return (
            np.array(obs),
            np.array(acts),
            np.array(rews, dtype=np.float32),
            np.array(next_obs),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class AgentQNetwork(nn.Module):
    """Simple MLP Q-network for a single agent."""
    def __init__(self, obs_dim, hidden_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, obs):
        return self.net(obs)


# ======================================================================
# IQL — Independent Q-Learning
# ======================================================================
class IQLAgent:
    """Each agent learns independently using its own Q-network. Simplest MARL baseline."""
    def __init__(self, n_agents, obs_dim, num_actions, hidden_dim=64, lr=1e-3, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=5000, target_update=1000,
                 replay_size=50000, batch_size=64, device="cpu"):
        self.n_agents = n_agents
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.steps_done = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update

        # Shared network (parameter sharing across agents)
        self.q_net = AgentQNetwork(obs_dim, hidden_dim, num_actions).to(self.device)
        self.target_net = AgentQNetwork(obs_dim, hidden_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = MARLReplayBuffer(replay_size)

    def get_epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.steps_done / self.eps_decay)

    def select_actions(self, obs_all):
        """obs_all: np.array (N, obs_dim). Returns list of actions."""
        epsilon = self.get_epsilon()
        self.steps_done += 1

        if random.random() < epsilon:
            return [random.randrange(self.num_actions) for _ in range(self.n_agents)]

        with torch.no_grad():
            obs_t = torch.tensor(obs_all, dtype=torch.float32).to(self.device)
            q_vals = self.q_net(obs_t)  # (N, num_actions)
            return q_vals.argmax(dim=1).cpu().tolist()

    def store(self, obs, actions, reward, next_obs, done):
        self.replay.push(obs, actions, reward, next_obs, done)

    def update(self):
        if len(self.replay) < self.batch_size:
            return 0.0

        obs_b, act_b, rew_b, nobs_b, done_b = self.replay.sample(self.batch_size)
        # obs_b: (B, N, obs_dim), act_b: (B, N), rew_b: (B,), done_b: (B,)

        B, N, D = obs_b.shape
        obs_flat = torch.tensor(obs_b.reshape(B * N, D), dtype=torch.float32).to(self.device)
        nobs_flat = torch.tensor(nobs_b.reshape(B * N, D), dtype=torch.float32).to(self.device)
        act_flat = torch.tensor(act_b.reshape(B * N), dtype=torch.long).to(self.device)
        rew_rep = torch.tensor(np.repeat(rew_b, N), dtype=torch.float32).to(self.device)
        done_rep = torch.tensor(np.repeat(done_b, N), dtype=torch.float32).to(self.device)

        q_vals = self.q_net(obs_flat).gather(1, act_flat.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(nobs_flat).max(1)[0]
            target = rew_rep + self.gamma * next_q * (1 - done_rep)

        loss = F.mse_loss(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())


# ======================================================================
# VDN — Value Decomposition Network
# ======================================================================
class VDNAgent:
    """
    Q_total = sum(Q_i). Additive decomposition.
    Same parameter-shared Q-network, but loss is on the summed Q.
    """
    def __init__(self, n_agents, obs_dim, num_actions, hidden_dim=64, lr=1e-3, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=5000, target_update=1000,
                 replay_size=50000, batch_size=64, device="cpu"):
        self.n_agents = n_agents
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.steps_done = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update

        self.q_net = AgentQNetwork(obs_dim, hidden_dim, num_actions).to(self.device)
        self.target_net = AgentQNetwork(obs_dim, hidden_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = MARLReplayBuffer(replay_size)

    def get_epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.steps_done / self.eps_decay)

    def select_actions(self, obs_all):
        epsilon = self.get_epsilon()
        self.steps_done += 1
        if random.random() < epsilon:
            return [random.randrange(self.num_actions) for _ in range(self.n_agents)]
        with torch.no_grad():
            obs_t = torch.tensor(obs_all, dtype=torch.float32).to(self.device)
            q_vals = self.q_net(obs_t)
            return q_vals.argmax(dim=1).cpu().tolist()

    def store(self, obs, actions, reward, next_obs, done):
        self.replay.push(obs, actions, reward, next_obs, done)

    def update(self):
        if len(self.replay) < self.batch_size:
            return 0.0

        obs_b, act_b, rew_b, nobs_b, done_b = self.replay.sample(self.batch_size)
        B, N, D = obs_b.shape

        obs_t = torch.tensor(obs_b, dtype=torch.float32).to(self.device)   # (B, N, D)
        nobs_t = torch.tensor(nobs_b, dtype=torch.float32).to(self.device)
        act_t = torch.tensor(act_b, dtype=torch.long).to(self.device)       # (B, N)
        rew_t = torch.tensor(rew_b, dtype=torch.float32).to(self.device)    # (B,)
        done_t = torch.tensor(done_b, dtype=torch.float32).to(self.device)

        # Per-agent Q values
        q_per_agent = []
        next_q_per_agent = []
        for i in range(N):
            qi = self.q_net(obs_t[:, i, :])                               # (B, num_actions)
            qi_selected = qi.gather(1, act_t[:, i].unsqueeze(1)).squeeze(1)  # (B,)
            q_per_agent.append(qi_selected)

            with torch.no_grad():
                nqi = self.target_net(nobs_t[:, i, :]).max(1)[0]
                next_q_per_agent.append(nqi)

        # VDN: Q_total = sum(Q_i)
        q_total = torch.stack(q_per_agent, dim=1).sum(dim=1)              # (B,)
        next_q_total = torch.stack(next_q_per_agent, dim=1).sum(dim=1)

        target = rew_t + self.gamma * next_q_total * (1 - done_t)
        loss = F.mse_loss(q_total, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())


# ======================================================================
# QMIX — QMIX with Hypernetwork Mixing
# ======================================================================
class QMIXMixingNetwork(nn.Module):
    """Monotonic mixing network conditioned on global state."""
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents = n_agents
        # Hypernetwork for weights (output must be non-negative)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents),
        )
        self.hyper_b1 = nn.Linear(state_dim, 1)

    def forward(self, agent_qs, state):
        """
        agent_qs: (B, N) — per-agent Q values
        state: (B, state_dim) — global state
        Returns: (B, 1) — Q_total
        """
        w1 = torch.abs(self.hyper_w1(state))     # (B, N), non-negative weights
        b1 = self.hyper_b1(state)                 # (B, 1)
        q_total = (agent_qs * w1).sum(dim=1, keepdim=True) + b1  # (B, 1)
        return q_total


class QMIXAgent:
    """QMIX: Monotonic value factorization with hypernetwork mixing."""
    def __init__(self, n_agents, obs_dim, num_actions, hidden_dim=64, embed_dim=32,
                 lr=1e-3, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=5000,
                 target_update=1000, replay_size=50000, batch_size=64, device="cpu"):
        self.n_agents = n_agents
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.steps_done = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update

        state_dim = n_agents * obs_dim  # Global state = concatenation of all obs

        self.q_net = AgentQNetwork(obs_dim, hidden_dim, num_actions).to(self.device)
        self.target_net = AgentQNetwork(obs_dim, hidden_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.mixer = QMIXMixingNetwork(n_agents, state_dim, embed_dim).to(self.device)
        self.target_mixer = QMIXMixingNetwork(n_agents, state_dim, embed_dim).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_mixer.eval()

        params = list(self.q_net.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.replay = MARLReplayBuffer(replay_size)

    def get_epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.steps_done / self.eps_decay)

    def select_actions(self, obs_all):
        epsilon = self.get_epsilon()
        self.steps_done += 1
        if random.random() < epsilon:
            return [random.randrange(self.num_actions) for _ in range(self.n_agents)]
        with torch.no_grad():
            obs_t = torch.tensor(obs_all, dtype=torch.float32).to(self.device)
            q_vals = self.q_net(obs_t)
            return q_vals.argmax(dim=1).cpu().tolist()

    def store(self, obs, actions, reward, next_obs, done):
        self.replay.push(obs, actions, reward, next_obs, done)

    def update(self):
        if len(self.replay) < self.batch_size:
            return 0.0

        obs_b, act_b, rew_b, nobs_b, done_b = self.replay.sample(self.batch_size)
        B, N, D = obs_b.shape

        obs_t = torch.tensor(obs_b, dtype=torch.float32).to(self.device)
        nobs_t = torch.tensor(nobs_b, dtype=torch.float32).to(self.device)
        act_t = torch.tensor(act_b, dtype=torch.long).to(self.device)
        rew_t = torch.tensor(rew_b, dtype=torch.float32).to(self.device)
        done_t = torch.tensor(done_b, dtype=torch.float32).to(self.device)

        # Global state = flattened observations
        state = obs_t.view(B, -1)       # (B, N*D)
        next_state = nobs_t.view(B, -1)

        # Per-agent Q values
        agent_qs = []
        next_agent_qs = []
        for i in range(N):
            qi = self.q_net(obs_t[:, i, :])
            qi_selected = qi.gather(1, act_t[:, i].unsqueeze(1)).squeeze(1)
            agent_qs.append(qi_selected)
            with torch.no_grad():
                nqi = self.target_net(nobs_t[:, i, :]).max(1)[0]
                next_agent_qs.append(nqi)

        agent_qs_t = torch.stack(agent_qs, dim=1)          # (B, N)
        next_agent_qs_t = torch.stack(next_agent_qs, dim=1)

        # Mix
        q_total = self.mixer(agent_qs_t, state).squeeze(1)  # (B,)
        with torch.no_grad():
            next_q_total = self.target_mixer(next_agent_qs_t, next_state).squeeze(1)
            target = rew_t + self.gamma * next_q_total * (1 - done_t)

        loss = F.mse_loss(q_total, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        return loss.item()

    def save(self, path):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'mixer': self.mixer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.mixer.load_state_dict(ckpt['mixer'])
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
