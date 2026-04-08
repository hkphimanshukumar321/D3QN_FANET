# marl_baselines.py
# MARL baselines for padded cluster-head agents.

from __future__ import annotations

import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MARLReplayBuffer:
    """Replay buffer storing local reward vectors and alive masks."""

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs_all, actions, rewards, next_obs_all, done, alive_mask=None):
        if alive_mask is None:
            alive_mask = np.ones(len(actions), dtype=np.bool_)
        self.buffer.append((obs_all, actions, rewards, next_obs_all, done, alive_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, acts, rews, next_obs, dones, masks = zip(*batch)
        return (
            np.array(obs),
            np.array(acts),
            np.array(rews, dtype=np.float32),
            np.array(next_obs),
            np.array(dones, dtype=np.float32),
            np.array(masks, dtype=np.bool_),
        )

    def __len__(self):
        return len(self.buffer)


class AgentQNetwork(nn.Module):
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


class BaseValueDecompositionAgent:
    def __init__(
        self,
        n_agents,
        obs_dim,
        num_actions,
        hidden_dim=64,
        lr=1e-3,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=5000,
        target_update=1000,
        replay_size=50000,
        batch_size=64,
        device="cpu",
    ):
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

    def select_actions(self, obs_all, alive_mask=None):
        epsilon = self.get_epsilon()
        self.steps_done += 1
        if alive_mask is None:
            alive_mask = np.ones(self.n_agents, dtype=np.bool_)

        if random.random() < epsilon:
            actions = [random.randrange(self.num_actions) if alive_mask[i] else 0 for i in range(self.n_agents)]
            return actions

        with torch.no_grad():
            obs_t = torch.tensor(obs_all, dtype=torch.float32, device=self.device)
            q_vals = self.q_net(obs_t)
            actions = q_vals.argmax(dim=1).cpu().tolist()
            return [actions[i] if alive_mask[i] else 0 for i in range(self.n_agents)]

    def store(self, obs, actions, rewards, next_obs, done, alive_mask=None):
        self.replay.push(obs, actions, rewards, next_obs, done, alive_mask)

    def _maybe_sync_target(self):
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


class IQLAgent(BaseValueDecompositionAgent):
    """Independent Q-learning with parameter sharing and local reward vectors."""

    def update(self):
        if len(self.replay) < self.batch_size:
            return 0.0

        obs_b, act_b, rew_b, nobs_b, done_b, mask_b = self.replay.sample(self.batch_size)
        bsz, n_agents, obs_dim = obs_b.shape

        obs_flat = torch.tensor(obs_b.reshape(bsz * n_agents, obs_dim), dtype=torch.float32, device=self.device)
        nobs_flat = torch.tensor(nobs_b.reshape(bsz * n_agents, obs_dim), dtype=torch.float32, device=self.device)
        act_flat = torch.tensor(act_b.reshape(bsz * n_agents), dtype=torch.long, device=self.device)
        rew_flat = torch.tensor(rew_b.reshape(bsz * n_agents), dtype=torch.float32, device=self.device)
        done_rep = torch.tensor(np.repeat(done_b, n_agents), dtype=torch.float32, device=self.device)
        mask_flat = torch.tensor(mask_b.reshape(bsz * n_agents), dtype=torch.float32, device=self.device)

        q_vals = self.q_net(obs_flat).gather(1, act_flat.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(nobs_flat).max(1)[0]
            target = rew_flat + self.gamma * next_q * (1 - done_rep)

        active_idx = mask_flat > 0.5
        if not torch.any(active_idx):
            return 0.0
        loss = F.mse_loss(q_vals[active_idx], target[active_idx])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._maybe_sync_target()
        return float(loss.item())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())


class VDNAgent(BaseValueDecompositionAgent):
    """Value Decomposition Network using summed local rewards as team targets."""

    def update(self):
        if len(self.replay) < self.batch_size:
            return 0.0

        obs_b, act_b, rew_b, nobs_b, done_b, mask_b = self.replay.sample(self.batch_size)
        bsz, n_agents, _ = obs_b.shape

        obs_t = torch.tensor(obs_b, dtype=torch.float32, device=self.device)
        nobs_t = torch.tensor(nobs_b, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(act_b, dtype=torch.long, device=self.device)
        done_t = torch.tensor(done_b, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask_b, dtype=torch.float32, device=self.device)
        rew_t = torch.tensor(rew_b, dtype=torch.float32, device=self.device)
        team_reward = (rew_t * mask_t).sum(dim=1)

        q_per_agent = []
        next_q_per_agent = []
        for i in range(n_agents):
            qi = self.q_net(obs_t[:, i, :])
            qi_sel = qi.gather(1, act_t[:, i].unsqueeze(1)).squeeze(1)
            q_per_agent.append(qi_sel)
            with torch.no_grad():
                nqi = self.target_net(nobs_t[:, i, :]).max(1)[0]
                next_q_per_agent.append(nqi)

        q_total = (torch.stack(q_per_agent, dim=1) * mask_t).sum(dim=1)
        next_q_total = (torch.stack(next_q_per_agent, dim=1) * mask_t).sum(dim=1)
        target = team_reward + self.gamma * next_q_total * (1 - done_t)

        loss = F.mse_loss(q_total, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._maybe_sync_target()
        return float(loss.item())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())


class QMIXMixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs, state, alive_mask=None):
        batch_size = agent_qs.size(0)
        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.n_agents, self.embed_dim)
        if alive_mask is not None:
            w1 = w1 * alive_mask.unsqueeze(-1)
        b1 = self.hyper_b1(state).unsqueeze(1)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        w2 = torch.abs(self.hyper_w2(state)).unsqueeze(2)
        b2 = self.hyper_b2(state)
        return torch.bmm(hidden, w2).squeeze(2) + b2


class QMIXAgent(BaseValueDecompositionAgent):
    def __init__(self, *args, embed_dim=32, **kwargs):
        super().__init__(*args, **kwargs)
        state_dim = self.n_agents * self.q_net.net[0].in_features
        self.mixer = QMIXMixingNetwork(self.n_agents, state_dim, embed_dim).to(self.device)
        self.target_mixer = QMIXMixingNetwork(self.n_agents, state_dim, embed_dim).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_mixer.eval()
        self.optimizer = torch.optim.Adam(
            list(self.q_net.parameters()) + list(self.mixer.parameters()),
            lr=kwargs.get("lr", 1e-3),
        )

    def update(self):
        if len(self.replay) < self.batch_size:
            return 0.0

        obs_b, act_b, rew_b, nobs_b, done_b, mask_b = self.replay.sample(self.batch_size)
        bsz, n_agents, _ = obs_b.shape

        obs_t = torch.tensor(obs_b, dtype=torch.float32, device=self.device)
        nobs_t = torch.tensor(nobs_b, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(act_b, dtype=torch.long, device=self.device)
        rew_t = torch.tensor(rew_b, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done_b, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask_b, dtype=torch.float32, device=self.device)

        state = obs_t.view(bsz, -1)
        next_state = nobs_t.view(bsz, -1)
        team_reward = (rew_t * mask_t).sum(dim=1)

        q_agents = []
        next_q_agents = []
        for i in range(n_agents):
            qi = self.q_net(obs_t[:, i, :])
            q_agents.append(qi.gather(1, act_t[:, i].unsqueeze(1)).squeeze(1))
            with torch.no_grad():
                nqi = self.target_net(nobs_t[:, i, :]).max(1)[0]
                next_q_agents.append(nqi)

        q_agent_tensor = torch.stack(q_agents, dim=1) * mask_t
        next_q_agent_tensor = torch.stack(next_q_agents, dim=1) * mask_t
        q_total = self.mixer(q_agent_tensor, state, alive_mask=mask_t).squeeze(1)
        with torch.no_grad():
            next_q_total = self.target_mixer(next_q_agent_tensor, next_state, alive_mask=mask_t).squeeze(1)
            target = team_reward + self.gamma * next_q_total * (1 - done_t)

        loss = F.mse_loss(q_total, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save({"q_net": self.q_net.state_dict(), "mixer": self.mixer.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.mixer.load_state_dict(ckpt["mixer"])
        self.target_mixer.load_state_dict(self.mixer.state_dict())
