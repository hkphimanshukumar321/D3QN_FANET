# mappo_agent.py
# Multi-Agent Proximal Policy Optimization (MAPPO) with Centralized Training and
# Decentralized Execution (CTDE).
#
# Actor: Maps local observation to action logits. (Parameter Shared)
# Critic: Maps global observation to state value. (Centralized)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np

class SharedActorNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, local_obs):
        # local_obs: [batch_size, obs_dim]
        logits = self.net(local_obs)
        return logits


class CentralizedCriticNetwork(nn.Module):
    def __init__(self, n_agents, obs_dim, hidden_dim):
        super().__init__()
        # Global state is formed by concatenating local observations of all agents.
        global_obs_dim = n_agents * obs_dim
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_obs):
        # global_obs: [batch_size, n_agents * obs_dim]
        value = self.net(global_obs)
        return value


class MAPPOAgent:
    def __init__(
        self,
        n_agents,
        obs_dim,
        num_actions,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=10,
        batch_size=64,
        device="cpu",
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.actor = SharedActorNetwork(obs_dim, hidden_dim, num_actions).to(self.device)
        self.critic = CentralizedCriticNetwork(n_agents, obs_dim, hidden_dim).to(self.device)

        # Single optimizer for both
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        self.reset_buffer()

    def reset_buffer(self):
        """Clear the rollout buffer"""
        self.buffer = {
            "obs": [],
            "global_obs": [],
            "actions": [],
            "logprobs": [],
            "rewards": [],
            "values": [],
            "dones": [],
            "alive_masks": []
        }

    def get_epsilon(self):
        return 0.0 # PPO is an on-policy stochastic method, no epsilon-greedy

    @torch.no_grad()
    def select_actions(self, obs_all, alive_mask=None, store=False):
        """
        Select actions for all agents. 
        If store=True, acts for rollout collection (computes value + log_prob).
        If store=False, acts for evaluation / inference (greedy or sample).
        """
        if alive_mask is None:
            alive_mask = np.ones(self.n_agents, dtype=np.bool_)

        obs_t = torch.tensor(obs_all, dtype=torch.float32, device=self.device) # [N, obs_dim]
        
        logits = self.actor(obs_t)
        dist = Categorical(logits=logits)
        
        if not store:  # Evaluation: greedy execution
            actions = logits.argmax(dim=-1).cpu().numpy()
            return [int(actions[i]) if alive_mask[i] else 0 for i in range(self.n_agents)]

        # Collection: sample execution
        actions = dist.sample()
        logprobs = dist.log_prob(actions)
        
        # Centralized Critic
        global_obs_t = obs_t.view(-1).unsqueeze(0) # [1, n_agents * obs_dim]
        # In a batch context, global_obs_t is [1, N*obs_dim]
        value = self.critic(global_obs_t).squeeze() # scalar
        
        return actions.cpu().numpy(), logprobs.cpu().numpy(), value.item(), global_obs_t.squeeze().cpu().numpy()

    def store(self, obs_all, global_obs, actions, logprobs, values, rewards, done, alive_mask=None):
        if alive_mask is None:
            alive_mask = np.ones(self.n_agents, dtype=np.bool_)
            
        self.buffer["obs"].append(obs_all)
        self.buffer["global_obs"].append(global_obs)
        self.buffer["actions"].append(actions)
        self.buffer["logprobs"].append(logprobs)
        self.buffer["values"].append(values)
        self.buffer["rewards"].append(rewards)
        self.buffer["dones"].append(done)
        self.buffer["alive_masks"].append(alive_mask)

    def update(self):
        """
        Performs PPO multi-epoch updates based on collected rollout buffer.
        """
        bsz = len(self.buffer["rewards"])
        if bsz == 0:
            return 0.0

        # Convert to tensors
        obs_batch = torch.tensor(np.array(self.buffer["obs"]), dtype=torch.float32, device=self.device) # [T, N, obs_dim]
        global_obs_batch = torch.tensor(np.array(self.buffer["global_obs"]), dtype=torch.float32, device=self.device) # [T, N*obs_dim]
        actions_batch = torch.tensor(np.array(self.buffer["actions"]), dtype=torch.long, device=self.device) # [T, N]
        logprobs_batch = torch.tensor(np.array(self.buffer["logprobs"]), dtype=torch.float32, device=self.device) # [T, N]
        values_batch = torch.tensor(np.array(self.buffer["values"]), dtype=torch.float32, device=self.device) # [T]
        rewards_batch = torch.tensor(np.array(self.buffer["rewards"]), dtype=torch.float32, device=self.device) # [T, N]
        dones_batch = torch.tensor(np.array(self.buffer["dones"]), dtype=torch.float32, device=self.device) # [T]
        alive_masks_batch = torch.tensor(np.array(self.buffer["alive_masks"]), dtype=torch.bool, device=self.device) # [T, N]
        
        # Calculate Team Reward
        team_rewards = (rewards_batch * alive_masks_batch).sum(dim=1) # [T]

        # Calculate Advantages & Returns using GAE
        advantages = torch.zeros_like(team_rewards).to(self.device)
        returns = torch.zeros_like(team_rewards).to(self.device)
        
        # Determine last value for GAE
        with torch.no_grad():
            last_global_obs = global_obs_batch[-1].unsqueeze(0)
            last_val = self.critic(last_global_obs).squeeze(-1) # [1]
            next_val = last_val
            
        gae = 0.0
        for step in reversed(range(bsz)):
            if step == bsz - 1:
                next_non_terminal = 1.0 - dones_batch[step]
                next_values = next_val
            else:
                next_non_terminal = 1.0 - dones_batch[step]
                next_values = values_batch[step + 1]
                
            delta = team_rewards[step] + self.gamma * next_values * next_non_terminal - values_batch[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[step] = gae
            returns[step] = gae + values_batch[step]

        # NaN guard: if advantages or returns contain NaN, skip update
        if torch.isnan(advantages).any() or torch.isnan(returns).any():
            self.reset_buffer()
            return 0.0

        # Flatten the tensors for batch updates
        # Critic uses Team aspects (flatten T only)
        # Actor uses individual agent aspects (flatten T*N)
        
        b_global_obs = global_obs_batch # [T, N*obs_dim]
        b_returns = returns # [T]
        b_advantages = advantages # [T]

        b_obs = obs_batch.view(-1, self.obs_dim) # [T*N, obs_dim]
        b_actions = actions_batch.view(-1) # [T*N]
        b_logprobs = logprobs_batch.view(-1) # [T*N]
        b_alive_masks = alive_masks_batch.view(-1) # [T*N]

        # Expand advantages so each agent in step T gets the same team advantage
        b_expanded_advantages = b_advantages.unsqueeze(1).repeat(1, self.n_agents).view(-1) # [T*N]
        
        # Normalize advantages
        b_expanded_advantages = (b_expanded_advantages - b_expanded_advantages.mean()) / (b_expanded_advantages.std() + 1e-8)

        total_loss_sum = 0.0
        
        # PPO Multi-Epoch Update
        dataset_size = bsz
        indices = np.arange(dataset_size)
        
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                mb_indices = indices[start:end]
                
                # --- CRITIC UPDATE ---
                mb_global_obs = b_global_obs[mb_indices]
                mb_returns = b_returns[mb_indices]
                
                new_values = self.critic(mb_global_obs).squeeze(-1)
                value_loss = F.mse_loss(new_values, mb_returns)
                
                # --- ACTOR UPDATE ---
                # To get corresponding actor batches, we multiply mb_indices by n_agents and get all agents for those steps
                # Alternatively, map indices directly:
                mb_actor_indices = [i * self.n_agents + a for i in mb_indices for a in range(self.n_agents)]
                
                mb_obs = b_obs[mb_actor_indices]
                mb_actions = b_actions[mb_actor_indices]
                mb_logprobs = b_logprobs[mb_actor_indices]
                mb_alive_masks = b_alive_masks[mb_actor_indices]
                mb_adv = b_expanded_advantages[mb_actor_indices]
                
                # Filter out dead agents
                if not mb_alive_masks.any():
                    # If all chosen agents are dead, just update critic. Rare.
                    loss = value_loss * self.value_loss_coef
                else:
                    live_obs = mb_obs[mb_alive_masks]
                    live_actions = mb_actions[mb_alive_masks]
                    live_old_logprobs = mb_logprobs[mb_alive_masks]
                    live_adv = mb_adv[mb_alive_masks]
                    
                    logits = self.actor(live_obs)
                    dist = Categorical(logits=logits)
                    new_logprobs = dist.log_prob(live_actions)
                    entropy = dist.entropy().mean()
                    
                    ratio = torch.exp(new_logprobs - live_old_logprobs)
                    surr1 = ratio * live_adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * live_adv
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # NaN guard: skip this minibatch if loss is NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), max_norm=0.5)
                self.optimizer.step()
                
                total_loss_sum += loss.item()

        self.reset_buffer()
        return total_loss_sum / (self.ppo_epochs * max((dataset_size // self.batch_size), 1))

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
