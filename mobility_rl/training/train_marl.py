import os
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs import config as params
from configs.marl_config import MARLConfig
from envs.marl_mac_env import MARLMacEnv
from algorithms.rl.gnn_marl import MAGAT_D3QN_QNetwork

def train_marl_agent(algo_name, agent_class, save_filename, extra_kwargs=None):
    """Generic training loop for independent MARL baselines (IQL, VDN, QMIX)."""
    env = MARLMacEnv()
    N = params.N
    obs_dim = MARLConfig.OBS_DIM
    num_actions = MARLConfig.NUM_ACTIONS

    kwargs = dict(
        n_agents=N, obs_dim=obs_dim, num_actions=num_actions,
        hidden_dim=MARLConfig.HIDDEN_DIM, lr=MARLConfig.LR,
        gamma=MARLConfig.GAMMA, eps_start=MARLConfig.EPSILON_START,
        eps_end=MARLConfig.EPSILON_END, eps_decay=MARLConfig.EPSILON_DECAY,
        target_update=MARLConfig.TARGET_UPDATE_FREQ,
        replay_size=MARLConfig.REPLAY_SIZE, batch_size=MARLConfig.BATCH_SIZE,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    agent = agent_class(**kwargs)
    episodes = MARLConfig.EPISODES
    ep_rewards = []
    
    print("=" * 60)
    print(f"Starting {algo_name} Training...")
    print("=" * 60)

    pbar = tqdm(range(episodes), desc=f"Training {algo_name}", unit="ep")

    for ep in pbar:
        obs_dict, _ = env.reset()
        obs_all = np.stack([obs_dict[a] for a in env.possible_agents])

        ep_reward = 0.0
        while env.agents:
            actions_list = agent.select_actions(obs_all)
            actions_dict = {a: actions_list[i] for i, a in enumerate(env.agents)}

            next_obs_dict, rewards, terminations, truncations, infos = env.step(actions_dict)
            reward_val = list(rewards.values())[0] if rewards else 0.0

            next_obs_all = np.stack([next_obs_dict[a] for a in env.possible_agents])
            done = terminations[env.possible_agents[0]] if env.possible_agents[0] in terminations else True

            agent.store(obs_all, actions_list, reward_val, next_obs_all, done)
            agent.update()

            obs_all = next_obs_all
            ep_reward += reward_val

        ep_rewards.append(ep_reward)
        if ep % 10 == 0:
            pbar.set_postfix({"Avg Reward (last 50)": f"{np.mean(ep_rewards[-50:]):.2f}"})

    cp_dir = MARLConfig.get_checkpoint_dir()
    save_path = os.path.join(cp_dir, save_filename)
    agent.save(save_path)
    print(f"✅ {algo_name} checkpoint saved: {save_path}")

def train_gnn_marl():
    env = MARLMacEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    obs_dim = MARLConfig.OBS_DIM
    hidden_dim = MARLConfig.HIDDEN_DIM
    num_actions = MARLConfig.NUM_ACTIONS
    heads = MARLConfig.GNN_HEADS
    
    policy_net = MAGAT_D3QN_QNetwork(node_in_dim=obs_dim, hidden_dim=hidden_dim, num_actions=num_actions, heads=heads).to(device)
    target_net = MAGAT_D3QN_QNetwork(node_in_dim=obs_dim, hidden_dim=hidden_dim, num_actions=num_actions, heads=heads).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=MARLConfig.LR)
    memory = deque(maxlen=MARLConfig.REPLAY_SIZE)
    batch_size = MARLConfig.BATCH_SIZE
    gamma = MARLConfig.GAMMA
    epsilon_start = MARLConfig.EPSILON_START
    epsilon_end = MARLConfig.EPSILON_END
    epsilon_decay = MARLConfig.EPSILON_DECAY
    
    episodes = MARLConfig.EPISODES
    steps_done = 0
    episode_rewards = []
    
    print("=" * 60)
    print(f"Starting Multi-Agent GNN Training... Device: {device}")
    print("=" * 60)
    
    pbar = tqdm(range(episodes), desc="Training MARL-GNN", unit="ep")
    for ep in pbar:
        obs, _ = env.reset()
        x, edge_index = env.get_global_graph_state()
        ep_reward = 0
        
        while env.agents:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps_done / epsilon_decay)
            steps_done += 1
            
            x_t = torch.tensor(x, dtype=torch.float32).to(device)
            edge_t = torch.tensor(edge_index, dtype=torch.long).to(device)
            
            actions = {}
            if random.random() < epsilon:
                for agent in env.agents:
                    actions[agent] = env.action_space(agent).sample()
            else:
                with torch.no_grad():
                    q_vals = policy_net(x_t, edge_t)
                    for i, agent in enumerate(env.agents):
                        actions[agent] = q_vals[i].argmax().item()
                        
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            reward_val = list(rewards.values())[0] if rewards else 0.0
            ep_reward += reward_val
            
            next_x, next_edge_index = env.get_global_graph_state()
            
            if env.agents:
                act_list = [actions[a] for a in env.agents]
                is_done = terminations[env.agents[0]]
                memory.append((x, edge_index, act_list, reward_val, next_x, next_edge_index, is_done))
                
            x = next_x
            edge_index = next_edge_index
            
            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                loss = 0
                for (b_x, b_edge, b_act, b_r, b_nx, b_nedge, b_d) in batch:
                    b_xt = torch.tensor(b_x, dtype=torch.float32).to(device)
                    b_edget = torch.tensor(b_edge, dtype=torch.long).to(device)
                    b_nxt = torch.tensor(b_nx, dtype=torch.float32).to(device)
                    b_nedget = torch.tensor(b_nedge, dtype=torch.long).to(device)
                    
                    q_all = policy_net(b_xt, b_edget)
                    q_a = q_all[range(params.N), b_act] 
                    with torch.no_grad():
                        q_next = target_net(b_nxt, b_nedget).max(1)[0]
                        target = b_r + gamma * q_next * (1 - int(b_d))
                    loss += F.mse_loss(q_a, target)
                    
                loss = loss / batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if steps_done % 1000 == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
        episode_rewards.append(ep_reward)
        if ep % 5 == 0:
            pbar.set_postfix({"Avg Reward": f"{ep_reward:.2f}", "Epsilon": f"{epsilon:.3f}"})
        
    model_path = os.path.join(MARLConfig.get_checkpoint_dir(), "gnn_marl_model.pth")
    torch.save(policy_net.state_dict(), model_path)
    print(f"✅ MARL-GNN Topology Mapping Complete. Model saved to {model_path}")

if __name__ == '__main__':
    from algorithms.rl.marl_baselines import IQLAgent, VDNAgent, QMIXAgent
    import argparse
    parser = argparse.ArgumentParser(description="Standalone MARL Trainer")
    parser.add_argument('--algo', type=str, default='all', choices=['all', 'iql', 'vdn', 'qmix', 'magat_d3qn'])
    args = parser.parse_args()
    
    if args.algo in ['all', 'iql']:
        train_marl_agent("IQL", IQLAgent, "marl_iql_model.pth")
    if args.algo in ['all', 'vdn']:
        train_marl_agent("VDN", VDNAgent, "marl_vdn_model.pth")
    if args.algo in ['all', 'qmix']:
        train_marl_agent("QMIX", QMIXAgent, "marl_qmix_model.pth", extra_kwargs={"embed_dim": MARLConfig.QMIX_EMBED_DIM})
    if args.algo in ['all', 'magat_d3qn']:
        train_gnn_marl()
