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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from configs import config as params
from envs.marl_mac_env import MARLMacEnv
from algorithms.rl.gnn_marl import MAGAT_D3QN_QNetwork

import multiprocessing

def worker_step(seed):
    env = MARLMacEnv(seed=seed)
    env.reset()
    x, edge_index = env.get_global_graph_state()
    # Execute a simple rapid random walk to generate diverse warmup buffers, 
    # or process entire episodes here natively depending on design.
    # To keep the PyTorch Gradient Graph synchronized, the cleanest parallelization
    # is collecting episodes asynchronously and updating globally. Let's do batch episode collection.
    
    local_memory = []
    ep_reward = 0
    steps_done = 0
    epsilon = 0.5 # Dynamic exploration per thread
    
    # We will just use random actions if detached from the main GPU purely to fill the buffer, 
    # OR we pass the network to each worker. Given torch multiprocessing complexity, 
    # let's just use a native Pool to collect environment steps using a copy of the CPU policy.
    # Since this introduces heavy complexity, let's keep it strictly process-based.
    pass

def train_gnn_marl():
    # Actually, the most robust way to parallelize PyTorch RL without massive SubprocVecEnv overhead 
    # that breaks custom graph states is Ray/RLlib or PyTorch's native `DistributedDataParallel`.
    # Since we want a simple drop-in fix for the CPU: Let's optimize the existing loop to just run N environments at once!
    
    num_envs = min(multiprocessing.cpu_count() - 2, 48) # Subproc 48 workers max
    envs = [MARLMacEnv(seed=params.SEED + i) for i in range(num_envs)]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_net = MAGAT_D3QN_QNetwork(node_in_dim=8, hidden_dim=64, num_actions=2, heads=4).to(device)
    target_net = MAGAT_D3QN_QNetwork(node_in_dim=8, hidden_dim=64, num_actions=2, heads=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = deque(maxlen=200000)
    batch_size = 128
    gamma = 0.99
    
    episodes = 5000
    steps_done = 0
    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 5000
    episode_rewards = []
    
    print("=" * 60)
    print(f"Starting Multi-Agent GNN Training... Device: {device} | Parallel Envs: {num_envs}")
    print("=" * 60)
    
    # Initialize all envs
    for env in envs:
        env.reset()
        
    for ep in range(episodes // num_envs): # Step by num_envs
        ep_rewards = [0] * num_envs
        active_envs = list(range(num_envs))
        
        # Loop until all parallel environments in this batch complete their max_steps
        while active_envs:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps_done / epsilon_decay)
            steps_done += 1
            
            for i in list(active_envs):
                env = envs[i]
                if not env.agents:
                    active_envs.remove(i)
                    continue
                    
                x, edge_index = env.get_global_graph_state()
                x_t = torch.tensor(x, dtype=torch.float32).to(device)
                edge_t = torch.tensor(edge_index, dtype=torch.long).to(device)
                
                actions = {}
                if random.random() < epsilon:
                    for agent in env.agents:
                        actions[agent] = env.action_space(agent).sample()
                else:
                    with torch.no_grad():
                        q_vals = policy_net(x_t, edge_t)
                        for idx, agent in enumerate(env.agents):
                            actions[agent] = q_vals[idx].argmax().item()
                            
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                
                reward_val = list(rewards.values())[0] if rewards else 0.0
                ep_rewards[i] += reward_val
                
                next_x, next_edge_index = env.get_global_graph_state()
                if env.agents:
                    act_list = [actions[a] for a in env.agents]
                    is_done = terminations[env.agents[0]]
                    memory.append((x, edge_index, act_list, reward_val, next_x, next_edge_index, is_done))
                    
            # Optimization Phase (batching across all collected step data)
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
                
        # Batch complete, reset environments
        for i in range(num_envs):
            if envs[i].agents == []:
                envs[i].reset()
                
        avg_batch_reward = sum(ep_rewards) / num_envs
        episode_rewards.extend(ep_rewards)
        
        print(f"Episodes {(ep+1)*num_envs:4d}/{episodes} | Batch Avg Shared Reward: {avg_batch_reward:7.2f} | Epsilon: {epsilon:.3f}")
        
    os.makedirs(os.path.join(project_root, "results", "checkpoints"), exist_ok=True)
    model_path = os.path.join(project_root, "results", "checkpoints", "gnn_marl_model.pth")
    torch.save(policy_net.state_dict(), model_path)
    print(f"✅ MARL-GNN Topology Mapping Complete. Model saved to {model_path}")
    
    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards, label="Shared Graph Reward")
    plt.title("MARL-GNN Training Over Dynamic MANET Topology")
    plt.xlabel("Episode")
    plt.ylabel("Reward (Throughput - Packet Loss)")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(project_root, "results", "gnn_marl_training.png"))
    
if __name__ == '__main__':
    train_gnn_marl()
