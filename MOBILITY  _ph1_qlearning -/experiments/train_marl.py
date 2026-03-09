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

from envs.marl_mac_env import MARLMacEnv
from algorithms.rl.gnn_marl import MAGAT_D3QN_QNetwork

def train_gnn_marl():
    env = MARLMacEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Node features: 8 -> Hidden: 64 -> Actions: 2 -> Heads: 4
    policy_net = MAGAT_D3QN_QNetwork(node_in_dim=8, hidden_dim=64, num_actions=2, heads=4).to(device)
    target_net = MAGAT_D3QN_QNetwork(node_in_dim=8, hidden_dim=64, num_actions=2, heads=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    
    memory = deque(maxlen=50000)
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 5000
    
    episodes = 5000
    steps_done = 0
    
    episode_rewards = []
    
    print("=" * 60)
    print(f"Starting Multi-Agent GNN Training... Device: {device}")
    print("=" * 60)
    
    for ep in range(episodes):
        obs, _ = env.reset()
        x, edge_index = env.get_global_graph_state()
        
        ep_reward = 0
        
        while env.agents:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                math.exp(-1. * steps_done / epsilon_decay)
            steps_done += 1
            
            x_t = torch.tensor(x, dtype=torch.float32).to(device)
            edge_t = torch.tensor(edge_index, dtype=torch.long).to(device)
            
            actions = {}
            if random.random() < epsilon:
                for agent in env.agents:
                    actions[agent] = env.action_space(agent).sample()
            else:
                with torch.no_grad():
                    q_vals = policy_net(x_t, edge_t) # Shape: (N, 2)
                    for i, agent in enumerate(env.agents):
                        actions[agent] = q_vals[i].argmax().item()
                        
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Cooperative MARL uses a shared global reward
            reward_val = list(rewards.values())[0] if rewards else 0.0
            ep_reward += reward_val
            
            next_x, next_edge_index = env.get_global_graph_state()
            
            # Parameter-Sharing Replay Buffer Storage
            # We store the topological graph state, not just 1D arrays
            if env.agents:
                act_list = [actions[a] for a in env.agents]
                is_done = terminations[env.agents[0]]
                memory.append((x, edge_index, act_list, reward_val, next_x, next_edge_index, is_done))
                
            x = next_x
            edge_index = next_edge_index
            
            # Optimization Phase
            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                
                loss = 0
                for (b_x, b_edge, b_act, b_r, b_nx, b_nedge, b_d) in batch:
                    b_xt = torch.tensor(b_x, dtype=torch.float32).to(device)
                    b_edget = torch.tensor(b_edge, dtype=torch.long).to(device)
                    b_nxt = torch.tensor(b_nx, dtype=torch.float32).to(device)
                    b_nedget = torch.tensor(b_nedge, dtype=torch.long).to(device)
                    
                    # Q(s, a) for all N agents simultaneously 
                    q_all = policy_net(b_xt, b_edget)
                    q_a = q_all[range(env.N), b_act] # Gather Q-values for the precisely chosen actions
                    
                    # Target Q(s', a')
                    with torch.no_grad():
                        q_next = target_net(b_nxt, b_nedget).max(1)[0]
                        target = b_r + gamma * q_next * (1 - int(b_d))
                        
                    # Calculate MSE Loss over the entire graph structure natively
                    loss += F.mse_loss(q_a, target)
                    
                loss = loss / batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if steps_done % 1000 == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
        episode_rewards.append(ep_reward)
        if ep % 5 == 0:
            print(f"Episode {ep:3d}/{episodes} | Avg Shared Reward: {ep_reward:7.2f} | Epsilon: {epsilon:.3f}")
        
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
