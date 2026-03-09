import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from envs.marl_mac_env import MARLMacEnv
from algorithms.rl.gnn_marl import GNNMARL_QNetwork
from configs import config as params
from algorithms.mac.baseline import Config, Logger, simulate_tdma, simulate_csma_ca

def evaluate_marl_gnn(render=False):
    """
    Evaluates the trained Graph Neural Network over a traffic sweep
    and extracts throughput performance to compare against baselines.
    """
    env = MARLMacEnv()
    device = torch.device("cpu")
    
    # Load Model
    model_path = os.path.join(project_root, "results", "checkpoints", "gnn_marl_model.pth")
    if not os.path.exists(model_path):
        print("ERROR: Trained GNN MARL model missing. Run train_marl.py first.")
        return None
        
    policy_net = GNNMARL_QNetwork(node_in_dim=8, hidden_dim=64, num_actions=2).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()
    
    traffic_pps_list = np.linspace(params.SWEEP_MIN_PPS, params.SWEEP_MAX_PPS, params.SWEEP_STEPS).astype(int)
    results = []
    
    # Baseline comparison models
    print("=" * 60)
    print("Evaluating GNN-MARL Policy vs Traffic Rate...")
    print("=" * 60)
    
    for idx, pps in enumerate(traffic_pps_list):
        # Override the env simulation physics momentarily for testing the traffic parameter
        params.SWEEP_MAX_PPS = pps  
        
        obs, _ = env.reset()
        x, edge_index = env.get_global_graph_state()
        
        x_t = torch.tensor(x, dtype=torch.float32).to(device)
        edge_t = torch.tensor(edge_index, dtype=torch.long).to(device)
        
        actions = {}
        with torch.no_grad():
            q_vals = policy_net(x_t, edge_t) # Shape (N, 2)
            for i, agent in enumerate(env.agents):
                actions[agent] = q_vals[i].argmax().item()
                
        # Calculate resulting votes
        votes = [actions.get(a, 0) for a in env.agents]
        vote_tdma = sum(1 for v in votes if v == 0)
        vote_csma = sum(1 for v in votes if v == 1)
        chosen_mac = 1 if vote_csma > vote_tdma else 0
        
        # Test accurate performance using MAC engine
        cfg = Config(N=params.N, sim_time_s=params.SIM_TIME_S, QMAX=params.QMAX, tdma_guard_time_s=params.TDMA_GUARD_TIME_S)
        log_marl = Logger()
        
        if chosen_mac == 0:
            simulate_tdma(cfg, pps, log_marl)
        else:
            simulate_csma_ca(cfg, pps, log_marl)
            
        throughput_marl = log_marl.get_throughput_bps(params.SIM_TIME_S) / 1e6
        
        results.append({
            'Offered_Load_pps': pps,
            'MARL_GNN_Throughput_Mbps': throughput_marl,
            'TDMA_Votes': vote_tdma,
            'CSMA_Votes': vote_csma,
            'Majority_Action': "CSMA" if chosen_mac == 1 else "TDMA"
        })
        print(f"  Load: {pps:4d} pps | MARL Votes (TDMA: {vote_tdma:2d}, CSMA: {vote_csma:2d}) -> {results[-1]['Majority_Action']:4s} | Throughput: {throughput_marl:.2f} Mbps")

    df = pd.DataFrame(results)
    csv_path = os.path.join(project_root, "results", "RL_comparison", "csv", "marl_gnn_evaluation.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results exported to {csv_path}")
    
    # Plotting Output
    plt.figure(figsize=(10,6))
    plt.plot(df['Offered_Load_pps'], df['MARL_GNN_Throughput_Mbps'], label='MARL-GNN (PettingZoo)', marker='D', color='purple', linewidth=2)
    plt.xlabel('Offered Traffic Rate (packets/sec total)')
    plt.ylabel('Throughput (Mbps)')
    plt.title(f'Multi-Agent GNN Evaluation ({params.N} Nodes, Spatial Topology Graph)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(project_root, "results", "RL_comparison", "images", "marl_gnn_throughput.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    
if __name__ == '__main__':
    evaluate_marl_gnn(render=True)
