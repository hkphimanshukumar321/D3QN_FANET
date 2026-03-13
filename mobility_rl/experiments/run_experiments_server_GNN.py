import os
import sys
import json
import datetime
import subprocess
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing
import psutil
from tqdm import tqdm

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from algorithms.mac.baseline import Config, Logger
from algorithms.mac.baseline import simulate_slotted_aloha, simulate_tdma, simulate_csma_ca
from configs import config as params
from configs.sarl_config import RLConfig
from configs.marl_config import MARLConfig
from envs.marl_mac_env import MARLMacEnv
from experiments.run_experiments import generate_baseline_plots

# =====================================================================
# Constants
# =====================================================================
MAC_NAMES = {0: "TDMA", 1: "CSMA_CA"}
MARL_TRAINING_EPISODES = MARLConfig.EPISODES

# =====================================================================
# Utility Functions
# =====================================================================
def get_git_commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('utf-8').strip()
    except Exception:
        return "Unknown"

def make_result_dirs(N, phy_rate_bps, QMAX, rts_cts_enabled, ack_enabled):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_name = f"trial_MARL_{timestamp}"
    
    phy_mbps = int(phy_rate_bps / 1e6)
    config_folder_name = f"N{N}_PHY{phy_mbps}M_Q{QMAX}"
    if rts_cts_enabled and ack_enabled:
        config_folder_name += "_RTSCTS_ACK_enabled"
    elif ack_enabled:
        config_folder_name += "_ACK_enabled"
    
    base_dir = os.path.join(project_root, "results", config_folder_name, trial_name)
    os.makedirs(os.path.join(base_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    
    marl_dir = os.path.join(base_dir, "RL_comparison", "MARL")
    os.makedirs(os.path.join(marl_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(marl_dir, "csv"), exist_ok=True)
    
    return base_dir, marl_dir, trial_name

def estimate_ram_footprint():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)

def get_model_file_size_kb(filepath):
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / 1024.0
    return 0.0


# =====================================================================
# Phase 1: Baseline MAC Simulations
# =====================================================================
def run_baseline_simulations(cfg, traffic_pps_list, sim_time_s, N, seed, log_print):
    results = []
    for idx, pps in enumerate(tqdm(traffic_pps_list, desc="Baseline MAC Sweep", unit="load")):
        log_print(f"  [Baseline] Testing load {pps} pps ({idx+1}/{len(traffic_pps_list)})")
        
        log_tdma = Logger(load_pps=pps)
        cfg.seed = seed + idx
        simulate_tdma(cfg, pps, log_tdma)
        
        log_csma = Logger(load_pps=pps)
        cfg.seed = seed + idx
        simulate_csma_ca(cfg, pps, log_csma)
        
        results.append({
            'Offered_Load_pps': pps,
            'TDMA_Throughput_Mbps': log_tdma.get_throughput_bps(sim_time_s) / 1e6,
            'CSMA_Throughput_Mbps': log_csma.get_throughput_bps(sim_time_s) / 1e6,
            'TDMA_Delay_s': log_tdma.get_avg_end_to_end_delay_s(),
            'CSMA_Delay_s': log_csma.get_avg_end_to_end_delay_s(),
            'TDMA_Drops': log_tdma.pkts_dropped_qfull + log_tdma.pkts_dropped_mac,
            'CSMA_Drops': log_csma.pkts_dropped_qfull + log_csma.pkts_dropped_mac,
        })
    return results


# =====================================================================
# Phase 2: MARL Training (Multiprocessed)
# =====================================================================
def train_single_marl_model(kwargs):
    """Worker function to train a single MARL agent in an isolated process."""
    algo_name = kwargs['algo_name']
    episodes = kwargs['episodes']
    cp_dir = kwargs['cp_dir']
    marl_out_dir = kwargs['marl_out_dir']
    seed = kwargs['seed']
    
    import pandas as pd
    from envs.marl_mac_env import MARLMacEnv
    from configs.marl_config import MARLConfig
    from configs import config as params
    
    pid = os.getpid()
    print(f"  [Worker {pid}] Starting MARL training: {algo_name.upper()}...")
    
    env = MARLMacEnv(seed=seed)
    N = params.N
    obs_dim = MARLConfig.OBS_DIM
    num_actions = MARLConfig.NUM_ACTIONS
    
    common_kwargs = dict(
        n_agents=N, obs_dim=obs_dim, num_actions=num_actions,
        hidden_dim=MARLConfig.HIDDEN_DIM, lr=MARLConfig.LR,
        gamma=MARLConfig.GAMMA, eps_start=MARLConfig.EPSILON_START,
        eps_end=MARLConfig.EPSILON_END, eps_decay=MARLConfig.EPSILON_DECAY,
        target_update=MARLConfig.TARGET_UPDATE_FREQ,
        replay_size=MARLConfig.REPLAY_SIZE, batch_size=MARLConfig.BATCH_SIZE,
    )
    
    if algo_name == 'iql':
        from algorithms.rl.marl_baselines import IQLAgent
        save_path = os.path.join(cp_dir, "marl_iql_model.pth")
        if os.path.exists(save_path):
            return f"{algo_name.upper()} skipped (checkpoint exists)"
        agent = IQLAgent(**common_kwargs)
        
    elif algo_name == 'vdn':
        from algorithms.rl.marl_baselines import VDNAgent
        save_path = os.path.join(cp_dir, "marl_vdn_model.pth")
        if os.path.exists(save_path):
            return f"{algo_name.upper()} skipped (checkpoint exists)"
        agent = VDNAgent(**common_kwargs)
        
    elif algo_name == 'qmix':
        from algorithms.rl.marl_baselines import QMIXAgent
        save_path = os.path.join(cp_dir, "marl_qmix_model.pth")
        if os.path.exists(save_path):
            return f"{algo_name.upper()} skipped (checkpoint exists)"
        common_kwargs['embed_dim'] = MARLConfig.QMIX_EMBED_DIM
        agent = QMIXAgent(**common_kwargs)
        
    elif algo_name == 'magat_d3qn':
        import torch
        from algorithms.rl.gnn_marl import MAGAT_D3QN_QNetwork
        save_path = os.path.join(cp_dir, "gnn_marl_model.pth")
        if os.path.exists(save_path):
            return f"{algo_name.upper()} skipped (checkpoint exists)"
        
        # GNN training (separate logic, uses graph state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_net = MAGAT_D3QN_QNetwork(node_in_dim=obs_dim, hidden_dim=64, num_actions=2, heads=4).to(device)
        target_net = MAGAT_D3QN_QNetwork(node_in_dim=obs_dim, hidden_dim=64, num_actions=2, heads=4).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
        from collections import deque
        import random, math
        memory = deque(maxlen=50000)
        batch_size = 64
        gamma = 0.99
        
        ep_rewards = []

        for ep in tqdm(range(episodes), desc=f"[{pid}] MAGAT-D3QN", unit="ep"):
            obs, _ = env.reset()
            x, edge_index = env.get_global_graph_state()
            ep_reward = 0
            
            epsilon = 0.05 + 0.95 * math.exp(-float(ep) / 500.0)
            
            while env.agents:
                x_t = torch.tensor(x, dtype=torch.float32).to(device)
                edge_t = torch.tensor(edge_index, dtype=torch.long).to(device)
                
                actions = {}
                if random.random() < epsilon:
                    for agent_name in env.agents:
                        actions[agent_name] = env.action_space(agent_name).sample()
                else:
                    with torch.no_grad():
                        q_vals = policy_net(x_t, edge_t)
                        for i, agent_name in enumerate(env.agents):
                            actions[agent_name] = q_vals[i].argmax().item()
                
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                reward_val = list(rewards.values())[0] if rewards else 0.0
                ep_reward += reward_val
                
                next_x, next_edge_index = env.get_global_graph_state()
                
                if env.agents:
                    act_list = [actions[a] for a in env.agents]
                    is_done = terminations[env.agents[0]]
                    memory.append((x, edge_index, act_list, reward_val, next_x, next_edge_index, is_done))
                
                x, edge_index = next_x, next_edge_index
                
                # Batch update
                if len(memory) > batch_size:
                    batch = random.sample(memory, batch_size)
                    loss = torch.tensor(0.0, device=device)
                    for (b_x, b_edge, b_act, b_r, b_nx, b_nedge, b_d) in batch:
                        b_xt = torch.tensor(b_x, dtype=torch.float32).to(device)
                        b_et = torch.tensor(b_edge, dtype=torch.long).to(device)
                        b_nxt = torch.tensor(b_nx, dtype=torch.float32).to(device)
                        b_net = torch.tensor(b_nedge, dtype=torch.long).to(device)
                        
                        q_all = policy_net(b_xt, b_et)
                        q_a = q_all[range(len(b_act)), b_act]
                        with torch.no_grad():
                            q_next = target_net(b_nxt, b_net).max(1)[0]
                            target = b_r + gamma * q_next * (1 - int(b_d))
                        loss += torch.nn.functional.mse_loss(q_a, target)
                    
                    loss = loss / batch_size
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                if ep % 100 == 0 and env.agents:
                    target_net.load_state_dict(policy_net.state_dict())
            
            ep_rewards.append(ep_reward)
        
        torch.save(policy_net.state_dict(), save_path)
        
        if ep_rewards:
            pd.DataFrame({"episode": range(len(ep_rewards)), "reward": ep_rewards}).to_csv(
                os.path.join(marl_out_dir, "csv", "magat_d3qn_training.csv"), index=False)
        
        return f"MAGAT-D3QN trained."
    else:
        return f"Unknown MARL algorithm: {algo_name}"
    
    # --- Common IQL/VDN/QMIX training loop ---
    ep_rewards = []
    for ep in tqdm(range(episodes), desc=f"[{pid}] {algo_name.upper()}", unit="ep"):
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
    
    agent.save(save_path)
    
    if ep_rewards:
        pd.DataFrame({"episode": range(len(ep_rewards)), "reward": ep_rewards}).to_csv(
            os.path.join(marl_out_dir, "csv", f"{algo_name}_training.csv"), index=False)
    
    return f"{algo_name.upper()} trained."


def execute_marl_multiprocess_training(marl_out_dir, episodes, log_print):
    """Launch MARL training across multiple CPU cores."""
    cp_dir = MARLConfig.get_checkpoint_dir()
    os.makedirs(cp_dir, exist_ok=True)
    
    tasks = []
    if getattr(params, "RUN_MARL_IQL", True): tasks.append('iql')
    if getattr(params, "RUN_MARL_VDN", True): tasks.append('vdn')
    if getattr(params, "RUN_MARL_QMIX", True): tasks.append('qmix')
    if getattr(params, "RUN_MARL_GNN", True): tasks.append('magat_d3qn')
    
    kwargs_list = [
        {'algo_name': t, 'episodes': episodes, 'cp_dir': cp_dir,
         'marl_out_dir': marl_out_dir, 'seed': params.SEED}
        for t in tasks
    ]
    
    cpu_cores = os.cpu_count() or 1
    workers = min(cpu_cores, len(tasks))
    log_print(f"  [MARL Train] Detected {cpu_cores} CPU cores — using {workers} workers for {len(tasks)} MARL algorithms")
    
    with multiprocessing.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(train_single_marl_model, kwargs_list):
            log_print(f"    --> {result}")
    
    log_print("  [MARL Train] All parallel MARL training complete.")


# =====================================================================
# Phase 3: MARL Evaluation Sweep
# =====================================================================
def evaluate_marl_models(traffic_pps_list, marl_out_dir, log_print):
    """Load trained MARL models and evaluate across traffic sweep."""
    import torch
    from envs.marl_mac_env import MARLMacEnv
    from algorithms.rl.marl_baselines import IQLAgent, VDNAgent, QMIXAgent
    
    cp_dir = MARLConfig.get_checkpoint_dir()
    N = params.N
    obs_dim = MARLConfig.OBS_DIM
    
    env = MARLMacEnv()
    
    marl_models = {}
    
    # Load IQL
    iql_path = os.path.join(cp_dir, "marl_iql_model.pth")
    if os.path.exists(iql_path):
        iql = IQLAgent(N, obs_dim, 2)
        iql.load(iql_path)
        marl_models["IQL"] = iql
    
    # Load VDN
    vdn_path = os.path.join(cp_dir, "marl_vdn_model.pth")
    if os.path.exists(vdn_path):
        vdn = VDNAgent(N, obs_dim, 2)
        vdn.load(vdn_path)
        marl_models["VDN"] = vdn
    
    # Load QMIX
    qmix_path = os.path.join(cp_dir, "marl_qmix_model.pth")
    if os.path.exists(qmix_path):
        qmix = QMIXAgent(N, obs_dim, 2)
        qmix.load(qmix_path)
        marl_models["QMIX"] = qmix
    
    # Load MAGAT-D3QN (GNN)
    gnn_path = os.path.join(cp_dir, "gnn_marl_model.pth")
    if os.path.exists(gnn_path):
        from algorithms.rl.gnn_marl import MAGAT_D3QN_QNetwork
        device = torch.device("cpu")
        gnn = MAGAT_D3QN_QNetwork(node_in_dim=obs_dim, hidden_dim=64, num_actions=2, heads=4).to(device)
        gnn.load_state_dict(torch.load(gnn_path, map_location=device))
        gnn.eval()
        marl_models["MAGAT-D3QN"] = gnn
    
    if not marl_models:
        log_print("  [Eval] No MARL models found. Skipping evaluation.")
        return pd.DataFrame()
    
    log_print(f"  [Eval] Loaded {len(marl_models)} MARL models: {list(marl_models.keys())}")
    
    results = []
    for pps in tqdm(traffic_pps_list, desc="MARL Evaluation Sweep", unit="load"):
        for model_name, model in marl_models.items():
            params.SWEEP_MAX_PPS = pps
            obs_dict, _ = env.reset()
            
            if model_name == "MAGAT-D3QN":
                x, edge_index = env.get_global_graph_state()
                x_t = torch.tensor(x, dtype=torch.float32)
                edge_t = torch.tensor(edge_index, dtype=torch.long)
                with torch.no_grad():
                    q_vals = model(x_t, edge_t)
                    actions_dict = {a: q_vals[i].argmax().item() for i, a in enumerate(env.agents)}
            else:
                obs_all = np.stack([obs_dict[a] for a in env.possible_agents])
                model.steps_done = model.eps_decay * 10  # Force greedy
                actions_list = model.select_actions(obs_all)
                actions_dict = {a: actions_list[i] for i, a in enumerate(env.agents)}
            
            # Run actual MAC sim
            votes = list(actions_dict.values())
            chosen_mac = 1 if sum(votes) > len(votes) / 2 else 0
            
            cfg = Config(N=N, sim_time_s=params.SIM_TIME_S, QMAX=params.QMAX,
                         tdma_guard_time_s=params.TDMA_GUARD_TIME_S)
            log = Logger()
            if chosen_mac == 0:
                simulate_tdma(cfg, pps, log)
            else:
                simulate_csma_ca(cfg, pps, log)
            
            results.append({
                'Model': model_name,
                'Offered_Load_pps': pps,
                'Throughput_Mbps': log.get_throughput_bps(params.SIM_TIME_S) / 1e6,
                'Delay_ms': log.get_avg_end_to_end_delay_s() * 1000,
                'Drops': log.pkts_dropped_qfull + log.pkts_dropped_mac,
                'Chosen_MAC': "TDMA" if chosen_mac == 0 else "CSMA",
            })
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(marl_out_dir, "csv", "marl_eval_sweep.csv")
    df.to_csv(csv_path, index=False)
    log_print(f"  [Eval] Results saved: {csv_path}")
    
    # Comparison plot
    plt.figure(figsize=(12, 6))
    for model_name in df['Model'].unique():
        sub = df[df['Model'] == model_name]
        plt.plot(sub['Offered_Load_pps'], sub['Throughput_Mbps'], marker='o', label=model_name, linewidth=2)
    plt.xlabel('Offered Traffic (PPS)')
    plt.ylabel('Throughput (Mbps)')
    plt.title('MARL Baselines — Throughput vs Traffic Load')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(marl_out_dir, "images", "marl_throughput_comparison.png"), dpi=150)
    plt.close()
    
    return df


# =====================================================================
# Main
# =====================================================================
def run_experiment_server():
    N = params.N
    sim_time_s = params.SIM_TIME_S
    phy_rate_bps = params.PHY_RATE_BPS
    QMAX = params.QMAX
    rts_cts_enabled = params.RTS_CTS_ENABLED
    ack_enabled = params.ACK_ENABLED
    
    out_dir, marl_dir, trial_name = make_result_dirs(N, phy_rate_bps, QMAX, rts_cts_enabled, ack_enabled)
    log_file_path = os.path.join(out_dir, "logs", "run.log")
    
    def log_print(msg):
        print(msg)
        with open(log_file_path, "a") as lf:
            lf.write(msg + "\n")
    
    log_print("=" * 60)
    log_print(" MARL SERVER-GRADE EXPERIMENT RUNNER")
    log_print(f" (Multiprocessing + tqdm + CTDE Baselines)")
    log_print("=" * 60)

    import torch
    num_cpus = os.cpu_count()
    has_gpu = torch.cuda.is_available()
    log_print(f"Hardware: {num_cpus} CPU Cores | GPU: {has_gpu}")

    cfg = Config()
    traffic_pps_list = np.linspace(params.SWEEP_MIN_PPS, params.SWEEP_MAX_PPS, params.SWEEP_STEPS).astype(int)
    
    # [Step 1] Baseline MAC
    log_print(f"\n--- Step 1: Baseline MAC Simulations ---")
    results = run_baseline_simulations(cfg, traffic_pps_list, sim_time_s, N, params.SEED, log_print)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "csv", "baseline_results.csv"), index=False)
    
    # [Step 2] MARL Training (Multiprocessed)
    log_print(f"\n--- Step 2: MARL Parallel Training ---")
    execute_marl_multiprocess_training(marl_dir, MARL_TRAINING_EPISODES, log_print)
    
    # [Step 3] MARL Evaluation
    log_print(f"\n--- Step 3: MARL Evaluation Sweep ---")
    eval_df = evaluate_marl_models(traffic_pps_list, marl_dir, log_print)
    
    # [Step 4] Ablation Study
    run_marl_on_sarl_ablation(traffic_pps_list, marl_dir, log_print)
    
    # [Step 5] Combined Metrics Plotting
    generate_combined_sarl_marl_plot(out_dir, log_print)
    
    log_print("\nMARL Experiment Runner Completed Successfully.")

def generate_combined_sarl_marl_plot(base_dir, log_print):
    """
    Looks for both SARL and MARL telemetry in the results directory.
    If both exist, generates a master combined Throughput vs Load curve graph.
    """
    log_print("\n--- Step 5: Generating Combined SARL vs MARL Plots ---")
    sarl_csv = os.path.join(base_dir, "RL_comparison", "SARL", "csv", "rl_eval_sweep.csv")
    marl_csv = os.path.join(base_dir, "RL_comparison", "MARL", "csv", "marl_eval_sweep.csv")
    
    if not os.path.exists(sarl_csv) or not os.path.exists(marl_csv):
        log_print("  [Plot] Missing logs. Both servers must finish their sweeps to plot combined graphs.")
        return
        
    sarl_df = pd.read_csv(sarl_csv)
    marl_df = pd.read_csv(marl_csv)
    
    combined_img_dir = os.path.join(base_dir, "RL_comparison", "images")
    os.makedirs(combined_img_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    if 'Model' in sarl_df.columns:
        for m in sarl_df['Model'].unique():
            sub = sarl_df[sarl_df['Model'] == m]
            plt.plot(sub['Offered_Load_pps'], sub['Throughput_Mbps'], linestyle='--', marker='o', label=f"SARL: {m}")
            
    if 'Model' in marl_df.columns:
        for m in marl_df['Model'].unique():
            sub = marl_df[marl_df['Model'] == m]
            plt.plot(sub['Offered_Load_pps'], sub['Throughput_Mbps'], linestyle='-', marker='s', label=f"MARL: {m}")
            
    plt.xlabel("Offered Load (pps)")
    plt.ylabel("Throughput (Mbps)")
    plt.title("Master Comparison: SARL vs MARL Swarm Intelligence")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    comb_path = os.path.join(combined_img_dir, "combined_sarl_marl_throughput.png")
    plt.savefig(comb_path, dpi=200)
    plt.close()
    log_print(f"  [Plot] Combined Master Graph Saved -> {comb_path}")

def run_marl_on_sarl_ablation(traffic_pps_list, marl_dir, log_print):
    """
    === ABLATION STUDY: MARL ON SARL ENV ===
    To test multi-agent (MARL) models inside the single-agent gym environment,
    we bridge dimensions mathematically:
    1. SARL Env outputs 1 Global State -> We duplicate/pad this into N states for the MARL agents.
    2. MARL Model predicts N Actions -> We take a MAS (Majority Vote) to determine the 1 Action the SARL Env receives.
    """
    log_print("\n=== ABLATION STUDY: MARL ON SARL ENV ===")
    try:
        from envs.adaptive_mac_env import AdaptiveMacEnv
        from algorithms.rl.marl_baselines import IQLAgent, VDNAgent, QMIXAgent
        import torch
    except ImportError:
        log_print("  [Ablation] Could not import dependencies. Skipping.")
        return
        
    ablation_dir = os.path.join(marl_dir, "MARL_on_SARL_Ablation")
    os.makedirs(ablation_dir, exist_ok=True)
    
    env = AdaptiveMacEnv()
    cp_dir = MARLConfig.get_checkpoint_dir()
    N = params.N
    obs_dim = 16
    
    marl_models = {}
    if os.path.exists(os.path.join(cp_dir, "marl_qmix_model.pth")):
        m = QMIXAgent(N, obs_dim, 2); m.load(os.path.join(cp_dir, "marl_qmix_model.pth"))
        marl_models["QMIX"] = m
    if os.path.exists(os.path.join(cp_dir, "marl_vdn_model.pth")):
        m = VDNAgent(N, obs_dim, 2); m.load(os.path.join(cp_dir, "marl_vdn_model.pth"))
        marl_models["VDN"] = m
    if os.path.exists(os.path.join(cp_dir, "gnn_marl_model.pth")):
        from algorithms.rl.gnn_marl import MAGAT_D3QN_QNetwork
        device = torch.device("cpu")
        gnn = MAGAT_D3QN_QNetwork(node_in_dim=obs_dim, hidden_dim=64, num_actions=2, heads=4).to(device)
        gnn.load_state_dict(torch.load(os.path.join(cp_dir, "gnn_marl_model.pth"), map_location=device))
        gnn.eval()
        marl_models["MAGAT-D3QN"] = gnn
        
    for model_name, model in marl_models.items():
        log_print(f"  [Ablation] Evaluating {model_name} on SARL Environment...")
        
        for pps in traffic_pps_list:
            sarl_obs_dict, _ = env.reset()
            sarl_obs = sarl_obs_dict["scalars"]
            
            done = False
            while not done:
                # 1. Bridge SARL (14-dim) into N identical MARL vectors (16-dim)
                marl_obs_all = np.zeros((N, 16), dtype=np.float32)
                marl_obs_all[:, 0] = sarl_obs[0]  # load
                marl_obs_all[:, 1] = sarl_obs[13] # prev_act -> mac_norm
                marl_obs_all[:, 2] = sarl_obs[8]  # mean_neighbor_dist -> avg_dist
                marl_obs_all[:, 3] = sarl_obs[10] # mean_neighbor_speed -> mean_speed
                marl_obs_all[:, 4] = sarl_obs[9]  # neighbor_density -> link_up_ratio
                marl_obs_all[:, 5] = sarl_obs[12] # fading_gain -> succ_prob proxy
                marl_obs_all[:, 6] = sarl_obs[12] # succ_prob
                marl_obs_all[:, 7] = sarl_obs[8]  # dist_to_target -> avg_dist
                marl_obs_all[:, 8] = 1.0          # tx_rate_norm
                marl_obs_all[:, 9] = sarl_obs[2]  # queue_util
                marl_obs_all[:, 10] = sarl_obs[5] # delay
                marl_obs_all[:, 11] = sarl_obs[5] # max_delay -> delay
                marl_obs_all[:, 12] = 0.0         # jitter
                marl_obs_all[:, 13] = sarl_obs[4] # throughput
                marl_obs_all[:, 14] = sarl_obs[7] # drop_rate
                marl_obs_all[:, 15] = 1.0         # battery
                
                # 2. MARL Predicts N actions
                if model_name == "MAGAT-D3QN":
                    # For ablation, mock graph edges just as a cycle (all-to-all proxy)
                    edges = [[i, j] for i in range(N) for j in range(N) if i != j]
                    x_t = torch.tensor(marl_obs_all, dtype=torch.float32)
                    edge_t = torch.tensor(edges, dtype=torch.long).t()
                    with torch.no_grad():
                        q_vals = model(x_t, edge_t)
                        actions = [q_vals[i].argmax().item() for i in range(N)]
                else:
                    model.steps_done = 999999 # Force greedy
                    actions = model.select_actions(marl_obs_all)
                
                # 3. Majority rule vote for SARL action
                final_sarl_action = int(np.bincount(actions).argmax())
                next_sarl_dict, reward, terminated, truncated, _ = env.step(final_sarl_action)
                sarl_obs = next_sarl_dict["scalars"]
                done = terminated or truncated

if __name__ == '__main__':
    run_experiment_server()
