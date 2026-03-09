import os
import sys
import json
import datetime
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to sys.path so we can import algorithms
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from algorithms.mac.baseline import Config, Logger
from algorithms.mac.baseline import simulate_slotted_aloha, simulate_tdma, simulate_csma_ca
from configs import config as params
from configs.rl_config import RLConfig

# =====================================================================
# Constants
# =====================================================================
MAC_NAMES = {0: "TDMA", 1: "CSMA_CA"}
RL_TRAINING_TIMESTEPS = 1_000  # Training timesteps per RL selector

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
    trial_name = f"trial_{timestamp}"
    
    # Example format: N20_PHY2M_Q20
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
    rl_out_dir = os.path.join(base_dir, "RL_comparison")
    os.makedirs(os.path.join(rl_out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(rl_out_dir, "csv"), exist_ok=True)
    
    return base_dir, rl_out_dir, trial_name


# =====================================================================
# Phase 1: Baseline MAC Simulations (unchanged)
# =====================================================================

def run_baseline_simulations(cfg, traffic_pps_list, sim_time_s, N, seed):
    """Run ALOHA, TDMA, CSMA/CA baseline simulations across traffic loads."""
    results = []
    q_log_aloha = []
    q_log_tdma = []
    q_log_csma = []

    for idx, pps in enumerate(traffic_pps_list):
        print(f"  [Baseline] Testing load {pps} pps ({idx+1}/{len(traffic_pps_list)})")
        
        log_aloha = Logger(load_pps=pps)
        cfg.seed = seed + idx
        simulate_slotted_aloha(cfg, pps, log_aloha)
        q_log_aloha.extend(log_aloha.q_log)
        
        log_tdma = Logger(load_pps=pps)
        cfg.seed = seed + idx
        simulate_tdma(cfg, pps, log_tdma)
        q_log_tdma.extend(log_tdma.q_log)
        
        log_csma = Logger(load_pps=pps)
        cfg.seed = seed + idx
        simulate_csma_ca(cfg, pps, log_csma)
        q_log_csma.extend(log_csma.q_log)
        
        results.append({
            'Offered_Load_pps': pps,
            
            'ALOHA_Throughput_Mbps': log_aloha.get_throughput_bps(sim_time_s) / 1e6,
            'TDMA_Throughput_Mbps': log_tdma.get_throughput_bps(sim_time_s) / 1e6,
            'CSMA_Throughput_Mbps': log_csma.get_throughput_bps(sim_time_s) / 1e6,
            
            'ALOHA_Drops': log_aloha.pkts_dropped_qfull + log_aloha.pkts_dropped_mac,
            'TDMA_Drops': log_tdma.pkts_dropped_qfull + log_tdma.pkts_dropped_mac,
            'CSMA_Drops': log_csma.pkts_dropped_qfull + log_csma.pkts_dropped_mac,
            
            'ALOHA_Q_len': log_aloha.get_avg_q_len(N),
            'TDMA_Q_len': log_tdma.get_avg_q_len(N),
            'CSMA_Q_len': log_csma.get_avg_q_len(N),
            
            'ALOHA_Delay_s': log_aloha.get_avg_end_to_end_delay_s(),
            'TDMA_Delay_s': log_tdma.get_avg_end_to_end_delay_s(),
            'CSMA_Delay_s': log_csma.get_avg_end_to_end_delay_s(),
            
            'ALOHA_Util': log_aloha.get_channel_utilization(),
            'TDMA_Util': log_tdma.get_channel_utilization(),
            'CSMA_Util': log_csma.get_channel_utilization(),
            
            'ALOHA_Collisions': log_aloha.collision_events,
            'TDMA_Collisions': log_tdma.collision_events,
            'CSMA_Collisions': log_csma.collision_events,
        })
    
    return results, q_log_aloha, q_log_tdma, q_log_csma


# =====================================================================
# Phase 2: RL Selector Training implementation
# =====================================================================

from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggerCallback(BaseCallback):
    """Custom callback to log episodic average rewards during training."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.steps = []

    def _on_step(self) -> bool:
        # Check if an episode just ended in the monitor wrapper
        if "episode" in self.locals.get("infos", [{}])[0]:
            ep_info = self.locals["infos"][0]["episode"]
            self.episode_rewards.append(ep_info["r"])
            self.steps.append(self.num_timesteps)
        return True

def train_all_rl_selectors(rl_out_dir, log_print, training_timesteps=None):
    """
    Train all RL selectors over AdaptiveMacEnv.
    Skips training if a checkpoint already exists.
    Returns the checkpoint directory path.
    """
    from envs.adaptive_mac_env import AdaptiveMacEnv
    from algorithms.rl.tabular_qlearning import TabularQLearning
    from algorithms.rl.sb3_baselines import create_sb3_baseline
    from algorithms.rl.custom_mca_d3qn import create_mca_d3qn
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    timesteps = training_timesteps or RL_TRAINING_TIMESTEPS
    cp_dir = os.path.join(RLConfig.get_results_dir(), "checkpoints")
    os.makedirs(cp_dir, exist_ok=True)

    # --- Tabular Q-Learning ---
    if getattr(params, "RUN_TABULAR_QLEARNING", True):
        tab_path = os.path.join(cp_dir, "tabular_q_model.json")
        if os.path.exists(tab_path):
            log_print(f"  [Train] Tabular Q-Learning — checkpoint exists, skipping.")
        else:
            log_print(f"  [Train] Tabular Q-Learning — training for {timesteps} steps...")
            env = AdaptiveMacEnv()
            model = TabularQLearning(seed=RLConfig.SEED)
            obs, info = env.reset(seed=RLConfig.SEED)
            tab_rewards = []
            tab_steps = []
            episode_reward = 0.0
            for step in range(timesteps):
                action, _ = model.predict(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = env.step(action)
                model.learn(obs, action, reward, next_obs, done=(terminated or truncated))
                obs = next_obs
                episode_reward += reward
                if terminated or truncated:
                    obs, info = env.reset()
                    tab_rewards.append(episode_reward)
                    tab_steps.append(step + 1)
                    episode_reward = 0.0
                if (step + 1) % 10000 == 0:
                    log_print(f"    Step {step+1}/{timesteps} — Eps: {model.epsilon:.4f}")
            model.save(tab_path)
            log_print(f"    Saved to {tab_path}")
            # Save tabular rewards log
            if tab_steps:
                pd.DataFrame({"step": tab_steps, "reward": tab_rewards}).to_csv(
                    os.path.join(rl_out_dir, "csv", "tabular_training_rewards.csv"), index=False
                )

    # --- SB3 Baselines: DQN, PPO, A2C ---
    sb3_configs = []
    if getattr(params, "RUN_DQN", True):
        sb3_configs.append("dqn")
    if getattr(params, "RUN_PPO", True):
        sb3_configs.append("ppo")
    if getattr(params, "RUN_A2C", True):
        sb3_configs.append("a2c")

    for algo_name in sb3_configs:
        save_path = os.path.join(cp_dir, f"{algo_name}_baseline_model")
        zip_path = save_path + ".zip"
        if os.path.exists(zip_path):
            log_print(f"  [Train] {algo_name.upper()} — checkpoint exists, skipping.")
        else:
            log_print(f"  [Train] {algo_name.upper()} — training for {timesteps} steps...")
            env = AdaptiveMacEnv()
            env = Monitor(env)
            vec_env = DummyVecEnv([lambda: env])
            model = create_sb3_baseline(vec_env, algo_name=algo_name, seed=RLConfig.SEED)
            cb = RewardLoggerCallback()
            model.learn(total_timesteps=timesteps, progress_bar=True, callback=cb)
            model.save(save_path)
            log_print(f"    Saved to {save_path}")
            if cb.steps:
                pd.DataFrame({"step": cb.steps, "reward": cb.episode_rewards}).to_csv(
                    os.path.join(rl_out_dir, "csv", f"{algo_name}_training_rewards.csv"), index=False
                )

    # --- MCA-D3QN (Custom Model / "Our Model") ---
    if getattr(params, "RUN_CUSTOM_RL", True):
        mca_path = os.path.join(cp_dir, "mca_d3qn_model")
        mca_zip = mca_path + ".zip"
        if os.path.exists(mca_zip):
            log_print(f"  [Train] MCA-D3QN — checkpoint exists, skipping.")
        else:
            log_print(f"  [Train] MCA-D3QN — training for {timesteps} steps...")
            env = AdaptiveMacEnv()
            env = Monitor(env)
            vec_env = DummyVecEnv([lambda: env])
            model = create_mca_d3qn(vec_env, seed=RLConfig.SEED)
            cb = RewardLoggerCallback()
            model.learn(total_timesteps=timesteps, progress_bar=True, callback=cb)
            model.save(mca_path)
            log_print(f"    Saved to {mca_path}")
            if cb.steps:
                pd.DataFrame({"step": cb.steps, "reward": cb.episode_rewards}).to_csv(
                    os.path.join(rl_out_dir, "csv", "mca_d3qn_training_rewards.csv"), index=False
                )

    log_print("  [Train] All RL selector training complete.")
    return cp_dir


# =====================================================================
# Phase 3: Load Trained Models
# =====================================================================

def load_trained_models(log_print):
    """Load all available trained RL models from checkpoints."""
    from stable_baselines3 import DQN, PPO, A2C
    from algorithms.rl.tabular_qlearning import TabularQLearning
    import torch
    from algorithms.rl.gnn_marl import MAGAT_D3QN_QNetwork

    models = {}
    cp_dir = os.path.join(RLConfig.get_results_dir(), "checkpoints")

    # MCA-D3QN ("Our Model")
    if getattr(params, "RUN_CUSTOM_RL", True):
        path = os.path.join(cp_dir, "mca_d3qn_model.zip")
        if os.path.exists(path):
            models["MCA_D3QN"] = DQN.load(path)
            log_print(f"  [Load] MCA-D3QN loaded.")

    # SB3 Baselines
    algo_map = [("dqn", DQN), ("ppo", PPO), ("a2c", A2C)]
    flag_map = {"dqn": "RUN_DQN", "ppo": "RUN_PPO", "a2c": "RUN_A2C"}
    for algo, cls in algo_map:
        if getattr(params, flag_map[algo], True):
            path = os.path.join(cp_dir, f"{algo}_baseline_model.zip")
            if os.path.exists(path):
                models[algo.upper()] = cls.load(path)
                log_print(f"  [Load] {algo.upper()} loaded.")

    # Tabular Q-Learning
    if getattr(params, "RUN_TABULAR_QLEARNING", True):
        path = os.path.join(cp_dir, "tabular_q_model.json")
        if os.path.exists(path):
            tab = TabularQLearning()
            tab.load(path)
            tab.epsilon = 0.0  # Greedy evaluation
            models["TABULAR"] = tab
            log_print(f"  [Load] Tabular Q-Learning loaded.")
            
    # MARL GNN
    if getattr(params, "RUN_MARL_GNN", True):
        path = os.path.join(cp_dir, "gnn_marl_model.pth")
        if os.path.exists(path):
            device = torch.device("cpu")
            model = MAGAT_D3QN_QNetwork(node_in_dim=8, hidden_dim=64, num_actions=2, heads=4).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            models["MARL_GNN"] = model
            log_print("  [Load] MARL_GNN loaded.")

    return models


# =====================================================================
# Phase 4: Evaluate RL Selectors & Generate Decision Graphs
# =====================================================================

def evaluate_rl_selectors(models, df, out_dir, log_print):
    """
    Evaluate each RL selector on the baseline simulation environment.
    For each selector, simulate decisions across traffic loads and
    generate separate decision graphs + CSV files.
    
    Uses the simulation results in df (TDMA/CSMA baselines) to create
    per-traffic-load evaluations — the same approach as qlearning_selector.
    """
    from envs.adaptive_mac_env import AdaptiveMacEnv
    from envs.marl_mac_env import MARLMacEnv
    import torch
    
    all_selector_results = {}

    for model_name, model in models.items():
        log_print(f"  [Eval] Evaluating {model_name}...")
        
        env = MARLMacEnv() if model_name == "MARL_GNN" else AdaptiveMacEnv()
        selector_rows = []
        
        # Iterate over the same traffic loads as baseline
        for i in range(len(df)):
            row = df.iloc[i]
            traffic_rate = float(row["Offered_Load_pps"])
            
            # Set the offered PPS on the global config so env uses it
            params.OFFERED_PPS = int(traffic_rate)
            params.SWEEP_MAX_PPS = int(traffic_rate) # MARL env loads this dynamically
            
            if i == 0:
                if model_name == "MARL_GNN":
                    env.reset()
                else:
                    obs, _ = env.reset(seed=params.SEED)
            
            # Get RL action
            if model_name == "MARL_GNN":
                x, edge_index = env.get_global_graph_state()
                x_t = torch.tensor(x, dtype=torch.float32)
                edge_t = torch.tensor(edge_index, dtype=torch.long)
                
                actions = {}
                with torch.no_grad():
                    q_vals = model(x_t, edge_t) # Shape (N, 2)
                    for idx_a, agent in enumerate(env.agents):
                        actions[agent] = q_vals[idx_a].argmax().item()
                
                obs, reward, terminations, truncations, infos = env.step(actions)
                
                # We need the global chosen MAC and throughput
                uav_0_info = infos[env.agents[0]]
                action = uav_0_info["chosen_mac"]
                selected_mac = MAC_NAMES[action]
                
                # In MARL, reward is a dict, we take the shared value
                reward_val = reward[env.agents[0]] 
                
                metrics = {
                    "throughput_mbps": uav_0_info["throughput"],
                    "delay_ms": 0.0, # MARL Env currently uses fast-forward simulation returning throughput/drops
                    "drops": uav_0_info["drops"],
                    "collisions": 0
                }
                terminated = terminations[env.agents[0]]
                truncated = truncations[env.agents[0]]

            else:
                if model_name == "TABULAR":
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                
                action = int(action)
                selected_mac = MAC_NAMES[action]
                
                # Step the environment to get the actual metrics
                obs, reward_val, terminated, truncated, info = env.step(action)
                metrics = info.get("metrics", {})
            
            # Also record what baselines achieved at this load
            selector_rows.append({
                "traffic_rate_pps": traffic_rate,
                "selected_mac": selected_mac,
                "action": action,
                "reward": reward_val,
                "throughput_mbps": metrics.get("throughput_mbps", 0.0),
                "delay_ms": metrics.get("delay_ms", 0.0),
                "drops": metrics.get("drops", 0),
                "collisions": metrics.get("collisions", 0),
                "baseline_tdma_throughput": float(row.get("TDMA_Throughput_Mbps", 0)),
                "baseline_csma_throughput": float(row.get("CSMA_Throughput_Mbps", 0)),
                "baseline_tdma_delay": float(row.get("TDMA_Delay_s", 0)),
                "baseline_csma_delay": float(row.get("CSMA_Delay_s", 0)),
            })
            
            # Reset if episode ended
            if terminated or truncated:
                if model_name == "MARL_GNN":
                    env.reset()
                else:
                    obs, _ = env.reset(seed=params.SEED + i + 1)
        
        sel_df = pd.DataFrame(selector_rows)
        all_selector_results[model_name] = sel_df
        
        # --- Save CSV ---
        csv_path = os.path.join(out_dir, "csv", f"{model_name.lower()}_mac_selection.csv")
        sel_df.to_csv(csv_path, index=False)
        log_print(f"    Saved {model_name} selection CSV to {csv_path}")
        
        # --- Plot 1: Selected MAC vs Traffic Rate ---
        _plot_mac_selection(sel_df, model_name, out_dir)
        
        # --- Plot 2: Delay Curves with RL Choice Overlay ---
        _plot_delay_overlay(sel_df, df, model_name, out_dir)
        
        # --- Plot 3: Throughput Curves with RL Choice Overlay ---
        _plot_throughput_overlay(sel_df, df, model_name, out_dir)
        
        log_print(f"    Generated 3 decision graphs for {model_name}")
    
    return all_selector_results


def _plot_mac_selection(sel_df, model_name, out_dir):
    """Plot: Selected MAC vs Traffic Rate (scatter)."""
    plt.figure(figsize=(10, 6))
    y_vals = [0 if m == "TDMA" else 1 for m in sel_df["selected_mac"]]
    plt.scatter(
        sel_df["traffic_rate_pps"], y_vals,
        c=y_vals, cmap="coolwarm", s=100, edgecolor="k"
    )
    plt.yticks([0, 1], ["TDMA", "CSMA/CA"])
    plt.xlabel("Traffic Rate (pps)")
    plt.ylabel("Selected MAC")
    rts_tag = "ON" if getattr(params, 'RTS_CTS_ENABLED', False) else "OFF"
    ack_tag = "ON" if getattr(params, 'ACK_ENABLED', False) else "OFF"
    plt.title(
        f"{model_name} - Selected MAC vs Traffic Rate\n"
        f"(N={params.N}, RTS/CTS={rts_tag}, ACK={ack_tag})"
    )
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "images", f"{model_name.lower()}_selected_mac_vs_traffic_rate.png"))
    plt.close()


def _plot_delay_overlay(sel_df, df, model_name, out_dir):
    """Plot: Delay curves with RL choice overlay."""
    plt.figure(figsize=(10, 6))
    plt.plot(df["Offered_Load_pps"], df["TDMA_Delay_s"],
             label="TDMA (Base)", color="blue", alpha=0.5, linestyle="--")
    plt.plot(df["Offered_Load_pps"], df["CSMA_Delay_s"],
             label="CSMA/CA (Base)", color="red", alpha=0.5, linestyle="--")
    
    tdma_sel = sel_df[sel_df["selected_mac"] == "TDMA"]
    csma_sel = sel_df[sel_df["selected_mac"] == "CSMA_CA"]
    
    plt.scatter(tdma_sel["traffic_rate_pps"], tdma_sel["delay_ms"] / 1000.0,
                color="blue", s=100, label=f"{model_name} -> TDMA", marker="o", edgecolor="k")
    plt.scatter(csma_sel["traffic_rate_pps"], csma_sel["delay_ms"] / 1000.0,
                color="red", s=100, label=f"{model_name} -> CSMA/CA", marker="^", edgecolor="k")
    
    plt.xlabel("Traffic Rate (pps)")
    plt.ylabel("End-to-End Delay (s)")
    plt.title(f"{model_name} - Delay Curves with MAC Selection Overlay")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "images", f"{model_name.lower()}_delay_curves_with_choice.png"))
    plt.close()


def _plot_throughput_overlay(sel_df, df, model_name, out_dir):
    """Plot: Throughput curves with RL choice overlay."""
    plt.figure(figsize=(10, 6))
    plt.plot(df["Offered_Load_pps"], df["TDMA_Throughput_Mbps"],
             label="TDMA (Base)", color="blue", alpha=0.5, linestyle="--")
    plt.plot(df["Offered_Load_pps"], df["CSMA_Throughput_Mbps"],
             label="CSMA/CA (Base)", color="red", alpha=0.5, linestyle="--")
    
    tdma_sel = sel_df[sel_df["selected_mac"] == "TDMA"]
    csma_sel = sel_df[sel_df["selected_mac"] == "CSMA_CA"]
    
    plt.scatter(tdma_sel["traffic_rate_pps"], tdma_sel["throughput_mbps"],
                color="blue", s=100, label=f"{model_name} -> TDMA", marker="o", edgecolor="k")
    plt.scatter(csma_sel["traffic_rate_pps"], csma_sel["throughput_mbps"],
                color="red", s=100, label=f"{model_name} -> CSMA/CA", marker="^", edgecolor="k")
    
    plt.xlabel("Traffic Rate (pps)")
    plt.ylabel("Throughput (Mbps)")
    plt.title(f"{model_name} - Throughput Curves with MAC Selection Overlay")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "images", f"{model_name.lower()}_throughput_curves_with_choice.png"))
    plt.close()


# =====================================================================
# Phase 5: Lookup Table Generation
# =====================================================================

def generate_rl_lookup_tables(all_selector_results, df, out_dir, log_print):
    """
    Generate a consolidated lookup table for each RL selector.
    Shows per-traffic-load: which MAC was selected, reward, perf metrics.
    """
    for model_name, sel_df in all_selector_results.items():
        lookup_rows = []
        for i in range(len(sel_df)):
            row = sel_df.iloc[i]
            lookup_rows.append({
                "traffic_load_pps": row["traffic_rate_pps"],
                "selected_mac": row["selected_mac"],
                "reward": round(row["reward"], 4),
                "throughput_mbps": round(row["throughput_mbps"], 4),
                "delay_ms": round(row["delay_ms"], 4),
                "drops": int(row["drops"]),
                "collisions": int(row["collisions"]),
                "baseline_tdma_throughput_mbps": round(row.get("baseline_tdma_throughput", 0), 4),
                "baseline_csma_throughput_mbps": round(row.get("baseline_csma_throughput", 0), 4),
                "baseline_tdma_delay_s": round(row.get("baseline_tdma_delay", 0), 6),
                "baseline_csma_delay_s": round(row.get("baseline_csma_delay", 0), 6),
            })
        
        lt_df = pd.DataFrame(lookup_rows)
        lt_path = os.path.join(out_dir, "csv", f"{model_name.lower()}_lookup_table.csv")
        lt_df.to_csv(lt_path, index=False)
        log_print(f"  [Lookup] Saved {model_name} lookup table -> {lt_path}")


# =====================================================================
# Phase 7: Unified Aggregated RL Plots
# =====================================================================

def generate_aggregated_rl_plots(all_selector_results, df, q_df, rl_out_dir, log_print):
    """
    Generate unified plots comparing all RL algorithms natively on single graphs.
    Includes Action Selection, Delay Override, Throughput Override, and Training Rewards.
    """
    import glob
    
    # 1. Training Reward Curves
    _plot_reward_curves(rl_out_dir, log_print)
    
    # Gather everything together
    comparison_dict = all_selector_results.copy()
    
    if q_df is not None and not q_df.empty:
        # Standardize Q-learning output to match our dynamic env schema
        fmt_q_df = pd.DataFrame()
        fmt_q_df["traffic_rate_pps"] = q_df["traffic_rate"]
        fmt_q_df["selected_mac"] = q_df["qlearning_selected_mac"]
        fmt_q_df["delay_ms"] = [
            r["tdma_delay"]*1000 if r["qlearning_selected_mac"] == "TDMA" else r["csma_delay"]*1000
            for _, r in q_df.iterrows()
        ]
        fmt_q_df["throughput_mbps"] = [
            r["tdma_throughput"] if r["qlearning_selected_mac"] == "TDMA" else r["csma_throughput"]
            for _, r in q_df.iterrows()
        ]
        comparison_dict["Offline_Q_Learning"] = fmt_q_df
        
    models_list = list(comparison_dict.keys())
    if not models_list:
        return
        
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    
    # 2. Aggregated MAC Selection
    plt.figure(figsize=(10, 6))
    for idx, model_name in enumerate(models_list):
        sel_df = comparison_dict[model_name]
        y_vals = [0 if m == "TDMA" else 1 for m in sel_df["selected_mac"]]
        # Add slight jitter to y to separate overlapping dots
        y_jitter = np.array(y_vals) + (idx - len(models_list)/2.0) * 0.05
        plt.scatter(
            sel_df["traffic_rate_pps"], y_jitter,
            color=colors[idx % len(colors)], s=60, marker=markers[idx % len(markers)],
            label=model_name, edgecolor="k", alpha=0.8
        )
    plt.yticks([0, 1], ["TDMA", "CSMA/CA"])
    plt.xlabel("Traffic Rate (pps)")
    plt.ylabel("Selected MAC")
    plt.title("Aggregated MAC Selection Across all RL Algorithms")
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(rl_out_dir, "images", "aggregated_mac_selection.png"))
    plt.close()
    
    # 3. Aggregated Delay Overlay
    plt.figure(figsize=(12, 7))
    plt.plot(df["Offered_Load_pps"], df["TDMA_Delay_s"], label="TDMA (Base)", color="gray", alpha=0.5, linestyle="--")
    plt.plot(df["Offered_Load_pps"], df["CSMA_Delay_s"], label="CSMA/CA (Base)", color="gray", alpha=0.5, linestyle=":")
    
    for idx, model_name in enumerate(models_list):
        sel_df = comparison_dict[model_name]
        plt.plot(
            sel_df["traffic_rate_pps"], sel_df["delay_ms"] / 1000.0,
            color=colors[idx % len(colors)], marker=markers[idx % len(markers)],
            label=f"{model_name}", linewidth=2, markersize=8, alpha=0.8
        )
        
    plt.xlabel("Traffic Rate (pps)")
    plt.ylabel("End-to-End Delay (s)")
    plt.title("Aggregated Delay Overlays Across all RL Algorithms")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(rl_out_dir, "images", "aggregated_delay_overlay.png"))
    plt.close()

    # 4. Aggregated Throughput Overlay
    plt.figure(figsize=(12, 7))
    plt.plot(df["Offered_Load_pps"], df["TDMA_Throughput_Mbps"], label="TDMA (Base)", color="gray", alpha=0.5, linestyle="--")
    plt.plot(df["Offered_Load_pps"], df["CSMA_Throughput_Mbps"], label="CSMA/CA (Base)", color="gray", alpha=0.5, linestyle=":")
    
    for idx, model_name in enumerate(models_list):
        sel_df = comparison_dict[model_name]
        plt.plot(
            sel_df["traffic_rate_pps"], sel_df["throughput_mbps"],
            color=colors[idx % len(colors)], marker=markers[idx % len(markers)],
            label=f"{model_name}", linewidth=2, markersize=8, alpha=0.8
        )
        
    plt.xlabel("Traffic Rate (pps)")
    plt.ylabel("Throughput (Mbps)")
    plt.title("Aggregated Throughput Overlays Across all RL Algorithms")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(rl_out_dir, "images", "aggregated_throughput_overlay.png"))
    plt.close()
    
    log_print("  [Aggregate] Saved unified decision graphs to RL_comparison/images/")


def _plot_reward_curves(rl_out_dir, log_print):
    import glob
    csv_files = glob.glob(os.path.join(rl_out_dir, "csv", "*_training_rewards.csv"))
    if not csv_files:
        return
        
    plt.figure(figsize=(10, 6))
    markers = ['-', '--', '-.', ':']
    
    for idx, f in enumerate(csv_files):
        try:
            name = os.path.basename(f).replace("_training_rewards.csv", "").upper()
            d = pd.read_csv(f)
            if not d.empty and "step" in d.columns and "reward" in d.columns:
                # Smoothing
                d['reward_smooth'] = d['reward'].rolling(window=max(1, len(d)//20), min_periods=1).mean()
                plt.plot(d['step'], d['reward_smooth'], label=name, 
                         linestyle=markers[idx % len(markers)], linewidth=2)
        except Exception as e:
            log_print(f"  [Reward Plot Error] Could not parse {f}: {e}")
            
    plt.xlabel("Training Timesteps")
    plt.ylabel("Episodic Average Reward (Smoothed)")
    plt.title(f"Reward vs. Training Steps Overview\n(Total Timesteps: {RL_TRAINING_TIMESTEPS})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(rl_out_dir, "images", "reward_vs_training_steps.png"))
    plt.close()
    log_print("  [Aggregate] Saved reward vs training steps plot.")


# =====================================================================
# Baseline Plots (UNCHANGED from original)
# =====================================================================

def generate_baseline_plots(df, out_dir, N, payload_bytes, phy_rate_bps, QMAX,
                            rts_cts_enabled, ack_enabled):
    """Generate the 5 standard baseline comparison plots. These are NOT modified."""
    
    # 1. Throughput
    plt.figure(figsize=(10,6))
    plt.plot(df['Offered_Load_pps'], df['ALOHA_Throughput_Mbps'], label='Slotted ALOHA', marker='o')
    plt.plot(df['Offered_Load_pps'], df['TDMA_Throughput_Mbps'], label='TDMA', marker='s')
    plt.plot(df['Offered_Load_pps'], df['CSMA_Throughput_Mbps'], label='CSMA/CA', marker='^')
    plt.xlabel('Offered Traffic Rate (packets/sec total)')
    plt.ylabel('Throughput (Mbps)')
    plt.title(f'Protocol Comparison: Throughput\n(N={N}, Pkt={payload_bytes}B, PHY={phy_rate_bps/1e6}Mbps)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "images", "throughput_vs_load.png"))
    plt.close()
    
    # 2. Utilization
    plt.figure(figsize=(10,6))
    plt.plot(df['Offered_Load_pps'], df['ALOHA_Util'], label='Slotted ALOHA', marker='o')
    plt.plot(df['Offered_Load_pps'], df['TDMA_Util'], label='TDMA', marker='s')
    plt.plot(df['Offered_Load_pps'], df['CSMA_Util'], label='CSMA/CA', marker='^')
    plt.xlabel('Offered Traffic Rate (packets/sec total)')
    plt.ylabel('Channel Utilization')
    plt.title(f'Protocol Comparison: Channel Utilization (N={N})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "images", "utilization_vs_load.png"))
    plt.close()
    
    # 3. Queues
    plt.figure(figsize=(10,6))
    plt.plot(df['Offered_Load_pps'], df['TDMA_Q_len'], label='TDMA', marker='s')
    plt.plot(df['Offered_Load_pps'], df['CSMA_Q_len'], label='CSMA/CA', marker='^')
    plt.xlabel('Offered Traffic Rate (packets/sec total)')
    plt.ylabel('Average Queue Length (packets)')
    plt.title(f'Protocol Comparison: Queue Occupancy\n(N={N}, Buffer Size={QMAX})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "images", "queues_vs_load.png"))
    plt.close()
    
    # 4. Drops
    plt.figure(figsize=(10,6))
    plt.plot(df['Offered_Load_pps'], df['ALOHA_Drops'], label='Slotted ALOHA', marker='o')
    plt.plot(df['Offered_Load_pps'], df['TDMA_Drops'], label='TDMA', marker='s')
    plt.plot(df['Offered_Load_pps'], df['CSMA_Drops'], label='CSMA/CA', marker='^')
    plt.xlabel('Offered Traffic Rate (packets/sec total)')
    plt.ylabel('Dropped Packets (Buffer + Retries)')
    plt.title(f'Protocol Comparison: Packet Drops (N={N})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "images", "drops_vs_load.png"))
    plt.close()
    
    # 5. End-To-End Delay
    plt.figure(figsize=(10,6))
    plt.plot(df['Offered_Load_pps'], df['ALOHA_Delay_s'], label='Slotted ALOHA', marker='o')
    plt.plot(df['Offered_Load_pps'], df['TDMA_Delay_s'], label='TDMA', marker='s')
    plt.plot(df['Offered_Load_pps'], df['CSMA_Delay_s'], label='CSMA/CA', marker='^')
    plt.xlabel('Offered Traffic Rate (packets/sec total)')
    plt.ylabel('End-to-End Delay (seconds)')
    rts_tag = "ON" if rts_cts_enabled else "OFF"
    ack_tag = "ON" if ack_enabled else "OFF"
    plt.title(f'Protocol Comparison: End-to-End Delay vs Traffic Rate\n(N={N}, QMAX={QMAX}, RTS/CTS={rts_tag}, ACK={ack_tag})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "images", "end_to_end_delay_vs_traffic_rate.png"))
    plt.close()


# =====================================================================
# Main Experiment Runner
# =====================================================================

def run_experiment():
    N = params.N
    sim_time_s = params.SIM_TIME_S
    slot_time_s = params.SLOT_TIME_S
    phy_rate_bps = params.PHY_RATE_BPS
    payload_bytes = params.PAYLOAD_BYTES
    QMAX = params.QMAX
    
    sweep_min_pps = params.SWEEP_MIN_PPS
    sweep_max_pps = params.SWEEP_MAX_PPS
    sweep_steps = params.SWEEP_STEPS
    
    # Advanced / Generic Tunable Parameters
    cw_min = params.CW_MIN
    cw_max = params.CW_MAX
    difs_slots = params.DIFS_SLOTS
    sifs_slots = params.SIFS_SLOTS
    ack_slots = params.ACK_SLOTS
    ack_timeout_slots = params.ACK_TIMEOUT_SLOTS
    max_retry = params.MAX_RETRY
    rts_cts_enabled = params.RTS_CTS_ENABLED
    ack_enabled = params.ACK_ENABLED
    rts_slots = params.RTS_SLOTS
    cts_slots = params.CTS_SLOTS
    
    log_interval_slots = params.LOG_INTERVAL_SLOTS

    seed = params.SEED
    
    out_dir, rl_out_dir, trial_name = make_result_dirs(N, phy_rate_bps, QMAX, rts_cts_enabled, ack_enabled)
    
    log_file_path = os.path.join(out_dir, "logs", "run.log")
    
    # Save metadata
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
        "script": os.path.basename(__file__),
        "seed_used": seed,
        "output_directory": out_dir,
        "parameters": {k: v for k, v in vars(params).items() if not k.startswith('__')}
    }
    
    with open(os.path.join(out_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4, default=str)
        
    def log_print(msg):
        print(msg)
        with open(log_file_path, "a") as lf:
            lf.write(msg + "\n")

    log_print(f"{'='*60}")
    log_print(f"Starting Experiment Runner (config.py aligned)")
    log_print(f"{'='*60}")
    log_print(f"Directory: {out_dir}")
    log_print(f"Parameters: N={N}, PHY={phy_rate_bps/1e6}Mbps, Pkt={payload_bytes}B, QMAX={QMAX}")
    log_print(f"RTS/CTS Enabled: {rts_cts_enabled}, ACK Enabled: {ack_enabled}")
    
    cfg = Config(N=N, sim_time_s=sim_time_s, slot_time_s=slot_time_s, 
                 phy_rate_bps=phy_rate_bps, payload_bytes=payload_bytes, QMAX=QMAX,
                 cw_min=cw_min, cw_max=cw_max, difs_slots=difs_slots, sifs_slots=sifs_slots,
                 ack_slots=ack_slots, ack_timeout_slots=ack_timeout_slots, max_retry=max_retry,
                 log_interval_slots=log_interval_slots, rts_cts_enabled=rts_cts_enabled,
                 ack_enabled=ack_enabled, rts_slots=rts_slots, cts_slots=cts_slots)

    traffic_pps_list = np.linspace(sweep_min_pps, sweep_max_pps, sweep_steps).astype(int)
    
    # ----------------------------------------------------------------
    # STEP 1: Baseline MAC Simulations (ALOHA, TDMA, CSMA/CA)
    # ----------------------------------------------------------------
    log_print(f"\n--- Step 1: Running Baseline MAC Simulations ---")
    results, q_log_aloha, q_log_tdma, q_log_csma = run_baseline_simulations(
        cfg, traffic_pps_list, sim_time_s, N, seed
    )
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "csv", "summary_metrics.csv")
    df.to_csv(csv_path, index=False)
    log_print(f"Saved metrics to {csv_path}")
    
    # Save End-to-End Delay standalone CSV
    delay_rows = []
    for idx, row in df.iterrows():
        delay_rows.append({"traffic_rate": row["Offered_Load_pps"], "traffic_rate_unit": "packets/sec", "end_to_end_delay": row["ALOHA_Delay_s"], "delay_unit": "seconds", "protocol": "ALOHA", "N": N, "trial_id": trial_name})
        delay_rows.append({"traffic_rate": row["Offered_Load_pps"], "traffic_rate_unit": "packets/sec", "end_to_end_delay": row["TDMA_Delay_s"], "delay_unit": "seconds", "protocol": "TDMA", "N": N, "trial_id": trial_name})
        delay_rows.append({"traffic_rate": row["Offered_Load_pps"], "traffic_rate_unit": "packets/sec", "end_to_end_delay": row["CSMA_Delay_s"], "delay_unit": "seconds", "protocol": "CSMA_CA", "N": N, "trial_id": trial_name})
    pd.DataFrame(delay_rows).to_csv(os.path.join(out_dir, "csv", "end_to_end_delay_vs_traffic_rate.csv"), index=False)
    log_print(f"Saved delay data to csv/end_to_end_delay_vs_traffic_rate.csv")
    
    # Save Time-series Queuing Files
    pd.DataFrame(q_log_aloha).to_csv(os.path.join(out_dir, "csv", "buffer_usage_ALOHA.csv"), index=False)
    pd.DataFrame(q_log_tdma).to_csv(os.path.join(out_dir, "csv", "buffer_usage_TDMA.csv"), index=False)
    pd.DataFrame(q_log_csma).to_csv(os.path.join(out_dir, "csv", "buffer_usage_CSMA_CA.csv"), index=False)
    log_print(f"Saved detailed buffer time-series to csv/buffer_usage_PROTOCOL.csv")
    
    # Save ACK Tracking Log (from last iteration's loggers)
    # NOTE: This captures the last traffic rate's ACK logs only
    
    # ----------------------------------------------------------------
    # STEP 2: Generate Baseline Plots (UNCHANGED)
    # ----------------------------------------------------------------
    log_print(f"\n--- Step 2: Generating Baseline Plots ---")
    generate_baseline_plots(df, out_dir, N, payload_bytes, phy_rate_bps, QMAX,
                            rts_cts_enabled, ack_enabled)
    log_print("Saved 5 baseline comparison plots.")

    # ----------------------------------------------------------------
    # STEP 3: Q-Learning Selector Post-Processing (UNCHANGED logic, but saves DF)
    # ----------------------------------------------------------------
    q_df = pd.DataFrame()
    if getattr(params, "ENABLE_RL_SELECTOR", False) or getattr(params, "RUN_QLEARNING_SELECTOR", False):
        log_print(f"\n--- Step 3: Q-Learning Selector Post-Processing ---")
        try:
            from algorithms.rl.qlearning_selector import run_rl_postprocess
            q_df = run_rl_postprocess(rl_out_dir, df, params, log_print)
        except Exception as e:
            log_print(f"ERROR in Q-Learning Selector Post-Processing: {e}")
    
    # ----------------------------------------------------------------
    # STEP 4: Train All RL Selectors Over AdaptiveMacEnv
    # ----------------------------------------------------------------
    log_print(f"\n--- Step 4: Training RL Selectors ---")
    try:
        cp_dir = train_all_rl_selectors(rl_out_dir, log_print)
    except Exception as e:
        log_print(f"ERROR in RL Training: {e}")
        import traceback
        log_print(traceback.format_exc())
        cp_dir = None
    
    # ----------------------------------------------------------------
    # STEP 5: Evaluate RL Selectors & Generate Decision Graphs
    # ----------------------------------------------------------------
    if cp_dir:
        log_print(f"\n--- Step 5: Evaluating RL Selectors ---")
        try:
            models = load_trained_models(log_print)
            if models:
                all_selector_results = evaluate_rl_selectors(models, df, rl_out_dir, log_print)
                
                # --------------------------------------------------------
                # STEP 6: Generate Lookup Tables
                # --------------------------------------------------------
                log_print(f"\n--- Step 6: Generating RL Lookup Tables ---")
                generate_rl_lookup_tables(all_selector_results, df, rl_out_dir, log_print)
                
                # --------------------------------------------------------
                # STEP 7: Generate Aggregated Plots
                # --------------------------------------------------------
                log_print(f"\n--- Step 7: Generating Aggregated RL Comparisons ---")
                generate_aggregated_rl_plots(all_selector_results, df, q_df, rl_out_dir, log_print)
                
            else:
                log_print("WARNING: No trained models could be loaded. Skipping RL evaluation.")
        except Exception as e:
            log_print(f"ERROR in RL Evaluation: {e}")
            import traceback
            log_print(traceback.format_exc())

    log_print(f"\n{'='*60}")
    log_print(f"Experiment completed successfully. Outputs saved to {out_dir}")
    log_print(f"{'='*60}")

if __name__ == '__main__':
    run_experiment()
