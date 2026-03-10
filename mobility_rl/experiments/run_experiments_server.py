import os
import sys
import json
import datetime
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing
import psutil

# Add the project root to sys.path so we can import algorithms
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# We must import these after path setup
from algorithms.mac.baseline import Config, Logger
from algorithms.mac.baseline import simulate_slotted_aloha, simulate_tdma, simulate_csma_ca
from configs import config as params
from configs.rl_config import RLConfig

# =====================================================================
# Constants & Hyperparameters
# =====================================================================
MAC_NAMES = {0: "TDMA", 1: "CSMA_CA"}
# In server environments, we typically want more training steps.
# You can override this dynamically.
RL_TRAINING_TIMESTEPS = 50000  

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
    trial_name = f"trial_SERVER_{timestamp}"
    
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
# Telemetry: Memory and Size Fetchers
# =====================================================================
def get_model_file_size_kb(filepath):
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / 1024.0
    elif os.path.exists(filepath + ".zip"):
        return os.path.getsize(filepath + ".zip") / 1024.0
    return 0.0

def estimate_ram_footprint():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0) # MB

# =====================================================================
# Phase 1: Baseline MAC Simulations 
# =====================================================================
def run_baseline_simulations(cfg, traffic_pps_list, sim_time_s, N, seed, log_print):
    results = []
    q_log_aloha = []
    q_log_tdma = []
    q_log_csma = []

    for idx, pps in enumerate(traffic_pps_list):
        log_print(f"  [Baseline] Testing load {pps} pps ({idx+1}/{len(traffic_pps_list)})")
        
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
            'TDMA_Throughput_Mbps': log_tdma.get_throughput_bps(sim_time_s) / 1e6,
            'CSMA_Throughput_Mbps': log_csma.get_throughput_bps(sim_time_s) / 1e6,
            'TDMA_Delay_s': log_tdma.get_avg_end_to_end_delay_s(),
            'CSMA_Delay_s': log_csma.get_avg_end_to_end_delay_s(),
            # ... keeping it brief for the essential baseline comparisons needed later
            'TDMA_Drops': log_tdma.pkts_dropped_qfull + log_tdma.pkts_dropped_mac,
            'CSMA_Drops': log_csma.pkts_dropped_qfull + log_csma.pkts_dropped_mac,
            'TDMA_Collisions': log_tdma.collision_events,
            'CSMA_Collisions': log_csma.collision_events,
        })
    return results

# =====================================================================
# Phase 2: Multiprocessed RL Training
# =====================================================================
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

class TqdmRewardLoggerCallback(BaseCallback):
    def __init__(self, algo_name, total_timesteps, position=0, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.steps = []
        self.algo_name = algo_name
        self.total_timesteps = total_timesteps
        self.position = position
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc=f"Training {self.algo_name.upper()}", position=self.position, leave=True, unit="step")

    def _on_step(self) -> bool:
        self.pbar.update(1)
        if "episode" in self.locals.get("infos", [{}])[0]:
            ep_info = self.locals["infos"][0]["episode"]
            self.episode_rewards.append(ep_info["r"])
            self.steps.append(self.num_timesteps)
            self.pbar.set_postfix({"Ep Reward": f"{ep_info['r']:.2f}"})
        return True

    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()

def train_single_model(kwargs):
    """
    Worker function to train a single RL model in an isolated process.
    """
    algo_name = kwargs['algo_name']
    timesteps = kwargs['timesteps']
    cp_dir = kwargs['cp_dir']
    rl_out_dir = kwargs['rl_out_dir']
    seed = kwargs['seed']
    position = kwargs['position']
    
    import pandas as pd
    from envs.adaptive_mac_env import AdaptiveMacEnv
    
    pid = os.getpid()
    print(f"  [Worker {pid}] Starting training for {algo_name.upper()}...")
    
    if algo_name == 'tabular':
        from algorithms.rl.tabular_qlearning import TabularQLearning
        tab_path = os.path.join(cp_dir, "tabular_q_model.json")
        if os.path.exists(tab_path):
            return f"{algo_name.upper()} skipped (checkpoint exists)"
            
        env = AdaptiveMacEnv()
        model = TabularQLearning(seed=seed)
        obs, info = env.reset(seed=seed)
        tab_rewards = []
        tab_steps = []
        episode_reward = 0.0
        
        pbar = tqdm(total=timesteps, desc=f"Training {algo_name.upper()}", position=position, leave=True, unit="step")
        
        for step in range(timesteps):
            action, _ = model.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            model.learn(obs, action, reward, next_obs, done=(terminated or truncated))
            obs = next_obs
            episode_reward += reward
            pbar.update(1)
            
            if terminated or truncated:
                obs, info = env.reset()
                tab_rewards.append(episode_reward)
                tab_steps.append(step + 1)
                pbar.set_postfix({"Ep Reward": f"{episode_reward:.2f}"})
                episode_reward = 0.0
                
        pbar.close()
        model.save(tab_path)
        if tab_steps:
            pd.DataFrame({"step": tab_steps, "reward": tab_rewards}).to_csv(
                os.path.join(rl_out_dir, "csv", "tabular_training_rewards.csv"), index=False
            )
        return f"{algo_name.upper()} trained."
        
    elif algo_name in ['dqn', 'ppo', 'a2c']:
        from algorithms.rl.sb3_baselines import create_sb3_baseline
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        save_path = os.path.join(cp_dir, f"{algo_name}_baseline_model")
        if os.path.exists(save_path + ".zip"):
            return f"{algo_name.upper()} skipped (checkpoint exists)"
            
        env = Monitor(AdaptiveMacEnv())
        vec_env = DummyVecEnv([lambda: env])
        model = create_sb3_baseline(vec_env, algo_name=algo_name, seed=seed)
        cb = TqdmRewardLoggerCallback(algo_name, timesteps, position)
        model.learn(total_timesteps=timesteps, progress_bar=False, callback=cb)  # disable internal TQDM
        model.save(save_path)
        
        if cb.steps:
            pd.DataFrame({"step": cb.steps, "reward": cb.episode_rewards}).to_csv(
                os.path.join(rl_out_dir, "csv", f"{algo_name}_training_rewards.csv"), index=False
            )
        return f"{algo_name.upper()} trained."
        
    elif algo_name == 'mca_d3qn':
        from algorithms.rl.custom_mca_d3qn import create_mca_d3qn
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        mca_path = os.path.join(cp_dir, "mca_d3qn_model")
        if os.path.exists(mca_path + ".zip"):
            return f"{algo_name.upper()} skipped (checkpoint exists)"
        
        env = Monitor(AdaptiveMacEnv())
        vec_env = DummyVecEnv([lambda: env])
        model = create_mca_d3qn(vec_env, seed=seed)
        cb = TqdmRewardLoggerCallback(algo_name, timesteps, position)
        model.learn(total_timesteps=timesteps, progress_bar=False, callback=cb)
        model.save(mca_path)
        
        if cb.steps:
            pd.DataFrame({"step": cb.steps, "reward": cb.episode_rewards}).to_csv(
                os.path.join(rl_out_dir, "csv", "mca_d3qn_training_rewards.csv"), index=False
            )
        return f"{algo_name.upper()} trained."

def execute_multiprocess_training(rl_out_dir, timesteps, log_print):
    cp_dir = os.path.join(RLConfig.get_results_dir(), "checkpoints")
    os.makedirs(cp_dir, exist_ok=True)
    
    # 5 Models
    tasks = []
    if getattr(params, "RUN_TABULAR_QLEARNING", True): tasks.append('tabular')
    if getattr(params, "RUN_DQN", True): tasks.append('dqn')
    if getattr(params, "RUN_PPO", True): tasks.append('ppo')
    if getattr(params, "RUN_A2C", True): tasks.append('a2c')
    if getattr(params, "RUN_CUSTOM_RL", True): tasks.append('mca_d3qn')
    
    kwargs_list = [
        {'algo_name': task, 'timesteps': timesteps, 'cp_dir': cp_dir, 'rl_out_dir': rl_out_dir, 'seed': params.SEED, 'position': i}
        for i, task in enumerate(tasks)
    ]
    
    cpu_cores = min(os.cpu_count() or 1, len(tasks))
    log_print(f"  [Train] Launching continuous Multiprocessing Pool with {cpu_cores} workers for {len(tasks)} algorithms...")
    
    # Multiprocessing
    import sys
    if sys.platform == 'win32':
        # Need to protect __main__ in windows, but we wrap this explicitly in run.
        pass

    with multiprocessing.Pool(processes=cpu_cores) as pool:
        for result in pool.imap_unordered(train_single_model, kwargs_list):
            log_print(f"    --> {result}")
            
    log_print("  [Train] Parallel Matrix complete.")

# =====================================================================
# Phase 3: Loading Models & Benchmarking
# =====================================================================
def load_and_benchmark_models(log_print):
    from stable_baselines3 import DQN, PPO, A2C
    from algorithms.rl.tabular_qlearning import TabularQLearning
    import torch
    from algorithms.rl.gnn_marl import MAGAT_D3QN_QNetwork
    from envs.marl_mac_env import MARLMacEnv
    
    models = {}
    benchmarks = {}
    cp_dir = os.path.join(RLConfig.get_results_dir(), "checkpoints")
    
    # Baseline RAM
    base_ram = estimate_ram_footprint()
    
    def _bench_model(name, load_func, path):
        if not os.path.exists(path) and not os.path.exists(path + ".zip"):
            return
            
        # 1. Size
        size_kb = get_model_file_size_kb(path)
        
        # 2. Memory
        m_before = estimate_ram_footprint()
        if name == "TABULAR":
            model = load_func()
            model.load(path)
            model.epsilon = 0.0
        elif name == "MARL_GNN":
            device = torch.device("cpu")
            model = MAGAT_D3QN_QNetwork(node_in_dim=8, hidden_dim=64, num_actions=2, heads=4).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
        else:
            model = load_func(path)
        m_after = estimate_ram_footprint()
        mem_mb = max(0, m_after - m_before)
        
        # 3. Latency
        from envs.adaptive_mac_env import AdaptiveMacEnv
        if name == "MARL_GNN":
            dummy_marl = MARLMacEnv()
            dummy_marl.reset()
            x, edge_index = dummy_marl.get_global_graph_state()
            x_t = torch.tensor(x, dtype=torch.float32)
            edge_t = torch.tensor(edge_index, dtype=torch.long)
            
            for _ in range(5):
                with torch.no_grad():
                    model(x_t, edge_t)
            times = []
            for _ in range(100):
                t0 = time.perf_counter_ns()
                with torch.no_grad():
                    model(x_t, edge_t)
                t1 = time.perf_counter_ns()
                times.append(t1 - t0)
                
        else:
            dummy_env = AdaptiveMacEnv()
            obs, _ = dummy_env.reset()
            
            # Warmup
            for _ in range(5):
                model.predict(obs, deterministic=True)
                
            times = []
            for _ in range(100):
                t0 = time.perf_counter_ns()
                model.predict(obs, deterministic=True)
                t1 = time.perf_counter_ns()
                times.append(t1 - t0)
            
        avg_latency_us = sum(times) / len(times) / 1000.0  # nanoseconds to microseconds
        
        models[name] = model
        benchmarks[name] = {
            "model_size_kb": round(size_kb, 2),
            "ram_footprint_mb": round(mem_mb, 2),
            "inference_latency_us": round(avg_latency_us, 2)
        }
        log_print(f"  [Load] {name} loaded (Size: {size_kb:.1f}KB | RAM: ~{mem_mb:.1f}MB | Latency: {avg_latency_us:.1f}us)")

    if getattr(params, "RUN_CUSTOM_RL", True):
        _bench_model("MCA_D3QN", DQN.load, os.path.join(cp_dir, "mca_d3qn_model"))
        
    for algo, cls in [("dqn", DQN), ("ppo", PPO), ("a2c", A2C)]:
        if getattr(params, f"RUN_{algo.upper()}", True):
            _bench_model(algo.upper(), cls.load, os.path.join(cp_dir, f"{algo}_baseline_model"))
            
    if getattr(params, "RUN_TABULAR_QLEARNING", True):
        _bench_model("TABULAR", lambda: TabularQLearning(), os.path.join(cp_dir, "tabular_q_model.json"))

    # if getattr(params, "RUN_MARL_GNN", True):
    #     _bench_model("MARL_GNN", None, os.path.join(cp_dir, "gnn_marl_model.pth"))

    return models, benchmarks

# =====================================================================
# Main Execution Loop
# =====================================================================
def run_experiment_server():
    # 1. Setup Dirs
    N = params.N
    sim_time_s = params.SIM_TIME_S
    phy_rate_bps = params.PHY_RATE_BPS
    QMAX = params.QMAX
    rts_cts_enabled = params.RTS_CTS_ENABLED
    ack_enabled = params.ACK_ENABLED
    
    out_dir, rl_out_dir, trial_name = make_result_dirs(N, phy_rate_bps, QMAX, rts_cts_enabled, ack_enabled)
    log_file_path = os.path.join(out_dir, "logs", "run.log")
    
    def log_print(msg):
        print(msg)
        with open(log_file_path, "a") as lf:
            lf.write(msg + "\n")

    log_print("==========================================================")
    log_print(f" SERVER-GRADE EXPERIMENT RUNNER (Multiprocessing & Telemetry)")
    log_print("==========================================================")
    
    # Move hardware check after potential multiprocessing initialization if needed,
    # or just keep it simple. We'll move the torch import inside to be safe.
    import torch
    num_cpus = os.cpu_count()
    # has_gpu = torch.cuda.is_available() # Avoid calling this here if possible
    log_print(f"Hardware Scan: {num_cpus} CPU Cores Detected")
    
    cfg = Config() # Loads defaults from params natively in base script architecture

    traffic_pps_list = np.linspace(params.SWEEP_MIN_PPS, params.SWEEP_MAX_PPS, params.SWEEP_STEPS).astype(int)
    
    # [Step 1] Baseline
    log_print(f"\n--- Step 1: Baseline MAC Simulations ---")
    results = run_baseline_simulations(cfg, traffic_pps_list, sim_time_s, N, params.SEED, log_print)
    df = pd.DataFrame(results)
    
    # [Step 2] QLearning Offline (Legacy)
    from algorithms.rl.qlearning_selector import run_rl_postprocess
    q_df = pd.DataFrame()
    if getattr(params, "ENABLE_RL_SELECTOR", False) or getattr(params, "RUN_QLEARNING_SELECTOR", False):
        log_print(f"\n--- Step 2: Q-Learning Offline Selector ---")
        q_df = run_rl_postprocess(rl_out_dir, df, params, log_print)
        
    # [Step 3] Parallel Training
    log_print(f"\n--- Step 3: Multiprocessed RL Training ---")
    execute_multiprocess_training(rl_out_dir, RL_TRAINING_TIMESTEPS, log_print)
    
    # [Step 4] Evaluation & Telemetry
    log_print(f"\n--- Step 4: Loading Models & Telemetry Profiling ---")
    models, benchmarks = load_and_benchmark_models(log_print)
    
    # Save Benchmarks JSON
    bench_path = os.path.join(rl_out_dir, "csv", "model_performance_benchmarks.json")
    with open(bench_path, 'w') as f:
        json.dump(benchmarks, f, indent=4)
    log_print(f"Saved strict telemetry to {bench_path}")

    # [Step 5] Sequential Prediction Sweeps
    log_print(f"\n--- Step 5: Sequential Traffic Evaluation ---")
    from experiments.run_experiments import evaluate_rl_selectors, generate_aggregated_rl_plots
    
    if models:
        all_selector_results = evaluate_rl_selectors(models, df, rl_out_dir, log_print)
        log_print(f"\n--- Step 6: Full Aggregated Visualizations ---")
        generate_aggregated_rl_plots(all_selector_results, df, q_df, rl_out_dir, log_print)

    log_print("Experiment Runner Terminated Successfully.")

if __name__ == '__main__':
    # Standard protection for python multiprocessing on Windows/Linux
    # Use 'spawn' to avoid CUDA re-initialization errors in forked subprocesses
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    run_experiment_server()
