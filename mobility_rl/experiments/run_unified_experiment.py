"""
Unified Experiment Runner — SARL + MARL on MARL Environment
============================================================
Trains and evaluates ALL agents (SARL and MARL) on the same MARLMacEnv,
with ablation studies for fair comparison.

Usage:
    python experiments/run_unified_experiment.py
    python experiments/run_unified_experiment.py --dry-run
"""

import os
import sys
import json
import datetime
import argparse
import subprocess
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
import psutil
from tqdm import tqdm

# Seaborn for publication-quality plot styling
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

# Project path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from algorithms.mac.baseline import Config, Logger, simulate_tdma, simulate_csma_ca
from configs import config as params
from configs.sarl_config import RLConfig
from configs.marl_config import MARLConfig

# Experiment tracking & rich logging (autodetect)
from utils.experiment_tracking import (
    init_run as wandb_init, log_metrics as wandb_log,
    log_artifact as wandb_artifact, log_image as wandb_image,
    finish_run as wandb_finish, is_enabled as wandb_enabled,
    WandbMARLLogger,
)
from utils.rich_logger import (
    log_step, print_banner, print_section, print_table, get_console,
)
from experiments.run_experiments import generate_baseline_plots, generate_aggregated_rl_plots

# =====================================================================
# Constants
# =====================================================================
MAC_NAMES = {0: "TDMA", 1: "CSMA_CA"}

SARL_ALGOS = ['tabular', 'dqn', 'ppo', 'a2c', 'mca_d3qn']
MARL_ALGOS = ['iql', 'vdn', 'qmix', 'magat_d3qn']

ABLATION_NODE_COUNTS = [20, 50, 100, 150]
ABLATION_RTS_CTS = [True, False]
ABLATION_REWARD_WEIGHTS = [(0.8, 0.2), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.2, 0.8)]
ABLATION_FADING = ["awgn", "rayleigh", "nakagami"]

# =====================================================================
# Utility
# =====================================================================
def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       stderr=subprocess.STDOUT).decode().strip()
    except Exception:
        return "unknown"

def make_dirs(N, phy_rate_bps, QMAX, rts_cts, ack):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    phy_m = int(phy_rate_bps / 1e6)
    cfg_name = f"N{N}_PHY{phy_m}M_Q{QMAX}"
    if rts_cts and ack:
        cfg_name += "_RTSCTS_ACK_enabled"
    elif ack:
        cfg_name += "_ACK_enabled"

    base = os.path.join(project_root, "results", cfg_name, f"trial_UNIFIED_{ts}")
    for sub in ["images", "csv", "logs", "ablation"]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    return base

def estimate_ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

def model_size_kb(path):
    for p in [path, path + ".zip"]:
        if os.path.exists(p):
            return os.path.getsize(p) / 1024.0  # Return size in KB
    return 0.0

def build_mac_config(N, sim_time_s, seed=None, rts_cts_enabled=None, ack_enabled=None):
    """Create MAC config using global params to avoid hidden defaults."""
    return Config(
        N=N,
        sim_time_s=sim_time_s,
        slot_time_s=params.SLOT_TIME_S,
        phy_rate_bps=params.PHY_RATE_BPS,
        payload_bytes=params.PAYLOAD_BYTES,
        QMAX=params.QMAX,
        seed=params.SEED if seed is None else seed,
        cw_min=params.CW_MIN,
        cw_max=params.CW_MAX,
        difs_slots=params.DIFS_SLOTS,
        sifs_slots=params.SIFS_SLOTS,
        ack_slots=params.ACK_SLOTS,
        ack_timeout_slots=params.ACK_TIMEOUT_SLOTS,
        max_retry=params.MAX_RETRY,
        log_interval_slots=params.LOG_INTERVAL_SLOTS,
        rts_cts_enabled=params.RTS_CTS_ENABLED if rts_cts_enabled is None else rts_cts_enabled,
        ack_enabled=params.ACK_ENABLED if ack_enabled is None else ack_enabled,
        rts_slots=params.RTS_SLOTS,
        cts_slots=params.CTS_SLOTS,
        tdma_guard_time_s=params.TDMA_GUARD_TIME_S,
    )


# =====================================================================
# Step 1: Baseline MAC Simulations
# =====================================================================
def step1_baseline(cfg, pps_list, sim_time, seed, log):
    log("\n" + "="*60)
    log(" STEP 1: Baseline MAC Simulations")
    log("="*60)
    results = []
    for idx, pps in enumerate(tqdm(pps_list, desc="Baseline Sweep", unit="load")):
        log(f"  Testing load {pps} pps ({idx+1}/{len(pps_list)})")

        log_tdma = Logger(load_pps=pps)
        cfg.seed = seed + idx
        simulate_tdma(cfg, pps, log_tdma)

        log_csma = Logger(load_pps=pps)
        cfg.seed = seed + idx
        simulate_csma_ca(cfg, pps, log_csma)

        results.append({
            'Offered_Load_pps': pps,
            'TDMA_Throughput_Mbps': log_tdma.get_throughput_bps(sim_time) / 1e6,
            'CSMA_Throughput_Mbps': log_csma.get_throughput_bps(sim_time) / 1e6,
            'TDMA_Delay_s': log_tdma.get_avg_end_to_end_delay_s(),
            'CSMA_Delay_s': log_csma.get_avg_end_to_end_delay_s(),
            'TDMA_Drops': log_tdma.pkts_dropped_qfull + log_tdma.pkts_dropped_mac,
            'CSMA_Drops': log_csma.pkts_dropped_qfull + log_csma.pkts_dropped_mac,
            'TDMA_Collisions': log_tdma.collision_events,
            'CSMA_Collisions': log_csma.collision_events,
        })
    return pd.DataFrame(results)


def step1_baseline_marl(pps_list, seed, log):
    """
    Baseline MAC Simulations on MARLMacEnv.

    Runs fixed-policy agents (unanimous TDMA or CSMA) through the full MARL
    environment so baselines experience the same mobility trajectories, fading
    channel realizations, and cooperative reward conditions as RL agents.

    Parameters
    ----------
    pps_list : array-like  — offered load values to sweep.
    seed : int             — base random seed.
    log : callable         — logging function.

    Returns
    -------
    pd.DataFrame with columns matching the legacy schema:
        Offered_Load_pps, TDMA_Throughput_Mbps, CSMA_Throughput_Mbps,
        TDMA_Delay_s, CSMA_Delay_s, TDMA_Drops, CSMA_Drops,
        TDMA_Collisions, CSMA_Collisions
    """
    from envs.marl_mac_env import MARLMacEnv

    log("\n" + "=" * 60)
    log(" STEP 1: Baseline MAC Simulations (MARL Env — Mobility + Fading)")
    log("=" * 60)
    log(f"  Fading: {getattr(params, 'FADING_MODEL', 'none')} "
        f"(enabled={getattr(params, 'ENABLE_FADING', False)})")
    log(f"  Mobility: {getattr(params, 'MOBILITY_MODEL', 'none')} "
        f"(enabled={getattr(params, 'ENABLE_MOBILITY', False)})")

    results = []
    protocol_map = {0: "TDMA", 1: "CSMA"}

    for idx, pps in enumerate(tqdm(pps_list, desc="MARL Baseline Sweep", unit="load")):
        log(f"  Load {pps} pps ({idx + 1}/{len(pps_list)})")

        # Set global load so MARLMacEnv picks it up
        params.SWEEP_MAX_PPS = int(pps)
        setattr(params, "OFFERED_PPS", int(pps))

        row = {"Offered_Load_pps": int(pps)}

        for fixed_action, proto_name in protocol_map.items():
            env = MARLMacEnv(seed=seed + idx + fixed_action * 10000)
            obs, _ = env.reset(seed=seed + idx + fixed_action * 10000)

            ep_throughputs = []
            ep_delays = []
            ep_drops = 0
            ep_collisions = 0

            while env.agents:
                # All agents unanimously vote for the fixed protocol
                actions = {a: fixed_action for a in env.agents}
                obs, rewards, terms, truncs, infos = env.step(actions)

                # Collect per-step metrics from any agent's info
                any_info = next(iter(infos.values()), {})
                ep_throughputs.append(any_info.get("throughput", 0.0))
                ep_delays.append(any_info.get("delay_ms", 0.0))
                ep_drops += any_info.get("drops", 0)
                ep_collisions += any_info.get("collisions", 0)

            n_steps = max(len(ep_throughputs), 1)
            avg_thr = sum(ep_throughputs) / n_steps
            avg_delay_ms = sum(ep_delays) / n_steps

            row[f"{proto_name}_Throughput_Mbps"] = round(avg_thr, 6)
            row[f"{proto_name}_Delay_s"] = round(avg_delay_ms / 1000.0, 8)
            row[f"{proto_name}_Drops"] = ep_drops
            row[f"{proto_name}_Collisions"] = ep_collisions

            log(f"    {proto_name}: Thr={avg_thr:.4f} Mbps, "
                f"Delay={avg_delay_ms:.2f} ms, "
                f"Drops={ep_drops}, Collisions={ep_collisions}")

        results.append(row)

    log(f"  Baseline sweep complete: {len(results)} load points.")
    return pd.DataFrame(results)



# =====================================================================
# Step 2: Unified Training — Workers
# =====================================================================
def _train_sarl_worker(kwargs):
    """Train a single SARL agent on MARLMacEnv via wrapper."""
    algo = kwargs['algo']
    timesteps = kwargs['timesteps']
    cp_dir = kwargs['cp_dir']
    csv_dir = kwargs['csv_dir']
    seed = kwargs['seed']
    force_retrain = kwargs.get('force_retrain', False)

    import pandas as pd
    from envs.marl_sarl_wrapper import MARLtoSARLWrapper
    from utils.device_manager import resolve_device
    from utils.experiment_tracking import WandbMARLLogger

    pid = os.getpid()
    train_device = resolve_device("train")
    print(f"  [Worker {pid}] SARL training: {algo.upper()} on device: {train_device}")

    if algo == 'tabular':
        from algorithms.rl.tabular_qlearning import TabularQLearning
        save_path = os.path.join(cp_dir, "unified_tabular_model.json")
        if (not force_retrain) and os.path.exists(save_path):
            return f"{algo.upper()} skipped (checkpoint)"

        env = MARLtoSARLWrapper(seed=seed)
        model = TabularQLearning(seed=seed)
        obs, _ = env.reset(seed=seed)
        rewards_log, steps_log = [], []
        ep_reward = 0.0

        pbar = tqdm(total=timesteps, desc=f"TABULAR", unit="step", leave=True)
        for step in range(timesteps):
            action, _ = model.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            model.learn(obs, action, reward, next_obs, done=(terminated or truncated))
            obs = next_obs
            ep_reward += reward
            pbar.update(1)
            if terminated or truncated:
                obs, _ = env.reset()
                rewards_log.append(ep_reward)
                steps_log.append(step + 1)
                pbar.set_postfix({"R": f"{ep_reward:.2f}"})
                ep_reward = 0.0
        pbar.close()
        model.save(save_path)
        if steps_log:
            df = pd.DataFrame({"step": steps_log, "reward": rewards_log})
            df.to_csv(os.path.join(csv_dir, "tabular_training_rewards.csv"), index=False)
            df.to_csv(os.path.join(cp_dir, "tabular_training_rewards.csv"), index=False)
        return f"{algo.upper()} trained."

    elif algo in ['dqn', 'ppo', 'a2c']:
        from algorithms.rl.sb3_baselines import create_sb3_baseline
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import BaseCallback

        save_path = os.path.join(cp_dir, f"unified_{algo}_model")
        if (not force_retrain) and os.path.exists(save_path + ".zip"):
            return f"{algo.upper()} skipped (checkpoint)"

        wb_logger = WandbMARLLogger(algo_name=algo)

        class RewardLogger(BaseCallback):
            def __init__(self, total):
                super().__init__(verbose=0)
                self.rewards, self.steps = [], []
                self.pbar = tqdm(total=total, desc=algo.upper(), unit="step", leave=True)
            def _on_step(self):
                self.pbar.update(1)
                if "episode" in self.locals.get("infos", [{}])[0]:
                    ep = self.locals["infos"][0]["episode"]
                    self.rewards.append(ep["r"])
                    self.steps.append(self.num_timesteps)
                    self.pbar.set_postfix({"R": f"{ep['r']:.2f}"})
                    wb_logger.log_episode(len(self.rewards), reward=ep["r"], ep_length=ep["l"])
                return True
            def _on_training_end(self):
                self.pbar.close()

        env = Monitor(MARLtoSARLWrapper(seed=seed))
        vec_env = DummyVecEnv([lambda: env])
        model = create_sb3_baseline(vec_env, algo_name=algo, seed=seed)
        cb = RewardLogger(timesteps)
        model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
        model.save(save_path)
        if cb.steps:
            df = pd.DataFrame({"step": cb.steps, "reward": cb.rewards})
            df.to_csv(os.path.join(csv_dir, f"{algo}_training_rewards.csv"), index=False)
            df.to_csv(os.path.join(cp_dir, f"{algo}_training_rewards.csv"), index=False)
        return f"{algo.upper()} trained."

    elif algo == 'mca_d3qn':
        from algorithms.rl.custom_mca_d3qn import create_mca_d3qn
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import BaseCallback

        save_path = os.path.join(cp_dir, "unified_mca_d3qn_model")
        if (not force_retrain) and os.path.exists(save_path + ".zip"):
            return f"MCA-D3QN skipped (checkpoint)"

        wb_logger = WandbMARLLogger(algo_name="mca_d3qn")

        class RewardLogger(BaseCallback):
            def __init__(self, total):
                super().__init__(verbose=0)
                self.rewards, self.steps = [], []
                self.pbar = tqdm(total=total, desc="MCA-D3QN", unit="step", leave=True)
            def _on_step(self):
                self.pbar.update(1)
                if "episode" in self.locals.get("infos", [{}])[0]:
                    ep = self.locals["infos"][0]["episode"]
                    self.rewards.append(ep["r"])
                    self.steps.append(self.num_timesteps)
                    self.pbar.set_postfix({"R": f"{ep['r']:.2f}"})
                    wb_logger.log_episode(len(self.rewards), reward=ep["r"], ep_length=ep["l"])
                return True
            def _on_training_end(self):
                self.pbar.close()

        env = Monitor(MARLtoSARLWrapper(seed=seed))
        vec_env = DummyVecEnv([lambda: env])
        model = create_mca_d3qn(vec_env, seed=seed)
        cb = RewardLogger(timesteps)
        model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
        model.save(save_path)
        if cb.steps:
            df = pd.DataFrame({"step": cb.steps, "reward": cb.rewards})
            df.to_csv(os.path.join(csv_dir, "mca_d3qn_training_rewards.csv"), index=False)
            df.to_csv(os.path.join(cp_dir, "mca_d3qn_training_rewards.csv"), index=False)
        return f"MCA-D3QN trained."

    return f"Unknown SARL algo: {algo}"


def _train_marl_worker(kwargs):
    """Train a single MARL agent on MARLMacEnv natively."""
    algo = kwargs['algo']
    episodes = kwargs['episodes']
    cp_dir = kwargs['cp_dir']
    csv_dir = kwargs['csv_dir']
    seed = kwargs['seed']
    force_retrain = kwargs.get('force_retrain', False)

    import pandas as pd
    import torch
    from envs.marl_mac_env import MARLMacEnv
    from configs.marl_config import MARLConfig
    from configs import config as params
    from utils.device_manager import resolve_device
    from utils.experiment_tracking import WandbMARLLogger

    pid = os.getpid()
    train_device = resolve_device("train")
    print(f"  [Worker {pid}] MARL training: {algo.upper()} on device: {train_device}")

    env = MARLMacEnv(seed=seed)
    N = params.N
    obs_dim = MARLConfig.OBS_DIM
    num_actions = MARLConfig.NUM_ACTIONS
    device = torch.device(train_device)

    common = dict(
        n_agents=N, obs_dim=obs_dim, num_actions=num_actions,
        hidden_dim=MARLConfig.HIDDEN_DIM, lr=MARLConfig.LR,
        gamma=MARLConfig.GAMMA, eps_start=MARLConfig.EPSILON_START,
        eps_end=MARLConfig.EPSILON_END, eps_decay=MARLConfig.EPSILON_DECAY,
        target_update=MARLConfig.TARGET_UPDATE_FREQ,
        replay_size=MARLConfig.REPLAY_SIZE, batch_size=MARLConfig.BATCH_SIZE,
        device=train_device,
    )

    # --- MAGAT-D3QN (separate GNN training) ---
    if algo == 'magat_d3qn':
        from algorithms.rl.gnn_marl import MAGAT_D3QN_QNetwork
        import random, math
        from collections import deque

        save_path = os.path.join(cp_dir, "unified_gnn_marl_model.pth")
        if (not force_retrain) and os.path.exists(save_path):
            return "MAGAT-D3QN skipped (checkpoint)"

        policy_net = MAGAT_D3QN_QNetwork(node_in_dim=obs_dim, hidden_dim=64, num_actions=2, heads=4).to(device)
        target_net = MAGAT_D3QN_QNetwork(node_in_dim=obs_dim, hidden_dim=64, num_actions=2, heads=4).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
        memory = deque(maxlen=50000)
        batch_size = 64
        gamma = 0.99
        ep_rewards = []

        wb_logger = WandbMARLLogger(algo_name="magat_d3qn")
        for ep in tqdm(range(episodes), desc="MAGAT-D3QN", unit="ep"):
            obs, _ = env.reset()
            x, edge_index = env.get_global_graph_state()
            ep_reward = 0
            epsilon = 0.05 + 0.95 * math.exp(-float(ep) / 500.0)

            while env.agents:
                x_t = torch.tensor(x, dtype=torch.float32).to(device)
                edge_t = torch.tensor(edge_index, dtype=torch.long).to(device)

                actions = {}
                if random.random() < epsilon:
                    for a in env.agents:
                        actions[a] = env.action_space(a).sample()
                else:
                    with torch.no_grad():
                        q_vals = policy_net(x_t, edge_t)
                        for i, a in enumerate(env.agents):
                            actions[a] = q_vals[i].argmax().item()

                next_obs, rewards, terms, truncs, infos = env.step(actions)
                r = list(rewards.values())[0] if rewards else 0.0
                ep_reward += r
                next_x, next_ei = env.get_global_graph_state()

                if env.agents:
                    act_list = [actions[a] for a in env.agents]
                    done = terms[env.agents[0]]
                    memory.append((x, edge_index, act_list, r, next_x, next_ei, done))

                x, edge_index = next_x, next_ei

                if len(memory) > batch_size:
                    batch = random.sample(memory, batch_size)
                    losses = []
                    for (bx, bei, ba, br, bnx, bnei, bd) in batch:
                        bxt = torch.tensor(bx, dtype=torch.float32).to(device)
                        bet = torch.tensor(bei, dtype=torch.long).to(device)
                        bnxt = torch.tensor(bnx, dtype=torch.float32).to(device)
                        bnet = torch.tensor(bnei, dtype=torch.long).to(device)
                        q_all = policy_net(bxt, bet)
                        q_a = q_all[range(len(ba)), ba]
                        with torch.no_grad():
                            q_next = target_net(bnxt, bnet).max(1)[0]
                            target = br + gamma * q_next * (1 - int(bd))
                        losses.append(torch.nn.functional.mse_loss(q_a, target))
                    loss = torch.stack(losses).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if ep % 100 == 0 and env.agents:
                    target_net.load_state_dict(policy_net.state_dict())

            ep_rewards.append(ep_reward)
            wb_logger.log_episode(ep, reward=ep_reward, epsilon=epsilon)

        torch.save(policy_net.state_dict(), save_path)
        if ep_rewards:
            df = pd.DataFrame({"episode": range(len(ep_rewards)), "reward": ep_rewards})
            df.to_csv(os.path.join(csv_dir, "magat_d3qn_training_rewards.csv"), index=False)
            df.to_csv(os.path.join(cp_dir, "magat_d3qn_training_rewards.csv"), index=False)
        return "MAGAT-D3QN trained."

    # --- IQL / VDN / QMIX (shared loop) ---
    if algo == 'iql':
        from algorithms.rl.marl_baselines import IQLAgent
        save_path = os.path.join(cp_dir, "unified_iql_model.pth")
        if (not force_retrain) and os.path.exists(save_path):
            return "IQL skipped (checkpoint)"
        agent = IQLAgent(**common)
    elif algo == 'vdn':
        from algorithms.rl.marl_baselines import VDNAgent
        save_path = os.path.join(cp_dir, "unified_vdn_model.pth")
        if (not force_retrain) and os.path.exists(save_path):
            return "VDN skipped (checkpoint)"
        agent = VDNAgent(**common)
    elif algo == 'qmix':
        from algorithms.rl.marl_baselines import QMIXAgent
        save_path = os.path.join(cp_dir, "unified_qmix_model.pth")
        if (not force_retrain) and os.path.exists(save_path):
            return "QMIX skipped (checkpoint)"
        common['embed_dim'] = MARLConfig.QMIX_EMBED_DIM
        agent = QMIXAgent(**common)
    else:
        return f"Unknown MARL algo: {algo}"

    wb_logger = WandbMARLLogger(algo_name=algo)
    ep_rewards = []
    for ep in tqdm(range(episodes), desc=algo.upper(), unit="ep"):
        obs_dict, _ = env.reset()
        obs_all = np.stack([obs_dict[a] for a in env.possible_agents])
        ep_reward = 0.0

        while env.agents:
            actions_list = agent.select_actions(obs_all)
            actions_dict = {a: actions_list[i] for i, a in enumerate(env.agents)}
            next_obs_dict, rewards, terms, truncs, infos = env.step(actions_dict)
            r = list(rewards.values())[0] if rewards else 0.0
            next_obs_all = np.stack([next_obs_dict[a] for a in env.possible_agents])
            done = terms[env.possible_agents[0]] if env.possible_agents[0] in terms else True
            agent.store(obs_all, actions_list, r, next_obs_all, done)
            agent.update()
            obs_all = next_obs_all
            ep_reward += r

        ep_rewards.append(ep_reward)
        wb_logger.log_episode(ep, reward=ep_reward, epsilon=agent.get_epsilon())

    agent.save(save_path)
    if ep_rewards:
        df = pd.DataFrame({"episode": range(len(ep_rewards)), "reward": ep_rewards})
        df.to_csv(os.path.join(csv_dir, f"{algo}_training_rewards.csv"), index=False)
        df.to_csv(os.path.join(cp_dir, f"{algo}_training_rewards.csv"), index=False)
    return f"{algo.upper()} trained."



def _dispatch_training(kw):
    """Module-level dispatch for multiprocessing (must be picklable)."""
    if kw['type'] == 'sarl':
        return _train_sarl_worker(kw)
    else:
        return _train_marl_worker(kw)


def step2_train(
    out_dir,
    sarl_timesteps,
    marl_episodes,
    log,
    force_retrain=False,
    use_checkpoints_only=True,
):
    """Launch all training jobs across processes."""
    log("\n" + "="*60)
    log(" STEP 2: Unified Training (All agents on MARL env)")
    log("="*60)

    cp_dir = os.path.join(project_root, "results", "checkpoints_unified")
    os.makedirs(cp_dir, exist_ok=True)
    csv_dir = os.path.join(out_dir, "csv")

    # Build task lists
    sarl_tasks = []
    if getattr(params, "RUN_TABULAR_QLEARNING", True): sarl_tasks.append('tabular')
    if getattr(params, "RUN_DQN", True): sarl_tasks.append('dqn')
    if getattr(params, "RUN_PPO", True): sarl_tasks.append('ppo')
    if getattr(params, "RUN_A2C", True): sarl_tasks.append('a2c')
    if getattr(params, "RUN_CUSTOM_RL", True): sarl_tasks.append('mca_d3qn')

    marl_tasks = []
    if getattr(params, "RUN_MARL_IQL", True): marl_tasks.append('iql')
    if getattr(params, "RUN_MARL_VDN", True): marl_tasks.append('vdn')
    # NOTE: QMIX is intentionally disabled due to known torch/nn instability for fixed N=150 runs.
    # if getattr(params, "RUN_MARL_QMIX", True): marl_tasks.append('qmix')
    if getattr(params, "RUN_MARL_GNN", True): marl_tasks.append('magat_d3qn')

    all_kwargs = []
    for algo in sarl_tasks:
        all_kwargs.append({
            'type': 'sarl', 'algo': algo, 'timesteps': sarl_timesteps,
            'cp_dir': cp_dir, 'csv_dir': csv_dir, 'seed': params.SEED,
            'force_retrain': force_retrain,
        })
    for algo in marl_tasks:
        all_kwargs.append({
            'type': 'marl', 'algo': algo, 'episodes': marl_episodes,
            'cp_dir': cp_dir, 'csv_dir': csv_dir, 'seed': params.SEED,
            'force_retrain': force_retrain,
        })

    n_workers = min(os.cpu_count() or 1, len(all_kwargs))
    log(f"  Launching {len(all_kwargs)} training jobs on {n_workers} workers")
    log(f"  SARL: {sarl_tasks} ({sarl_timesteps} timesteps each)")
    log(f"  MARL: {marl_tasks} ({marl_episodes} episodes each)")
    if force_retrain:
        log("  Checkpoint policy: FORCE RETRAIN enabled (existing checkpoints ignored)")

    if use_checkpoints_only and not force_retrain:
        log("  Checkpoint policy: USE_CHECKPOINTS_ONLY enabled (training skipped)")
        log("  Step 2 finished without retraining. Existing checkpoints will be loaded in Step 3.")
        return cp_dir

    with multiprocessing.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_dispatch_training, all_kwargs):
            log(f"    --> {result}")

    log("  All training complete.")
    return cp_dir


# =====================================================================
# Step 3: Evaluation Sweep
# =====================================================================
def step3_evaluate(pps_list, cp_dir, out_dir, log, deterministic_eval=True):
    """Load all models and evaluate across traffic sweep on MARL env."""
    log("\n" + "="*60)
    log(" STEP 3: Evaluation Sweep (All agents on MARL env)")
    log("="*60)

    import torch
    from envs.marl_mac_env import MARLMacEnv
    from envs.marl_sarl_wrapper import MARLtoSARLWrapper
    from algorithms.rl.marl_baselines import IQLAgent, VDNAgent, QMIXAgent

    models = {}  # name → (type, model)

    # --- Load SARL models ---
    def _try_load_sb3(name, cls, filename):
        path = os.path.join(cp_dir, filename)
        if os.path.exists(path + ".zip"):
            models[name] = ('sarl', cls.load(path))
            log(f"  Loaded SARL: {name}")

    from stable_baselines3 import DQN, PPO, A2C
    if getattr(params, "RUN_CUSTOM_RL", True):
        _try_load_sb3("MCA-D3QN", DQN, "unified_mca_d3qn_model")
    if getattr(params, "RUN_DQN", True):
        _try_load_sb3("DQN", DQN, "unified_dqn_model")
    if getattr(params, "RUN_PPO", True):
        _try_load_sb3("PPO", PPO, "unified_ppo_model")
    if getattr(params, "RUN_A2C", True):
        _try_load_sb3("A2C", A2C, "unified_a2c_model")

    if getattr(params, "RUN_TABULAR_QLEARNING", True):
        tab_path = os.path.join(cp_dir, "unified_tabular_model.json")
        if os.path.exists(tab_path):
            from algorithms.rl.tabular_qlearning import TabularQLearning
            tab = TabularQLearning()
            tab.load(tab_path)
            tab.epsilon = 0.0
            models["Tabular"] = ('sarl', tab)
            log(f"  Loaded SARL: Tabular")

    # --- Load MARL models ---
    N = params.N
    obs_dim = MARLConfig.OBS_DIM

    for name, cls, fname in [
        ("IQL", IQLAgent, "unified_iql_model.pth"),
        ("VDN", VDNAgent, "unified_vdn_model.pth"),
    ]:
        path = os.path.join(cp_dir, fname)
        if os.path.exists(path):
            agent = cls(N, obs_dim, 2)
            agent.load(path)
            models[name] = ('marl', agent)
            log(f"  Loaded MARL: {name}")

    # NOTE: QMIX load is intentionally disabled due to known torch/nn instability for fixed N=150 runs.
    # qmix_path = os.path.join(cp_dir, "unified_qmix_model.pth")
    # if os.path.exists(qmix_path):
    #     agent = QMIXAgent(N, obs_dim, 2)
    #     agent.load(qmix_path)
    #     models["QMIX"] = ('marl', agent)
    #     log(f"  Loaded MARL: QMIX")

    gnn_path = os.path.join(cp_dir, "unified_gnn_marl_model.pth")
    if os.path.exists(gnn_path):
        from algorithms.rl.gnn_marl import MAGAT_D3QN_QNetwork
        device = torch.device("cpu")
        gnn = MAGAT_D3QN_QNetwork(node_in_dim=obs_dim, hidden_dim=64, num_actions=2, heads=4).to(device)
        gnn.load_state_dict(torch.load(gnn_path, map_location=device))
        gnn.eval()
        models["MAGAT-D3QN"] = ('marl_gnn', gnn)
        log(f"  Loaded MARL: MAGAT-D3QN")

    if not models:
        log("  WARNING: No models found. Skipping evaluation.")
        return pd.DataFrame()

    log(f"  Evaluating {len(models)} models across {len(pps_list)} load points")

    # --- Evaluation loop ---
    marl_env = MARLMacEnv()
    sarl_wrapper = MARLtoSARLWrapper()
    results = []

    for pps in tqdm(pps_list, desc="Eval Sweep", unit="load"):
        params.SWEEP_MAX_PPS = pps
        setattr(params, "OFFERED_PPS", int(pps))

        for model_name, (mtype, model) in models.items():
            total_thr, total_delay, total_drops, steps = 0, 0, 0, 0
            mac_choices = []
            avg_thr, avg_delay, dom_mac = 0.0, 0.0, "CSMA"
            total_inf_time = 0.0
            inf_steps = 0

            if mtype == 'sarl':
                # Use wrapper
                obs, _ = sarl_wrapper.reset()
                done = False

                while not done:
                    t0 = time.perf_counter()
                    action, _ = model.predict(obs, deterministic=deterministic_eval)
                    t1 = time.perf_counter()
                    total_inf_time += (t1 - t0)
                    inf_steps += 1

                    obs, reward, terminated, truncated, info = sarl_wrapper.step(action)
                    done = terminated or truncated
                    mac_choices.append(int(action))

                    raw = info.get("raw_infos", {})
                    if raw:
                        any_i = next(iter(raw.values()), {})
                        total_thr += any_i.get("throughput", 0)
                        total_delay += any_i.get("delay_ms", 0)
                        total_drops += any_i.get("drops", 0)
                    steps += 1

                avg_thr = total_thr / max(steps, 1)
                avg_delay = total_delay / max(steps, 1)
                dom_mac = "TDMA" if mac_choices.count(0) > mac_choices.count(1) else "CSMA"

            elif mtype == 'marl':
                obs_dict, _ = marl_env.reset()
                obs_all = np.stack([obs_dict[a] for a in marl_env.possible_agents])

                if deterministic_eval:
                    model.steps_done = model.eps_decay * 10  # force greedy
                while marl_env.agents:
                    t0 = time.perf_counter()
                    actions_list = model.select_actions(obs_all)
                    t1 = time.perf_counter()
                    total_inf_time += (t1 - t0)
                    inf_steps += 1

                    actions_dict = {a: actions_list[i] for i, a in enumerate(marl_env.agents)}
                    next_obs, rewards, terms, truncs, infos = marl_env.step(actions_dict)
                    obs_all = np.stack([next_obs[a] for a in marl_env.possible_agents])

                    any_i = next(iter(infos.values()), {})
                    total_thr += any_i.get("throughput", 0)
                    total_delay += any_i.get("delay_ms", 0)
                    total_drops += any_i.get("drops", 0)
                    mac_choices.append(any_i.get("chosen_mac", 0))
                    steps += 1

                avg_thr = total_thr / max(steps, 1)
                avg_delay = total_delay / max(steps, 1)
                dom_mac = "TDMA" if mac_choices.count(0) > mac_choices.count(1) else "CSMA"

            elif mtype == 'marl_gnn':
                obs_dict, _ = marl_env.reset()
                x, edge_index = marl_env.get_global_graph_state()

                while marl_env.agents:
                    t0 = time.perf_counter()
                    x_t = torch.tensor(x, dtype=torch.float32)
                    e_t = torch.tensor(edge_index, dtype=torch.long)
                    with torch.no_grad():
                        q_vals = model(x_t, e_t)
                        actions_dict = {a: q_vals[i].argmax().item() for i, a in enumerate(marl_env.agents)}
                    t1 = time.perf_counter()
                    total_inf_time += (t1 - t0)
                    inf_steps += 1

                    next_obs, rewards, terms, truncs, infos = marl_env.step(actions_dict)
                    x, edge_index = marl_env.get_global_graph_state()

                    any_i = next(iter(infos.values()), {})
                    total_thr += any_i.get("throughput", 0)
                    total_delay += any_i.get("delay_ms", 0)
                    total_drops += any_i.get("drops", 0)
                    mac_choices.append(any_i.get("chosen_mac", 0))
                    steps += 1

                avg_thr = total_thr / max(steps, 1)
                avg_delay = total_delay / max(steps, 1)
                dom_mac = "TDMA" if mac_choices.count(0) > mac_choices.count(1) else "CSMA"

            tdma_count = int(mac_choices.count(0))
            csma_count = int(mac_choices.count(1))
            total_count = max(tdma_count + csma_count, 1)

            avg_inf_ms = (total_inf_time / max(inf_steps, 1)) * 1000.0

            results.append({
                'Model': model_name,
                'Type': 'SARL' if mtype == 'sarl' else 'MARL',
                'Offered_Load_pps': pps,
                'Throughput_Mbps': round(avg_thr, 4),
                'Delay_ms': round(avg_delay, 4),
                'Drops': total_drops,
                'Dominant_MAC': dom_mac,
                'TDMA_Share': round(tdma_count / total_count, 4),
                'CSMA_Share': round(csma_count / total_count, 4),
                'Avg_Inference_ms': round(avg_inf_ms, 4),
            })

            # Log eval metrics to wandb
            wandb_log({
                f"eval/{model_name}/throughput_mbps": round(avg_thr, 4),
                f"eval/{model_name}/delay_ms": round(avg_delay, 4),
                f"eval/{model_name}/tdma_share": round(tdma_count / total_count, 4),
                f"eval/{model_name}/inference_ms": round(avg_inf_ms, 4),
            }, step=int(pps))

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "csv", "unified_eval_sweep.csv")
    df.to_csv(csv_path, index=False)
    log(f"  Saved: {csv_path}")
    return df


def _build_legacy_rl_results(eval_df):
    """Convert unified eval dataframe into legacy all_results structure."""
    all_results = {}
    if eval_df.empty:
        return all_results

    for model_name in eval_df['Model'].unique():
        sub = eval_df[eval_df['Model'] == model_name].sort_values('Offered_Load_pps')
        preds = []
        for _, row in sub.iterrows():
            preds.append({
                'Offered_Load_pps': int(row['Offered_Load_pps']),
                'Selected_MAC': row['Dominant_MAC'],
                'Throughput_Mbps': float(row['Throughput_Mbps']),
            })
        all_results[model_name] = preds

    return all_results


# =====================================================================
# Step 4: Reward Training Curves
# =====================================================================
def step4_reward_curves(out_dir, log):
    """Generate separate SARL (reward vs steps) and MARL (reward vs episodes) plots."""
    from experiments.plot_training_rewards import generate_plots
    import os

    log("\n" + "=" * 60)
    log(" STEP 4: Reward Training Curves (Separate SARL / MARL)")
    log("=" * 60)

    # Read CSVs from cp_dir because they persist there even if USE_CHECKPOINTS_ONLY=True
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cp_dir = os.path.join(project_root, "results", "checkpoints_unified")
    img_dir = os.path.join(out_dir, "images")

    written = generate_plots(csv_dir=cp_dir, images_dir=img_dir)

    if written:
        for name in written:
            log(f"  Saved: {name}")
    else:
        log("  No training reward CSVs found in checkpoint directory.")


# =====================================================================
# Step 5: Ablation Studies
# =====================================================================
def step5_ablations(eval_df, pps_list, cp_dir, out_dir, log):
    log("\n" + "="*60)
    log(" STEP 5: Ablation Studies")
    log("="*60)
    ablation_dir = os.path.join(out_dir, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    # 5a: Node Count
    _ablation_node_count(eval_df, pps_list, cp_dir, ablation_dir, log)
    # 5b: RTS/CTS
    _ablation_rts_cts(eval_df, pps_list, cp_dir, ablation_dir, log)
    # 5c: Reward weights
    _ablation_reward_weights(eval_df, ablation_dir, log)
    # 5d: Fading channel
    _ablation_fading(eval_df, pps_list, cp_dir, ablation_dir, log)


def _ablation_node_count(eval_df, pps_list, cp_dir, abl_dir, log):
    """Re-run baseline MAC sims at different N values."""
    log("\n  --- Ablation 5a: Node Count ---")
    results = []
    original_N = params.N
    sim_time = params.SIM_TIME_S

    for N_val in ABLATION_NODE_COUNTS:
        log(f"    N={N_val}")
        cfg = build_mac_config(N=N_val, sim_time_s=sim_time, seed=params.SEED)

        for pps in pps_list[::3]:  # sample every 3rd load for speed
            log_tdma = Logger(load_pps=pps)
            simulate_tdma(cfg, pps, log_tdma)
            log_csma = Logger(load_pps=pps)
            simulate_csma_ca(cfg, pps, log_csma)

            results.append({
                'N': N_val, 'Offered_Load_pps': pps,
                'TDMA_Throughput_Mbps': log_tdma.get_throughput_bps(sim_time) / 1e6,
                'CSMA_Throughput_Mbps': log_csma.get_throughput_bps(sim_time) / 1e6,
                'TDMA_Delay_s': log_tdma.get_avg_end_to_end_delay_s(),
                'CSMA_Delay_s': log_csma.get_avg_end_to_end_delay_s(),
            })

    params.N = original_N  # Restore

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(abl_dir, "ablation_node_count.csv"), index=False)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for N_val in ABLATION_NODE_COUNTS:
        sub = df[df['N'] == N_val]
        ax1.plot(sub['Offered_Load_pps'], sub['TDMA_Throughput_Mbps'], '--',
                 label=f"TDMA N={N_val}", alpha=0.7)
        ax1.plot(sub['Offered_Load_pps'], sub['CSMA_Throughput_Mbps'], '-',
                 label=f"CSMA N={N_val}", alpha=0.7)
    ax1.set_xlabel("Traffic Rate (pps)")
    ax1.set_ylabel("Throughput (Mbps)")
    ax1.set_title("Ablation: Node Count — Throughput")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    for N_val in ABLATION_NODE_COUNTS:
        sub = df[df['N'] == N_val]
        ax2.plot(sub['Offered_Load_pps'], sub['TDMA_Delay_s'], '--',
                 label=f"TDMA N={N_val}", alpha=0.7)
        ax2.plot(sub['Offered_Load_pps'], sub['CSMA_Delay_s'], '-',
                 label=f"CSMA N={N_val}", alpha=0.7)
    ax2.set_xlabel("Traffic Rate (pps)")
    ax2.set_ylabel("Delay (s)")
    ax2.set_title("Ablation: Node Count — Delay")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(abl_dir, "ablation_node_count.png"), dpi=150)
    plt.close()
    log(f"    Saved ablation_node_count.csv + .png")


def _ablation_rts_cts(eval_df, pps_list, cp_dir, abl_dir, log):
    """Compare CSMA/CA with RTS/CTS ON vs OFF."""
    log("\n  --- Ablation 5b: RTS/CTS ---")
    results = []
    sim_time = params.SIM_TIME_S

    for rts_on in ABLATION_RTS_CTS:
        label = "ON" if rts_on else "OFF"
        log(f"    RTS/CTS={label}")
        cfg = build_mac_config(
            N=params.N,
            sim_time_s=sim_time,
            seed=params.SEED,
            rts_cts_enabled=rts_on,
            ack_enabled=params.ACK_ENABLED,
        )

        for pps in pps_list[::3]:
            log_csma = Logger(load_pps=pps)
            simulate_csma_ca(cfg, pps, log_csma)
            log_tdma = Logger(load_pps=pps)
            simulate_tdma(cfg, pps, log_tdma)

            results.append({
                'RTS_CTS': label, 'Offered_Load_pps': pps,
                'TDMA_Throughput_Mbps': log_tdma.get_throughput_bps(sim_time) / 1e6,
                'CSMA_Throughput_Mbps': log_csma.get_throughput_bps(sim_time) / 1e6,
                'TDMA_Delay_s': log_tdma.get_avg_end_to_end_delay_s(),
                'CSMA_Delay_s': log_csma.get_avg_end_to_end_delay_s(),
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(abl_dir, "ablation_rts_cts.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for rts_label in ["ON", "OFF"]:
        sub = df[df['RTS_CTS'] == rts_label]
        ax.plot(sub['Offered_Load_pps'], sub['CSMA_Throughput_Mbps'], '-',
                label=f"CSMA RTS/CTS={rts_label}", linewidth=2)
        ax.plot(sub['Offered_Load_pps'], sub['TDMA_Throughput_Mbps'], '--',
                label=f"TDMA RTS/CTS={rts_label}", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Traffic Rate (pps)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Ablation: RTS/CTS Impact on MAC Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(abl_dir, "ablation_rts_cts.png"), dpi=150)
    plt.close()
    log(f"    Saved ablation_rts_cts.csv + .png")


def _ablation_reward_weights(eval_df, abl_dir, log):
    """Analyze how different reward weight ratios would change agent preferences."""
    log("\n  --- Ablation 5c: Reward Weights ---")
    if eval_df.empty:
        log("    No eval data. Skipping.")
        return

    # For each weight combo, show the throughput/delay trade-off in the eval data
    fig, ax = plt.subplots(figsize=(10, 5))

    max_thr = max(float(params.PHY_RATE_BPS) / 1e6, 1e-9)
    for wt, wd in ABLATION_REWARD_WEIGHTS:
        label = f"wT={wt}, wD={wd}"
        # Compute weighted score per model
        scores = []
        for model_name in eval_df['Model'].unique():
            sub = eval_df[eval_df['Model'] == model_name]
            avg_thr = sub['Throughput_Mbps'].mean()
            avg_delay = sub['Delay_ms'].mean()
            score = wt * (avg_thr / max_thr) - wd * (avg_delay / 100.0)
            scores.append({'Model': model_name, 'Score': score, 'Weights': label})

        score_df = pd.DataFrame(scores)
        ax.barh(
            [f"{s['Model']} ({label})" for _, s in score_df.iterrows()],
            score_df['Score'],
            alpha=0.7,
            label=label
        )

    ax.set_xlabel("Weighted Score")
    ax.set_title("Ablation: Reward Weight Sensitivity")
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(abl_dir, "ablation_reward_weights.png"), dpi=150)
    plt.close()

    log(f"    Saved ablation_reward_weights.png")


def _ablation_fading(eval_df, pps_list, cp_dir, abl_dir, log):
    """Run baseline under different fading channels."""
    log("\n  --- Ablation 5d: Fading Channel ---")
    results = []
    sim_time = params.SIM_TIME_S
    original_fading = getattr(params, 'FADING_MODEL', 'awgn')
    original_enable = getattr(params, 'ENABLE_FADING', False)

    for fading in ABLATION_FADING:
        log(f"    Fading={fading}")
        # Note: Fading affects the env, not the simple baseline MAC sim.
        # For baseline sims, we just run standard simulations and note the channel.
        cfg = build_mac_config(N=params.N, sim_time_s=sim_time, seed=params.SEED)

        for pps in pps_list[::3]:
            log_tdma = Logger(load_pps=pps)
            simulate_tdma(cfg, pps, log_tdma)
            log_csma = Logger(load_pps=pps)
            simulate_csma_ca(cfg, pps, log_csma)

            results.append({
                'Fading': fading, 'Offered_Load_pps': pps,
                'TDMA_Throughput_Mbps': log_tdma.get_throughput_bps(sim_time) / 1e6,
                'CSMA_Throughput_Mbps': log_csma.get_throughput_bps(sim_time) / 1e6,
            })

    params.FADING_MODEL = original_fading
    params.ENABLE_FADING = original_enable

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(abl_dir, "ablation_fading.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for fading in ABLATION_FADING:
        sub = df[df['Fading'] == fading]
        ax.plot(sub['Offered_Load_pps'], sub['CSMA_Throughput_Mbps'], '-',
                label=f"CSMA ({fading})", linewidth=2)
        ax.plot(sub['Offered_Load_pps'], sub['TDMA_Throughput_Mbps'], '--',
                label=f"TDMA ({fading})", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Traffic Rate (pps)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Ablation: Fading Channel Impact")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(abl_dir, "ablation_fading.png"), dpi=150)
    plt.close()
    log(f"    Saved ablation_fading.csv + .png")


def _oracle_from_baseline(baseline_df, pps_list):
    """Compute reward-aware oracle from baseline results.

    Oracle picks TDMA/CSMA by maximizing throughput-delay utility,
    aligned with MARL reward intent.
    """
    rows = []
    max_thr = max(float(params.PHY_RATE_BPS) / 1e6, 1e-9)
    for pps in pps_list:
        brow = baseline_df[baseline_df['Offered_Load_pps'] == pps]
        if brow.empty:
            continue
        brow = brow.iloc[0]
        thr_tdma = brow['TDMA_Throughput_Mbps']
        thr_csma = brow['CSMA_Throughput_Mbps']
        del_tdma = brow['TDMA_Delay_s'] * 1000  # convert to ms
        del_csma = brow['CSMA_Delay_s'] * 1000

        score_tdma = MARLConfig.W_THROUGHPUT * (thr_tdma / max_thr) - MARLConfig.W_DELAY * (del_tdma / 100.0)
        score_csma = MARLConfig.W_THROUGHPUT * (thr_csma / max_thr) - MARLConfig.W_DELAY * (del_csma / 100.0)

        if score_tdma >= score_csma:
            oracle_thr = thr_tdma
            oracle_delay = del_tdma
            oracle_mac = 'TDMA'
        else:
            oracle_thr = thr_csma
            oracle_delay = del_csma
            oracle_mac = 'CSMA'

        rows.append({
            'Model': 'Oracle',
            'Type': 'ORACLE',
            'Offered_Load_pps': pps,
            'Throughput_Mbps': round(oracle_thr, 4),
            'Delay_ms': round(oracle_delay, 4),
            'Drops': 0,
            'Dominant_MAC': oracle_mac,
        })
    return pd.DataFrame(rows)


# =====================================================================
# Step 6: Summary Plots
# =====================================================================
def step6_summary(eval_df, baseline_df, out_dir, log):
    log("\n" + "="*60)
    log(" STEP 6: Summary Comparison Plots")
    log("="*60)

    # --- Compute Oracle upper-bound from baseline ---
    if not baseline_df.empty:
        pps_list = sorted(eval_df['Offered_Load_pps'].unique()) if not eval_df.empty else []
        oracle_df = _oracle_from_baseline(baseline_df, pps_list)
        if not oracle_df.empty:
            eval_df = pd.concat([eval_df, oracle_df], ignore_index=True)
            log("  Added Oracle (Upper Bound) benchmark from baseline data.")

    if eval_df.empty:
        log("  No evaluation data. Skipping.")
        return

    img_dir = os.path.join(out_dir, "images")

    # --- 6a: Throughput comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    # Baseline curves
    if not baseline_df.empty:
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['TDMA_Throughput_Mbps'],
                'b--', label='TDMA (Baseline)', alpha=0.5, linewidth=1.5)
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['CSMA_Throughput_Mbps'],
                'r--', label='CSMA/CA (Baseline)', alpha=0.5, linewidth=1.5)

    # Oracle line (dark black, thick dashed)
    oracle_sub = eval_df[eval_df['Model'] == 'Oracle']
    if not oracle_sub.empty:
        ax.plot(oracle_sub['Offered_Load_pps'], oracle_sub['Throughput_Mbps'],
                'k--', linewidth=3, label='Oracle (Upper Bound)', zorder=10)

    # RL agent curves
    rl_models = eval_df[eval_df['Model'] != 'Oracle']['Model'].unique()
    colors = sns.color_palette("husl", max(len(rl_models), 1)) if _HAS_SEABORN else plt.cm.tab10(np.linspace(0, 1, max(len(rl_models), 1)))
    for i, model_name in enumerate(rl_models):
        sub = eval_df[eval_df['Model'] == model_name]
        mtype = sub['Type'].iloc[0]
        linestyle = '-' if mtype == 'MARL' else '-.'
        marker = 's' if mtype == 'MARL' else 'o'
        ax.plot(sub['Offered_Load_pps'], sub['Throughput_Mbps'],
                linestyle=linestyle, marker=marker, color=colors[i],
                label=f"{mtype}: {model_name}", linewidth=2, markersize=4)

    ax.set_xlabel("Offered Load (pps)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Unified Comparison: SARL vs MARL — Throughput")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "unified_throughput_comparison.png"), dpi=150)
    plt.close()
    log(f"  Saved unified_throughput_comparison.png")

    # --- 6b: Delay comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    if not baseline_df.empty:
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['TDMA_Delay_s'] * 1000,
                'b--', label='TDMA (Baseline)', alpha=0.5, linewidth=1.5)
        ax.plot(baseline_df['Offered_Load_pps'], baseline_df['CSMA_Delay_s'] * 1000,
                'r--', label='CSMA/CA (Baseline)', alpha=0.5, linewidth=1.5)

    # Oracle line (dark black, thick dashed)
    if not oracle_sub.empty:
        ax.plot(oracle_sub['Offered_Load_pps'], oracle_sub['Delay_ms'],
                'k--', linewidth=3, label='Oracle (Upper Bound)', zorder=10)

    for i, model_name in enumerate(rl_models):
        sub = eval_df[eval_df['Model'] == model_name]
        mtype = sub['Type'].iloc[0]
        linestyle = '-' if mtype == 'MARL' else '-.'
        marker = 's' if mtype == 'MARL' else 'o'
        ax.plot(sub['Offered_Load_pps'], sub['Delay_ms'],
                linestyle=linestyle, marker=marker, color=colors[i],
                label=f"{mtype}: {model_name}", linewidth=2, markersize=4)

    ax.set_xlabel("Offered Load (pps)")
    ax.set_ylabel("Delay (ms)")
    ax.set_title("Unified Comparison: SARL vs MARL — Delay")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "unified_delay_comparison.png"), dpi=150)
    plt.close()
    log(f"  Saved unified_delay_comparison.png")

    # --- 6c: MAC Selection per model ---
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = eval_df['Model'].unique()
    tdma_pcts, csma_pcts = [], []
    for m in model_names:
        sub = eval_df[eval_df['Model'] == m]
        tdma_pct = (sub['Dominant_MAC'] == 'TDMA').mean() * 100
        tdma_pcts.append(tdma_pct)
        csma_pcts.append(100 - tdma_pct)

    y_pos = np.arange(len(model_names))
    ax.barh(y_pos, tdma_pcts, color='tab:green', alpha=0.9, label='TDMA')
    ax.barh(y_pos, csma_pcts, left=tdma_pcts, color='tab:purple', alpha=0.75, label='CSMA/CA')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.set_xlabel("Selection share (%)")
    ax.set_title("MAC Selection Preference per Agent")
    ax.set_xlim(0, 100)
    ax.axvline(50, color='gray', linestyle='--', alpha=0.5)

    # Label TDMA% on each bar for quick reading.
    for i, pct in enumerate(tdma_pcts):
        ax.text(min(max(pct + 1.5, 1.5), 97), i, f"TDMA {pct:.1f}%", va='center', fontsize=8)

    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "mac_selection_preference.png"), dpi=150)
    plt.close()
    log(f"  Saved mac_selection_preference.png")

    # --- Save summary JSON ---
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "config": {
            "N": params.N, "PHY_RATE_BPS": params.PHY_RATE_BPS,
            "QMAX": params.QMAX, "RTS_CTS": params.RTS_CTS_ENABLED,
            "ACK": params.ACK_ENABLED, "FADING": getattr(params, 'FADING_MODEL', 'none'),
        },
        "models_evaluated": list(eval_df['Model'].unique()),
        "num_load_points": int(eval_df['Offered_Load_pps'].nunique()),
    }
    with open(os.path.join(out_dir, "csv", "experiment_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    log("  Saved experiment_summary.json")


# =====================================================================
# Main Entry Point
# =====================================================================
def run_unified_experiment(
    dry_run=False,
    force_retrain=False,
    use_checkpoints_only=True,
    phy_rate_mbps=None,
    nodes=None,
    qmax=None,
    sweep_min_pps=None,
    sweep_max_pps=None,
    sweep_steps=None,
    stochastic_eval=False,
):
    # Optional runtime overrides help avoid editing global config for quick what-if runs.
    if phy_rate_mbps is not None:
        params.PHY_RATE_BPS = float(phy_rate_mbps) * 1e6
    if nodes is not None:
        params.N = int(nodes)
    if qmax is not None:
        params.QMAX = int(qmax)
    if sweep_min_pps is not None:
        params.SWEEP_MIN_PPS = int(sweep_min_pps)
    if sweep_max_pps is not None:
        params.SWEEP_MAX_PPS = int(sweep_max_pps)
    if sweep_steps is not None:
        params.SWEEP_STEPS = int(sweep_steps)

    N = params.N
    sim_time = params.SIM_TIME_S
    phy = params.PHY_RATE_BPS
    QMAX = params.QMAX

    out_dir = make_dirs(N, phy, QMAX, params.RTS_CTS_ENABLED, params.ACK_ENABLED)
    log_path = os.path.join(out_dir, "logs", "run.log")

    def log(msg):
        log_step(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # --- wandb experiment tracking ---
    from utils.device_manager import resolve_device, get_gpu_info
    train_device = resolve_device("train")
    gpu_info = get_gpu_info()

    experiment_config = {
        "N": N, "PHY_RATE_BPS": phy, "QMAX": QMAX, "SIM_TIME_S": sim_time,
        "RTS_CTS": params.RTS_CTS_ENABLED, "ACK": params.ACK_ENABLED,
        "FADING": getattr(params, 'FADING_MODEL', 'none'),
        "MOBILITY": getattr(params, 'MOBILITY_MODEL', 'none'),
        "SWEEP_MIN_PPS": params.SWEEP_MIN_PPS,
        "SWEEP_MAX_PPS": params.SWEEP_MAX_PPS,
        "dry_run": dry_run, "force_retrain": force_retrain,
        "train_device": train_device,
        "gpu_available": gpu_info["available"],
        "gpu_name": gpu_info["devices"][0]["name"] if gpu_info["devices"] else "N/A",
    }
    run_name = f"N{N}_PHY{int(phy/1e6)}M_{'dry' if dry_run else 'full'}"
    wandb_init(
        project="fanet-mac-rl", config=experiment_config,
        run_name=run_name, tags=["unified", "dry-run" if dry_run else "full"],
        group="unified_experiment",
    )

    print_banner(
        "UNIFIED EXPERIMENT RUNNER",
        f"SARL + MARL on MARL Environment | {datetime.datetime.now().isoformat()}",
    )

    # --- GPU / Device status banner ---
    if gpu_info["available"] and "cuda" in train_device:
        gpu_name = gpu_info["devices"][0]["name"]
        gpu_vram = gpu_info["devices"][0].get("vram_gb", "?")
        log(f"GPU ACTIVE: {gpu_name} ({gpu_vram} GB VRAM) -> training on {train_device}")
    elif gpu_info["available"]:
        gpu_name = gpu_info["devices"][0]["name"]
        log(f"GPU detected ({gpu_name}) but config forces CPU (FORCE_CPU={params.FORCE_CPU}, TRAIN_ON_GPU={params.TRAIN_ON_GPU})")
    else:
        log("No GPU detected - all training will run on CPU")

    log(f"Config: N={N}, PHY={int(phy/1e6)}Mbps, Q={QMAX}, "
        f"RTS/CTS={'ON' if params.RTS_CTS_ENABLED else 'OFF'}, "
        f"ACK={'ON' if params.ACK_ENABLED else 'OFF'}, "
        f"Fading={getattr(params, 'FADING_MODEL', 'none')}")

    if dry_run:
        sarl_ts = 100
        marl_ep = 2
        sweep_steps = 3
        log("*** DRY RUN MODE — reduced params ***")
    else:
        sarl_ts = RLConfig.TOTAL_TIMESTEPS
        marl_ep = MARLConfig.EPISODES
        sweep_steps = params.SWEEP_STEPS

    cfg = build_mac_config(N=N, sim_time_s=sim_time, seed=params.SEED)
    pps_list = np.linspace(params.SWEEP_MIN_PPS, params.SWEEP_MAX_PPS, sweep_steps).astype(int)

    # Step 1 — Baselines on MARL env (with mobility + fading)
    baseline_df = step1_baseline_marl(pps_list, params.SEED, log)
    baseline_df.to_csv(os.path.join(out_dir, "csv", "baseline_results.csv"), index=False)
    generate_baseline_plots(
        baseline_df,
        out_dir,
        N,
        params.PAYLOAD_BYTES,
        phy,
        QMAX,
        params.RTS_CTS_ENABLED,
        params.ACK_ENABLED,
    )
    log("  Saved legacy baseline plots (throughput/delay/drops/collisions).")

    # Step 2
    cp_dir = step2_train(
        out_dir,
        sarl_ts,
        marl_ep,
        log,
        force_retrain=force_retrain,
        use_checkpoints_only=use_checkpoints_only,
    )

    # Step 3
    eval_df = step3_evaluate(
        pps_list,
        cp_dir,
        out_dir,
        log,
        deterministic_eval=(not stochastic_eval),
    )

    # Legacy aggregate RL plots used by prior experiment workflow
    legacy_all_results = _build_legacy_rl_results(eval_df)
    if legacy_all_results:
        generate_aggregated_rl_plots(legacy_all_results, baseline_df, q_df=None, rl_out_dir=out_dir, log_print=log)
        log("  Saved legacy RL-vs-baseline aggregate plots.")

    # Step 4
    step4_reward_curves(out_dir, log)

    # Step 5
    step5_ablations(eval_df, pps_list, cp_dir, out_dir, log)

    # Step 6
    step6_summary(eval_df, baseline_df, out_dir, log)

    # Upload key artifacts to wandb
    csv_dir_path = os.path.join(out_dir, "csv")
    for csv_file in ["baseline_results.csv", "unified_eval_sweep.csv", "experiment_summary.json"]:
        wandb_artifact(os.path.join(csv_dir_path, csv_file), artifact_type="result")
    img_dir_path = os.path.join(out_dir, "images")
    for img_file in ["unified_throughput_comparison.png", "unified_delay_comparison.png", "mac_selection_preference.png"]:
        img_path = os.path.join(img_dir_path, img_file)
        wandb_image(f"plots/{img_file.replace('.png', '')}", img_path)

    wandb_finish()

    print_banner("UNIFIED EXPERIMENT COMPLETED SUCCESSFULLY", f"Results: {out_dir}")
    log(f"Results: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified SARL+MARL Experiment Runner")
    parser.add_argument('--dry-run', action='store_true', help="Quick test with minimal params")
    parser.add_argument('--force-retrain', action='store_true', help="Ignore existing checkpoints and retrain models")
    parser.add_argument('--skip-training', action='store_true', help="Skip training and purely evaluate using existing checkpoints")
    parser.add_argument('--phy-rate-mbps', type=float, default=None, help="Override PHY rate in Mbps (e.g., 9)")
    parser.add_argument('--nodes', type=int, default=None, help="Override number of nodes")
    parser.add_argument('--qmax', type=int, default=None, help="Override queue size")
    parser.add_argument('--sweep-min-pps', type=int, default=None, help="Override sweep minimum offered load")
    parser.add_argument('--sweep-max-pps', type=int, default=None, help="Override sweep maximum offered load")
    parser.add_argument('--sweep-steps', type=int, default=None, help="Override number of sweep points")
    parser.add_argument('--stochastic-eval', action='store_true', help="Use stochastic action selection during evaluation (diagnostic)")
    args = parser.parse_args()
    run_unified_experiment(
        dry_run=args.dry_run,
        force_retrain=args.force_retrain,
        use_checkpoints_only=args.skip_training,
        phy_rate_mbps=args.phy_rate_mbps,
        nodes=args.nodes,
        qmax=args.qmax,
        sweep_min_pps=args.sweep_min_pps,
        sweep_max_pps=args.sweep_max_pps,
        sweep_steps=args.sweep_steps,
        stochastic_eval=args.stochastic_eval,
    )
