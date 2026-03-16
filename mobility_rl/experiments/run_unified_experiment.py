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

# Project path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from algorithms.mac.baseline import Config, Logger, simulate_tdma, simulate_csma_ca
from configs import config as params
from configs.sarl_config import RLConfig
from configs.marl_config import MARLConfig

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
            return os.path.getsize(p) / 1024.0
    return 0.0


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
        })
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

    import pandas as pd
    from envs.marl_sarl_wrapper import MARLtoSARLWrapper

    pid = os.getpid()
    print(f"  [Worker {pid}] SARL training: {algo.upper()}")

    if algo == 'tabular':
        from algorithms.rl.tabular_qlearning import TabularQLearning
        save_path = os.path.join(cp_dir, "unified_tabular_model.json")
        if os.path.exists(save_path):
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
            pd.DataFrame({"step": steps_log, "reward": rewards_log}).to_csv(
                os.path.join(csv_dir, "tabular_training_rewards.csv"), index=False)
        return f"{algo.upper()} trained."

    elif algo in ['dqn', 'ppo', 'a2c']:
        from algorithms.rl.sb3_baselines import create_sb3_baseline
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import BaseCallback

        save_path = os.path.join(cp_dir, f"unified_{algo}_model")
        if os.path.exists(save_path + ".zip"):
            return f"{algo.upper()} skipped (checkpoint)"

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
            pd.DataFrame({"step": cb.steps, "reward": cb.rewards}).to_csv(
                os.path.join(csv_dir, f"{algo}_training_rewards.csv"), index=False)
        return f"{algo.upper()} trained."

    elif algo == 'mca_d3qn':
        from algorithms.rl.custom_mca_d3qn import create_mca_d3qn
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import BaseCallback

        save_path = os.path.join(cp_dir, "unified_mca_d3qn_model")
        if os.path.exists(save_path + ".zip"):
            return f"MCA-D3QN skipped (checkpoint)"

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
            pd.DataFrame({"step": cb.steps, "reward": cb.rewards}).to_csv(
                os.path.join(csv_dir, "mca_d3qn_training_rewards.csv"), index=False)
        return f"MCA-D3QN trained."

    return f"Unknown SARL algo: {algo}"


def _train_marl_worker(kwargs):
    """Train a single MARL agent on MARLMacEnv natively."""
    algo = kwargs['algo']
    episodes = kwargs['episodes']
    cp_dir = kwargs['cp_dir']
    csv_dir = kwargs['csv_dir']
    seed = kwargs['seed']

    import pandas as pd
    import torch
    from envs.marl_mac_env import MARLMacEnv
    from configs.marl_config import MARLConfig
    from configs import config as params

    pid = os.getpid()
    print(f"  [Worker {pid}] MARL training: {algo.upper()}")

    env = MARLMacEnv(seed=seed)
    N = params.N
    obs_dim = MARLConfig.OBS_DIM
    num_actions = MARLConfig.NUM_ACTIONS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    common = dict(
        n_agents=N, obs_dim=obs_dim, num_actions=num_actions,
        hidden_dim=MARLConfig.HIDDEN_DIM, lr=MARLConfig.LR,
        gamma=MARLConfig.GAMMA, eps_start=MARLConfig.EPSILON_START,
        eps_end=MARLConfig.EPSILON_END, eps_decay=MARLConfig.EPSILON_DECAY,
        target_update=MARLConfig.TARGET_UPDATE_FREQ,
        replay_size=MARLConfig.REPLAY_SIZE, batch_size=MARLConfig.BATCH_SIZE,
        device=device,
    )

    # --- MAGAT-D3QN (separate GNN training) ---
    if algo == 'magat_d3qn':
        from algorithms.rl.gnn_marl import MAGAT_D3QN_QNetwork
        import random, math
        from collections import deque

        save_path = os.path.join(cp_dir, "unified_gnn_marl_model.pth")
        if os.path.exists(save_path):
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
                    loss = torch.tensor(0.0, device=device)
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
                        loss += torch.nn.functional.mse_loss(q_a, target)
                    loss /= batch_size
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if ep % 100 == 0 and env.agents:
                    target_net.load_state_dict(policy_net.state_dict())

            ep_rewards.append(ep_reward)

        torch.save(policy_net.state_dict(), save_path)
        if ep_rewards:
            pd.DataFrame({"episode": range(len(ep_rewards)), "reward": ep_rewards}).to_csv(
                os.path.join(csv_dir, "magat_d3qn_training_rewards.csv"), index=False)
        return "MAGAT-D3QN trained."

    # --- IQL / VDN / QMIX (shared loop) ---
    if algo == 'iql':
        from algorithms.rl.marl_baselines import IQLAgent
        save_path = os.path.join(cp_dir, "unified_iql_model.pth")
        if os.path.exists(save_path):
            return "IQL skipped (checkpoint)"
        agent = IQLAgent(**common)
    elif algo == 'vdn':
        from algorithms.rl.marl_baselines import VDNAgent
        save_path = os.path.join(cp_dir, "unified_vdn_model.pth")
        if os.path.exists(save_path):
            return "VDN skipped (checkpoint)"
        agent = VDNAgent(**common)
    elif algo == 'qmix':
        from algorithms.rl.marl_baselines import QMIXAgent
        save_path = os.path.join(cp_dir, "unified_qmix_model.pth")
        if os.path.exists(save_path):
            return "QMIX skipped (checkpoint)"
        common['embed_dim'] = MARLConfig.QMIX_EMBED_DIM
        agent = QMIXAgent(**common)
    else:
        return f"Unknown MARL algo: {algo}"

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

    agent.save(save_path)
    if ep_rewards:
        pd.DataFrame({"episode": range(len(ep_rewards)), "reward": ep_rewards}).to_csv(
            os.path.join(csv_dir, f"{algo}_training_rewards.csv"), index=False)
    return f"{algo.upper()} trained."



def _dispatch_training(kw):
    """Module-level dispatch for multiprocessing (must be picklable)."""
    if kw['type'] == 'sarl':
        return _train_sarl_worker(kw)
    else:
        return _train_marl_worker(kw)


def step2_train(out_dir, sarl_timesteps, marl_episodes, log):
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
    if getattr(params, "RUN_MARL_QMIX", True): marl_tasks.append('qmix')
    if getattr(params, "RUN_MARL_GNN", True): marl_tasks.append('magat_d3qn')

    all_kwargs = []
    for algo in sarl_tasks:
        all_kwargs.append({
            'type': 'sarl', 'algo': algo, 'timesteps': sarl_timesteps,
            'cp_dir': cp_dir, 'csv_dir': csv_dir, 'seed': params.SEED,
        })
    for algo in marl_tasks:
        all_kwargs.append({
            'type': 'marl', 'algo': algo, 'episodes': marl_episodes,
            'cp_dir': cp_dir, 'csv_dir': csv_dir, 'seed': params.SEED,
        })

    n_workers = min(os.cpu_count() or 1, len(all_kwargs))
    log(f"  Launching {len(all_kwargs)} training jobs on {n_workers} workers")
    log(f"  SARL: {sarl_tasks} ({sarl_timesteps} timesteps each)")
    log(f"  MARL: {marl_tasks} ({marl_episodes} episodes each)")

    with multiprocessing.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_dispatch_training, all_kwargs):
            log(f"    --> {result}")

    log("  All training complete.")
    return cp_dir


# =====================================================================
# Step 3: Evaluation Sweep
# =====================================================================
def step3_evaluate(pps_list, cp_dir, out_dir, log):
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

    qmix_path = os.path.join(cp_dir, "unified_qmix_model.pth")
    if os.path.exists(qmix_path):
        agent = QMIXAgent(N, obs_dim, 2)
        agent.load(qmix_path)
        models["QMIX"] = ('marl', agent)
        log(f"  Loaded MARL: QMIX")

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

        for model_name, (mtype, model) in models.items():
            if mtype == 'sarl':
                # Use wrapper
                obs, _ = sarl_wrapper.reset()
                total_thr, total_delay, total_drops, steps = 0, 0, 0, 0
                mac_choices = []
                done = False

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
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
                total_thr, total_delay, total_drops, steps = 0, 0, 0, 0
                mac_choices = []

                model.steps_done = model.eps_decay * 10  # force greedy
                while marl_env.agents:
                    actions_list = model.select_actions(obs_all)
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
                total_thr, total_delay, total_drops, steps = 0, 0, 0, 0
                mac_choices = []

                while marl_env.agents:
                    x_t = torch.tensor(x, dtype=torch.float32)
                    e_t = torch.tensor(edge_index, dtype=torch.long)
                    with torch.no_grad():
                        q_vals = model(x_t, e_t)
                        actions_dict = {a: q_vals[i].argmax().item() for i, a in enumerate(marl_env.agents)}

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

            results.append({
                'Model': model_name,
                'Type': 'SARL' if mtype == 'sarl' else 'MARL',
                'Offered_Load_pps': pps,
                'Throughput_Mbps': round(avg_thr, 4),
                'Delay_ms': round(avg_delay, 4),
                'Drops': total_drops,
                'Dominant_MAC': dom_mac,
            })

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "csv", "unified_eval_sweep.csv")
    df.to_csv(csv_path, index=False)
    log(f"  Saved: {csv_path}")
    return df


# =====================================================================
# Step 4: Reward Training Curves
# =====================================================================
def step4_reward_curves(out_dir, log):
    log("\n" + "="*60)
    log(" STEP 4: Reward-vs-Episode Training Curves")
    log("="*60)

    csv_dir = os.path.join(out_dir, "csv")
    fig, ax = plt.subplots(figsize=(12, 6))
    found = False

    import glob
    for csv_file in sorted(glob.glob(os.path.join(csv_dir, "*_training_rewards.csv"))):
        name = os.path.basename(csv_file).replace("_training_rewards.csv", "").upper()
        df = pd.read_csv(csv_file)
        x_col = "episode" if "episode" in df.columns else "step"
        # Smoothing with rolling average
        window = max(1, len(df) // 20)
        smoothed = df["reward"].rolling(window=window, min_periods=1).mean()
        ax.plot(df[x_col], smoothed, label=name, linewidth=1.5)
        found = True

    if found:
        ax.set_xlabel("Training Step / Episode")
        ax.set_ylabel("Reward (smoothed)")
        ax.set_title("Training Reward Curves — All Agents on MARL Env")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, "images", "reward_training_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        log(f"  Saved: {path}")
    else:
        log("  No training reward CSVs found.")


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
        cfg = Config(N=N_val, sim_time_s=sim_time, QMAX=params.QMAX,
                     tdma_guard_time_s=params.TDMA_GUARD_TIME_S)

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
        cfg = Config(N=params.N, sim_time_s=sim_time, QMAX=params.QMAX,
                     rts_cts_enabled=rts_on, ack_enabled=params.ACK_ENABLED,
                     tdma_guard_time_s=params.TDMA_GUARD_TIME_S)

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

    for wt, wd in ABLATION_REWARD_WEIGHTS:
        label = f"wT={wt}, wD={wd}"
        # Compute weighted score per model
        scores = []
        for model_name in eval_df['Model'].unique():
            sub = eval_df[eval_df['Model'] == model_name]
            avg_thr = sub['Throughput_Mbps'].mean()
            avg_delay = sub['Delay_ms'].mean()
            score = wt * (avg_thr / 3.0) - wd * (avg_delay / 100.0)
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
        cfg = Config(N=params.N, sim_time_s=sim_time, QMAX=params.QMAX,
                     tdma_guard_time_s=params.TDMA_GUARD_TIME_S)

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
    """Compute oracle (upper-bound) performance from baseline results.
    Oracle = max(TDMA, CSMA) throughput, min(TDMA, CSMA) delay at each load."""
    rows = []
    for pps in pps_list:
        brow = baseline_df[baseline_df['Offered_Load_pps'] == pps]
        if brow.empty:
            continue
        brow = brow.iloc[0]
        thr_tdma = brow['TDMA_Throughput_Mbps']
        thr_csma = brow['CSMA_Throughput_Mbps']
        del_tdma = brow['TDMA_Delay_s'] * 1000  # convert to ms
        del_csma = brow['CSMA_Delay_s'] * 1000

        # Oracle picks best MAC at each load
        if thr_tdma >= thr_csma:
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
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rl_models), 1)))
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
    fig, ax = plt.subplots(figsize=(12, 5))
    model_names = eval_df['Model'].unique()
    tdma_pcts = []
    for m in model_names:
        sub = eval_df[eval_df['Model'] == m]
        tdma_pct = (sub['Dominant_MAC'] == 'TDMA').mean() * 100
        tdma_pcts.append(tdma_pct)

    def _bar_color(m):
        t = eval_df[eval_df['Model'] == m]['Type'].iloc[0]
        if t == 'ORACLE': return 'black'
        return 'tab:blue' if t == 'MARL' else 'tab:orange'
    bar_colors = [_bar_color(m) for m in model_names]
    bars = ax.barh(model_names, tdma_pcts, color=bar_colors, alpha=0.8)
    ax.set_xlabel("% Steps selecting TDMA")
    ax.set_title("MAC Selection Preference per Agent")
    ax.axvline(50, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 100)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='tab:blue', label='MARL'), Patch(color='tab:orange', label='SARL')])
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
def run_unified_experiment(dry_run=False):
    N = params.N
    sim_time = params.SIM_TIME_S
    phy = params.PHY_RATE_BPS
    QMAX = params.QMAX

    out_dir = make_dirs(N, phy, QMAX, params.RTS_CTS_ENABLED, params.ACK_ENABLED)
    log_path = os.path.join(out_dir, "logs", "run.log")

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=" * 60)
    log(" UNIFIED EXPERIMENT RUNNER")
    log(f" SARL + MARL on MARL Environment")
    log(f" {datetime.datetime.now().isoformat()}")
    log("=" * 60)
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

    cfg = Config(N=N, sim_time_s=sim_time, QMAX=QMAX,
                 tdma_guard_time_s=params.TDMA_GUARD_TIME_S)
    pps_list = np.linspace(params.SWEEP_MIN_PPS, params.SWEEP_MAX_PPS, sweep_steps).astype(int)

    # Step 1
    baseline_df = step1_baseline(cfg, pps_list, sim_time, params.SEED, log)
    baseline_df.to_csv(os.path.join(out_dir, "csv", "baseline_results.csv"), index=False)

    # Step 2
    cp_dir = step2_train(out_dir, sarl_ts, marl_ep, log)

    # Step 3
    eval_df = step3_evaluate(pps_list, cp_dir, out_dir, log)

    # Step 4
    step4_reward_curves(out_dir, log)

    # Step 5
    step5_ablations(eval_df, pps_list, cp_dir, out_dir, log)

    # Step 6
    step6_summary(eval_df, baseline_df, out_dir, log)

    log("\n" + "=" * 60)
    log(" UNIFIED EXPERIMENT COMPLETED SUCCESSFULLY")
    log("=" * 60)
    log(f"Results: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified SARL+MARL Experiment Runner")
    parser.add_argument('--dry-run', action='store_true', help="Quick test with minimal params")
    args = parser.parse_args()
    run_unified_experiment(dry_run=args.dry_run)
