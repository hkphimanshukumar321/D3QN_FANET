"""
Bottleneck Profiler — Measure where time goes in one MARL training trial.

Run on the server:
    python experiments/profile_bottleneck.py --algo iql --episodes 50
    python experiments/profile_bottleneck.py --algo magat_d3qn --episodes 50

Outputs a breakdown of:
  - env.step() time (CPU: mobility + fading + MAC sim)
  - model.select_actions() time (GPU or CPU: neural net forward)
  - model training time (GPU or CPU: backward + optimizer)
  - everything else
"""

import os, sys, time, argparse
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from configs.marl_config import MARLConfig
from configs.cluster_config import ClusterConfig as CC
from configs import config as params
from envs.marl_mac_env import MARLMacEnv
from utils.device_manager import resolve_device, get_gpu_info, get_cpu_info


def profile_algo(algo: str, episodes: int):
    import torch

    device_str = resolve_device("train")
    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()

    print("=" * 60)
    print(f"  Bottleneck Profiler: {algo.upper()}")
    print(f"  Episodes: {episodes}")
    print(f"  Device: {device_str}")
    print(f"  CPU: {cpu_info['logical_cores']} logical cores, {cpu_info['ram_gb']:.1f} GB RAM")
    if gpu_info["available"]:
        for d in gpu_info["devices"]:
            print(f"  GPU: {d['name']} ({d.get('vram_gb', '?')} GB)")
    print("=" * 60)

    # --- Build agent ---
    n_agents = CC.C_MAX
    obs_dim = MARLConfig.OBS_DIM
    num_actions = MARLConfig.NUM_ACTIONS
    common = dict(
        n_agents=n_agents, obs_dim=obs_dim, num_actions=num_actions,
        hidden_dim=MARLConfig.HIDDEN_DIM, lr=MARLConfig.LR,
        gamma=MARLConfig.GAMMA, eps_start=MARLConfig.EPSILON_START,
        eps_end=MARLConfig.EPSILON_END, eps_decay=MARLConfig.EPSILON_DECAY,
        target_update=MARLConfig.TARGET_UPDATE_FREQ,
        replay_size=MARLConfig.REPLAY_SIZE, batch_size=MARLConfig.BATCH_SIZE,
        device=device_str,
    )

    if algo == "magat_d3qn":
        from algorithms.rl.gnn_marl import MAGAT_D3QN_QNetwork
        from collections import deque
        model = MAGAT_D3QN_QNetwork(
            node_in_dim=obs_dim, num_actions=num_actions,
            hidden_dim=MARLConfig.HIDDEN_DIM,
            heads=MARLConfig.GNN_HEADS,
            use_gru=MARLConfig.MAGAT_USE_GRU,
        ).to(device_str)
        target_model = MAGAT_D3QN_QNetwork(
            node_in_dim=obs_dim, num_actions=num_actions,
            hidden_dim=MARLConfig.HIDDEN_DIM,
            heads=MARLConfig.GNN_HEADS,
            use_gru=MARLConfig.MAGAT_USE_GRU,
        ).to(device_str)
        target_model.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=MARLConfig.LR)
        memory = deque(maxlen=MARLConfig.REPLAY_SIZE)
        is_gnn = True
    elif algo == "iql":
        from algorithms.rl.marl_baselines import IQLAgent
        agent = IQLAgent(**common)
        is_gnn = False
    elif algo == "vdn":
        from algorithms.rl.marl_baselines import VDNAgent
        agent = VDNAgent(**common)
        is_gnn = False
    elif algo == "qmix":
        from algorithms.rl.marl_baselines import QMIXAgent
        agent = QMIXAgent(**common, state_dim=obs_dim * n_agents)
        is_gnn = False
    elif algo == "mappo":
        from algorithms.rl.mappo_agent import MAPPOAgent
        agent = MAPPOAgent(
            n_agents=n_agents, obs_dim=obs_dim, num_actions=num_actions,
            hidden_dim=MARLConfig.HIDDEN_DIM, lr=MARLConfig.LR,
            gamma=MARLConfig.GAMMA, device=device_str,
        )
        is_gnn = False
    else:
        raise ValueError(f"Unknown algo: {algo}")

    # --- GPU VRAM baseline ---
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        vram_baseline_mb = torch.cuda.memory_allocated() / (1024**2)
        vram_samples = []
    else:
        vram_baseline_mb = 0.0
        vram_samples = []

    # --- Profile ---
    env = MARLMacEnv(seed=42)
    t_env_reset = 0.0
    t_env_step = 0.0
    t_action_select = 0.0
    t_train = 0.0
    t_other = 0.0
    total_steps = 0

    for ep in range(episodes):
        t0 = time.perf_counter()
        obs_dict, _ = env.reset()
        t_env_reset += time.perf_counter() - t0

        if is_gnn:
            x, edge_index, alive_mask = env.get_global_graph_state()
        else:
            obs_all = np.stack([obs_dict[a] for a in env.possible_agents])
            alive_mask = np.ones(n_agents, dtype=bool)

        while env.agents:
            # --- Action selection (forward pass) ---
            t0 = time.perf_counter()
            if is_gnn:
                x_t = torch.tensor(x, dtype=torch.float32).to(device_str)
                e_t = torch.tensor(edge_index, dtype=torch.long).to(device_str)
                a_t = torch.tensor(alive_mask, dtype=torch.bool).to(device_str)
                with torch.no_grad():
                    q_vals = model(x_t, e_t, a_t)
                actions_dict = {
                    a: q_vals[i].argmax().item()
                    for i, a in enumerate(env.possible_agents)
                }
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            else:
                actions_list = agent.select_actions(obs_all, alive_mask=alive_mask)
                actions_dict = {a: actions_list[i] for i, a in enumerate(env.possible_agents)}
            t_action_select += time.perf_counter() - t0

            # --- Env step (CPU: mobility + fading + MAC sim) ---
            t0 = time.perf_counter()
            next_obs, rewards, terms, truncs, infos = env.step(actions_dict)
            t_env_step += time.perf_counter() - t0

            # --- Store transitions + Training step ---
            t0 = time.perf_counter()
            if is_gnn:
                from torch_geometric.data import Data, Batch as PyGBatch
                import random
                next_x, next_ei, next_alive = env.get_global_graph_state()
                act_list = np.array([actions_dict[a] for a in env.possible_agents], dtype=np.int64)
                reward_arr = np.array([rewards.get(a, 0.0) for a in env.possible_agents], dtype=np.float32)
                done = not env.agents
                memory.append((x, edge_index, alive_mask, act_list, reward_arr, next_x, next_ei, next_alive, done))

                if len(memory) > MARLConfig.BATCH_SIZE and total_steps % 4 == 0:
                    batch = random.sample(memory, MARLConfig.BATCH_SIZE)
                    data_c, data_n, act_b, rew_b, m_b, mn_b, d_b = [], [], [], [], [], [], []
                    for (bx, bei, bmask, ba, br, bnx, bnei, bnmask, bd) in batch:
                        n_nodes = bx.shape[0]
                        data_c.append(Data(x=torch.tensor(bx, dtype=torch.float32), edge_index=torch.tensor(bei, dtype=torch.long)))
                        data_n.append(Data(x=torch.tensor(bnx, dtype=torch.float32), edge_index=torch.tensor(bnei, dtype=torch.long)))
                        act_b.append(torch.tensor(ba, dtype=torch.long))
                        rew_b.append(torch.tensor(br, dtype=torch.float32))
                        m_b.append(torch.tensor(bmask, dtype=torch.bool))
                        mn_b.append(torch.tensor(bnmask, dtype=torch.bool))
                        d_b.append(torch.full((n_nodes,), float(bd), dtype=torch.float32))

                    batch_c = PyGBatch.from_data_list(data_c).to(device_str)
                    batch_n = PyGBatch.from_data_list(data_n).to(device_str)
                    act_cat = torch.cat(act_b).to(device_str)
                    rew_cat = torch.cat(rew_b).to(device_str)
                    mask_cat = torch.cat(m_b).to(device_str)
                    mask_n_cat = torch.cat(mn_b).to(device_str)
                    done_cat = torch.cat(d_b).to(device_str)

                    model.reset_memory()
                    q_all = model.forward_batched(batch_c.x, batch_c.edge_index, batch_c.batch, mask_cat)
                    q_a = q_all[torch.arange(q_all.size(0), device=device_str), act_cat]

                    with torch.no_grad():
                        target_model.reset_memory()
                        q_next = target_model.forward_batched(batch_n.x, batch_n.edge_index, batch_n.batch, mask_n_cat).max(1)[0]
                        target = rew_cat + MARLConfig.GAMMA * q_next * (1.0 - done_cat)

                    active = mask_cat
                    if torch.any(active):
                        loss = torch.nn.functional.mse_loss(q_a[active], target[active])
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                if total_steps % MARLConfig.TARGET_UPDATE_FREQ == 0:
                    target_model.load_state_dict(model.state_dict())
            else:
                next_obs_all = np.stack([next_obs[a] for a in env.possible_agents])
                reward_arr = np.array([rewards.get(a, 0.0) for a in env.possible_agents])
                done = not env.agents
                if hasattr(agent, 'store'):
                    agent.store(obs_all, actions_list, reward_arr, next_obs_all, done, alive_mask)
                if hasattr(agent, 'update') and hasattr(agent, 'replay') and len(agent.replay) > MARLConfig.BATCH_SIZE:
                    agent.update()
            t_train += time.perf_counter() - t0

            # Bookkeeping
            t0 = time.perf_counter()
            if is_gnn:
                x, edge_index, alive_mask = next_x, next_ei, next_alive
            else:
                obs_all = next_obs_all
            t_other += time.perf_counter() - t0

            # Sample GPU VRAM
            if gpu_available:
                vram_samples.append(torch.cuda.memory_allocated() / (1024**2))

            total_steps += 1

        if (ep + 1) % 10 == 0:
            vram_now = f", VRAM={torch.cuda.memory_allocated()/(1024**2):.1f}MB" if gpu_available else ""
            print(f"  Episode {ep+1}/{episodes} done ({total_steps} steps{vram_now})")

    # --- Report ---
    total = t_env_reset + t_env_step + t_action_select + t_train + t_other
    print("\n" + "=" * 60)
    print(f"  RESULTS: {algo.upper()} — {episodes} episodes, {total_steps} steps")
    print("=" * 60)
    print(f"  {'Component':<30} {'Time (s)':>10} {'%':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*8}")
    for label, t in [
        ("env.reset() [CPU]", t_env_reset),
        ("env.step()  [CPU: sim/fading]", t_env_step),
        ("select_actions [NN forward]", t_action_select),
        ("train_step  [NN backward]", t_train),
        ("other (obs stacking etc)", t_other),
    ]:
        pct = (t / total * 100) if total > 0 else 0
        print(f"  {label:<30} {t:>10.2f} {pct:>7.1f}%")
    print(f"  {'TOTAL':<30} {total:>10.2f} {'100.0':>7}%")
    print()
    print(f"  Avg per step: {total/max(total_steps,1)*1000:.2f} ms")
    print(f"  Avg env.step(): {t_env_step/max(total_steps,1)*1000:.2f} ms")
    print(f"  Avg select_actions(): {t_action_select/max(total_steps,1)*1000:.2f} ms")
    print()

    cpu_pct = (t_env_reset + t_env_step + t_other) / total * 100
    nn_pct = (t_action_select + t_train) / total * 100
    print(f"  >>> CPU-bound (env sim):  {cpu_pct:.1f}%")
    print(f"  >>> NN-bound (GPU/CPU):   {nn_pct:.1f}%")
    print()

    # --- GPU VRAM Report ---
    if gpu_available:
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        avg_vram_mb = np.mean(vram_samples) if vram_samples else 0.0
        max_vram_mb = max(vram_samples) if vram_samples else 0.0
        total_gpu_mb = gpu_info["devices"][0].get("vram_gb", 32) * 1024
        # Reserve 2GB for system/driver overhead
        usable_mb = total_gpu_mb - 2048
        max_concurrent = int(usable_mb / max(peak_vram_mb, 1))

        print(f"  === GPU VRAM PER TRIAL ===")
        print(f"  {'Baseline (model load)':<30} {vram_baseline_mb:>10.1f} MB")
        print(f"  {'Peak (torch.cuda.max_memory)':<30} {peak_vram_mb:>10.1f} MB")
        print(f"  {'Avg during training':<30} {avg_vram_mb:>10.1f} MB")
        print(f"  {'Max sampled':<30} {max_vram_mb:>10.1f} MB")
        print()
        print(f"  GPU total: {total_gpu_mb:.0f} MB | Usable (minus 2GB reserve): {usable_mb:.0f} MB")
        print(f"  >>> Max concurrent trials on this GPU: {max_concurrent}")
        print(f"      ({usable_mb:.0f} MB usable / {peak_vram_mb:.1f} MB per trial)")
    else:
        print(f"  No GPU — all trials run on CPU. Concurrency limited by CPU cores only.")
        peak_vram_mb = 0
        max_concurrent = 96  # CPU-only

    print()
    if cpu_pct > 60:
        print(f"  VERDICT: CPU-bound — parallelizing across CPU cores will help most.")
        print(f"           Non-GNN algos can run with CUDA_VISIBLE_DEVICES=\"\" safely.")
    elif nn_pct > 60:
        print(f"  VERDICT: GPU-bound — increase --max-concurrent-gpu-trials if VRAM allows.")
    else:
        print(f"  VERDICT: Mixed — both CPU and GPU parallelism will help.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=["iql", "vdn", "qmix", "mappo", "magat_d3qn"])
    parser.add_argument("--episodes", type=int, default=50, help="Episodes to profile (50 is enough)")
    args = parser.parse_args()
    profile_algo(args.algo, args.episodes)
