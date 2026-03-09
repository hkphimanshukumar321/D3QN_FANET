import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO, A2C

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.adaptive_mac_env import AdaptiveMacEnv
from algorithms.rl.tabular_qlearning import TabularQLearning
from configs.rl_config import RLConfig

def load_models():
    models = {}
    cp_dir = os.path.join(RLConfig.get_results_dir(), "checkpoints")
    
    # MCA-D3QN
    path = os.path.join(cp_dir, "mca_d3qn_model.zip")
    if os.path.exists(path):
        models["MCA_D3QN"] = DQN.load(path)
        
    # Baselines
    for algo, cls in [("dqn", DQN), ("ppo", PPO), ("a2c", A2C)]:
        path = os.path.join(cp_dir, f"{algo}_baseline_model.zip")
        if os.path.exists(path):
            models[algo.upper()] = cls.load(path)
            
    # Tabular
    path = os.path.join(cp_dir, "tabular_q_model.json")
    if os.path.exists(path):
        tab = TabularQLearning()
        tab.load(path)
        # Force greedy evaluation
        tab.epsilon = 0.0
        models["TABULAR"] = tab
        
    return models

def evaluate_step_oracle(env):
    """
    To find the Oracle action, we must simulate BOTH actions from the EXACT same state.
    Since stepping advances the environment internally, we will instead leverage the _step_mobility
    and then simulate both MACs manually using the traces, just as it's done inside step().
    """
    import copy
    from algorithms.mac.baseline import Config, Logger
    from algorithms.mac.channel_aware_mac import simulate_tdma_aware, simulate_csma_aware
    from configs import config as global_cfg
    
    # 1. Advance mobility once for this decision interval
    lu_sched, sp_sched, mob_stats = env._step_mobility()
    offered_pps = getattr(global_cfg, 'OFFERED_PPS', 400)
    
    # Base configuration
    cfg = Config(
        N=global_cfg.N, sim_time_s=env.decision_interval, slot_time_s=global_cfg.SLOT_TIME_S,
        phy_rate_bps=global_cfg.PHY_RATE_BPS, payload_bytes=global_cfg.PAYLOAD_BYTES,
        QMAX=global_cfg.QMAX, seed=int(env.rng.integers(0, 10000)),
        cw_min=global_cfg.CW_MIN, cw_max=global_cfg.CW_MAX,
        difs_slots=global_cfg.DIFS_SLOTS, sifs_slots=global_cfg.SIFS_SLOTS,
        ack_slots=global_cfg.ACK_SLOTS, ack_timeout_slots=global_cfg.ACK_TIMEOUT_SLOTS,
        max_retry=global_cfg.MAX_RETRY, log_interval_slots=global_cfg.LOG_INTERVAL_SLOTS,
        rts_cts_enabled=global_cfg.RTS_CTS_ENABLED, ack_enabled=global_cfg.ACK_ENABLED,
        rts_slots=global_cfg.RTS_SLOTS, cts_slots=global_cfg.CTS_SLOTS,
        tdma_guard_time_s=global_cfg.TDMA_GUARD_TIME_S,
    )
    
    sp_arr = sp_sched if global_cfg.ENABLE_PATHLOSS else None
    
    # 2. Simulate TDMA
    log_tdma = Logger(load_pps=offered_pps, protocol_name="TDMA")
    simulate_tdma_aware(cfg, offered_pps, log_tdma, lu_sched, sp_arr, env.dt)
    m_tdma = {
        "throughput_mbps": log_tdma.get_throughput_bps(env.decision_interval) / 1e6,
        "delay_ms": log_tdma.get_avg_end_to_end_delay_s() * 1000,
        "drops": log_tdma.pkts_dropped_qfull + log_tdma.pkts_dropped_mac,
        "pkts_generated": log_tdma.pkts_generated,
        "collisions": log_tdma.collision_events
    }
    r_tdma = env._compute_reward(m_tdma)
    
    # 3. Simulate CSMA/CA
    log_csma = Logger(load_pps=offered_pps, protocol_name="CSMA_CA")
    simulate_csma_aware(cfg, offered_pps, log_csma, lu_sched, sp_arr, env.dt)
    m_csma = {
        "throughput_mbps": log_csma.get_throughput_bps(env.decision_interval) / 1e6,
        "delay_ms": log_csma.get_avg_end_to_end_delay_s() * 1000,
        "drops": log_csma.pkts_dropped_qfull + log_csma.pkts_dropped_mac,
        "pkts_generated": log_csma.pkts_generated,
        "collisions": log_csma.collision_events
    }
    r_csma = env._compute_reward(m_csma)
    
    # 4. Oracle determination
    oracle_action = 0 if r_tdma >= r_csma else 1
    
    # Return everything so the env can just adopt the metrics of the chosen action
    # without running it a third time.
    return {
        "oracle_action": oracle_action,
        "r_tdma": r_tdma, "m_tdma": m_tdma,
        "r_csma": r_csma, "m_csma": m_csma,
        "mob_stats": mob_stats
    }

def main():
    print("Initiating Evaluation Pipeline...")
    models = load_models()
    if not models:
        print("No trained models found! Please run train scripts first.")
        return
        
    print(f"Loaded models: {list(models.keys())}")
    np.random.seed(42)
    
    results = []
    
    # Evaluation settings
    eval_episodes = 5
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        env = AdaptiveMacEnv()
        
        for ep in range(eval_episodes):
            obs, _ = env.reset(seed=100 + ep)
            done = False
            step = 0
            
            while not done:
                # Get RL action
                if model_name == "TABULAR":
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                
                # Get generalized Oracle action using custom step logic
                # We need to compute oracle first, then let the env apply the selected action stats
                oracle_data = evaluate_step_oracle(env)
                
                # We bypass env.step() directly to avoid double-stepping mobility.
                # Instead, we inject the selected action metrics as the env.step output
                sel_m = oracle_data["m_tdma"] if action == 0 else oracle_data["m_csma"]
                sel_r = oracle_data["r_tdma"] if action == 0 else oracle_data["r_csma"]
                sel_m["prev_action"] = action
                sel_m["avg_queue"] = 0.0 # Fast approximation 
                sel_m["backlog"] = 0.0
                
                obs = env._get_obs(sel_m, oracle_data["mob_stats"])
                env.current_step += 1
                env.global_sim_time += env.decision_interval
                
                done = bool(env.current_step >= env.max_steps)
                
                accuracy = 1 if int(action) == int(oracle_data["oracle_action"]) else 0
                
                results.append({
                    "model": model_name,
                    "episode": ep,
                    "step": step,
                    "sim_time": env.global_sim_time,
                    "rl_action": int(action),
                    "oracle_action": int(oracle_data["oracle_action"]),
                    "accuracy": accuracy,
                    "throughput_mbps": sel_m["throughput_mbps"],
                    "delay_ms": sel_m["delay_ms"],
                    "drops": sel_m["drops"],
                    "reward": sel_r
                })
                
                step += 1
                
    # Save raw logs
    df = pd.DataFrame(results)
    out_dir = os.path.join(RLConfig.get_results_dir(), "evaluations")
    os.makedirs(out_dir, exist_ok=True)
    
    df.to_csv(os.path.join(out_dir, "rl_eval_mac_selection.csv"), index=False)
    
    # Create Summary
    summary = df.groupby("model").agg(
        accuracy=("accuracy", "mean"),
        avg_reward=("reward", "mean"),
        avg_throughput=("throughput_mbps", "mean"),
        avg_delay=("delay_ms", "mean"),
        avg_drops=("drops", "mean")
    ).reset_index()
    
    print("\n--- Evaluation Summary ---")
    print(summary.to_string(index=False))
    
    summary.to_csv(os.path.join(out_dir, "rl_baseline_comparison.csv"), index=False)

if __name__ == "__main__":
    main()
