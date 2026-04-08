"""
Complexity and deployability benchmark.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from configs import config as params
from configs.cluster_config import ClusterConfig as CC
from configs.marl_config import MARLConfig
from envs.sarl_central_env import SARLCentralEnv
from experiments.evidence_metrics import model_memory_mb, parameter_count
from experiments.run_unified_experiment import get_magat_arch_kwargs, step3_evaluate


def collect_model_stats(cp_dir: str):
    rows = []

    def append_row(name, model_obj):
        rows.append(
            {
                "Model": name,
                "Parameter_Count": parameter_count(model_obj),
                "Model_Memory_MB": round(model_memory_mb(model_obj), 4),
            }
        )

    if getattr(params, "RUN_CUSTOM_RL", True):
        from algorithms.rl.custom_mca_d3qn import MCABranchingD3QNAgent

        path = os.path.join(cp_dir, "unified_mca_d3qn_model")
        if os.path.exists(path):
            try:
                env = SARLCentralEnv(seed=params.SEED)
                agent = MCABranchingD3QNAgent.load(path, env=env)
                append_row("MCA-D3QN", agent.q_network)
            except Exception:
                pass

    try:
        from stable_baselines3 import DQN, PPO, A2C

        for name, cls, fname in [
            ("DQN", DQN, "unified_dqn_model"),
            ("PPO", PPO, "unified_ppo_model"),
            ("A2C", A2C, "unified_a2c_model"),
        ]:
            path = os.path.join(cp_dir, fname)
            if os.path.exists(path + ".zip"):
                try:
                    model = cls.load(path)
                    append_row(name, model.policy)
                except Exception:
                    pass
    except Exception:
        pass

    from algorithms.rl.marl_baselines import IQLAgent, VDNAgent, QMIXAgent

    for name, cls, fname in [
        ("IQL", IQLAgent, "unified_iql_model.pth"),
        ("VDN", VDNAgent, "unified_vdn_model.pth"),
    ]:
        path = os.path.join(cp_dir, fname)
        if os.path.exists(path):
            try:
                agent = cls(CC.C_MAX, MARLConfig.OBS_DIM, MARLConfig.NUM_ACTIONS)
                agent.load(path)
                append_row(name, agent.q_net)
            except Exception:
                pass

    qmix_path = os.path.join(cp_dir, "unified_qmix_model.pth")
    if os.path.exists(qmix_path):
        try:
            agent = QMIXAgent(CC.C_MAX, MARLConfig.OBS_DIM, MARLConfig.NUM_ACTIONS, embed_dim=MARLConfig.QMIX_EMBED_DIM)
            agent.load(qmix_path)
            append_row("QMIX-QNet", agent.q_net)
            append_row("QMIX-Mixer", agent.mixer)
        except Exception:
            pass

    gnn_path = os.path.join(cp_dir, "unified_gnn_marl_model.pth")
    if os.path.exists(gnn_path):
        try:
            model = torch.load(gnn_path, map_location="cpu")
            gnn = None
            if isinstance(model, dict):
                from algorithms.rl.gnn_marl import MAGAT_D3QN_QNetwork

                gnn = MAGAT_D3QN_QNetwork(**get_magat_arch_kwargs())
                gnn.load_state_dict(model)
            if gnn is not None:
                append_row("MAGAT-D3QN", gnn)
        except Exception:
            pass

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Complexity benchmark")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--load-pps", type=int, default=None, help="Single offered load used for latency measurement")
    args = parser.parse_args()

    if args.out_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(project_root, "results", "journal_complexity", ts)
    else:
        out_dir = os.path.abspath(args.out_dir)
    os.makedirs(os.path.join(out_dir, "csv"), exist_ok=True)

    def log(msg):
        print(msg)

    ref_pps = int(args.load_pps if args.load_pps is not None else 0.5 * (params.SWEEP_MIN_PPS + params.SWEEP_MAX_PPS))
    eval_df = step3_evaluate(
        np.array([ref_pps], dtype=int),
        args.checkpoint_dir,
        out_dir,
        log,
        deterministic_eval=True,
        env_reset_options={},
        csv_name="complexity_eval.csv",
        raw_cluster_csv_name="complexity_cluster_steps.csv",
        summary_json_name="complexity_summary.json",
        save_raw_cluster_logs=True,
    )
    stats_df = collect_model_stats(args.checkpoint_dir)
    stats_csv = os.path.join(out_dir, "csv", "model_complexity_stats.csv")
    stats_df.to_csv(stats_csv, index=False)

    if not eval_df.empty and not stats_df.empty:
        merged = eval_df.merge(stats_df, on="Model", how="left")
        merged.to_csv(os.path.join(out_dir, "csv", "complexity_benchmark.csv"), index=False)

    log(f"[complexity] complete -> {out_dir}")


if __name__ == "__main__":
    main()
