"""
Apply the selected Optuna trial config and run the final unified pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager

import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from configs import config as params
from configs.marl_config import MARLConfig
from configs.sarl_config import RLConfig
from experiments.run_unified_experiment import run_unified_experiment


ALGO_FLAG_MAP = {
    "tabular": "RUN_TABULAR_QLEARNING",
    "dqn": "RUN_DQN",
    "ppo": "RUN_PPO",
    "a2c": "RUN_A2C",
    "mca_d3qn": "RUN_CUSTOM_RL",
    "iql": "RUN_MARL_IQL",
    "vdn": "RUN_MARL_VDN",
    "qmix": "RUN_MARL_QMIX",
    "mappo": "RUN_MARL_MAPPO",
    "magat_d3qn": "RUN_MARL_GNN",
}


@contextmanager
def temporary_attrs(target, **overrides):
    snapshot = {key: getattr(target, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(target, key, value)
        yield
    finally:
        for key, value in snapshot.items():
            setattr(target, key, value)


def select_trial(study_dir: str, representative_label: str | None, trial_number: int | None) -> dict:
    if trial_number is not None:
        config_path = os.path.join(study_dir, f"trial_{trial_number}", "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    reps_csv = os.path.join(study_dir, "representative_pareto_points.csv")
    if not os.path.exists(reps_csv):
        raise FileNotFoundError(f"Representative Pareto CSV not found: {reps_csv}")
    reps_df = pd.read_csv(reps_csv)
    if reps_df.empty:
        raise ValueError(f"No representative Pareto points in {reps_csv}")

    label = representative_label or "balanced_min_L2_to_ideal"
    selected = reps_df[reps_df["representative_label"] == label]
    if selected.empty:
        raise ValueError(f"Representative label '{label}' not found in {reps_csv}")
    trial_num = int(selected.iloc[0]["trial_number"])
    config_path = os.path.join(study_dir, f"trial_{trial_num}", "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_trial_config(trial_cfg: dict, final_all_algos: bool):
    hyper = trial_cfg.get("hyperparameters", {})
    reward = trial_cfg.get("reward_weights", {})
    algo = trial_cfg["algorithm"]

    if algo in {"tabular", "dqn"}:
        raise ValueError(
            f"{algo} is unsupported for the current centralized MultiDiscrete burst-action baseline. "
            "Do not include it in the tuned final pipeline."
        )

    marl_overrides = {
        "LR": hyper.get("lr", MARLConfig.LR),
        "BATCH_SIZE": hyper.get("batch_size", MARLConfig.BATCH_SIZE),
        "HIDDEN_DIM": hyper.get("hidden_dim", MARLConfig.HIDDEN_DIM),
        "EPSILON_DECAY": hyper.get("epsilon_decay", MARLConfig.EPSILON_DECAY),
        "TARGET_UPDATE_FREQ": hyper.get("target_update_freq", MARLConfig.TARGET_UPDATE_FREQ),
        "GNN_HEADS": hyper.get("gnn_heads", MARLConfig.GNN_HEADS),
        "W_THROUGHPUT": reward.get("wT", MARLConfig.W_THROUGHPUT),
        "W_DELAY": reward.get("wD", MARLConfig.W_DELAY),
        "W_DROPS": reward.get("wDrop", MARLConfig.W_DROPS),
        "W_COLLISIONS": reward.get("wCol", MARLConfig.W_COLLISIONS),
    }
    sarl_overrides = {
        "REWARD_W_THROUGHPUT": reward.get("wT", RLConfig.REWARD_W_THROUGHPUT),
        "REWARD_W_DELAY": reward.get("wD", RLConfig.REWARD_W_DELAY),
        "REWARD_W_FAILURES": reward.get("wDrop", RLConfig.REWARD_W_FAILURES),
        "REWARD_W_JITTER": reward.get("wCol", RLConfig.REWARD_W_JITTER),
    }

    run_flag_overrides = {flag: False for flag in ALGO_FLAG_MAP.values()}
    if final_all_algos:
        run_flag_overrides = {flag: getattr(params, flag) for flag in ALGO_FLAG_MAP.values()}
    else:
        run_flag_overrides[ALGO_FLAG_MAP[algo]] = True

    return marl_overrides, sarl_overrides, run_flag_overrides


def main():
    parser = argparse.ArgumentParser(description="Run final pipeline from selected Optuna trial")
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--representative-label", default="balanced_min_L2_to_ideal")
    parser.add_argument("--trial-number", type=int, default=None)
    parser.add_argument("--final-all-algos", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--phy-rate-mbps", type=float, default=None)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--qmax", type=int, default=None)
    parser.add_argument("--sweep-min-pps", type=int, default=None)
    parser.add_argument("--sweep-max-pps", type=int, default=None)
    parser.add_argument("--sweep-steps", type=int, default=None)
    parser.add_argument("--artifact-json", default=None)
    args = parser.parse_args()

    trial_cfg = select_trial(args.study_dir, args.representative_label, args.trial_number)
    marl_overrides, sarl_overrides, run_flag_overrides = apply_trial_config(
        trial_cfg, final_all_algos=args.final_all_algos
    )

    with temporary_attrs(MARLConfig, **marl_overrides):
        with temporary_attrs(RLConfig, **sarl_overrides):
            with temporary_attrs(params, **run_flag_overrides):
                artifacts = run_unified_experiment(
                    dry_run=args.dry_run,
                    force_retrain=args.force_retrain,
                    use_checkpoints_only=args.skip_training,
                    phy_rate_mbps=args.phy_rate_mbps,
                    nodes=args.nodes,
                    qmax=args.qmax,
                    sweep_min_pps=args.sweep_min_pps,
                    sweep_max_pps=args.sweep_max_pps,
                    sweep_steps=args.sweep_steps,
                    stochastic_eval=False,
                )

    artifact_json = args.artifact_json
    if artifact_json:
        with open(artifact_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "selected_trial": trial_cfg,
                    "artifacts": artifacts,
                    "execution_mode": {
                        "marl_env": "MARLMacEnv",
                        "marl_control": "decentralized_cluster_head",
                        "sarl_env": "SARLCentralEnv",
                        "burst_split_action": "(m_k, rho_k)",
                    },
                },
                f,
                indent=2,
            )
    else:
        print(json.dumps({"selected_trial": trial_cfg, "artifacts": artifacts}, indent=2))


if __name__ == "__main__":
    main()
