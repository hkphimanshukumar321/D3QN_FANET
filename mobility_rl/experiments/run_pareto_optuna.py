"""
Optuna Multi-Objective Pareto Tuning for FANET MAC RL
=====================================================
Wraps the unified SARL+MARL experiment pipeline with Optuna's NSGA-II
sampler for multi-objective hyperparameter and reward-weight tuning.

Design invariants
-----------------
* One **study** = one **algorithm** (CLI: ``--algo dqn``).
* Reward weights are *search variables*, but Pareto dominance is
  computed on *evaluation metrics* only.
* First 15 trials use fixed anchor weight combinations; subsequent
  trials use Optuna's continuous normalised sampling.
* Production default: ``n_jobs=1`` — only one GPU trial at a time
  on the single V100.

Usage
-----
    # Single-algorithm study (dry-run, 4 trials):
    python experiments/run_pareto_optuna.py --algo dqn --n-trials 4 --dry-run

    # Multi-algorithm sequential run:
    python experiments/run_pareto_optuna.py --algos dqn,iql,vdn,magat_d3qn --dry-run

    # All recommended algorithms:
    python experiments/run_pareto_optuna.py --algos all --n-trials 48 --study-name full_sweep
"""

import os
import sys
import json
import copy
import shutil
import datetime
import argparse
import traceback
import platform
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import optuna

# Wandb integration — gracefully no-op if unavailable
from utils.experiment_tracking import (
    init_run as wandb_init, log_metrics as wandb_log,
    log_artifact as wandb_artifact, log_image as wandb_image,
    finish_run as wandb_finish, is_enabled as wandb_enabled,
)
from optuna.samplers import NSGAIISampler

# =====================================================================
# Constants & Canonical Mappings
# =====================================================================

# CLI name -> display name (same as in run_unified_experiment.py)
ALGO_NAME_MAP = {
    "tabular": "Tabular",
    "dqn": "DQN",
    "ppo": "PPO",
    "a2c": "A2C",
    "mca_d3qn": "MCA-D3QN",
    "iql": "IQL",
    "vdn": "VDN",
    "qmix": "QMIX",
    "magat_d3qn": "MAGAT-D3QN",
}

VALID_ALGOS = list(ALGO_NAME_MAP.keys())

# Default recommended algorithms for --algos all
DEFAULT_ALL_ALGOS = ["dqn", "iql", "vdn", "qmix", "magat_d3qn"]

# Is this a GNN algorithm?  (gets reduced trial budget)
GNN_ALGOS = {"magat_d3qn"}

# 15 fixed anchor weight combinations: (wT, wD, wDrop, wCol)
ANCHOR_WEIGHTS = [
    (0.25, 0.25, 0.25, 0.25),   # 1.  Uniform
    (0.55, 0.15, 0.15, 0.15),   # 2.  Throughput emphasis
    (0.15, 0.55, 0.15, 0.15),   # 3.  Delay emphasis
    (0.15, 0.15, 0.55, 0.15),   # 4.  Drop emphasis
    (0.15, 0.15, 0.15, 0.55),   # 5.  Collision emphasis
    (0.40, 0.40, 0.10, 0.10),   # 6.  T+D emphasis
    (0.40, 0.10, 0.40, 0.10),   # 7.  T+Drop emphasis
    (0.40, 0.10, 0.10, 0.40),   # 8.  T+Col emphasis
    (0.10, 0.40, 0.40, 0.10),   # 9.  D+Drop emphasis
    (0.10, 0.40, 0.10, 0.40),   # 10. D+Col emphasis
    (0.10, 0.10, 0.40, 0.40),   # 11. Drop+Col emphasis
    (0.40, 0.20, 0.20, 0.20),   # 12. Practical T-focused
    (0.35, 0.25, 0.25, 0.15),   # 13. Practical balanced 1
    (0.35, 0.25, 0.15, 0.25),   # 14. Practical balanced 2
    (0.30, 0.30, 0.20, 0.20),   # 15. Practical balanced 3
]
NUM_ANCHORS = len(ANCHOR_WEIGHTS)

# Trial budget recommendations
TRIAL_BUDGET = {
    "dry_run": 4,       # 2 anchor + 2 Optuna
    "pilot": 16,
    "non_gnn": 48,      # 15 anchor + 33 Optuna
    "magat_d3qn": 24,   # 15 anchor + 9 Optuna
    "paper_quality": 64,
}

# Objective definitions
# NOTE: 'inference' is intentionally excluded from DEFAULT_OBJECTIVES.
# It varies minimally across trials and adds noise to the Pareto front.
# It is always *logged* as a metric but only included as a Pareto
# direction when the user passes --include-inference.
DEFAULT_OBJECTIVES = ["throughput", "delay", "drops", "collisions"]
ALL_OBJECTIVES = ["throughput", "delay", "drops", "collisions", "inference"]
OBJECTIVE_DIRECTIONS = {
    "throughput": "maximize",
    "delay": "minimize",
    "drops": "minimize",
    "collisions": "minimize",
    "inference": "minimize",
}

# =====================================================================
# Helpers
# =====================================================================

def _get_device_info():
    """Gather device information for trial documentation."""
    info = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    except ImportError:
        info["cuda_available"] = False
    return info


def _is_pareto_efficient(costs):
    """Return boolean mask of Pareto-efficient rows.

    Parameters
    ----------
    costs : np.ndarray of shape (n_trials, n_objectives)
        Already sign-adjusted so that *lower is better* for every column.
    """
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        # Keep i; remove points dominated by i
        dominated = np.all(costs[i] <= costs, axis=1) & np.any(costs[i] < costs, axis=1)
        is_efficient[dominated] = False
        is_efficient[i] = True  # keep self
    return is_efficient


def _compute_costs(df, objectives):
    """Build a cost matrix from a trials dataframe (lower = better)."""
    cols = []
    for obj in objectives:
        if OBJECTIVE_DIRECTIONS[obj] == "maximize":
            cols.append(-df[f"obj_{obj}"].values)
        else:
            cols.append(df[f"obj_{obj}"].values)
    return np.column_stack(cols)


# =====================================================================
# Objective Function
# =====================================================================

def create_objective(
    algo_cli,
    objectives,
    study_dir,
    dry_run,
    force_retrain,
    base_seed,
    extra_kwargs,
    gpu_ids,
    intra_trial_workers,
):
    """Return an Optuna objective callable for one algorithm."""

    algo_display = ALGO_NAME_MAP[algo_cli]
    num_anchors_in_study = min(NUM_ANCHORS, extra_kwargs.get("_n_trials", 999))
    # For dry-run: use 2 anchor + 2 Optuna
    if dry_run:
        num_anchors_in_study = min(2, extra_kwargs.get("_n_trials", 4))

    def objective(trial):
        trial_start = datetime.datetime.now()
        trial_num = trial.number
        trial_dir = os.path.join(study_dir, f"trial_{trial_num}")
        os.makedirs(trial_dir, exist_ok=True)

        # --- GPU visibility ---
        if gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

        try:
            # =========================================================
            # 1. Sample reward weights (two-stage)
            # =========================================================
            if trial_num < num_anchors_in_study:
                # Stage 7A: fixed anchor
                anchor_idx = trial_num % NUM_ANCHORS
                wT, wD, wDrop, wCol = ANCHOR_WEIGHTS[anchor_idx]
                weight_source = "anchor"
                trial.set_user_attr("weight_source", "anchor")
                trial.set_user_attr("anchor_index", anchor_idx)
            else:
                # Stage 7B: Optuna continuous refinement
                raw_t = trial.suggest_float("raw_wT", 0.05, 1.0)
                raw_d = trial.suggest_float("raw_wD", 0.05, 1.0)
                raw_dr = trial.suggest_float("raw_wDrop", 0.05, 1.0)
                raw_co = trial.suggest_float("raw_wCol", 0.05, 1.0)
                s = raw_t + raw_d + raw_dr + raw_co
                wT = round(raw_t / s, 4)
                wD = round(raw_d / s, 4)
                wDrop = round(raw_dr / s, 4)
                wCol = round(1.0 - wT - wD - wDrop, 4)  # ensure exact sum=1
                weight_source = "optuna"
                trial.set_user_attr("weight_source", "optuna")

            trial.set_user_attr("wT", wT)
            trial.set_user_attr("wD", wD)
            trial.set_user_attr("wDrop", wDrop)
            trial.set_user_attr("wCol", wCol)

            # =========================================================
            # 2. Sample hyperparameters
            # =========================================================
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
            epsilon_decay = trial.suggest_int("epsilon_decay", 500, 5000)
            target_update_freq = trial.suggest_int("target_update_freq", 200, 2000)

            gnn_heads = 4
            if algo_cli == "magat_d3qn":
                gnn_heads = trial.suggest_categorical("gnn_heads", [2, 4, 8])

            # =========================================================
            # 3. Override configs and run unified experiment
            # =========================================================
            from configs.marl_config import MARLConfig
            from configs.sarl_config import RLConfig

            # Save original values for restoration (MARL + SARL)
            orig_marl = {
                "LR": MARLConfig.LR,
                "BATCH_SIZE": MARLConfig.BATCH_SIZE,
                "HIDDEN_DIM": MARLConfig.HIDDEN_DIM,
                "EPSILON_DECAY": MARLConfig.EPSILON_DECAY,
                "TARGET_UPDATE_FREQ": MARLConfig.TARGET_UPDATE_FREQ,
                "GNN_HEADS": MARLConfig.GNN_HEADS,
                "W_THROUGHPUT": MARLConfig.W_THROUGHPUT,
                "W_DELAY": MARLConfig.W_DELAY,
                "W_DROPS": MARLConfig.W_DROPS,
                "W_COLLISIONS": MARLConfig.W_COLLISIONS,
            }
            orig_sarl = {
                "REWARD_W_THROUGHPUT": RLConfig.REWARD_W_THROUGHPUT,
                "REWARD_W_DELAY": RLConfig.REWARD_W_DELAY,
                "REWARD_W_FAILURES": RLConfig.REWARD_W_FAILURES,
                "REWARD_W_JITTER": RLConfig.REWARD_W_JITTER,
            }

            # Apply sampled hyperparameters
            MARLConfig.LR = lr
            MARLConfig.BATCH_SIZE = batch_size
            MARLConfig.HIDDEN_DIM = hidden_dim
            MARLConfig.EPSILON_DECAY = epsilon_decay
            MARLConfig.TARGET_UPDATE_FREQ = target_update_freq
            MARLConfig.GNN_HEADS = gnn_heads

            # Apply sampled reward weights to BOTH configs
            MARLConfig.W_THROUGHPUT = wT
            MARLConfig.W_DELAY = wD
            MARLConfig.W_DROPS = wDrop
            MARLConfig.W_COLLISIONS = wCol

            RLConfig.REWARD_W_THROUGHPUT = wT
            RLConfig.REWARD_W_DELAY = wD
            RLConfig.REWARD_W_FAILURES = wDrop
            RLConfig.REWARD_W_JITTER = wCol

            # Run the inner experiment
            from experiments.run_unified_experiment import run_unified_experiment

            run_kwargs = dict(
                dry_run=dry_run,
                force_retrain=True,  # always retrain for tuning
                use_checkpoints_only=False,
                stochastic_eval=False,
            )
            # Pass through sweep overrides
            for k in ["phy_rate_mbps", "nodes", "qmax",
                       "sweep_min_pps", "sweep_max_pps", "sweep_steps"]:
                if k in extra_kwargs and extra_kwargs[k] is not None:
                    run_kwargs[k] = extra_kwargs[k]

            # Set seed per trial
            from configs import config as params
            params.SEED = base_seed + trial_num

            result = run_unified_experiment(**run_kwargs)

            # Restore original configs (MARL + SARL)
            for k, v in orig_marl.items():
                setattr(MARLConfig, k, v)
            for k, v in orig_sarl.items():
                setattr(RLConfig, k, v)

            # =========================================================
            # 4. Aggregate metrics from eval CSV
            # =========================================================
            eval_csv = result["eval_csv"]
            if not os.path.exists(eval_csv):
                raise FileNotFoundError(f"Eval CSV not found: {eval_csv}")

            df = pd.read_csv(eval_csv)
            algo_rows = df[df["Model"] == algo_display]

            if algo_rows.empty:
                raise ValueError(
                    f"No rows for {algo_display} in {eval_csv}. "
                    f"Available: {list(df['Model'].unique())}"
                )

            # Always collect ALL metrics (including inference)
            # even if inference is not a Pareto objective.
            metrics = {
                "throughput": float(algo_rows["Throughput_Mbps"].mean()),
                "delay": float(algo_rows["Delay_ms"].mean()),
                "drops": float(algo_rows["Drops"].sum()),
                "collisions": float(algo_rows["Collisions"].sum()),
                "inference": float(algo_rows["Avg_Inference_ms"].mean()),
            }

            # =========================================================
            # 5. Save trial documentation (Task 13)
            # =========================================================
            trial_end = datetime.datetime.now()
            trial_config = {
                "trial_number": trial_num,
                "algorithm": algo_cli,
                "algorithm_display": algo_display,
                "objectives": objectives,
                "aggregation": "mean/sum per objective",
                "seed": base_seed + trial_num,
                "device_info": _get_device_info(),
                "trial_start": trial_start.isoformat(),
                "trial_end": trial_end.isoformat(),
                "trial_duration_s": round((trial_end - trial_start).total_seconds(), 2),
                "trial_state": "COMPLETE",
                "weight_source": weight_source,
                "reward_weights": {
                    "wT": wT, "wD": wD, "wDrop": wDrop, "wCol": wCol,
                },
                "hyperparameters": {
                    "lr": lr,
                    "batch_size": batch_size,
                    "hidden_dim": hidden_dim,
                    "epsilon_decay": epsilon_decay,
                    "target_update_freq": target_update_freq,
                    "gnn_heads": gnn_heads,
                },
                "objective_values": {f"obj_{k}": v for k, v in metrics.items() if k in objectives},
                "all_metrics": metrics,
                "unified_run_dir": result["out_dir"],
            }

            # --- wandb: log per-trial metrics ---
            if wandb_enabled():
                wandb_log({
                    f"optuna/{algo_cli}/trial": trial_num,
                    f"optuna/{algo_cli}/throughput": metrics["throughput"],
                    f"optuna/{algo_cli}/delay": metrics["delay"],
                    f"optuna/{algo_cli}/drops": metrics["drops"],
                    f"optuna/{algo_cli}/collisions": metrics["collisions"],
                    f"optuna/{algo_cli}/inference": metrics["inference"],
                    f"optuna/{algo_cli}/wT": wT,
                    f"optuna/{algo_cli}/wD": wD,
                    f"optuna/{algo_cli}/wDrop": wDrop,
                    f"optuna/{algo_cli}/wCol": wCol,
                    f"optuna/{algo_cli}/weight_source": 1.0 if weight_source == "anchor" else 0.0,
                    f"optuna/{algo_cli}/lr": lr,
                    f"optuna/{algo_cli}/duration_s": round((trial_end - trial_start).total_seconds(), 2),
                }, step=trial_num)

            with open(os.path.join(trial_dir, "config.json"), "w") as f:
                json.dump(trial_config, f, indent=2)

            # Copy key artifacts into trial directory
            for src_key, fname in [
                ("eval_csv", "unified_eval_sweep.csv"),
                ("baseline_csv", "baseline_results.csv"),
                ("summary_json", "experiment_summary.json"),
            ]:
                src = result.get(src_key)
                if src and os.path.exists(src):
                    shutil.copy2(src, os.path.join(trial_dir, fname))

            # =========================================================
            # 6. Return objectives to Optuna
            # =========================================================
            obj_values = tuple(metrics[obj] for obj in objectives)
            return obj_values

        except Exception as e:
            # Record failure but do not crash the study
            trial_end = datetime.datetime.now()
            error_info = {
                "trial_number": trial_num,
                "trial_state": "FAILED",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "trial_start": trial_start.isoformat(),
                "trial_end": trial_end.isoformat(),
                "trial_duration_s": round((trial_end - trial_start).total_seconds(), 2),
            }
            with open(os.path.join(trial_dir, "config.json"), "w") as f:
                json.dump(error_info, f, indent=2)

            # Return NaN — Optuna will mark trial as failed
            return tuple(float("nan") for _ in objectives)

    return objective


# =====================================================================
# Post-Study Analysis
# =====================================================================

def save_study_artifacts(study, study_dir, objectives, algo_cli):
    """Save all study-level artifacts (Task 13, 14, 15)."""
    print(f"\n{'='*60}")
    print(f"  Post-Study Analysis: {algo_cli}")
    print(f"{'='*60}")

    # --- trials_summary.csv / .json (Task 13) ---
    rows = []
    for t in study.trials:
        row = {
            "trial_number": t.number,
            "state": t.state.name,
        }
        # Objective values
        if t.values is not None:
            for i, obj in enumerate(objectives):
                row[f"obj_{obj}"] = t.values[i] if i < len(t.values) else None
        else:
            for obj in objectives:
                row[f"obj_{obj}"] = None
        # User attributes (weights, hyperparams)
        for k, v in t.user_attrs.items():
            row[k] = v
        # Params
        for k, v in t.params.items():
            row[f"param_{k}"] = v
        row["duration_s"] = (
            (t.datetime_complete - t.datetime_start).total_seconds()
            if t.datetime_start and t.datetime_complete else None
        )
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(study_dir, "trials_summary.csv"), index=False)
    with open(os.path.join(study_dir, "trials_summary.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"  Saved trials_summary.csv + .json ({len(rows)} trials)")

    # --- Filter to complete trials with valid objectives ---
    complete = summary_df[summary_df["state"] == "COMPLETE"].copy()
    obj_cols = [f"obj_{o}" for o in objectives]
    for c in obj_cols:
        if c in complete.columns:
            complete[c] = pd.to_numeric(complete[c], errors="coerce")
    complete = complete.dropna(subset=obj_cols)

    if complete.empty:
        print("  WARNING: No complete trials with valid objectives. Skipping Pareto analysis.")
        # Still save empty files
        pd.DataFrame().to_csv(os.path.join(study_dir, "pareto_front.csv"), index=False)
        with open(os.path.join(study_dir, "pareto_front.json"), "w") as f:
            json.dump([], f)
        pd.DataFrame().to_csv(os.path.join(study_dir, "representative_pareto_points.csv"), index=False)
        with open(os.path.join(study_dir, "representative_pareto_points.json"), "w") as f:
            json.dump([], f)
        return

    # --- Pareto front extraction ---
    costs = _compute_costs(complete, objectives)
    pareto_mask = _is_pareto_efficient(costs)
    pareto_df = complete[pareto_mask].copy()

    pareto_df.to_csv(os.path.join(study_dir, "pareto_front.csv"), index=False)
    with open(os.path.join(study_dir, "pareto_front.json"), "w") as f:
        json.dump(pareto_df.to_dict(orient="records"), f, indent=2, default=str)
    print(f"  Pareto front: {len(pareto_df)} non-dominated trials out of {len(complete)}")

    # --- weight_anchors.csv (Task 7A) ---
    anchor_df = pd.DataFrame(ANCHOR_WEIGHTS, columns=["wT", "wD", "wDrop", "wCol"])
    anchor_df.index.name = "anchor_index"
    anchor_df.to_csv(os.path.join(study_dir, "weight_anchors.csv"))
    print(f"  Saved weight_anchors.csv ({len(ANCHOR_WEIGHTS)} anchors)")

    # --- Representative Pareto points (Task 15) ---
    _save_representative_points(pareto_df, objectives, study_dir)

    # --- Plots (Task 14) ---
    plots_dir = os.path.join(study_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    _generate_pareto_plots(complete, pareto_df, objectives, plots_dir, algo_cli)


def _save_representative_points(pareto_df, objectives, study_dir):
    """Identify and save representative Pareto points (Task 15)."""
    reps = []
    obj_cols = [f"obj_{o}" for o in objectives]

    # Best per objective
    for obj, col in zip(objectives, obj_cols):
        direction = OBJECTIVE_DIRECTIONS[obj]
        if direction == "maximize":
            idx = pareto_df[col].idxmax()
        else:
            idx = pareto_df[col].idxmin()
        row = pareto_df.loc[idx].to_dict()
        row["representative_label"] = f"best_{obj}"
        reps.append(row)

    # Balanced point: min L2 distance to normalised ideal
    # Each objective is normalised to [0,1] within the Pareto front,
    # then flipped so lower = better. The point closest to the origin
    # (the ideal corner) in this normalised space is selected.
    if len(pareto_df) > 1:
        normed = pd.DataFrame()
        for obj, col in zip(objectives, obj_cols):
            vals = pareto_df[col].values.astype(float)
            vmin, vmax = vals.min(), vals.max()
            rng = max(vmax - vmin, 1e-12)
            if OBJECTIVE_DIRECTIONS[obj] == "maximize":
                normed[col] = (vmax - vals) / rng  # lower is better after flip
            else:
                normed[col] = (vals - vmin) / rng
        dists = np.sqrt((normed.values ** 2).sum(axis=1))
        balanced_idx = pareto_df.index[np.argmin(dists)]
        row = pareto_df.loc[balanced_idx].to_dict()
        row["representative_label"] = "balanced_min_L2_to_ideal"
        reps.append(row)

    reps_df = pd.DataFrame(reps)
    reps_df.to_csv(os.path.join(study_dir, "representative_pareto_points.csv"), index=False)
    with open(os.path.join(study_dir, "representative_pareto_points.json"), "w") as f:
        json.dump(reps, f, indent=2, default=str)
    print(f"  Saved representative_pareto_points ({len(reps)} points)")


def _generate_pareto_plots(complete_df, pareto_df, objectives, plots_dir, algo_cli):
    """Generate all required Pareto charts (Task 14)."""
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", font_scale=1.1)
    except ImportError:
        pass

    algo_display = ALGO_NAME_MAP.get(algo_cli, algo_cli)
    dom_label = "Non-dominated (Pareto)"
    nondom_label = "Dominated"

    # Helper: 2D Pareto scatter
    def _scatter_2d(fname, x_col, y_col, x_label, y_label, title):
        fig, ax = plt.subplots(figsize=(10, 7))
        # Dominated trials
        dominated = complete_df[~complete_df.index.isin(pareto_df.index)]
        if not dominated.empty:
            ax.scatter(dominated[x_col], dominated[y_col],
                       c="gray", alpha=0.4, s=40, label=nondom_label, edgecolors="none")
        # Pareto front
        pf_sorted = pareto_df.sort_values(x_col)
        ax.scatter(pf_sorted[x_col], pf_sorted[y_col],
                   c="crimson", s=80, zorder=5, label=dom_label, edgecolors="black", linewidth=0.5)
        ax.plot(pf_sorted[x_col], pf_sorted[y_col],
                "r--", alpha=0.5, linewidth=1.2, zorder=4)

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f"{algo_display}: {title}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, frameon=True)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, fname), dpi=200)
        plt.close()
        # Save source data
        pf_sorted.to_csv(os.path.join(plots_dir, fname.replace(".png", "_data.csv")), index=False)

    # 1. Throughput vs Delay
    if "throughput" in objectives and "delay" in objectives:
        _scatter_2d("throughput_vs_delay_pareto.png",
                     "obj_delay", "obj_throughput",
                     "Mean Delay (ms)", "Mean Throughput (Mbps)",
                     "Throughput vs Delay Pareto Front")

    # 2. Throughput vs Drops
    if "throughput" in objectives and "drops" in objectives:
        _scatter_2d("throughput_vs_drops_pareto.png",
                     "obj_drops", "obj_throughput",
                     "Total Drops", "Mean Throughput (Mbps)",
                     "Throughput vs Drops Pareto Front")

    # 3. Throughput vs Collisions
    if "throughput" in objectives and "collisions" in objectives:
        _scatter_2d("throughput_vs_collisions_pareto.png",
                     "obj_collisions", "obj_throughput",
                     "Total Collisions", "Mean Throughput (Mbps)",
                     "Throughput vs Collisions Pareto Front")

    # 4. Throughput vs Inference
    if "throughput" in objectives and "inference" in objectives:
        _scatter_2d("throughput_vs_inference_pareto.png",
                     "obj_inference", "obj_throughput",
                     "Mean Inference Time (ms)", "Mean Throughput (Mbps)",
                     "Throughput vs Inference Pareto Front")

    # 5. Parallel coordinates (Pareto front only, or top trials)
    _plot_parallel_coordinates(pareto_df, objectives, plots_dir, algo_display)

    # 6. Reward weights vs objectives
    _plot_weights_vs_objectives(complete_df, objectives, plots_dir, algo_display)

    print(f"  Saved Pareto plots to {plots_dir}")


def _plot_parallel_coordinates(pareto_df, objectives, plots_dir, algo_display):
    """Parallel coordinates plot for Pareto-front trials (Task 14.5)."""
    if pareto_df.empty:
        return

    weight_cols = ["wT", "wD", "wDrop", "wCol"]
    obj_cols = [f"obj_{o}" for o in objectives]
    avail_cols = [c for c in weight_cols + obj_cols if c in pareto_df.columns]
    if len(avail_cols) < 3:
        return

    plot_df = pareto_df[avail_cols].copy()
    # Normalize each column to [0, 1] for visualization
    for c in avail_cols:
        vals = plot_df[c].values.astype(float)
        vmin, vmax = vals.min(), vals.max()
        rng = max(vmax - vmin, 1e-12)
        plot_df[c] = (vals - vmin) / rng

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(avail_cols))
    for _, row in plot_df.iterrows():
        ax.plot(x, row[avail_cols].values.astype(float), alpha=0.6, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(avail_cols, rotation=30, ha="right")
    ax.set_ylabel("Normalised Value [0, 1]")
    ax.set_title(f"{algo_display}: Parallel Coordinates — Pareto Front Trials", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "parallel_coordinates_pareto.png"), dpi=200)
    plt.close()


def _plot_weights_vs_objectives(complete_df, objectives, plots_dir, algo_display):
    """How reward weights map to objective outcomes (Task 14.6)."""
    weight_cols = ["wT", "wD", "wDrop", "wCol"]
    obj_cols = [f"obj_{o}" for o in objectives]
    avail_w = [c for c in weight_cols if c in complete_df.columns]
    avail_o = [c for c in obj_cols if c in complete_df.columns]
    if not avail_w or not avail_o:
        return

    n_obj = len(avail_o)
    fig, axes = plt.subplots(len(avail_w), n_obj, figsize=(5 * n_obj, 4 * len(avail_w)), squeeze=False)
    for i, w_col in enumerate(avail_w):
        for j, o_col in enumerate(avail_o):
            ax = axes[i][j]
            ax.scatter(complete_df[w_col], complete_df[o_col],
                       alpha=0.5, s=30, c="steelblue", edgecolors="none")
            ax.set_xlabel(w_col)
            ax.set_ylabel(o_col)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"{algo_display}: Reward Weights → Objective Outcomes", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "reward_weights_vs_objectives.png"), dpi=150)
    plt.close()


# =====================================================================
# Study Runner
# =====================================================================

def run_single_study(
    algo_cli,
    n_trials,
    study_name,
    storage,
    objectives,
    dry_run,
    force_retrain,
    seed,
    n_jobs,
    gpu_ids,
    max_concurrent_gpu_trials,
    intra_trial_workers,
    timeout,
    extra_kwargs,
):
    """Run one Optuna study for one algorithm."""
    algo_display = ALGO_NAME_MAP.get(algo_cli, algo_cli)
    full_study_name = f"{study_name}_{algo_cli}"

    print(f"\n{'#'*70}")
    print(f"  OPTUNA PARETO STUDY: {algo_display}")
    print(f"  Study name: {full_study_name}")
    print(f"  Trials: {n_trials}  |  Objectives: {objectives}")
    print(f"  n_jobs={n_jobs}  |  GPU IDs: {gpu_ids}")
    print(f"  Dry-run: {dry_run}  |  Force retrain: {force_retrain}")
    print(f"{'#'*70}\n")

    # --- wandb: initialise a study-level run ---
    wandb_init(
        project="fanet-mac-rl",
        config={
            "study_name": full_study_name,
            "algorithm": algo_cli,
            "algorithm_display": algo_display,
            "objectives": objectives,
            "n_trials": n_trials,
            "dry_run": dry_run,
            "seed": seed,
            "n_jobs": n_jobs,
            "gpu_ids": gpu_ids,
        },
        run_name=f"optuna_{full_study_name}",
        tags=["optuna", "pareto", algo_cli],
        group="optuna_pareto",
        notes=f"Pareto tuning for {algo_display}, {n_trials} trials",
    )

    # Resolve storage
    study_base = os.path.join(project_root, "results", "optuna", full_study_name)
    os.makedirs(study_base, exist_ok=True)

    if storage is None:
        db_path = os.path.join(study_base, f"{full_study_name}.db")
        storage = f"sqlite:///{db_path}"

    # Create study
    directions = [OBJECTIVE_DIRECTIONS[o] for o in objectives]
    study = optuna.create_study(
        study_name=full_study_name,
        storage=storage,
        sampler=NSGAIISampler(seed=seed),
        directions=directions,
        load_if_exists=True,
    )

    # Store metadata
    study.set_user_attr("algorithm", algo_cli)
    study.set_user_attr("algorithm_display", algo_display)
    study.set_user_attr("objectives", objectives)
    study.set_user_attr("directions", directions)
    study.set_user_attr("dry_run", dry_run)
    study.set_user_attr("n_jobs", n_jobs)
    study.set_user_attr("max_concurrent_gpu_trials", max_concurrent_gpu_trials)
    study.set_user_attr("intra_trial_workers", intra_trial_workers)
    study.set_user_attr("seed", seed)

    # Build objective
    extra_kwargs["_n_trials"] = n_trials
    obj_fn = create_objective(
        algo_cli=algo_cli,
        objectives=objectives,
        study_dir=study_base,
        dry_run=dry_run,
        force_retrain=force_retrain,
        base_seed=seed,
        extra_kwargs=extra_kwargs,
        gpu_ids=gpu_ids,
        intra_trial_workers=intra_trial_workers,
    )

    # Run optimization
    # Enforce safe GPU policy: production always n_jobs=1
    safe_n_jobs = min(n_jobs, max_concurrent_gpu_trials)
    study.optimize(
        obj_fn,
        n_trials=n_trials,
        n_jobs=safe_n_jobs,
        timeout=timeout,
        show_progress_bar=True,
    )

    # Post-study analysis
    save_study_artifacts(study, study_base, objectives, algo_cli)

    # Save study-level execution summary
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    exec_summary = {
        "study_name": full_study_name,
        "algorithm": algo_cli,
        "algorithm_display": algo_display,
        "objectives": objectives,
        "directions": directions,
        "n_trials_requested": n_trials,
        "n_trials_completed": n_completed,
        "n_trials_failed": n_failed,
        "dry_run": dry_run,
        "n_jobs": safe_n_jobs,
        "max_concurrent_gpu_trials": max_concurrent_gpu_trials,
        "intra_trial_workers": intra_trial_workers,
        "gpu_ids": gpu_ids,
        "seed": seed,
        "storage": storage,
        "completed_at": datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(study_base, "study_execution_summary.json"), "w") as f:
        json.dump(exec_summary, f, indent=2)

    # --- wandb: log study summary + upload Pareto artifacts ---
    if wandb_enabled():
        wandb_log({
            f"optuna/{algo_cli}/n_completed": n_completed,
            f"optuna/{algo_cli}/n_failed": n_failed,
        })
        # Upload Pareto artifacts
        for artifact_name in ["pareto_front.csv", "representative_pareto_points.csv",
                               "trials_summary.csv", "study_execution_summary.json"]:
            fpath = os.path.join(study_base, artifact_name)
            if os.path.exists(fpath):
                wandb_artifact(fpath, artifact_type="result")
        # Upload Pareto plots
        plots_dir = os.path.join(study_base, "plots")
        if os.path.isdir(plots_dir):
            for img_file in os.listdir(plots_dir):
                if img_file.endswith(".png"):
                    img_path = os.path.join(plots_dir, img_file)
                    wandb_image(f"optuna/{algo_cli}/{img_file.replace('.png', '')}", img_path)
    wandb_finish()

    return study, study_base


# =====================================================================
# CLI
# =====================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Optuna Multi-Objective Pareto Tuning for FANET MAC RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run for DQN (4 trials):
  python experiments/run_pareto_optuna.py --algo dqn --n-trials 4 --dry-run

  # Multi-algorithm run:
  python experiments/run_pareto_optuna.py --algos dqn,iql,vdn,magat_d3qn --dry-run

  # All recommended algorithms:
  python experiments/run_pareto_optuna.py --algos all --n-trials 48
        """,
    )

    # Algorithm selection (mutually exclusive: --algo XOR --algos)
    algo_group = p.add_mutually_exclusive_group(required=True)
    algo_group.add_argument(
        "--algo", type=str, choices=VALID_ALGOS,
        help="Single algorithm to tune."
    )
    algo_group.add_argument(
        "--algos", type=str,
        help="Comma-separated list of algorithms, or 'all'."
    )

    # Study configuration
    p.add_argument("--study-name", type=str, default="pareto_study",
                   help="Base study name (algo is appended).")
    p.add_argument("--storage", type=str, default=None,
                   help="Optuna storage URL. Default: sqlite in results/optuna/.")
    p.add_argument("--n-trials", type=int, default=None,
                   help="Number of trials. Default: auto based on algo + mode.")
    p.add_argument("--timeout", type=float, default=None,
                   help="Timeout in seconds for the entire study.")

    # Objective selection
    p.add_argument("--objectives", nargs="+", default=DEFAULT_OBJECTIVES,
                   choices=ALL_OBJECTIVES,
                   help="Objective metrics for Pareto optimization. Default: throughput, delay, drops, collisions. "
                        "Inference is always logged but excluded from Pareto directions by default.")
    p.add_argument("--include-inference", action="store_true",
                   help="Add inference time as an objective.")

    # Experiment overrides
    p.add_argument("--dry-run", action="store_true",
                   help="Use reduced training for smoke testing.")
    p.add_argument("--force-retrain", action="store_true",
                   help="Force retraining even if checkpoints exist.")
    p.add_argument("--seed", type=int, default=42, help="Base random seed.")
    p.add_argument("--nodes", type=int, default=None)
    p.add_argument("--qmax", type=int, default=None)
    p.add_argument("--phy-rate-mbps", type=float, default=None)
    p.add_argument("--sweep-min-pps", type=int, default=None)
    p.add_argument("--sweep-max-pps", type=int, default=None)
    p.add_argument("--sweep-steps", type=int, default=None)

    # Execution policy
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Optuna parallel jobs. Default: 1 (single V100).")
    p.add_argument("--gpu-ids", type=str, default=None,
                   help="CUDA_VISIBLE_DEVICES string, e.g. '0'.")
    p.add_argument("--max-concurrent-gpu-trials", type=int, default=1,
                   help="Max GPU trials in parallel. Default: 1.")
    p.add_argument("--intra-trial-workers", type=int, default=1,
                   help="Workers within each trial. Default: 1.")

    return p


def resolve_trial_count(args, algo_cli):
    """Determine the number of trials for the given algo."""
    if args.n_trials is not None:
        return args.n_trials
    if args.dry_run:
        return TRIAL_BUDGET["dry_run"]
    if algo_cli in GNN_ALGOS:
        return TRIAL_BUDGET["magat_d3qn"]
    return TRIAL_BUDGET["non_gnn"]


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolve algorithm list
    if args.algo:
        algo_list = [args.algo]
    else:
        raw = args.algos.strip()
        if raw.lower() == "all":
            algo_list = DEFAULT_ALL_ALGOS
        else:
            algo_list = [a.strip() for a in raw.split(",")]
            for a in algo_list:
                if a not in VALID_ALGOS:
                    parser.error(f"Unknown algorithm: {a}. Valid: {VALID_ALGOS}")

    # Add inference to objectives if requested
    objectives = list(args.objectives)
    if args.include_inference and "inference" not in objectives:
        objectives.append("inference")

    # Extra kwargs to pass through to inner experiment
    extra_kwargs = {
        "phy_rate_mbps": args.phy_rate_mbps,
        "nodes": args.nodes,
        "qmax": args.qmax,
        "sweep_min_pps": args.sweep_min_pps,
        "sweep_max_pps": args.sweep_max_pps,
        "sweep_steps": args.sweep_steps,
    }

    print("=" * 70)
    print("  OPTUNA PARETO TUNING — FANET MAC RL")
    print(f"  Algorithms: {algo_list}")
    print(f"  Objectives: {objectives}")
    print(f"  Dry-run: {args.dry_run}")
    print(f"  n_jobs={args.n_jobs}  |  GPU IDs: {args.gpu_ids}")
    print(f"  max_concurrent_gpu_trials={args.max_concurrent_gpu_trials}")
    print("=" * 70)

    all_study_results = []

    for algo_cli in algo_list:
        n_trials = resolve_trial_count(args, algo_cli)
        print(f"\n  >>> Algorithm: {ALGO_NAME_MAP[algo_cli]} — {n_trials} trials")

        study, study_dir = run_single_study(
            algo_cli=algo_cli,
            n_trials=n_trials,
            study_name=args.study_name,
            storage=args.storage,
            objectives=objectives,
            dry_run=args.dry_run,
            force_retrain=args.force_retrain,
            seed=args.seed,
            n_jobs=args.n_jobs,
            gpu_ids=args.gpu_ids,
            max_concurrent_gpu_trials=args.max_concurrent_gpu_trials,
            intra_trial_workers=args.intra_trial_workers,
            timeout=args.timeout,
            extra_kwargs=extra_kwargs,
        )

        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        all_study_results.append({
            "algorithm": algo_cli,
            "display_name": ALGO_NAME_MAP[algo_cli],
            "n_trials": n_trials,
            "n_completed": completed,
            "study_dir": study_dir,
        })

    # Master summary across all studies
    if len(algo_list) > 1:
        master_dir = os.path.join(project_root, "results", "optuna", args.study_name + "_master")
        os.makedirs(master_dir, exist_ok=True)
        with open(os.path.join(master_dir, "master_summary.json"), "w") as f:
            json.dump({
                "study_name": args.study_name,
                "algorithms": all_study_results,
                "objectives": objectives,
                "dry_run": args.dry_run,
                "completed_at": datetime.datetime.now().isoformat(),
            }, f, indent=2)
        print(f"\n  Master summary: {master_dir}/master_summary.json")

    print("\n" + "=" * 70)
    print("  ALL STUDIES COMPLETE")
    for r in all_study_results:
        print(f"    {r['display_name']}: {r['n_completed']}/{r['n_trials']} trials — {r['study_dir']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
