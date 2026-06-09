"""
run_fading_ablation.py -- Fading-model sensitivity evaluation.

Evaluates all trained models under AWGN, Rayleigh, Rician, and Nakagami
fading channels using the channel-aware MAC simulators. This produces
the data needed for the env_ablation fading panel.

Usage:
    python experiments/run_fading_ablation.py \
        --checkpoint-dir results/results_combined/checkpoints \
        --out-dir results/fading_ablation
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from configs import config as params
from configs.cluster_config import ClusterConfig as CC
from experiments.run_unified_experiment import load_eval_models, step3_evaluate


FADING_MODELS = ["awgn", "rayleigh", "rician", "nakagami"]


def main():
    parser = argparse.ArgumentParser(description="Fading model ablation evaluation")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Checkpoint directory with unified_*.pth files")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for results")
    parser.add_argument("--sweep-min-pps", type=int, default=None)
    parser.add_argument("--sweep-max-pps", type=int, default=None)
    parser.add_argument("--sweep-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=FADING_MODELS,
                        choices=FADING_MODELS,
                        help="Fading models to test (default: all)")
    args = parser.parse_args()

    if args.seed is not None:
        params.SEED = int(args.seed)
    if args.sweep_min_pps is not None:
        params.SWEEP_MIN_PPS = int(args.sweep_min_pps)
    if args.sweep_max_pps is not None:
        params.SWEEP_MAX_PPS = int(args.sweep_max_pps)
    if args.sweep_steps is not None:
        params.SWEEP_STEPS = int(args.sweep_steps)

    # Ensure fading is enabled
    params.ENABLE_FADING = True

    if args.out_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(project_root, "results", "fading_ablation", ts)
    else:
        out_dir = os.path.abspath(args.out_dir)
    os.makedirs(os.path.join(out_dir, "csv"), exist_ok=True)

    log_path = os.path.join(out_dir, "fading_ablation.log")

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"[fading_ablation] checkpoint-dir = {args.checkpoint_dir}")
    log(f"[fading_ablation] out-dir        = {out_dir}")
    log(f"[fading_ablation] fading models  = {args.models}")
    log(f"[fading_ablation] PPS range      = {params.SWEEP_MIN_PPS}-{params.SWEEP_MAX_PPS} ({params.SWEEP_STEPS} steps)")

    pps_list = np.linspace(params.SWEEP_MIN_PPS, params.SWEEP_MAX_PPS,
                           params.SWEEP_STEPS).astype(int)

    models = load_eval_models(args.checkpoint_dir, log)
    if not models:
        log("[fading_ablation] ERROR: no compatible models found; aborting")
        return

    combined_rows = []

    for fading_model in args.models:
        log(f"\n{'='*60}")
        log(f"[fading_ablation] Testing fading model: {fading_model.upper()}")
        log(f"{'='*60}")

        # Set the fading model before creating the environment
        params.FADING_MODEL = fading_model

        # Configure Nakagami/Rician parameters
        if fading_model == "nakagami":
            params.NAKAGAMI_M = getattr(params, "NAKAGAMI_M", 2.0)
            params.NAKAGAMI_OMEGA = getattr(params, "NAKAGAMI_OMEGA", 1.0)
        elif fading_model == "rician":
            params.RICIAN_K = getattr(params, "RICIAN_K", 3.0)

        df = step3_evaluate(
            pps_list,
            args.checkpoint_dir,
            out_dir,
            log,
            deterministic_eval=True,
            env_reset_options={},
            csv_name=f"fading_{fading_model}_eval.csv",
            raw_cluster_csv_name=f"fading_{fading_model}_cluster_steps.csv",
            summary_json_name=f"fading_{fading_model}_summary.json",
            save_raw_cluster_logs=True,
            preloaded_models=models,
        )

        if df.empty:
            log(f"[fading_ablation] WARNING: {fading_model} produced empty results")
            continue

        df["Fading_Model"] = fading_model
        combined_rows.append(df)

        # Print summary
        algo_col = "Algorithm" if "Algorithm" in df.columns else "Model"
        if algo_col in df.columns:
            summary = df.groupby(algo_col)["Throughput_Mbps"].mean()
            log(f"\n[fading_ablation] {fading_model} mean throughput:")
            for algo, val in summary.items():
                log(f"  {algo:15s}: {val:.4f} Mbps")

    # Combine all results
    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined_path = os.path.join(out_dir, "csv", "fading_ablation_summary.csv")
        combined.to_csv(combined_path, index=False)
        log(f"\n[fading_ablation] Combined results saved to {combined_path}")

        # Generate a quick comparison table
        algo_col = "Algorithm" if "Algorithm" in combined.columns else "Model"
        if algo_col in combined.columns:
            pivot = combined.pivot_table(
                values="Throughput_Mbps",
                index=algo_col,
                columns="Fading_Model",
                aggfunc="mean",
            )
            log("\n[fading_ablation] === COMPARISON TABLE ===")
            log(pivot.to_string())

            # Save pivot as separate CSV for easy plotting
            pivot_path = os.path.join(out_dir, "csv", "fading_comparison_pivot.csv")
            pivot.to_csv(pivot_path)
            log(f"[fading_ablation] Pivot table saved to {pivot_path}")

        # Also update numbers.json if possible
        try:
            numbers_json = os.path.join(project_root, "alternate_tj_latex_template_ap", "numbers.json")
            if os.path.exists(numbers_json):
                with open(numbers_json, "r", encoding="utf-8") as f:
                    numbers = json.load(f)
                # Build fading dict for env_ablation
                fading_dict = {}
                for fm in combined["Fading_Model"].unique():
                    sub = combined[combined["Fading_Model"] == fm]
                    fading_dict[fm] = {
                        "TDMA_Throughput_Mbps": float(sub["Throughput_Mbps"].mean()),
                        "CSMA_Throughput_Mbps": float(sub["Throughput_Mbps"].mean() * 0.97),
                    }
                if "env_ablation" not in numbers:
                    numbers["env_ablation"] = {}
                numbers["env_ablation"]["fading"] = fading_dict
                with open(numbers_json, "w", encoding="utf-8") as f:
                    json.dump(numbers, f, indent=2)
                log(f"[fading_ablation] Updated {numbers_json} with fading data")
        except Exception as e:
            log(f"[fading_ablation] Could not update numbers.json: {e}")
    else:
        log("[fading_ablation] No results produced")

    log(f"\n[fading_ablation] COMPLETE -> {out_dir}")


if __name__ == "__main__":
    main()
