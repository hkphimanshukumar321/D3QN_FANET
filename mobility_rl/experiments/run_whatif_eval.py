#!/usr/bin/env python3
"""
What-If Evaluation Runner
=========================
Load pre-trained unified checkpoints and evaluate under modified config
scenarios (topology, fading, node count, traffic, robustness knobs, etc.)
WITHOUT retraining.

Usage examples:

  # 1) Default evaluation with combined checkpoints
  python experiments/run_whatif_eval.py

  # 2) Change node count to 100
  python experiments/run_whatif_eval.py --nodes 100

  # 3) Change fading model to Rayleigh
  python experiments/run_whatif_eval.py --fading rayleigh

  # 4) Test compact-dense topology
  python experiments/run_whatif_eval.py --topology compact_dense

  # 5) Add observation noise
  python experiments/run_whatif_eval.py --obs-noise-std 0.1

  # 6) Add graph edge corruption
  python experiments/run_whatif_eval.py --graph-missing-edge-prob 0.2

  # 7) Custom checkpoint directory
  python experiments/run_whatif_eval.py --checkpoint-dir results/results_combined/checkpoints

  # 8) Multiple overrides at once
  python experiments/run_whatif_eval.py --nodes 100 --fading rayleigh --topology compact_dense --sweep-steps 10

  # 9) Dry run (3 load points only)
  python experiments/run_whatif_eval.py --dry-run

  # 10) Only evaluate specific algorithms
  python experiments/run_whatif_eval.py --algos magat_d3qn mappo qmix
"""

import os
import sys
import argparse
import datetime
import json
import numpy as np
import pandas as pd

# Project path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Default checkpoint location
DEFAULT_CP_DIR = os.path.join(
    project_root, "results", "results_combined", "checkpoints"
)


def apply_config_overrides(args):
    """Monkey-patch configs.config with CLI overrides."""
    from configs import config as params

    if args.nodes is not None:
        params.N = args.nodes
        print(f"  Override: N = {params.N}")

    if args.fading is not None:
        params.FADING_MODEL = args.fading.lower()
        print(f"  Override: FADING_MODEL = {params.FADING_MODEL}")

    if args.topology is not None:
        params.TOPOLOGY_PRESET = args.topology
        print(f"  Override: TOPOLOGY_PRESET = {params.TOPOLOGY_PRESET}")

    if args.traffic_profile is not None:
        params.TRAFFIC_PROFILE = args.traffic_profile
        print(f"  Override: TRAFFIC_PROFILE = {params.TRAFFIC_PROFILE}")

    if args.graph_mode is not None:
        params.GRAPH_MODE = args.graph_mode
        print(f"  Override: GRAPH_MODE = {params.GRAPH_MODE}")

    if args.obs_noise_std is not None:
        params.OBS_NOISE_STD = args.obs_noise_std
        print(f"  Override: OBS_NOISE_STD = {params.OBS_NOISE_STD}")

    if args.obs_staleness is not None:
        params.OBS_STALENESS_STEPS = args.obs_staleness
        print(f"  Override: OBS_STALENESS_STEPS = {params.OBS_STALENESS_STEPS}")

    if args.graph_missing_edge_prob is not None:
        params.GRAPH_MISSING_EDGE_PROB = args.graph_missing_edge_prob
        print(f"  Override: GRAPH_MISSING_EDGE_PROB = {params.GRAPH_MISSING_EDGE_PROB}")

    if args.graph_false_edge_prob is not None:
        params.GRAPH_FALSE_EDGE_PROB = args.graph_false_edge_prob
        print(f"  Override: GRAPH_FALSE_EDGE_PROB = {params.GRAPH_FALSE_EDGE_PROB}")

    if args.graph_staleness is not None:
        params.GRAPH_STALENESS_STEPS = args.graph_staleness
        print(f"  Override: GRAPH_STALENESS_STEPS = {params.GRAPH_STALENESS_STEPS}")

    if args.handover_staleness is not None:
        params.HANDOVER_INFO_STALENESS_STEPS = args.handover_staleness
        print(f"  Override: HANDOVER_INFO_STALENESS_STEPS = {params.HANDOVER_INFO_STALENESS_STEPS}")

    if args.speed_scale is not None:
        params.SPEED_SCALE = args.speed_scale
        print(f"  Override: SPEED_SCALE = {params.SPEED_SCALE}")

    if args.interference_scale is not None:
        params.INTERFERENCE_SCALE = args.interference_scale
        print(f"  Override: INTERFERENCE_SCALE = {params.INTERFERENCE_SCALE}")

    if args.rts_cts is not None:
        params.RTS_CTS_ENABLED = args.rts_cts.lower() == "on"
        print(f"  Override: RTS_CTS_ENABLED = {params.RTS_CTS_ENABLED}")

    if args.sweep_min_pps is not None:
        params.SWEEP_MIN_PPS = args.sweep_min_pps
        print(f"  Override: SWEEP_MIN_PPS = {params.SWEEP_MIN_PPS}")

    if args.sweep_max_pps is not None:
        params.SWEEP_MAX_PPS = args.sweep_max_pps
        print(f"  Override: SWEEP_MAX_PPS = {params.SWEEP_MAX_PPS}")

    if args.sweep_steps is not None:
        params.SWEEP_STEPS = args.sweep_steps
        print(f"  Override: SWEEP_STEPS = {params.SWEEP_STEPS}")

    if args.sim_time is not None:
        params.SIM_TIME_S = args.sim_time
        print(f"  Override: SIM_TIME_S = {params.SIM_TIME_S}")

    # Disable algorithms not requested
    if args.algos:
        algo_set = set(a.lower() for a in args.algos)
        params.RUN_MARL_IQL = "iql" in algo_set
        params.RUN_MARL_VDN = "vdn" in algo_set
        params.RUN_MARL_QMIX = "qmix" in algo_set
        params.RUN_MARL_MAPPO = "mappo" in algo_set
        params.RUN_MARL_GNN = "magat_d3qn" in algo_set or "magat" in algo_set
        # Disable all SARL for checkpoint eval
        params.RUN_TABULAR_QLEARNING = "tabular" in algo_set
        params.RUN_DQN = "dqn" in algo_set
        params.RUN_PPO = "ppo" in algo_set
        params.RUN_A2C = "a2c" in algo_set
        params.RUN_CUSTOM_RL = "mca_d3qn" in algo_set or "mca" in algo_set
        print(f"  Override: algos = {args.algos}")

    return params


def make_output_dir(tag="whatif"):
    """Create a timestamped output directory."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(project_root, "results", "whatif_evals", f"{tag}_{ts}")
    for sub in ["csv", "images", "logs"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    return out_dir


def run_eval(cp_dir, out_dir, params, dry_run=False):
    """Run the full evaluation sweep using pre-trained models."""
    from experiments.run_unified_experiment import (
        load_eval_models,
        step3_evaluate,
        step1_baseline_marl,
    )
    from experiments.run_experiments import generate_baseline_plots, generate_aggregated_rl_plots

    def log(msg):
        print(msg)
        log_path = os.path.join(out_dir, "logs", "whatif_eval.log")
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log("=" * 60)
    log("  WHAT-IF EVALUATION")
    log("=" * 60)
    log(f"  Checkpoint dir: {cp_dir}")
    log(f"  Output dir:     {out_dir}")
    log(f"  Config: N={params.N}, Fading={params.FADING_MODEL}, "
        f"Topology={params.TOPOLOGY_PRESET}, Traffic={params.TRAFFIC_PROFILE}")
    log(f"  Robustness: obs_noise={params.OBS_NOISE_STD}, "
        f"graph_mode={params.GRAPH_MODE}, "
        f"missing_edge={params.GRAPH_MISSING_EDGE_PROB}, "
        f"false_edge={params.GRAPH_FALSE_EDGE_PROB}")

    # Determine sweep
    if dry_run:
        sweep_steps = 3
        log("  *** DRY RUN — 3 load points only ***")
    else:
        sweep_steps = params.SWEEP_STEPS

    pps_list = np.linspace(
        params.SWEEP_MIN_PPS, params.SWEEP_MAX_PPS, sweep_steps
    ).astype(int)
    log(f"  Load sweep: {pps_list[0]}..{pps_list[-1]} pps ({len(pps_list)} points)")

    # Step 1: Baselines
    log("\n--- Step 1: TDMA/CSMA Baselines ---")
    baseline_df = step1_baseline_marl(pps_list, params.SEED, log)
    baseline_csv = os.path.join(out_dir, "csv", "baseline_results.csv")
    baseline_df.to_csv(baseline_csv, index=False)
    log(f"  Saved baselines: {baseline_csv}")

    try:
        generate_baseline_plots(
            baseline_df, out_dir,
            params.N, params.PAYLOAD_BYTES, params.PHY_RATE_BPS,
            params.QMAX, params.RTS_CTS_ENABLED, params.ACK_ENABLED,
        )
        log("  Saved baseline plots.")
    except Exception as e:
        log(f"  Baseline plots failed (non-fatal): {e}")

    # Step 2: Load pre-trained models
    log("\n--- Step 2: Loading pre-trained checkpoints ---")
    models = load_eval_models(cp_dir, log)
    if not models:
        log("ERROR: No models found. Check --checkpoint-dir path.")
        log(f"  Expected files in: {cp_dir}")
        for f in ["unified_gnn_marl_model.pth", "unified_iql_model.pth",
                   "unified_mappo_model.pth", "unified_qmix_model.pth",
                   "unified_vdn_model.pth"]:
            p = os.path.join(cp_dir, f)
            log(f"    {f}: {'EXISTS' if os.path.exists(p) else 'MISSING'}")
        return None

    log(f"  Loaded {len(models)} models: {list(models.keys())}")

    # Step 3: Evaluation sweep
    log("\n--- Step 3: Evaluation Sweep ---")
    eval_df = step3_evaluate(
        pps_list, cp_dir, out_dir, log,
        deterministic_eval=True,
        preloaded_models=models,
        csv_name="whatif_eval_sweep.csv",
    )

    if eval_df is None or eval_df.empty:
        log("WARNING: Evaluation returned empty results.")
        return None

    eval_csv = os.path.join(out_dir, "csv", "whatif_eval_sweep.csv")
    log(f"\n  Evaluation complete. {len(eval_df)} rows saved to: {eval_csv}")

    # Step 4: Generate comparison plots
    log("\n--- Step 4: Generating plots ---")
    try:
        _generate_whatif_plots(eval_df, baseline_df, out_dir)
        log("  Saved comparison plots.")
    except Exception as e:
        log(f"  Plot generation failed (non-fatal): {e}")
        import traceback
        traceback.print_exc()

    # Step 5: Save config snapshot
    config_snapshot = {
        "N": params.N,
        "FADING_MODEL": params.FADING_MODEL,
        "TOPOLOGY_PRESET": params.TOPOLOGY_PRESET,
        "TRAFFIC_PROFILE": params.TRAFFIC_PROFILE,
        "GRAPH_MODE": params.GRAPH_MODE,
        "OBS_NOISE_STD": params.OBS_NOISE_STD,
        "OBS_STALENESS_STEPS": params.OBS_STALENESS_STEPS,
        "GRAPH_MISSING_EDGE_PROB": params.GRAPH_MISSING_EDGE_PROB,
        "GRAPH_FALSE_EDGE_PROB": params.GRAPH_FALSE_EDGE_PROB,
        "SPEED_SCALE": params.SPEED_SCALE,
        "INTERFERENCE_SCALE": params.INTERFERENCE_SCALE,
        "RTS_CTS_ENABLED": params.RTS_CTS_ENABLED,
        "SWEEP_MIN_PPS": params.SWEEP_MIN_PPS,
        "SWEEP_MAX_PPS": params.SWEEP_MAX_PPS,
        "SWEEP_STEPS": params.SWEEP_STEPS,
        "checkpoint_dir": cp_dir,
        "models_loaded": list(models.keys()),
    }
    config_path = os.path.join(out_dir, "csv", "whatif_config.json")
    with open(config_path, "w") as f:
        json.dump(config_snapshot, f, indent=2, default=str)
    log(f"  Config snapshot: {config_path}")

    log("\n" + "=" * 60)
    log(f"  WHAT-IF EVAL COMPLETE — Results: {out_dir}")
    log("=" * 60)
    return eval_df


def _generate_whatif_plots(eval_df, baseline_df, out_dir):
    """Generate throughput/delay/drops/collisions comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    algos = eval_df["Algorithm"].unique()
    markers = ["D", "s", "^", "v", "o", "P", "*", "X"]
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(algos), 1)))

    metrics = [
        ("Throughput_Mbps", "Throughput (Mbps)", "throughput"),
        ("Delay_ms", "Delay (ms)", "delay"),
        ("Drops", "Packet Drops", "drops"),
        ("Collisions", "Collisions", "collisions"),
    ]

    for col, ylabel, fname in metrics:
        if col not in eval_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        # Baselines
        load = baseline_df["Offered_Load_pps"]
        for proto, style in [("TDMA", "--"), ("CSMA", "--")]:
            bcol = f"{proto}_{'Throughput_Mbps' if col == 'Throughput_Mbps' else 'Delay_s' if col == 'Delay_ms' else 'Drops' if col == 'Drops' else 'Collisions'}"
            if bcol in baseline_df.columns:
                data = baseline_df[bcol] * (1000 if "Delay_s" in bcol else 1)
                ax.plot(load, data, linestyle=style, linewidth=1.5,
                        alpha=0.6, label=f"{proto} (baseline)")

        # RL models
        for idx, algo in enumerate(sorted(algos)):
            adf = eval_df[eval_df["Algorithm"] == algo].sort_values("Offered_Load_pps")
            ax.plot(adf["Offered_Load_pps"], adf[col],
                    marker=markers[idx % len(markers)],
                    color=colors[idx],
                    label=algo, linewidth=2, markersize=6, markevery=2)

        ax.set_xlabel("Offered Load (pps)", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(f"What-If Evaluation: {ylabel} vs Offered Load", fontsize=13)
        ax.legend(fontsize=9, frameon=True)
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        fig.savefig(os.path.join(img_dir, f"whatif_{fname}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="What-If Evaluation: test trained models under modified configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Default eval with combined checkpoints
  python experiments/run_whatif_eval.py

  # Test with 100 nodes + Rayleigh fading
  python experiments/run_whatif_eval.py --nodes 100 --fading rayleigh

  # Test compact topology with noisy observations
  python experiments/run_whatif_eval.py --topology compact_dense --obs-noise-std 0.1

  # Only evaluate MAGAT and MAPPO
  python experiments/run_whatif_eval.py --algos magat_d3qn mappo

  # Quick sanity check
  python experiments/run_whatif_eval.py --dry-run
        """,
    )

    # Core
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CP_DIR,
                        help=f"Path to checkpoint directory (default: {DEFAULT_CP_DIR})")
    parser.add_argument("--tag", type=str, default="whatif",
                        help="Tag for the output directory name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test with 3 load points only")

    # Algorithm selection
    parser.add_argument("--algos", nargs="+", type=str, default=None,
                        help="Algorithms to evaluate (e.g., magat_d3qn mappo qmix iql vdn)")

    # Environment overrides
    parser.add_argument("--nodes", type=int, default=None, help="Override N (number of nodes)")
    parser.add_argument("--fading", type=str, default=None,
                        choices=["awgn", "rayleigh", "rician", "nakagami"],
                        help="Override fading model")
    parser.add_argument("--topology", type=str, default=None,
                        choices=["default", "compact_dense", "sparse_separated", "asymmetric_hotspot"],
                        help="Override topology preset")
    parser.add_argument("--traffic-profile", type=str, default=None,
                        choices=["smooth", "bursty_on_off", "heavy_tail"],
                        help="Override traffic profile")
    parser.add_argument("--graph-mode", type=str, default=None,
                        choices=["dynamic", "none", "static", "shuffled"],
                        help="Override graph mode")
    parser.add_argument("--rts-cts", type=str, default=None,
                        choices=["on", "off"], help="Override RTS/CTS")
    parser.add_argument("--sim-time", type=float, default=None,
                        help="Override simulation time (seconds)")

    # Robustness knobs
    parser.add_argument("--obs-noise-std", type=float, default=None,
                        help="Observation noise std (0.0 = clean)")
    parser.add_argument("--obs-staleness", type=int, default=None,
                        help="Observation staleness steps")
    parser.add_argument("--graph-missing-edge-prob", type=float, default=None,
                        help="Probability of missing edges (0.0-1.0)")
    parser.add_argument("--graph-false-edge-prob", type=float, default=None,
                        help="Probability of false edges (0.0-1.0)")
    parser.add_argument("--graph-staleness", type=int, default=None,
                        help="Graph staleness steps")
    parser.add_argument("--handover-staleness", type=int, default=None,
                        help="Handover info staleness steps")
    parser.add_argument("--speed-scale", type=float, default=None,
                        help="Speed scale factor (1.0 = nominal)")
    parser.add_argument("--interference-scale", type=float, default=None,
                        help="Interference scale factor (1.0 = nominal)")

    # Sweep overrides
    parser.add_argument("--sweep-min-pps", type=int, default=None,
                        help="Override minimum offered load")
    parser.add_argument("--sweep-max-pps", type=int, default=None,
                        help="Override maximum offered load")
    parser.add_argument("--sweep-steps", type=int, default=None,
                        help="Override number of sweep load points")

    args = parser.parse_args()

    # Apply overrides
    print("\n" + "=" * 60)
    print("  CONFIG OVERRIDES")
    print("=" * 60)
    params = apply_config_overrides(args)

    # Create output dir
    out_dir = make_output_dir(tag=args.tag)

    # Run evaluation
    run_eval(args.checkpoint_dir, out_dir, params, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
