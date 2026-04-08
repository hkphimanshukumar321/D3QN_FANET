"""
Imperfect-information robustness evaluation harness.
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
from experiments.run_unified_experiment import load_eval_models, step3_evaluate


def build_scenarios():
    return [
        {"study_block": "stale_obs", "scenario": "stale_1", "env_options": {"obs_staleness_steps": 1}},
        {"study_block": "stale_obs", "scenario": "stale_3", "env_options": {"obs_staleness_steps": 3}},
        {"study_block": "noisy_obs", "scenario": "noise_0p02", "env_options": {"obs_noise_std": 0.02}},
        {"study_block": "noisy_obs", "scenario": "noise_0p05", "env_options": {"obs_noise_std": 0.05}},
        {"study_block": "graph_corruption", "scenario": "missing_0p10", "env_options": {"graph_missing_edge_prob": 0.10}},
        {"study_block": "graph_corruption", "scenario": "missing_0p25", "env_options": {"graph_missing_edge_prob": 0.25}},
        {"study_block": "graph_corruption", "scenario": "false_0p10", "env_options": {"graph_false_edge_prob": 0.10}},
        {"study_block": "graph_corruption", "scenario": "false_0p25", "env_options": {"graph_false_edge_prob": 0.25}},
        {"study_block": "graph_corruption", "scenario": "static_graph", "env_options": {"graph_mode": "static"}},
        {"study_block": "graph_corruption", "scenario": "shuffled_graph", "env_options": {"graph_mode": "shuffled"}},
        {
            "study_block": "handover_info",
            "scenario": "handover_stale_2",
            "env_options": {
                "handover_info_staleness_steps": 2,
                "failure_schedule": [{"step": 20, "target": "random"}],
            },
        },
        {
            "study_block": "handover_info",
            "scenario": "handover_stale_4",
            "env_options": {
                "handover_info_staleness_steps": 4,
                "failure_schedule": [{"step": 20, "target": "max_backlog"}],
            },
        },
    ]


def filter_scenarios(scenarios, scenario_names=None, study_blocks=None):
    wanted_scenarios = set(scenario_names or [])
    wanted_blocks = set(study_blocks or [])
    filtered = []
    for scenario in scenarios:
        if wanted_scenarios and scenario["scenario"] not in wanted_scenarios:
            continue
        if wanted_blocks and scenario["study_block"] not in wanted_blocks:
            continue
        filtered.append(scenario)
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation suite")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory created by run_unified_experiment")
    parser.add_argument("--out-dir", default=None, help="Output directory for robustness CSVs")
    parser.add_argument("--sweep-min-pps", type=int, default=None)
    parser.add_argument("--sweep-max-pps", type=int, default=None)
    parser.add_argument("--sweep-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scenario", action="append", default=[], help="Run only the named scenario(s)")
    parser.add_argument("--study-block", action="append", default=[], help="Run only the named study block(s)")
    args = parser.parse_args()

    if args.seed is not None:
        params.SEED = int(args.seed)
    if args.sweep_min_pps is not None:
        params.SWEEP_MIN_PPS = int(args.sweep_min_pps)
    if args.sweep_max_pps is not None:
        params.SWEEP_MAX_PPS = int(args.sweep_max_pps)
    if args.sweep_steps is not None:
        params.SWEEP_STEPS = int(args.sweep_steps)

    if args.out_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(project_root, "results", "journal_robustness", ts)
    else:
        out_dir = os.path.abspath(args.out_dir)
    os.makedirs(os.path.join(out_dir, "csv"), exist_ok=True)

    log_path = os.path.join(out_dir, "robustness.log")

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    pps_list = np.linspace(params.SWEEP_MIN_PPS, params.SWEEP_MAX_PPS, params.SWEEP_STEPS).astype(int)
    combined_rows = []
    manifest = []
    scenarios = filter_scenarios(build_scenarios(), args.scenario, args.study_block)
    models = load_eval_models(args.checkpoint_dir, log)
    if not models:
        log("[robustness] no compatible models found; aborting")
        return

    for scenario_cfg in scenarios:
        study_block = scenario_cfg["study_block"]
        scenario = scenario_cfg["scenario"]
        log(f"[robustness] {study_block} :: {scenario}")
        df = step3_evaluate(
            pps_list,
            args.checkpoint_dir,
            out_dir,
            log,
            deterministic_eval=True,
            env_reset_options=scenario_cfg["env_options"],
            csv_name=f"{study_block}_{scenario}_eval.csv",
            raw_cluster_csv_name=f"{study_block}_{scenario}_cluster_steps.csv",
            summary_json_name=f"{study_block}_{scenario}_summary.json",
            save_raw_cluster_logs=True,
            preloaded_models=models,
        )
        if df.empty:
            continue
        df["Study_Block"] = study_block
        df["Scenario"] = scenario
        combined_rows.append(df)
        manifest.append({
            "study_block": study_block,
            "scenario": scenario,
            "env_options": scenario_cfg["env_options"],
        })

    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined.to_csv(os.path.join(out_dir, "csv", "robustness_suite_summary.csv"), index=False)
    else:
        combined = pd.DataFrame()

    with open(os.path.join(out_dir, "csv", "robustness_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log(f"[robustness] complete -> {out_dir}")


if __name__ == "__main__":
    main()
