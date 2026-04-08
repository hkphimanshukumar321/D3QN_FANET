"""
Generalization and regime-mismatch evaluation harness.

This script reuses the already trained decentralized/centralized models and
evaluates them under controlled test-time mismatches without changing the core
cluster-head architecture.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from configs import config as params
from experiments.run_unified_experiment import load_eval_models, step3_evaluate


@contextmanager
def temporary_params(**overrides):
    snapshot = {key: getattr(params, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(params, key, value)
        yield
    finally:
        for key, value in snapshot.items():
            setattr(params, key, value)


def build_scenarios(base_nodes: int):
    smaller_nodes = max(20, int(round(0.6 * base_nodes)))
    larger_nodes = max(base_nodes + 10, int(round(1.4 * base_nodes)))
    return [
        {
            "study_block": "node_count",
            "scenario": f"n_{smaller_nodes}",
            "param_overrides": {"N": smaller_nodes},
            "env_options": {},
        },
        {
            "study_block": "node_count",
            "scenario": f"n_{base_nodes}",
            "param_overrides": {"N": base_nodes},
            "env_options": {},
        },
        {
            "study_block": "node_count",
            "scenario": f"n_{larger_nodes}",
            "param_overrides": {"N": larger_nodes},
            "env_options": {},
        },
        {
            "study_block": "mobility_regime",
            "scenario": "slow_random_walk",
            "param_overrides": {},
            "env_options": {"mobility_model": "random_walk", "speed_scale": 0.6},
        },
        {
            "study_block": "mobility_regime",
            "scenario": "fast_random_walk",
            "param_overrides": {},
            "env_options": {"mobility_model": "random_walk", "speed_scale": 1.6},
        },
        {
            "study_block": "mobility_regime",
            "scenario": "gauss_markov",
            "param_overrides": {},
            "env_options": {"mobility_model": "gauss_markov", "speed_scale": 1.0},
        },
        {
            "study_block": "topology_scale",
            "scenario": "compact_dense",
            "param_overrides": {},
            "env_options": {"topology_preset": "compact_dense"},
        },
        {
            "study_block": "topology_scale",
            "scenario": "sparse_separated",
            "param_overrides": {},
            "env_options": {"topology_preset": "sparse_separated"},
        },
        {
            "study_block": "topology_scale",
            "scenario": "asymmetric_hotspot",
            "param_overrides": {},
            "env_options": {"topology_preset": "asymmetric_hotspot"},
        },
        {
            "study_block": "interference_regime",
            "scenario": "lighter_interference",
            "param_overrides": {},
            "env_options": {"interference_scale": 0.8, "coordination_capacity_scale": 1.1},
        },
        {
            "study_block": "interference_regime",
            "scenario": "heavier_interference",
            "param_overrides": {},
            "env_options": {"interference_scale": 1.25, "coordination_capacity_scale": 0.8},
        },
        {
            "study_block": "traffic_regime",
            "scenario": "smooth",
            "param_overrides": {},
            "env_options": {"traffic_profile": "smooth"},
        },
        {
            "study_block": "traffic_regime",
            "scenario": "bursty_on_off",
            "param_overrides": {},
            "env_options": {"traffic_profile": "bursty_on_off"},
        },
        {
            "study_block": "traffic_regime",
            "scenario": "heavy_tail",
            "param_overrides": {},
            "env_options": {"traffic_profile": "heavy_tail"},
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
    parser = argparse.ArgumentParser(description="Generalization / mismatch evaluation suite")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory created by run_unified_experiment")
    parser.add_argument("--out-dir", default=None, help="Output directory for study CSVs")
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
        out_dir = os.path.join(project_root, "results", "journal_generalization", ts)
    else:
        out_dir = os.path.abspath(args.out_dir)
    os.makedirs(os.path.join(out_dir, "csv"), exist_ok=True)

    log_path = os.path.join(out_dir, "generalization.log")

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    pps_list = np.linspace(params.SWEEP_MIN_PPS, params.SWEEP_MAX_PPS, params.SWEEP_STEPS).astype(int)
    combined_rows = []
    manifest = []
    scenarios = filter_scenarios(build_scenarios(params.N), args.scenario, args.study_block)
    models = load_eval_models(args.checkpoint_dir, log)
    if not models:
        log("[generalization] no compatible models found; aborting")
        return

    for scenario_cfg in scenarios:
        study_block = scenario_cfg["study_block"]
        scenario = scenario_cfg["scenario"]
        log(f"[generalization] {study_block} :: {scenario}")
        with temporary_params(**scenario_cfg["param_overrides"]):
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
            "param_overrides": scenario_cfg["param_overrides"],
        })

    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined.to_csv(os.path.join(out_dir, "csv", "generalization_suite_summary.csv"), index=False)
    else:
        combined = pd.DataFrame()

    with open(os.path.join(out_dir, "csv", "generalization_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log(f"[generalization] complete -> {out_dir}")


if __name__ == "__main__":
    main()
