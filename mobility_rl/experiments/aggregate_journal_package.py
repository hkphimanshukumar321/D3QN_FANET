"""
Aggregate tuned runs and journal study outputs into one paper-package directory.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [df for df in frames if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def find_files(root: str, pattern: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, pattern), recursive=True))


def collect_per_algo_artifacts(pipeline_root: str) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    manifests: list[dict] = []
    for path in sorted(glob.glob(os.path.join(pipeline_root, "final_artifacts_*.json"))):
        data = read_json(path)
        trial = data.get("selected_trial", {})
        artifacts = data.get("artifacts", {})
        algo = trial.get("algorithm") or Path(path).stem.replace("final_artifacts_", "")
        reward = trial.get("reward_weights", {})
        hyper = trial.get("hyperparameters", {})
        rows.append(
            {
                "Algo": algo,
                "Trial_Number": trial.get("trial_number"),
                "Study_Run_Dir": trial.get("unified_run_dir"),
                "Checkpoint_Dir": artifacts.get("checkpoint_dir"),
                "Out_Dir": artifacts.get("out_dir"),
                "wT": reward.get("wT"),
                "wD": reward.get("wD"),
                "wDrop": reward.get("wDrop"),
                "wCol": reward.get("wCol"),
                "lr": hyper.get("lr"),
                "batch_size": hyper.get("batch_size"),
                "hidden_dim": hyper.get("hidden_dim"),
                "epsilon_decay": hyper.get("epsilon_decay"),
                "target_update_freq": hyper.get("target_update_freq"),
                "gnn_heads": hyper.get("gnn_heads"),
            }
        )
        manifests.append({"algo": algo, "path": os.path.abspath(path), "data": data})
    return pd.DataFrame(rows), manifests


def collect_unified_eval_rows(per_algo_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_frames: list[pd.DataFrame] = []
    baseline_frames: list[pd.DataFrame] = []
    for _, row in per_algo_df.iterrows():
        algo = row["Algo"]
        out_dir = row.get("Out_Dir")
        if not isinstance(out_dir, str) or not out_dir:
            continue
        eval_csv = os.path.join(out_dir, "csv", "unified_eval_sweep.csv")
        baseline_csv = os.path.join(out_dir, "csv", "baseline_results.csv")
        if os.path.exists(eval_csv):
            df = pd.read_csv(eval_csv)
            df["Tuned_Algo"] = algo
            eval_frames.append(df)
        if os.path.exists(baseline_csv):
            df = pd.read_csv(baseline_csv)
            df["Tuned_Algo"] = algo
            baseline_frames.append(df)
    return safe_concat(eval_frames), safe_concat(baseline_frames)


def collect_training_rewards(checkpoint_dir: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return pd.DataFrame()
    for path in sorted(glob.glob(os.path.join(checkpoint_dir, "*_training_rewards.csv"))):
        df = pd.read_csv(path)
        if df.empty:
            continue
        stem = Path(path).stem.replace("_training_rewards", "")
        df["Algo"] = stem
        df["Source_File"] = os.path.abspath(path)
        rows.append(df)
    return safe_concat(rows)


def collect_study_summary_files(paths: Iterable[str], label: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["Source_File"] = os.path.abspath(path)
        df["Suite_Label"] = label
        rows.append(df)
    return safe_concat(rows)


def collect_optuna_statuses(pipeline_root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_rows = []
    exec_rows = []
    for path in sorted(glob.glob(os.path.join(pipeline_root, "target_trials_status*.json"))):
        data = read_json(path)
        data["Source_File"] = os.path.abspath(path)
        target_rows.append(data)
    for path in sorted(glob.glob(os.path.join(pipeline_root, "study_execution_summary*.json"))):
        data = read_json(path)
        data["Source_File"] = os.path.abspath(path)
        exec_rows.append(data)
    return pd.DataFrame(target_rows), pd.DataFrame(exec_rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate journal pipeline outputs into one package")
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    pipeline_root = os.path.abspath(args.pipeline_root)
    if not os.path.isdir(pipeline_root):
        raise FileNotFoundError(f"Pipeline root not found: {pipeline_root}")

    out_dir = os.path.abspath(args.out_dir or os.path.join(pipeline_root, "paper_package"))
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    package_manifest: dict[str, object] = {
        "pipeline_root": pipeline_root,
        "package_root": out_dir,
        "generated_files": {},
    }

    per_algo_df, per_algo_manifests = collect_per_algo_artifacts(pipeline_root)
    per_algo_csv = os.path.join(csv_dir, "tuned_algorithms_summary.csv")
    per_algo_df.to_csv(per_algo_csv, index=False)
    package_manifest["generated_files"]["tuned_algorithms_summary"] = per_algo_csv

    unified_eval_df, baseline_df = collect_unified_eval_rows(per_algo_df)
    unified_eval_csv = os.path.join(csv_dir, "main_unified_eval_master.csv")
    baseline_csv = os.path.join(csv_dir, "baseline_eval_master.csv")
    unified_eval_df.to_csv(unified_eval_csv, index=False)
    baseline_df.to_csv(baseline_csv, index=False)
    package_manifest["generated_files"]["main_unified_eval_master"] = unified_eval_csv
    package_manifest["generated_files"]["baseline_eval_master"] = baseline_csv

    final_artifacts_json = os.path.join(pipeline_root, "final_artifacts.json")
    combined_checkpoint_dir = ""
    if os.path.exists(final_artifacts_json):
        final_data = read_json(final_artifacts_json)
        combined_checkpoint_dir = final_data.get("artifacts", {}).get("checkpoint_dir", "")
    training_rewards_df = collect_training_rewards(combined_checkpoint_dir)
    training_rewards_csv = os.path.join(csv_dir, "training_rewards_master.csv")
    training_rewards_df.to_csv(training_rewards_csv, index=False)
    package_manifest["generated_files"]["training_rewards_master"] = training_rewards_csv

    generalization_df = collect_study_summary_files(
        find_files(pipeline_root, "journal_suites/generalization/**/csv/generalization_suite_summary.csv"),
        "generalization",
    )
    robustness_df = collect_study_summary_files(
        find_files(pipeline_root, "journal_suites/robustness/**/csv/robustness_suite_summary.csv"),
        "robustness",
    )
    failure_df = collect_study_summary_files(
        find_files(pipeline_root, "journal_suites/failure_recovery/**/csv/failure_recovery_suite_summary.csv"),
        "failure_recovery",
    )
    complexity_df = collect_study_summary_files(
        find_files(pipeline_root, "journal_suites/complexity/**/csv/complexity_benchmark.csv"),
        "complexity",
    )

    for name, df in [
        ("generalization_master.csv", generalization_df),
        ("robustness_master.csv", robustness_df),
        ("failure_recovery_master.csv", failure_df),
        ("complexity_master.csv", complexity_df),
    ]:
        path = os.path.join(csv_dir, name)
        df.to_csv(path, index=False)
        package_manifest["generated_files"][name.replace(".csv", "")] = path

    target_df, exec_df = collect_optuna_statuses(pipeline_root)
    target_csv = os.path.join(csv_dir, "optuna_target_status_master.csv")
    exec_csv = os.path.join(csv_dir, "optuna_execution_master.csv")
    target_df.to_csv(target_csv, index=False)
    exec_df.to_csv(exec_csv, index=False)
    package_manifest["generated_files"]["optuna_target_status_master"] = target_csv
    package_manifest["generated_files"]["optuna_execution_master"] = exec_csv

    manifest_json = os.path.join(out_dir, "package_manifest.json")
    package_manifest["per_algo_artifacts"] = per_algo_manifests
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(package_manifest, f, indent=2)

    print(out_dir)


if __name__ == "__main__":
    main()
