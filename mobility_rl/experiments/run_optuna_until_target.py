"""
Run an Optuna study until a target number of COMPLETE trials exists.

This prevents redundant reruns when a pipeline is resumed after partial
completion or downstream failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager

import optuna

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from experiments.run_pareto_optuna import DEFAULT_OBJECTIVES, OBJECTIVE_DIRECTIONS, resolve_trial_count, save_study_artifacts


@contextmanager
def file_lock(lock_path: str, timeout_s: int = 3600, poll_s: float = 2.0):
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n{int(time.time())}\n".encode("utf-8"))
            os.close(fd)
            break
        except FileExistsError:
            stale = False
            try:
                with open(lock_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines()]
                lock_pid = int(lines[0]) if lines else -1
                ts = int(lines[1]) if len(lines) > 1 else 0
                try:
                    os.kill(lock_pid, 0)
                except OSError:
                    stale = True
                if time.time() - ts > timeout_s:
                    stale = True
            except Exception:
                stale = True
            if stale:
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
                continue
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(poll_s)
    try:
        yield
    finally:
        try:
            os.remove(lock_path)
        except OSError:
            pass


def load_or_create_study(
    study_name: str,
    storage: str,
    directions: list[str],
    objectives: list[str],
) -> optuna.Study:
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        saved_objectives = list(study.user_attrs.get("objectives", []))
        saved_directions = list(study.user_attrs.get("directions", []))
        if saved_objectives and saved_objectives != objectives:
            raise ValueError(
                f"Existing study objectives {saved_objectives} do not match requested {objectives}."
            )
        if saved_directions and saved_directions != directions:
            raise ValueError(
                f"Existing study directions {saved_directions} do not match requested {directions}."
            )
    except KeyError:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            directions=directions,
        )
        study.set_user_attr("objectives", objectives)
        study.set_user_attr("directions", directions)
    return study


def count_complete_trials(
    study_name: str,
    storage: str,
    directions: list[str],
    objectives: list[str],
) -> int:
    study = load_or_create_study(study_name, storage, directions, objectives)
    return sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)


def _ensure_study_artifacts(study_name, storage, directions, objectives, study_dir, algo_cli):
    """Regenerate study-level artifacts if missing (e.g. after a crash)."""
    reps_csv = os.path.join(study_dir, "representative_pareto_points.csv")
    if os.path.exists(reps_csv):
        return  # Already present, nothing to do
    print(f"[run_optuna_until_target] Regenerating missing study artifacts for {study_name}...")
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        save_study_artifacts(study, study_dir, objectives, algo_cli)
        print(f"[run_optuna_until_target] Study artifacts regenerated successfully.")
    except Exception as e:
        print(f"[run_optuna_until_target] WARNING: Could not regenerate artifacts: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run Optuna until target COMPLETE trials is reached")
    parser.add_argument("--algo", required=True)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--target-complete-trials", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--objectives", nargs="+", default=None)
    parser.add_argument("--include-inference", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--gpu-ids", type=str, default=None)
    parser.add_argument("--max-concurrent-gpu-trials", type=int, default=1)
    parser.add_argument("--intra-trial-workers", type=int, default=1)
    parser.add_argument("--phy-rate-mbps", type=float, default=None)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--qmax", type=int, default=None)
    parser.add_argument("--sweep-min-pps", type=int, default=None)
    parser.add_argument("--sweep-max-pps", type=int, default=None)
    parser.add_argument("--sweep-steps", type=int, default=None)
    parser.add_argument("--tuning-sim-time", type=float, default=None,
                        help="Override SIM_TIME_S during tuning (e.g. 150).")
    parser.add_argument("--tuning-episodes", type=int, default=None,
                        help="Override MARL EPISODES during tuning (e.g. 1000).")
    parser.add_argument("--skip-ablations", action="store_true",
                        help="Skip ablation studies during tuning trials.")
    args = parser.parse_args()

    objectives = list(args.objectives or DEFAULT_OBJECTIVES)
    if args.include_inference and "inference" not in objectives:
        objectives.append("inference")
    directions = [OBJECTIVE_DIRECTIONS[obj] for obj in objectives]

    if args.target_complete_trials is not None and args.n_trials is not None:
        raise ValueError("Use either --target-complete-trials or --n-trials, not both.")

    target_complete = args.target_complete_trials
    if target_complete is None:
        args_for_default = argparse.Namespace(n_trials=args.n_trials, dry_run=args.dry_run)
        target_complete = resolve_trial_count(args_for_default, args.algo)

    full_study_name = f"{args.study_name}_{args.algo}"
    study_dir = os.path.join(project_root, "results", "optuna", full_study_name)
    os.makedirs(study_dir, exist_ok=True)
    db_path = os.path.join(study_dir, f"{full_study_name}.db")
    storage = f"sqlite:///{db_path}"
    lock_path = os.path.join(study_dir, ".target_trials.lock")

    with file_lock(lock_path):
        complete = count_complete_trials(full_study_name, storage, directions, objectives)
        remaining = max(0, target_complete - complete)
        summary = {
            "study_name": full_study_name,
            "objectives": objectives,
            "directions": directions,
            "target_complete_trials": target_complete,
            "complete_before": complete,
            "remaining_to_run": remaining,
        }
        with open(os.path.join(study_dir, "target_trials_status.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if remaining <= 0:
            print(json.dumps(summary, indent=2))
            # Regenerate study artifacts in case a prior crash left them missing
            # (e.g. representative_pareto_points.csv needed by run_best_optuna_pipeline.py)
            _ensure_study_artifacts(full_study_name, storage, directions, objectives, study_dir, args.algo)
            return

    # ----------------------------------------------------------------
    # Retry loop: re-check COMPLETE trials after each batch.
    # If trials fail (OOM, BrokenPipeError), remaining > 0 and we
    # re-launch with only the deficit.  Cap at MAX_RETRY_ROUNDS to
    # prevent infinite loops when an algo is fundamentally broken.
    # ----------------------------------------------------------------
    MAX_RETRY_ROUNDS = 5
    retry_round = 0

    while remaining > 0 and retry_round < MAX_RETRY_ROUNDS:
        retry_round += 1
        print(f"\n[run_optuna_until_target] Round {retry_round}/{MAX_RETRY_ROUNDS}: "
              f"{complete} COMPLETE, {remaining} remaining to reach {target_complete}")

        # Build command OUTSIDE the lock so all workers can train concurrently
        cmd = [
            sys.executable,
            os.path.join(project_root, "experiments", "run_pareto_optuna.py"),
            "--algo", args.algo,
            "--study-name", args.study_name,
            "--objectives", *objectives,
            "--n-trials", str(remaining),
            "--seed", str(args.seed),
            "--n-jobs", str(args.n_jobs),
            "--max-concurrent-gpu-trials", str(args.max_concurrent_gpu_trials),
            "--intra-trial-workers", str(args.intra_trial_workers),
        ]
        if args.include_inference:
            cmd.append("--include-inference")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.force_retrain:
            cmd.append("--force-retrain")
        if args.gpu_ids is not None:
            cmd.extend(["--gpu-ids", args.gpu_ids])
        if args.phy_rate_mbps is not None:
            cmd.extend(["--phy-rate-mbps", str(args.phy_rate_mbps)])
        if args.nodes is not None:
            cmd.extend(["--nodes", str(args.nodes)])
        if args.qmax is not None:
            cmd.extend(["--qmax", str(args.qmax)])
        if args.sweep_min_pps is not None:
            cmd.extend(["--sweep-min-pps", str(args.sweep_min_pps)])
        if args.sweep_max_pps is not None:
            cmd.extend(["--sweep-max-pps", str(args.sweep_max_pps)])
        if args.sweep_steps is not None:
            cmd.extend(["--sweep-steps", str(args.sweep_steps)])
        if getattr(args, "tuning_sim_time", None) is not None:
            cmd.extend(["--tuning-sim-time", str(args.tuning_sim_time)])
        if getattr(args, "tuning_episodes", None) is not None:
            cmd.extend(["--tuning-episodes", str(args.tuning_episodes)])
        if getattr(args, "skip_ablations", False):
            cmd.append("--skip-ablations")

        # Don't use check=True: Optuna handles trial failures internally.
        # The subprocess may exit non-zero if ALL trials fail, but partial
        # completions are still saved in SQLite.
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[run_optuna_until_target] WARNING: subprocess exited with rc={result.returncode}")

        # Re-check how many COMPLETE trials we now have
        with file_lock(lock_path):
            complete = count_complete_trials(full_study_name, storage, directions, objectives)
            remaining = max(0, target_complete - complete)
            summary["complete_after_round"] = complete
            summary["remaining_after_round"] = remaining
            summary["retry_round"] = retry_round
            with open(os.path.join(study_dir, "target_trials_status.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        if remaining <= 0:
            print(f"[run_optuna_until_target] Target reached: {complete} COMPLETE trials")
            break

        # If no progress was made (all new trials failed), warn but retry
        prev_complete = summary.get("complete_before", 0)
        if complete <= prev_complete:
            print(f"[run_optuna_until_target] WARNING: No new COMPLETE trials this round "
                  f"({complete} == {prev_complete}). Will retry {MAX_RETRY_ROUNDS - retry_round} more times.")

    if remaining > 0:
        print(f"[run_optuna_until_target] ERROR: After {MAX_RETRY_ROUNDS} rounds, "
              f"only {complete}/{target_complete} COMPLETE trials. {remaining} still missing.")
        sys.exit(1)

    # Ensure study artifacts exist
    _ensure_study_artifacts(full_study_name, storage, directions, objectives, study_dir, args.algo)


if __name__ == "__main__":
    main()
