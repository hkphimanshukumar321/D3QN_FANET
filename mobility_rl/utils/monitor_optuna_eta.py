import argparse
import json
import os
import sys
import datetime
import optuna

def main():
    parser = argparse.ArgumentParser("Monitor Optuna ETA")
    parser.add_argument("pipeline_root", type=str)
    args = parser.parse_args()

    pipeline_root = args.pipeline_root
    if not os.path.exists(pipeline_root):
        return

    # Find status files
    status_files = []
    for f in os.listdir(pipeline_root):
        if f.startswith("target_trials_status") and f.endswith(".json"):
            status_files.append(os.path.join(pipeline_root, f))
    
    if not status_files:
        return
        
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    print("Optuna Study Progress & ETA:")
    for status_file in status_files:
        try:
            with open(status_file, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        
        study_name = data.get("study_name")
        target = data.get("target_complete_trials")
        
        if not study_name or target is None:
            continue
        
        # Check DB
        study_dir = os.path.join(project_root, "results", "optuna", study_name)
        db_path = os.path.join(study_dir, f"{study_name}.db")
        if not os.path.exists(db_path):
            continue
        
        storage = f"sqlite:///{db_path}"
        try:
            # Suppress optuna logging
            optuna.logging.set_verbosity(optuna.logging.ERROR)
            study = optuna.load_study(study_name=study_name, storage=storage)
            trials = study.trials
        except Exception:
            continue
        
        complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        running = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
        
        num_complete = len(complete)
        remaining = max(0, target - num_complete)
        
        durations = []
        for t in complete:
            if t.datetime_start and t.datetime_complete:
                durations.append((t.datetime_complete - t.datetime_start).total_seconds())
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        eta_seconds = remaining * avg_duration
        
        if remaining == 0:
            eta_str = "COMPLETED"
        else:
            eta_delta = datetime.timedelta(seconds=int(eta_seconds))
            eta_str = f"ETA: {eta_delta} ({len(running)} running)"
            
        print(f"  {study_name:<30} | {num_complete:>2}/{target:<2} trials | Avg: {avg_duration:5.1f}s | {eta_str}")

if __name__ == "__main__":
    main()
