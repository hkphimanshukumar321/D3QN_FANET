import optuna
import time
import os
from tqdm import tqdm
from datetime import timedelta

def monitor_sweep():
    """
    Connects to the Optuna SQLite databases to track the global progress.
    Calculates accurate collective ETAs based on parallel n-jobs workers.
    """
    
    # Target trials per algorithm
    targets = {
        "full_v100_sweep_magat_d3qn": 20,
        "full_v100_sweep_qmix": 20,
        "full_v100_sweep_vdn": 20,
        "full_v100_sweep_iql": 20,
        "full_v100_sweep_dqn": 48,
        "full_v100_sweep_mca_d3qn": 48,
        "full_v100_sweep_ppo": 48,
        "full_v100_sweep_a2c": 48
    }

    # Knowing how many background processes we launched per algo lets us 
    # perfectly calculate the remaining wall-clock time.
    workers_count = {
        "magat_d3qn": 4,
        "qmix": 4,
        "vdn": 4,
        "iql": 4,
        "dqn": 2,
        "mca_d3qn": 2,
        "ppo": 2,
        "a2c": 2
    }

    print("======================================================")
    print("  LIVE OPTUNA SWEEP TRACKER (Ctrl+C to exit)")
    print("  Calculating true collective global ETA...")
    print("======================================================\n")
    
    bars = {}
    
    try:
        while True:
            max_eta_seconds = 0
            
            for study_name, target in targets.items():
                db_path = os.path.join("results", "optuna", study_name, f"{study_name}.db")
                storage_url = f"sqlite:///{db_path}"
                
                # Check DB exists
                if not os.path.exists(db_path):
                    continue
                    
                try:
                    # Load the study quietly
                    optuna.logging.set_verbosity(optuna.logging.ERROR)
                    study = optuna.load_study(study_name=study_name, storage=storage_url)
                    
                    # Count fully completed trials
                    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                    completed_count = len(completed_trials)
                    
                    algo_name = study_name.replace("full_v100_sweep_", "")
                    
                    # --- CALCULATE PREDICTIVE ETA ---
                    remaining_trials = target - completed_count
                    if remaining_trials > 0 and completed_count > 0:
                        # Grab the average duration of the last 5 successful trials for high accuracy
                        recent = completed_trials[-5:]
                        avg_duration = sum((t.datetime_complete - t.datetime_start).total_seconds() for t in recent) / len(recent)
                        
                        # Divide duration by identical parallel workers
                        n_jobs = workers_count.get(algo_name, 1)
                        study_eta = (remaining_trials / n_jobs) * avg_duration
                        
                        if study_eta > max_eta_seconds:
                            max_eta_seconds = study_eta

                    # --- UPDATE TQDM BARS ---
                    if study_name not in bars:
                        bars[study_name] = tqdm(total=target, desc=f"{algo_name.upper():<10}", position=len(bars), leave=True)
                        bars[study_name].update(completed_count)
                    else:
                        current_val = bars[study_name].n
                        if completed_count > current_val:
                            bars[study_name].update(completed_count - current_val)
                            
                except Exception:
                    pass
            
            # Print the Net Global ETA without messing up the tqdm bar stack
            if max_eta_seconds > 0:
                global_eta = timedelta(seconds=int(max_eta_seconds))
                days = global_eta.days
                hours, remainder = divmod(global_eta.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                
                tqdm.write(f"\r[GLOBAL SWEEP ETA] Net Total Time Left: {days} Days, {hours} Hours, {minutes} Minutes    ", end="")
            elif len(bars) > 0 and all(bars[s].n == targets[s] for s in bars):
                tqdm.write("\r[GLOBAL SWEEP ETA] ★ ALL TRIALS COMPLETED! ★                               ", end="")
                    
            # Refresh every 30 seconds
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\nExiting monitor. The background sweep is still running safely!")

if __name__ == "__main__":
    monitor_sweep()
