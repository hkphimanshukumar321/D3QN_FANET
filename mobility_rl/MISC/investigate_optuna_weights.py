"""Quick investigation: why do MAGAT-D3QN and MAPPO have identical Optuna weights?"""
import pandas as pd, os, glob

ROOT = os.path.abspath(".")

files = {
    "MAGAT-D3QN": "results/results_MAGAT/optuna/journal_tune_magat_d3qn/representative_pareto_points.csv",
    "MAPPO":      "results/results_MAPPO_VDN/optuna/journal_tune_mappo/representative_pareto_points.csv",
    "VDN":        "results/results_MAPPO_VDN/optuna/journal_tune_vdn/representative_pareto_points.csv",
    "QMIX":       "results/results_iql_qmix/optuna/journal_tune_qmix/representative_pareto_points.csv",
    "IQL":        "results/results_iql_qmix/optuna/journal_tune_iql/representative_pareto_points.csv",
}

print("=" * 90)
print("INVESTIGATION: Why do MAGAT-D3QN and MAPPO share identical Optuna weights?")
print("=" * 90)

# 1. Compare the balanced_min_L2_to_ideal row for each algo
print("\n--- balanced_min_L2_to_ideal comparison ---")
for name, f in files.items():
    df = pd.read_csv(f)
    bal = df[df["representative_label"] == "balanced_min_L2_to_ideal"]
    if bal.empty:
        print(f"  {name}: NO balanced row")
        continue
    r = bal.iloc[0]
    print(f"  {name:12s} | trial={int(r['trial_number']):3d} | "
          f"wT={r['wT']:.4f} wD={r['wD']:.4f} wDrop={r['wDrop']:.4f} wCol={r['wCol']:.4f} | "
          f"lr={r['param_lr']:.6e} B={int(r['param_batch_size'])} h={int(r['param_hidden_dim'])}")

# 2. Check ALL trials in each representative CSV — are ALL MAPPO trials identical?
print("\n--- All trials in each representative CSV ---")
for name, f in files.items():
    df = pd.read_csv(f)
    unique_wt = df["wT"].nunique()
    unique_lr = df["param_lr"].nunique()
    print(f"  {name:12s} | {len(df)} rows | unique wT values: {unique_wt} | unique lr values: {unique_lr}")
    for _, r in df.iterrows():
        print(f"    trial={int(r['trial_number']):3d} label={r['representative_label']:30s} "
              f"wT={r['wT']:.4f} lr={r['param_lr']:.6e} | "
              f"raw_wT={r.get('param_raw_wT', 'N/A')}")

# 3. Check the Optuna DB files — how many unique parameter sets exist?
print("\n--- Checking Optuna SQLite databases ---")
db_locations = {
    "MAGAT-D3QN": "results/results_MAGAT/optuna/journal_tune_magat_d3qn",
    "MAPPO":      "results/results_MAPPO_VDN/optuna/journal_tune_mappo",
    "VDN":        "results/results_MAPPO_VDN/optuna/journal_tune_vdn",
    "QMIX":       "results/results_iql_qmix/optuna/journal_tune_qmix",
    "IQL":        "results/results_iql_qmix/optuna/journal_tune_iql",
}

for name, study_dir in db_locations.items():
    db_files = glob.glob(os.path.join(study_dir, "*.db"))
    config_files = glob.glob(os.path.join(study_dir, "trial_*/config.json"))
    print(f"  {name:12s} | DB files: {len(db_files)} | config.json files: {len(config_files)}")
    for db in db_files:
        print(f"    DB: {os.path.basename(db)} ({os.path.getsize(db) / 1024:.0f} KB)")

# 4. Read the actual DB to check unique params
print("\n--- Unique params from Optuna DB ---")
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    for name, study_dir in db_locations.items():
        db_files = glob.glob(os.path.join(study_dir, "*.db"))
        if not db_files:
            print(f"  {name}: No DB found")
            continue
        db_path = db_files[0]
        study_name_in_db = os.path.splitext(os.path.basename(db_path))[0]
        # Try loading with the filename as study name
        try:
            study = optuna.load_study(
                study_name=study_name_in_db,
                storage=f"sqlite:///{os.path.abspath(db_path)}"
            )
        except KeyError:
            # Try listing studies
            summaries = optuna.study.get_all_study_summaries(
                storage=f"sqlite:///{os.path.abspath(db_path)}"
            )
            if summaries:
                study = optuna.load_study(
                    study_name=summaries[0].study_name,
                    storage=f"sqlite:///{os.path.abspath(db_path)}"
                )
            else:
                print(f"  {name}: No studies in DB")
                continue
        
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"  {name:12s} | study='{study.study_name}' | {len(complete)} complete trials")
        
        # Show unique parameter sets
        param_sets = set()
        for t in complete[:10]:  # first 10
            key = tuple(sorted(t.params.items()))
            param_sets.add(key)
            if len(complete) <= 10 or t.number in [complete[0].number, complete[-1].number]:
                wt = t.params.get("w_throughput", t.params.get("wT", "?"))
                lr = t.params.get("lr", "?")
                print(f"    trial {t.number}: wT={wt} lr={lr}")
        
        print(f"    -> {len(param_sets)} unique param sets in first {min(10, len(complete))} trials")
except ImportError:
    print("  optuna not installed, skipping DB check")

print("\n" + "=" * 90)
print("CONCLUSION")
print("=" * 90)
