import optuna
import pandas as pd

study = optuna.load_study(study_name='full_v100_sweep_dqn', storage='sqlite:///full_v100_sweep_dqn (1).db')
df = study.trials_dataframe()

print(f"\n=======================================================")
print(f"  DATABASE OVERVIEW: full_v100_sweep_dqn (1).db")
print(f"=======================================================")
print(f"Total Trials: {len(df)}")
print(f"Completed Trials: {len(df[df['state'] == 'COMPLETE'])}")
print(f"Failed/Running/Pruned: {len(df[df['state'] != 'COMPLETE'])}")

print(f"\n--- OBJECTIVES (METRICS LOGGED) ---")
for col in df.columns:
    if 'values_' in col:
        print(f" - {col}")

print(f"\n--- HYPERPARAMETERS TUNED ---")
for col in df.columns:
    if 'params_' in col:
        print(f" - {col}")

print(f"\n--- TOP 5 BEST HIGHEST THROUGHPUT OVERVIEWS ---")
if 'values_0' in df.columns:
    top_throughput = df[df['state'] == 'COMPLETE'].sort_values(by='values_0', ascending=False).head(5)
    cols_to_show = ['number', 'values_0', 'values_1'] + [c for c in df.columns if 'params_w_' in c]
    print(top_throughput[cols_to_show].to_string(index=False))
