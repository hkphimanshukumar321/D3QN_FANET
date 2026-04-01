import optuna
import pandas as pd
import plotly.express as px
import os

# Define the base directory where all the algorithm folders are stored
# (Assuming you run this script from the project root)
BASE_DIR = "experiments/results/optuna"

# Dictionary mapping 'Algorithm Name' to 'Folder/Study Name'
# Notice how we replaced 'full_v100_sweep_magat_d3qn' with your new dedicated sweep!
studies_to_load = {
    "A2C": "full_v100_sweep_a2c",
    "DQN": "full_v100_sweep_dqn",
    "IQL": "full_v100_sweep_iql",
    "MCA_D3QN": "full_v100_sweep_mca_d3qn",
    "PPO": "full_v100_sweep_ppo",
    "QMIX": "full_v100_sweep_qmix",
    "VDN": "full_v100_sweep_vdn",
    "MAGAT_D3QN": "magat_dedicated_sweep" # <--- The new sweep from the second server
}

all_dataframes = []

for algo_name, study_db_name in studies_to_load.items():
    print(f"Loading {algo_name} ...")
    
    # Construct the path to the DB file based on your image
    # Format: results/optuna/folder_name/folder_name.db
    db_path = f"{BASE_DIR}/{study_db_name}/{study_db_name}.db"
        
    full_db_url = f"sqlite:///{db_path}"
    
    try:
        # Load the study
        study = optuna.load_study(study_name=study_db_name, storage=full_db_url)
        
        # Get dataframe
        df = study.trials_dataframe()
        
        # Filter only COMPLETED trials (ignore hanging/running/failed ones like the canceled MAGAT)
        df = df[df['state'] == 'COMPLETE'].copy()
        
        if len(df) > 0:
            # Add a column for the algorithm name so we can color-code the plot
            df['Algorithm'] = algo_name
            all_dataframes.append(df)
            print(f" -> Found {len(df)} completed trials for {algo_name}")
        else:
            print(f" -> No completed trials found for {algo_name}")
            
    except Exception as e:
        print(f" -> Could not load {study_db_name}: {e}")

# Combine everything into one giant Pandas dataframe
if not all_dataframes:
    print("No data found! Check your database paths.")
    exit()

combined_df = pd.concat(all_dataframes, ignore_index=True)

# ---------------------------------------------------------
# Create the Pareto Front Scatter Plot using Plotly
# ---------------------------------------------------------
# Note: In Optuna multi-objective sweeps, the final metric values 
# are stored in 'values_0' and 'values_1'. 

fig = px.scatter(
    combined_df,
    x="values_0", 
    y="values_1", 
    color="Algorithm",      # This creates the legend and colors the dots by Algorithm!
    hover_data=["number", "params_w_throughput", "params_w_delay"], # Shows weights when you hover your mouse
    title="Hybrid Communication Protocol: Pareto Optimality Comparison",
    labels={
        "values_0": "Objective 0 Value (e.g., Throughput)",
        "values_1": "Objective 1 Value (e.g., Delay)"
    },
    template="plotly_dark",
    # You can change opacity if dots overlap a lot
    opacity=0.8
)

# Optional: Add gridlines and size
fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(width=1000, height=800)

# Save to an interactive HTML file
output_file = "experiments/results/combined_pareto_front.html"
os.makedirs("experiments/results", exist_ok=True)
fig.write_html(output_file)

print(f"\nSuccess! Open {output_file} in your browser to interact with the chart.")
