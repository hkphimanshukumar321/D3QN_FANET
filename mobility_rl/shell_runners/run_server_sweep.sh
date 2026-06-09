#!/bin/bash
# ==============================================================================
# Single-Server MARL Sweep Runner (Reliable)
# ==============================================================================
# Runs the Optuna sweep for ONE algorithm with a small number of speed workers,
# then automatically triggers the full journal pipeline.
#
# Uses AUTO_NOHUP (proven working on JupyterHub) for the pipeline watcher.
# Speed workers use nohup+& and are EXPENDABLE — if they die, the watcher
# continues alone.
#
# Usage:
#   ALGO=iql bash run_server_sweep.sh
#   ALGO=vdn WORKERS=4 bash run_server_sweep.sh
#
# Required env vars:
#   ALGO           - algorithm name (iql, vdn, qmix, magat_d3qn)
#
# Optional env vars:
#   WORKERS=4      - number of speed workers (default: 4, keep LOW)
#   N_TRIALS=48    - total Optuna trials target
#   STUDY_NAME=journal_tune
# ==============================================================================

set -euo pipefail

ALGO="${ALGO:?'ALGO env var required (iql, vdn, qmix, magat_d3qn)'}"
WORKERS="${WORKERS:-4}"
N_TRIALS="${N_TRIALS:-48}"
STUDY_NAME="${STUDY_NAME:-journal_tune}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

mkdir -p logs

echo "============================================================"
echo "  Server Sweep Runner"
echo "  Algorithm: $ALGO"
echo "  Speed workers: $WORKERS"
echo "  Target trials: $N_TRIALS"
echo "============================================================"

# Step 1: Launch expendable speed workers
TRIALS_PER_WORKER=$(( (N_TRIALS + WORKERS - 1) / WORKERS ))
echo "Launching $WORKERS speed workers ($TRIALS_PER_WORKER trials each)..."

for i in $(seq 1 $WORKERS); do
    nohup python experiments/run_pareto_optuna.py \
        --algo "$ALGO" \
        --study-name "$STUDY_NAME" \
        --n-trials "$TRIALS_PER_WORKER" \
        --force-retrain \
        --n-jobs 1 \
        > "logs/speed_${ALGO}_w${i}.log" 2>&1 &
    sleep 3
done

echo "Speed workers launched (expendable — watcher handles completion)."
echo ""

# Step 2: Launch the PROVEN master pipeline watcher
# This script uses AUTO_NOHUP internally and is proven to survive on JupyterHub.
# It will: sweep until target → extract best config → final train → journal suites
echo "Launching master pipeline watcher..."
export ALGO
export STUDY_NAME
export N_TRIALS
export FORCE_RETRAIN=1
export DRY_RUN=0

bash launch_optuna_then_full_pipeline.sh

echo ""
echo "============================================================"
echo "  Pipeline launched for $ALGO!"
echo "  Monitor: python utils/monitor_optuna_eta.py ."
echo "============================================================"
