#!/bin/bash
# ==============================================================================
# 96-Core Multi-Process Optuna Sweep Launcher
# ==============================================================================
# Bypasses the Python GIL by launching separate Python processes.
# This ensures that your 96 virtual cores are actively utilized because
# the environment simulation steps (which are CPU-bottlenecked) happen
# in entirely distinct processes.
# ==============================================================================

set -e

STUDY_NAME="full_v100_sweep"
SEED=42
GPU_ID=0

mkdir -p logs

echo "============================================================"
echo "  LAUNCHING MULTI-PROCESS ISOLATED OPTUNA SWEEP"
echo "  96 Cores | V100 GPU | Isolated Checkpoints"
echo "============================================================"

# Helper function to launch N background processes for an algorithm
launch_workers() {
    local algo=$1
    local trials=$2
    local workers=$3

    echo "Launching $workers isolated processes for $algo ($trials total trials)..."
    for (( i=1; i<=$workers; i++ ))
    do
        # NOTE: We set --n-jobs 1 here because the parallelism is handled
        # by the bash script running multiple isolated Python processes!
        nohup python experiments/run_pareto_optuna.py \
            --algo $algo \
            --n-trials $trials \
            --study-name "$STUDY_NAME" \
            --seed $SEED \
            --gpu-ids $GPU_ID \
            --force-retrain \
            --n-jobs 1 \
            > logs/optuna_${algo}_worker_${i}.log 2>&1 &
        
        # Sleep randomly between 1-5 seconds to prevent SQLite lock collisions
        # right at the exact launch millisecond.
        sleep $((1 + $RANDOM % 5))
    done
}

# --- MARL Algorithms (Heavy Compute) ---
# Each worker is a dedicated Python process. 
# We launch 4 workers for each MARL method (16 processes total)
# They will all hit the same SQLite DB, figure out what trials are pending,
# and chew through the 20-trial queue collaboratively!

launch_workers "magat_d3qn" 20 4
launch_workers "qmix" 20 4
launch_workers "vdn" 20 4
launch_workers "iql" 20 4

# --- SARL Algorithms (Light Compute) ---
launch_workers "dqn" 48 2
launch_workers "mca_d3qn" 48 2
launch_workers "ppo" 48 2
launch_workers "a2c" 48 2

echo ""
echo "============================================================"
echo "  ALL 24 PYTHON PROCESSES LAUNCHED AND DETACHED!"
echo "============================================================"
echo "You can safely close this SSH session."
echo ""
echo "To monitor:"
echo "  tail -f logs/optuna_magat_d3qn_worker_1.log"
echo "  htop        (Watch your 96 cores get used!)"
echo "  nvidia-smi  (Watch the V100 Memory and Compute util)"
echo "============================================================"
