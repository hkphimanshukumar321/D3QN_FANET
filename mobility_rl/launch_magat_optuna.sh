#!/bin/bash
# ==============================================================================
# Isolated MAGAT-D3QN 96-Core Multi-Process Optuna Sweep Launcher
# ==============================================================================
# Dedicated script to spawn 20 concurrent Python processes exclusively for 
# MAGAT-D3QN to maximize the V100 GPU and 96-Core Xeon CPU utilization
# on the secondary server.
# ==============================================================================

set -e

# --- CRITICAL CPU OPTIMIZATION ---
# 20 processes * 4 threads = 80 cores perfectly saturated.
# Leaves 16 cores free for OS and I/O.
# Without this, PyTorch will spawn 96 threads per process and choke the CPU.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

STUDY_NAME="magat_dedicated_sweep"
SEED=42
GPU_ID=0

# Hardware allocation matching 96-core Xeon / V100 32GB / 1.4TB RAM
WORKERS=20
TRIALS_PER_WORKER=20 # Each worker queries the same DB up to 20 times. Optuna handles deduplication.

mkdir -p logs

echo "============================================================"
echo "  LAUNCHING ISOLATED MAGAT-D3QN OPTUNA SWEEP"
echo "  Hardware Scale: 96 Cores | V100 32GB GPU | 1400GB RAM"
echo "  Workers: $WORKERS Concurrent Processes"
echo "============================================================"

echo "Launching $WORKERS isolated processes for magat_d3qn ($TRIALS_PER_WORKER target trials)..."
for (( i=1; i<=$WORKERS; i++ ))
do
    # NOTE: We set --n-jobs 1 here because the parallelism is handled
    # by the bash script running multiple isolated Python processes!
    nohup python experiments/run_pareto_optuna.py \
        --algo magat_d3qn \
        --n-trials $TRIALS_PER_WORKER \
        --study-name "$STUDY_NAME" \
        --seed $SEED \
        --gpu-ids $GPU_ID \
        --force-retrain \
        --n-jobs 1 \
        > logs/optuna_magat_d3qn_worker_${i}.log 2>&1 &
    
    # Sleep randomly between 1-5 seconds to prevent SQLite lock collisions
    # right at the exact launch millisecond.
    sleep $((1 + $RANDOM % 5))
done

echo ""
echo "============================================================"
echo "  ALL $WORKERS MAGAT PROCESSES LAUNCHED AND DETACHED!"
echo "============================================================"
echo "You can safely close this SSH session."
echo ""
echo "To monitor on the new server:"
echo "  tail -f logs/optuna_magat_d3qn_worker_1.log"
echo "  htop        (Watch the 80 cores light up!)"
echo "  nvidia-smi  (Watch the V100 memory fill and compute utilize)"
echo "============================================================"
