#!/bin/bash
# ==============================================================================
# MASSSIVE PARALLEL OPTUNA -> FULL PIPELINE LAUNCHER (96-CORE OPTIMIZED)
# ==============================================================================
# This scripts marries the multi-processing power of `launch_parallel_optuna.sh`
# with the full automated artifact packaging of `launch_optuna_then_full_pipeline.sh`.
#
# It will run IQL, VDN, QMIX, and MAGAT-D3QN entirely simultaneously,
# allocating multiple isolated CPU cores to each.
# ==============================================================================

set -e

# 1. Thread tuning so PyTorch doesn't crash the server
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

STUDY_NAME="journal_tune"
TRIALS_PER_ALGO=48
WORKERS_PER_ALGO=15  # 15 workers * 4 algos = 60 CPU cores effectively pegged

mkdir -p logs

echo "============================================================"
echo "  LAUNCHING MULTI-PROCESS OPTUNA + FINAL PIPELINE"
echo "  Target: IQL, VDN, QMIX, MAGAT-D3QN"
echo "============================================================"

# Helper function to launch N background workers PLUS a finalizer
launch_algo_pipeline() {
    local algo=$1
    local trials=$2
    local workers=$3

    echo "Launching $workers completely isolated processes for $algo ($trials total trials)..."
    
    # Start the parallel Optuna sweep workers
    for (( i=1; i<=$workers; i++ ))
    do
        nohup python experiments/run_optuna_until_target.py \
            --algo $algo \
            --study-name "$STUDY_NAME" \
            --n-trials $trials \
            --force-retrain \
            --n-jobs 1 \
            > logs/massive_${algo}_optuna_worker_${i}.log 2>&1 &
        
        sleep $((1 + $RANDOM % 3))
    done

    # Now we launch the FULL pipeline script for this specific algo in the background.
    # It will mathematically merge into the same SQLite DB, wait for the target to hit,
    # and then automatically proceed to Steps 2 & 3 (best config extract + journal suites).
    export ALGO=$algo
    export STUDY_NAME=$STUDY_NAME
    export N_TRIALS=$trials
    export DRY_RUN=0
    export FORCE_RETRAIN=1
    
    nohup bash launch_optuna_then_full_pipeline.sh \
        > logs/massive_${algo}_final_pipeline.log 2>&1 &
    
    echo " -> $algo pipeline watcher attached."
}

# Launch the 4 target MARL algorithms simultaneously
launch_algo_pipeline "iql" $TRIALS_PER_ALGO $WORKERS_PER_ALGO
launch_algo_pipeline "vdn" $TRIALS_PER_ALGO $WORKERS_PER_ALGO
launch_algo_pipeline "qmix" $TRIALS_PER_ALGO $WORKERS_PER_ALGO
launch_algo_pipeline "magat_d3qn" $TRIALS_PER_ALGO $WORKERS_PER_ALGO

echo ""
echo "============================================================"
echo "  ALL PIPELINES LAUNCHED SUCCESSFULLY!"
echo "============================================================"
echo "You now have ~60 isolated Python processes running in parallel."
echo "They will crunch through the Optuna sweeps, and automatically"
echo "extract checkpoints when finished."
echo ""
echo "To monitor Optuna progress, run:"
echo "  tail -f logs/massive_magat_d3qn_optuna_worker_1.log"
echo "To monitor the final extraction:"
echo "  tail -f logs/massive_magat_d3qn_final_pipeline.log"
echo "============================================================"
