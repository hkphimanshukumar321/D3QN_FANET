#!/bin/bash
# ==============================================================================
# Parallel Journal Evidence Launcher
# ==============================================================================
# Runs the journal-readiness evaluation suites in parallel across CPU cores while
# avoiding thread oversubscription inside NumPy / PyTorch.
#
# Usage:
#   WANDB_API_KEY=... bash launch_parallel_journal_experiments.sh /path/to/checkpoints
#   WANDB_DISABLED=1 bash launch_parallel_journal_experiments.sh /path/to/checkpoints
#   cp .env.wandb.example .env.wandb && edit key, then:
#   bash launch_parallel_journal_experiments.sh /path/to/checkpoints
#
# Optional env vars:
#   MAX_PARALLEL=6
#   OMP_THREADS_PER_WORKER=2
#   SWEEP_MIN_PPS=50
#   SWEEP_MAX_PPS=1000
#   SWEEP_STEPS=20
#   RUN_MAGAT_ISOLATION=0
# ==============================================================================

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <checkpoint_dir>"
    exit 1
fi

CHECKPOINT_DIR="$1"
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
OMP_THREADS_PER_WORKER="${OMP_THREADS_PER_WORKER:-2}"
RUN_MAGAT_ISOLATION="${RUN_MAGAT_ISOLATION:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${RESULT_ROOT:-results/journal_parallel/$TS}"
LOG_DIR="$RESULT_ROOT/logs"

mkdir -p "$LOG_DIR"

# Load local wandb secrets if present.
if [[ -f ".env.wandb" ]]; then
    # shellcheck disable=SC1091
    source ".env.wandb"
elif [[ -f ".env.wandb.local" ]]; then
    # shellcheck disable=SC1091
    source ".env.wandb.local"
fi

# Avoid the classic bottleneck: each worker spawning too many BLAS/OpenMP threads.
export OMP_NUM_THREADS="$OMP_THREADS_PER_WORKER"
export MKL_NUM_THREADS="$OMP_THREADS_PER_WORKER"
export OPENBLAS_NUM_THREADS="$OMP_THREADS_PER_WORKER"
export NUMEXPR_NUM_THREADS="$OMP_THREADS_PER_WORKER"
export TORCH_NUM_THREADS="$OMP_THREADS_PER_WORKER"
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

if [[ "${WANDB_DISABLED:-0}" != "1" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is not set. Using existing wandb session/netrc if available."
    echo "If you want zero wandb interaction, export WANDB_DISABLED=1."
fi

SWEEP_ARGS=()
if [[ -n "${SWEEP_MIN_PPS:-}" ]]; then
    SWEEP_ARGS+=(--sweep-min-pps "$SWEEP_MIN_PPS")
fi
if [[ -n "${SWEEP_MAX_PPS:-}" ]]; then
    SWEEP_ARGS+=(--sweep-max-pps "$SWEEP_MAX_PPS")
fi
if [[ -n "${SWEEP_STEPS:-}" ]]; then
    SWEEP_ARGS+=(--sweep-steps "$SWEEP_STEPS")
fi
if [[ -n "${SEED:-}" ]]; then
    SWEEP_ARGS+=(--seed "$SEED")
fi

wait_for_slot() {
    while [[ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]]; do
        sleep 2
    done
}

launch_job() {
    local name="$1"
    shift
    wait_for_slot
    echo "[launch] $name"
    nohup "$@" > "$LOG_DIR/${name}.log" 2>&1 &
}

echo "============================================================"
echo "  Parallel Journal Evidence Launcher"
echo "  checkpoint_dir=$CHECKPOINT_DIR"
echo "  result_root=$RESULT_ROOT"
echo "  max_parallel=$MAX_PARALLEL"
echo "  threads_per_worker=$OMP_THREADS_PER_WORKER"
echo "============================================================"

# ------------------------------------------------------------------
# Generalization blocks
# ------------------------------------------------------------------
for block in node_count mobility_regime topology_scale interference_regime traffic_regime
do
    launch_job "generalization_${block}" \
        "$PYTHON_BIN" experiments/run_generalization_suite.py \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --out-dir "$RESULT_ROOT/generalization/${block}" \
        --study-block "$block" \
        "${SWEEP_ARGS[@]}"
done

# ------------------------------------------------------------------
# Robustness blocks
# ------------------------------------------------------------------
for block in stale_obs noisy_obs graph_corruption handover_info
do
    launch_job "robustness_${block}" \
        "$PYTHON_BIN" experiments/run_robustness_suite.py \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --out-dir "$RESULT_ROOT/robustness/${block}" \
        --study-block "$block" \
        "${SWEEP_ARGS[@]}"
done

# ------------------------------------------------------------------
# Failure recovery
# ------------------------------------------------------------------
launch_job "failure_recovery" \
    "$PYTHON_BIN" experiments/run_failure_recovery_suite.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --out-dir "$RESULT_ROOT/failure_recovery" \
    "${SWEEP_ARGS[@]}"

# ------------------------------------------------------------------
# Complexity benchmark
# ------------------------------------------------------------------
launch_job "complexity" \
    "$PYTHON_BIN" experiments/run_complexity_benchmark.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --out-dir "$RESULT_ROOT/complexity"

# ------------------------------------------------------------------
# Optional MAGAT isolation retraining / evaluation
# ------------------------------------------------------------------
if [[ "$RUN_MAGAT_ISOLATION" == "1" ]]; then
    launch_job "magat_isolation" \
        "$PYTHON_BIN" experiments/run_magat_isolation_suite.py \
        "${SWEEP_ARGS[@]}"
fi

echo ""
echo "All jobs launched."
echo "Logs: $LOG_DIR"
echo "Monitor with:"
echo "  tail -f $LOG_DIR/generalization_node_count.log"
echo "  htop"
echo ""
echo "Wait for completion with:"
echo "  wait"
