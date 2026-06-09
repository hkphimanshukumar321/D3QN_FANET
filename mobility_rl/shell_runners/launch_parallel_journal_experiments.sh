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
OMP_THREADS_PER_WORKER="${OMP_THREADS_PER_WORKER:-2}"
RUN_MAGAT_ISOLATION="${RUN_MAGAT_ISOLATION:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${RESULT_ROOT:-results/journal_parallel/$TS}"
LOG_DIR="$RESULT_ROOT/logs"
PID_DIR="$RESULT_ROOT/pids"
STATUS_DIR="$RESULT_ROOT/status"
MANIFEST="$RESULT_ROOT/jobs_manifest.tsv"

mkdir -p "$LOG_DIR" "$PID_DIR" "$STATUS_DIR"
printf "name\tpid_file\tlog_file\tstatus_file\n" > "$MANIFEST"

if [[ "${AUTO_NOHUP:-1}" == "1" && -z "${SELF_NOHUP_LAUNCHED:-}" ]]; then
    export SELF_NOHUP_LAUNCHED=1
    export RESULT_ROOT="$RESULT_ROOT"
    nohup bash "$0" "$@" > "$LOG_DIR/launcher.log" 2>&1 &
    LAUNCHER_PID=$!
    printf "%s\n" "$LAUNCHER_PID" > "$PID_DIR/launcher.pid"
    printf "DETACHED\n" > "$STATUS_DIR/launcher.status"
    echo "Detached journal launcher."
    echo "  pid=$LAUNCHER_PID"
    echo "  root=$RESULT_ROOT"
    echo "  log=$LOG_DIR/launcher.log"
    exit 0
fi

if [[ -n "${SELF_NOHUP_LAUNCHED:-}" ]]; then
    finalize_launcher_status() {
        local rc=$?
        if [[ $rc -eq 0 ]]; then
            printf "COMPLETED\n" > "$STATUS_DIR/launcher.status"
        else
            printf "FAILED:%s\n" "$rc" > "$STATUS_DIR/launcher.status"
        fi
    }
    trap finalize_launcher_status EXIT
fi

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

CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 8)"
if [[ -z "${MAX_PARALLEL:-}" ]]; then
    MAX_PARALLEL=$(( CPU_COUNT / OMP_THREADS_PER_WORKER ))
    if [[ "$MAX_PARALLEL" -lt 1 ]]; then
        MAX_PARALLEL=1
    fi
    if [[ "$MAX_PARALLEL" -gt 8 ]]; then
        MAX_PARALLEL=8
    fi
else
    MAX_PARALLEL="${MAX_PARALLEL}"
fi

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
    local log_file="$LOG_DIR/${name}.log"
    local pid_file="$PID_DIR/${name}.pid"
    local status_file="$STATUS_DIR/${name}.status"
    printf "RUNNING\n" > "$status_file"
    nohup bash -lc '
        status_file="$1"
        shift
        "$@"
        rc=$?
        if [[ $rc -eq 0 ]]; then
            printf "COMPLETED\n" > "$status_file"
        else
            printf "FAILED:%s\n" "$rc" > "$status_file"
        fi
        exit $rc
    ' _ "$status_file" "$@" > "$log_file" 2>&1 &
    local pid=$!
    printf "%s\n" "$pid" > "$pid_file"
    printf "%s\t%s\t%s\t%s\n" "$name" "$pid_file" "$log_file" "$status_file" >> "$MANIFEST"
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
echo "PIDs: $PID_DIR"
echo "Manifest: $MANIFEST"
echo "Monitor with:"
echo "  bash monitor_pipeline.sh $RESULT_ROOT"
echo "Finalize package:"
echo "  bash finalize_journal_package.sh $RESULT_ROOT"
echo ""
echo "Wait for completion with:"
echo "  wait"
