#!/bin/bash
# ==============================================================================
# Full Journal Pipeline Launcher
# ==============================================================================
# 1. Runs unified training/evaluation once to produce checkpoints, reward curves,
#    baseline plots, and unified eval CSVs.
# 2. Launches the parallel journal evidence suites against the generated
#    checkpoint directory.
#
# Usage:
#   bash launch_full_journal_pipeline.sh
#
# Optional env vars:
#   WANDB_API_KEY=...
#   WANDB_DISABLED=1
#   DRY_RUN=1
#   FORCE_RETRAIN=1
#   SKIP_TRAINING=0
#   PHY_RATE_MBPS=5
#   NODES=50
#   QMAX=100
#   SWEEP_MIN_PPS=50
#   SWEEP_MAX_PPS=1000
#   SWEEP_STEPS=20
#   MAX_PARALLEL=6
#   OMP_THREADS_PER_WORKER=2
#   RUN_MAGAT_ISOLATION=0
# ==============================================================================

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
PIPELINE_ROOT="${PIPELINE_ROOT:-results/full_journal_pipeline/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$PIPELINE_ROOT"
LOG_DIR="$PIPELINE_ROOT/logs"
PID_DIR="$PIPELINE_ROOT/pids"
STATUS_DIR="$PIPELINE_ROOT/status"
mkdir -p "$LOG_DIR" "$PID_DIR" "$STATUS_DIR"

if [[ "${AUTO_NOHUP:-1}" == "1" && -z "${SELF_NOHUP_LAUNCHED:-}" ]]; then
    export SELF_NOHUP_LAUNCHED=1
    export PIPELINE_ROOT="$PIPELINE_ROOT"
    nohup bash "$0" "$@" > "$LOG_DIR/launcher.log" 2>&1 &
    LAUNCHER_PID=$!
    printf "%s\n" "$LAUNCHER_PID" > "$PID_DIR/launcher.pid"
    printf "DETACHED\n" > "$STATUS_DIR/launcher.status"
    echo "Detached full pipeline launcher."
    echo "  pid=$LAUNCHER_PID"
    echo "  root=$PIPELINE_ROOT"
    echo "  log=$LOG_DIR/launcher.log"
    exit 0
fi

export DRY_RUN
export FORCE_RETRAIN
export SKIP_TRAINING
export PIPELINE_ROOT
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

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

ARTIFACT_JSON="$PIPELINE_ROOT/unified_artifacts.json"
export ARTIFACT_JSON
export PHY_RATE_MBPS="${PHY_RATE_MBPS:-}"
export NODES="${NODES:-}"
export QMAX="${QMAX:-}"
export SWEEP_MIN_PPS="${SWEEP_MIN_PPS:-}"
export SWEEP_MAX_PPS="${SWEEP_MAX_PPS:-}"
export SWEEP_STEPS="${SWEEP_STEPS:-}"
export SEED="${SEED:-}"

run_stage() {
    local name="$1"
    shift
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
    wait "$pid"
}

echo "============================================================"
echo "  Full Journal Pipeline"
echo "  pipeline_root=$PIPELINE_ROOT"
echo "============================================================"
echo "Step 1/2: unified training + checkpoints + reward curves"

run_stage "unified_training" \
    "$PYTHON_BIN" experiments/run_unified_and_dump_artifacts.py

if [[ ! -f "$ARTIFACT_JSON" ]]; then
    echo "Unified artifact JSON not created: $ARTIFACT_JSON"
    exit 1
fi

CHECKPOINT_DIR="$("$PYTHON_BIN" - <<'PY'
import json, os
path = os.path.abspath(os.environ["ARTIFACT_JSON"])
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(data["checkpoint_dir"])
PY
)"

OUT_DIR="$("$PYTHON_BIN" - <<'PY'
import json, os
path = os.path.abspath(os.environ["ARTIFACT_JSON"])
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(data["out_dir"])
PY
)"

echo "Unified run complete."
echo "  out_dir=$OUT_DIR"
echo "  checkpoint_dir=$CHECKPOINT_DIR"
echo "  reward_curves_dir=$OUT_DIR/images"
echo ""
echo "Step 2/2: journal evidence suites in parallel"

export RESULT_ROOT="$PIPELINE_ROOT/journal_suites"
export SWEEP_MIN_PPS="${SWEEP_MIN_PPS:-}"
export SWEEP_MAX_PPS="${SWEEP_MAX_PPS:-}"
export SWEEP_STEPS="${SWEEP_STEPS:-}"
export SEED="${SEED:-}"
export OMP_THREADS_PER_WORKER="${OMP_THREADS_PER_WORKER:-2}"
export RUN_MAGAT_ISOLATION="${RUN_MAGAT_ISOLATION:-0}"
if [[ -n "${MAX_PARALLEL:-}" ]]; then
    export MAX_PARALLEL
fi

bash launch_parallel_journal_experiments.sh "$CHECKPOINT_DIR"

echo ""
echo "============================================================"
echo "Full pipeline launched."
echo "Training artifacts:"
echo "  $OUT_DIR"
echo "Parallel journal suites:"
echo "  $RESULT_ROOT"
echo "Monitor:"
echo "  bash monitor_pipeline.sh $PIPELINE_ROOT"
echo "Finalize package:"
echo "  bash finalize_journal_package.sh $PIPELINE_ROOT"
echo "============================================================"
