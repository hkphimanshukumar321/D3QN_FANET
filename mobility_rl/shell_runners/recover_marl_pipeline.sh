#!/bin/bash
# ==============================================================================
# MARL Recovery Script — Parallel Tuning for 96-core + GPU Server
# ==============================================================================
# Runs: IQL, VDN, QMIX, MAPPO, MAGAT-D3QN
#
# Parallelization strategy (96 cores, 1 GPU):
#   Phase 1: IQL + VDN + QMIX + MAPPO in PARALLEL (CPU-based, no GPU contention)
#   Phase 2: MAGAT-D3QN alone (GNN needs GPU)
#
# Within each algorithm:
#   --n-jobs N               → N concurrent Optuna trials
#   --max-concurrent-gpu-trials N → GPU guard
#   --tuning-sim-time 150    → 4× faster sim
#   --tuning-episodes 500    → 4× fewer episodes
#   --skip-ablations         → skip ablation sweeps during tuning
#
# Usage:
#   bash shell_runners/recover_marl_pipeline.sh
#
# Optional env vars:
#   PIPELINE_ROOT=results/optuna_then_pipeline/20260809_165642
#   N_TRIALS=24
#   TUNING_SIM_TIME=150
#   TUNING_EPISODES=500
#   PHASE1_JOBS_PER_ALGO=4  (concurrent trials per non-GNN algo)
#   PHASE2_JOBS=2           (concurrent trials for MAGAT-D3QN)
# ==============================================================================

set -uo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE_ROOT="${PIPELINE_ROOT:-results/optuna_then_pipeline/20260809_165642}"
STUDY_NAME="${STUDY_NAME:-journal_tune}"
REPRESENTATIVE_LABEL="${REPRESENTATIVE_LABEL:-balanced_min_L2_to_ideal}"

LOG_DIR="$PIPELINE_ROOT/logs"
PID_DIR="$PIPELINE_ROOT/pids"
STATUS_DIR="$PIPELINE_ROOT/status"
mkdir -p "$LOG_DIR" "$PID_DIR" "$STATUS_DIR"

export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# --- Tuning speed overrides ---
TUNING_SIM_TIME="${TUNING_SIM_TIME:-150}"
TUNING_EPISODES="${TUNING_EPISODES:-500}"
N_TRIALS="${N_TRIALS:-24}"

# --- Parallelism (tuned from profiling: env.step = 97-99% of wall-clock) ---
# Phase 1: Non-GNN MARL algos (IQL, VDN, QMIX, MAPPO)
#   Profiled: IQL env.step = 98.9%, NN = 1.1%. GPU essentially idle.
#   Each trial is single-core (Python env sim). With 96 cores:
#   4 algos × 20 trials = 80 concurrent processes → 80 of 96 cores utilized.
#   Memory: ~1GB per process × 80 = ~80GB (well within 1.5TB RAM).
PHASE1_JOBS_PER_ALGO="${PHASE1_JOBS_PER_ALGO:-20}"

# Phase 2: MAGAT-D3QN (GNN)
#   Profiled: env.step = 96.8%, GNN forward = 2.5%. Still CPU-bound!
#   Measured: 1,204 MB VRAM per trial → 20 × 1,200 = 24,000 MB (73% of V100-32GB).
#   CPU is the real constraint: 20 trials × 1 core = 20 cores.
PHASE2_JOBS="${PHASE2_JOBS:-20}"

# --- Algorithm lists ---
PHASE1_ALGOS="iql vdn qmix mappo"
PHASE2_ALGOS="magat_d3qn"

COMBINED_ROOT="$PIPELINE_ROOT/final_combined"
COMBINED_CHECKPOINT_DIR="$COMBINED_ROOT/checkpoints"
COMBINED_MANIFEST="$PIPELINE_ROOT/combined_runs_manifest.jsonl"
mkdir -p "$COMBINED_CHECKPOINT_DIR" "$COMBINED_ROOT"

echo "============================================================"
echo "  MARL Recovery Pipeline (Parallel)"
echo "  Phase 1: $PHASE1_ALGOS (${PHASE1_JOBS_PER_ALGO} concurrent trials each)"
echo "  Phase 2: $PHASE2_ALGOS (${PHASE2_JOBS} concurrent GPU trials)"
echo "  tuning_sim_time: ${TUNING_SIM_TIME}s  tuning_episodes: ${TUNING_EPISODES}"
echo "  n_trials: $N_TRIALS"
echo "  started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ======================================================================
# Helper: run tuning + eval for one algorithm
# ======================================================================
run_algo() {
    local algo="$1"
    local n_jobs="$2"
    local gpu_trials="$3"
    local MAX_RETRIES="${MAX_RETRIES:-3}"
    local RETRY_BACKOFF="${RETRY_BACKOFF:-60}"

    echo "  [$algo] Starting (n_jobs=$n_jobs, gpu_trials=$gpu_trials, max_retries=$MAX_RETRIES) — $(date '+%H:%M:%S')"

    # --- Step 1: Optuna tuning (with retry) ---
    local tuning_ok=false
    for attempt in $(seq 1 "$MAX_RETRIES"); do
        printf "RUNNING:attempt_%s\n" "$attempt" > "$STATUS_DIR/optuna_${algo}.status"
        echo "  [$algo] Tuning attempt $attempt/$MAX_RETRIES — $(date '+%H:%M:%S')"

        "$PYTHON_BIN" experiments/run_optuna_until_target.py \
            --algo "$algo" \
            --study-name "$STUDY_NAME" \
            --target-complete-trials "$N_TRIALS" \
            --force-retrain \
            --n-jobs "$n_jobs" \
            --max-concurrent-gpu-trials "$gpu_trials" \
            --tuning-sim-time "$TUNING_SIM_TIME" \
            --tuning-episodes "$TUNING_EPISODES" \
            --skip-ablations \
            >> "$LOG_DIR/optuna_${algo}.log" 2>&1
        local rc=$?

        if [[ $rc -eq 0 ]]; then
            tuning_ok=true
            printf "COMPLETED\n" > "$STATUS_DIR/optuna_${algo}.status"
            echo "  [$algo] Tuning COMPLETED on attempt $attempt"
            break
        fi

        echo "  [$algo] Tuning FAILED (rc=$rc) on attempt $attempt. " >&2
        if [[ $attempt -lt $MAX_RETRIES ]]; then
            echo "  [$algo] Retrying in ${RETRY_BACKOFF}s..." >&2
            sleep "$RETRY_BACKOFF"
        fi
    done

    if [[ "$tuning_ok" != "true" ]]; then
        printf "FAILED:tuning_exhausted\n" > "$STATUS_DIR/optuna_${algo}.status"
        echo "  [$algo] TUNING FAILED after $MAX_RETRIES attempts" >&2
        return 1
    fi

    local study_dir="results/optuna/${STUDY_NAME}_${algo}"
    if [[ ! -d "$study_dir" ]]; then
        echo "  [$algo] study_dir missing: $study_dir" >&2
        return 1
    fi

    # --- Step 2: Final unified eval (with retry) ---
    local artifact_json="$PIPELINE_ROOT/final_artifacts_${algo}.json"
    local eval_ok=false
    for attempt in $(seq 1 "$MAX_RETRIES"); do
        printf "RUNNING:final_attempt_%s\n" "$attempt" > "$STATUS_DIR/final_unified_${algo}.status"
        echo "  [$algo] Final eval attempt $attempt/$MAX_RETRIES — $(date '+%H:%M:%S')"

        "$PYTHON_BIN" experiments/run_best_optuna_pipeline.py \
            --study-dir "$study_dir" \
            --representative-label "$REPRESENTATIVE_LABEL" \
            --artifact-json "$artifact_json" \
            --force-retrain \
            >> "$LOG_DIR/final_unified_${algo}.log" 2>&1
        rc=$?

        if [[ $rc -eq 0 ]]; then
            eval_ok=true
            printf "COMPLETED\n" > "$STATUS_DIR/final_unified_${algo}.status"
            echo "  [$algo] Final eval COMPLETED on attempt $attempt"
            break
        fi

        echo "  [$algo] Final eval FAILED (rc=$rc) on attempt $attempt" >&2
        if [[ $attempt -lt $MAX_RETRIES ]]; then
            echo "  [$algo] Retrying in ${RETRY_BACKOFF}s..." >&2
            sleep "$RETRY_BACKOFF"
        fi
    done

    if [[ "$eval_ok" != "true" ]]; then
        printf "FAILED:eval_exhausted\n" > "$STATUS_DIR/final_unified_${algo}.status"
        echo "  [$algo] FINAL EVAL FAILED after $MAX_RETRIES attempts" >&2
        return 1
    fi

    # --- Step 3: Merge checkpoints ---
    if [[ -f "$artifact_json" ]]; then
        local ckpt_dir
        ckpt_dir="$("$PYTHON_BIN" -c "
import json
with open('$artifact_json') as f:
    print(json.load(f)['artifacts']['checkpoint_dir'])
" 2>/dev/null || echo "")"

        local out_d
        out_d="$("$PYTHON_BIN" -c "
import json
with open('$artifact_json') as f:
    print(json.load(f)['artifacts']['out_dir'])
" 2>/dev/null || echo "")"

        if [[ -n "$ckpt_dir" && -d "$ckpt_dir" ]]; then
            mkdir -p "$COMBINED_ROOT/runs/$algo"
            cp "$artifact_json" "$COMBINED_ROOT/runs/$algo/artifacts.json"
            cp -r "$ckpt_dir"/. "$COMBINED_CHECKPOINT_DIR"/
            printf '{"algo":"%s","study_dir":"%s","out_dir":"%s","checkpoint_dir":"%s"}\n' \
                "$algo" "$study_dir" "$out_d" "$ckpt_dir" >> "$COMBINED_MANIFEST"
        fi
    fi

    echo "  [$algo] DONE — $(date '+%H:%M:%S')"
    return 0
}

# ======================================================================
# PHASE 1: Non-GNN MARL algos in PARALLEL
# ======================================================================
# IQL, VDN, QMIX are value-decomposition MLPs — CPU-dominated training.
# MAPPO is actor-critic MLP — also CPU-dominated.
# Train on CPU to free GPU for Phase 2; use n_jobs for intra-algo concurrency.
# ======================================================================
echo ""
echo "============================================================"
echo "  PHASE 1: Non-GNN MARL (parallel) — $(date '+%H:%M:%S')"
echo "============================================================"

PHASE1_PIDS=()
PHASE1_ALGOS_ARR=($PHASE1_ALGOS)

for algo in "${PHASE1_ALGOS_ARR[@]}"; do
    (
        # Force CPU for non-GNN algos to avoid GPU contention
        export CUDA_VISIBLE_DEVICES=""
        run_algo "$algo" "$PHASE1_JOBS_PER_ALGO" "$PHASE1_JOBS_PER_ALGO"
    ) &
    PHASE1_PIDS+=($!)
    echo "  Launched $algo (PID=$!)"
done

# Wait for all Phase 1 to complete
PHASE1_FAILED=""
for i in "${!PHASE1_ALGOS_ARR[@]}"; do
    algo="${PHASE1_ALGOS_ARR[$i]}"
    pid="${PHASE1_PIDS[$i]}"
    wait "$pid" || {
        echo "  WARNING: $algo failed (PID=$pid)" >&2
        PHASE1_FAILED="$PHASE1_FAILED $algo"
    }
done

echo ""
echo "  Phase 1 complete — $(date '+%H:%M:%S')"
if [[ -n "$PHASE1_FAILED" ]]; then
    echo "  Phase 1 failures:$PHASE1_FAILED"
fi

# ======================================================================
# PHASE 2: GNN MARL (MAGAT-D3QN) — GPU required
# ======================================================================
echo ""
echo "============================================================"
echo "  PHASE 2: MAGAT-D3QN (GPU, ${PHASE2_JOBS} concurrent trials)"
echo "  — $(date '+%H:%M:%S')"
echo "============================================================"

PHASE2_FAILED=""
for algo in $PHASE2_ALGOS; do
    run_algo "$algo" "$PHASE2_JOBS" "$PHASE2_JOBS" || {
        PHASE2_FAILED="$PHASE2_FAILED $algo"
    }
done

# ======================================================================
# Summary
# ======================================================================
ALL_FAILED="${PHASE1_FAILED}${PHASE2_FAILED}"

echo ""
echo "============================================================"
echo "  MARL Recovery Complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Phase 1 (CPU parallel):  $PHASE1_ALGOS"
echo "  Phase 2 (GPU):           $PHASE2_ALGOS"
if [[ -n "$ALL_FAILED" ]]; then
    echo "  FAILED:${ALL_FAILED}"
fi
echo "============================================================"

if [[ -n "$ALL_FAILED" ]]; then
    echo ""
    echo "Failed algorithm logs:"
    for algo in $ALL_FAILED; do
        echo "  $LOG_DIR/optuna_${algo}.log"
        echo "  $LOG_DIR/final_unified_${algo}.log"
    done
    exit 1
fi
