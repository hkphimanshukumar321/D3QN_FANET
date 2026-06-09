#!/bin/bash
# ==============================================================================
# Optuna -> Final Pipeline Launcher
# ==============================================================================
# 1. Runs Optuna for one target algorithm.
# 2. Selects a representative Pareto trial (balanced by default).
# 3. Runs the final unified training/evaluation with that selected config.
# 4. Launches the parallel journal evidence suites on the resulting checkpoints.
#
# Usage:
#   ALGO=magat_d3qn bash launch_optuna_then_full_pipeline.sh
#
# Optional env vars:
#   ALGO=magat_d3qn
#   ALGOS=dqn,ppo,a2c,mca_d3qn,iql,vdn,qmix,magat_d3qn
#   STUDY_NAME=journal_tune
#   REPRESENTATIVE_LABEL=balanced_min_L2_to_ideal
#   TRIAL_NUMBER=12
#   N_TRIALS=24
#   DRY_RUN=0
#   FORCE_RETRAIN=1
#   FINAL_ALL_ALGOS=0
#   WANDB_API_KEY=...
#   WANDB_DISABLED=1
# ==============================================================================

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ALGO="${ALGO:-magat_d3qn}"
ALGOS="${ALGOS:-}"
STUDY_NAME="${STUDY_NAME:-journal_tune}"
REPRESENTATIVE_LABEL="${REPRESENTATIVE_LABEL:-balanced_min_L2_to_ideal}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-1}"
FINAL_ALL_ALGOS="${FINAL_ALL_ALGOS:-0}"
PIPELINE_ROOT="${PIPELINE_ROOT:-results/optuna_then_pipeline/$(date +%Y%m%d_%H%M%S)}"
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
    echo "Detached Optuna -> pipeline launcher."
    echo "  pid=$LAUNCHER_PID"
    echo "  root=$PIPELINE_ROOT"
    echo "  log=$LOG_DIR/launcher.log"
    exit 0
fi

export DRY_RUN
export FORCE_RETRAIN
export FINAL_ALL_ALGOS
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

resolve_algos() {
    local raw="$1"
    if [[ -z "$raw" ]]; then
        raw="$ALGO"
    fi
    if [[ "$raw" == "all" || "$raw" == "paper" ]]; then
        echo "ppo a2c mca_d3qn iql vdn qmix magat_d3qn"
        return
    fi
    if [[ "$raw" == "all_with_tabular" ]]; then
        echo "tabular dqn ppo a2c mca_d3qn iql vdn qmix magat_d3qn"
        return
    fi
    echo "$raw" | tr ',' ' '
}

filter_supported_algos() {
    local filtered=()
    local warned=0
    for algo in "$@"; do
        if [[ "$algo" == "tabular" || "$algo" == "dqn" ]]; then
            warned=1
            continue
        fi
        filtered+=("$algo")
    done
    if [[ "$warned" -eq 1 ]]; then
        echo "Unsupported algorithms were excluded from the tuned final pipeline." >&2
        echo "Excluded: tabular, dqn" >&2
        echo "Reason: they do not support the current centralized MultiDiscrete burst-action baseline." >&2
    fi
    printf "%s\n" "${filtered[@]}"
}

if [[ -f ".env.wandb" ]]; then
    # shellcheck disable=SC1091
    source ".env.wandb"
elif [[ -f ".env.wandb.local" ]]; then
    # shellcheck disable=SC1091
    source ".env.wandb.local"
fi

COMMON_OPTUNA_ARGS=(
    --study-name "$STUDY_NAME"
    --force-retrain
)

if [[ -n "${N_TRIALS:-}" ]]; then
    COMMON_OPTUNA_ARGS+=(--target-complete-trials "$N_TRIALS")
fi
if [[ "${INCLUDE_INFERENCE_OBJECTIVE:-0}" == "1" ]]; then
    COMMON_OPTUNA_ARGS+=(--include-inference)
fi
if [[ -n "${OPTUNA_OBJECTIVES:-}" ]]; then
    # shellcheck disable=SC2206
    OBJECTIVE_WORDS=($OPTUNA_OBJECTIVES)
    COMMON_OPTUNA_ARGS+=(--objectives "${OBJECTIVE_WORDS[@]}")
fi
if [[ "$DRY_RUN" == "1" ]]; then
    COMMON_OPTUNA_ARGS+=(--dry-run)
fi
if [[ -n "${PHY_RATE_MBPS:-}" ]]; then
    COMMON_OPTUNA_ARGS+=(--phy-rate-mbps "$PHY_RATE_MBPS")
fi
if [[ -n "${NODES:-}" ]]; then
    COMMON_OPTUNA_ARGS+=(--nodes "$NODES")
fi
if [[ -n "${QMAX:-}" ]]; then
    COMMON_OPTUNA_ARGS+=(--qmax "$QMAX")
fi
if [[ -n "${SWEEP_MIN_PPS:-}" ]]; then
    COMMON_OPTUNA_ARGS+=(--sweep-min-pps "$SWEEP_MIN_PPS")
fi
if [[ -n "${SWEEP_MAX_PPS:-}" ]]; then
    COMMON_OPTUNA_ARGS+=(--sweep-max-pps "$SWEEP_MAX_PPS")
fi
if [[ -n "${SWEEP_STEPS:-}" ]]; then
    COMMON_OPTUNA_ARGS+=(--sweep-steps "$SWEEP_STEPS")
fi
if [[ -n "${SEED:-}" ]]; then
    COMMON_OPTUNA_ARGS+=(--seed "$SEED")
fi
if [[ -n "${OPTUNA_N_JOBS:-}" ]]; then
    COMMON_OPTUNA_ARGS+=(--n-jobs "$OPTUNA_N_JOBS" --max-concurrent-gpu-trials "$OPTUNA_N_JOBS")
fi

COMMON_FINAL_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
    COMMON_FINAL_ARGS+=(--dry-run)
fi
if [[ "$FORCE_RETRAIN" == "1" ]]; then
    COMMON_FINAL_ARGS+=(--force-retrain)
fi
if [[ -n "${PHY_RATE_MBPS:-}" ]]; then
    COMMON_FINAL_ARGS+=(--phy-rate-mbps "$PHY_RATE_MBPS")
fi
if [[ -n "${NODES:-}" ]]; then
    COMMON_FINAL_ARGS+=(--nodes "$NODES")
fi
if [[ -n "${QMAX:-}" ]]; then
    COMMON_FINAL_ARGS+=(--qmax "$QMAX")
fi
if [[ -n "${SWEEP_MIN_PPS:-}" ]]; then
    COMMON_FINAL_ARGS+=(--sweep-min-pps "$SWEEP_MIN_PPS")
fi
if [[ -n "${SWEEP_MAX_PPS:-}" ]]; then
    COMMON_FINAL_ARGS+=(--sweep-max-pps "$SWEEP_MAX_PPS")
fi
if [[ -n "${SWEEP_STEPS:-}" ]]; then
    COMMON_FINAL_ARGS+=(--sweep-steps "$SWEEP_STEPS")
fi

RESOLVED_ALGOS="$(resolve_algos "$ALGOS")"
# shellcheck disable=SC2206
RESOLVED_ARRAY=($RESOLVED_ALGOS)
SUPPORTED_ALGOS="$(filter_supported_algos "${RESOLVED_ARRAY[@]}")"
ALGO_LIST="$(echo "$SUPPORTED_ALGOS" | tr '\n' ' ' | xargs)"
if [[ -z "$ALGO_LIST" ]]; then
    echo "No supported algorithms left after filtering."
    exit 1
fi
export ALGO_LIST_ENV="$ALGO_LIST"
COMBINED_ROOT="$PIPELINE_ROOT/final_combined"
COMBINED_CHECKPOINT_DIR="$COMBINED_ROOT/checkpoints"
COMBINED_MANIFEST="$PIPELINE_ROOT/combined_runs_manifest.jsonl"
mkdir -p "$COMBINED_CHECKPOINT_DIR" "$COMBINED_ROOT"
touch "$COMBINED_MANIFEST"

if [[ "$(echo "$ALGO_LIST" | wc -w)" -gt 1 && "$FINAL_ALL_ALGOS" == "1" ]]; then
    echo "FINAL_ALL_ALGOS=1 is redundant when running multiple tuned algorithms."
    echo "Each algorithm is already retrained with its own selected Optuna config."
    FINAL_ALL_ALGOS=0
fi

echo "============================================================"
echo "  Optuna -> Final Pipeline"
echo "  algos=$ALGO_LIST"
echo "  study_name=$STUDY_NAME"
echo "  pipeline_root=$PIPELINE_ROOT"
echo "============================================================"
echo "Step 1/3: Optuna + tuned final runs"

LAST_STUDY_DIR=""
for CURRENT_ALGO in $ALGO_LIST; do
    echo "------------------------------------------------------------"
    echo "Algorithm: $CURRENT_ALGO"

    OPTUNA_ARGS=(--algo "$CURRENT_ALGO" "${COMMON_OPTUNA_ARGS[@]}")
    run_stage "optuna_${CURRENT_ALGO}" \
        "$PYTHON_BIN" experiments/run_optuna_until_target.py "${OPTUNA_ARGS[@]}"

    STUDY_DIR="results/optuna/${STUDY_NAME}_${CURRENT_ALGO}"
    LAST_STUDY_DIR="$STUDY_DIR"
    if [[ ! -d "$STUDY_DIR" ]]; then
        echo "Expected study dir not found: $STUDY_DIR"
        exit 1
    fi

    if [[ -f "$STUDY_DIR/target_trials_status.json" ]]; then
        cp "$STUDY_DIR/target_trials_status.json" "$PIPELINE_ROOT/target_trials_status_${CURRENT_ALGO}.json"
        cp "$STUDY_DIR/target_trials_status.json" "$PIPELINE_ROOT/target_trials_status.json"
    fi
    if [[ -f "$STUDY_DIR/study_execution_summary.json" ]]; then
        cp "$STUDY_DIR/study_execution_summary.json" "$PIPELINE_ROOT/study_execution_summary_${CURRENT_ALGO}.json"
        cp "$STUDY_DIR/study_execution_summary.json" "$PIPELINE_ROOT/study_execution_summary.json"
    fi

    ARTIFACT_JSON="$PIPELINE_ROOT/final_artifacts_${CURRENT_ALGO}.json"
    export ARTIFACT_JSON

    BEST_ARGS=(
        --study-dir "$STUDY_DIR"
        --representative-label "$REPRESENTATIVE_LABEL"
        --artifact-json "$ARTIFACT_JSON"
        "${COMMON_FINAL_ARGS[@]}"
    )
    if [[ -n "${TRIAL_NUMBER:-}" ]]; then
        BEST_ARGS+=(--trial-number "$TRIAL_NUMBER")
    fi
    if [[ "$FINAL_ALL_ALGOS" == "1" ]]; then
        BEST_ARGS+=(--final-all-algos)
    fi

    run_stage "final_unified_${CURRENT_ALGO}" \
        "$PYTHON_BIN" experiments/run_best_optuna_pipeline.py "${BEST_ARGS[@]}"

    if [[ ! -f "$ARTIFACT_JSON" ]]; then
        echo "Final artifact JSON not created: $ARTIFACT_JSON"
        exit 1
    fi

    CURRENT_CHECKPOINT_DIR="$("$PYTHON_BIN" - <<'PY'
import json, os
path = os.path.abspath(os.environ["ARTIFACT_JSON"])
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(data["artifacts"]["checkpoint_dir"])
PY
)"

    CURRENT_OUT_DIR="$("$PYTHON_BIN" - <<'PY'
import json, os
path = os.path.abspath(os.environ["ARTIFACT_JSON"])
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(data["artifacts"]["out_dir"])
PY
)"

    mkdir -p "$COMBINED_ROOT/runs/$CURRENT_ALGO"
    cp "$ARTIFACT_JSON" "$COMBINED_ROOT/runs/$CURRENT_ALGO/artifacts.json"
    cp -r "$CURRENT_CHECKPOINT_DIR"/. "$COMBINED_CHECKPOINT_DIR"/
    printf '{"algo":"%s","study_dir":"%s","out_dir":"%s","checkpoint_dir":"%s"}\n' \
        "$CURRENT_ALGO" "$STUDY_DIR" "$CURRENT_OUT_DIR" "$CURRENT_CHECKPOINT_DIR" >> "$COMBINED_MANIFEST"
done

CHECKPOINT_DIR="$COMBINED_CHECKPOINT_DIR"
OUT_DIR="$COMBINED_ROOT"
FINAL_ARTIFACTS_JSON="$PIPELINE_ROOT/final_artifacts.json"
export CHECKPOINT_DIR OUT_DIR FINAL_ARTIFACTS_JSON COMBINED_MANIFEST

"$PYTHON_BIN" - <<'PY'
import os, sys
project_root = os.path.abspath(".")
sys.path.insert(0, project_root)
from experiments.plot_training_rewards import generate_plots
generate_plots(
    csv_dir=os.path.abspath(os.environ["CHECKPOINT_DIR"]),
    images_dir=os.path.join(os.path.abspath(os.environ["OUT_DIR"]), "images"),
)
PY

"$PYTHON_BIN" - <<'PY'
import json, os
with open(os.path.abspath(os.environ["FINAL_ARTIFACTS_JSON"]), "w", encoding="utf-8") as f:
    json.dump(
        {
            "artifacts": {
                "checkpoint_dir": os.path.abspath(os.environ["CHECKPOINT_DIR"]),
                "out_dir": os.path.abspath(os.environ["OUT_DIR"]),
                "combined_manifest": os.path.abspath(os.environ["COMBINED_MANIFEST"]),
            },
            "execution_mode": {
                "marl_env": "MARLMacEnv",
                "marl_control": "decentralized_cluster_head",
                "sarl_env": "SARLCentralEnv",
                "burst_split_action": "(m_k, rho_k)",
                "tuned_algorithms": os.environ.get("ALGO_LIST_ENV", "").split(),
            },
        },
        f,
        indent=2,
    )
PY

echo "Step 3/3: launch journal evidence suites"
export RESULT_ROOT="$PIPELINE_ROOT/journal_suites"
export OMP_THREADS_PER_WORKER="${OMP_THREADS_PER_WORKER:-2}"
export RUN_MAGAT_ISOLATION="${RUN_MAGAT_ISOLATION:-0}"
if [[ -n "${MAX_PARALLEL:-}" ]]; then
    export MAX_PARALLEL
fi

bash launch_parallel_journal_experiments.sh "$CHECKPOINT_DIR"

echo ""
echo "============================================================"
echo "Optuna study dir:"
echo "  $LAST_STUDY_DIR"
echo "Final unified run:"
echo "  $OUT_DIR"
echo "Final checkpoints:"
echo "  $CHECKPOINT_DIR"
echo "Parallel journal suites:"
echo "  $RESULT_ROOT"
echo "Monitor:"
echo "  bash monitor_pipeline.sh $PIPELINE_ROOT"
echo "Finalize package:"
echo "  bash finalize_journal_package.sh $PIPELINE_ROOT"
echo "============================================================"
