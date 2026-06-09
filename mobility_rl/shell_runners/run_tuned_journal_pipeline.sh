#!/bin/bash
# ==============================================================================
# Tuned Journal Pipeline — Skip Optuna, Use Existing Best Weights
# ==============================================================================
# Reads the best trial config from existing Optuna study directories
# (representative_pareto_points.csv), retrains with those weights, and
# launches journal evidence suites.
#
# This is a LEAN alternative to launch_optuna_then_full_pipeline.sh:
#   - NO Optuna step (assumes tuning is already complete)
#   - Directly calls run_best_optuna_pipeline.py per algorithm
#   - Launches journal evidence suites on the resulting checkpoints
#
# Usage:
#   ALGOS=mappo,vdn bash run_tuned_journal_pipeline.sh
#
# Optional env vars:
#   ALGOS=mappo,vdn         (comma-separated list of algos to run)
#   STUDY_NAME=journal_tune (must match the existing Optuna study)
#   REPRESENTATIVE_LABEL=balanced_min_L2_to_ideal
#   FORCE_RETRAIN=1
#   DRY_RUN=0
# ==============================================================================

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ALGOS="${ALGOS:?'ALGOS env var required (e.g., ALGOS=mappo,vdn)'}"
STUDY_NAME="${STUDY_NAME:-journal_tune}"
REPRESENTATIVE_LABEL="${REPRESENTATIVE_LABEL:-balanced_min_L2_to_ideal}"
FORCE_RETRAIN="${FORCE_RETRAIN:-1}"
DRY_RUN="${DRY_RUN:-0}"

PIPELINE_ROOT="${PIPELINE_ROOT:-results/tuned_journal_pipeline/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$PIPELINE_ROOT"
LOG_DIR="$PIPELINE_ROOT/logs"
mkdir -p "$LOG_DIR"

export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# Parse algo list
ALGO_LIST="$(echo "$ALGOS" | tr ',' ' ')"

COMBINED_ROOT="$PIPELINE_ROOT/final_combined"
COMBINED_CHECKPOINT_DIR="$COMBINED_ROOT/checkpoints"
COMBINED_MANIFEST="$PIPELINE_ROOT/combined_runs_manifest.jsonl"
mkdir -p "$COMBINED_CHECKPOINT_DIR" "$COMBINED_ROOT"
touch "$COMBINED_MANIFEST"

echo "============================================================"
echo "  Tuned Journal Pipeline (Optuna-Free)"
echo "  algos=$ALGO_LIST"
echo "  study_name=$STUDY_NAME"
echo "  representative=$REPRESENTATIVE_LABEL"
echo "  pipeline_root=$PIPELINE_ROOT"
echo "============================================================"

# ==============================================================================
# Step 1: Verify all study dirs exist BEFORE training
# ==============================================================================
echo ""
echo "Step 1: Verifying existing Optuna study directories..."
for CURRENT_ALGO in $ALGO_LIST; do
    STUDY_DIR="results/optuna/${STUDY_NAME}_${CURRENT_ALGO}"
    PARETO_CSV="$STUDY_DIR/representative_pareto_points.csv"

    if [[ ! -d "$STUDY_DIR" ]]; then
        echo "  ERROR: Study dir not found: $STUDY_DIR"
        echo "  Optuna tuning must be completed first."
        exit 1
    fi
    if [[ ! -f "$PARETO_CSV" ]]; then
        echo "  ERROR: Pareto CSV not found: $PARETO_CSV"
        echo "  Run post-study analysis to generate it."
        exit 1
    fi

    # Show what trial will be selected
    echo "  ✓ $CURRENT_ALGO: $PARETO_CSV"
    "$PYTHON_BIN" -c "
import pandas as pd, sys
df = pd.read_csv('$PARETO_CSV')
sel = df[df['representative_label'] == '$REPRESENTATIVE_LABEL']
if sel.empty:
    print('    WARNING: label \"$REPRESENTATIVE_LABEL\" not found. Available:', list(df['representative_label'].unique()))
    sys.exit(1)
row = sel.iloc[0]
tn = int(row['trial_number'])
print(f'    Selected trial #{tn}')
# Show key metrics from the row
for col in ['obj_throughput', 'obj_delay', 'obj_drops', 'obj_collisions', 'wT', 'wD', 'wDrop', 'wCol']:
    if col in row.index:
        print(f'      {col}: {row[col]}')
"
done
echo ""

# ==============================================================================
# Step 2: Final unified training + evaluation with best config
# ==============================================================================
echo "Step 2: Final training & evaluation with best Optuna config..."
LAST_STUDY_DIR=""

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

for CURRENT_ALGO in $ALGO_LIST; do
    echo "------------------------------------------------------------"
    echo "Algorithm: $CURRENT_ALGO"

    STUDY_DIR="results/optuna/${STUDY_NAME}_${CURRENT_ALGO}"
    LAST_STUDY_DIR="$STUDY_DIR"

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

    echo "  Running: $PYTHON_BIN experiments/run_best_optuna_pipeline.py ${BEST_ARGS[*]}"
    "$PYTHON_BIN" experiments/run_best_optuna_pipeline.py "${BEST_ARGS[@]}" \
        2>&1 | tee "$LOG_DIR/final_unified_${CURRENT_ALGO}.log"

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

    echo "  ✓ $CURRENT_ALGO complete."
done

CHECKPOINT_DIR="$COMBINED_CHECKPOINT_DIR"
OUT_DIR="$COMBINED_ROOT"
export CHECKPOINT_DIR OUT_DIR

# Generate combined training reward plots
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

echo ""
echo "Step 2 complete. Checkpoints: $CHECKPOINT_DIR"
echo ""

# ==============================================================================
# Step 3: Launch journal evidence suites
# ==============================================================================
echo "Step 3: Launching journal evidence suites..."
export RESULT_ROOT="$PIPELINE_ROOT/journal_suites"
export OMP_THREADS_PER_WORKER="${OMP_THREADS_PER_WORKER:-2}"
export RUN_MAGAT_ISOLATION="${RUN_MAGAT_ISOLATION:-0}"
if [[ -n "${MAX_PARALLEL:-}" ]]; then
    export MAX_PARALLEL
fi

bash launch_parallel_journal_experiments.sh "$CHECKPOINT_DIR"

echo ""
echo "============================================================"
echo "  Tuned Journal Pipeline Complete"
echo "  Pipeline root: $PIPELINE_ROOT"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Journal suites: $RESULT_ROOT"
echo "  Monitor:"
echo "    bash monitor_pipeline.sh $PIPELINE_ROOT"
echo "  Finalize:"
echo "    bash finalize_journal_package.sh $PIPELINE_ROOT"
echo "============================================================"
