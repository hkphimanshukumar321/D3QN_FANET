#!/bin/bash
# ==============================================================================
# Direct Journal Suites — Assemble best checkpoints, skip all training
# ==============================================================================
# Reads best trial configs from Optuna Pareto CSVs, finds the matching
# checkpoints from previous runs, assembles them, and launches journal suites.
#
# Usage:
#   bash run_journal_direct.sh
#   nohup bash run_journal_direct.sh > logs/journal_direct.log 2>&1 &
#
# Optional env vars:
#   STUDY_NAME=journal_tune
#   CHECKPOINT_SEARCH_ROOT=results/N50_PHY5M_Q100_RTSCTS_ACK_enabled
# ==============================================================================

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
STUDY_NAME="${STUDY_NAME:-journal_tune}"
SEARCH_ROOT="${CHECKPOINT_SEARCH_ROOT:-results/N50_PHY5M_Q100_RTSCTS_ACK_enabled}"
TS="$(date +%Y%m%d_%H%M%S)"
COMBINED_DIR="results/journal_best_checkpoints_${TS}"

export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

mkdir -p "$COMBINED_DIR" "logs"

echo "============================================================"
echo "  Direct Journal Pipeline"
echo "  study=$STUDY_NAME"
echo "  search_root=$SEARCH_ROOT"
echo "  combined_dir=$COMBINED_DIR"
echo "============================================================"

# ==============================================================================
# Step 1: Read best trial configs and display weights
# ==============================================================================
echo ""
echo "Step 1: Reading best trial configs from Optuna Pareto CSVs..."

"$PYTHON_BIN" - <<'PYEOF'
import pandas as pd
import json
import os
import sys

study_name = os.environ.get("STUDY_NAME", "journal_tune")

for algo in ["mappo", "vdn"]:
    csv_path = f"results/optuna/{study_name}_{algo}/representative_pareto_points.csv"
    if not os.path.exists(csv_path):
        print(f"  SKIP {algo}: {csv_path} not found")
        continue

    df = pd.read_csv(csv_path)
    sel = df[df["representative_label"] == "balanced_min_L2_to_ideal"]
    if sel.empty:
        print(f"  WARN {algo}: balanced_min_L2_to_ideal not found")
        continue

    row = sel.iloc[0]
    trial_num = int(row["trial_number"])
    print(f"\n  {algo.upper()} — Best Trial #{trial_num}")

    # Load config
    cfg_path = f"results/optuna/{study_name}_{algo}/trial_{trial_num}/config.json"
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        hp = cfg.get("hyperparameters", {})
        rw = cfg.get("reward_weights", {})
        print(f"    lr={hp.get('lr')}, hidden_dim={hp.get('hidden_dim')}, batch_size={hp.get('batch_size')}")
        print(f"    wT={rw.get('wT')}, wD={rw.get('wD')}, wDrop={rw.get('wDrop')}, wCol={rw.get('wCol')}")
    else:
        print(f"    Config: {cfg_path} NOT FOUND")

    for col in ["obj_throughput", "obj_delay", "obj_drops", "obj_collisions"]:
        if col in row.index:
            print(f"    {col}: {row[col]}")
PYEOF

# ==============================================================================
# Step 2: Find and assemble the best checkpoints
# ==============================================================================
echo ""
echo "Step 2: Assembling best checkpoints from $SEARCH_ROOT..."

# For each model type, find the MOST RECENT checkpoint (newest = trained with
# the latest config from the pipeline that just ran)
MODEL_PATTERNS=(
    "unified_mappo_model.pth"
    "unified_vdn_model.pth"
    "unified_iql_model.pth"
    "unified_qmix_model.pth"
    "unified_magat_d3qn_model.pth"
)

FOUND_COUNT=0
for pattern in "${MODEL_PATTERNS[@]}"; do
    # Find the most recently modified file matching this pattern
    latest=$(find "$SEARCH_ROOT" -name "$pattern" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
    if [[ -n "$latest" ]]; then
        cp "$latest" "$COMBINED_DIR/"
        FOUND_COUNT=$((FOUND_COUNT + 1))
        # Show which trial dir it came from
        trial_dir=$(dirname "$(dirname "$latest")")
        echo "  ✓ $pattern"
        echo "    <- $trial_dir"
    else
        echo "  ✗ $pattern (not found)"
    fi
done

# Also grab SARL baselines if they exist
for sarl_pattern in "sarl_dqn_model.zip" "sarl_ppo_model.zip" "sarl_a2c_model.zip" "sarl_custom_mca_model.pth"; do
    latest=$(find "$SEARCH_ROOT" -name "$sarl_pattern" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
    if [[ -n "$latest" ]]; then
        cp "$latest" "$COMBINED_DIR/"
        FOUND_COUNT=$((FOUND_COUNT + 1))
        echo "  ✓ $sarl_pattern (SARL baseline)"
    fi
done

# Copy any training reward CSVs for reference (one per algo, newest wins)
find "$SEARCH_ROOT" -name "*_training_rewards.csv" -printf '%T@ %p\n' 2>/dev/null \
  | sort -rn | while IFS=' ' read -r _ts fpath; do
    [ -z "$fpath" ] && continue
    base=$(basename "$fpath")
    if [[ ! -f "$COMBINED_DIR/$base" ]]; then
        cp "$fpath" "$COMBINED_DIR/" 2>/dev/null || true
    fi
done

echo ""
echo "Assembled $FOUND_COUNT model files:"
ls -lh "$COMBINED_DIR/" 2>/dev/null || true
echo ""

if [[ "$FOUND_COUNT" -lt 2 ]]; then
    echo "ERROR: Found fewer than 2 model files. Cannot proceed."
    echo "Check that training has completed and checkpoints exist in $SEARCH_ROOT"
    exit 1
fi

# ==============================================================================
# Step 3: Launch journal evidence suites
# ==============================================================================
echo "Step 3: Launching journal evidence suites..."
echo ""

export RESULT_ROOT="results/journal_evidence_${TS}"
export AUTO_NOHUP=0
export WANDB_DISABLED="${WANDB_DISABLED:-1}"

bash launch_parallel_journal_experiments.sh "$COMBINED_DIR"

echo ""
echo "============================================================"
echo "  Journal Pipeline Complete"
echo "  Checkpoints: $COMBINED_DIR"
echo "  Results: $RESULT_ROOT"
echo "  Monitor: bash monitor_pipeline.sh $RESULT_ROOT"
echo "============================================================"
