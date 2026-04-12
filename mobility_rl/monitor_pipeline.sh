#!/bin/bash
# ==============================================================================
# Pipeline Monitor
# ==============================================================================
# Monitors the long-running journal/Optuna pipeline jobs by checking:
# - pid files and liveness
# - status files
# - key artifact files
# - recent log error signatures
#
# Usage:
#   bash monitor_pipeline.sh <pipeline_root>
#   bash monitor_pipeline.sh <pipeline_root> --once
#
# Optional env vars:
#   REFRESH_SECS=20
#   ERROR_TAIL_LINES=200
# ==============================================================================

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <pipeline_root> [--once]"
    exit 1
fi

PIPELINE_ROOT="$1"
MODE="${2:-}"
REFRESH_SECS="${REFRESH_SECS:-20}"
ERROR_TAIL_LINES="${ERROR_TAIL_LINES:-200}"

if [[ ! -d "$PIPELINE_ROOT" ]]; then
    echo "Pipeline root not found: $PIPELINE_ROOT"
    exit 1
fi

print_header() {
    echo "============================================================"
    echo " Pipeline Monitor"
    echo " root=$PIPELINE_ROOT"
    echo " time=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
}

check_pid_file() {
    local pid_file="$1"
    local name
    name="$(basename "$pid_file" .pid)"
    
    # Verify status file first to avoid PID recycling false-positives
    local status_file
    status_file="$(dirname "$(dirname "$pid_file")")/status/${name}.status"
    if [[ -f "$status_file" ]]; then
        local st
        st="$(head -n 1 "$status_file" 2>/dev/null || echo "MISSING")"
        if [[ "$st" == "COMPLETED" || "$st" == DETACHED* || "$st" == FAILED* ]]; then
            # If the stage is definitively done or detached, just show that status
            printf "%-32s %-10s %s\n" "$name" "($st)" "$pid_file"
            return
        fi
    fi

    local pid
    pid="$(tr -d '[:space:]' < "$pid_file" 2>/dev/null || true)"
    local status="UNKNOWN"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        status="RUNNING"
    else
        status="DEAD"
    fi
    printf "%-32s %-10s %s\n" "$name" "$status" "$pid_file"
}

check_status_file() {
    local status_file="$1"
    local name
    name="$(basename "$status_file" .status)"
    local value
    value="$(head -n 1 "$status_file" 2>/dev/null || echo "MISSING")"
    printf "%-32s %s\n" "$name" "$value"
}

scan_logs() {
    local any_issue=0
    while IFS= read -r -d '' log_file; do
        local matches
        matches="$(tail -n "$ERROR_TAIL_LINES" "$log_file" | grep -Ein "traceback|runtimeerror|exception|error:|failed|killed|oom|cuda out of memory" || true)"
        if [[ -n "$matches" ]]; then
            any_issue=1
            echo "[log-alert] $log_file"
            echo "$matches" | tail -n 5
        fi
    done < <(find "$PIPELINE_ROOT" -type f -path "*/logs/*.log" -print0 2>/dev/null)
    if [[ "$any_issue" -eq 0 ]]; then
        echo "No recent error signatures found in log tails."
    fi
}

check_artifacts() {
    local missing=0
    local candidate
    for candidate in \
        "$PIPELINE_ROOT/unified_artifacts.json" \
        "$PIPELINE_ROOT/final_artifacts.json" \
        "$PIPELINE_ROOT/journal_suites" \
        "$PIPELINE_ROOT/logs"; do
        if [[ -e "$candidate" ]]; then
            printf "[artifact-ok] %s\n" "$candidate"
        else
            printf "[artifact-missing] %s\n" "$candidate"
            missing=1
        fi
    done

    while IFS= read -r -d '' status_json; do
        printf "[status-json] %s\n" "$status_json"
    done < <(find "$PIPELINE_ROOT" -type f \( -name "target_trials_status.json" -o -name "study_execution_summary.json" \) -print0 2>/dev/null)

    return "$missing"
}

check_checkpoint_targets() {
    local json_file
    for json_file in "$PIPELINE_ROOT/unified_artifacts.json" "$PIPELINE_ROOT/final_artifacts.json"; do
        if [[ -f "$json_file" ]]; then
            python - "$json_file" <<'PY'
import json, os, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
if "checkpoint_dir" in data:
    ckpt = data["checkpoint_dir"]
elif "artifacts" in data and isinstance(data["artifacts"], dict):
    ckpt = data["artifacts"].get("checkpoint_dir")
else:
    ckpt = None
if ckpt:
    print(f"[checkpoint {'ok' if os.path.isdir(ckpt) else 'missing'}] {ckpt}")
PY
        fi
    done
}

run_once() {
    print_header
    echo ""
    echo "PID status:"
    if find "$PIPELINE_ROOT" -type f -path "*/pids/*.pid" | grep -q .; then
        while IFS= read -r -d '' pid_file; do
            check_pid_file "$pid_file"
        done < <(find "$PIPELINE_ROOT" -type f -path "*/pids/*.pid" -print0 2>/dev/null | sort -z)
    else
        echo "No pid files found."
    fi

    echo ""
    echo "Stage status:"
    if find "$PIPELINE_ROOT" -type f -path "*/status/*.status" | grep -q .; then
        while IFS= read -r -d '' status_file; do
            check_status_file "$status_file"
        done < <(find "$PIPELINE_ROOT" -type f -path "*/status/*.status" -print0 2>/dev/null | sort -z)
    else
        echo "No status files found."
    fi

    echo ""
    echo "Artifacts:"
    check_artifacts || true
    check_checkpoint_targets

    echo ""
    if [[ -f "utils/monitor_optuna_eta.py" ]]; then
        python utils/monitor_optuna_eta.py "$PIPELINE_ROOT"
    fi

    echo ""
    echo "Recent log scan:"
    scan_logs
    echo ""
}

if [[ "$MODE" == "--once" ]]; then
    run_once
    exit 0
fi

while true; do
    clear || true
    run_once
    sleep "$REFRESH_SECS"
done
