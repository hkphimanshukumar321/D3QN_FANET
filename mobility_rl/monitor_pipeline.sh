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

get_status() {
    local pid_file="$1"
    local name
    name="$(basename "$pid_file" .pid)"
    
    local status_file
    status_file="$(dirname "$(dirname "$pid_file")")/status/${name}.status"
    if [[ -f "$status_file" ]]; then
        local st
        st="$(head -n 1 "$status_file" 2>/dev/null || echo "MISSING")"
        if [[ "$st" == "COMPLETED" || "$st" == DETACHED* || "$st" == FAILED* ]]; then
            echo "($st)"
            return
        fi
    fi

    local pid
    pid="$(tr -d '[:space:]' < "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "RUNNING"
    else
        echo "DEAD"
    fi
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
    return "$missing"
}

run_once() {
    print_header
    echo ""
    echo "=== Active Pipeline Stages ==="
    
    local has_optuna=0
    local has_unified=0
    local has_journal=0

    # Collect status
    if find "$PIPELINE_ROOT" -type f -path "*/pids/*.pid" | grep -q .; then
        while IFS= read -r -d '' pid_file; do
            local name
            name="$(basename "$pid_file" .pid)"
            local st
            st="$(get_status "$pid_file")"
            
            # Skip old launchers and dead processes from old pipelines if multiple exist
            if [[ "$name" == "launcher" && "$st" != "RUNNING" && "$st" != "(FAILED"* ]]; then
                continue
            fi
            
            if [[ "$name" == optuna_* ]]; then
                has_optuna=1
                printf " [Tuning]       %-24s : %s\n" "${name#optuna_}" "$st"
            elif [[ "$name" == final_unified_* ]]; then
                has_unified=1
                printf " [Unified Eval] %-24s : %s\n" "${name#final_unified_}" "$st"
            elif [[ "$name" == generalization_* || "$name" == robustness_* || "$name" == failure_recovery || "$name" == complexity || "$name" == magat_isolation ]]; then
                has_journal=1
                printf " [Journal Test] %-24s : %s\n" "$name" "$st"
            fi
        done < <(find "$PIPELINE_ROOT" -type f -path "*/pids/*.pid" -print0 2>/dev/null | sort -z)
    fi

    if [[ "$has_optuna" == 0 && "$has_unified" == 0 && "$has_journal" == 0 ]]; then
        echo " No active processes found in this pipeline."
    fi

    echo ""
    echo "=== Artifacts ==="
    check_artifacts || true

    echo ""
    echo "=== Optuna Study Estimates ==="
    if [[ -f "utils/monitor_optuna_eta.py" ]]; then
        # Run it but filter out older dead studies if we want to reduce confusion
        if [[ -n "${N_TRIALS:-}" ]]; then
            python utils/monitor_optuna_eta.py . --force-target "$N_TRIALS" | grep -v "COMPLETED" || true
        else
            python utils/monitor_optuna_eta.py . --force-target 24 | grep -v "COMPLETED" || true
        fi
        # Always show COMPLETED explicitly if pipeline handles it nicely
        echo " (Any algorithm not listed here has successfully finished tuning)"
    fi

    echo ""
    echo "=== Recent Log Scan ==="
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
