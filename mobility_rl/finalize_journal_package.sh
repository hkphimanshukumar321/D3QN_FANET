#!/bin/bash
# ==============================================================================
# Finalize Journal Package
# ==============================================================================
# Waits for the pipeline jobs to finish, then aggregates all tuned runs and
# journal suite outputs into one paper_package/ directory.
#
# Usage:
#   bash finalize_journal_package.sh <pipeline_root>
#   bash finalize_journal_package.sh <pipeline_root> --no-wait
# ==============================================================================

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <pipeline_root> [--no-wait]"
    exit 1
fi

PIPELINE_ROOT="$1"
MODE="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REFRESH_SECS="${REFRESH_SECS:-20}"

if [[ ! -d "$PIPELINE_ROOT" ]]; then
    echo "Pipeline root not found: $PIPELINE_ROOT"
    exit 1
fi

all_done() {
    local status_file
    while IFS= read -r -d '' status_file; do
        local status
        status="$(head -n 1 "$status_file" 2>/dev/null || echo "")"
        case "$status" in
            RUNNING|DETACHED|"")
                return 1
                ;;
        esac
    done < <(find "$PIPELINE_ROOT" -type f -path "*/status/*.status" -print0 2>/dev/null)
    return 0
}

if [[ "$MODE" != "--no-wait" ]]; then
    echo "Waiting for pipeline completion: $PIPELINE_ROOT"
    until all_done; do
        bash monitor_pipeline.sh "$PIPELINE_ROOT" --once || true
        sleep "$REFRESH_SECS"
    done
fi

echo "Aggregating paper package..."
PACKAGE_DIR="$("$PYTHON_BIN" experiments/aggregate_journal_package.py --pipeline-root "$PIPELINE_ROOT")"
echo "Paper package ready:"
echo "  $PACKAGE_DIR"
