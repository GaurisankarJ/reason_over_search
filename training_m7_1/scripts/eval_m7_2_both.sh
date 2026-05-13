#!/usr/bin/env bash
# Run M7.2 eval on both saved ckpts back-to-back:
#   1) M7.1 short100 step_100 (F1-only reward)
#   2) M7.4 short100 step_50  (F1+format reward)
#
# Both use the M4.3 base lock prompt (qwen35_minimal_no_system, byte-equal
# to the training render) on the Plan A 7-dataset suite. Full splits.
#
# Outputs:
#   logs/m7_2_eval_*_step*.log               -- per-eval driver log
#   logs/m7_2_eval_both_status.txt           -- watcher status snapshot
#   evaluation_qwen35/results/<dataset>/...  -- per-dataset metric_score.txt
#
# Usage:
#   nohup bash training_m7_1/scripts/eval_m7_2_both.sh > logs/m7_2_eval_both.log 2>&1 &

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

STATUS="logs/m7_2_eval_both_status.txt"
DRIVER_LOG="logs/m7_2_eval_both.log"
EVAL_SCRIPT="training_m7_1/scripts/eval_m7_2.sh"

# Override the hardcoded HPC python path in scripts/run_m4.sh to use the
# Vast eval venv. evaluation_qwen35/ local flashrag wins (no system install
# in this venv); confirmed via `cd evaluation_qwen35 && python -c 'import flashrag'`.
export PY="/venv/evaluation_search_r1/bin/python"

status() {
    {
        echo "=== M7.2 eval-both watcher status ==="
        echo "Updated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "$*"
    } > "$STATUS"
}

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

START="$(date -u +%s)"
log "M7.2 eval-both starting"
status "STATE: STARTING"

# Retriever sanity
if ! curl -sS --max-time 3 http://127.0.0.1:3005/health 2>/dev/null | grep -q healthy ; then
    log "FATAL: retriever not healthy at 127.0.0.1:3005"
    status "STATE: ABORTED — retriever not healthy"
    exit 1
fi

# Cell 1: M7.1 step_100
log "=== cell 1/2: M7.1 step_100 (F1-only reward) ==="
status "STATE: RUNNING (cell 1/2 — M7.1 step_100)"
if CKPT_BASE="results/grpo/m7_short100/seed42" RUN_TAG="m7_1" \
       bash "$EVAL_SCRIPT" 100 ; then
    log "cell 1 DONE"
else
    log "cell 1 FAILED — aborting before cell 2"
    status "STATE: FAILED on cell 1 (M7.1 step_100)"
    exit 1
fi

# Cell 2: M7.4 step_50
log "=== cell 2/2: M7.4 step_50 (F1+format reward) ==="
status "STATE: RUNNING (cell 2/2 — M7.4 step_50)"
if CKPT_BASE="results/grpo/m7_m74_short100/seed42" RUN_TAG="m7_4" \
       bash "$EVAL_SCRIPT" 50 ; then
    log "cell 2 DONE"
else
    log "cell 2 FAILED"
    status "STATE: FAILED on cell 2 (M7.4 step_50)"
    exit 1
fi

ELAPSED_MIN=$(( ($(date -u +%s) - START) / 60 ))
log "M7.2 eval-both complete in ${ELAPSED_MIN} min"
status "STATE: COMPLETE
Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Wall: ${ELAPSED_MIN} min
Result files: evaluation_qwen35/results/<dataset>/<dataset>_*_m4_qwen3.5_0.8b_base_qwen35_minimal_no_system_seed1/
Driver logs: logs/m7_2_eval_*.log"
