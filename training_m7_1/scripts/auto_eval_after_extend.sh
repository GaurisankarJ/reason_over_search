#!/usr/bin/env bash
# Auto-fires M7.2 eval when the M7.1-extend training process exits cleanly.
#
# Polls the training PID; once it's gone, examines the log for clean
# completion ("Max number of steps has been reached"), picks the highest
# step_N ckpt, and invokes scripts/eval_m7_2.sh on it.
#
# DOES NOT fire eval if the run exited with an error (Traceback, OOM, etc.) —
# user must inspect and re-launch manually.
#
# Usage:
#   nohup bash training_m7_1/scripts/auto_eval_after_extend.sh <extend_pid> > /dev/null 2>&1 &
#
# Outputs:
#   logs/auto_eval_watcher.log  — audit trail of polls + decision + eval invocation
#   logs/auto_eval_status.txt   — current state (overwritten)

set -u
TARGET_PID="${1:?usage: $0 <extend_pid>}"
REPO_ROOT="/workspace/reason_over_search"
cd "$REPO_ROOT"

WATCHER_LOG="logs/auto_eval_watcher.log"
STATUS_FILE="logs/auto_eval_status.txt"
TRAIN_LOG="logs/m7_extend.log"
EVAL_SCRIPT="training_m7_1/scripts/eval_m7_2.sh"
CKPT_BASE="results/grpo/m7_short100/seed42"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$WATCHER_LOG"
}
status() {
    {
        echo "=== auto-eval watcher status ==="
        echo "Updated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "Target PID: ${TARGET_PID}"
        echo "$*"
    } > "$STATUS_FILE"
}

log "started — watching PID=${TARGET_PID}, will auto-fire ${EVAL_SCRIPT} on clean exit"
status "STATE: WAITING_FOR_EXTEND_TO_FINISH"

# Poll every 60s until the training PID exits
while kill -0 "$TARGET_PID" 2>/dev/null; do
    sleep 60
done

log "PID $TARGET_PID exited; analyzing"

# Determine outcome
CLEAN=0
if grep -q "Max number of steps has been reached" "$TRAIN_LOG" 2>/dev/null; then
    CLEAN=1
fi
HAS_ERROR=0
if grep -q -E "Traceback|OutOfMemoryError|CUDA out of memory|No space left|FAILED|AssertionError" "$TRAIN_LOG" 2>/dev/null; then
    HAS_ERROR=1
fi

if [[ $CLEAN -ne 1 ]]; then
    log "no clean-completion marker; aborting auto-eval"
    status "STATE: ABORTED_NO_CLEAN_COMPLETION
Train log: $TRAIN_LOG (no 'Max number of steps' marker)
Action: inspect log + decide whether to relaunch."
    exit 0
fi
if [[ $HAS_ERROR -eq 1 ]]; then
    log "found error markers in train log; aborting auto-eval out of caution"
    err_snip="$(grep -E 'Traceback|OutOfMemoryError|No space left|FAILED' "$TRAIN_LOG" 2>/dev/null | tail -3)"
    status "STATE: ABORTED_ERROR_DETECTED
Error snippet:
${err_snip}
Action: inspect $TRAIN_LOG; auto-eval skipped."
    exit 0
fi

# Find latest ckpt
LATEST_STEP=$(ls "$CKPT_BASE" 2>/dev/null | grep -oE 'step_[0-9]+' | sort -t_ -k2 -n | tail -1 | sed 's/step_//')
if [[ -z "$LATEST_STEP" ]]; then
    log "no step_N ckpt under $CKPT_BASE; aborting"
    status "STATE: ABORTED_NO_CKPT
Action: ckpt dir empty — training may have crashed before any save."
    exit 1
fi

log "clean completion + latest ckpt = step ${LATEST_STEP}"
status "STATE: LAUNCHING_EVAL
Target ckpt: step_${LATEST_STEP}
Eval driver: ${EVAL_SCRIPT}
Started at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Retriever sanity
if ! curl -sS --max-time 3 http://127.0.0.1:3005/health 2>/dev/null | grep -q healthy ; then
    log "retriever not healthy; aborting eval"
    status "STATE: ABORTED_RETRIEVER_DOWN
Action: restart retriever (see local_retriever/), then re-invoke ${EVAL_SCRIPT} ${LATEST_STEP} manually."
    exit 1
fi

# Run the eval — block until done so this watcher's log records the outcome
log "invoking ${EVAL_SCRIPT} ${LATEST_STEP}"
if bash "${EVAL_SCRIPT}" "${LATEST_STEP}" >> "$WATCHER_LOG" 2>&1; then
    log "eval completed cleanly"
    status "STATE: EVAL_COMPLETE
Target ckpt: step_${LATEST_STEP}
Finished at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Result files: evaluation_qwen35/results/<dataset>/metric_score.txt
Driver log:   logs/m7_2_eval.log"
else
    rc=$?
    log "eval exited with non-zero status: ${rc}"
    status "STATE: EVAL_FAILED
Target ckpt: step_${LATEST_STEP}
Exit code: ${rc}
Action: inspect logs/m7_2_eval.log + $WATCHER_LOG"
fi
