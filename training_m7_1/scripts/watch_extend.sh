#!/usr/bin/env bash
# Watcher for the M7.1-extend training run.
#
# Polls the training PID until it exits, then writes a status report
# documenting whether the run completed cleanly or crashed.
# DOES NOT auto-restart on error — that's a deliberate choice so a
# crash-loop can't quietly burn GPU time.
#
# Usage: bash training_m7_1/scripts/watch_extend.sh <PID>
#
# Outputs:
#   logs/m7_extend_watcher.log  — append-only audit trail of watcher actions
#   logs/m7_extend_status.txt   — single-file current status (overwrite)

set -u
TARGET_PID="${1:-}"
if [[ -z "$TARGET_PID" ]]; then
    echo "usage: $0 <PID>" >&2
    exit 2
fi

REPO_ROOT="/workspace/reason_over_search"
cd "${REPO_ROOT}"

WATCHER_LOG="logs/m7_extend_watcher.log"
STATUS_FILE="logs/m7_extend_status.txt"
TRAIN_LOG="logs/m7_extend.log"

log() {
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[${ts}] $*" >> "${WATCHER_LOG}"
}

write_status() {
    {
        echo "=== M7.1-extend watcher status ==="
        echo "Updated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "Target PID: ${TARGET_PID}"
        echo "$@"
    } > "${STATUS_FILE}"
}

log "started watching PID=${TARGET_PID}"
write_status "STATE: RUNNING"

# Poll every 60s
while kill -0 "${TARGET_PID}" 2>/dev/null; do
    sleep 60
done

# Process is gone. Determine outcome.
EXIT_DETECTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "PID ${TARGET_PID} no longer exists; analyzing exit"

CLEAN_EXIT=0
if grep -q "Max number of steps has been reached" "${TRAIN_LOG}" 2>/dev/null; then
    CLEAN_EXIT=1
fi

ERROR_TRACE=""
if grep -q -E "OOM|OutOfMemoryError|No space left|FAILED|Traceback" "${TRAIN_LOG}" 2>/dev/null; then
    # Capture the last 3 error lines for the status file
    ERROR_TRACE="$(grep -E 'OOM|OutOfMemoryError|No space left|FAILED|Traceback' "${TRAIN_LOG}" 2>/dev/null | tail -3)"
fi

# Latest step jsonl
LATEST_STEP_FILE="$(ls -t logs/exp_018/train_data_step*.jsonl 2>/dev/null | head -1 || true)"
LATEST_STEP_NUM=""
if [[ -n "${LATEST_STEP_FILE}" ]]; then
    LATEST_STEP_NUM="$(basename "${LATEST_STEP_FILE}" | grep -oE '[0-9]+' | head -1)"
fi

# Last 10 rewards summary
LAST10="(no jsonl yet)"
if [[ -n "${LATEST_STEP_FILE}" && -n "${LATEST_STEP_NUM}" ]]; then
    LAST10="$(training_m7_1/nemo_rl/.venv/bin/python <<PY 2>/dev/null || echo "(reward parse failed)"
import json
from pathlib import Path
start = max(1, ${LATEST_STEP_NUM} - 9)
out = []
for s in range(start, ${LATEST_STEP_NUM} + 1):
    p = Path(f"logs/exp_018/train_data_step{s}.jsonl")
    if not p.exists():
        continue
    rs = []
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            rs.append(float(r.get("rewards", [0.0])[0]))
    if rs:
        out.append(f"  step {s:>3d}: rew_mean={sum(rs)/len(rs):.4f}  pct_pos={sum(1 for x in rs if x>0)/len(rs)*100:.1f}%")
print("\n".join(out))
PY
)"
fi

# Ckpts saved
CKPT_LIST="$(ls -1 results/grpo/m7_short100/seed42/ 2>/dev/null | tr '\n' ' ')"

# Determine state
if [[ ${CLEAN_EXIT} -eq 1 ]]; then
    STATE="COMPLETED_CLEAN"
    log "clean completion detected (Max number of steps reached)"
elif [[ -n "${ERROR_TRACE}" ]]; then
    STATE="EXITED_WITH_ERROR"
    log "error detected in log: ${ERROR_TRACE}"
else
    STATE="EXITED_UNEXPECTED"
    log "process exited but no clean-completion marker and no traceback"
fi

write_status "STATE: ${STATE}
Exit detected: ${EXIT_DETECTED_AT}
Last completed step: ${LATEST_STEP_NUM}
Checkpoints on disk: ${CKPT_LIST}

Last 10 steps' rewards:
${LAST10}

Error trace (if any):
${ERROR_TRACE:-(none)}

NOTE: This watcher does NOT auto-restart. To resume from the latest
checkpoint after fixing the cause:
  cd ${REPO_ROOT}
  bash training_m7_1/scripts/run.sh --mode extend
"

log "watcher done — state=${STATE}"
