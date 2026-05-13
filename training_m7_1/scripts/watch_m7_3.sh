#!/usr/bin/env bash
# Watcher for the M7.3-short100 training run.
# Polls PID until exit, then writes status report. No auto-restart.

set -u
TARGET_PID="${1:?usage: $0 <pid>}"
REPO_ROOT="/workspace/reason_over_search"
cd "$REPO_ROOT"

WATCHER_LOG="logs/m7_3_watcher.log"
STATUS_FILE="logs/m7_3_status.txt"
TRAIN_LOG="logs/m7_3_short100.log"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$WATCHER_LOG"; }
status() {
    {
        echo "=== M7.3-short100 watcher status ==="
        echo "Updated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "Target PID: ${TARGET_PID}"
        echo "$@"
    } > "$STATUS_FILE"
}

log "started watching PID=${TARGET_PID}"
status "STATE: RUNNING"

while kill -0 "${TARGET_PID}" 2>/dev/null; do
    sleep 60
done

EXIT_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "PID ${TARGET_PID} no longer exists; analyzing"

CLEAN=0
if grep -q "Max number of steps has been reached" "${TRAIN_LOG}" 2>/dev/null; then
    CLEAN=1
fi
ERR=""
if grep -q -E "OOM|OutOfMemoryError|No space left|FAILED|Traceback" "${TRAIN_LOG}" 2>/dev/null; then
    ERR="$(grep -E 'OOM|OutOfMemoryError|No space left|FAILED|Traceback' "${TRAIN_LOG}" 2>/dev/null | tail -3)"
fi

# Find latest exp jsonl + step number
EXP_DIR="$(ls -td logs/exp_* 2>/dev/null | head -1)"
LATEST=""
LATEST_NUM=""
if [[ -n "${EXP_DIR}" ]]; then
    LATEST="$(ls -t ${EXP_DIR}/train_data_step*.jsonl 2>/dev/null | head -1 || true)"
    if [[ -n "${LATEST}" ]]; then
        LATEST_NUM="$(basename "${LATEST}" | grep -oE '[0-9]+' | head -1)"
    fi
fi

LAST10="(none)"
if [[ -n "${LATEST_NUM}" && -n "${EXP_DIR}" ]]; then
    LAST10="$(training_m7_1/nemo_rl/.venv/bin/python <<PY 2>/dev/null
import json
from pathlib import Path
start = max(1, ${LATEST_NUM} - 9)
out = []
for s in range(start, ${LATEST_NUM} + 1):
    p = Path(f"${EXP_DIR}/train_data_step{s}.jsonl")
    if not p.exists():
        continue
    rs = []
    tc = 0
    for line in open(p):
        r = json.loads(line)
        rs.append(float(r["rewards"][0]))
        for i, c in enumerate(r["content"][0]):
            if i == 0 or not c: continue
            if i % 2 == 1:
                if "<tool_call>" in c: tc += 1
                break
    if rs:
        out.append(f"  step {s:>3d}: rew_mean={sum(rs)/len(rs):.4f}  pct_pos={sum(1 for x in rs if x>0)/len(rs)*100:.1f}%  pct_tc={tc/len(rs)*100:.1f}%")
print("\n".join(out))
PY
)"
fi

CKPTS="$(ls -1 results/grpo/m7_m73_short100/seed42/ 2>/dev/null | tr '\n' ' ')"

if [[ ${CLEAN} -eq 1 ]]; then
    STATE="COMPLETED_CLEAN"
elif [[ -n "${ERR}" ]]; then
    STATE="EXITED_WITH_ERROR"
else
    STATE="EXITED_UNEXPECTED"
fi
log "state=${STATE}"

status "STATE: ${STATE}
Exit detected: ${EXIT_AT}
Last completed step: ${LATEST_NUM}
Checkpoints: ${CKPTS}

Last 10 steps (rew_mean, pct_pos, pct_trajectories_with_tool_call):
${LAST10}

Error trace (if any):
${ERR:-(none)}
"
log "done"
