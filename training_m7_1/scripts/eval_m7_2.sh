#!/usr/bin/env bash
# M7.2 eval: run the M4.3 base lock (`qwen35_minimal_no_system` template +
# `enable_thinking=True`) on a trained M7.1 checkpoint across the Plan A
# 7-dataset suite (bamboogle, nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa,
# musique).
#
# Reuses scripts/run_m4.sh — the only delta vs untrained-baseline eval is
# QWEN35_0_8B_BASE_PATH pointed at the materialized trained ckpt.
#
# Usage:
#   bash training_m7_1/scripts/eval_m7_2.sh <ckpt_step>
#   bash training_m7_1/scripts/eval_m7_2.sh latest   # picks highest step_N in m7_short100/seed42
#
# Outputs:
#   eval/qwen3.5_0.8b_base_m7_step<N>/   — materialized HF model dir (symlinks where possible)
#   evaluation_qwen35/results/<dataset>/  — eval result JSON + metric_score.txt per dataset
#   logs/m7_2_eval.log                    — driver log
#
# Prerequisites:
#   - Retriever live at 127.0.0.1:3005 (curl -sS http://127.0.0.1:3005/health → healthy)
#   - evaluation_qwen35 venv exists and run_m4.sh works (already proven by M4 eval)
#   - Trained ckpt with policy/weights/model/consolidated/*.safetensors + policy/tokenizer/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# CKPT_BASE may be overridden via env (e.g. CKPT_BASE=results/grpo/m7_m74_short100/seed42).
# Default targets the M7.1 short100 ckpts.
CKPT_BASE="${CKPT_BASE:-results/grpo/m7_short100/seed42}"
# RUN_TAG is used to namespace the materialized HF model dir + eval result files
# so multiple runs (M7.1, M7.4, ...) don't collide. Default derived from CKPT_BASE.
RUN_TAG="${RUN_TAG:-$(basename "$(dirname "$CKPT_BASE")")_$(basename "$CKPT_BASE")}"
ARG_STEP="${1:?missing arg: ckpt step number, or 'latest'}"

# Resolve step number
if [[ "$ARG_STEP" == "latest" ]]; then
    STEP=$(ls "$CKPT_BASE" 2>/dev/null | grep -oE 'step_[0-9]+' | sort -t_ -k2 -n | tail -1 | sed 's/step_//')
    if [[ -z "$STEP" ]]; then
        echo "ERROR: no step_N directories found under $CKPT_BASE" >&2
        exit 1
    fi
else
    STEP="$ARG_STEP"
fi
CKPT_DIR="$CKPT_BASE/step_${STEP}"
if [[ ! -d "$CKPT_DIR" ]]; then
    echo "ERROR: ckpt dir not found: $CKPT_DIR" >&2
    exit 1
fi
echo "[eval_m7_2] target ckpt: step ${STEP}  (${CKPT_DIR})"

# Materialize an HF model dir (eval/qwen3.5_0.8b_base_m7_step<N>/) combining:
#   policy/weights/model/consolidated/*  (config.json + safetensors + index)
#   policy/tokenizer/*                   (tokenizer files + chat template)
MODEL_DIR="eval/qwen3.5_0.8b_base_${RUN_TAG}_step${STEP}"
SRC_WEIGHTS="${REPO_ROOT}/${CKPT_DIR}/policy/weights/model/consolidated"
SRC_TOKENIZER="${REPO_ROOT}/${CKPT_DIR}/policy/tokenizer"

if [[ ! -f "${SRC_WEIGHTS}/config.json" ]]; then
    echo "ERROR: expected consolidated config.json at ${SRC_WEIGHTS}/config.json" >&2
    exit 1
fi
if [[ ! -f "${SRC_TOKENIZER}/tokenizer.json" ]]; then
    echo "ERROR: expected tokenizer.json at ${SRC_TOKENIZER}/tokenizer.json" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"
# Symlink everything (read-only, no disk duplication)
for f in "$SRC_WEIGHTS"/* ; do
    ln -sf "$f" "$MODEL_DIR/$(basename "$f")"
done
for f in "$SRC_TOKENIZER"/* ; do
    # tokenizer files take precedence (chat_template etc.); won't clash with consolidated/
    ln -sf "$f" "$MODEL_DIR/$(basename "$f")"
done
echo "[eval_m7_2] materialized HF model dir: ${MODEL_DIR}"
ls -la "$MODEL_DIR" | head -15

# Retriever sanity
if ! curl -sS --max-time 3 http://127.0.0.1:3005/health 2>/dev/null | grep -q healthy ; then
    echo "ERROR: retriever not healthy at 127.0.0.1:3005; aborting eval" >&2
    exit 1
fi
echo "[eval_m7_2] retriever OK"

# Eval driver setup. Override QWEN35_0_8B_BASE_PATH so run_m4.sh picks up our trained ckpt.
# PROMPT_MODE=qwen35_minimal_no_system matches the M4.3 base lock (and the training arm).
export QWEN35_0_8B_BASE_PATH="${REPO_ROOT}/${MODEL_DIR}"
export PROMPT_MODE="qwen35_minimal_no_system"
# Namespace save_note so the trained-ckpt eval doesn't collide with the
# M4 untrained-baseline results (same variant + prompt_mode, different
# weights). run_m4.sh appends this string after mode_tag.
export SAVE_NOTE_SUFFIX="_${RUN_TAG}_step${STEP}"

DATASETS=(bamboogle nq triviaqa popqa hotpotqa 2wikimultihopqa musique)
SEED=1
LOG_FILE="logs/m7_2_eval_${RUN_TAG}_step${STEP}.log"

mkdir -p logs
{
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] M7.2 eval starting"
    echo "  ckpt step: ${STEP}"
    echo "  model dir: ${MODEL_DIR}"
    echo "  PROMPT_MODE: ${PROMPT_MODE}"
    echo "  datasets: ${DATASETS[*]}"
    echo ""
} >> "$LOG_FILE"

OVERALL_START="$(date -u +%s)"
for dataset in "${DATASETS[@]}"; do
    START="$(date -u +%s)"
    echo "[$(date -u +%H:%M:%SZ)] eval dataset=${dataset}" | tee -a "$LOG_FILE"
    if bash scripts/run_m4.sh qwen3.5_0.8b_base "${dataset}" "${SEED}" 2>&1 | tee -a "$LOG_FILE" ; then
        END="$(date -u +%s)"
        echo "  ✓ done in $((END-START))s" | tee -a "$LOG_FILE"
    else
        END="$(date -u +%s)"
        echo "  ✗ FAILED in $((END-START))s" | tee -a "$LOG_FILE"
    fi
done

OVERALL_END="$(date -u +%s)"
TOTAL_MIN="$(( (OVERALL_END - OVERALL_START) / 60 ))"
echo "" | tee -a "$LOG_FILE"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] M7.2 eval complete (${TOTAL_MIN} min)" | tee -a "$LOG_FILE"

# Summary: collect metric_score.txt for each dataset
echo "" | tee -a "$LOG_FILE"
echo "=== M7.2 metric_score summary (step_${STEP}) ===" | tee -a "$LOG_FILE"
for dataset in "${DATASETS[@]}"; do
    # save_note pattern: m4_qwen35_minimal_no_system_seed1 (no _n100 suffix for full run)
    score_files=$(find "evaluation_qwen35/results/${dataset}" -name "metric_score.txt" -newer "$LOG_FILE.0" 2>/dev/null || \
                  find "evaluation_qwen35/results/${dataset}" -name "metric_score.txt" 2>/dev/null | tail -1)
    if [[ -n "$score_files" ]]; then
        echo "  ${dataset}:" | tee -a "$LOG_FILE"
        cat "$score_files" 2>/dev/null | head -3 | sed 's/^/    /' | tee -a "$LOG_FILE"
    else
        echo "  ${dataset}: (no metric_score.txt found)" | tee -a "$LOG_FILE"
    fi
done

echo "[eval_m7_2] DONE — see $LOG_FILE"
