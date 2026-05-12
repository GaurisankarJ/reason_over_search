#!/usr/bin/env bash
# M5.1 GRPO launcher for Qwen3.5-0.8B on MuSiQue, 1× A100 80GB.
#
# Configs supported, selected via --mode:
#   --mode smoke       → configs/m7_smoke.yaml             (20 traj × 50 steps; seq=4096)
#   --mode smoke_8192  → configs/m7_smoke_8192.yaml        (20 traj × 50 steps; seq=8192)
#   --mode prod_shape  → configs/m7_prod_shape_smoke.yaml  (320 traj × 3 steps; prod shape)
#   --mode short100    → configs/m7_1_short100.yaml        (100-step prod probe, ckpts at 50+100)
#   --mode prod        → configs/m7_1_research_paper.yaml  (full 622-step ReSearch recipe)
#
# Common knobs:
#   --seed N      → GRPO seed (RNG; defaults to 42)
#   --            → everything after is forwarded as Hydra overrides
#
# Examples:
#   bash training_m7_1/scripts/run.sh --mode smoke
#   bash training_m7_1/scripts/run.sh --mode smoke --seed 7
#   bash training_m7_1/scripts/run.sh --mode prod  --seed 42
#   bash training_m7_1/scripts/run.sh --mode smoke -- policy.train_micro_batch_size=8
#
# Prerequisites:
#   1. Retriever live at 127.0.0.1:3005 (curl -sS http://127.0.0.1:3005/health → healthy)
#   2. training_m7_1/.env populated (WANDB_API_KEY); copy from .env.example if missing.
#   3. training_m7_1/nemo_rl/.venv materialized (training_m7_1/setup.sh or uv sync).
#   4. data/training/musique/train.parquet present
#      (training_m7_1/scripts/prep_musique.py downloads + reshapes it).
#   5. training_m7_1/src/prompts/m7_qwen35_base_user.txt populated
#      (training_m7_1/scripts/sync_m4_prompts.py --mode <M4 winner>; pre-staged
#      at qwen35_minimal, M4.2 canonical).

set -euo pipefail

MODE=""
SEED="42"
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --) shift; EXTRA_OVERRIDES=("$@"); break ;;
        -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1 (run with --help for usage)" >&2; exit 2 ;;
    esac
done

case "$MODE" in
    smoke)           CONFIG="training_m7_1/configs/m7_smoke.yaml" ;;
    smoke_8192)      CONFIG="training_m7_1/configs/m7_smoke_8192.yaml" ;;            # [M7.0.7b] seq=8192 smoke variant
    prod_shape)      CONFIG="training_m7_1/configs/m7_prod_shape_smoke.yaml" ;;       # [M7.0.7c] prod batch (320 traj) × 3 steps
    short100)        CONFIG="training_m7_1/configs/m7_1_short100.yaml" ;;             # [M7.1] 100-step probe, ckpts at 50+100
    prod)            CONFIG="training_m7_1/configs/m7_1_research_paper.yaml" ;;
    "")              echo "error: --mode is required (smoke|smoke_8192|prod_shape|short100|prod)" >&2; exit 2 ;;
    *)               echo "error: --mode must be smoke|smoke_8192|prod_shape|short100|prod (got: $MODE)" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "error: config not found at ${CONFIG}" >&2
    if [[ "$MODE" == "prod" ]]; then
        echo "       (m7_1_research_paper.yaml is built in M5.1 step 6;" >&2
        echo "        see docs/milestone_5/MILESTONE_5.md §\"Run sequence — M5.1\")" >&2
    fi
    exit 1
fi

VENV_PYTHON="${REPO_ROOT}/training_m7_1/nemo_rl/.venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "error: venv not found at ${VENV_PYTHON}" >&2
    echo "       run training_m7_1/setup.sh first" >&2
    exit 1
fi

if [[ -f "${REPO_ROOT}/training_m7_1/.env" ]]; then
    set -a; source "${REPO_ROOT}/training_m7_1/.env"; set +a
fi

# NOTE: PYTORCH_ALLOC_CONF=expandable_segments:True was tried for log_softmax
# fragmentation but breaks NeMo-RL's CUDA-IPC weight sharing between policy
# worker and vLLM worker (pidfd_getfd "Operation not permitted"). Stay on the
# default allocator; reduce memory pressure via train_micro_batch_size +
# vllm_cfg.gpu_memory_utilization in the yaml instead.

TS="$(date -u +%Y%m%dT%H%MZ)"
# [M7] Run names + ckpt dirs now use m7_ prefix and include "base" qualifier
# to distinguish from M5.1 hybrid runs in W&B + on-disk artifacts.
RUN_NAME="qwen3.5-0.8b-base-musique-m7_${MODE}-seed${SEED}-${TS}"
# Checkpoint dir keyed by (mode, seed); not timestamped, so resumes work.
CKPT_DIR="${CHECKPOINT_DIR_BASE:-results/grpo}/m7_${MODE}/seed${SEED}"

OVERRIDES=(
    "grpo.seed=${SEED}"
    "logger.wandb.name=${RUN_NAME}"
    "checkpointing.checkpoint_dir=${CKPT_DIR}"
)

# Hybrid Qwen3.5-0.8B needs enable_thinking=True so the <think> block opens
# (matches M4 eval). Forced-add since the config keeps chat_template_kwargs=null.
OVERRIDES+=("++policy.tokenizer.chat_template_kwargs.enable_thinking=true")

echo "[run.sh] mode=${MODE} seed=${SEED}"
echo "[run.sh] config=${CONFIG}"
echo "[run.sh] run name=${RUN_NAME}"
echo "[run.sh] ckpt dir=${CKPT_DIR}"

exec "${VENV_PYTHON}" training_m7_1/scripts/run_grpo.py \
    --config="${CONFIG}" \
    "${OVERRIDES[@]}" \
    "${EXTRA_OVERRIDES[@]}"
