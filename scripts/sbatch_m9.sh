#!/usr/bin/env bash
# SLURM batch script: run all 7 M9 datasets for ONE M5.1 checkpoint variant.
# Submit:
#   cd /zfsstore/user/s4374886/omega/reason_over_search_m9
#   sbatch scripts/sbatch_m9.sh 10
#   sbatch scripts/sbatch_m9.sh 50
#   sbatch scripts/sbatch_m9.sh 100
#   sbatch scripts/sbatch_m9.sh 150
#   sbatch scripts/sbatch_m9.sh 180
#
# Adapted from scripts/sbatch_m4.sh. Key changes:
#   - VARIANT = step number (10|50|100|150|180); resolves to eval/m9/step_<N>_hf
#   - prompt_mode locked to m5_qwen35_train (byte-exact training prompt)
#   - save_note = m9_step<N>_seed1

#SBATCH --job-name=m9_eval
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=160g
#SBATCH --time=04:00:00
#SBATCH --output=logs/m9_%j_%x.out
#SBATCH --error=logs/m9_%j_%x.err

set -euo pipefail

STEP="${1:?Usage: sbatch scripts/sbatch_m9.sh <10|50|100|150|180>}"

# Use SLURM_SUBMIT_DIR (per CODE_SETUP_m3 #12) so the script resolves correctly under sbatch.
REPO_ROOT="${SLURM_SUBMIT_DIR:-/zfsstore/user/s4374886/omega/reason_over_search_m9}"
IVF_INDEX="$REPO_ROOT/indexes/wiki18_100w_e5_ivf4096_sq8.index"
MODEL_PATH="$REPO_ROOT/eval/m9/step_${STEP}_hf"

# Bootstrap lmod (same rationale as sbatch_m4.sh).
if ! command -v module >/dev/null 2>&1; then
  export MODULEPATH="${MODULEPATH:-/trinity/shared/modulefiles}"
  . /opt/ohpc/admin/lmod/lmod/init/bash >/dev/null 2>&1 || true
fi
module purge
module load ALICE/default
module load Miniconda3/24.7.1-0
module load CUDA/12.4.0

cd "$REPO_ROOT"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: ckpt not found: $MODEL_PATH" >&2
  echo "Run scripts/convert_m5_ckpt_to_hf.py first." >&2
  exit 1
fi
if [[ ! -f "$IVF_INDEX" ]]; then
  echo "ERROR: IVF index not found: $IVF_INDEX" >&2
  exit 1
fi

# ── Retriever ────────────────────────────────────────────────────────────────
mkdir -p logs
RETRIEVER_LOG="logs/m9_${SLURM_JOB_ID:-local}_retriever.log"
export OMP_NUM_THREADS=4
echo "Using IVF-SQ8 index (× 8 workers): $IVF_INDEX"
/home/s4374886/.conda/envs/retriever/bin/python -u local_retriever/retriever_serving.py \
  --config local_retriever/retriever_config.yaml \
  --num_retriever 8 \
  --index "$IVF_INDEX" \
  --port 3005 > "$RETRIEVER_LOG" 2>&1 &
RETRIEVER_PID=$!
echo "Retriever PID: $RETRIEVER_PID  log: $RETRIEVER_LOG"

# 1200 s wait — same budget as sbatch_m4.sh
echo "Waiting for retriever to load index (up to 1200 s) ..."
for i in $(seq 1 240); do
  sleep 5
  if curl -sf http://127.0.0.1:3005/health > /dev/null 2>&1; then
    echo "Retriever healthy after $((i*5)) s"
    break
  fi
  if ! kill -0 "$RETRIEVER_PID" 2>/dev/null; then
    echo "ERROR: retriever process died after $((i*5)) s — last log lines:" >&2
    tail -30 "$RETRIEVER_LOG" >&2
    exit 1
  fi
done
if ! curl -sf http://127.0.0.1:3005/health > /dev/null 2>&1; then
  echo "ERROR: retriever still not healthy after 1200 s — last log lines:" >&2
  tail -30 "$RETRIEVER_LOG" >&2
  kill "$RETRIEVER_PID" 2>/dev/null || true
  exit 1
fi
curl -sS http://127.0.0.1:3005/health

# ── SGLang server ────────────────────────────────────────────────────────────
SGLANG_LOG="logs/m9_${SLURM_JOB_ID:-local}_sglang.log"
export SGLANG_DISABLE_CUDNN_CHECK=1
export PATH="/home/s4374886/.conda/envs/evaluation_search_r1/bin:$PATH"
/home/s4374886/.conda/envs/evaluation_search_r1/bin/python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 --port 3000 \
  --tp 1 --context-length 8192 --dtype bfloat16 --trust-remote-code > "$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!
echo "SGLang PID: $SGLANG_PID  log: $SGLANG_LOG"

echo "Waiting for SGLang to load model (up to 600 s) ..."
for i in $(seq 1 120); do
  sleep 5
  if curl -sf http://127.0.0.1:3000/health > /dev/null 2>&1; then
    echo "SGLang healthy after $((i*5)) s"
    break
  fi
  if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
    echo "ERROR: SGLang process died after $((i*5)) s — last log lines:" >&2
    tail -30 "$SGLANG_LOG" >&2
    exit 1
  fi
done
if ! curl -sf http://127.0.0.1:3000/health > /dev/null 2>&1; then
  echo "ERROR: SGLang still not healthy after 600 s — last log lines:" >&2
  tail -30 "$SGLANG_LOG" >&2
  kill "$SGLANG_PID" 2>/dev/null || true
  exit 1
fi

# ── Eval loop ─────────────────────────────────────────────────────────────────
for DATASET in bamboogle nq triviaqa popqa hotpotqa 2wikimultihopqa musique; do
  bash scripts/run_m9.sh "$STEP" "$DATASET" 1 "$REPO_ROOT/data"
done

# ── Cleanup ───────────────────────────────────────────────────────────────────
kill "$SGLANG_PID" "$RETRIEVER_PID" 2>/dev/null || true
wait "$SGLANG_PID" "$RETRIEVER_PID" 2>/dev/null || true

echo "All 7 datasets done for step=$STEP"
