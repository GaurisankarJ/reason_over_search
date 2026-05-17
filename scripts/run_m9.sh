#!/usr/bin/env bash
# Run one (m9 step ckpt, dataset, seed) evaluation via evaluation_qwen35.
# Usage:
#   scripts/run_m9.sh <step_N> <dataset> <seed> [data_dir]
# Where <step_N> is one of: 10 | 50 | 100 | 150 | 180
#
# Uses prompt_mode=m5_qwen35_train (byte-exact mirror of training prompt
# m5_qwen35_user.txt), enable_thinking=True, apply_chat=True, and the
# converted HF ckpt at eval/m9/step_<N>_hf/. Writes to
# evaluation_qwen35/results/<dataset>/<dataset>_*_m9_step<N>_seed<seed>/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$REPO_ROOT/evaluation_qwen35"
PY="${PY:-/home/s4374886/.conda/envs/evaluation_search_r1/bin/python}"

step="${1:?missing step (10|50|100|150|180)}"
dataset="${2:?missing dataset}"
seed="${3:?missing seed}"
data_dir="${4:-$REPO_ROOT/data}"

case "$dataset" in
  bamboogle|nq|triviaqa|popqa) split=test ;;
  hotpotqa|2wikimultihopqa|musique) split=dev ;;
  *) echo "unknown dataset: $dataset" >&2; exit 2 ;;
esac

model_path="$REPO_ROOT/eval/m9/step_${step}_hf"
if [[ ! -d "$model_path" ]]; then
  echo "ERROR: ckpt not found: $model_path" >&2
  exit 1
fi

save_note="m9_step${step}_seed${seed}"
save_dir="$EVAL_DIR/results/$dataset"
if compgen -G "$save_dir/${dataset}_*_${save_note}/metric_score.txt" > /dev/null 2>&1; then
  existing="$(ls -d "$save_dir"/${dataset}_*_${save_note} | tail -1)"
  echo "[skip] step=${step}/$dataset/seed=$seed already done -> $existing"
  exit 0
fi

echo "[run]  step=${step}/$dataset/seed=$seed split=$split model=$model_path prompt_mode=m5_qwen35_train"
cd "$EVAL_DIR"
"$PY" run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir "$data_dir" \
  --dataset_name "$dataset" \
  --split "$split" \
  --save_dir "$save_dir" \
  --save_note "$save_note" \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model "$model_path" \
  --apply_chat True \
  --prompt_mode m5_qwen35_train \
  --enable_thinking True
