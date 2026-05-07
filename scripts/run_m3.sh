#!/usr/bin/env bash
# Run one (model_variant, dataset, seed) M3 evaluation using evaluation_research/.
# Usage:
#   scripts/run_m3.sh <qwen3_0.6b|qwen3_0.6b_v0> <dataset> <seed> [data_dir]
#
# Both variants use prompt_mode=qwen3 (system-message format, <result> tags).
# Writes results to evaluation_research/results/<dataset>/.
# Skips if a metric_score.txt already exists for this save_note (resume-friendly).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$REPO_ROOT/evaluation_research"
PY="${PY:-/home/s4374886/.conda/envs/evaluation_search_r1/bin/python}"

variant="${1:?missing variant (qwen3_0.6b|qwen3_0.6b_v0)}"
dataset="${2:?missing dataset}"
seed="${3:?missing seed}"
data_dir="${4:-$REPO_ROOT/data}"

case "$dataset" in
  bamboogle)         split=test ;;
  nq)                split=test ;;
  triviaqa)          split=test ;;
  popqa)             split=test ;;
  hotpotqa)          split=dev  ;;
  2wikimultihopqa)   split=dev  ;;
  musique)           split=dev  ;;
  *) echo "unknown dataset: $dataset" >&2; exit 2 ;;
esac

case "$variant" in
  qwen3_0.6b)
    model_path="$REPO_ROOT/eval/qwen_3_0.6b"
    ;;
  qwen3_0.6b_v0)
    model_path="$REPO_ROOT/eval/qwen_3_0.6b_v0"
    ;;
  *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac

save_note="m3_${variant}_seed${seed}"
save_dir="$EVAL_DIR/results/$dataset"

if compgen -G "$save_dir/${dataset}_*_${save_note}/metric_score.txt" > /dev/null 2>&1; then
  existing="$(ls -d "$save_dir"/${dataset}_*_${save_note} | tail -1)"
  echo "[skip] $variant/$dataset/seed=$seed already done -> $existing"
  exit 0
fi

echo "[run]  $variant/$dataset/seed=$seed split=$split model=$model_path"
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
  --prompt_mode qwen3 \
  --enable_thinking True
