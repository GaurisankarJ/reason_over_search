#!/usr/bin/env bash
# Run one (model_variant, dataset, seed) evaluation.
# Usage:
#   scripts/run_one.sh <base|instruct> <dataset> <seed> [data_dir]
#
# Looks up split/apply_chat from the canonical mapping.
# Writes results to evaluation_search_r1/results/<dataset>/.
# Skips if a metric_score.txt already exists for this save_note (resume-friendly).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$REPO_ROOT/evaluation_search_r1"
PY="${PY:-/venv/evaluation_search_r1/bin/python}"

variant="${1:?missing variant (base|instruct|qwen_25_3b_instruct)}"
dataset="${2:?missing dataset}"
seed="${3:?missing seed}"
data_dir="${4:-$REPO_ROOT/data}"
SGL_PORT="${SGL_PORT:-3000}"
RETRIEVER_URL="${RETRIEVER_URL:-127.0.0.1:3005}"

# Canonical split per dataset (matches Search-R1: test if exists, else dev).
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
  base)
    apply_chat=True
    generator_model=search_r1_base_model
    ;;
  instruct)
    apply_chat=True
    generator_model=search_r1_instruct_model
    ;;
  qwen_25_3b_instruct)
    apply_chat=True
    generator_model=qwen_25_3b_instruct
    ;;
  *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac

save_note="search_r1_${variant}_seed${seed}"
save_dir="$EVAL_DIR/results/$dataset"

# Resume: if a previous run with this save_note produced a metric file, skip.
if compgen -G "$save_dir/${dataset}_*_${save_note}/metric_score.txt" > /dev/null; then
  existing="$(ls -d "$save_dir"/${dataset}_*_${save_note} | tail -1)"
  echo "[skip] $variant/$dataset/seed=$seed already done -> $existing"
  exit 0
fi

echo "[run]  $variant/$dataset/seed=$seed split=$split apply_chat=$apply_chat"
cd "$EVAL_DIR"
"$PY" run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir "$data_dir" \
  --dataset_name "$dataset" \
  --split "$split" \
  --save_dir "$save_dir" \
  --save_note "$save_note" \
  --sgl_remote_url 127.0.0.1:$SGL_PORT \
  --remote_retriever_url "$RETRIEVER_URL" \
  --generator_model "$generator_model" \
  --apply_chat "$apply_chat"
