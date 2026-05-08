#!/usr/bin/env bash
# Run one (model_variant, dataset, seed) M4 evaluation using evaluation_qwen35/.
# Usage:
#   scripts/run_m4.sh <variant> <dataset> <seed> [data_dir] [test_sample_num]
#
# Variants — Qwen3.5-0.8B baseline (untrained; M4 floor for any future GRPO
# checkpoint trained with the same `<tool_call>` / `<tool_response>` prompt):
#   qwen3.5_0.8b           hybrid (instruct + thinking)  -> Qwen/Qwen3.5-0.8B
#   qwen3.5_0.8b_base      base (pretrained only)        -> Qwen/Qwen3.5-0.8B-Base
#
# Both variants use prompt_mode=qwen35 (QWEN35_0_8B_TEMPLATE: same prose as the
# M3 p1_basic_w_ex Qwen3 prompt with `<search>` ↔ `<tool_call>` and `<result>`
# ↔ `<tool_response>`). Hybrid runs with enable_thinking=True; base runs with
# enable_thinking=False (the chat template's `<think>\n\n</think>` auto-injection
# is harmless on a base model that ignores it; we toggle it to mirror how the
# eventual M4 trained checkpoints will be served).
#
# Pass test_sample_num=100 for the quick eval; omit for the full sweep.
# Writes results to evaluation_qwen35/results/<dataset>/.
# Skip if a metric_score.txt already exists for this save_note.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$REPO_ROOT/evaluation_qwen35"
PY="${PY:-/home/s4374886/.conda/envs/evaluation_search_r1/bin/python}"

variant="${1:?missing variant (qwen3.5_0.8b|qwen3.5_0.8b_base)}"
dataset="${2:?missing dataset}"
seed="${3:?missing seed}"
data_dir="${4:-$REPO_ROOT/data}"
test_sample_num="${5:-}"

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
  qwen3.5_0.8b)
    model_path="${QWEN35_0_8B_HYBRID_PATH:-$REPO_ROOT/eval/qwen3.5_0.8b}"
    enable_thinking=True
    ;;
  qwen3.5_0.8b_base)
    model_path="${QWEN35_0_8B_BASE_PATH:-$REPO_ROOT/eval/qwen3.5_0.8b_base}"
    enable_thinking=False
    ;;
  *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac

prompt_mode=qwen35

if [[ -n "$test_sample_num" ]]; then
  save_note="m4_${variant}_seed${seed}_n${test_sample_num}"
  sample_args=(--test_sample_num "$test_sample_num" --random_sample True --seed "$seed")
else
  save_note="m4_${variant}_seed${seed}"
  sample_args=()
fi
save_dir="$EVAL_DIR/results/$dataset"

if compgen -G "$save_dir/${dataset}_*_${save_note}/metric_score.txt" > /dev/null 2>&1; then
  existing="$(ls -d "$save_dir"/${dataset}_*_${save_note} | tail -1)"
  echo "[skip] $variant/$dataset/seed=$seed already done -> $existing"
  exit 0
fi

echo "[run]  $variant/$dataset/seed=$seed split=$split model=$model_path prompt_mode=$prompt_mode sample_num=${test_sample_num:-FULL}"
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
  --prompt_mode "$prompt_mode" \
  --enable_thinking "$enable_thinking" \
  "${sample_args[@]}"
