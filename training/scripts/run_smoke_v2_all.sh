#!/usr/bin/env bash
# Run all four Phase-2 smoke combos at "v2" scale (longer than SETUP_CLAUDE v1 smokes).
# After each run, renames logs/exp_NNN -> logs/smoke_v2_<combo> for extract_smoke_samples.py --suffix v2.
#
# v1 knobs (SETUP_CLAUDE): 2 steps × 4 prompts × GBS 20 → 40 traj/combo
# v2 knobs: 8 steps × 8 prompts × GBS 40 → 320 traj/combo
#
# Prerequisites: retriever on :3005, bootstrap + v2 venv done, training/.env for W&B.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
ulimit -n 65536

SMOKE_V2=(
  policy.sequence_packing.enabled=false
  policy.dynamic_batching.enabled=true
  policy.train_micro_batch_size=2
  grpo.max_num_steps=8
  grpo.num_prompts_per_step=8
  policy.train_global_batch_size=40
)

latest_exp_dir() {
  local n
  n="$(ls logs 2>/dev/null | grep -E '^exp_[0-9]+$' | sed 's/exp_//' | sort -n | tail -1)"
  # Force decimal: bare 008/009 break printf/%d (invalid octal).
  n="$((10#$n))"
  printf 'logs/exp_%03d' "$n"
}

move_after_run() {
  local combo="$1"
  local dest="logs/smoke_v2_${combo}"
  local src
  src="$(latest_exp_dir)"
  if [[ ! -d "$src" ]]; then
    echo "error: expected experiment dir under logs/, got: $src" >&2
    exit 1
  fi
  echo "[run_smoke_v2_all] mv $src -> $dest"
  rm -rf "$dest"
  mv "$src" "$dest"
}

run_one() {
  local variant="$1" arm="$2" combo="$3"
  echo "[run_smoke_v2_all] ========== ${variant} × ${arm} (${combo}) =========="
  rm -rf /tmp/ray
  bash training/scripts/run_grpo_1xa100.sh --variant "$variant" --seed 42 --arm "$arm" -- \
    "${SMOKE_V2[@]}"
  move_after_run "$combo"
}

run_one base qwen_native base_qwen_native
run_one base paper base_paper
run_one hybrid qwen_native hybrid_qwen_native
run_one hybrid paper hybrid_paper

echo "[run_smoke_v2_all] all four done. Run: python3 training/scripts/extract_smoke_samples.py --suffix v2"
