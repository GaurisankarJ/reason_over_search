#!/usr/bin/env bash
# PLAN A (1 seed) on 8×4090 + raw Qwen2.5-3B-Instruct baseline.
# 1 seed × 7 datasets × 3 variants = 21 runs.
# Estimated wall-clock: ~6 h (NQ-bound per variant ~2 h × 3 + ~5 min model swaps).
#
# Architecture:
#   * 8 paired IVF-SQ8 retriever processes on ports 3005..3012 (one per SGLang
#     server). Depends on the async fix in local_retriever/retriever_serving.py
#     and on OMP_NUM_THREADS being capped per-process (env OMP_RETRIEVER, default 8).
#   * 8 SGLang servers on ports 3000..3007, one per GPU. All 8 serve the SAME
#     variant per phase; we tear down + bring up between variants. Retriever
#     fleet stays up across all phases.
#   * 7 datasets dispatched in parallel across GPUs 0..6 (GPU 7 idle this phase).
#     NQ (~2 h, longest) is pinned to GPU 0 so it starts first.
#   * GPU N pairs with SGLang :300N and retriever :$((3005+N)).
#
# Variants in execution order:
#   1. base                — Search-R1 GRPO from Qwen2.5-3B-Base
#   2. instruct            — Search-R1 GRPO from Qwen2.5-3B-Instruct
#   3. qwen_25_3b_instruct — raw Qwen/Qwen2.5-3B-Instruct (un-tuned baseline)
#
# Usage:
#   nohup scripts/sweep_8gpu_one_seed.sh > /tmp/sweep_8gpu.log 2>&1 &
#   tail -f /tmp/sweep_8gpu.log

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="$REPO_ROOT/logs/sweep_8gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"

# Defensive HF cache pin — matches docker/reason-over-search-v1/Dockerfile:138.
# Ensures SGLang's hub-id model-paths (e.g. Qwen/Qwen2.5-3B-Instruct) land on the
# persistent /workspace volume, not the host's ~/.cache/huggingface.
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

SEED=1
DATASETS=(nq triviaqa popqa hotpotqa 2wikimultihopqa musique bamboogle)
VARIANTS=(base instruct qwen_25_3b_instruct)
RESULTS_OUT="$REPO_ROOT/docs/eval/plan_a_8gpu/RESULTS.md"
mkdir -p "$(dirname "$RESULTS_OUT")"

echo "=== PLAN A 8×4090: 1 seed × ${#DATASETS[@]} datasets × ${#VARIANTS[@]} variants = $((${#DATASETS[@]} * ${#VARIANTS[@]})) runs ==="
date
echo "log dir: $LOG_ROOT"

# Trap: ensure both fleets are stopped on any exit (success, failure, signal).
cleanup() {
  echo "[sweep] cleanup: stopping SGLang + retriever fleets"
  "$REPO_ROOT/scripts/manage_sglang.sh" stop_fleet || true
  "$REPO_ROOT/local_retriever/launch_ivfsq8.sh" stop_fleet || true
}
trap cleanup EXIT

# Retriever fleet (8 procs, ports 3005..3012) lives across all variant phases.
echo
echo "=== starting IVF-SQ8 retriever fleet ==="
"$REPO_ROOT/local_retriever/launch_ivfsq8.sh" start_fleet 8
"$REPO_ROOT/local_retriever/launch_ivfsq8.sh" wait_fleet 600

for variant in "${VARIANTS[@]}"; do
  echo
  echo "=== phase: $variant ==="
  date
  "$REPO_ROOT/scripts/manage_sglang.sh" start_fleet "$variant"
  "$REPO_ROOT/scripts/manage_sglang.sh" wait_fleet 900

  echo "[sweep] dispatching ${#DATASETS[@]} datasets across GPUs 0..$((${#DATASETS[@]} - 1))"
  for i in "${!DATASETS[@]}"; do
    ds="${DATASETS[$i]}"
    SGL_PORT=$((3000 + i)) \
    RETRIEVER_URL="127.0.0.1:$((3005 + i))" \
      "$REPO_ROOT/scripts/run_one.sh" "$variant" "$ds" "$SEED" \
      > "$LOG_ROOT/${variant}_${ds}.log" 2>&1 &
    echo "  GPU $i  SGLang :$((3000 + i))  retriever :$((3005 + i))  $variant/$ds  pid $!"
  done

  # One bad dataset shouldn't kill the rest of the sweep — capture rc but continue.
  set +e
  wait
  rc=$?
  set -e
  if (( rc != 0 )); then
    echo "[sweep] phase '$variant' had failed jobs (rc=$rc); see $LOG_ROOT/" >&2
  fi

  "$REPO_ROOT/scripts/manage_sglang.sh" stop_fleet
done

echo
echo "=== aggregating ==="
"$REPO_ROOT/scripts/aggregate.py" --output "$RESULTS_OUT"
echo "done — see ${RESULTS_OUT#$REPO_ROOT/}"
date
