#!/usr/bin/env bash
# PLAN A — Full sweep: 5 seeds × 7 datasets × 2 model variants = 70 runs.
# Estimated wall-clock: ~17 days on a single 4090 with the current 1-worker retriever.
# Designed to be run with nohup and resumed if interrupted (run_one.sh skips
# already-completed runs).
#
# Usage:
#   nohup scripts/sweep_a_full.sh > /tmp/sweep_a.log 2>&1 &
#   tail -f /tmp/sweep_a.log

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEEDS="1 2 3 4 5"
DATASETS="bamboogle nq triviaqa popqa musique 2wikimultihopqa hotpotqa"

echo "=== PLAN A: 5 × 7 × 2 = 70 runs ==="
date

for variant in base instruct; do
  echo
  echo "=== switching SGLang to $variant ==="
  "$REPO_ROOT/scripts/manage_sglang.sh" switch "$variant"
  echo "=== running 5 seeds × 7 datasets on $variant ==="
  "$REPO_ROOT/scripts/run_variant_sweep.sh" "$variant" "$SEEDS" "$DATASETS"
done

echo
echo "=== aggregating ==="
"$REPO_ROOT/scripts/aggregate.py" --output "$REPO_ROOT/RESULTS_PLAN_A.md"
echo "done — see RESULTS_PLAN_A.md"
date
