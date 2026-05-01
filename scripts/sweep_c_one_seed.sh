#!/usr/bin/env bash
# PLAN C — 1 seed × full datasets × 2 variants = 14 runs.
# Estimated wall-clock: ~80 hours (~3.4 days) on this box.
#
# Usage:
#   nohup scripts/sweep_c_one_seed.sh > /tmp/sweep_c.log 2>&1 &
#   tail -f /tmp/sweep_c.log

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEEDS="1"
DATASETS="bamboogle nq triviaqa popqa musique 2wikimultihopqa hotpotqa"

echo "=== PLAN C: 1 seed × 7 datasets × 2 variants = 14 runs ==="
date

for variant in base instruct; do
  echo
  echo "=== switching SGLang to $variant ==="
  "$REPO_ROOT/scripts/manage_sglang.sh" switch "$variant"
  echo "=== running on $variant ==="
  "$REPO_ROOT/scripts/run_variant_sweep.sh" "$variant" "$SEEDS" "$DATASETS"
done

echo
echo "=== aggregating ==="
"$REPO_ROOT/scripts/aggregate.py" --output "$REPO_ROOT/docs/RESULTS_PLAN_C.md"
echo "done — see docs/RESULTS_PLAN_C.md"
date
