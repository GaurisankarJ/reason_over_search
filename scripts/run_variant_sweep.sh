#!/usr/bin/env bash
# For one model variant (already loaded in SGLang), iterate datasets × seeds.
# Usage:
#   scripts/run_variant_sweep.sh <base|instruct> "<seeds>" "<datasets>" [data_dir]
# Examples:
#   scripts/run_variant_sweep.sh base "1 2 3 4 5" "bamboogle nq triviaqa popqa musique 2wikimultihopqa hotpotqa"
#   scripts/run_variant_sweep.sh instruct "1" "bamboogle"
#
# Order: outer = seed, inner = dataset (so all datasets are touched per seed).
# Each (variant,dataset,seed) is resume-friendly via run_one.sh.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

variant="${1:?missing variant}"
seeds="${2:?missing seeds}"
datasets="${3:?missing datasets}"
data_dir="${4:-$REPO_ROOT/data}"

# Sanity: SGLang must be serving the requested variant.
expected_model="search_r1_${variant}"
served="$(curl -sS --max-time 5 http://127.0.0.1:3000/get_model_info | \
          /venv/main/bin/python -c 'import sys,json; print(json.load(sys.stdin).get("model_path",""))' || true)"
case "$served" in
  *"${expected_model}_model"*) : ;;  # ok
  *)
    echo "[sweep] SGLang is serving '$served', expected variant '$variant'" >&2
    echo "[sweep] run: scripts/manage_sglang.sh switch $variant" >&2
    exit 3
    ;;
esac

for seed in $seeds; do
  for ds in $datasets; do
    "$REPO_ROOT/scripts/run_one.sh" "$variant" "$ds" "$seed" "$data_dir"
  done
done
