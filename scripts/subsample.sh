#!/usr/bin/env bash
# Create a subsampled copy of the evaluation data under data_subsample/.
# For datasets with very large eval splits, take a deterministic random sample.
# Bamboogle (125), MuSiQue (2,417) stay full size — already small.
# Usage:
#   scripts/subsample.sh [N=1000] [SEED=42]

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N="${1:-1000}"
SEED="${2:-42}"
SRC="$REPO_ROOT/data"
DST="$REPO_ROOT/data_subsample"

mkdir -p "$DST"

# Map: dataset -> split file to subsample
declare -A SPLITS=(
  [nq]=test
  [triviaqa]=test
  [popqa]=test
  [hotpotqa]=dev
  [2wikimultihopqa]=dev
)

for ds in "${!SPLITS[@]}"; do
  split="${SPLITS[$ds]}"
  src_file="$SRC/$ds/$split.jsonl"
  dst_dir="$DST/$ds"
  dst_file="$dst_dir/$split.jsonl"
  mkdir -p "$dst_dir"
  if [[ -s "$dst_file" ]]; then
    echo "[subsample] $ds/$split.jsonl already present ($(wc -l < "$dst_file") rows) — skipping"
    continue
  fi
  total="$(wc -l < "$src_file")"
  if (( total <= N )); then
    cp "$src_file" "$dst_file"
    echo "[subsample] $ds/$split.jsonl: copied all $total rows (<= N=$N)"
  else
    /venv/main/bin/python - "$src_file" "$dst_file" "$N" "$SEED" <<'PY'
import sys, random
src, dst, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(src) as f:
    lines = f.readlines()
random.Random(seed).shuffle(lines)
with open(dst, 'w') as f:
    f.writelines(lines[:n])
PY
    echo "[subsample] $ds/$split.jsonl: sampled $N from $total (seed=$SEED)"
  fi
done

# Bamboogle and MuSiQue: full copy (small).
for ds in bamboogle musique; do
  split=test
  [[ "$ds" == "musique" ]] && split=dev
  mkdir -p "$DST/$ds"
  if [[ ! -s "$DST/$ds/$split.jsonl" ]]; then
    cp "$SRC/$ds/$split.jsonl" "$DST/$ds/$split.jsonl"
    echo "[subsample] $ds/$split.jsonl: copied full ($(wc -l < "$SRC/$ds/$split.jsonl") rows)"
  fi
done

echo "[subsample] done -> $DST"
