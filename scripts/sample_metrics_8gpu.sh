#!/usr/bin/env bash
# Sample host + GPU metrics every 30 s, append to docs/eval/plan_a_8gpu/system_metrics.tsv.
# Schema mirrors docs/plan_a_5090x4_metrics.tsv from the 4×5090 predecessor run,
# expanded to 8 GPUs. Writes header if the file is empty.
#
# Usage:
#   nohup scripts/sample_metrics_8gpu.sh > /tmp/sampler.log 2>&1 &
#
# To stop:
#   pkill -f sample_metrics_8gpu.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$REPO_ROOT/docs/eval/plan_a_8gpu/system_metrics.tsv}"
INTERVAL="${INTERVAL:-30}"
RETRIEVER_PORT="${RETRIEVER_PORT:-3005}"
mkdir -p "$(dirname "$OUT")"

if [[ ! -s "$OUT" ]]; then
  {
    printf 'ts_utc\tepoch'
    for i in 0 1 2 3 4 5 6 7; do
      printf '\tg%d_util\tg%d_mem' "$i" "$i"
    done
    printf '\tret_cpu\tret_rss_gb\tsg_sch_cpu_avg\testab_%d\tfree_gb\tcache_gb\tload1\tload5\n' "$RETRIEVER_PORT"
  } > "$OUT"
fi

retriever_pid() {
  pgrep -f "retriever_serving.py.*--port ${RETRIEVER_PORT}" | head -1 || true
}

while true; do
  ts="$(date -u +%H:%M:%S)"
  epoch="$(date -u +%s)"

  # GPUs 0..7 — utilisation,memory.used (MiB)
  gpu_csv="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
  gpu_fields=""
  i=0
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    util="$(echo "$line" | awk -F',' '{gsub(/ /, "", $1); print $1}')"
    mem="$(echo "$line"  | awk -F',' '{gsub(/ /, "", $2); print $2}')"
    gpu_fields="${gpu_fields}\t${util}\t${mem}"
    i=$((i + 1))
  done <<< "$gpu_csv"
  # pad to 8 GPUs if fewer cards present
  while (( i < 8 )); do
    gpu_fields="${gpu_fields}\t\t"
    i=$((i + 1))
  done

  # retriever
  rpid="$(retriever_pid)"
  if [[ -n "$rpid" ]]; then
    ret_cpu="$(ps -p "$rpid" -o %cpu= 2>/dev/null | awk '{print int($1)}')"
    ret_rss_kb="$(ps -p "$rpid" -o rss= 2>/dev/null | awk '{print $1}')"
    ret_rss_gb="$(awk -v k="$ret_rss_kb" 'BEGIN{printf "%.1f", k/1048576}')"
  else
    ret_cpu=""
    ret_rss_gb=""
  fi

  # SGLang scheduler average %CPU (sum / count) — comm gets truncated to "sglang::schedu" on linux ps
  sg_lines="$(ps -eo pid,comm,%cpu --no-headers 2>/dev/null | grep -E 'sglang::schedu|sglang.launch' || true)"
  if [[ -n "$sg_lines" ]]; then
    sg_sch_cpu_avg="$(echo "$sg_lines" | awk '{s+=$3; n++} END{if(n>0) printf "%d", s/n}')"
  else
    sg_sch_cpu_avg=""
  fi

  # established conns to retriever port
  estab="$(ss -tn state established "( sport = :${RETRIEVER_PORT} or dport = :${RETRIEVER_PORT} )" 2>/dev/null | tail -n +2 | wc -l | awk '{print $1}')"

  # mem (Linux)
  if [[ -r /proc/meminfo ]]; then
    free_kb="$(awk '/^MemFree:/ {print $2}'   /proc/meminfo)"
    cache_kb="$(awk '/^Cached:/ {print $2; exit}' /proc/meminfo)"
    free_gb="$(awk -v k="$free_kb"  'BEGIN{printf "%d", k/1048576}')"
    cache_gb="$(awk -v k="$cache_kb" 'BEGIN{printf "%d", k/1048576}')"
  else
    free_gb=""
    cache_gb=""
  fi

  # load
  if [[ -r /proc/loadavg ]]; then
    read -r load1 load5 _ < /proc/loadavg
  else
    load1=""
    load5=""
  fi

  printf '%s\t%s%b\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$ts" "$epoch" "$gpu_fields" \
    "$ret_cpu" "$ret_rss_gb" "$sg_sch_cpu_avg" "$estab" \
    "$free_gb" "$cache_gb" "$load1" "$load5" \
    >> "$OUT"

  sleep "$INTERVAL"
done
