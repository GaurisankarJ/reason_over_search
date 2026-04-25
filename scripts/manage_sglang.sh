#!/usr/bin/env bash
# Start / stop / wait-for-ready helpers for the SGLang server.
# Usage:
#   scripts/manage_sglang.sh stop
#   scripts/manage_sglang.sh start base|instruct
#   scripts/manage_sglang.sh wait
#   scripts/manage_sglang.sh switch base|instruct   # = stop + start + wait
#
# Notes:
#   * Loads model from evaluation_search_r1/search_r1_{base,instruct}_model.
#   * Listens on 127.0.0.1:3000 with --context-length 8192 to match the README.
#   * Sends server stdout/stderr to /tmp/sglang_<variant>.log.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$REPO_ROOT/evaluation_search_r1"
PORT=3000
HOST=127.0.0.1
PY="${PY:-/venv/evaluation_search_r1/bin/python}"

stop_sglang() {
  local pids
  pids="$(pgrep -f 'sglang.launch_server' || true)"
  if [[ -n "$pids" ]]; then
    echo "[sglang] stopping pids: $pids"
    kill $pids 2>/dev/null || true
    sleep 5
    pids="$(pgrep -f 'sglang.launch_server' || true)"
    if [[ -n "$pids" ]]; then
      echo "[sglang] force kill: $pids"
      kill -9 $pids 2>/dev/null || true
    fi
  else
    echo "[sglang] no running server"
  fi
  # GPU memory takes a few seconds to release
  sleep 5
}

start_sglang() {
  local variant="$1"
  local model_path served
  case "$variant" in
    base)
      model_path="$EVAL_DIR/search_r1_base_model"
      served="search_r1_base"
      ;;
    instruct)
      model_path="$EVAL_DIR/search_r1_instruct_model"
      served="search_r1_instruct"
      ;;
    *)
      echo "[sglang] unknown variant: $variant" >&2
      exit 2
      ;;
  esac
  local logfile="/tmp/sglang_${variant}.log"
  echo "[sglang] starting $variant from $model_path (log: $logfile)"
  cd "$EVAL_DIR"
  nohup "$PY" -m sglang.launch_server \
    --served-model-name "$served" \
    --model-path "$model_path" \
    --tp 1 \
    --context-length 8192 \
    --enable-metrics \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --disable-overlap \
    --disable-radix-cache \
    > "$logfile" 2>&1 &
  echo "[sglang] launched pid $!"
}

wait_ready() {
  local timeout="${1:-300}"
  local elapsed=0
  echo "[sglang] waiting up to ${timeout}s for /get_model_info..."
  while (( elapsed < timeout )); do
    if curl -sS --max-time 2 "http://${HOST}:${PORT}/get_model_info" >/dev/null 2>&1; then
      echo "[sglang] ready"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "[sglang] timeout after ${timeout}s" >&2
  return 1
}

switch() {
  stop_sglang
  start_sglang "$1"
  wait_ready 600
}

case "${1:-}" in
  stop) stop_sglang ;;
  start) start_sglang "${2:?missing variant}" ;;
  wait) wait_ready "${2:-300}" ;;
  switch) switch "${2:?missing variant}" ;;
  *) echo "usage: $0 {stop|start <variant>|wait [timeout]|switch <variant>}" >&2; exit 2 ;;
esac
