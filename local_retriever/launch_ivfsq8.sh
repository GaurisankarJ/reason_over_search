#!/usr/bin/env bash
# Start / stop / health-check the IVF-SQ8 retriever — single-instance or fleet.
# After the async fix in retriever_serving.py, FastAPI's thread pool handles
# concurrency within each process; we still run a paired fleet (one retriever
# per SGLang server) to avoid OMP-thread oversubscription on multi-GPU sweeps.
#
# Single-instance usage (smoke / debugging):
#   local_retriever/launch_ivfsq8.sh start [port]
#   local_retriever/launch_ivfsq8.sh stop  [port]
#   local_retriever/launch_ivfsq8.sh wait  [port] [timeout_seconds]
#
# Fleet usage (full 8-GPU sweep):
#   local_retriever/launch_ivfsq8.sh start_fleet [count=8]
#   local_retriever/launch_ivfsq8.sh wait_fleet  [timeout=600]
#   local_retriever/launch_ivfsq8.sh stop_fleet
#
# Tunables (env vars):
#   OMP_RETRIEVER     — OMP_NUM_THREADS per retriever process (default 8).
#                       Cap so 8 procs × OMP_RETRIEVER ≤ host_cores; otherwise
#                       FAISS will oversubscribe (~24 cores per search default).
#   FLEET_PIDFILE     — pidfile listing fleet pids (default /tmp/retriever_fleet.pids).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RETRIEVER_DIR="$REPO_ROOT/local_retriever"
PY="${PY:-/venv/retriever/bin/python}"
INDEX="${INDEX:-$RETRIEVER_DIR/indexes/wiki18_100w_e5_ivf4096_sq8.index}"
OMP_RETRIEVER="${OMP_RETRIEVER:-8}"
FLEET_PIDFILE="${FLEET_PIDFILE:-/tmp/retriever_fleet.pids}"

ACTION="${1:?missing action: start|stop|wait|start_fleet|wait_fleet|stop_fleet}"

# Launch one retriever process bound to $port. Echoes pid on stdout, log path on stderr.
_launch_one() {
  local port="$1"
  local logfile="/tmp/retriever_ivfsq8_${port}.log"
  if [[ ! -f "$INDEX" ]]; then
    echo "[retriever] index not found: $INDEX" >&2
    echo "[retriever] download per docs/setup/BOOTSTRAP_NEW_INSTANCE.md step 4" >&2
    return 1
  fi
  echo "[retriever] starting on port $port (OMP=$OMP_RETRIEVER, log: $logfile)" >&2
  cd "$RETRIEVER_DIR"
  OMP_NUM_THREADS="$OMP_RETRIEVER" nohup "$PY" retriever_serving.py \
    --config retriever_config.yaml \
    --num_retriever 1 \
    --index "$INDEX" \
    --port "$port" \
    > "$logfile" 2>&1 &
  echo $!
}

# Wait until /health on $port returns 200, up to $timeout seconds.
_wait_one() {
  local port="$1"
  local timeout="${2:-300}"
  local elapsed=0
  echo "[retriever] waiting up to ${timeout}s for /health on port $port..."
  while (( elapsed < timeout )); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[retriever] healthy on port $port"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "[retriever] timeout after ${timeout}s on port $port" >&2
  return 1
}

start_retriever() {
  local port="${1:-3005}"
  local pidfile="/tmp/retriever_ivfsq8_${port}.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "[retriever] already running pid $(cat "$pidfile") on port $port"
    return 0
  fi
  local pid
  pid="$(_launch_one "$port")"
  echo "$pid" > "$pidfile"
  echo "[retriever] launched pid $pid"
}

stop_retriever() {
  local port="${1:-3005}"
  local pidfile="/tmp/retriever_ivfsq8_${port}.pid"
  if [[ ! -f "$pidfile" ]]; then
    echo "[retriever] no pidfile $pidfile"
    return 0
  fi
  local pid
  pid="$(cat "$pidfile")"
  echo "[retriever] stopping pid $pid (port $port)"
  kill "$pid" 2>/dev/null || true
  sleep 5
  if kill -0 "$pid" 2>/dev/null; then
    echo "[retriever] force kill $pid"
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pidfile"
}

start_fleet() {
  local count="${1:-8}"
  : > "$FLEET_PIDFILE"
  for ((i=0; i<count; i++)); do
    local port=$((3005 + i))
    local pid
    pid="$(_launch_one "$port")"
    echo "$pid" >> "$FLEET_PIDFILE"
    # also write the per-port pidfile so single-instance stop works for any one
    echo "$pid" > "/tmp/retriever_ivfsq8_${port}.pid"
  done
  echo "[retriever] fleet of $count launched on ports 3005..$((3005 + count - 1)); pids in $FLEET_PIDFILE"
}

wait_fleet() {
  local timeout="${1:-600}"
  local count
  count="$(wc -l < "$FLEET_PIDFILE" 2>/dev/null | awk '{print $1}')"
  count="${count:-8}"
  for ((i=0; i<count; i++)); do
    _wait_one "$((3005 + i))" "$timeout" || return 1
  done
  echo "[retriever] fleet of $count ready"
}

stop_fleet() {
  if [[ ! -f "$FLEET_PIDFILE" ]]; then
    echo "[retriever] no fleet pidfile at $FLEET_PIDFILE"
    return 0
  fi
  local pids
  pids="$(tr '\n' ' ' < "$FLEET_PIDFILE")"
  echo "[retriever] stopping fleet pids: $pids"
  kill $pids 2>/dev/null || true
  sleep 5
  local remaining=""
  for p in $pids; do
    if kill -0 "$p" 2>/dev/null; then
      remaining="$remaining $p"
    fi
  done
  if [[ -n "$remaining" ]]; then
    echo "[retriever] force kill:$remaining"
    kill -9 $remaining 2>/dev/null || true
  fi
  rm -f "$FLEET_PIDFILE"
  # also clear per-port pidfiles
  rm -f /tmp/retriever_ivfsq8_300[5-9].pid /tmp/retriever_ivfsq8_3010.pid /tmp/retriever_ivfsq8_3011.pid /tmp/retriever_ivfsq8_3012.pid
  echo "[retriever] fleet stopped"
}

case "$ACTION" in
  start)        start_retriever "${2:-3005}" ;;
  stop)         stop_retriever  "${2:-3005}" ;;
  wait)         _wait_one       "${2:-3005}" "${3:-300}" ;;
  start_fleet)  start_fleet     "${2:-8}" ;;
  wait_fleet)   wait_fleet      "${2:-600}" ;;
  stop_fleet)   stop_fleet ;;
  *)
    cat >&2 <<EOF
usage:
  $0 start  [port]
  $0 stop   [port]
  $0 wait   [port] [timeout]
  $0 start_fleet [count=8]
  $0 wait_fleet  [timeout=600]
  $0 stop_fleet
EOF
    exit 2
    ;;
esac
