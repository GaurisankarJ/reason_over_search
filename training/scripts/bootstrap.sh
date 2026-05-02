#!/usr/bin/env bash
# Idempotent one-shot bootstrap for a fresh `pantomiman/reason-over-search-v1:v1`
# Vast box. Runs everything that the docker image cannot bake (anything that
# needs GPU access at build time, or content that would balloon the image).
#
# Designed to be re-runnable: each step checks existing state and skips if
# already done. First run on a fresh box: ~30–45 min (the v2/automodel uv
# venv compile is the long pole, ~25 min). Subsequent runs: < 1 min.
#
# Usage:
#   bash training/scripts/bootstrap.sh                      # full setup
#   SKIP_V2_BUILD=1 bash training/scripts/bootstrap.sh      # skip the long step
#   SKIP_RETRIEVER=1 bash training/scripts/bootstrap.sh     # don't start retriever
#
# Exits non-zero on any unrecoverable failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GREEN=$'\033[0;32m'; YELLOW=$'\033[0;33m'; RED=$'\033[0;31m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
log()  { echo "${BOLD}${GREEN}▶${RESET} $*"; }
warn() { echo "${BOLD}${YELLOW}⚠${RESET} $*"; }
err()  { echo "${BOLD}${RED}✗${RESET} $*" >&2; }

# ---------------------------------------------------------------------------
# 0. Sanity
# ---------------------------------------------------------------------------
log "Sanity checks…"

if [[ ! -f "training/scripts/run_grpo_1xa100.sh" ]]; then
  err "Run from a clone of reason_over_search (or pass an explicit REPO_ROOT)."
  exit 1
fi

free_gb_workspace=$(df --output=avail -BG /workspace 2>/dev/null | tail -1 | tr -d 'G ' || echo 0)
if (( free_gb_workspace < 30 )); then
  warn "/workspace has only ${free_gb_workspace}G free; recommend ≥ 30 G."
fi

ram_gb=$(free -g | awk '/^Mem:/{print $2}')
if (( ram_gb < 150 )); then
  warn "Only ${ram_gb} GB RAM; IVF-SQ8 + 8 workers loads ~134 GB (8 × 16 GB index copies). Recommend ≥ 150 GB."
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  err "nvidia-smi not found — is this a GPU instance?"
  exit 1
fi
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
log "GPU: ${gpu_name}"

# Conda envs
if ! /opt/miniforge3/bin/conda env list 2>/dev/null | grep -q '^retriever '; then
  err "Conda env 'retriever' missing. This script expects pantomiman/reason-over-search-v1:v1 (or rebuild)."
  exit 1
fi

# .env
if [[ ! -f training/.env ]]; then
  if [[ -f training/.env.example ]]; then
    warn "training/.env missing — copying from .env.example. Fill WANDB_API_KEY before training."
    cp training/.env.example training/.env
  else
    warn "training/.env missing and no .env.example. Set WANDB_MODE=disabled if you don't have a key."
  fi
fi

# Git LFS for parquets / eval jsonls
if command -v git-lfs >/dev/null 2>&1; then
  if [[ ! -s data/training/nq_hotpotqa_train/train.parquet ]]; then
    log "Pulling Git LFS objects…"
    git lfs install >/dev/null
    git lfs pull
  fi
fi

# ---------------------------------------------------------------------------
# 1. HF model weights (~4 min if missing, instant if cached)
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "$HF_HOME"

need_models=0
for m in "models--Qwen--Qwen3.5-2B-Base" "models--Qwen--Qwen3.5-2B"; do
  if ! ls "$HF_HOME/hub/$m"/snapshots/*/model.safetensors-*.safetensors >/dev/null 2>&1; then
    need_models=1
    break
  fi
done

if (( need_models )); then
  log "Downloading Qwen3.5-2B-Base + Qwen3.5-2B to ${HF_HOME} (~8 GB)…"
  source /opt/miniforge3/etc/profile.d/conda.sh
  conda activate retriever
  hf download Qwen/Qwen3.5-2B-Base
  hf download Qwen/Qwen3.5-2B
  conda deactivate
else
  log "Qwen3.5 weights already cached at ${HF_HOME}."
fi

# ---------------------------------------------------------------------------
# 2. NeMo-RL project venv (training/nemo_rl/.venv) — fast, wheels from /.uv/cache
# ---------------------------------------------------------------------------
if [[ ! -x training/nemo_rl/.venv/bin/python ]] || \
   ! training/nemo_rl/.venv/bin/python -c "import nemo_rl" 2>/dev/null; then
  log "Materializing project venv at training/nemo_rl/.venv (uv sync --extra vllm)…"
  ( cd training/nemo_rl && uv sync --extra vllm )
else
  log "Project venv at training/nemo_rl/.venv already present."
fi

# ---------------------------------------------------------------------------
# 3. v2 / automodel uv venv
# Two paths to materialize this:
#   a. Fast path — download pre-built tarball from HuggingFace Hub and extract
#      (~3 min for 5 GB compressed). Default.
#   b. Slow path — compile transformer-engine + nv-grouped-gemm + deep_ep
#      from source via `uv sync --extra automodel`. ~25 min on first run.
# Both produce the same on-disk venv. The fast path is preferred because the
# slow path can't run inside NeMo-RL's lazy GPU-less Ray _env_builder actor:
# nv-grouped-gemm's setup.py calls torch.cuda.init() at install time.
# ---------------------------------------------------------------------------
V2_VENV="$REPO_ROOT/training/nemo_rl/venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"
V2_VENV_HF_REPO="${V2_VENV_HF_REPO:-pantomiman/reason-over-search-v1-venvs}"
V2_VENV_HF_FILE="${V2_VENV_HF_FILE:-dtensor_policy_worker_v2.tar.gz}"

if [[ "${SKIP_V2_BUILD:-0}" == "1" ]]; then
  warn "SKIP_V2_BUILD=1 — assuming the v2 venv exists or that you'll only use _v2=false."
elif [[ -x "$V2_VENV/bin/python" ]] && "$V2_VENV/bin/python" -c "import nemo_automodel" 2>/dev/null; then
  log "v2 (automodel) venv already built at $V2_VENV."
elif [[ "${V2_BUILD_FROM_SOURCE:-0}" == "1" ]]; then
  log "V2_BUILD_FROM_SOURCE=1 → compiling v2 venv from source (~25 min)…"
  ( cd training/nemo_rl
    export NEMO_RL_VENV_DIR="$REPO_ROOT/training/nemo_rl/venvs"
    uv venv --allow-existing "$V2_VENV"
    UV_PROJECT_ENVIRONMENT="$V2_VENV" uv sync --locked --extra automodel
  )
else
  log "Downloading pre-built v2 venv from HF dataset $V2_VENV_HF_REPO (~5 GB)…"
  source /opt/miniforge3/etc/profile.d/conda.sh
  conda activate retriever
  TARBALL_DIR="$(mktemp -d)"
  trap 'rm -rf "$TARBALL_DIR"' EXIT
  if hf download "$V2_VENV_HF_REPO" "$V2_VENV_HF_FILE" \
        --repo-type dataset --local-dir "$TARBALL_DIR" 2>&1 | tail -3; then
    log "Extracting tarball to $(dirname "$V2_VENV") (~2 min)…"
    mkdir -p "$(dirname "$V2_VENV")"
    tar -xzf "$TARBALL_DIR/$V2_VENV_HF_FILE" -C "$(dirname "$V2_VENV")"
    rm -rf "$TARBALL_DIR"
    trap - EXIT
    if [[ -x "$V2_VENV/bin/python" ]] && "$V2_VENV/bin/python" -c "import nemo_automodel" 2>/dev/null; then
      log "v2 (automodel) venv ready (downloaded path)."
    else
      err "Tarball extracted but venv doesn't import nemo_automodel. Falling back to source build."
      rm -rf "$V2_VENV"
      V2_BUILD_FROM_SOURCE=1 exec "$0" "$@"
    fi
  else
    warn "HF download failed or repo unavailable. Falling back to source build (~25 min)…"
    rm -rf "$TARBALL_DIR"
    trap - EXIT
    ( cd training/nemo_rl
      export NEMO_RL_VENV_DIR="$REPO_ROOT/training/nemo_rl/venvs"
      uv venv --allow-existing "$V2_VENV"
      UV_PROJECT_ENVIRONMENT="$V2_VENV" uv sync --locked --extra automodel
    )
  fi
  conda deactivate
fi

# ---------------------------------------------------------------------------
# 4. Retriever (IVF-SQ8 + 8 workers, v1 default)
# ---------------------------------------------------------------------------
if [[ "${SKIP_RETRIEVER:-0}" == "1" ]]; then
  warn "SKIP_RETRIEVER=1 — not starting the retriever."
else
  if curl -sS -m 2 http://127.0.0.1:3005/health 2>/dev/null | grep -q healthy; then
    log "Retriever already healthy at http://127.0.0.1:3005."
  else
    log "Starting IVF-SQ8 retriever with 8 workers (cold start ~30–60 s)…"
    pkill -9 -f "retriever_serving" 2>/dev/null || true
    rm -f /tmp/retriever.log
    (
      source /opt/miniforge3/etc/profile.d/conda.sh
      conda activate retriever
      cd "$REPO_ROOT/local_retriever"
      nohup python retriever_serving.py \
        --config retriever_config.yaml \
        --num_retriever 8 \
        --port 3005 \
        > /tmp/retriever.log 2>&1 &
      disown
    )
    # Wait for ready (poll every 5 s up to 5 min)
    log "Waiting for retriever to come up…"
    for i in $(seq 1 60); do
      if grep -q "Uvicorn running" /tmp/retriever.log 2>/dev/null; then
        break
      fi
      sleep 5
    done
    if ! grep -q "Uvicorn running" /tmp/retriever.log 2>/dev/null; then
      err "Retriever did not come up within 5 min. Last lines of /tmp/retriever.log:"
      tail -20 /tmp/retriever.log
      exit 1
    fi
  fi

  # Smoke-check the retriever
  health=$(curl -sS http://127.0.0.1:3005/health || echo "ERROR")
  if echo "$health" | grep -q '"status":"healthy"'; then
    log "Retriever health: $health"
  else
    err "Retriever health check failed: $health"
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log "${BOLD}Bootstrap complete.${RESET}"
echo
echo "Next: pick a combo and run training. Recommended smoke shape:"
echo "  bash training/scripts/run_grpo_1xa100.sh \\"
echo "    --variant {base|hybrid} --seed 42 --arm {qwen_native|paper} \\"
echo "    -- grpo.max_num_steps=2 grpo.num_prompts_per_step=4 policy.train_global_batch_size=20 \\"
echo "       policy.sequence_packing.enabled=false policy.dynamic_batching.enabled=true \\"
echo "       policy.train_micro_batch_size=2"
echo
echo "For real Phase-2 runs (1005 steps), drop the smoke overrides but keep:"
echo "       policy.sequence_packing.enabled=false policy.dynamic_batching.enabled=true \\"
echo "       policy.train_micro_batch_size=2"
