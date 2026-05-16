#!/usr/bin/env bash
# One-shot system bootstrap for M5.5 on a fresh Verda B300 (or any bare
# Ubuntu 24.04 box with B300 GPUs + driver 580 + CUDA 13.0 default toolkit).
#
# Bakes in every fix learned during the 2026-05-15 bring-up. Full root-cause
# discussion: docs/setup/B300_RUNBOOK.md.
#
# Idempotent: each step checks if its outcome is already true and skips if so.
# Run from anywhere; cd's to the repo root before doing work.
#
# Usage:
#   bash training_m5_5/scripts/bootstrap_b300.sh
#
# Knobs (env):
#   NVTE_CUDA_ARCHS    arch list passed to transformer-engine. Default "90;120"
#                      (Hopper + Blackwell-Ultra). Must include at least one of
#                      70/75/80/89/90 — passing only "120" fails because TE
#                      strips it into NVTE_SPECIFIC_ARCHS and leaves
#                      CMAKE_CUDA_ARCHITECTURES empty.
#   MAX_JOBS           parallel compile jobs. Default 32 (60 vCPU on Verda).
#   SKIP_QWEN_CACHE    skip pre-downloading Qwen3.5-0.8B (default: no).
#   HF_TOKEN           if set, used for HF downloads (faster, higher rate limit).
#                      Auto-loaded from training_m5_5/.env if present. Used for
#                      both step 7 (V2 venv tarball) and step 8 (Qwen weights).
#
# Idempotency: skips steps whose outcome already holds. Safe to re-run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Load secrets early so HF_TOKEN is available to both step 7 (V2 venv tarball)
# and step 8 (Qwen weight cache). training_m5_5/.env is gitignored.
ENV_FILE="${REPO_ROOT}/training_m5_5/.env"
if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
fi

# -----------------------------------------------------------------------------
NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-90;120}"
MAX_JOBS="${MAX_JOBS:-32}"
SKIP_QWEN_CACHE="${SKIP_QWEN_CACHE:-}"

ok()   { printf '\033[1;32m[ok]\033[0m   %s\n' "$*"; }
info() { printf '\033[1;34m[bs]\033[0m   %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[fail]\033[0m %s\n' "$*"; exit 1; }

interactive() { [[ -t 0 ]] && [[ -t 1 ]]; }

# ============================================================
info "step 1/9 — apt prereqs (ninja, cmake, tmux, IB dev, build-essential)"
# ============================================================
APT_NEEDED=()
command -v ninja >/dev/null 2>&1 || APT_NEEDED+=(ninja-build)
command -v tmux  >/dev/null 2>&1 || APT_NEEDED+=(tmux)
[[ -f /usr/include/infiniband/mlx5dv.h ]] || APT_NEEDED+=(libibverbs-dev libmlx5-1 libnuma-dev rdma-core)
# build-essential for cc/c++ that some wheels invoke
command -v c++ >/dev/null 2>&1 || APT_NEEDED+=(build-essential)

if [[ ${#APT_NEEDED[@]} -gt 0 ]]; then
    info "installing: ${APT_NEEDED[*]}"
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${APT_NEEDED[@]}"
fi
ok "apt prereqs present"

# ============================================================
info "step 2/9 — CUDA 12.9 toolkit (torch 2.10 is cu129; default cuda-13.0 mismatches)"
# ============================================================
NVCC_REAL="$(readlink -f /usr/local/cuda/bin/nvcc 2>/dev/null || true)"
if [[ "${NVCC_REAL}" != *cuda-12.9* ]]; then
    if [[ ! -d /usr/local/cuda-12.9 ]]; then
        info "installing cuda-toolkit-12-9 (apt)"
        ( cd /tmp
          wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
          dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null
          apt-get update -qq
          DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-12-9 )
    fi
    info "repointing /usr/local/cuda → cuda-12.9"
    rm -f /etc/alternatives/cuda /usr/local/cuda
    ln -s /usr/local/cuda-12.9 /etc/alternatives/cuda
    ln -s /usr/local/cuda-12.9 /usr/local/cuda
fi
/usr/local/cuda/bin/nvcc --version | tail -1
ok "CUDA 12.9 default at $(readlink -f /usr/local/cuda)"

# ============================================================
info "step 3/9 — cuDNN dev headers (transformer-engine pytorch ext needs cudnn.h)"
# ============================================================
if [[ ! -f /usr/include/x86_64-linux-gnu/cudnn.h && ! -f /usr/include/cudnn.h ]]; then
    info "installing cudnn9-cuda-12 + libcudnn9-dev-cuda-12"
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cudnn9-cuda-12 libcudnn9-dev-cuda-12 || \
        warn "apt cudnn install failed; TE pytorch ext will fall back to venv-bundled cuDNN if accessible"
fi
ok "cudnn.h reachable"

# ============================================================
info "step 4/9 — uv (Astral) + system-wide symlinks (Ray actors need it on PATH)"
# ============================================================
if ! command -v uv >/dev/null 2>&1; then
    info "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
[[ -x /usr/local/bin/uv ]]  || ln -sf "$(command -v uv)"  /usr/local/bin/uv
[[ -x /usr/local/bin/uvx ]] || ln -sf "$(command -v uvx)" /usr/local/bin/uvx
ok "uv $(uv --version | awk '{print $2}') visible system-wide"

# ============================================================
info "step 5/9 — cmake 4.x (Ubuntu 24.04 ships 3.28; sm_120 needs 3.31+)"
# ============================================================
CMAKE_VER="$(cmake --version 2>/dev/null | head -1 | awk '{print $3}' | cut -d. -f1)"
if [[ "${CMAKE_VER:-0}" -lt 4 ]]; then
    info "installing cmake 4.x via uv tool"
    uv tool install --force cmake >/dev/null
    if [[ -e /usr/bin/cmake && ! -L /usr/bin/cmake ]]; then
        mv /usr/bin/cmake /usr/bin/cmake.apt
    elif [[ -L /usr/bin/cmake ]]; then
        rm /usr/bin/cmake
    fi
    ln -sf "$HOME/.local/bin/cmake" /usr/bin/cmake
fi
cmake --version | head -1
ok "cmake supports sm_120"

# ============================================================
info "step 6/9 — main NeMo-RL venv (training_m5_5/setup.sh)"
# ============================================================
if [[ ! -x training_m5_5/nemo_rl/.venv/bin/python ]]; then
    info "running training_m5_5/setup.sh"
    bash training_m5_5/setup.sh
fi
training_m5_5/nemo_rl/.venv/bin/python -c "import torch; print(f'  torch={torch.__version__} cuda={torch.version.cuda} devices={torch.cuda.device_count()}')"
ok "main venv built"

# ============================================================
info "step 7/9 — V2 worker venv (tarball fast-path → falls back to source compile)"
# ============================================================
#
# Two paths to materialize the DTensorPolicyWorkerV2 venv:
#   a. FAST: download the pre-built tarball from HF dataset
#      `pantomiman/reason-over-search-v1-venvs` and extract (~3-5 min for
#      ~5 GB). Smoke-test that imports work on B300 (sm_120) — the tarball
#      was built on Hopper-class hardware, so the cuda kernels may not have
#      sm_120 SASS but should have PTX for JIT fallback. If any import
#      fails (no PTX, ABI drift, etc.), wipe and fall through to (b).
#   b. SLOW: compile transformer-engine + nv-grouped-gemm + deep-ep +
#      causal-conv1d + mamba-ssm from source (~15-25 min). Runs from the
#      host shell, not a Ray actor — nv-grouped-gemm.setup.py calls
#      torch.cuda.init() at install time and the actor has no GPU.
#
# Override:
#   V2_BUILD_FROM_SOURCE=1   skip the tarball path; always compile
#   V2_VENV_HF_REPO=<repo>   override the HF repo for the tarball
V2_VENV="${REPO_ROOT}/training/nemo_rl/venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"
V2_VENV_HF_REPO="${V2_VENV_HF_REPO:-pantomiman/reason-over-search-v1-venvs}"
V2_VENV_HF_FILE="${V2_VENV_HF_FILE:-dtensor_policy_worker_v2.tar.gz}"

# Smoke-test the V2 venv at runtime: import the kernels we actually use.
# More aggressive than Vast's `import nemo_automodel` because we want to
# catch sm_120 mismatches (PTX missing → "no kernel image available") early.
v2_imports_ok() {
    [[ -x "${V2_VENV}/bin/python" ]] || return 1
    "${V2_VENV}/bin/python" - <<'PY' 2>/dev/null
import sys
try:
    import torch
    import transformer_engine
    import causal_conv1d
    import mamba_ssm
    import deep_ep
    import nemo_automodel
    # quick smoke that CUDA actually works at this arch
    assert torch.cuda.is_available()
    torch.zeros(1, device="cuda")
    print("OK")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
PY
}

build_from_source() {
    info "compiling V2 venv (NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS}, MAX_JOBS=${MAX_JOBS}; ~15-25 min cold)"
    rm -rf "${V2_VENV}"
    rm -rf /root/.cache/uv/git-v0/checkouts/*/build 2>/dev/null || true
    env \
      PATH="/root/.local/bin:/usr/local/bin:/usr/local/cuda-12.9/bin:/usr/bin:/bin" \
      CUDA_HOME=/usr/local/cuda-12.9 \
      LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 \
      NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS}" \
      TORCH_CUDA_ARCH_LIST="9.0;12.0+PTX" \
      MAX_JOBS="${MAX_JOBS}" \
      CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}" \
      UV_PROJECT_ENVIRONMENT="${V2_VENV}" \
      uv sync --locked --extra automodel --directory training_m5_5/nemo_rl
}

if v2_imports_ok; then
    ok "V2 worker venv already usable"
elif [[ "${V2_BUILD_FROM_SOURCE:-0}" == "1" ]]; then
    build_from_source
else
    info "attempting fast-path: tarball from HF ${V2_VENV_HF_REPO}:${V2_VENV_HF_FILE}"
    if [[ -n "${HF_TOKEN:-}" ]]; then
        info "  (authenticated via HF_TOKEN from .env)"
    else
        warn "  no HF_TOKEN in env — anonymous download (may rate-limit on large tarball)"
    fi
    rm -rf "${V2_VENV}"
    mkdir -p "$(dirname "${V2_VENV}")"
    TARBALL_DIR="$(mktemp -d)"
    VENV_HF="${REPO_ROOT}/training_m5_5/nemo_rl/.venv/bin/hf"
    # Use parent venv's hf CLI if available, else fall back to uv tool's hf
    [[ -x "${VENV_HF}" ]] || VENV_HF="$(command -v hf)"
    if HF_HUB_ENABLE_HF_TRANSFER=1 HF_TOKEN="${HF_TOKEN:-}" \
            "${VENV_HF}" download "${V2_VENV_HF_REPO}" "${V2_VENV_HF_FILE}" \
            --repo-type dataset --local-dir "${TARBALL_DIR}" 2>&1 | tail -3; then
        info "extracting (~2 min, ~5 GB → ~10 GB on disk)"
        tar -xzf "${TARBALL_DIR}/${V2_VENV_HF_FILE}" -C "$(dirname "${V2_VENV}")"
        rm -rf "${TARBALL_DIR}"
        if v2_imports_ok; then
            ok "V2 venv usable via tarball (saved ~20-30 min vs compile)"
        else
            warn "tarball extracted but imports failed on B300 (likely no sm_120 SASS/PTX). Wiping + compiling."
            rm -rf "${V2_VENV}"
            build_from_source
        fi
    else
        warn "HF tarball download failed; falling back to source compile"
        rm -rf "${TARBALL_DIR}"
        build_from_source
    fi
fi
v2_imports_ok || fail "V2 venv built but final smoke import fails — inspect ${V2_VENV}"
ok "V2 worker venv ready"

# ============================================================
info "step 8/9 — Qwen3.5-0.8B weights in HF cache (avoid slow anonymous download at first launch)"
# ============================================================
if [[ -z "${SKIP_QWEN_CACHE}" ]]; then
    QWEN_CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B"
    QWEN_SAFETENSORS="$(find "$QWEN_CACHE/snapshots" -name 'model.safetensors*' 2>/dev/null | head -1)"
    if [[ -n "${QWEN_SAFETENSORS}" && -s "${QWEN_SAFETENSORS}" ]]; then
        ok "Qwen3.5-0.8B already cached at ${QWEN_SAFETENSORS}"
    else
        if [[ -n "${HF_TOKEN:-}" ]]; then
            info "downloading Qwen3.5-0.8B (authenticated via HF_TOKEN from .env)"
        else
            info "downloading Qwen3.5-0.8B (anonymous — may rate-limit)"
        fi
        # Prefer the parent venv's hf CLI; install if missing
        VENV_PY="${REPO_ROOT}/training_m5_5/nemo_rl/.venv/bin/python"
        "${VENV_PY}" -m pip install -q --upgrade "huggingface_hub[hf_xet]" hf_transfer 2>/dev/null || true
        VENV_HF="${REPO_ROOT}/training_m5_5/nemo_rl/.venv/bin/hf"

        try_download() {
            HF_HUB_ENABLE_HF_TRANSFER=1 HF_TOKEN="${HF_TOKEN:-}" \
                "${VENV_HF}" download Qwen/Qwen3.5-0.8B 2>&1 | tail -5
        }

        if ! try_download; then
            warn "HF download failed (rate-limit, expired token, or transient error)"
            if interactive; then
                printf '\nPaste an HF_TOKEN to retry, or press Enter to abort: '
                read -r USER_HF_TOKEN
                if [[ -n "${USER_HF_TOKEN}" ]]; then
                    info "retrying with provided HF_TOKEN"
                    if HF_TOKEN="${USER_HF_TOKEN}" try_download; then
                        # persist for future launches if not already saved
                        if ! grep -q '^HF_TOKEN=' "${REPO_ROOT}/training_m5_5/.env" 2>/dev/null; then
                            echo "HF_TOKEN=${USER_HF_TOKEN}" >> "${REPO_ROOT}/training_m5_5/.env"
                            chmod 600 "${REPO_ROOT}/training_m5_5/.env"
                            ok "HF_TOKEN saved to training_m5_5/.env"
                        fi
                    else
                        fail "authenticated HF download also failed — check token / network"
                    fi
                else
                    fail "no token provided; aborting. Re-run with HF_TOKEN=<…> in training_m5_5/.env, or SKIP_QWEN_CACHE=1."
                fi
            else
                fail "non-interactive shell and HF download failed. Set HF_TOKEN in training_m5_5/.env (re-run will source it) or SKIP_QWEN_CACHE=1 to defer to vLLM."
            fi
        fi
    fi
fi

# ============================================================
info "step 9/9 — retriever venv (faiss-cpu)"
# ============================================================
if [[ ! -x local_retriever/.venv_cpu/bin/python ]]; then
    info "creating local_retriever/.venv_cpu (python 3.10)"
    uv venv local_retriever/.venv_cpu --python 3.10
    uv pip install --python local_retriever/.venv_cpu/bin/python -q -r local_retriever/requirements.txt
fi
ok "retriever venv ready"

echo
ok "bootstrap complete — next:"
echo "    bash training_m5_5/scripts/start_b300.sh --dry-run    # validate pre-flight"
echo "    bash training_m5_5/scripts/start_b300.sh --mode smoke # 4-step smoke"
echo "    bash training_m5_5/scripts/start_b300.sh              # prod_b300 seed 42"
