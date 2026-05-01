#!/usr/bin/env bash
# Set up NeMo-RL for Search-R1-style GRPO training.
# Idempotent: safe to re-run. Skips clone + install if the venv is already healthy.
#
# What it does:
#   1. Installs uv if missing (Astral's Rust-based pip replacement; NeMo-RL's official tooling)
#   2. Clones NVIDIA-NeMo/RL at a pinned tag with submodules (Megatron-LM, Megatron-Bridge, Automodel, Gym)
#   3. Removes the cloned .git directory so it's not a nested repo
#   4. Creates a uv venv inside training/nemo_rl/.venv (Python 3.13)
#   5. Runs `uv sync` with the vllm extra (rollout backend)
#
# Run:
#   bash training/setup.sh                # default — pins to v0.6.0
#   NEMO_RL_REF=main bash training/setup.sh   # override with a branch / tag / commit hash
#   FORCE_RECLONE=1 bash training/setup.sh    # wipe nemo_rl/ and re-clone
#
# Reads:
#   NEMO_RL_REF       — git ref to check out (default: v0.6.0)
#   FORCE_RECLONE     — if set, delete training/nemo_rl/ before cloning
#   UV_EXTRAS         — extras passed to uv sync (default: "vllm")
#                       Common options: "vllm", "vllm,fsdp", "vllm,nemo_gym"
#
# Notes:
#   - The clone is gitignored (training/.gitignore). We don't track NeMo-RL source in this repo;
#     reproducibility comes from the pinned NEMO_RL_REF + uv.lock inside the clone.
#   - uv manages Python 3.13 itself — no system Python upgrade needed.

set -euo pipefail

NEMO_RL_REF="${NEMO_RL_REF:-v0.6.0}"
UV_EXTRAS="${UV_EXTRAS:-vllm}"
FORCE_RECLONE="${FORCE_RECLONE:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_RL_DIR="${SCRIPT_DIR}/nemo_rl"

log() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }

# 1. Install uv if missing
if ! command -v uv >/dev/null 2>&1; then
    log "uv not found — installing via the Astral installer"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer drops uv at ~/.local/bin/uv (or ~/.cargo/bin/uv on some setups).
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv installed but not on PATH. Add ~/.local/bin (or ~/.cargo/bin) to PATH and re-run."
        exit 1
    fi
fi
log "uv version: $(uv --version)"

# 2. Clone NeMo-RL (or reuse existing clone)
if [[ -n "${FORCE_RECLONE}" && -d "${NEMO_RL_DIR}" ]]; then
    log "FORCE_RECLONE set — removing existing ${NEMO_RL_DIR}"
    rm -rf "${NEMO_RL_DIR}"
fi

if [[ ! -d "${NEMO_RL_DIR}" ]]; then
    log "cloning NVIDIA-NeMo/RL @ ${NEMO_RL_REF} into ${NEMO_RL_DIR} (with submodules)"
    git clone --recursive --branch "${NEMO_RL_REF}" \
        https://github.com/NVIDIA-NeMo/RL.git "${NEMO_RL_DIR}"

    # 3. Apply any local patches BEFORE removing .git (git apply is more forgiving inside a repo).
    PATCHES_DIR="${SCRIPT_DIR}/patches"
    if [[ -d "${PATCHES_DIR}" ]]; then
        shopt -s nullglob
        patches=( "${PATCHES_DIR}"/*.patch )
        shopt -u nullglob
        if (( ${#patches[@]} > 0 )); then
            log "applying ${#patches[@]} patch(es) from ${PATCHES_DIR}"
            for p in "${patches[@]}"; do
                log "  $(basename "${p}")"
                ( cd "${NEMO_RL_DIR}" && git apply --3way "${p}" ) || {
                    echo "ERROR: failed to apply ${p}. Resolve and rerun with FORCE_RECLONE=1."
                    exit 1
                }
            done
        fi
    fi

    # 4. Remove .git so this isn't a nested repo. Submodule .git dirs go too.
    log "removing .git directories so it's not a nested repo"
    find "${NEMO_RL_DIR}" -name ".git" -prune -exec rm -rf {} + 2>/dev/null || true
else
    log "reusing existing ${NEMO_RL_DIR} (set FORCE_RECLONE=1 to start over)"
fi

# 4 + 5. Create uv venv and install
cd "${NEMO_RL_DIR}"
if [[ ! -d ".venv" ]]; then
    log "creating uv venv"
    uv venv
fi

log "installing NeMo-RL with extras: ${UV_EXTRAS}"
# uv sync respects the lockfile; --extra adds optional dependency groups.
EXTRA_ARGS=()
IFS=',' read -r -a EXTRAS <<< "${UV_EXTRAS}"
for e in "${EXTRAS[@]}"; do
    EXTRA_ARGS+=(--extra "${e}")
done
uv sync "${EXTRA_ARGS[@]}"

log "done."
log ""
log "Activate the env:"
log "  source ${NEMO_RL_DIR}/.venv/bin/activate"
log ""
log "Or run commands via uv directly (recommended):"
log "  cd ${NEMO_RL_DIR}"
log "  uv run python examples/run_grpo.py --help"
