# reason-over-search v1.
#
# Hybrid paradigm:
#   - retriever  + evaluation_search_r1: code AND env baked into image (stable,
#     no need to clone the repo on Vast for these to work).
#   - training (NeMo-RL): only the env (uv wheel cache + uv installed) is
#     baked. The repo is cloned on Vast and the venv is materialized at
#     training/nemo_rl/.venv via `uv sync`. Reason: training is the active
#     iteration surface; rebuilding the image on every code edit is wasteful.
#
# Pin base to linux/amd64: Vast hosts and "expected amd64" builds are x86_64.
# Without this, a `docker build` on Apple Silicon produces an arm64 image,
# which will not run as a base (or on Vast) for amd64-only workflows.

ARG VAST_BASE_TAG=cuda-12.8.1-auto
ARG TARGETPLATFORM=linux/amd64
FROM --platform=${TARGETPLATFORM} vastai/base-image:${VAST_BASE_TAG}

LABEL org.opencontainers.image.title="reason-over-search-v1"
LABEL org.opencontainers.image.version="1"
LABEL org.opencontainers.image.description="Conda envs for retriever + evaluation_search_r1 (with code baked in); uv + pre-warmed NeMo-RL wheel cache for training (code cloned at runtime)."

ENV APP_ROOT=/app
WORKDIR ${APP_ROOT}

# === 1. Conda env `retriever` (Python 3.10) — code + env baked in ===
ENV PIP_NO_CACHE_DIR=1
COPY local_retriever/requirements.txt /tmp/retriever-requirements.txt
RUN /opt/miniforge3/bin/conda create -n retriever python=3.10 -y && \
    /opt/miniforge3/bin/conda run -n retriever \
      python -m pip install --no-cache-dir -r /tmp/retriever-requirements.txt && \
    rm -f /tmp/retriever-requirements.txt && \
    /opt/miniforge3/bin/conda clean -afy

# === 2. Conda env `evaluation_search_r1` (Python 3.11) — code + env baked in ===
# eval requirements pin ipython==9.10.1, which requires Python >=3.11.
COPY evaluation_search_r1/requirements.txt /tmp/evaluation-requirements.txt
RUN /opt/miniforge3/bin/conda create -n evaluation_search_r1 python=3.11 -y && \
    /opt/miniforge3/bin/conda run -n evaluation_search_r1 \
      python -m pip install --no-cache-dir -r /tmp/evaluation-requirements.txt && \
    rm -f /tmp/evaluation-requirements.txt && \
    /opt/miniforge3/bin/conda clean -afy

# Source trees for retriever + eval (data dirs excluded via .dockerignore;
# mount or download at runtime). Editable install of the eval package so
# `import flashrag` works without re-running setup at container start.
COPY local_retriever ${APP_ROOT}/local_retriever
COPY evaluation_search_r1 ${APP_ROOT}/evaluation_search_r1
RUN /opt/miniforge3/bin/conda run -n evaluation_search_r1 \
      /bin/bash -lc "cd /app/evaluation_search_r1 && python setup.py develop --no-deps"

# === 3. Training env: uv + pre-warmed NeMo-RL wheel cache (NO repo code) ===
# The training/nemo_rl/ source is committed to the repo and cloned at runtime
# on Vast. The image only needs (a) uv installed, and (b) NeMo-RL's wheels
# pre-downloaded into uv's cache so the on-Vast `uv sync` is fast.
ARG NEMO_RL_REF=v0.6.0
ARG UV_EXTRAS=vllm

# vastai/base-image already ships a `uv` at /usr/local/bin/uv; we still install
# our own copy via the Astral installer to pin a known version, then `ln -sf`
# to make sure the symlinks point at /root/.local/bin/* (not the base image's).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv && \
    ln -sf /root/.local/bin/uvx /usr/local/bin/uvx

# Pre-warm wheels for NeMo-RL. We download every wheel that an
# `uv sync --extra vllm` would need into a flat directory at /opt/uv-wheels/,
# then set UV_FIND_LINKS so any subsequent `uv sync` (on Vast, against the
# user's cloned training/nemo_rl/) picks them up locally instead of going to
# PyPI.
#
# Why download (no install) — the install path would extract wheels into a
# temporary venv (~3GB unpack of torch alone) inside the build container's
# writable layer, which OOMs the Docker VM on Apple Silicon under QEMU.
# Download-only never extracts; it just persists ~5GB of .whl files.
RUN git clone --depth=1 --branch ${NEMO_RL_REF} --recursive \
        https://github.com/NVIDIA-NeMo/RL.git /tmp/nemo-rl-deps && \
    cd /tmp/nemo-rl-deps && \
    EXTRA_FLAGS="" && \
    for e in $(echo ${UV_EXTRAS} | tr ',' ' '); do EXTRA_FLAGS="${EXTRA_FLAGS} --extra ${e}"; done && \
    # Resolve the lockfile to a flat requirements.txt with all transitive deps pinned.
    uv export ${EXTRA_FLAGS} --no-dev --no-emit-project --format requirements-txt -o /tmp/nemo-rl-reqs.txt && \
    # Download every wheel to /opt/uv-wheels/ (no extraction, no venv).
    mkdir -p /opt/uv-wheels && \
    uv pip download --python-version 3.13 --no-deps -r /tmp/nemo-rl-reqs.txt -d /opt/uv-wheels && \
    rm -rf /tmp/nemo-rl-deps /tmp/nemo-rl-reqs.txt

# Tell uv to look in /opt/uv-wheels/ before going to PyPI. On Vast, when the
# user runs `cd training/nemo_rl && uv sync --extra vllm`, uv will resolve
# every dep against this directory and skip PyPI for anything found.
ENV UV_FIND_LINKS=/opt/uv-wheels

# === 4. Vast boot hook — fix SSH key permissions so OpenSSH accepts auth ===
COPY docker/reason-over-search-v1/10-fix-ssh-perms.sh /etc/vast_boot.d/10-fix-ssh-perms.sh
RUN chmod +x /etc/vast_boot.d/10-fix-ssh-perms.sh

# === Runtime defaults ===
ENV RETRIEVER_PORT=3005
ENV NUM_RETRIEVER=1
EXPOSE 3005

RUN env-hash > /.env_hash
