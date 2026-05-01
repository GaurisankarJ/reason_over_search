# reason-over-search v1 — base environment only (no auto-started services).
# Extends Vast’s image for good layer cache; add v2 / v3 sibling folders for other envs.
# See: https://hub.docker.com/r/vastai/base-image/
#
# Pin base to linux/amd64: Vast hosts and “expected amd64” builds are x86_64. Without this, a
# `docker build` on Apple Silicon produces an arm64 image, which will not run as a base (or
# on Vast) for amd64-only workflows.

ARG VAST_BASE_TAG=cuda-12.8.1-auto
ARG TARGETPLATFORM=linux/amd64
FROM --platform=${TARGETPLATFORM} vastai/base-image:${VAST_BASE_TAG}

LABEL org.opencontainers.image.title="reason-over-search-v1"
LABEL org.opencontainers.image.version="1"
LABEL org.opencontainers.image.description="Conda envs (retriever Py3.10, evaluation_search_r1 Py3.11) + uv-managed NeMo-RL training env (Py3.13, baked at build time)"

ENV APP_ROOT=/app
WORKDIR ${APP_ROOT}

# Conda env #1: `retriever` + pip dependencies (aligns with local_retriever README)
ENV PIP_NO_CACHE_DIR=1
COPY local_retriever/requirements.txt /tmp/retriever-requirements.txt
RUN /opt/miniforge3/bin/conda create -n retriever python=3.10 -y && \
    /opt/miniforge3/bin/conda run -n retriever \
      python -m pip install --no-cache-dir -r /tmp/retriever-requirements.txt && \
    rm -f /tmp/retriever-requirements.txt && \
    /opt/miniforge3/bin/conda clean -afy

# Conda env #2: `evaluation_search_r1` + eval stack dependencies.
# eval requirements currently pin ipython==9.10.1, which requires Python >=3.11.
COPY evaluation_search_r1/requirements.txt /tmp/evaluation-requirements.txt
RUN /opt/miniforge3/bin/conda create -n evaluation_search_r1 python=3.11 -y && \
    /opt/miniforge3/bin/conda run -n evaluation_search_r1 \
      python -m pip install --no-cache-dir -r /tmp/evaluation-requirements.txt && \
    rm -f /tmp/evaluation-requirements.txt && \
    /opt/miniforge3/bin/conda clean -afy

# App tree (data dirs excluded via root .dockerignore; mount or copy at runtime)
COPY local_retriever ${APP_ROOT}/local_retriever
COPY evaluation_search_r1 ${APP_ROOT}/evaluation_search_r1
RUN /opt/miniforge3/bin/conda run -n evaluation_search_r1 \
      /bin/bash -lc "cd /app/evaluation_search_r1 && python setup.py develop --no-deps"

# Training env (Milestone 2): NeMo-RL @ pinned tag, installed via uv.
# Same paradigm as the conda envs above — fully baked in so a Vast.ai instance
# starts with everything ready. The clone is gitignored on the host (.dockerignore
# also excludes training/nemo_rl/ from build context); the Dockerfile clones
# NeMo-RL itself so the image is reproducible from a clean checkout.
ARG NEMO_RL_REF=v0.6.0
ARG UV_EXTRAS=vllm

# 1. Install uv (Astral's pip replacement; NeMo-RL's official tooling)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv && \
    ln -s /root/.local/bin/uvx /usr/local/bin/uvx

# 2. Copy our overlay (configs / src / scripts / setup.sh / patches/).
#    .dockerignore excludes training/nemo_rl/ and training/.venv so the host clone
#    (if present) is not shipped — Dockerfile owns the clone.
COPY training ${APP_ROOT}/training
RUN chmod +x ${APP_ROOT}/training/setup.sh

# 3. Clone NeMo-RL with submodules at the pinned ref, apply any patches/, strip .git.
#    Single RUN keeps the layer cohesive — if you change NEMO_RL_REF the whole
#    train install layer rebuilds (which is what you want).
RUN git clone --recursive --branch ${NEMO_RL_REF} \
        https://github.com/NVIDIA-NeMo/RL.git ${APP_ROOT}/training/nemo_rl && \
    if ls ${APP_ROOT}/training/patches/*.patch >/dev/null 2>&1; then \
        cd ${APP_ROOT}/training/nemo_rl && \
        for p in ${APP_ROOT}/training/patches/*.patch; do \
            echo "applying patch $p" && git apply --3way "$p"; \
        done; \
    fi && \
    find ${APP_ROOT}/training/nemo_rl -name ".git" -prune -exec rm -rf {} + 2>/dev/null || true

# 4. Install: uv venv (Py 3.13) + uv sync with extras. This is the heavy step —
#    pulls torch 2.10, vLLM 0.17, Megatron-LM core, etc. Cached as one layer so
#    rebuilds only re-trigger if NEMO_RL_REF or UV_EXTRAS change.
#    `uv cache clean` at the end trims the wheel cache to reduce final image size.
RUN cd ${APP_ROOT}/training/nemo_rl && \
    uv venv && \
    EXTRA_FLAGS="" && \
    for e in $(echo ${UV_EXTRAS} | tr ',' ' '); do EXTRA_FLAGS="${EXTRA_FLAGS} --extra ${e}"; done && \
    uv sync ${EXTRA_FLAGS} && \
    uv cache clean

# 5. Convenience: ensure the training venv is on PATH for interactive shells
ENV TRAINING_VENV=${APP_ROOT}/training/nemo_rl/.venv

# Run on every Vast boot before SSH auth so key perms are accepted.
COPY docker/reason-over-search-v1/10-fix-ssh-perms.sh /etc/vast_boot.d/10-fix-ssh-perms.sh
RUN chmod +x /etc/vast_boot.d/10-fix-ssh-perms.sh

# Retrieval server settings
ENV RETRIEVER_PORT=3005
ENV NUM_RETRIEVER=1

EXPOSE 3005

RUN env-hash > /.env_hash
