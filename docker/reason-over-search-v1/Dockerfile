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
LABEL org.opencontainers.image.description="Conda env `reason-over-search` (Python 3.10)"

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

# Run on every Vast boot before SSH auth so key perms are accepted.
COPY docker/reason-over-search-v1/10-fix-ssh-perms.sh /etc/vast_boot.d/10-fix-ssh-perms.sh
RUN chmod +x /etc/vast_boot.d/10-fix-ssh-perms.sh

# Retrieval server settings
ENV RETRIEVER_PORT=3005
ENV NUM_RETRIEVER=1

EXPOSE 3005

RUN env-hash > /.env_hash
