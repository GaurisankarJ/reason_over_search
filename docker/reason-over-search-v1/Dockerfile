# reason-over-search v1 — base environment only (no auto-started services).
# Extends Vast’s image for good layer cache; add v2 / v3 sibling folders for other envs.
# See: https://hub.docker.com/r/vastai/base-image/

ARG VAST_BASE_TAG=cuda-12.8.1-auto
FROM vastai/base-image:${VAST_BASE_TAG}

LABEL org.opencontainers.image.title="reason-over-search-v1"
LABEL org.opencontainers.image.version="1"
LABEL org.opencontainers.image.description="Conda env `reason-over-search` (Python 3.10)"

ENV APP_ROOT=/app
WORKDIR ${APP_ROOT}

# Conda `retriever` + pip dependencies (aligns with local_retriever README)
ENV PIP_NO_CACHE_DIR=1
COPY local_retriever/requirements.txt /tmp/retriever-requirements.txt
RUN /opt/miniforge3/bin/conda create -n retriever python=3.10 -y && \
    /opt/miniforge3/bin/conda run -n retriever \
      python -m pip install --no-cache-dir -r /tmp/retriever-requirements.txt && \
    rm -f /tmp/retriever-requirements.txt && \
    /opt/miniforge3/bin/conda clean -afy

# App tree (data dirs excluded via root .dockerignore; mount or copy at runtime)
COPY local_retriever ${APP_ROOT}/local_retriever

# Retrieval server settings
ENV RETRIEVER_PORT=3005
ENV NUM_RETRIEVER=1

EXPOSE 3005

RUN env-hash > /.env_hash
