# reason-over-search v1 (Docker)

## Build (from repository root)

```bash
docker build -f docker/reason-over-search-v1/Dockerfile -t reason-over-search-v1:v1 .
```

## Run an interactive shell

The Vast base image has its own `ENTRYPOINT`. For a normal shell, override it, then use conda and your app from `/app`.

```bash
docker run --rm -it --entrypoint /bin/bash \
  -p 3005:3005 \
  -v "$(pwd)/local_retriever/models":/app/local_retriever/models:ro \
  -v "$(pwd)/local_retriever/indexes":/app/local_retriever/indexes:ro \
  -v "$(pwd)/local_retriever/corpus":/app/local_retriever/corpus:ro \
  reason-over-search-v1:v1
```

Inside the container:

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate retriever
cd /app/local_retriever

python retriever_serving.py --config retriever_config_mini.yaml --num_retriever 1 --port 3005
```

In another shell on the host, check health:

```bash
curl -sS http://127.0.0.1:3005/health
```

## One-liner (no shell session)

```bash
docker run --rm -it -p 3005:3005 \
  -v "$(pwd)/local_retriever/models":/app/local_retriever/models:ro \
  -v "$(pwd)/local_retriever/indexes":/app/local_retriever/indexes:ro \
  -v "$(pwd)/local_retriever/corpus":/app/local_retriever/corpus:ro \
  --entrypoint /opt/miniforge3/bin/conda \
  reason-over-search-v1:v1 \
  run --no-capture-output -n retriever python /app/local_retriever/retriever_serving.py \
  --config /app/local_retriever/retriever_config_mini.yaml --port 3005 --num_retriever 1
```

## Build OOMs on pip/conda

Raise Docker / Colima / Desktop **RAM (8G+)** and try again.

## Push (e.g. Vast or Hub)

```bash
docker tag reason-over-search-v1:v1 pantomiman/reason-over-search-v1:v1
docker login
docker push pantomiman/reason-over-search-v1:v1
```

On Vast, you can `ssh` in or use their terminal, `conda activate retriever`, and run the same `python` command as above.
