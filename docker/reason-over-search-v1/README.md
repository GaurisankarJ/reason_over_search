# reason-over-search v1 (Docker)

## What this image contains

- Conda env `retriever` for `local_retriever/`
- Conda env `evaluation_search_r1` (Python 3.11) for `evaluation_search_r1/`
- App code copied to `/app/local_retriever` and `/app/evaluation_search_r1`
- Eval package installed in editable mode during build: `python setup.py develop --no-deps`
- **`uv` + NeMo-RL @ `v0.6.0` baked in** at `/app/training/nemo_rl/` with the `vllm` extra installed (Milestone 2). Venv at `/app/training/nemo_rl/.venv/` (Python 3.13). No setup needed at runtime.

## Build (from repository root)

The `Dockerfile` uses `FROM --platform=linux/amd64` so the result matches **Vast.ai** and typical cloud GPUs (**x86_64**), even when you build on an **Apple Silicon** machine. The first such build on a Mac can be slower (QEMU) while it runs the `RUN` steps.

```bash
docker build -f docker/reason-over-search-v1/Dockerfile -t reason-over-search-v1:v1 .
```

**If a remote builder still complains about base platform**, use Buildx and load into the local engine:

```bash
docker buildx build --platform linux/amd64 \
  -f docker/reason-over-search-v1/Dockerfile \
  -t reason-over-search-v1:v1 --load .
```

Pushing a previously **arm64-only** image to Hub and reusing it as a base for an **amd64** build causes `InvalidBaseImagePlatform`. Rebuild with `linux/amd64` and push that tag.

This image also includes a Vast boot hook that normalizes `/root/.ssh/authorized_keys` permissions on startup to avoid OpenSSH rejecting key auth with:
`Authentication refused: bad ownership or modes for file /root/.ssh/authorized_keys`.

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

# Retriever env
conda activate retriever
cd /app/local_retriever

python retriever_serving.py --config retriever_config_mini.yaml --num_retriever 4 --port 3005
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
  --config /app/local_retriever/retriever_config_mini.yaml --port 3005 --num_retriever 4
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

## Run evaluation env (inside container)

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate evaluation_search_r1
cd /app/evaluation_search_r1

# Example:
python run_eval.py \
  --config_path flashrag/config/basic_config.yaml \
  --method_name search-r1 \
  --data_dir /app/data \
  --dataset_name bamboogle \
  --split test \
  --save_dir /app/evaluation_search_r1/results/bamboogle \
  --save_note search_r1_base \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model search_r1_base_model \
  --apply_chat False
```

## Run training env (inside container)

NeMo-RL is **already installed and ready** — the venv lives at `/app/training/nemo_rl/.venv/` (Python 3.13).

```bash
# Option A — activate explicitly
source /app/training/nemo_rl/.venv/bin/activate
cd /app/training/nemo_rl
python examples/run_grpo.py --help

# Option B — uv (no activation needed)
cd /app/training/nemo_rl
uv run python examples/run_grpo.py --help
```

The training env coexists with `retriever` and `evaluation_search_r1` because uv manages its own Python 3.13 venv inside `training/nemo_rl/.venv/` — it doesn't conflict with the conda envs.

### Pinning a different NeMo-RL version

The Dockerfile pins via build args. To rebuild against a different ref or with extra optional deps:

```bash
docker build \
  --build-arg NEMO_RL_REF=main \
  --build-arg UV_EXTRAS=vllm,nemo_gym \
  -f docker/reason-over-search-v1/Dockerfile \
  -t reason-over-search-v1:v1 .
```

For local dev *outside* docker, see [`training/README.md`](../../training/README.md) — `training/setup.sh` does the same clone + install and is the path to use on a Mac without Docker.

## Vast SSH troubleshooting

If you see:

- `No ED25519 host key is known ... and you have requested strict checking`
- `Authentication refused: bad ownership or modes for file /root/.ssh/authorized_keys`

use:

```bash
# First connect once and accept/add host key (or use StrictHostKeyChecking=accept-new)
ssh -o StrictHostKeyChecking=accept-new -p <PORT> root@ssh<HOST>.vast.ai
```

Then your tunnel command works:

```bash
ssh -o StrictHostKeyChecking=accept-new -p <PORT> root@ssh<HOST>.vast.ai -L 8080:localhost:8080
```
