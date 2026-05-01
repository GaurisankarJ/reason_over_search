# Training

> **How the training paradigm works + Search-R1 vs Qwen3.5 differences:** see [docs/training/](../docs/training/).
>
> TL;DR — `pantomiman/reason-over-search-v1` ships `uv` + a pre-warmed NeMo-RL wheel cache. NeMo-RL source is committed at [`nemo_rl/`](nemo_rl/) (pinned to `v0.6.0`). On Vast: clone the repo, `cd training/nemo_rl`, `uv sync --extra vllm`, activate. Same pattern as `local_retriever/` and `evaluation_search_r1/` (env from image, code from repo).

## Environment Setup

The standard install path uses `uv` + the wheel cache pre-warmed in the docker image (`/root/.cache/uv/`). Inside the docker container:

```bash
cd training/nemo_rl
uv sync --extra vllm
source .venv/bin/activate
```

The `uv sync` materializes the venv at `training/nemo_rl/.venv/` (Python 3.13). On the Vast docker image this is fast (~30s–2min) because the wheels are cached. On a fresh local Mac/Linux setup with no cache, the first run downloads ~5 GB of wheels (10–20 min); subsequent runs hit uv's local cache.

If `uv` is missing entirely (e.g. running outside the docker image and outside any other env that has it), use the helper script:

```bash
bash training/setup.sh
```

`setup.sh` installs `uv` if needed, then runs the same `uv sync --extra ${UV_EXTRAS:-vllm}` against the committed NeMo-RL source.

### Adding optional dependency groups

NeMo-RL exposes extras: `vllm`, `fsdp`, `automodel`, `mcore`, `nemo_gym`, `sglang`, `nvrx`. Default is `vllm`. Add more:

```bash
UV_EXTRAS="vllm,nemo_gym" bash training/setup.sh
```

The same `UV_EXTRAS` is exposed at docker build time so the wheel cache covers it:

```bash
docker build --build-arg UV_EXTRAS=vllm,nemo_gym \
  -f docker/reason-over-search-v1/Dockerfile -t reason-over-search-v1:v1 .
```

### Bumping the pinned NeMo-RL version

```bash
NEMO_RL_REF=v0.7.0 FORCE_RECLONE=1 bash training/setup.sh
git add training/nemo_rl/
git commit -m "training: bump NeMo-RL to v0.7.0"
# rebuild the docker image so the wheel cache pre-warm matches:
docker build --build-arg NEMO_RL_REF=v0.7.0 \
  -f docker/reason-over-search-v1/Dockerfile -t reason-over-search-v1:v1 .
```

`FORCE_RECLONE=1` wipes any uncommitted local edits to `nemo_rl/`. To preserve a clean diff against upstream while bumping, drop a `*.patch` into [`patches/`](patches/) — `setup.sh` re-applies patches after re-clone.

## Download steps

### Training dataset

```bash
training/scripts/prepare_dataset.py
```

`prepare_dataset.py` is a `uv` inline-script (idempotent). It pulls the [`PeterJinGo/nq_hotpotqa_train`](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train) parquets via `hf_hub_download` (NQ + HotpotQA mix, 169,615 train + 51,713 test) and **strips the prebaked Search-R1 template** — replaces each row's `prompt[0].content` with the bare `question`. Output: `data/training/nq_hotpotqa_train/{train,test}.parquet`. Schema in [docs/training/TRAINING_DATA.md](../docs/training/TRAINING_DATA.md).

```bash
training/scripts/prepare_dataset.py --force   # regenerate even if outputs exist
```

### Models

```bash
cd training
mkdir -p models

# Install once if needed
pip install -U "huggingface_hub[cli]"

# Base
huggingface-cli download Qwen/Qwen3.5-2B-Base \
  --local-dir models/Qwen3.5-2B-Base \
  --local-dir-use-symlinks False

# Hybrid (default soft-switch reasoning)
huggingface-cli download Qwen/Qwen3.5-2B \
  --local-dir models/Qwen3.5-2B \
  --local-dir-use-symlinks False
```

## Configure W&B (one-time per Vast instance)

```bash
cat > training/.env <<'EOF'
WANDB_API_KEY=...
WANDB_ENTITY=...
WANDB_PROJECT=reason-over-search-training
EOF
source training/.env
```

`training/.env` is gitignored.

## Run Training

> **Status (2026-05-01)**: NeMo-RL env + dataset prep are wired up. Concrete launch scripts are pending — Milestone 2 step 6. The runs below are sketches; final commands will live in `training/scripts/`.

### Sanity check the env

```bash
cd training/nemo_rl
python -c "import nemo_rl; print(nemo_rl.__version__)"   # → 0.6.0
python examples/run_grpo.py --help
```

### Run NeMo-RL's bundled GRPO recipe (smoke test)

Useful to confirm the env works before wiring our Search-R1 retrieval env:

```bash
cd training/nemo_rl
uv run python examples/run_grpo.py --config=examples/configs/grpo_math_1B.yaml
```

### Run Search-R1 GRPO (TBD)

Will live at `training/scripts/run_grpo_qwen3.5_2b_{base,hybrid}_{1,2}_a100_80gb.sh`. Each script:

1. Reads `training/.env` for W&B keys.
2. Loads `training/configs/grpo_qwen3.5_2b_{base,hybrid}.yaml` (concrete starting yaml in [docs/training/NEMO_RL_KNOBS.md](../docs/training/NEMO_RL_KNOBS.md) §7).
3. Registers the Search-R1 retrieval env (`training/src/environments/search_r1_env.py`) via NeMo-RL's `register_env` (see [docs/training/NEMO_RL_KNOBS.md](../docs/training/NEMO_RL_KNOBS.md) §9).
4. Invokes `uv run python examples/run_grpo.py --config=...`.

Before launching, the retriever must be live (`local_retriever/README.md`, port 3005); the rollout env POSTs to `/batch_search`.

## Folder layout

```
training/
├── README.md                    # this file
├── setup.sh                     # installs uv + runs uv sync (local dev / docker fallback)
├── nemo_rl/                     # vendored NeMo-RL source @ v0.6.0 (committed)
│   └── .venv/                   # uv-managed Python 3.13 venv (gitignored, materialized on Vast)
├── scripts/
│   └── prepare_dataset.py       # download + strip Search-R1 template
├── patches/                     # optional *.patch overlays for nemo_rl/ (applied by setup.sh)
├── configs/                     # GRPO + memory configs (TBD — Milestone 2 step 6)
├── src/                         # Search-R1 chat template, reward, retrieval env (TBD — Milestone 2 step 4)
└── .env                         # W&B key (gitignored)
```

## See also

- [docs/training/NEMO_RL_KNOBS.md](../docs/training/NEMO_RL_KNOBS.md) — config knobs + concrete starting yaml for 1× A100 80GB
- [docs/training/CHAT_TEMPLATE.md](../docs/training/CHAT_TEMPLATE.md) — Qwen3.5 native tool-call template (the baseline)
- [docs/training/TRAINING_DATA.md](../docs/training/TRAINING_DATA.md) — `PeterJinGo/nq_hotpotqa_train` schema + conversion recipe
- [docs/training/PAPER_VS_OURS_TRAINING.md](../docs/training/PAPER_VS_OURS_TRAINING.md) — divergences from paper, with rationale
- [docs/training/VERL_REFERENCE.md](../docs/training/VERL_REFERENCE.md) — porting reference distilled from verl-tested scripts
- [docs/training/VALIDATION.md](../docs/training/VALIDATION.md) — in-loop validation plan
- [docs/milestone_two/MILESTONE_2.md](../docs/milestone_two/MILESTONE_2.md) — overall milestone scope
