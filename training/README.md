# Training

NeMo-RL setup for Search-R1-style GRPO training of Qwen3.5-2B (base + hybrid). Scoped in [`docs/milestone_two/MILESTONE_2.md`](../docs/milestone_two/MILESTONE_2.md).

## Quick start

### 1. Run the setup script (first time, locally or on Vast.ai)

```bash
bash training/setup.sh
```

This installs `uv` (NeMo-RL's official package manager), clones [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) at the pinned tag (default: `v0.6.0`) into `training/nemo_rl/` with all 4 submodules (Megatron-LM, Megatron-Bridge, Automodel, Gym), removes the cloned `.git` so it's not a nested repo, and runs `uv sync --extra vllm` to install NeMo-RL with the vLLM rollout backend.

The clone is **gitignored** (large; reproducibility comes from the pinned `NEMO_RL_REF` in [`setup.sh`](setup.sh) plus `uv.lock` inside the clone).

The script is idempotent — re-running skips the clone if `nemo_rl/` exists.

#### Knobs

```bash
NEMO_RL_REF=main bash training/setup.sh        # use main branch instead of v0.6.0
NEMO_RL_REF=<commit-sha> bash training/setup.sh
FORCE_RECLONE=1 bash training/setup.sh         # wipe nemo_rl/ and start over
UV_EXTRAS="vllm,nemo_gym" bash training/setup.sh    # add NeMo Gym for tool integration Path B
```

### 2. Activate the env

After setup, NeMo-RL's venv lives at `training/nemo_rl/.venv/`. Two ways to use it:

```bash
# Option A — activate explicitly (familiar to most users)
source training/nemo_rl/.venv/bin/activate

# Option B — uv handles activation for you (recommended)
cd training/nemo_rl/
uv run python examples/run_grpo.py --help
```

### 3. (Future) launch training

Training scripts live in `training/scripts/` (not yet written — Milestone 2 step 6). They will load configs from `training/configs/` and invoke NeMo-RL via `uv run`.

## Layout (planned)

```
training/
├── setup.sh                    # idempotent NeMo-RL clone + install (DONE)
├── .gitignore                  # ignores nemo_rl/, .env, .venv (DONE)
├── README.md                   # this file
├── nemo_rl/                    # cloned by setup.sh, gitignored
│   └── .venv/                  # uv-managed Python 3.13 venv
├── configs/                    # GRPO + memory configs (TBD — Milestone 2 step 6)
├── scripts/                    # 1× A100 and 2× A100 launch scripts (TBD)
├── src/                        # Search-R1-style chat templates, reward, retrieval env (TBD)
│   ├── chat_template/
│   ├── environments/           # search_r1_env.py — Ray actor that bridges <tool_call> → /batch_search
│   └── reward/
└── .env                        # W&B key (gitignored, user-supplied)
```

## What `setup.sh` does, step by step

1. Installs `uv` if missing (Astral's installer; lands at `~/.local/bin/uv`).
2. Clones `NVIDIA-NeMo/RL` at `${NEMO_RL_REF}` (default `v0.6.0`) into `training/nemo_rl/` with `--recursive` (submodules: Megatron-LM, Megatron-Bridge, Automodel, Gym).
3. Removes the cloned `.git` and submodule `.git` directories so it's not a nested repo (per Milestone 2 design — local edits are tracked in *this* repo, not upstream's).
4. Runs `uv venv` to create a Python 3.13 venv inside `training/nemo_rl/.venv/` (uv downloads Python 3.13 itself if not on PATH).
5. Runs `uv sync --extra vllm` to install all of NeMo-RL's pinned deps (torch 2.10, transformers 5.3, ray 2.54, vLLM 0.17, etc.) plus the editable NeMo-RL package.

Total disk after install: roughly **6–8 GB** (most of it torch + vLLM + cudnn). On Vast.ai this fits comfortably in the 150 GB persistent storage Milestone 2 requires.

## Running in Docker

The image `pantomiman/reason-over-search-v1` (built from [`docker/reason-over-search-v1/`](../docker/reason-over-search-v1/)) ships `uv` pre-installed and `training/` copied to `/app/training/`. The heavy install (clone + `uv sync`) is **deferred to runtime** to keep the image small and avoid freezing a NeMo-RL version into the image.

Inside a running container:

```bash
bash /app/training/setup.sh
# then either:
source /app/training/nemo_rl/.venv/bin/activate
# or:
cd /app/training/nemo_rl && uv run python ...
```

## See also

- [`docs/training/NEMO_RL_KNOBS.md`](../docs/training/NEMO_RL_KNOBS.md) — config knobs + concrete starting yaml for 1× A100 80GB
- [`docs/training/CHAT_TEMPLATE.md`](../docs/training/CHAT_TEMPLATE.md) — Qwen3.5 native tool-call template (the baseline)
- [`docs/training/TRAINING_DATA.md`](../docs/training/TRAINING_DATA.md) — `PeterJinGo/nq_hotpotqa_train` schema + conversion recipe
- [`docs/training/PAPER_VS_OURS_TRAINING.md`](../docs/training/PAPER_VS_OURS_TRAINING.md) — divergences from paper, with rationale
- [`docs/training/VERL_REFERENCE.md`](../docs/training/VERL_REFERENCE.md) — porting reference distilled from verl-tested scripts
- [`docs/training/VALIDATION.md`](../docs/training/VALIDATION.md) — in-loop validation plan
