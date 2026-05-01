# Milestone 2: Training [Search-R1](https://www.alphaxiv.org/abs/2503.09516)

## Context

Search-R1 trains [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) (and 7B) with GRPO, on both **base** and **instruct** variants. We want to do the same training on **[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)**, but `verl` does not support Qwen3.5, so we port the Search-R1-style training loop to **[NeMo-RL](https://github.com/NVIDIA-NeMo/RL)**, which supports GRPO and RLVR for Qwen3.5.

The Qwen3.5 lineup differs from Qwen2.5: it has **base** and a default model with a **soft-switch reasoning mode** (`enable_thinking=true/false`) — there is no separate "instruct" variant. We will train both:
- **base** → [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base)
- **hybrid** (Qwen's default soft-switch reasoning model) → [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)

The **reward function** and **chat template** must be **decoupled from the core training code** so they can be swapped cleanly. In Milestone 2 we use the **Search-R1 baseline reward unchanged** — the goal is to verify training works on Qwen3.5 with the paper's reward before ablating in Milestone 3.

## Goal

Reproduce the Search-R1 training pipeline on Qwen3.5-2B, for both base and hybrid variants.

Use the official Search-R1 training dataset (linked from the paper).

Create a `training/` folder at the repo root containing the NeMo-RL setup with the Search-R1-style modifications. Add a new `training` env to the docker image.

## Step-by-step

1. **Set up the training env in docker (hybrid paradigm).** The image at [`docker/reason-over-search-v1/Dockerfile`](../../docker/reason-over-search-v1/Dockerfile) splits two concerns:
   - **Stable surfaces (`retriever`, `evaluation_search_r1`)**: code + conda env baked in, **as before**. No change.
   - **Active surface (`training` / NeMo-RL)**: env baked, code cloned at runtime. Image ships `uv` + a pre-warmed wheel cache for NeMo-RL. The repo gets `git clone`d on Vast; `uv sync` materializes the venv from the cached wheels in seconds-to-minutes.

   **Source layout**: NeMo-RL @ `v0.6.0` is **committed to this repo** at [`training/nemo_rl/`](../../training/nemo_rl/) with submodules (Megatron-LM, Megatron-Bridge, Automodel, Gym; `.git` stripped so it's not a nested repo). Local edits land in this repo's git history. Bump via [`training/setup.sh`](../../training/setup.sh) with `NEMO_RL_REF=<ref> FORCE_RECLONE=1`, commit the new tree, and rebuild the image with `--build-arg NEMO_RL_REF=<ref>` so the wheel cache matches.

   **Dockerfile** additions for Milestone 2:
   - Installs [`uv`](https://docs.astral.sh/uv/) (NeMo-RL's official package manager).
   - **Pre-warms `/root/.cache/uv/`** by shallow-cloning NeMo-RL @ `${NEMO_RL_REF}` into `/tmp`, running `uv sync --extra ${UV_EXTRAS} --no-install-project` to download all wheels, then deleting the temp clone. The wheels stay in the cache.
   - **Does not COPY `training/`** — the source comes from `git clone` on Vast. `.dockerignore` excludes `training/nemo_rl/` from the build context to keep `docker build` fast.

   Build + push:

   ```bash
   docker build \
     --build-arg NEMO_RL_REF=v0.6.0 \
     --build-arg UV_EXTRAS=vllm \
     -f docker/reason-over-search-v1/Dockerfile -t reason-over-search-v1:v1 .

   docker tag reason-over-search-v1:v1 pantomiman/reason-over-search-v1:v1
   docker login
   docker push pantomiman/reason-over-search-v1:v1
   ```

   **On Vast** (per fresh instance):

   ```bash
   cd /workspace
   git clone https://github.com/<your-user>/reason_over_search.git
   cd reason_over_search/training/nemo_rl
   uv sync --extra vllm                  # fast — wheel cache pre-warmed in image
   source .venv/bin/activate
   ```

   For local dev *outside* docker, [`training/setup.sh`](../../training/setup.sh) does the same `uv sync` (first run downloads ~5 GB; subsequent runs hit uv's local cache).

2. **Download and prep the Search-R1 training dataset.** Run [`training/scripts/prepare_dataset.py`](../../training/scripts/prepare_dataset.py) — uv inline-script, idempotent. It pulls [`PeterJinGo/nq_hotpotqa_train`](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train) (169,615 train + 51,713 test, parquet) and **strips the prebaked Search-R1 template**, replacing each row's `prompt[0].content` with the bare `question`. Output: `data/training/nq_hotpotqa_train/{train,test}.parquet` (gitignored). Schema details in [`docs/training/TRAINING_DATA.md`](../training/TRAINING_DATA.md).

   Stripping keeps the dataset **template-agnostic** so the chat template (Qwen3.5 native `<tool_call>` default vs. the paper's `<search>` ablation arm) is applied at **rollout time** via run config — not baked into the dataset. Rationale: [`CHAT_TEMPLATE.md`](../training/CHAT_TEMPLATE.md) §5.

3. **Map to NeMo-RL's expected row schema.** Done *after* (1), once NeMo-RL's data-loader requirements are visible inside `training/nemo_rl/examples/`. The current `prompt: [{role,content}]` shape may already be enough; if NeMo-RL expects `messages` or another shape, add a final reshape pass to [`prepare_dataset.py`](../../training/scripts/prepare_dataset.py) rather than introducing a second conversion script.

   > **TODO**: NeMo-RL's exact row-level data schema for GRPO. Inspect `examples/configs/` and `examples/run_grpo.py` in the cloned NeMo-RL repo (after step 1) and update [`TRAINING_DATA.md`](../training/TRAINING_DATA.md) with the verified mapping before running this step.

4. **Apply Search-R1-style modifications** as an **overlay** under `training/src/` (NOT inside `training/nemo_rl/`, which is gitignored).
   - **Wire the retriever as a custom Ray-actor environment** ([`NEMO_RL_KNOBS.md`](../training/NEMO_RL_KNOBS.md) §9 Path A): write `training/src/environments/search_r1_env.py`, register it via NeMo-RL's `register_env` from a launch script. The actor parses `<tool_call><function=search><parameter=query>...` from rollouts, POSTs to `{retriever_url}/batch_search`, wraps the response as a `tool` message. Fall back to NeMo Gym (Path B) only if Path A blocks on multi-turn intercepts.
   - **Chat template, reward function, prompt builder** also live under `training/src/` (overlay, not patches to upstream code).
   - **If a true upstream patch is unavoidable** (e.g. NeMo-RL doesn't expose a hook we need), drop a `.patch` file in `training/patches/` and have `setup.sh` apply it after the clone — that way the patch is tracked in our repo and the gitignored clone stays clean.

5. **Verify the training setup matches the paper.** All confirmed paper hyperparameters are in [`PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md) §6, cross-checked against the user's verl scripts in [`VERL_REFERENCE.md`](../training/VERL_REFERENCE.md). Open items to verify against the official [Search-R1 GitHub](https://github.com/PeterGriffinJin/Search-R1) verl yaml during step 4:
   - **Reward function:** Search-R1 baseline reward unchanged (Reward ablation = Milestone 3). The EM scorer must be byte-identical to [`flashrag/search_r1/reward.py`](../../evaluation_search_r1/flashrag/search_r1/reward.py).
   - **Validation during training:** plan in [`VALIDATION.md`](../training/VALIDATION.md). Match Search-R1's validation cadence — verify `test_freq` against upstream verl yaml.

   > **TODO**: confirm `kl_loss_type` (verl uses `low_var_kl`; NeMo-RL's GRPO default may differ — set to match if the option exists, otherwise note the divergence).

6. **Write GRPO training scripts** for 1× A100 80GB and 2× A100 80GB.
   - Hyperparameters baked into [`NEMO_RL_KNOBS.md`](../training/NEMO_RL_KNOBS.md) §7 (concrete starting yaml for 1× A100 with paper's β=0.001, lr=1e-6, group=5, etc.).
   - **Separate concerns** in the script/config: memory-tuning knobs vs. Search-R1-specific training params. Both clearly named and easily tweakable.
   - **Chat template:** Qwen3.5 native `<tool_call>` template is the baseline ([`CHAT_TEMPLATE.md`](../training/CHAT_TEMPLATE.md)); paper template is the ablation arm.
   - **Reference settings** (cherry-picked, transferable bits only) in [`VERL_REFERENCE.md`](../training/VERL_REFERENCE.md). The original verl scripts at `~/Documents/Obsidian/code/omega/research/verl_latest/` stay local.
   - Folder structure inside `training/` should be intuitive (configs / scripts / src clearly separated).

7. **Connect to W&B.** API key in `training/.env` (gitignored). Metric set defined in [`VALIDATION.md`](../training/VALIDATION.md) §4.

8. **Write a paper-vs-ours training comparison.** Already drafted in [`PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md); fill in `TBD` rows (compute, GPU-hours, $/run) after the first run.

## Deliverables

1. Updated docker image pushed to Docker Hub.
2. Search-R1 training dataset downloaded, verified, and converted to NeMo-RL format.
3. NeMo-RL training code set up and verified (without running). W&B connected.
4. Code reviewed and cleaned up for clarity.
5. Docker setup verified; documentation updated for clear reproduction.
6. Repository state suitable for easy reproduction and publication.

## Phase 2 — run training

### Context

Once docker + training code are set up, run training on Vast.ai.

### Goal

Train Qwen3.5-2B (base + hybrid) via GRPO with the Search-R1 baseline reward. Verify training is stable and produces a checkpoint that improves on the base model.

### Success criteria

- **3 seeds** per variant (base, hybrid).
- Note **wall-clock time** and **compute** (GPU-hours, $) per run.
- Compare loss / reward / EM curves against the paper's reported training dynamics.
- **Eval gate**: run the trained checkpoint on **Bamboogle** first (fast, n=125). If results trend up vs. the untrained base model, run the full Milestone 1 benchmark suite (NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle) using the existing `evaluation_search_r1/` pipeline.

### Compute estimate

Calculated post-first-run. **A100 80GB** is the starting plan; alternative GPUs (H100, etc.) will be planned after the first run informs throughput / cost.

### Checkpoint persistence

Saved to **Vast.ai persistent storage** as a starting point. May move to HF Hub or repo LFS later — revisit after the first successful run.

### Step-by-step

1. **Prepare runtime on Vast.ai.**
   - Use [pantomiman/reason-over-search-v1](https://hub.docker.com/r/pantomiman/reason-over-search-v1) (with the new `training` env), or build and push your own image using `docker/reason-over-search-v1/README.md`.
   - Create a Vast.ai custom template from that image and start an instance with **persistent storage** attached (for checkpoints).
   - Storage: at least **150 GB** for indexes, model files, checkpoints.
   - GPU: **1× or 2× A100 80GB**.

2. **Set up the retriever** (Search-R1 rollouts query it during GRPO).
   - Follow [`local_retriever/README.md`](../../local_retriever/README.md).
   - Docker image already has the env; skip env creation.
   - Download corpus, indexes, embedding model.

3. **Validate the retriever service.**
   - Run health/search test calls.
   - Resource note: retriever runs on CPU and needs **~80 GB** RAM.

4. **Run training.** 3 seeds × {base, hybrid}. Monitor W&B for loss / reward / EM curves and verify training stability.

5. **Smoke-eval on Bamboogle.** Run the trained checkpoint through `evaluation_search_r1/` on Bamboogle only.

6. **If Bamboogle improves**, run the full benchmark suite (the seven datasets from Milestone 1). Compare against paper numbers and aggregate across the 3 seeds.
