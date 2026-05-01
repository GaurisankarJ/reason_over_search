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

1. **Set up NeMo-RL.** Run [`training/setup.sh`](../../training/setup.sh) — idempotent script that:
   - Installs [`uv`](https://docs.astral.sh/uv/) (NeMo-RL's official package manager).
   - Clones [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) at the pinned tag (default `v0.6.0`) into `training/nemo_rl/` with submodules (Megatron-LM, Megatron-Bridge, Automodel, Gym).
   - Removes the cloned `.git` so it's not a nested repo (per Milestone 2 design — local edits get tracked in *this* repo, not upstream's).
   - Creates a uv venv at `training/nemo_rl/.venv/` (Python 3.13 — uv downloads it itself).
   - Runs `uv sync --extra vllm` to install all of NeMo-RL's pinned deps + the editable package.

   **The clone is gitignored** (NeMo-RL + 4 submodules ≈ several hundred MB; reproducibility comes from `NEMO_RL_REF` + `uv.lock`). See [`training/README.md`](../../training/README.md) for setup-script knobs.

   **Docker integration**: the `pantomiman/reason-over-search-v1` image now ships `uv` pre-installed and `training/` copied to `/app/training/`. The heavy install is deferred to runtime (avoids freezing a NeMo-RL version into the image). On a fresh container: `bash /app/training/setup.sh` once. The Dockerfile lives at [`docker/reason-over-search-v1/Dockerfile`](../../docker/reason-over-search-v1/Dockerfile); push the rebuilt image after this step.

2. **Download and verify the Search-R1 training dataset.** Schema, splits, and a quick-load snippet are in [`docs/training/TRAINING_DATA.md`](../training/TRAINING_DATA.md). Source: [`PeterJinGo/nq_hotpotqa_train`](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train) (170k train + 51.7k test, parquet).

3. **Convert the dataset to NeMo-RL's expected format.** Done *after* (1) and (2), once NeMo-RL's input requirements are visible inside `training/nemo_rl/examples/`. The conversion **must rewrite the dataset's `prompt[0].content`** — the upstream parquet ships with paper's `<search>` template baked in, which clashes with our Qwen3.5 native `<tool_call>` template (rationale in [`CHAT_TEMPLATE.md`](../training/CHAT_TEMPLATE.md) §5; mapping recipe in [`TRAINING_DATA.md`](../training/TRAINING_DATA.md) §"Conversion to NeMo-RL format").

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
