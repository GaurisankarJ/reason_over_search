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

2. **Download and prep the Search-R1 training dataset, in NeMo-RL's row schema.** Run [`training/scripts/prepare_dataset.py`](../../training/scripts/prepare_dataset.py) — uv inline-script, idempotent. It pulls [`PeterJinGo/nq_hotpotqa_train`](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train) (169,615 train + 51,713 test, parquet) and applies two transforms in one pass:
   - **Strip the prebaked Search-R1 template** — `prompt[0].content := question`.
   - **Rename `prompt` → `messages`** — match the column name NeMo-RL's `ResponseDataset` routes on ([`response_dataset.py:64`](../../training/nemo_rl/nemo_rl/data/datasets/response_datasets/response_dataset.py#L64)). Length-1 list `[{"role": "user", "content": question}]`; `golden_answers` stays as its own column for the M2-step-4 processor to read.

   Output: `data/training/nq_hotpotqa_train/{train,test}.parquet`, committed via Git LFS (matches the `data/**/*.parquet` rule in [`.gitattributes`](../../.gitattributes)). Full schema + the verified NeMo-RL mapping in [`docs/training/TRAINING_DATA.md`](../training/TRAINING_DATA.md).

   Stripping keeps the dataset **template-agnostic** — the chat template (Qwen3.5 native `<tool_call>` default vs. the paper's `<search>` ablation arm) is applied at **rollout time** via run config (`prompt_file` + `system_prompt_file`) by `math_hf_data_processor`-style processing in [`processors.py:467-477`](../../training/nemo_rl/nemo_rl/data/processors.py#L467-L477). Rationale: [`CHAT_TEMPLATE.md`](../training/CHAT_TEMPLATE.md) §5.

3. *(Merged into step 2.)* The schema mapping was originally a separate step gated on inspecting `training/nemo_rl/examples/`. After verifying the routing rule in [`response_dataset.py:64`](../../training/nemo_rl/nemo_rl/data/datasets/response_datasets/response_dataset.py#L64), the reshape was small enough to fold into the prep script. Verified mapping documented in [`TRAINING_DATA.md`](../training/TRAINING_DATA.md) §"Mapping to NeMo-RL's row schema".

4. **Apply Search-R1-style modifications** as an **overlay** under `training/src/` (NOT inside `training/nemo_rl/`, which is gitignored).
   - **Wire the retriever as a custom Ray-actor environment** ([`NEMO_RL_KNOBS.md`](../training/NEMO_RL_KNOBS.md) §9 Path A): write `training/src/environments/search_r1_env.py`, register it via NeMo-RL's `register_env` from a launch script. The actor parses `<tool_call><function=search><parameter=query>...` from rollouts, POSTs to `{retriever_url}/batch_search`, wraps the response as a `tool` message. Fall back to NeMo Gym (Path B) only if Path A blocks on multi-turn intercepts.
   - **Chat template, reward function, prompt builder** also live under `training/src/` (overlay, not patches to upstream code).
   - **If a true upstream patch is unavoidable** (e.g. NeMo-RL doesn't expose a hook we need), drop a `.patch` file in `training/patches/` and have `setup.sh` apply it after the clone — that way the patch is tracked in our repo and the gitignored clone stays clean.

5. **Verify the training setup matches the paper.** All paper hyperparameters cross-checked against [`Search-R1/scripts/nq_hotpotqa/v0.2/train_grpo.sh`](https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/nq_hotpotqa/v0.2/train_grpo.sh) — the EM-only baseline that produced the published GRPO checkpoints we evaluated in Milestone 1. Canonical record: [`PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md) §6; verl-side mapping: [`VERL_REFERENCE.md`](../training/VERL_REFERENCE.md) §2.

   **Resolved during step 5:**
   - **Reward function:** byte-identical port at [`training/src/rewards/search_r1.py`](../../training/src/rewards/search_r1.py); 15 parity tests pass ([`training/tests/test_reward_parity.py`](../../training/tests/test_reward_parity.py)).
   - **`kl_loss_type=low_var_kl` ≡ NeMo-RL `loss.reference_policy_kl_type: k3`** — both compute Schulman 2020 k3 (`exp(ref-log) - (ref-log) - 1`) byte-identically; NeMo-RL's `k3` is the default, no override needed.
   - **`state_masking=true` ≡ automatic** in NeMo-RL via role-based `token_loss_mask` ([`grpo.py:1685-1693`](../../training/nemo_rl/nemo_rl/algorithms/grpo.py#L1685-L1693)) — assistant tokens get loss=1, env (`role: tool`) tokens get loss=0. No config knob.
   - **`test_freq=100`, `total_training_steps=1005`, `val_before_train=true`, `max_turns=4`** — confirmed in v0.2 verl yaml. Maps to NeMo-RL `grpo.{val_period, max_num_steps, val_at_start, max_rollout_turns}`.

   **Carried into step 6 as TODOs:**
   - **`max_obs_length=500`** (verl per-`<information>` block cap on retrieved docs) — not yet enforced in our env; needs truncation in [`training/src/environments/parsers.py`](../../training/src/environments/parsers.py)'s `format_docs_*` helpers.

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
