# Training fixes — May 2026 smoke-run unblock

These are the in-tree changes made to get a fresh `pantomiman/reason-over-search-v1:v1` Vast box from clone → first GRPO step. Each item lists the file, the symptom, and what changed.

## Code changes

### 1. `training/scripts/run_grpo_1xa100.sh`

- **Added `--` passthrough** so callers can append arbitrary Hydra overrides after the wrapper's flags (e.g. `--variant base ... -- grpo.max_num_steps=2 grpo.num_prompts_per_step=4`). Without this, the wrapper exits with `unknown arg: --` for any extra Hydra knobs.
- **Removed `data.validation.arm=${ARM}` from the `OVERRIDES` array.** The default config has `data.validation: null` (validation is gated off in Phase 2). Hydra cannot override a nested key under a `null` parent; the run errored at startup with `ConfigCompositionException: Could not override 'data.validation.arm'`.

### 2. `training/src/registry.py`

- **Patched `ACTOR_ENVIRONMENT_REGISTRY`** to map `training.src.environments.search_r1_env.SearchR1Environment` → `PY_EXECUTABLES.SYSTEM`. NeMo-RL's `create_env` (`environments/utils.py`) looks up `actor_class_fqn` in this dict to decide which uv venv to use for the Ray actor. Without this entry the run dies before training with `ValueError: No actor environment registered for training.src.environments.search_r1_env.SearchR1Environment`.

### 3. `training/src/datasets/search_r1.py`

- **Added a `task_name` column to the loaded HF Dataset.** `AllTaskProcessedDataset.__getitem__` (`processed_dataset.py:118`) reads `entry["task_name"]` whenever `task_data_processors` is a dict — which is always the case on the multi-task path that NeMo-RL takes by default. The reshaped parquet doesn't carry `task_name`, so every dataloader fetch raised `KeyError: 'task_name'`. Mirrors what every upstream `RawDataset` does (e.g. `clevr.py`, `geometry3k.py`, `daily_omni.py`).

## Infrastructure changes (automated in bootstrap.sh)

These are now fully automated by `training/scripts/bootstrap.sh`. The agent runbook is `docs/training/SETUP_CLAUDE.md`.

### 4. v2 (automodel) venv — downloaded from HF Hub, not compiled on Vast

The default config uses `policy.dtensor_cfg._v2: true` → `DTensorPolicyWorkerV2` → uv extra `automodel`. NeMo-RL builds these venvs lazily via Ray actor `_env_builder`, which has **no GPU allocated**. The `automodel` extra pulls `nv-grouped-gemm` whose `setup.py` calls `torch.cuda.init()` at install-time — which fails inside the GPU-less actor with `RuntimeError: No CUDA GPUs are available`, killing training before the first step.

Fix: `bootstrap.sh` (step 3) downloads a pre-built tarball from `pantomiman/reason-over-search-v1-venvs` on HF Hub (~5 GB, ~3 min) and extracts it to the correct venv path. Falls back to host-shell compile (~25 min) if the HF download fails. Either path produces an identical on-disk venv; NeMo-RL's `create_local_venv` detects the existing python and skips any rebuild.

### 5. Sequence packing must be disabled for Qwen3.5

Qwen3.5 is multimodal with interleaved `linear_attention` (Mamba/GatedDeltaNet) and `full_attention` layers. With `policy.sequence_packing.enabled: true`, the `torch_chunk_gated_delta_rule` kernel raises `CUDA error: an illegal memory access was encountered` during `policy.get_logprobs()`. Disabling sequence packing and using dynamic batching instead works.

**These are now the YAML defaults** in both `grpo_qwen3.5_2b_1xa100.yaml` and `grpo_qwen3.5_2b_2xa100.yaml`:
```
policy.sequence_packing.enabled=false
policy.dynamic_batching.enabled=true
policy.train_micro_batch_size=2
```

### 6. Retriever: IVF-SQ8 index + 8 workers is the default for training

With `num_retriever=2` and the flat IP index, the training rollout (~80 concurrent `/batch_search` queries) timed out on ~80% of requests. Rewards collapse to 0/format-only.

`local_retriever/retriever_config.yaml` now defaults to the IVF-SQ8 index. `bootstrap.sh` (step 4) starts the retriever with `--num_retriever 8`. Latency: `30 s timeout` → `~1.7 s` for a 3-query batch. Recall hit < 1% (acceptable for online RL; use flat IP for offline paper-quality eval).
