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

## Docker / venv changes (one-time on Vast box)

These are **not** committed code changes; they're setup steps captured in `docs/SETUP_CLAUDE.md` so a fresh box is reproducible.

### 4. Pre-build the v2 (automodel) venv from the host shell

The default config uses `policy.dtensor_cfg._v2: true` → `DTensorPolicyWorkerV2` → uv extra `automodel`. NeMo-RL builds these venvs lazily via Ray actor `_env_builder`, which has **no GPU allocated**. The `automodel` extra pulls `nv-grouped-gemm` whose `setup.py` calls `torch.cuda.init()` at install-time — which fails inside the GPU-less actor with `RuntimeError: No CUDA GPUs are available`, killing training before the first step.

Workaround: pre-create the venv from the main shell where CUDA is visible. NeMo-RL's `create_local_venv` checks for a pre-existing `python` and skips the rebuild:

```bash
cd /workspace/reason_over_search/training/nemo_rl
export NEMO_RL_VENV_DIR=$(pwd)/venvs
export V2_VENV=$NEMO_RL_VENV_DIR/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2
uv venv --allow-existing $V2_VENV
UV_PROJECT_ENVIRONMENT=$V2_VENV uv sync --locked --extra automodel
```

This compiles `transformer-engine` (~25 min CUDA-kernel build), `nv-grouped-gemm`, and `deep_ep` from source. Final venv ~11 GB.

This step belongs in the docker image (right next to the existing vLLM extras pre-warm). Until then it's a host-shell prerequisite.

### 5. Sequence packing must be disabled for Qwen3.5

Qwen3.5 is multimodal with interleaved `linear_attention` (Mamba/GatedDeltaNet) and `full_attention` layers. With `policy.sequence_packing.enabled: true`, the `torch_chunk_gated_delta_rule` kernel raises `CUDA error: an illegal memory access was encountered` during `policy.get_logprobs()`. Disabling sequence packing and using dynamic batching instead works.

Pass on the CLI:
```
policy.sequence_packing.enabled=false
policy.dynamic_batching.enabled=true
policy.train_micro_batch_size=2
```

If we end up running Qwen3.5 long-term we should flip the YAML defaults; for now the smoke harness scripts add these overrides explicitly.

### 6. Retriever index choice for training rollouts

Default config (`local_retriever/retriever_config.yaml`) loads the **flat IP** index. With `num_retriever=2` and the training rollout submitting 4 prompts × 5 generations × up-to-4 turns ≈ 80 concurrent `/batch_search` queries against an exact 21 M-passage search, ~80 % of queries time out at the 30 s default. Most rollouts come back with `Retriever failed: ... Read timed out`, which makes rewards collapse to 0 / format-only.

Use the IVF-SQ8 index + more workers for training:
```bash
cd /workspace/reason_over_search/local_retriever
python retriever_serving.py \
  --config retriever_config.yaml \
  --num_retriever 8 \
  --port 3005 \
  --index ./indexes/wiki18_100w_e5_ivf4096_sq8.index
```

Latency dropped from `30 s timeout` to `~1.7 s` for a 3-query batch on this box. Recall hit is < 1 % per the README.
