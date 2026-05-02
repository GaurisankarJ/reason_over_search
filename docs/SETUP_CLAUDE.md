# SETUP_CLAUDE.md — fresh-Vast bring-up for Phase 2 smoke / training

Hand this file to Claude (or a human). Following it end-to-end on a fresh Vast.ai instance running `pantomiman/reason-over-search-v1:v1` produces a working setup that can run the four GRPO smoke combos and any longer training run, reproducibly.

The image ships with: `uv`, conda envs (`base`, `retriever`, `evaluation_search_r1`, `main`), pre-warmed uv cache for the `vllm` extra, and the wiki-18 corpus + flat IP + IVF-SQ8 indexes already extracted under `local_retriever/`. The clone of `reason_over_search` brings the tracked code (incl. eval JSONLs via Git LFS) but not the Qwen3.5 weights or the v2/automodel uv-managed venv. Steps below close those gaps.

If anything in this doc disagrees with `training/README.md` / `local_retriever/README.md`, this doc wins for getting from zero to a successful smoke step. The longer-form READMEs have the design rationale; this is the operations checklist.

> **Where the fixes are documented.** The in-tree code edits made to unblock training are listed in [`training/fix/CHANGES.md`](../training/fix/CHANGES.md). This doc references them; it does not re-explain them.

---

## 0. Sanity (≈30 s)

```bash
cd /workspace/reason_over_search
which conda && conda env list                 # base / retriever / evaluation_search_r1 / main
df -h /workspace                              # need ≥ 30 GB free for v2 venv + work
free -g                                        # ≥ 100 GB; flat index needs ~65 GB if you ever switch back to it
git lfs version                                # 3.x
```

Eval `*.jsonl` files under `data/{nq,hotpotqa,...}/` should be full-size (e.g. `hotpotqa/dev.jsonl` ~48 MB). If they show as ~130 B LFS pointers, run `git lfs install && git lfs pull`.

The `data/training/nq_hotpotqa_train/{train,test}.parquet` training shards are produced by `training/scripts/prepare_dataset.py`; on this image they're already present. If not: `training/scripts/prepare_dataset.py`.

## 1. Retriever (≈30–60 s cold start)

The default in `local_retriever/retriever_config.yaml` was flipped to **IVF-SQ8** (v1, May 2026). Just pass `--num_retriever 8` and the right index loads automatically:

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate retriever
cd /workspace/reason_over_search/local_retriever

nohup python retriever_serving.py \
  --config retriever_config.yaml \
  --num_retriever 8 \
  --port 3005 \
  > /tmp/retriever.log 2>&1 &
disown
```

> The training rollouts hammer the retriever (≈80 concurrent `/batch_search` per step on the smoke shape; 2k+ on real config). Flat IP at 2 workers timed out on most calls; IVF-SQ8 at 8 workers does the same batch in ~1.7 s. See [`local_retriever/README_v1.md`](../local_retriever/README_v1.md) for the rationale and when to swap back to flat IP for paper-fidelity eval.

Wait for `Uvicorn running on http://0.0.0.0:3005` in `/tmp/retriever.log`, then:

```bash
curl -sS http://127.0.0.1:3005/health
# {"status":"healthy","retrievers":{"total":8,"available":8}}

curl -sS -X POST http://127.0.0.1:3005/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Who wrote The Lord of the Rings?","top_n":3,"return_score":false}'
# top hit should mention Tolkien
```

If you ever need higher recall (full retrieval-quality eval, not training rollouts), restart with `--index ./indexes/wiki18_100w_e5_flat_inner.index --num_retriever 2` instead. The flat index is ~65 GB resident per worker.

## 2. Model weights (≈4 min on a fast pull, one-time)

The image does not bundle the Qwen3.5 weights. Download both variants to a cache on the persistent `/workspace` volume so they survive instance restarts.

```bash
export HF_HOME=/workspace/hf_cache
mkdir -p $HF_HOME

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate retriever     # has the `hf` CLI

hf download Qwen/Qwen3.5-2B-Base
hf download Qwen/Qwen3.5-2B
```

Each is ~4.3 GB. They land at `$HF_HOME/hub/models--Qwen--Qwen3.5-2B{-Base,}/snapshots/<sha>/model.safetensors-00001-of-00001.safetensors`. The training launcher passes `policy.model_name=Qwen/Qwen3.5-2B-Base` (or `Qwen/Qwen3.5-2B`) by name — vLLM resolves via `HF_HOME`. Make sure `HF_HOME=/workspace/hf_cache` is **exported in the shell that starts training**, otherwise vLLM re-downloads to `/root/.cache/...` (small overlay disk, fills fast).

## 3. NeMo-RL venvs

### 3a. Project venv (≈2 min, fast — wheels from cache)

```bash
cd /workspace/reason_over_search/training/nemo_rl
uv sync --extra vllm
.venv/bin/python -c "import nemo_rl; print(nemo_rl.__version__)"   # → 0.6.0+...
```

### 3b. vLLM Ray-actor venv

This is built lazily on first training launch (≈3 min, wheels also from cache). No pre-step needed — the first `bash training/scripts/run_grpo_1xa100.sh ...` will create `training/nemo_rl/venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/`.

### 3c. v2 / automodel Ray-actor venv (≈25–40 min, **must pre-build**)

The default config uses `policy.dtensor_cfg._v2: true` → `DTensorPolicyWorkerV2` → uv extra `automodel`. The `automodel` extra pulls `nv-grouped-gemm`, whose `setup.py` calls `torch.cuda.init()` at install-time. NeMo-RL's lazy venv build runs inside a Ray actor (`_env_builder`) **with no GPU allocated**, so it dies with `RuntimeError: No CUDA GPUs are available`.

**Workaround: build the venv from the host shell** (CUDA visible), and the lazy builder will skip when it sees an existing `python`:

```bash
cd /workspace/reason_over_search/training/nemo_rl
export NEMO_RL_VENV_DIR=$(pwd)/venvs
export V2_VENV=$NEMO_RL_VENV_DIR/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2
uv venv --allow-existing $V2_VENV
UV_PROJECT_ENVIRONMENT=$V2_VENV uv sync --locked --extra automodel
```

This compiles `transformer-engine` (the long pole — building CUDA kernels for sm_70..sm_120 on all available cores), `nv-grouped-gemm`, and `deep_ep`. The output is a ~11 GB venv at `training/nemo_rl/venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/`. Do not delete it.

> If/when this is baked into the docker image, this step goes away.

## 4. W&B (one-time, optional but recommended)

```bash
cat > /workspace/reason_over_search/training/.env <<'EOF'
WANDB_API_KEY=...your_key_here...
WANDB_ENTITY=...your_entity...
WANDB_PROJECT=reason-over-search
EOF
```

The wrapper script sources this. Set `WANDB_MODE=disabled` in your shell to skip uploads while keeping the local run dirs. The image ships with the user's key already populated; rotate before publishing.

## 5. Smoke run — confirm the loop turns

Run all four combos (base/hybrid × qwen_native/paper). 2 outer steps × 4 prompts × group=5 = 40 trajectories per combo, ≈ 5–8 min each.

```bash
cd /workspace/reason_over_search
export HF_HOME=/workspace/hf_cache
ulimit -n 65536

for variant in base hybrid; do
  for arm in qwen_native paper; do
    rm -rf /tmp/ray
    bash training/scripts/run_grpo_1xa100.sh \
      --variant $variant --seed 42 --arm $arm \
      -- \
      grpo.max_num_steps=2 \
      grpo.num_prompts_per_step=4 \
      policy.train_global_batch_size=20 \
      policy.sequence_packing.enabled=false \
      policy.dynamic_batching.enabled=true \
      policy.train_micro_batch_size=2 \
      > /tmp/smoke_${variant}_${arm}.log 2>&1
  done
done
```

Per-combo trace lands at `logs/exp_NNN/train_data_step{1,2}.jsonl` (`exp_NNN` is auto-incremented; check the printed `📊 Using log directory:`). Each line in the JSONL is one trajectory with: `content` (rendered conversation list), `rewards` (final scalar), `token_loss_mask`, `advantages`, etc.

The `policy.sequence_packing.enabled=false` flag is **not optional** — Qwen3.5's GatedDeltaNet (Mamba) layers crash with `CUDA illegal memory access` under packed sequences. See `training/fix/CHANGES.md` §5.

## 6. Real training (once smoke is green)

Drop the smoke overrides; keep the seq-packing flag off. The full Phase-2 plan (3 seeds × 2 variants):

```bash
for v in base hybrid; do
  for s in 42 7 1337; do
    rm -rf /tmp/ray
    bash training/scripts/run_grpo_1xa100.sh --variant $v --seed $s \
      -- policy.sequence_packing.enabled=false \
         policy.dynamic_batching.enabled=true \
         policy.train_micro_batch_size=2
  done
done
```

Each run = 1005 steps. On 1× A100 80 GB this is several days — see `docs/milestone_two/PHASE_2_RUNBOOK.md` for the exact timing and how to resume.

## Troubleshooting cheatsheet

| Symptom | Cause | Fix |
|---|---|---|
| `unknown arg: --` from wrapper | wrapper without `--` passthrough | `git pull` — the patch is in `run_grpo_1xa100.sh` |
| `Could not override 'data.validation.arm'` | wrapper still has the bad override | same — check the wrapper has the validation line removed |
| `KeyError: 'task_name'` in DataLoader | parquet has no `task_name` col | same — `training/src/datasets/search_r1.py` adds the column |
| `No actor environment registered for ... SearchR1Environment` | registry not patched | same — `training/src/registry.py` patches `ACTOR_ENVIRONMENT_REGISTRY` |
| `RuntimeError: No CUDA GPUs are available` while uv-installing nv-grouped-gemm | v2 venv being built inside GPU-less Ray actor | pre-build v2 venv from host shell (step 3c) |
| `AttributeError: 'Qwen3_5Model' object has no attribute 'layers'` | running v1 (`_v2=false`) DTensor on Qwen3.5 multimodal | use the default `_v2: true`; ensure step 3c is done |
| `Retriever failed: ... Read timed out` in trajectory `<tool_response>` | flat IP retriever overloaded | restart retriever with IVF-SQ8 + 8 workers (step 1) |
| `CUDA error: an illegal memory access` in `torch_chunk_gated_delta_rule` | sequence packing on Qwen3.5 mamba layers | add `policy.sequence_packing.enabled=false` (step 5) |
| Ray refuses to start, `node ID ... not found` | stale ray session from previous run | `rm -rf /tmp/ray` before relaunch |
| `Did not find any active Ray processes` after kill | no-op (your prior `pkill` worked) | proceed |

## What this doc deliberately doesn't cover

- The **architecture / why** behind the overlay — lives in `docs/training/README.md` and per-topic deep dives.
- The **eval pipeline** (`evaluation_search_r1/`, FlashRAG configs) — separate runbook.
- **Multi-node** training — single 1× A100 box only here.
