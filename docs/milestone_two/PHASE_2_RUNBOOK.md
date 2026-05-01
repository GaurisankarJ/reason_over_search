# Milestone 2 Phase 2 — Runbook

Concrete sequence to run Search-R1 GRPO training on Vast.ai. Phase 1 (build the training pipeline) is documented in [MILESTONE_2.md](MILESTONE_2.md) §"Step-by-step"; this runbook covers Phase 2 (actually run training + smoke-eval).

> **TL;DR** — boot the docker image on Vast.ai, set up retriever, source `training/.env`, run `bash training/scripts/run_grpo_1xa100.sh --variant base --seed 42`. Repeat for hybrid + 3 seeds. Eval on Bamboogle through the Milestone-1 pipeline.

---

## Pre-flight checklist

Run through this before booting the Vast instance. Prevents wasted GPU-hours.

### Local-side

- [ ] Latest code on `main` branch (or whatever branch you intend to run): `git status` clean, `git push` done.
- [ ] Datasets on Git LFS: `git lfs ls-files` shows `data/training/nq_hotpotqa_train/{train,test}.parquet`.
- [ ] W&B account ready; API key copied to clipboard.
- [ ] Docker Hub credentials ready (only needed if pushing a new image).

### Verify the overlay still passes its tests

```bash
uv run --no-project --with pytest pytest training/tests/ -v
# Expected: 19 passed, 2 skipped (env/dataset modules need the training venv).
```

### Verify the docker image is current

The latest image on Docker Hub should match the committed `training/nemo_rl/` ref. If you bumped NeMo-RL recently, rebuild:

```bash
docker build \
  --build-arg NEMO_RL_REF=v0.6.0 \
  --build-arg UV_EXTRAS=vllm \
  -f docker/reason-over-search-v1/Dockerfile -t pantomiman/reason-over-search-v1:v1 .
docker push pantomiman/reason-over-search-v1:v1
```

---

## Vast.ai setup (per fresh instance)

### 1. Boot

- **Image**: `pantomiman/reason-over-search-v1:v1` (or your fork).
- **GPU**: 1× or 2× A100 80GB. Match the launch script you intend to use.
- **Persistent storage**: ≥ **150 GB** for indexes (~25 GB) + model weights (~15 GB) + checkpoints (~10–30 GB per run).
- **Disk**: container disk ≥ 60 GB (for the venv at `training/nemo_rl/.venv/` + uv cache + working set).
- **CPU/RAM**: ≥ 16 cores / **≥ 100 GB RAM** (retriever flat FAISS holds ~65 GB in host RAM; leave headroom).

### 2. Clone repo + materialize the training venv

```bash
cd /workspace
git clone https://github.com/<your-user>/reason_over_search.git
cd reason_over_search

# Pull LFS objects (training parquets + eval jsonls)
git lfs install
git lfs pull

# Materialize the NeMo-RL venv. First run downloads ~5 GB of wheels — torch,
# vLLM, cuDNN — taking ~10–20 min on Vast's network. The image's uv install
# does NOT carry a pre-warmed wheel cache (see docker/reason-over-search-v1/
# README.md "Build args" for why); subsequent runs in the same container are
# fast since uv keeps the cache at /root/.cache/uv/.
cd training/nemo_rl
uv sync --extra vllm
cd ../..
```

Smoke-check the venv:
```bash
training/nemo_rl/.venv/bin/python -c "import nemo_rl; print(nemo_rl.__version__)"
# → 0.6.0
```

### 3. Set up the retriever

Follow [`local_retriever/README.md`](../../local_retriever/README.md). Summary:

```bash
# Use the docker image's pre-built /venv/retriever (faiss-cpu).
# Download corpus + index + encoder model (~25 GB total):
cd local_retriever
mkdir -p corpus indexes models

huggingface-cli download PeterJinGo/wiki-18-corpus \
  --repo-type dataset --include "wiki-18.jsonl.gz" \
  --local-dir corpus --local-dir-use-symlinks False
gunzip -f corpus/wiki-18.jsonl.gz
mv corpus/wiki-18.jsonl corpus/wiki18_100w.jsonl

huggingface-cli download PeterJinGo/wiki-18-e5-index \
  --repo-type dataset --local-dir indexes --local-dir-use-symlinks False
cat indexes/part_aa indexes/part_ab > indexes/wiki18_100w_e5_flat_inner.index.gz
gunzip -f indexes/wiki18_100w_e5_flat_inner.index.gz

huggingface-cli download intfloat/e5-base-v2 \
  --local-dir models/e5-base-v2 --local-dir-use-symlinks False
cd ..
```

**Move heavy artifacts to persistent storage** if `/workspace` isn't already mounted persistent — checkpoint persistence assumes the run survives instance restarts.

### 4. Validate retriever

Start it in a `tmux` / `screen` session so it survives:

```bash
tmux new -s retriever
/venv/retriever/bin/python local_retriever/retriever_serving.py \
    --config local_retriever/retriever_config.yaml \
    --num_retriever 1 \
    --port 3005
# Ctrl-b d to detach
```

Health + sanity-search:
```bash
curl -sS http://127.0.0.1:3005/health
# → "healthy"

curl -sS -X POST http://127.0.0.1:3005/batch_search \
    -H "Content-Type: application/json" \
    -d '{"query":["who founded SpaceX"],"top_n":3,"return_score":false}' \
  | head -c 400
# Expect a list[list[{id, contents}]] with 3 docs.
```

If retriever startup fails: check RAM (flat FAISS index needs ~65 GB resident), check the corpus + index files exist at the paths in `retriever_config.yaml`.

### 5. Download model weights

```bash
mkdir -p training/models
huggingface-cli download Qwen/Qwen3.5-2B-Base \
  --local-dir training/models/Qwen3.5-2B-Base \
  --local-dir-use-symlinks False

# Hybrid (only if you'll run the hybrid variant):
huggingface-cli download Qwen/Qwen3.5-2B \
  --local-dir training/models/Qwen3.5-2B \
  --local-dir-use-symlinks False
```

The launch scripts pass model names as HF repo IDs (`Qwen/Qwen3.5-2B-Base`), not local paths — vLLM resolves via the HF cache. If you want to use a local copy, override `policy.model_name=/workspace/.../Qwen3.5-2B-Base` on the CLI.

### 6. Configure W&B

```bash
cp training/.env.example training/.env
$EDITOR training/.env
# Fill in WANDB_API_KEY (and optionally WANDB_ENTITY, CHECKPOINT_DIR_BASE).
```

The launch scripts source this file. `training/.env` is gitignored.

For checkpoint persistence, set:
```bash
CHECKPOINT_DIR_BASE=/workspace/persistent/checkpoints
```

---

## Smoke run (recommended before the real thing)

Run a tiny truncated training to verify all the pipes connect — retriever, dataset, env, GRPO loop, W&B logging, checkpointing.

```bash
# 5-step training run, single seed, 1× A100. Override max_num_steps + val cadence.
bash training/scripts/run_grpo_1xa100.sh --variant base --seed 999 \
    grpo.max_num_steps=5 grpo.val_period=2 grpo.val_at_start=true \
    logger.wandb.name=smoke-test
```

Expected outcome:
- W&B picks up the run; you see `train/reward_mean`, `val/accuracy`, etc.
- The retriever logs receive `/batch_search` calls during rollouts.
- Checkpoint saves to `${CHECKPOINT_DIR_BASE}/qwen3.5-2b-base/qwen_native/seed999/`.
- No NaN losses or KL spikes.

If anything blows up here, fix it before committing real GPU-hours.

---

## Real runs — 6-run plan

### 1× A100 (default; cheapest)

```bash
# Three seeds for base, qwen_native arm
bash training/scripts/run_grpo_1xa100.sh --variant base --seed 42
bash training/scripts/run_grpo_1xa100.sh --variant base --seed 7
bash training/scripts/run_grpo_1xa100.sh --variant base --seed 1337

# Three seeds for hybrid, qwen_native arm
bash training/scripts/run_grpo_1xa100.sh --variant hybrid --seed 42
bash training/scripts/run_grpo_1xa100.sh --variant hybrid --seed 7
bash training/scripts/run_grpo_1xa100.sh --variant hybrid --seed 1337
```

Each run:
- 1005 steps, ~30–50 h projected on 1× A100 ([PAPER_VS_OURS_TRAINING.md §7](../training/PAPER_VS_OURS_TRAINING.md#7-compute)).
- 11 validation points (step 0 + every 100 steps).
- Top-3 checkpoints kept by `val:accuracy`, plus `latest`.

### 2× A100 (faster; if 1× takes too long)

```bash
bash training/scripts/run_grpo_2xa100.sh --variant base --seed 42
# ...etc.
```

Wall-clock ~1.7× faster, total GPU-hours slightly higher (see [PAPER_VS_OURS_TRAINING.md §7](../training/PAPER_VS_OURS_TRAINING.md#7-compute)).

### Sequential vs concurrent

On a single Vast instance, runs are **sequential** (one process at a time per GPU). To run multiple seeds concurrently, spin up separate Vast instances or split GPUs. Concurrent runs share retriever traffic — the retriever handles batched requests but watch CPU saturation.

---

## Monitoring during training

### W&B

Per [VALIDATION.md §4](../training/VALIDATION.md#4-metrics-logged-to-wb), watch:

- `train/reward_mean` — should rise from ~0 to ~0.3 over the first 100 steps for `paper` arm; from ~0 to ~0.5 for `qwen_native` (different reward scale).
- `train/kl` — spikes followed by recovery are normal early. Sustained KL > 0.5 is a warning.
- `val/accuracy` (= mean reward on val) — should improve at each validation point.
- `val/em_hit_rate` — fraction of val rollouts with reward ≥ 0.8 (proxy for "got the answer").
- `val/fraction_of_samples_properly_ended` — fraction that emitted a proper stop token (not max-tokens-truncated). Should be ≥ 0.95 by step 200; lower means model is not learning the format.
- Truncation rate via `policy.generation` truncation events.

### Retriever

```bash
tmux attach -t retriever
# Watch /batch_search call rate during rollouts.
```

CPU saturation on the retriever node is the most likely throughput bottleneck. If it pegs 100 %, the GPU sits idle waiting.

### Failure modes + recovery

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM in policy forward | `train_micro_batch_size` too high | Drop to 2; or enable `dtensor_cfg.cpu_offload=true` |
| OOM in vLLM rollout | `vllm_cfg.gpu_memory_utilization` too high | Drop to 0.7 |
| Retriever HTTP timeout | retriever overloaded | Increase `env.search_r1.request_timeout_s` to 60; reduce `grpo.num_generations_per_prompt` to 4 temporarily |
| `val:accuracy` stays at 0 | format-validity ≈ 0 (model not emitting `<answer>`) | Inspect a few val rollouts in W&B. Likely needs more warmup or a higher LR. |
| KL explodes (> 5) | LR too high or KL coef too low | Check `loss_fn.reference_policy_kl_penalty=0.001` is set |

---

## Smoke-eval on Bamboogle

After each run completes, smoke-test the best checkpoint through the Milestone-1 eval pipeline.

### 1. Convert checkpoint to a vLLM-loadable format

NeMo-RL's checkpointing saves both DTensor sharded state and (with `save_consolidated: true`) a HF-compatible safetensors. Our config sets `save_consolidated: false` to save space; flip it for runs you want to eval, OR run NeMo-RL's `convert_dcp_to_hf.py` post-hoc.

```bash
training/nemo_rl/.venv/bin/python training/nemo_rl/examples/converters/convert_dcp_to_hf.py \
    --dcp-ckpt-path ${CHECKPOINT_DIR_BASE}/qwen3.5-2b-base/qwen_native/seed42/checkpoints/best/policy/ \
    --hf-output-path /workspace/persistent/checkpoints/qwen3.5-2b-base-grpo-seed42-hf/ \
    --tokenizer-name Qwen/Qwen3.5-2B-Base
```

### 2. Run M1 eval pipeline on Bamboogle

```bash
# Following docs/milestone_one/FROZEN_CONFIG_v1.md, but pointing at our trained checkpoint.
scripts/manage_sglang.sh switch trained -- \
    --model-path /workspace/persistent/checkpoints/qwen3.5-2b-base-grpo-seed42-hf
scripts/run_one.sh trained bamboogle 1 > bamboogle_seed42.log 2>&1

LATEST=$(ls -dt evaluation_search_r1/results/bamboogle/bamboogle_*_search_r1_trained_seed1 | head -1)
grep -E "^(em|f1):" "$LATEST/metric_score.txt"
```

> **Eval gate**: if Bamboogle EM trends up vs. the **untrained** Qwen3.5-2B-Base baseline (~5 min eval, n=125), proceed to the full Milestone-1 benchmark suite. If not, debug before committing 60 GPU-hours of full eval.

### 3. Full benchmark suite (only if Bamboogle improves)

```bash
# 7 datasets × variant × seed. See evaluation_search_r1/README.md and
# docs/setup/VAST_AI_PLAN_A.md for the cost-optimized fleet plan.
for ds in nq triviaqa popqa hotpotqa 2wikimultihopqa musique bamboogle; do
    scripts/run_one.sh trained $ds 1 > eval_${ds}_seed42.log 2>&1
done
```

---

## After Phase 2

1. **Fill the TBD rows** in [PAPER_VS_OURS_TRAINING.md §7](../training/PAPER_VS_OURS_TRAINING.md#7-compute): observed wall-clock, GPU-hours, $/run.
2. **Aggregate eval results** across the 3 seeds × 2 variants. Mirror the M1 results table at [docs/milestone_one/RESULTS_PLAN_B.md](../milestone_one/RESULTS_PLAN_B.md).
3. **Side-by-side write-up** vs. paper Table 3 numbers (§3 of [PAPER_VS_OURS_TRAINING.md](../training/PAPER_VS_OURS_TRAINING.md)).
4. **Decide on M3** based on what M2 reveals — likely reward-function ablations and a Qwen-native-aware format reward.

---

## Quick reference

| Need | Path |
|---|---|
| Launch script (1× A100) | [`training/scripts/run_grpo_1xa100.sh`](../../training/scripts/run_grpo_1xa100.sh) |
| Launch script (2× A100) | [`training/scripts/run_grpo_2xa100.sh`](../../training/scripts/run_grpo_2xa100.sh) |
| Hyperparameter audit | [`docs/training/PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md) |
| W&B metric set | [`docs/training/VALIDATION.md`](../training/VALIDATION.md#4-metrics-logged-to-wb) |
| Tuning knobs | [`docs/training/NEMO_RL_KNOBS.md`](../training/NEMO_RL_KNOBS.md) |
| Overlay architecture | [`docs/training/README.md`](../training/README.md#trainingsrc-overlay-architecture) |
| Retriever setup | [`local_retriever/README.md`](../../local_retriever/README.md) |
| M1 eval pipeline | [`docs/milestone_one/FROZEN_CONFIG_v1.md`](../milestone_one/FROZEN_CONFIG_v1.md) |
