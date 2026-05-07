---
title: STAGING — Plan A (1 seed) on 8×4090, stage 1 of 2
tags: [eval, plan-a, runbook, 8gpu, staging]
source: internal
created: 2026-05-07
updated: 2026-05-07
---

# Staging — download artifacts and validate fixes on a cheap instance

**Stage 1 of 2.** Stage 2 is [`BOOTSTRAP.md`](BOOTSTRAP.md) (the actual 6-hour 8×4090 sweep).

This stage runs on a **single small instance** with a **persistent volume** mounted at `/workspace`. The volume gets re-attached to the 8×4090 instance in stage 2. Goal: download the 110 GB of artifacts (corpus, IVF-SQ8 index, encoder, three model checkpoints) and validate that the retriever async fix actually parallelizes — both **without paying the 8×4090 hourly rate**.

Cost saved: a typical staging session is ~30–60 min wall-clock dominated by HF downloads (~25–60 min @ 1 Gbps). Running that on an 8×4090 at $1.50–$3/hr in idle GPU time is wasted money. Running the same thing on a 1×4090 at ~$0.35/hr (or CPU-only at $0.10–0.20/hr) is ~$0.10–$0.50.

## Step 0 — host requirements (smaller than stage 2)

Two acceptable shapes:

| Shape | Cost (Vast.ai marketplace) | Validates |
|---|---|---|
| **1× RTX 4090** (recommended) | ~$0.30–0.40/hr | retriever fix + Qwen-on-SGLang single-GPU smoke (Step 7) |
| **CPU-only (no GPU)** | ~$0.10–0.20/hr | retriever fix only — defer SGLang smoke to early stage 2 |

In both cases:

| Resource | Minimum |
|---|---|
| Host RAM | 32 GB (single retriever ~17 GB resident) |
| Disk | **150 GB on persistent volume** at `/workspace` (artifacts) — this is the value being preserved across stages |
| Network | ≥ 100 Mbps (downloads dominate wall-clock) |

**Critical:** the persistent volume must be detachable / re-attachable. On Vast.ai use "Vast.ai Storage" (persistent block storage), not the default ephemeral disk. The 110 GB of artifacts must survive instance teardown.

## Step 1 — get the image

```bash
docker pull pantomiman/reason-over-search-v1:v1
docker run --rm -it --gpus all \
  -p 3005:3005 \
  -v "$HOME/ros":/workspace \
  --entrypoint /bin/bash \
  pantomiman/reason-over-search-v1:v1
```

(`--gpus all` is harmless on a CPU-only host — just use `docker run --rm -it` without it.)

## Step 2 — clone the repo

```bash
cd /workspace
gh repo clone GaurisankarJ/reason_over_search
cd reason_over_search
git checkout plan-a-eval         # branch carrying the fleet support + async fix
```

## Step 3 — verify venvs

```bash
/venv/retriever/bin/python -c "import faiss, fastapi; print('retriever ok')"
/venv/evaluation_search_r1/bin/python -c "import flashrag, sglang; print('eval ok')"
```

## Step 4 — stage all artifacts (~110 GB total, 25–60 min @ 1 Gbps)

```bash
cd /workspace/reason_over_search/local_retriever
mkdir -p corpus indexes models

# Corpus (~14 GB)
huggingface-cli download PeterJinGo/wiki-18-corpus --repo-type dataset \
  --include "wiki-18.jsonl.gz" --local-dir corpus
gunzip -f corpus/wiki-18.jsonl.gz
mv corpus/wiki-18.jsonl corpus/wiki18_100w.jsonl

# IVF-SQ8 FAISS index (~16 GB) — required, no fallback
curl -L -o indexes/wiki18_100w_e5_ivf4096_sq8.index \
  https://huggingface.co/datasets/pantomiman/reason-over-search/resolve/main/retriever/wiki18_100w_e5_ivf4096_sq8.index

# Encoder (~2 GB)
huggingface-cli download intfloat/e5-base-v2 --local-dir models/e5-base-v2

# Search-R1 GRPO checkpoints (2 × 13 GB)
cd ../evaluation_search_r1
huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo \
  --local-dir search_r1_base_model
huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo \
  --local-dir search_r1_instruct_model

# Raw Qwen2.5-3B-Instruct (~7 GB) — pinned to /workspace/hf_cache so it survives
# container restart and is shared with stage 2.
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
huggingface-cli download Qwen/Qwen2.5-3B-Instruct
```

Verify Search-R1 checkpoint identity per [`docs/setup/BOOTSTRAP_NEW_INSTANCE.md:121-129`](../../setup/BOOTSTRAP_NEW_INSTANCE.md#step-4--download-data--indexes--checkpoints):

```bash
sha256sum search_r1_base_model/model-00001-of-00003.safetensors
# expect 7ac54e1b9762c3c6d639da28a2cca177fe7db092ff5cf6e5a9a7849a36a9dabf
sha256sum search_r1_instruct_model/model-00001-of-00003.safetensors
# expect 3d787062256210d1cc6c7c666a0ab0ac83a7a5d0296281b4811df72c968ccd35
```

Verify the Qwen cache location:

```bash
ls -d "$HF_HOME/hub/models--Qwen--Qwen2.5-3B-Instruct"
```

## Step 5 — start a single retriever (smoke, not the full fleet)

Stage 1 only needs ONE retriever to validate the async fix. Save the 8-process fleet for stage 2.

```bash
cd /workspace/reason_over_search
local_retriever/launch_ivfsq8.sh start 3005
local_retriever/launch_ivfsq8.sh wait  3005 300   # ~70 s for IVF-SQ8 cold load
```

## Step 6 — validate the async fix

This is the load-bearing check. If it fails, do **not** pay for the 8×4090 instance until the retriever code is fixed.

```bash
local_retriever/smoke_concurrent.sh 3005 8
# expected: [smoke] PASS — async fix is working (≥3× speedup)
```

The script:
1. hits `/health` on port 3005 — confirms the index loaded and the server is up
2. fires one `/search` and prints the first document — confirms a real query returns sensible content
3. times 8 sequential vs 8 parallel `/search` calls — speedup ≥3× confirms FastAPI's thread pool is dispatching sync handlers in parallel (i.e., the `async def → def` change is active)

If the script reports `FAIL` (parallel ≈ sequential), the async fix isn't active. Diagnose:

```bash
grep -c '^async def search' local_retriever/retriever_serving.py    # expect 0
git log -1 --oneline local_retriever/retriever_serving.py           # confirm the fix is in this commit
ls -la local_retriever/__pycache__/                                  # delete stale .pyc if present
```

Save the output of `smoke_concurrent.sh` into [`SESSION_LOG.md`](SESSION_LOG.md) under "Setup → Stage 1 validation".

Tear down:

```bash
local_retriever/launch_ivfsq8.sh stop 3005
```

## Step 7 — (optional, GPU instance only) single-GPU SGLang smoke

Skip this on CPU-only hosts. On a 1×4090 staging instance, this validates that raw `Qwen2.5-3B-Instruct` actually emits the Search-R1 paper protocol when you ask it to. ~10 min.

```bash
local_retriever/launch_ivfsq8.sh start 3005
local_retriever/launch_ivfsq8.sh wait  3005 300

CUDA_VISIBLE_DEVICES=0 PORT=3000 scripts/manage_sglang.sh start qwen_25_3b_instruct
PORT=3000 scripts/manage_sglang.sh wait 600

curl -sS http://127.0.0.1:3000/get_model_info | grep model_path
# → expect: "Qwen/Qwen2.5-3B-Instruct"

SGL_PORT=3000 RETRIEVER_URL=127.0.0.1:3005 scripts/run_one.sh qwen_25_3b_instruct bamboogle 1
```

Inspect 3 random rollouts in `evaluation_search_r1/results/bamboogle/bamboogle_*_search_r1_qwen_25_3b_instruct_seed1/intermediate_data.json`:

- `<search>...</search>` is emitted at least once
- closes with `<answer>...</answer>`
- `<information>...</information>` is injected back cleanly
- EM > 0 (raw model will be much lower than GRPO — that's the point)

If the model invents its own format, the paper-protocol prompt may need a one-shot example added.

Tear down:

```bash
scripts/manage_sglang.sh stop
local_retriever/launch_ivfsq8.sh stop 3005
```

## Step 8 — snapshot / detach the persistent volume

The artifacts on `/workspace` (and the `$HF_HOME` cache) are what stage 2 will mount. Confirm what's on disk:

```bash
du -sh /workspace/reason_over_search/local_retriever/{corpus,indexes,models}
# corpus: ~14 GB, indexes: ~16 GB, models: ~2 GB

du -sh /workspace/reason_over_search/evaluation_search_r1/search_r1_*_model
# 2 × ~13 GB

du -sh /workspace/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct
# ~7 GB

df -h /workspace
# total used ≈ 65–70 GB on a 150 GB volume
```

Then on Vast.ai:
- **Stop** (don't destroy) the staging instance — the persistent volume survives.
- Note the volume ID; you'll attach the same volume to the 8×4090 instance in stage 2.

If the platform doesn't have detachable volumes, use a tarball-to-blob-storage path:

```bash
tar --exclude='__pycache__' --exclude='*.pyc' \
  -cf /workspace/staging.tar /workspace/reason_over_search/local_retriever/{corpus,indexes,models} \
                              /workspace/reason_over_search/evaluation_search_r1/search_r1_*_model \
                              /workspace/hf_cache
# Upload to S3/R2/B2 and download on the stage-2 host.
```

## Decision criteria — Stage 1 is done when

- [ ] `/workspace/reason_over_search/local_retriever/{corpus,indexes,models}` populated, sizes match (~14 / 16 / 2 GB)
- [ ] Search-R1 base + instruct ckpts sha256-verified
- [ ] `$HF_HOME/hub/models--Qwen--Qwen2.5-3B-Instruct/` populated (~7 GB)
- [ ] `smoke_concurrent.sh 3005 8` returns PASS (≥3× speedup) — **hard gate**
- [ ] (GPU instance only) Step 7 single-GPU SGLang smoke produced non-zero EM with valid `<answer>` tags
- [ ] Persistent volume detached / snapshot taken

The PASS verdict from `smoke_concurrent.sh` is the load-bearing check. Without it, the 8-GPU sweep in stage 2 will be CPU-bound at ~40% GPU util (per `docs/PLAN_A_5090x4.md` §7,8), wasting ~$50 of compute on the 8×4090.

## See also

- [`BOOTSTRAP.md`](BOOTSTRAP.md) — stage 2 (the actual 6-h sweep on 8×4090)
- [`CODE_SETUP.md`](CODE_SETUP.md) — what changed in the scripts
- [`SESSION_LOG.md`](SESSION_LOG.md) — append stage-1 results here
- [`docs/setup/BOOTSTRAP_NEW_INSTANCE.md`](../../setup/BOOTSTRAP_NEW_INSTANCE.md) — single-GPU bootstrap (parent)
- [`docs/setup/VAST_AI_PLAN_A.md`](../../setup/VAST_AI_PLAN_A.md) — Vast.ai cost economics
