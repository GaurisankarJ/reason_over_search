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

## Step 0 — ideal instance shape

**Default recommendation: 1× RTX 4090, 32 GB RAM, 16 vCPU, ≥1 Gbps NIC, 150 GB persistent volume.** Cost on Vast.ai marketplace lows: ~$0.30–0.55/hr × ~1 h staging session ≈ **$0.50**.

| Resource | Ideal | Floor | Driven by |
|---|---|---|---|
| GPU | 1× RTX 4090 (24 GB VRAM) | 0 GB (CPU-only) | Qwen2.5-3B in bf16 @ ctx 8192 needs ~22 GB VRAM for the optional Step 7 SGLang smoke. Skip GPU only if you're willing to defer that smoke into stage 2. |
| Host RAM | 32 GB | 24 GB | One IVF-SQ8 index resident ~17 GB + encoder + OS/venv ~5 GB + headroom + (with GPU) SGLang host overhead ~3 GB. |
| vCPU | 16 cores | 16 cores | A single FAISS IVF search at nprobe=64 wants ~8–24 OMP threads. Below 16 cores a single search already saturates the box, so the seq-vs-parallel delta the smoke measures collapses to noise — the smoke can falsely OK a still-broken async path. (The `OMP_RETRIEVER=4` cap that lets 8×4=32 threads coexist is for the **stage-2 fleet**, not the stage-1 single-instance smoke; stage 1 runs one retriever and uses the launcher's default `OMP_RETRIEVER=8`.) |
| Network | ≥1 Gbps | ≥100 Mbps | 110 GB of artifacts. 1 Gbps → ~15 min download phase. 100 Mbps → ~150 min — staging session balloons to 3 h. |
| Disk | 150 GB **persistent volume at `/workspace`** | 130 GB | 110 GB artifacts + 20 GB headroom. Must survive instance teardown — that's the whole point of staging. Ephemeral container disk doesn't count. |

### Cheapest path (CPU-only)

If you just want to validate the retriever async fix and don't want to pay GPU rates:

| Resource | Value |
|---|---|
| GPU | none |
| RAM, CPU, network, disk | same as above |
| Cost | ~$0.10–0.15/hr × ~1 h ≈ **$0.15** |
| Trade-off | Step 7 (Qwen SGLang smoke) deferred to early stage 2 — costs ~10 extra min on the 8×4090 instance ≈ ~$0.50 of expensive time. Net delta vs the 1×4090 staging ≈ 0; mostly a complexity choice. |

### Why these numbers and not less

- **Why not 16 GB RAM** — the IVF-SQ8 index is 17 GB resident on its own. 16 GB host = swap thrashing or OOM at retriever start. This number is index-size-driven, not headroom-driven.
- **Why not 8 vCPUs** — at 8 cores, a single FAISS IVF search at nprobe=64 alone wants ~8–24 cores (its OMP fanout); sequential and parallel timings collapse together, and the speedup ratio loses signal. The smoke can't reliably distinguish PASS from FAIL.
- **Why not 100 Mbps** — pure operator-time concern. The download is unattended but you pay the hourly rate while it runs.
- **Why no GPU** is a valid choice — the load-bearing fix this stage validates is the retriever async fix, which is CPU-only. The SGLang smoke catches a different class of bug (Qwen-model prompt rendering) and isn't on the critical path of "did our changes break the multi-GPU sweep."

### Persistent volume — non-negotiable

On Vast.ai use **"Vast.ai Storage"** (persistent block storage), not the default ephemeral container disk. The 110 GB of artifacts is the value being preserved across stages; if it lives on ephemeral disk it dies with the instance and stage 2 has to re-stage from scratch on the 8×4090 at $5/hr.

Tarball-to-blob fallback (S3 / R2 / B2) is documented at the end of this doc if your platform doesn't support detachable volumes.

## Step 1 — get the image

```bash
docker pull pantomiman/reason-over-search-v1:v1
```

The `-v` mount must point at the **persistent volume** (the artifacts staged below survive into stage 2 only if they live there). On Vast.ai, "Vast.ai Storage" mounts at `/workspace` on the host by default; on other platforms substitute the persistent mountpoint. Don't use a path under the ephemeral container disk.

GPU host (1×4090 staging):

```bash
docker run --rm -it --gpus all \
  -p 3005:3005 \
  -v /workspace:/workspace \
  --entrypoint /bin/bash \
  pantomiman/reason-over-search-v1:v1
```

CPU-only host (no NVIDIA Container Toolkit needed; omit `--gpus all` — it errors with `could not select device driver` on hosts without it):

```bash
docker run --rm -it \
  -p 3005:3005 \
  -v /workspace:/workspace \
  --entrypoint /bin/bash \
  pantomiman/reason-over-search-v1:v1
```

## Step 2 — clone the repo

```bash
cd /workspace
git clone https://github.com/GaurisankarJ/reason_over_search.git
cd reason_over_search
git checkout plan-a-eval         # branch carrying the fleet support + async fix
```

## Step 3 — verify venvs

```bash
/venv/retriever/bin/python -c "import faiss, fastapi; print('retriever ok')"
/venv/evaluation_search_r1/bin/python -c "import flashrag, sglang; print('eval ok')"
```

## Step 4 — stage all artifacts (~110 GB total, 25–60 min @ 1 Gbps)

The `huggingface_hub` CLI is now `hf`; the legacy `huggingface-cli` entry point is deprecated and errors out in current images.

```bash
cd /workspace/reason_over_search/local_retriever
mkdir -p corpus indexes models

# Corpus (~14 GB compressed → ~14 GB uncompressed)
hf download PeterJinGo/wiki-18-corpus --repo-type dataset \
  --include "wiki-18.jsonl.gz" --local-dir corpus
gunzip -f corpus/wiki-18.jsonl.gz
mv corpus/wiki-18.jsonl corpus/wiki18_100w.jsonl

# IVF-SQ8 FAISS index (~16 GB) — required, no fallback
curl -L -o indexes/wiki18_100w_e5_ivf4096_sq8.index \
  https://huggingface.co/datasets/pantomiman/reason-over-search/resolve/main/retriever/wiki18_100w_e5_ivf4096_sq8.index

# Encoder (~2 GB)
hf download intfloat/e5-base-v2 --local-dir models/e5-base-v2

# Search-R1 GRPO checkpoints (2 × 13 GB)
cd ../evaluation_search_r1
hf download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo \
  --local-dir search_r1_base_model
hf download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo \
  --local-dir search_r1_instruct_model

# Raw Qwen2.5-3B-Instruct (~7 GB) — flat local dir, same convention as the
# Search-R1 base/instruct ckpts above. Both SGLang and the eval pipeline
# resolve via this absolute path; no HF cache, no symlink, no model2path entry.
hf download Qwen/Qwen2.5-3B-Instruct --local-dir qwen_25_3b_instruct
test -f qwen_25_3b_instruct/config.json    # sanity
```

Verify Search-R1 checkpoint identity per [`docs/setup/BOOTSTRAP_NEW_INSTANCE.md:121-129`](../../setup/BOOTSTRAP_NEW_INSTANCE.md#step-4--download-data--indexes--checkpoints):

```bash
sha256sum search_r1_base_model/model-00001-of-00003.safetensors
# expect 7ac54e1b9762c3c6d639da28a2cca177fe7db092ff5cf6e5a9a7849a36a9dabf
sha256sum search_r1_instruct_model/model-00001-of-00003.safetensors
# expect 3d787062256210d1cc6c7c666a0ab0ac83a7a5d0296281b4811df72c968ccd35
```

Verify the Qwen local dir (still in `evaluation_search_r1/` from above):

```bash
ls qwen_25_3b_instruct/ | head
# expect: config.json, generation_config.json, model-00001-of-00002.safetensors, tokenizer.json, ...
```

## Step 5 — start a single retriever (smoke, not the full fleet)

Stage 1 only needs ONE retriever to validate the async fix. Save the 8-process fleet for stage 2.

The HF cache redirects below are load-bearing: `flashrag/retriever/utils.py:130` calls `datasets.load_dataset('json', ...)` which materializes a ~14 GB parquet/arrow cache. The default location is `~/.cache/huggingface/datasets/` on the small overlay root disk; without `HF_DATASETS_CACHE` redirected to `/workspace`, the retriever crashes with `[Errno 28] No space left on device` partway through cold load.

```bash
cd /workspace/reason_over_search
export HF_HOME=/workspace/hf_cache \
       HF_DATASETS_CACHE=/workspace/hf_datasets_cache \
       TRANSFORMERS_CACHE=/workspace/hf_cache
local_retriever/launch_ivfsq8.sh start 3005
local_retriever/launch_ivfsq8.sh wait  3005 600   # ~70 s IVF-SQ8 load + arrow cache build on first run
```

## Step 6 — validate the async fix

This is the load-bearing check. If it fails, do **not** pay for the 8×4090 instance until the retriever code is fixed.

```bash
local_retriever/smoke_concurrent.sh 3005 8
# PASS (≥3×) → done. OK (2–3×) → retry at N=16:
local_retriever/smoke_concurrent.sh 3005 16
# PASS at N=8 OR N=16 satisfies the gate.
```

On hosts with ≥64 cores, fixed per-call overhead (encoder pass, JSON, mmap I/O) can cap the N=8 speedup at ~2.5× even with the fix active; the N=16 confirmation amortizes that overhead and gives a clean PASS signal. On a 16-core staging box, N=8 alone should already PASS.

The script:
1. hits `/health` on port 3005 — confirms the index loaded and the server is up
2. fires one `/search` and prints the first document — confirms a real query returns sensible content
3. times N sequential vs N parallel `/search` calls — speedup ≥3× confirms FastAPI's thread pool is dispatching sync handlers in parallel (i.e., the `async def → def` change is active)

If both runs report `FAIL` (parallel ≈ sequential), the async fix isn't active. Diagnose:

```bash
grep -c '^async def search' local_retriever/retriever_serving.py    # expect 0
sed -n '60,70p' local_retriever/retriever_serving.py                 # confirm 'def search' (no async) on /search route
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

The artifacts on `/workspace` are what stage 2 will mount. Confirm what's on disk:

```bash
du -sh /workspace/reason_over_search/local_retriever/{corpus,indexes,models}
# corpus: ~14 GB, indexes: ~16 GB, models: ~2 GB

du -sh /workspace/reason_over_search/evaluation_search_r1/{search_r1_base_model,search_r1_instruct_model,qwen_25_3b_instruct}
# search_r1_base_model: ~13 GB, search_r1_instruct_model: ~13 GB, qwen_25_3b_instruct: ~7 GB

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
                              /workspace/reason_over_search/evaluation_search_r1/{search_r1_base_model,search_r1_instruct_model,qwen_25_3b_instruct}
# Upload to S3/R2/B2 and download on the stage-2 host.
```

## Decision criteria — Stage 1 is done when

- [ ] `/workspace/reason_over_search/local_retriever/{corpus,indexes,models}` populated, sizes match (~14 / 16 / 2 GB)
- [ ] Search-R1 base + instruct ckpts sha256-verified
- [ ] `evaluation_search_r1/qwen_25_3b_instruct/` populated (~7 GB) — `config.json` present
- [ ] `smoke_concurrent.sh` returns PASS (≥3× speedup) at N=8 or, on high-core hosts, at N=16 — **hard gate**
- [ ] (GPU instance only) Step 7 single-GPU SGLang smoke produced non-zero EM with valid `<answer>` tags
- [ ] Persistent volume detached / snapshot taken

The PASS verdict from `smoke_concurrent.sh` is the load-bearing check. Without it, the 8-GPU sweep in stage 2 will be CPU-bound at ~40% GPU util (per `docs/PLAN_A_5090x4.md` §7,8), wasting ~$50 of compute on the 8×4090.

## See also

- [`BOOTSTRAP.md`](BOOTSTRAP.md) — stage 2 (the actual 6-h sweep on 8×4090)
- [`CODE_SETUP.md`](CODE_SETUP.md) — what changed in the scripts
- [`SESSION_LOG.md`](SESSION_LOG.md) — append stage-1 results here
- [`docs/setup/BOOTSTRAP_NEW_INSTANCE.md`](../../setup/BOOTSTRAP_NEW_INSTANCE.md) — single-GPU bootstrap (parent)
- [`docs/setup/VAST_AI_PLAN_A.md`](../../setup/VAST_AI_PLAN_A.md) — Vast.ai cost economics
