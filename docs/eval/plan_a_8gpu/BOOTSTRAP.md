---
title: BOOTSTRAP — Plan A (1 seed) on 8×4090, stage 2 of 2
tags: [eval, plan-a, runbook, 8gpu]
source: internal
created: 2026-05-07
updated: 2026-05-07
---

# Bootstrap — actual 8×4090 sweep (stage 2 of 2)

**Stage 1 prerequisite:** complete [`STAGING.md`](STAGING.md) on a small instance and detach the persistent volume. This doc assumes that volume is mounted at `/workspace` on a fresh 8×4090 instance.

Output of this run: [`RESULTS.md`](RESULTS.md). Predecessor 4-shard run: [`docs/PLAN_A_5090x4.md`](../../PLAN_A_5090x4.md).

## Step 0 — host requirements

| Resource | Minimum | Why |
|---|---|---|
| GPU | 8× RTX 4090 (24 GB each) | one SGLang server per GPU; Qwen2.5-3B bf16 + KV cache @ ctx 8192 ≈ 22 GB/GPU |
| Host RAM | **200 GB** | 8 paired IVF-SQ8 retrievers ~17 GB each (~136 GB) + 8 SGLang host overhead ~16 GB + corpus mmap + OS + headroom |
| Disk | persistent volume from stage 1 mounted at `/workspace` (~70 GB used) | re-uses staged artifacts; no re-download |
| CPU | **64 cores** | 8 retriever procs × `OMP_NUM_THREADS=8` = 64 FAISS threads + 8 SGLang schedulers (~100% each) |

**Architecture:** 8 retriever processes on ports 3005..3012 paired 1:1 with 8 SGLang servers on ports 3000..3007. All 8 SGLang serve the same variant per phase; we cycle through three variants. GPU N → SGLang :300N → retriever :$((3005+N)).

**4090 VRAM headroom is tight.** 22 GB est. on a 24 GB card leaves ~2 GB. Before launching the full sweep, validate one card:

```bash
CUDA_VISIBLE_DEVICES=0 PORT=3000 scripts/manage_sglang.sh start qwen_25_3b_instruct
PORT=3000 scripts/manage_sglang.sh wait 600
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits   # post-load
curl -sS -X POST http://127.0.0.1:3000/generate -H 'Content-Type: application/json' \
  -d '{"text":"Hello.","sampling_params":{"max_new_tokens":256,"temperature":0}}' >/dev/null
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits   # post one-call peak
scripts/manage_sglang.sh stop
```

If post-call MiB ≥ 22500, lower the per-server budget before the fleet boot: edit `scripts/manage_sglang.sh:_launch_one` to add `--mem-fraction-static 0.85`, or drop `--context-length` from 8192 to 6144 (cuts KV cache ~25 %). The 4×5090 predecessor run (`docs/PLAN_A_5090x4.md`) hit no OOMs at this config but had 32 GB cards; treat any `CUDA out of memory` in `/tmp/sglang_*_p3000.log` at fleet boot as a config fix, not a hardware mismatch.

No flat-FAISS fallback; IVF-SQ8 throughout (`docs/setup/VAST_AI_PLAN_A.md:54-56`).

## Step 1 — provision the instance + mount the staged volume

On Vast.ai create an 8× RTX 4090 instance, attaching the persistent volume from stage 1 at `/workspace`:

```bash
docker run --rm -it --gpus all \
  -p 3000-3007:3000-3007 -p 3005-3012:3005-3012 \
  -v /vast_volume_from_stage1:/workspace \
  --entrypoint /bin/bash \
  pantomiman/reason-over-search-v1:v1
```

Port range covers both fleets (SGLang 3000-3007 + retrievers 3005-3012; the 3005-3007 overlap is fine, those are retriever ports).

## Step 2 — verify staged artifacts (no re-download)

Everything below should already be on disk from stage 1. Fail fast if anything is missing — don't pay 8×4090 hourly rate to re-download.

```bash
cd /workspace/reason_over_search
git checkout plan-a-eval && git pull   # in case the branch advanced

# Volume contents check
test -f local_retriever/corpus/wiki18_100w.jsonl                     || echo "MISS: corpus"
test -f local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index     || echo "MISS: index"
test -d local_retriever/models/e5-base-v2                            || echo "MISS: encoder"
test -d evaluation_search_r1/search_r1_base_model                    || echo "MISS: base ckpt"
test -d evaluation_search_r1/search_r1_instruct_model                || echo "MISS: instruct ckpt"
test -d evaluation_search_r1/qwen_25_3b_instruct                     || echo "MISS: qwen2.5-3b-instruct"

# Async fix on this branch
[[ "$(grep -c '^async def search' local_retriever/retriever_serving.py)" == "0" ]] \
  || echo "MISS: async fix not present in this checkout"

# Persistent HF cache pin + datasets cache redirect (flashrag/retriever/utils.py:130
# materializes a ~14 GB parquet cache via datasets.load_dataset; default location
# would land on the ephemeral overlay root and OOM the box).
export HF_HOME="${HF_HOME:-/workspace/hf_cache}" \
       HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/workspace/hf_datasets_cache}" \
       TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf_cache}"

# Hardware
nvidia-smi --query-gpu=count --format=csv,noheader | head -1     # expect 8
free -g | awk '/^Mem/{print "RAM total: " $2 " GB"}'              # expect ≥ 200
nproc                                                             # expect ≥ 64
```

If any line printed `MISS:`, return to [`STAGING.md`](STAGING.md) and complete it before proceeding.

## Step 3 — start the retriever fleet

8 paired processes on ports 3005..3012, each `OMP_NUM_THREADS=8`. Lives across all variant phases.

```bash
local_retriever/launch_ivfsq8.sh start_fleet 8     # background; logs /tmp/retriever_ivfsq8_300{5..12}.log
local_retriever/launch_ivfsq8.sh wait_fleet 600    # ~70 s × 8 (with disk cache warm, ~150 s total)
```

`OMP_RETRIEVER` env var overrides the per-process OMP cap (default 8). For a 96-core host, bump to 12; for 48 cores, drop to 6.

### 3a — Re-validate the fleet (one retriever ≠ eight retrievers)

Stage 1 validated one retriever. Re-run the parallelism check on at least port 3005 (proves async fix survived re-launch on this host) and `/health` on the other 7 (proves the fleet is up):

```bash
local_retriever/smoke_concurrent.sh 3005 8
# PASS (≥3×) → done. OK (2–3×) on a ≥64-core host → retry at N=16:
local_retriever/smoke_concurrent.sh 3005 16
# PASS at N=8 OR N=16 satisfies the gate.

for p in 3006 3007 3008 3009 3010 3011 3012; do
  printf 'port %s: ' "$p"
  curl -sSf "http://127.0.0.1:$p/health" || echo "FAIL"
  echo
done
```

If any port FAILs, do not proceed to Step 5 — investigate `/tmp/retriever_ivfsq8_300X.log`.

## Step 4 — (optional, ~10 min) re-run single-GPU SGLang smoke if not done in stage 1

Skip if you already ran [`STAGING.md`](STAGING.md) Step 7 on a 1-GPU staging instance. If you used a CPU-only staging host, do this now:

```bash
CUDA_VISIBLE_DEVICES=0 PORT=3000 scripts/manage_sglang.sh start qwen_25_3b_instruct
PORT=3000 scripts/manage_sglang.sh wait 600

curl -sS http://127.0.0.1:3000/get_model_info | grep model_path
# → expect: "Qwen/Qwen2.5-3B-Instruct"

SGL_PORT=3000 RETRIEVER_URL=127.0.0.1:3005 scripts/run_one.sh qwen_25_3b_instruct bamboogle 1
```

Inspect 3 rollouts in `evaluation_search_r1/results/bamboogle/bamboogle_*_search_r1_qwen_25_3b_instruct_seed1/intermediate_data.json`:
`<search>` emitted, closed with `<answer>`, `<information>` injected, EM > 0.

Tear down:

```bash
scripts/manage_sglang.sh stop
```

## Step 5 — full 8×4090 sweep

```bash
nohup scripts/sweep_8gpu_one_seed.sh > /tmp/sweep_8gpu.log 2>&1 &
tail -f /tmp/sweep_8gpu.log

# In another terminal — watch GPU util
nvidia-smi dmon -s u -c 720

# Optional: 30 s system metrics sampler (writes docs/eval/plan_a_8gpu/system_metrics.tsv)
nohup scripts/sample_metrics_8gpu.sh > /tmp/sampler.log 2>&1 &
```

Three phases run sequentially: `base` → `instruct` → `qwen_25_3b_instruct`. Within each phase, all 8 GPUs serve the same variant; 7 datasets dispatched in parallel across GPUs 0–6 (GPU 7 idle). NQ (longest) is pinned to GPU 0. Per-phase wall-clock ~2 h + ~5 min model swap. Three phases ≈ 6 h.

Disk: the sweep produces ~1–2 GB of new results under `evaluation_search_r1/results/` (21 result dirs × 5–14 MB `intermediate_data.json` plus ~10 KB metric files). Combined with the staged ~67 GB, expect ~70 GB total used on `/workspace`. `system_metrics.tsv` adds ~200 KB.

The sweep is resume-aware: `run_one.sh` skips any `(variant, dataset, seed=1)` cell that already has a `metric_score.txt`, so a crashed sweep can be re-launched without re-running completed runs. `metric_score.txt` is written atomically at the end of `run_eval.py`'s pipeline (`flashrag/evaluator/evaluator.py:69`), so a partial-rollout crash leaves no metric file → that cell re-runs cleanly.

## Step 6 — aggregate + tear down

`sweep_8gpu_one_seed.sh` aggregates automatically into [`RESULTS.md`](RESULTS.md) on completion. Manual:

```bash
scripts/aggregate.py --output docs/eval/plan_a_8gpu/RESULTS.md
```

Cleanup is automatic via the sweep's `EXIT` trap (stops both fleets). Manual:

```bash
scripts/manage_sglang.sh stop_fleet
local_retriever/launch_ivfsq8.sh stop_fleet
pkill -f sample_metrics_8gpu.sh    # if you started it in Step 5
```

Detach the persistent volume after tear-down so you keep the result artifacts (and can iterate on a future Plan-A run without re-staging).

## Common pitfalls

- **Volume not mounted** — Step 2's `test -f` checks fail. Re-attach the persistent volume from stage 1, or fall back to the tarball path documented at the bottom of [`STAGING.md`](STAGING.md) Step 8.
- **Async fix not active on this checkout** — `grep -c '^async def search'` returns ≠ 0. The branch advanced and the fix was reverted, or you're on the wrong branch. Fix and re-run `smoke_concurrent.sh`.
- **`OMP_NUM_THREADS` not set on retriever fleet** — without `OMP_RETRIEVER` (or `OMP_NUM_THREADS`) capping each process, FAISS spawns ~24 OMP threads per search per process; 8 unbounded processes request ~192 cores on a 64-core box → severe contention. `launch_ivfsq8.sh` sets `OMP_NUM_THREADS=8` by default; verify via `cat /proc/$PID/status | grep Threads` on a fleet pid.
- **Port collision** — stale SGLang on 3000–3007 or stale retriever on 3005–3012 makes `wait_fleet` time out. `scripts/manage_sglang.sh stop` and `local_retriever/launch_ivfsq8.sh stop_fleet` clear them.
- **HF cache lost** — `$HF_HOME` must point to the persistent volume (`/workspace/hf_cache`). If it isn't pinned, the Qwen2.5-3B-Instruct model lands in `~/.cache/huggingface` (ephemeral) and gets downloaded again. `sweep_8gpu_one_seed.sh` exports `HF_HOME` defensively.
- **Datasets cache OOMs root disk** — without `HF_DATASETS_CACHE=/workspace/hf_datasets_cache` exported before the retriever fleet starts, `flashrag/retriever/utils.py:130` materializes ~14 GB of parquet to `~/.cache/huggingface/datasets/` on the ephemeral overlay and the retrievers crash with `[Errno 28] No space left on device` during cold load. Step 2 already exports it; verify with `echo $HF_DATASETS_CACHE` before launching the fleet.
- **`MISS: qwen2.5-3b-instruct` in Step 2** — re-run [`STAGING.md`](STAGING.md) Step 4's `hf download Qwen/Qwen2.5-3B-Instruct --local-dir qwen_25_3b_instruct` from inside `evaluation_search_r1/`. SGLang and the eval pipeline both resolve the variant via this absolute local directory; no HF cache or symlink involved.
- **`pkill -f` foot-gun** — don't kill SGLang or `run_eval.py` by elapsed-time substring; use the pidfiles at `/tmp/sglang_fleet.pids` and `/tmp/retriever_fleet.pids` (`docs/PLAN_A_5090x4.md` §9.3).
- **Resume hazard** — a successful run leaves `metric_score.txt` and `run_one.sh` skips it. To force a re-run: `rm -rf evaluation_search_r1/results/<dataset>/<dataset>_*_search_r1_<variant>_seed1`.
- **GPU 7 idle** — by design (7 datasets, 8 GPUs). Retriever 8 (port 3012) launches but goes unused. To use GPU 7, split the longest dataset (NQ) across two GPUs — out of scope for this 1-seed run.

## Total time budget (stage 2 only)

| Phase | Wall-clock |
|---|---|
| Instance provision + volume attach | ~5–10 min |
| Step 2 artifact verification | <1 min |
| Step 3 retriever fleet bring-up | ~3 min |
| Step 4 (skipped if Step 7 of STAGING was done) | 0 or ~10 min |
| Step 5 full sweep | ~6 h |
| Step 6 aggregate + tear down | <2 min |
| **Total on 8×4090 hourly rate** | **~6–6.5 h** (≈ $20–30 at $3.50–5/hr) |

Compare to a single-bootstrap run that does staging on the 8×4090 itself: 7–8 h × $3.50–5/hr ≈ $25–40, plus the iteration cost of any failed validation. Splitting saves $5–10 per attempt and isolates the validation cost.

## See also

- [`STAGING.md`](STAGING.md) — stage 1 (downloads + retriever fix validation on cheap instance)
- [`CODE_SETUP.md`](CODE_SETUP.md) — what changed in scripts vs prior `plan-a-eval` HEAD
- [`RESULTS.md`](RESULTS.md) — the per-(variant, dataset) results table
- [`SESSION_LOG.md`](SESSION_LOG.md) — running journal of this sweep
- [`docs/PLAN_A_5090x4.md`](../../PLAN_A_5090x4.md) — predecessor 4×5090 multi-instance run; bottleneck analysis
- [`docs/setup/BOOTSTRAP_NEW_INSTANCE.md`](../../setup/BOOTSTRAP_NEW_INSTANCE.md) — single-GPU bootstrap (parent)
- [`docs/setup/VAST_AI_PLAN_A.md`](../../setup/VAST_AI_PLAN_A.md) — fleet sizing economics
