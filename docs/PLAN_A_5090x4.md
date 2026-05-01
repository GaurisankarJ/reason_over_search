# Plan A on 4× RTX 5090 — Vast.ai run, 2026-05-01

Single-session experiment: run Plan A's base half (3 seeds × 7 datasets = 21 runs) end-to-end on a single Vast.ai instance with 4× RTX 5090 by sharding 4 SGLang servers across the GPUs and a shared retriever.

> **Status:** stopped early at 16:06 UTC (user request) before the extended deadline. 4 runs completed; 4 hotpotqa runs + triviaqa-1 were in flight and aborted (resume-aware — will be retried in a future session). 3 popqa runs were lost earlier to an orphan-kill incident.

## 1. Why and what changed vs. canonical Plan A

[`docs/VAST_AI_PLAN_A.md`](VAST_AI_PLAN_A.md) sized Plan A for an 8× RTX 4090 fleet running shards of `(variant, seed)` against one SGLang per machine. This run validates the same idea **on one machine** with 4× 5090 by running 4 SGLang servers + 4 sweep workers + 1 shared retriever.

Scope: **base variant only, 3 seeds**, IVF-SQ8 retriever. Goal is a working multi-shard recipe and per-run wall-clock numbers.

## 2. Hardware

| | |
|---|---|
| GPUs | 4× NVIDIA RTX 5090 (32 GB each) |
| CPU | AMD EPYC 7763 — 1 socket, 64 cores, 128 threads |
| RAM | 251 GB |
| Disk | local NVMe; wiki18 corpus + IVF-SQ8 index pre-staged |

Note: the repo's [`docs/HARDWARE.md`](HARDWARE.md) describes a single-4090 box. This run was on a different machine; do not edit that file.

## 3. Setup (live config)

- **Retriever** (PID 95361):
  - `python retriever_serving.py --num_retriever 8 --port 3005 --index ./indexes/wiki18_100w_e5_ivf4096_sq8.index`
  - FAISS-CPU, IVF4096-SQ8, `nprobe=64`, encoder `intfloat/e5-base-v2`
  - 8 retriever instances (each holds its own copy of the index ≈ 17.5 GB RES)
- **SGLang servers** — 4 instances pinned via `CUDA_VISIBLE_DEVICES`:
  | shard | GPU | port | served-model-name |
  |---|---|---|---|
  | 0 | 0 | 3000 | `search_r1_base_g0` |
  | 1 | 1 | 3001 | `search_r1_base_g1` |
  | 2 | 2 | 3002 | `search_r1_base_g2` |
  | 3 | 3 | 3003 | `search_r1_base_g3` |
  - All served from `evaluation_search_r1/search_r1_base_model`, `--tp 1 --context-length 8192 --dtype bfloat16 --enable-metrics`.
- **Sweep workers** — 4 parallel `bash` loops, each pinned to its SGLang via `SGL_PORT` env var.

### One-line patch to `scripts/run_one.sh`

To allow each shard worker to point at its own SGLang port without forking the script, added support for the `SGL_PORT` env var (default `3000`):

```bash
SGL_PORT="${SGL_PORT:-3000}"
...
  --sgl_remote_url 127.0.0.1:$SGL_PORT \
```

Preserves existing behavior (port 3000) for any callers that don't set the env var.

## 4. LPT shard plan

21 jobs (5 datasets × 3 seeds + bamboogle skipping seed 1 already-done) bin-packed by item count into 4 shards using LPT (longest-processing-time-first). Final imbalance: `max/min = 1.024`.

| shard | port | jobs (in execution order) | total items |
|---|---|---|---:|
| 0 | 3000 | popqa-3, 2wmh-1, hotpotqa-1, nq-1, bamboogle-3, bamboogle-2 | 38108 |
| 1 | 3001 | popqa-2, triviaqa-3, hotpotqa-3, nq-3, musique-2 | 39012 |
| 2 | 3002 | popqa-1, triviaqa-2, hotpotqa-2, nq-2, musique-1 | 39012 |
| 3 | 3003 | 2wmh-3, 2wmh-2, triviaqa-1, musique-3 | 38882 |

Item-count for sizing:

| dataset | split | items |
|---|---|---:|
| popqa | test | 14267 |
| 2wikimultihopqa | dev | 12576 |
| triviaqa | test | 11313 |
| hotpotqa | dev | 7405 |
| nq | test | 3610 |
| musique | dev | 2417 |
| bamboogle | test | 125 |

## 5. Run timeline

- **14:00–14:08 UTC** — Retriever (4w) + SGLang (instruct) running from a prior session. Sanity Bamboogle base seed=1 run completed earlier (EM 0.096, n=125).
- **14:09:08** — Switched to 4-shard SGLang base (4 GPUs, ports 3000-3003). All ready in ~30 s.
- **14:09:44** — First sweep launched (4 shards × 4-worker retriever). Quick benchmark: GPU avg ~55 % per GPU, 100 ESTAB conns to retriever. Inferred queue depth ~60 from the per-shard 16-thread `INFERENCE_MAX_WORKERS`.
- **14:39 (≈)** — User asked to bump retriever workers. **Incident:** my collateral kill of `run_eval.py` orphans accidentally killed the in-flight runs too (etime-string filter `> "12:00"` matched both old and new). The 4 sweep wrappers respawned new run_eval.py procs, but the 3 popqa runs in flight were lost — empty result dirs, no `metric_score.txt`, wrappers logged END (no exit-code check) and moved on.
- **14:44** — Stopped 4-worker retriever. Started 8-worker retriever. Ready 14:45:31. Free RAM 86 GB after; retriever now 131 GB RSS.
- **14:46:29** — Sweep 2 launched (sweep_start_epoch). Same 4 wrappers, same shard plan, fresh logs. **Original deadline 15:46:29.**
- **14:59:12** — Sweep wrappers all moved to their **second** job (2wmh-1, triviaqa-3, triviaqa-2, 2wmh-2) — first jobs were the lost popqa runs.
- **15:25 UTC** — Status check showed 4 in-flight runs at ~26 min each, ~30 % done at 8.7 SGLang req/s per port ≈ 2.2 items/s per shard. **Decided to extend deadline to 17:01:29 UTC.** Old deadline-killer was already gone (killed earlier in cleanup); new one + sampler armed at 15:34.
- **15:41:33** — triviaqa seed=2 + seed=3 completed (~42:21 each).
- **15:46:05–07** — 2wmh seed=1 + seed=2 completed (~46:53 each).
- **15:46–15:47** — hotpotqa-1, hotpotqa-2, hotpotqa-3, triviaqa-1 started.
- **16:06:39** — User stopped the sweep early. Sweep wrappers killed (PIDs 97070/97083/97096/97108) followed by their `run_eval.py` children. 4 in-flight runs aborted mid-flight; their result dirs (`*_2026_05_01_15_4{1,6}_search_r1_base_seed*`) have no `metric_score.txt`, so `run_one.sh` will retry them on the next session.

## 6. Results

### Completed runs _(updated as they land)_

| dataset | seed | n | EM | F1 | acc | wall clock | paper EM | Δ pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 1 | 12576 | 0.2562 | 0.3121 | 0.2701 | 46:53 | 0.274 | −1.8 |
| 2wikimultihopqa | 2 | 12576 | 0.2565 | 0.3135 | 0.2711 | 46:55 | 0.274 | −1.7 |
| triviaqa | 2 | 11313 | 0.5743 | 0.6422 | 0.6163 | 42:21 | 0.583 | −0.9 |
| triviaqa | 3 | 11313 | 0.5740 | 0.6428 | 0.6170 | 42:21 | 0.583 | −0.9 |

(Bamboogle base seed=1 from the earlier sanity run: EM 0.096 — known n=125 variance per [`docs/archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md`](archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md).)

All 4 completed runs are within **~2 pp of the paper baselines** — consistent with the v1 reproducer's noise band. No regression from running 4 shards in parallel.

### Lost runs (reproducible cleanly later)

popqa seed=1, 2, 3 — aborted at 14:59:12 by the orphan-kill incident. Empty result dirs at `evaluation_search_r1/results/popqa/popqa_2026_05_01_14_46_search_r1_base_seed{1,2,3}` have no `metric_score.txt`, so `run_one.sh` will not skip them on the next attempt; they will be retried in a future sweep.

### Aborted in-flight at 16:06:39 UTC (will be retried)

| shard | port | dataset | seed | n | started | aborted at | partial |
|---|---|---|---:|---:|---|---|---:|
| 0 | 3000 | hotpotqa | 1 | 7405 | 15:46:05 | 16:06:39 | ~20 min |
| 1 | 3001 | hotpotqa | 3 | 7405 | 15:41:33 | 16:06:39 | ~25 min |
| 2 | 3002 | hotpotqa | 2 | 7405 | 15:41:33 | 16:06:39 | ~25 min |
| 3 | 3003 | triviaqa | 1 | 11313 | 15:46:07 | 16:06:39 | ~20 min |

### Not started in this session

nq-1/2/3, musique-1/2/3, bamboogle-2/3, plus the 3 lost popqa seeds. Total runs that will be picked up by the next sweep session: **6 lost-or-aborted + 8 not-started = 14 of 21**.

## 7. Bottleneck analysis — the retriever async serialization

Initial hypothesis: more retriever workers → more concurrency. Empirically that's wrong.

| `--num_retriever` | retriever CPU | GPU avg util | per-port SGLang QPS |
|---:|---:|---:|---:|
| 4 | ~24 cores (2455 %) | ~55 % | n/a (popqa lost) |
| 8 | ~24 cores (2391 %) | ~41 % combined (43.7 / 36 / 38 / 44) | 7.9–9.6 req/s |

**Doubling workers did not double throughput.** Per-process retriever CPU stayed flat (~24 cores) regardless of `--num_retriever`. This is a code-level serialization, not a resource limit. From [`local_retriever/retriever_serving.py:70-101`](../local_retriever/retriever_serving.py):

```python
async def search(request: QueryRequest):
    ...
    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            results = retriever_list[retriever_idx].search(query, ...)  # SYNC FAISS call
```

`retriever.search()` is **synchronous** inside an `async def` handler. uvicorn here runs single-worker (no `--workers` flag), so the synchronous FAISS call **blocks the event loop for its entire duration**. The asyncio semaphore is decorative — even with `num_retriever=8`, the event loop only executes one FAISS search at a time. The other retrievers sit idle.

Consistent observations:
- Retriever process CPU = ~24 cores ≈ FAISS internal OMP fanout for one nprobe=64 IVF search; doesn't scale with `--num_retriever`.
- `ESTAB` sockets averaged 94 (peak 237) — many shards' threads queued at the single-threaded event loop.
- GPUs alternate 0 → 99 % each second — each shard waits its turn through the (serialized) retriever before continuing inference.

**Cheap fixes (not applied this run):**
1. Change `async def search` → `def search` (and the same for `batch_search`). FastAPI runs sync handlers in a 40-thread pool automatically. Remove or replace the asyncio semaphore. ~10-line change.
2. Or `await asyncio.to_thread(retriever.search, ...)`.
3. Or `uvicorn.run(..., workers=N)` (multi-process; doubles RAM since each process loads the index).

Recommended next experiment: option 1, on a separate branch, with the same 4-shard setup. The single-worker → ~3-4× speedup hypothesis is testable in <1 hour.

## 8. System metrics (sampled every 30 s)

134 samples from 14:58 to 16:06 UTC (~67 min). Window covers both 4-worker and 8-worker retriever phases.

| metric | avg | peak |
|---|---:|---:|
| GPU 0 util | 44 % | 99 % |
| GPU 1 util | 37 % | 100 % |
| GPU 2 util | 39 % | 100 % |
| GPU 3 util | 44 % | 100 % |
| GPU combined avg | 41 % | — |
| retriever CPU | 2407 % (= 24.1 cores) | — |
| retriever RSS | 140.5 GB | — |
| ESTAB conns to :3005 | 93 | — |
| free RAM | 3 GB | — |
| buff/cache | 85 GB | — |
| load1 | 30.0 | — |
| load5 | 29.9 | — |

Raw TSV: [`docs/plan_a_5090x4_metrics.tsv`](plan_a_5090x4_metrics.tsv) (135 lines incl. header). Schema: `ts_utc epoch g{0..3}_util g{0..3}_mem ret_cpu ret_rss_gb sg_sch_cpu_avg estab_3005 free_gb cache_gb load1 load5`. The `sg_sch_cpu_avg` column reads 0 in the TSV (the sampler's `ps -eo comm` truncated the scheduler comm to `sglang::schedu` so the regex missed); use the prose figure (~100 % per scheduler × 4) instead.

Interpretation:
- 41 % avg GPU util across 4 GPUs implies ~1.6× single-GPU effective speedup — **far below the ~3.5–4× ideal** for embarrassingly parallel work. Almost all of the gap is the retriever serialization (§7).
- Retriever process at ~24 cores out of 128 → CPU is not the constraint.
- RSS 139.4 GB matches 8 × ~17.5 GB index copies. RAM is the practical ceiling on `--num_retriever` without sharing the index.
- Load-avg 30 corresponds to the 4 SGLang schedulers (~100 % CPU each), 4 run_eval.py procs, retriever's OMP threads, and small overhead. Not CPU-saturated.

## 9. Lessons from the session

1. **Retriever's `async def` + sync FAISS = a hidden serialization point.** It looks like 8-way parallel concurrency but is effectively single-threaded. This is the dominant bottleneck for any multi-shard sweep on this code path. Fix is small.
2. **`run_one.sh`'s wrapper logs `END` regardless of run success.** If the inner `run_eval.py` is killed, the next iteration of the wrapper's for-loop just proceeds. There's no metric_score.txt for the failed run, so on the next session the run is retried (good), but **inside the same session the next dataset can lose a slot to an apparent-success step** (bad). Adding `set -e` or an exit-code check around the inner run would help. Out of scope for this run.
3. **Be careful with `pkill -f` selectors when a recent restart leaves siblings.** The string-match etime filter `> "12:00"` killed both old (19:12) and new (12:42) processes. A safer approach: filter by parent PID lineage of the current sweep wrappers.
4. **5090 + IVF-SQ8 + 4 shards is a working recipe** modulo the retriever fix. SGLang load time per GPU is ~30 s, retriever load with 8 workers ~70 s, sweep launch is one-line. Total bring-up ~3 minutes from a clean instance.

## 10. Reproducing this run

```bash
# 1. Retriever (8 workers, IVF-SQ8)
cd local_retriever
HF_HOME=/workspace/.hf_cache HF_DATASETS_CACHE=/workspace/.hf_cache/datasets \
  nohup python retriever_serving.py --config retriever_config.yaml \
    --num_retriever 8 --port 3005 \
    --index ./indexes/wiki18_100w_e5_ivf4096_sq8.index \
    > /workspace/logs/retriever_ivf_8w.log 2>&1 &
# wait for "Application startup complete"

# 2. 4 SGLang servers, one per GPU
EVAL_DIR=/workspace/reason_over_search/evaluation_search_r1
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i nohup /venv/evaluation_search_r1/bin/python \
    -m sglang.launch_server \
    --served-model-name "search_r1_base_g${i}" \
    --model-path "$EVAL_DIR/search_r1_base_model" \
    --tp 1 --context-length 8192 --enable-metrics --dtype bfloat16 \
    --host 127.0.0.1 --port $((3000+i)) --trust-remote-code \
    > /tmp/sglang_base_gpu${i}.log 2>&1 &
done
# poll all 4 ports' /get_model_info until ready

# 3. Compute LPT shard plan (see /tmp/shard_plan.txt for this run)

# 4. Launch 4 parallel sweep loops (one per port)
for SHARD in 0 1 2 3; do
  PORT=$((3000+SHARD))
  ( cd /workspace/reason_over_search
    export SGL_PORT=$PORT
    for J in $(grep "^shard${SHARD}\b" /tmp/shard_plan.txt | awk '{print $2"_"$3}'); do
      DS=${J%_*}; SEED=${J##*_}
      scripts/run_one.sh base "$DS" "$SEED"
    done
  ) > /tmp/sweep_shard${SHARD}.log 2>&1 &
done
```

## 11. Open follow-ups

- **Retriever async-fix branch.** Convert `search` and `batch_search` handlers from `async def` to `def`, drop the asyncio semaphore (FastAPI's threadpool already gates concurrency at 40). Re-run a single-shard Bamboogle to verify, then a 4-shard hotpotqa to measure throughput delta.
- **Resilient sweep wrapper.** Make the sweep loop check `run_one.sh` exit code before logging END.
- **Parallel sweep helper.** Add `scripts/sweep_a_parallel.sh --shards N` (or extend `sweep_a_full.sh`) so this multi-shard recipe is a single-command repeatable, not a paste-job.
- **Re-run popqa seeds 1/2/3** in the next session (resume-aware).
- **Apply same recipe to the instruct variant** if 5090 fleet stays available.

---

_This doc was written mid-run; it is updated through the deadline. Final EM table + final metrics aggregates are appended after the deadline-killer fires._
