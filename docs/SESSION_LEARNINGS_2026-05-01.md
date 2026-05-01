# Session learnings — 2026-05-01

Session: bring up Plan A on a 4× RTX 5090 Vast.ai box; pilot the base variant for 3 seeds × 7 datasets with the IVF-SQ8 retriever. Companion doc with run details: [`PLAN_A_5090x4.md`](PLAN_A_5090x4.md).

This file is the meta-doc — what we learned about the system and the process, separate from the run results.

## 1. Headline findings

### 1.1 The retriever has a hidden serialization point — `--num_retriever` is decorative

**Symptom:** doubling `--num_retriever 4 → 8` gave no aggregate throughput lift (per-process retriever CPU stayed flat at ~24 cores; combined GPU util stayed ~41 %).

**Root cause:** [`local_retriever/retriever_serving.py:70-101`](../local_retriever/retriever_serving.py) declares `async def search(...)` but calls a synchronous FAISS function inside it. uvicorn runs single-worker by default (no `--workers` flag), so the sync FAISS call blocks the entire event loop while it runs. The `asyncio.Semaphore(num_retriever)` gates a resource that is already serialized by the event loop — only one search executes at a time regardless of `num_retriever`.

**Test:** observed `ESTAB` socket count climb to 237 with `num_retriever=8` while the retriever process used ~24 cores total — same as `num_retriever=4`. If the semaphore actually unlocked concurrency we'd expect ~2× CPU usage.

**Fix (cheap, not applied this session):** change `async def search` and `async def batch_search` to plain `def`; FastAPI runs sync handlers in a 40-thread `ThreadPoolExecutor` automatically. Drop the asyncio semaphore (FastAPI's threadpool gates concurrency to 40). Or wrap the FAISS call in `await asyncio.to_thread(retriever.search, ...)`. Or add `uvicorn.run(..., workers=N)` (multi-process; doubles RAM since each process loads the index).

**Why this matters:** any multi-shard sweep on this code path is bottlenecked here. The full Plan A wall-clock estimate in `docs/EVAL_OPS.md` and `docs/VAST_AI_PLAN_A.md` assumed `--num_retriever` scales linearly — it does not.

### 1.2 Sharding 4 SGLang on 4 GPUs works modulo the retriever bug

The recipe (4 SGLang servers pinned via `CUDA_VISIBLE_DEVICES`, `SGL_PORT` env-var per worker, shared retriever, LPT bin-pack of jobs) is one-line-per-shard simple to launch and resume-aware via `run_one.sh`'s existing skip guard. The orchestration overhead is small. Once the retriever async fix lands, this should give close to 4× aggregate speedup.

## 2. What worked (patterns to repeat)

### 2.1 LPT bin-packing of `(dataset, seed)` jobs

Sort jobs by item count descending, assign each to the least-loaded bin. With 21 jobs over 4 bins it produced a `max/min = 1.024` imbalance ratio — effectively perfect balance. Tiny Python script in `/tmp/shard.py`; the bin assignments live in `/tmp/shard_plan.txt`.

A naïve round-robin gave 1.31×; greedy seed-sharded gave 1.26×. LPT is barely more code and significantly tighter.

### 2.2 `SGL_PORT` env var on `run_one.sh`

Two-line change:

```bash
SGL_PORT="${SGL_PORT:-3000}"
...
  --sgl_remote_url 127.0.0.1:$SGL_PORT \
```

Default keeps existing behavior; new shards just `export SGL_PORT=300X` before invoking. Backward-compatible, no flag plumbing.

### 2.3 Disowned subshell guards (deadline-killer + metric sampler)

Pattern:

```bash
( until [ "$(date +%s)" -ge "$DEADLINE" ]; do sleep 30; done
  kill -9 $SWEEP_PIDS
) > /tmp/x.log 2>&1 &
disown
```

A separate background subshell that polls a wall-clock condition and acts. Survives the parent Bash exit. Multiple of these can run in parallel (deadline-killer + metric sampler).

The harness emits a "background command completed" notification when the *parent* Bash exits — that's a false alarm; verify via `ps -p $PID` whether the disowned subshell is still alive.

### 2.4 Monitor task for run-completion events

`Monitor` with `tail -F .../sweep_*.log | grep -E "START|END|'em':|Traceback|FAILED"` gave one notification per dataset boundary plus EM scores at completion. Critical: **don't put the raw log into the monitor stream** — only the boundaries. Otherwise tqdm's per-batch progress lines flood the notification channel.

When a Monitor times out (default 1 h), re-arm it; events queued during the gap may flush in one batch.

### 2.5 Periodic 30 s metric sampler

Bash background loop writing TSV (`ts, epoch, gpu0..3 util/mem, retriever cpu/rss, scheduler cpu, ESTAB conns, free RAM, cache, load1/5`). Cheap: ~0 % CPU, accumulates time-series data automatically. Avoids "I wish I had measured X" regret. 134 samples × 22 columns over 67 minutes. The TSV lives at [`plan_a_5090x4_metrics.tsv`](plan_a_5090x4_metrics.tsv).

Watch out: `ps -eo comm` can truncate process names — our `sg_sch_cpu_avg` column read 0 because `comm` was returning `sglang::schedu` and our awk filter missed. Use `-o comm=` with width or filter on the full `cmd`.

## 3. What didn't work (mistakes to avoid)

### 3.1 String-comparison filter on `etime`

**Bad:**

```bash
ps -eo pid,etime,cmd | grep run_eval.py | awk '$2 > "12:00"{print $1}'
```

This was meant to kill orphan run_eval procs older than 12 minutes. But `"12:42" > "12:00"` is true lexicographically — string comparison killed both the orphans **and** the just-restarted new procs. Cost: 12 minutes of progress on three popqa runs (lost entirely; sweep wrappers logged `END` and moved on).

**Right way:** use a `--ppid` filter (kill only procs whose parent is not in the current sweep wrapper PID set) or compare numerical seconds (`awk -F: '($2*60 + $3) > 720{...}'`). Or — simpler — when you want to kill orphans, don't kill anything; kill the parent wrappers first and only those, then `pkill` children that are already orphan.

### 3.2 Sweep wrappers don't check exit codes

The shard loop is:

```bash
for J in $JOBS; do
  scripts/run_one.sh base "$DS" "$SEED" 2>&1 | tail -6
  echo "[shard$SHARD] ... END $DS seed=$SEED"
done
```

`run_one.sh` exiting non-zero (because `run_eval.py` was killed) does **not** stop the loop and does **not** suppress the `END` log. Combined with §3.1, it ate dataset slots silently — the wrappers moved on as if popqa had succeeded. The result-dir check from `run_one.sh:49` saves the data side (no `metric_score.txt` written → next session will retry) but the **current** session's wrapper is hopelessly forward-only.

**Fix to add later:**

```bash
if scripts/run_one.sh ... ; then
  echo "[shard$SHARD] ... END $DS seed=$SEED"
else
  rc=$?
  echo "[shard$SHARD] ... FAIL $DS seed=$SEED rc=$rc"
fi
```

Don't `set -e` the loop — we *want* to continue past one failed dataset.

### 3.3 Piping `run_one.sh` through `tail -6`

The wrapper does `scripts/run_one.sh ... 2>&1 | tail -6` to keep logs short. But tqdm's progress (most of the output) only flushes at end-of-stream, so mid-run progress is invisible to the wrapper log. We saw nothing useful until each run finished. Net cost: had to peek at SGLang `/metrics` for QPS to estimate progress.

Fix: drop the `| tail -6`. Or `tee` to a per-run log file and let progress stream live.

### 3.4 "Background command completed" notifications

The Bash tool fires this notification when the **launcher Bash exits**, not when the disowned subshell exits. The first time it happened I worried the deadline-killer had died; verifying via `ps -p $PID` showed it was fine. Now: ignore the notification, verify via `ps`.

### 3.5 Long sleeps blocked

The runtime blocks `sleep` longer than ~60 s as the leading instruction in a Bash call. To wait until a deadline, use `until [ condition ]; do sleep 30; done` (the runtime allows this) — or `run_in_background: true` for one-shot waits. Direct `sleep 3600` is rejected.

## 4. Process learnings

### 4.1 Default to small numbers when restarting services

Bumping `--num_retriever` from 4 to 8 cost ~70 GB RAM (two retriever copies of the 17.5 GB index each, × 4 new workers). Available RAM was 86 GB after — tight. Going to 16 would have OOMed (need 280 GB just for retrievers). Always compute the RAM cost before changing a worker count.

The actual ceiling on this box is: RAM permits ~8 retrievers, but the async-serialization bug means anything above 1 worker provides no benefit. With the fix (FastAPI's 40-thread pool), 1–2 workers should be enough; the threadpool absorbs concurrency.

### 4.2 Multi-shard kills are tricky — kill in dependency order

Kill order matters when wrappers respawn children:

1. **Wrappers first** (the `for J in $JOBS; do scripts/run_one.sh; done` loops) — otherwise they immediately spawn the next iteration after you kill the current `run_eval.py`.
2. **Then `run_one.sh`** processes.
3. **Then `run_eval.py`** processes.
4. **Then SGLang** if you want to free GPU memory.

Reverse order leaks new children continuously.

### 4.3 SGLang switch cost is real

`scripts/manage_sglang.sh switch` takes ~30–60 s per variant. The 4-shard recipe pins each SGLang to one variant for the whole session — switching mid-sweep would cost ~2 minutes per switch × number of switches. The Plan A doc's "shard by `(variant, seed)`" recommendation already accounts for this.

## 5. Architecture notes (gleaned this session)

### 5.1 SearchR1Pipeline runs ThreadPoolExecutor(max_workers=16) per shard

[`flashrag/pipeline/active_pipeline.py:178-180`](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py) — each `run_eval.py` fans out 16 concurrent `run_item` calls. Each `run_item` makes its own sequence of `/search` and `/generate` calls. So with 4 shards, the *demand* on the retriever can spike to 64 concurrent `/search` calls.

This is why the retriever-side async serialization hurts so much. The supply (1 effective worker) vs demand (64 concurrent callers) ratio is brutal.

### 5.2 Search-R1 is multi-turn; QPS ≠ items/s

Each item in Search-R1 averages ~3–4 turns (initial generation → search → continued generation → search → ... → answer). On SGLang `/metrics` we saw ~8.7 req/s per port; that's ~2.2 items/s per shard. Useful for ETA estimation: `items / (QPS / 4)`.

### 5.3 The 4× 5090 box reports as 4× ~32 GB

The repo's `docs/HARDWARE.md` describes a single-4090 box. This session's box was different — 4× RTX 5090 (32 GB each), AMD EPYC 7763 (64C/128T), 251 GB RAM, on Vast.ai (visible via the SSH tunnel to `ssh9.vast.ai`). Don't trust HARDWARE.md when running on Vast — re-check `nvidia-smi` and `lscpu`.

### 5.4 Result-dir convention

`evaluation_search_r1/results/<dataset>/<dataset>_YYYY_MM_DD_HH_MM_search_r1_<variant>_seed<N>/` contains `config.yaml`, `intermediate_data.json` (large — 50–115 MB), `metric_score.txt` (3 lines: em/acc/f1).

`run_one.sh:49` resume-skips on `metric_score.txt` existence — empty dirs (created at run-start, no metric file because killed mid-flight) won't trigger the skip; the next session will retry.

`intermediate_data.json` is too large to commit (committed `_archive_v0/` ones are 9–12 MB; ours are 46–115 MB — possibly because we run with the IVF-SQ8 retriever return format, vs flat).

## 6. Open follow-ups (in priority order)

1. **Retriever async-fix branch.** Convert `search`/`batch_search` to `def`; drop or sync-replace the semaphore. Verify on Bamboogle single-shard (sanity), then 4-shard hotpotqa (throughput delta). Hypothesis: 3–4× lift in items/s per shard.
2. **Sweep wrapper exit-code awareness.** Patch the `for J in $JOBS` loops to log `FAIL` on non-zero exit, and stop spawning new runs after N consecutive failures. Out of scope here; small change in our orchestrator script.
3. **`scripts/sweep_a_parallel.sh --shards N`.** Promote the inline 4-shard launcher used in this session to a versioned script. Should compute LPT, launch SGLang+sweep workers, write to per-shard logs, and ship a shutdown helper.
4. **Re-run the lost runs.** popqa-1/2/3 (orphan-kill loss) + hotpotqa-1/2/3 + triviaqa-1 + nq-1/2/3 + musique-1/2/3 + bamboogle-2/3. Resume-aware so just re-launch with the same shard plan.
5. **Apply the recipe to the instruct variant.** 21 more runs once base is finished.
6. **Update `docs/HARDWARE.md`** to mention "this varies on Vast" or move it out of canonical docs.

## 7. Quick-reference commands

```bash
# Status during a sweep
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
free -h
curl -sS http://127.0.0.1:3005/health
for p in 3000 3001 3002 3003; do curl -sS http://127.0.0.1:$p/get_model_info | jq .model_path; done

# Per-port SGLang request count (cumulative since boot)
for p in 3000 3001 3002 3003; do
  curl -sS http://127.0.0.1:$p/metrics | awk '/^sglang:num_requests_total/{printf "port=%s %s\n", "'"$p"'", $2}'
done

# Active retriever connections
ss -tn '( dport = :3005 or sport = :3005 )' | awk 'NR>1 && $1=="ESTAB"{c++}END{print c}'

# Find completed runs in a sweep window
START=$(cat /tmp/sweep_a_base.start_epoch)
find evaluation_search_r1/results -name 'metric_score.txt' -newermt "@$START" \
  -exec bash -c 'echo "$(basename $(dirname $1)): $(cat $1)"' _ {} \;

# Clean kill of multi-shard sweep
for P in $(cat /tmp/sweep_pids.txt); do kill -9 "$P"; done   # wrappers first
sleep 1
pkill -KILL -f "scripts/run_one.sh"
pkill -KILL -f "run_eval.py"
```

## 8. Time accounting

Total session: ~3 h. Rough breakdown:

| activity | time |
|---|---:|
| Bamboogle base sanity (single-shard, 4-worker retriever) | 5 min |
| 4-shard SGLang setup + first sweep launch | 10 min |
| First-pass benchmark + diagnosis (4-worker retriever) | 15 min |
| Stop, restart retriever with 8 workers, re-launch | 10 min |
| Bottleneck investigation (the async-serialization finding) | 25 min |
| Sweep run window (4 runs completed) | 80 min |
| Doc + commit | 15 min |
| **Lost to mistakes** (orphan-kill + popqa retry budget) | ~12 min |

The 12-minute orphan-kill cost is in the run window. If the retriever async fix is applied next session, that throughput gain (3–4×) dominates everything else.
