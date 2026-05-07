---
title: CODE_SETUP — Plan A (1 seed) on 8×4090
tags: [eval, plan-a, code-setup, 8gpu]
source: internal
created: 2026-05-07
updated: 2026-05-07
---

# Code setup — what changed for the 8×4090 1-seed Plan A sweep

**Date**: 2026-05-07
**Branch**: `plan-a-eval`
**Scope**: Documents what changed from `plan-a-eval` HEAD prior to this sweep to the version that runs the 8-GPU 3-variant sweep. Mirrors the v0 → v1 diff format from [`docs/report/CODE_SETUP_v1.md`](../../report/CODE_SETUP_v1.md), but for the eval-side (not training).

## 1. Headline diff

| Dimension | Before (`plan-a-eval` HEAD) | After (this sweep) |
|---|---|---|
| **Hardware target** | Single 4090 (sequential) | 8× 4090 (parallel) |
| **Concurrency model** | One SGLang on port 3000, sequential dataset loop | 8 SGLang servers on ports 3000-3007, parallel dataset dispatch |
| **Variants** | `base`, `instruct` (Search-R1 GRPO) | `base`, `instruct`, `qwen_25_3b_instruct` (raw Qwen2.5-3B-Instruct) |
| **Retriever concurrency** | `async def search` + sync FAISS = single-threaded under the hood | `def search` + FastAPI thread pool, paired 1:1 with SGLang clients |
| **Retriever topology** | 1 process | **8 paired processes**, ports 3005..3012 (one per SGLang server) |
| **Retriever RAM** | ~17 GB (single index) | ~136 GB (8 × 17 GB) — fits the 200-250 GB host; deliberate trade for predictable per-shard latency and `OMP_NUM_THREADS` bounds |
| **FAISS OMP cap** | unbounded (~24 cores per search) | `OMP_NUM_THREADS=8` per retriever process (8 × 8 = 64 cores total on 64-core host) |
| **Index** | IVF-SQ8 (default) or flat IP (paper-quality option) | IVF-SQ8 only — no flat fallback for this run |
| **`HF_HOME`** | unset, falls back to `~/.cache/huggingface` | exported to `/workspace/hf_cache` (matches Dockerfile); ancillary HF cache writes stay on the persistent volume |
| **`HF_DATASETS_CACHE` / `TRANSFORMERS_CACHE`** | unset → ephemeral overlay root | exported to `/workspace/hf_datasets_cache` and `/workspace/hf_cache`; required to keep the 14 GB corpus parquet off the 25 GB overlay |
| **`qwen_25_3b_instruct` model location** | n/a (variant didn't exist) | flat local dir at `evaluation_search_r1/qwen_25_3b_instruct/`, identical convention to `search_r1_base_model/`. SGLang and eval pipeline both resolve via this absolute path; no HF cache, no symlink, no `model2path` entry. |
| **Sweep wrapper** | `sweep_c_one_seed.sh` (sequential 14 runs) | `sweep_8gpu_one_seed.sh` (parallel 21 runs) |
| **Wall-clock** | ~80 h (Plan C, 14 runs) | ~6 h (this run, 21 runs) |
| **Output docs** | one-off `docs/RESULTS_PLAN_C.md` | folder `docs/eval/plan_a_8gpu/` (BOOTSTRAP / CODE_SETUP / RESULTS / SESSION_LOG) |

## 2. What's unchanged

These are paper-faithful and stay identical:

| Knob | Value |
|---|---|
| Prompt protocol | Search-R1 paper: `<search>` / `<information>` / `<answer>` |
| Chat template | `apply_chat=True` for all three variants |
| Sampling | greedy (`temperature=0`) per [`docs/eval/REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) |
| `retrieval_topk` | 3 |
| `max_search_turns` | per `evaluation_search_r1/flashrag/config/basic_config.yaml` |
| FAISS retriever encoder | `intfloat/e5-base-v2` |
| Datasets | NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle (paper's 7) |
| Splits | test where it exists, dev otherwise (per `run_one.sh` mapping) |
| Resume mechanism | `metric_score.txt` presence skip in `run_one.sh:48-53` |

## 3. Files changed

| File | Change | Why | Source |
|---|---|---|---|
| `local_retriever/retriever_serving.py` | `async def` → `def` for `/search` and `/batch_search`; drop `asyncio.Semaphore` and `available_retrievers` deque; use `retriever_list[0]` directly; init only one retriever (warn if `--num_retriever > 1`) | Unblock per-process throughput. Per `docs/PLAN_A_5090x4.md` §7, `async def` + sync FAISS serialized concurrent searches on a single-worker uvicorn event loop. Even with paired fleet, this is needed if SGLang issues overlapping `batch_search` calls within a process. | `PLAN_A_5090x4.md` §7,9 |
| `scripts/run_one.sh` | Add `SGL_PORT="${SGL_PORT:-3000}"`; replace `127.0.0.1:3000` with `127.0.0.1:$SGL_PORT` in `--sgl_remote_url` | Each shard worker can point at its own SGLang port without forking the script. | cherry-pick of commit `425c833` from `plan-a-5090x4` |
| `scripts/run_one.sh` | Add `qwen_25_3b_instruct` variant case (`apply_chat=True`, `generator_model=qwen_25_3b_instruct`) | New raw baseline — same family/size as the Search-R1 instruct ckpt's pre-training base, isolates GRPO contribution. | this work |
| `scripts/run_one.sh` | Add `RETRIEVER_URL="${RETRIEVER_URL:-127.0.0.1:3005}"`; thread into `--remote_retriever_url`. | Each shard worker hits its paired retriever port (3005..3012). Mirrors the SGL_PORT pattern. | Q3 delta |
| `scripts/manage_sglang.sh` | Parametrize `PORT` via env var (default 3000); add `qwen_25_3b_instruct` case using `$EVAL_DIR/qwen_25_3b_instruct` (mirrors the `base`/`instruct` cases — all three variants resolve to flat dirs under `evaluation_search_r1/`); add `_launch_one` helper; add `start_fleet` / `wait_fleet` / `stop_fleet` subcommands tracking pids in `/tmp/sglang_fleet.pids`; `wait_ready` accepts optional port | Programmatic 8-GPU bring-up; the 5090 session did this manually with ad-hoc bash loops, lost runs to `pkill -f` foot-guns. Uniform path resolution lets the eval pipeline's `model2path.get(name, name)` fallback work for all three variants without `model2path` entries. See §5d. | this work + `PLAN_A_5090x4.md` §9.3 |
| `local_retriever/launch_ivfsq8.sh` (new) | `start` / `stop` / `wait` for single-instance (smoke); `start_fleet` / `wait_fleet` / `stop_fleet` for paired fleet on ports 3005..3012 with `OMP_NUM_THREADS=${OMP_RETRIEVER:-8}`. Pidfiles at `/tmp/retriever_ivfsq8_PORT.pid` (per-port) and `/tmp/retriever_fleet.pids` (fleet list). | Standardize lifecycle; `OMP_NUM_THREADS` cap is the load-bearing knob to avoid CPU oversubscription with 8 paired retrievers. | this work + Q3 delta |
| `local_retriever/smoke_concurrent.sh` (new) | Health check + sample query + sequential-vs-parallel timing. PASS at ≥3× speedup, OK at 2-3×, FAIL at <2×. Run after `start_fleet` to confirm the async fix is live before launching the full sweep. | Single-command verification that the retriever is loaded AND parallelizing — catches a stale-`.pyc` regression in seconds. | this work |
| `scripts/sweep_8gpu_one_seed.sh` (new) | Top-level driver: defensive `export HF_HOME=${HF_HOME:-/workspace/hf_cache}`; bring up retriever fleet once; for each variant `start_fleet` (sglang) → parallel dispatch of 7 datasets via `SGL_PORT=$((3000+i))` + `RETRIEVER_URL=127.0.0.1:$((3005+i))` → `wait` → SGLang `stop_fleet`. Trap stops both fleets on exit. Aggregates into `RESULTS.md`. | Equivalent of what `plan-a-5090x4` ran as 4 manual bash for-loops, now with paired retriever pinning. | this work + Q3 delta + Q2 |
| `scripts/sample_metrics_8gpu.sh` (new) | 30-s sampler writing `docs/eval/plan_a_8gpu/system_metrics.tsv` (mirrors 5090 schema, expanded to 8 GPUs). | Live observability during the 6-h sweep; predecessor TSV format. | this work |
| `docs/eval/plan_a_8gpu/` (new folder) | `BOOTSTRAP.md`, `CODE_SETUP.md` (this), `RESULTS.md`, `SESSION_LOG.md` | User ask: scaffold output docs at run-start, fill live as the run progresses (mirroring `docs/report/CODE_SETUP_v1.md` style). | this work |
| `docs/PLAN_A_5090x4.md` + `docs/plan_a_5090x4_metrics.tsv` | Cherry-picked from sister branch `plan-a-5090x4`; added frontmatter and fixed `setup/` link paths. | Predecessor run-doc that all four new docs link into for bottleneck analysis (§7) and metric schema. | this work |

## 4. What's deliberately *not* changed

- **`scripts/sweep_a_full.sh`** — Plan A's canonical 5-seed 70-run script stays exactly as it was. This sweep is the 1-seed parallel version, not a replacement.
- **`scripts/sweep_c_one_seed.sh`** — kept as the canonical sequential 1-seed reference.
- **`evaluation_search_r1/run_eval.py`, `flashrag/search_r1/`** — no eval-pipeline code changes. Paper protocol stays identical, raw Qwen rides the same path.
- **`qwen_native` arm** from `docs/milestone_one/MILESTONE_1.1_QWEN_BASELINES.md` — that's a Qwen3.5-2B baseline with the `<tool_call>` protocol, out of scope here.
- **Flat FAISS index** — explicitly not used for this run. IVF-SQ8 throughout.

## 5. Critical architectural notes

### 5a. Retriever async fix (per-process throughput)

The async fix is still load-bearing even though we run paired retrievers. Without it:

```
SGLang client ──HTTP──> retriever process
                        └─> FastAPI single-worker uvicorn
                            └─> async def search:
                                await asyncio.Semaphore.__aenter__   # async
                                retriever.search(query, ...)         # SYNC (FAISS)  ← blocks event loop
```

If SGLang issues overlapping `batch_search` calls (concurrent rollouts within a request), the synchronous FAISS call inside `async def` blocks the event loop. The 5090 session measured this empirically — per-process retriever CPU stayed flat at ~24 cores from `--num_retriever 4` to `--num_retriever 8`; GPU util sat at 41% across 4 shards (`docs/PLAN_A_5090x4.md` §7,8).

After the fix:

```
SGLang client ──HTTP──> retriever process (1 of 8)
                        └─> FastAPI uvicorn (sync handler thread pool)
                            └─> def search:
                                retriever_list[0].search(query, ...)  # FAISS OMP, capped at OMP_NUM_THREADS=8
```

### 5b. Paired retrievers, OMP-capped (system throughput)

Single-shared retriever post-async-fix works for ≤4 clients but oversubscribes CPUs at 8: FastAPI thread pool (default 40) lets multiple FAISS searches run concurrently, each spawning the default ~24 OMP threads → severe contention on a 64-core box. Two ways to bound it:

1. **One process, OMP-capped, threadpool-capped**: 1 retriever, `OMP_NUM_THREADS=8`, threadpool=8. RAM ~17 GB. Throughput: 8 concurrent searches × 8 cores = 64 cores. Risk: any residual GIL or thread-pool weirdness still bites.

2. **8 processes, OMP-capped, 1:1 with SGLang** *(chosen)*: 8 retrievers on ports 3005..3012, each with `OMP_NUM_THREADS=8`. RAM ~136 GB (fits 200-250 GB host). Equivalent CPU profile, defensive against any cross-talk in option 1.

The 5090 session's 8-worker single-process retriever (~140 GB) was the wrong shape because the 8 in-process workers were a placebo (option 1 without the fix); we get the same RAM cost as a side-effect of the right shape (option 2 with the fix).

### 5c. HF cache pinning + datasets-cache redirect

Three env vars cooperate; missing any one breaks a different part of the pipeline.

`HF_HOME=/workspace/hf_cache` — model weights are now downloaded to flat dirs under `evaluation_search_r1/` (see §5d), so `HF_HOME` no longer governs Qwen storage directly. It still matters for any incidental cache writes by `huggingface_hub` (download metadata, partial-download recovery, and any HF tokenizer fetches that aren't redirected via `TRANSFORMERS_CACHE`). Pinning to `/workspace/hf_cache` keeps those off the ephemeral overlay root and matches `docker/reason-over-search-v1/Dockerfile:138`.

`HF_DATASETS_CACHE=/workspace/hf_datasets_cache` — `flashrag/retriever/utils.py:130` calls `datasets.load_dataset('json', data_files=corpus_path, split="train")` to ingest the 14 GB wiki corpus, materializing a parquet/arrow cache. The default location is `~/.cache/huggingface/datasets/` on the small overlay root (~25 GB total). Without redirect, the retriever crashes mid-cold-load with `[Errno 28] No space left on device`. STAGING.md Step 5 and BOOTSTRAP.md Step 2 both export this defensively.

`TRANSFORMERS_CACHE=/workspace/hf_cache` — the encoder (`intfloat/e5-base-v2`) and tokenizer caches; same root-disk concern as `HF_HOME`. Setting it to the same path keeps everything on the persistent volume.

### 5d. Qwen variant path resolution (`qwen_25_3b_instruct`)

The eval pipeline expects `generator_model_path` to be a local directory: `flashrag/utils/utils.py:47` opens `config.json` from it directly, and `flashrag/config/config.py:219` does `model2path.get(name, name)` — falling through to use the alias as the path itself if no entry exists. From `cd $EVAL_DIR` (which `run_eval.py` does), the alias `qwen_25_3b_instruct` resolves to the relative directory `evaluation_search_r1/qwen_25_3b_instruct/`. The `base`/`instruct` variants work the same way — their aliases (`search_r1_base_model`, `search_r1_instruct_model`) match local directory names under `evaluation_search_r1/`. No `model2path` entry needed for any of the three.

`scripts/manage_sglang.sh` mirrors this on the SGLang side: all three variant cases pass `$EVAL_DIR/<name>` as `--model-path`. SGLang accepts both HF identifiers and local paths; using the local path keeps everything one-source-of-truth and avoids an HF-hub round-trip when the model is already on disk.

STAGING.md Step 4 downloads the raw Qwen with `hf download Qwen/Qwen2.5-3B-Instruct --local-dir qwen_25_3b_instruct` (run from inside `evaluation_search_r1/`). BOOTSTRAP.md Step 2 verifies it survived the volume re-attach via `test -d evaluation_search_r1/qwen_25_3b_instruct`.

### 5e. Smoke gate at N=8 vs N=16

`smoke_concurrent.sh` PASS threshold is ≥3× speedup. On hosts with ≥64 cores, fixed per-call overhead (encoder pass, JSON parse, mmap I/O on the corpus arrow) caps the N=8 ratio at ~2.4–2.8× even with the async fix fully active — landing in the OK band. The canonical recipe is now: run `smoke 3005 8` first; if verdict is OK, retry at `smoke 3005 16`. PASS at either count satisfies the hard gate. On smaller staging boxes (16-core), N=8 alone clears 3× because per-call overhead is a smaller share of the wall-clock.

## 6. Smoke results (state at run-start)

To be filled in by the operator after Step 6 of [`BOOTSTRAP.md`](BOOTSTRAP.md):

| Metric | Value |
|---|---|
| Single-GPU Bamboogle on `qwen_25_3b_instruct` (n=125, greedy) | EM = TODO, F1 = TODO, wall-clock = TODO |
| 2-GPU fleet smoke (Bamboogle + PopQA, ports 3000-3001) | wall-clock = TODO, no port collision = TODO |
| Retriever `/health` post-async-fix | TODO |
| Average GPU util across 8 GPUs during `base` phase | TODO (from `nvidia-smi dmon`) |

The full per-(variant, dataset) results land in [`RESULTS.md`](RESULTS.md) once `aggregate.py` runs at sweep end.

## 7. Pointers

- Full audit of the retriever bottleneck: [`docs/PLAN_A_5090x4.md` §7,8](../../PLAN_A_5090x4.md)
- Predecessor session log: [`docs/PLAN_A_5090x4.md`](../../PLAN_A_5090x4.md)
- Bootstrap runbook: [`BOOTSTRAP.md`](BOOTSTRAP.md)
- Sweep wrapper: [`scripts/sweep_8gpu_one_seed.sh`](../../../scripts/sweep_8gpu_one_seed.sh)
- Fleet helpers: [`scripts/manage_sglang.sh`](../../../scripts/manage_sglang.sh) (`start_fleet`, `wait_fleet`, `stop_fleet`)
- IVF-SQ8 wrapper: [`local_retriever/launch_ivfsq8.sh`](../../../local_retriever/launch_ivfsq8.sh)
