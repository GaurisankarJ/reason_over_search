---
title: SESSION LOG — Plan A (1 seed) on 8×4090
tags: [eval, plan-a, session-log, 8gpu]
source: internal
created: 2026-05-07
updated: 2026-05-07
---

# Session log — Plan A 1-seed sweep on 8×4090

Running journal for this sweep. Append timestamped notes here as the run progresses; mirror the structure of [`docs/PLAN_A_5090x4.md`](../../PLAN_A_5090x4.md) §5,8,9.

## Stage 1 — staging + validation (cheap instance)

Per [`STAGING.md`](STAGING.md). Instance shape: `_____` (1×4090 or CPU-only). Hourly rate: `$_____/hr`. Wall-clock: `_____ min`.

- [ ] Image + venvs verified
- [ ] All artifacts staged on `/workspace`:
  - corpus (~14 GB), index (~16 GB), encoder (~2 GB), Search-R1 base + instruct ckpts (2 × ~13 GB), Qwen2.5-3B-Instruct in `$HF_HOME` (~7 GB)
- [ ] Search-R1 base ckpt sha256-verified: `7ac54e1b...`
- [ ] Search-R1 instruct ckpt sha256-verified: `3d787062...`
- [ ] **Async fix smoke** — `local_retriever/smoke_concurrent.sh 3005 8` returned: `_____` (target: PASS, ≥3× speedup)
  - sequential time: `_____ s`, parallel time: `_____ s`, speedup: `_____×`
- [ ] (GPU instance only) Single-GPU SGLang smoke EM on Bamboogle qwen_25_3b_instruct: `_____`
- [ ] Persistent volume detached / snapshot ID: `_____`

Stage 1 cost: $`_____`.

## Stage 2 — actual run (8×4090)

Per [`BOOTSTRAP.md`](BOOTSTRAP.md). Instance shape: `8× RTX 4090`. Hourly rate: `$_____/hr`.

### Setup

- [ ] Persistent volume from stage 1 mounted at `/workspace`
- [ ] Repo branch `plan-a-eval` at commit `<sha>`: `_____`
- [ ] Host RAM ≥ 200 GB confirmed (`free -h`): `___ GB`
- [ ] Host CPU ≥ 64 cores confirmed (`nproc`): `___ cores`
- [ ] `HF_HOME` exported to `/workspace/hf_cache`: `echo $HF_HOME` → `___`
- [ ] All 8 GPUs visible (`nvidia-smi --query-gpu=count --format=csv,noheader`): `___`
- [ ] All staged artifacts present (Step 2 of BOOTSTRAP, no `MISS:` printed)
- [ ] Async fix on this checkout: `grep -c '^async def search' local_retriever/retriever_serving.py` → `0`
- [ ] Retriever fleet `/health` on ports 3005..3012 — all 8 healthy
- [ ] Re-validation `smoke_concurrent.sh 3005 8` → PASS
- [ ] OMP cap verified: per-proc Threads via `cat /proc/$PID/status | grep Threads`

## Smoke (Step 6)

- [ ] Single-GPU `qwen_25_3b_instruct` × Bamboogle launched at `<UTC>`
- [ ] 3 random rollouts inspected — `<search>` / `<answer>` tags present, `<information>` injected cleanly
- [ ] EM > 0 confirmed: `EM = ___`
- [ ] Wall-clock for Bamboogle smoke: `___ min`
- [ ] No port collisions, no SGLang errors in `/tmp/sglang_qwen_25_3b_instruct_p3000.log`

## Phase 1 — base

Start: `<UTC>`

| GPU | port | dataset | n | started | ended | wall-clock | EM |
|---:|---:|---|---:|---|---|---:|---:|
| 0 | 3000 | nq | 3610 | | | | |
| 1 | 3001 | triviaqa | 11313 | | | | |
| 2 | 3002 | popqa | 14267 | | | | |
| 3 | 3003 | hotpotqa | 7405 | | | | |
| 4 | 3004 | 2wikimultihopqa | 12576 | | | | |
| 5 | 3005 | musique | 2417 | | | | |
| 6 | 3006 | bamboogle | 125 | | | | |
| 7 | — | (idle) | — | — | — | — | — |

End: `<UTC>` — phase wall-clock: `___`. Notes:

## Phase 2 — instruct

Start: `<UTC>`

| GPU | port | dataset | n | started | ended | wall-clock | EM |
|---:|---:|---|---:|---|---|---:|---:|
| 0 | 3000 | nq | 3610 | | | | |
| 1 | 3001 | triviaqa | 11313 | | | | |
| 2 | 3002 | popqa | 14267 | | | | |
| 3 | 3003 | hotpotqa | 7405 | | | | |
| 4 | 3004 | 2wikimultihopqa | 12576 | | | | |
| 5 | 3005 | musique | 2417 | | | | |
| 6 | 3006 | bamboogle | 125 | | | | |
| 7 | — | (idle) | — | — | — | — | — |

End: `<UTC>` — phase wall-clock: `___`. Notes:

## Phase 3 — qwen_25_3b_instruct (raw)

Start: `<UTC>`

| GPU | port | dataset | n | started | ended | wall-clock | EM |
|---:|---:|---|---:|---|---|---:|---:|
| 0 | 3000 | nq | 3610 | | | | |
| 1 | 3001 | triviaqa | 11313 | | | | |
| 2 | 3002 | popqa | 14267 | | | | |
| 3 | 3003 | hotpotqa | 7405 | | | | |
| 4 | 3004 | 2wikimultihopqa | 12576 | | | | |
| 5 | 3005 | musique | 2417 | | | | |
| 6 | 3006 | bamboogle | 125 | | | | |
| 7 | — | (idle) | — | — | — | — | — |

End: `<UTC>` — phase wall-clock: `___`. Notes:

## System metrics

Sample every 30 s with a sidecar (model on `docs/PLAN_A_5090x4.md` §8). Append the TSV path here once written.

| metric | avg | peak |
|---|---:|---:|
| GPU 0–7 util | TBD | TBD |
| GPU combined avg | TBD | — |
| retriever CPU (% of one core) | TBD | — |
| retriever RSS (GB) | TBD | — |
| ESTAB conns to :3005 | TBD | — |
| free RAM (GB) | TBD | — |
| load1 / load5 | TBD | — |

## Lessons

To be filled at run end. Mirror the format of [`docs/PLAN_A_5090x4.md`](../../PLAN_A_5090x4.md) §9 — what worked, what surprised, what to fix next time. Confirm or refute the central hypothesis from the predecessor run: that the async fix lifts 4-shard 41% GPU util to ≥80% across 8 shards.

### Hypotheses going in

1. **Async fix + paired-retriever fleet unblocks 8-GPU throughput** — 5090 run was 41% GPU util at 4 shards (single placebo retriever, no async fix). With both fixes and 1:1 SGLang↔retriever pairing at `OMP_NUM_THREADS=8`, expect ≥75% GPU util across 8 shards. Falsifiable by `nvidia-smi dmon -s u` averages and `system_metrics.tsv`.
2. **Paired-retriever RAM ~136 GB** — 8 processes × ~17 GB. Sum across `ps -o rss` on the 8 fleet pids should land in the 130-150 GB band; total host free RAM after fleet boot ≥ 50 GB.
3. **OMP cap holds** — total FAISS thread count across all 8 retriever processes stays ≤ 8 × 8 = 64 (sum of `Threads` in `/proc/PID/status`). If unbounded, contention will tank throughput.
4. **HF cache survives container restart** — `$HF_HOME/hub/models--Qwen--Qwen2.5-3B-Instruct/` persists; second `start_fleet qwen_25_3b_instruct` finishes in <30 s (no re-download).
5. **GRPO delta is positive on the instruct row** — Search-R1 instruct GRPO will outscore raw Qwen2.5-3B-Instruct on every benchmark. The interesting question is *by how much*, per dataset.

## Open questions to resolve before the next run

- TBD

## See also

- [`BOOTSTRAP.md`](BOOTSTRAP.md) — runbook
- [`CODE_SETUP.md`](CODE_SETUP.md) — code changes
- [`RESULTS.md`](RESULTS.md) — final per-(variant, dataset) table
- [`docs/PLAN_A_5090x4.md`](../../PLAN_A_5090x4.md) — predecessor 4-shard run
