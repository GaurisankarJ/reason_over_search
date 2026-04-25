# Search-R1 Evaluation Sweep — Plans & Analysis

This document describes the three sweep plans implemented in `scripts/`,
explains the trade-offs, and documents how to run each plan and aggregate
results. Run when you've already applied the fixes from
`SEARCH_R1_FIXES_REVIEW.md` and verified the smoke test passes.

---

## Why three plans

The originally requested sweep — **5 seeds × 7 datasets × 2 model variants =
70 runs** — covers ~517 K example evaluations. Each example is a multi-turn
conversation hitting both the SGLang generator (24 GB single 4090) and the
1-worker FAISS retriever. Smoke test on Bamboogle base (the slowest, hardest
multi-hop) measured **2.85 s/example** wall-clock end-to-end. Extrapolating:

| Plan | Runs | Examples | Wall-clock @ 2.85 s/ex |
|---|---:|---:|---|
| **A** Full (5 seeds × 7 × 2) | 70 | 517,130 | ≈ **410 h** (~17 days) |
| **C** 1 seed × 7 × 2 | 14 | 103,426 | ≈ **82 h** (~3.4 days) |
| **B** 1 seed, subsampled large datasets | 14 | ~15,084 | ≈ **12–18 h** (~1 day) |

Bamboogle is on the slow end (multi-hop, ~3 turns avg). NQ / TriviaQA / PopQA
are factoid, often answered in 1 turn, ~1 s/ex — so total runtimes will likely
land near the lower end of the bands above.

---

## Plan B is the recommended starting point

**Bottom-line recommendation: run Plan B first.**

Reasons:

1. **It finishes overnight.** ~12–18 h on this single 4090 vs 3+ days for
   Plan C and 17 days for Plan A.
2. **It produces paper-comparable EM numbers.** A random 1,000-example
   subsample on factoid datasets (NQ / TriviaQA / PopQA) gives EM standard
   error ~1.5 percentage points — small enough to detect a >3-pp deviation
   from the published Search-R1 numbers, which is the bar for "did the fixes
   land."
3. **Bamboogle (125) and MuSiQue (2,417) are kept full.** These are the two
   smallest splits; subsampling would add noise without saving meaningful
   time.
4. **It's the gating decision.** If Plan B reproduces the paper, then Plan A
   only buys tighter error bars on already-correct numbers — useful for the
   write-up, not for validation. If Plan B *doesn't* reproduce, run a
   regression hunt on the smaller plan rather than burning 17 days finding
   the same gap.

Plan A is the right choice once we know the pipeline reproduces and we want
the publication-quality means + variances over 5 seeds. Plan C is a middle
ground if we want full datasets but a single seed (e.g., to avoid
sampling-induced noise on the large datasets while keeping wall-clock
manageable).

---

## Smoke test outcome (already executed)

Bamboogle / base / 1 run after applying all 10 audit fixes:

```
em: 0.088   f1: 0.155   acc: 0.088     (125 examples, 6m 03s)
```

Inspecting `intermediate_data.json`:

* Successful traces follow the expected
  `<think>...</think><search>...</search>\n\n<information>Doc 1(Title: ...)
  ...</information>\n\n<answer>...</answer>` shape — the post-fix format is
  rendering correctly.
* 21/125 traces (17 %) hallucinate the `<information>` block themselves and
  never search; this is a known failure mode of 3B-base on hard multi-hop and
  is consistent with the published behavior.
* The 0.088 EM is in the expected range for 3B-base GRPO on Bamboogle (the
  hardest OOD benchmark; the model was trained on NQ + HotpotQA only).

The smoke test does **not** prove the full sweep will reproduce — but it
confirms the pipeline is wired correctly and the SGLang ↔ retriever loop is
functioning end-to-end.

---

## Scripts

All under `scripts/`. Made executable. Run from anywhere — they resolve their
own repo root.

| Script | Purpose |
|---|---|
| `manage_sglang.sh` | Stop / start / wait / **switch** SGLang for a given variant (`base`\|`instruct`). The `switch` subcommand stops, restarts, and waits for `/get_model_info`. |
| `subsample.sh` | Build `data_subsample/` with deterministic 1,000-row random samples of NQ/TriviaQA/PopQA/HotpotQA/2Wiki and full copies of Bamboogle/MuSiQue. Idempotent — safe to re-run. |
| `run_one.sh` | Runs one `(variant, dataset, seed)` evaluation. Skips if a `metric_score.txt` already exists for the same `save_note` (resume-friendly). |
| `run_variant_sweep.sh` | Iterates `seeds × datasets` for one variant. Refuses to run if SGLang is serving the wrong variant. |
| `sweep_a_full.sh` | **Plan A** — 5 seeds × 7 datasets × 2 variants. Use `nohup`. |
| `sweep_b_reduced.sh` | **Plan B** — 1 seed × 7 datasets × 2 variants on `data_subsample/`. |
| `sweep_c_one_seed.sh` | **Plan C** — 1 seed × 7 datasets × 2 variants on full data. |
| `aggregate.py` | Walks `evaluation_search_r1/results/`, parses `metric_score.txt`, groups by `(dataset, variant, seed)` and writes a markdown report with per-seed scores, means, and grand averages. |

### Save-note convention

Each run writes to:

```
evaluation_search_r1/results/<dataset>/<dataset>_<YYYY>_<MM>_<DD>_<HH>_<MM>_search_r1_<variant>_seed<N>/
```

`aggregate.py` parses the trailing `search_r1_<variant>_(seed|run)<N>` to
group runs. The smoke test's `_run1` suffix is also recognized.

### Resume semantics

`run_one.sh` checks for an existing `metric_score.txt` matching the
`save_note` and skips the eval if found. Plans A and C take many days; if a
run is interrupted or the box reboots, just re-launch — completed
`(variant, dataset, seed)` triples are not redone.

---

## How to run each plan

All three plans assume:

* The retriever is up on `127.0.0.1:3005` (`curl /health` returns "healthy").
* SGLang is launchable on `127.0.0.1:3000` (the scripts will switch models as
  needed; current process gets killed and restarted).
* The eval venv is at `/venv/evaluation_search_r1` (override with `PY=...`).

> ⚠️ Each plan kills and restarts SGLang twice (once per variant). Make sure
> nothing else needs the running process before launching.

### Plan A — full sweep (~17 days)

```bash
nohup scripts/sweep_a_full.sh > /tmp/sweep_a.log 2>&1 &
disown
tail -f /tmp/sweep_a.log
```

Produces `RESULTS_PLAN_A.md` at the end. Resumable.

### Plan B — reduced overnight (~12–18 h)

```bash
nohup scripts/sweep_b_reduced.sh > /tmp/sweep_b.log 2>&1 &
disown
tail -f /tmp/sweep_b.log
```

Produces `RESULTS_PLAN_B.md`. Subsample is deterministic
(`SEED=42`, `N=1000`), so the same script re-run gives identical inputs.

### Plan C — 1 seed × full datasets (~3.4 days)

```bash
nohup scripts/sweep_c_one_seed.sh > /tmp/sweep_c.log 2>&1 &
disown
tail -f /tmp/sweep_c.log
```

Produces `RESULTS_PLAN_C.md`. Resumable.

### Aggregate by hand (any time)

```bash
scripts/aggregate.py --output RESULTS_NOW.md
```

Picks up every completed run under `evaluation_search_r1/results/` regardless
of which plan produced it, including the smoke test.

---

## What to expect from the output

Per-metric tables look like:

```
### EM

| Dataset         | Variant   | seed=1 | seed=2 | … | mean  | std   | n |
|---               |---        |---     |---     |---|---    |---    |---|
| bamboogle       | base      | 0.088  | 0.080  | … | 0.085 | 0.005 | 5 |
| bamboogle       | instruct  | …      | …      | … | …     | …     | 5 |
| nq              | base      | …      | …      | … | …     | …     | 5 |
| …               | …         | …      | …      | … | …     | …     | … |
```

Plus a grand-average summary per variant (mean across all
dataset×seed cells).

---

## Operational risks & mitigations

| Risk | Mitigation |
|---|---|
| Mid-run crash (OOM, SGLang hang, retriever crash) | `run_one.sh` is resume-aware; restart the same `sweep_*.sh` script and it picks up where it left off. |
| SGLang fails to load on switch | `manage_sglang.sh switch` waits up to 10 min for `/get_model_info` and exits non-zero on timeout, so the sweep aborts loudly rather than silently producing zeros. |
| Disk fill (intermediate JSONs ~hundreds of MB per run) | Each run writes only one results dir; clean up old smoke runs in `evaluation_search_r1/results/` before kicking off a sweep. |
| Retriever 1-worker bottleneck | Most of the wall-clock is multi-turn LLM generation, not retrieval; bumping `--num_retriever` would help marginally but isn't on the critical path. Revisit only if Plan A takes longer than projected. |
| Determinism | SGLang sampling at temperature 1.0 — the "seed" in `save_note` is purely a label so different runs aren't conflated. SGLang ignores the FlashRAG `seed` field, so each run is genuinely stochastic. |

---

## Discussion points

1. **Plan B vs Plan A as the canonical run for the write-up.** If Plan B
   numbers fall within ~3 pp of the published Search-R1 EM table, the
   reproduction case is solid and Plan A becomes a "nice-to-have" for tighter
   error bars. If Plan B numbers diverge, we debug rather than scale up.

2. **Multi-machine option for Plan A.** ~17 days on one 4090 is real but
   expensive. Splitting datasets across two boxes (each switching variants)
   roughly halves wall-clock. Worth considering before committing the budget.

3. **Subsample seed.** Plan B's subsample uses `SEED=42`. If we want to
   sanity-check that subsampling didn't get unlucky, re-run `subsample.sh`
   with a different seed and compare; differences should be within sampling
   error.

4. **Instruct-variant `apply_chat=True`.** All three plans correctly toggle
   `apply_chat` based on the variant (per `run_one.sh`). The instruct model's
   chat template is applied automatically; no per-dataset overrides needed.

5. **What's *not* covered.** None of the plans evaluate the 7B variants — the
   3B base + 3B instruct released GRPO checkpoints are the only models loaded
   on disk. If we want 7B numbers, we need a separate download (and likely a
   bigger GPU than this 24 GB 4090).
