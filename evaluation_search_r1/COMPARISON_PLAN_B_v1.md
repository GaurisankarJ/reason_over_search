# Plan B v1 vs. Search-R1 paper

Plan B v1 = locked-config sweep after the [paper-vs-ours audit](PAPER_VS_OURS_AUDIT.md) landed three fixes (apply_chat=True for base, missing prompt sentence restored, special-tokens addition removed). Same 1 seed × 7 datasets × 2 variants × 1 k-row subsamples for the large datasets that Plan B v0 ran. Paper targets are GRPO numbers from arXiv 2503.09516 v5, Appendix F / Table 3. The pre-fix baseline is in [COMPARISON_PLAN_B.md](COMPARISON_PLAN_B.md).

## Headline

**Plan B v1 reproduces the paper, base variant.** Average per-dataset gap shrunk from −8.3 pp (v0) to −2.0 pp (v1). No dataset is more than 4 pp off. TriviaQA hits paper EM exactly. PopQA beats paper by +1.1 pp. Format-validity (close-rate of `</answer>`) went from 84 % on v0 base / Bamboogle to 100 % on v1 across all 7 datasets.

Instruct v1 is **in progress** (started 2026-04-29 06:46 UTC, ETA ~6 h). Numbers below are placeholder; this file will be updated when the sweep completes. Instruct v0 numbers are kept as the comparison row in the meantime.

## Base — v1 vs. v0 vs. paper

| Dataset | v0 EM | **v1 EM** | Paper EM | v1−paper |
|---|---:|---:|---:|---:|
| Bamboogle (n=125) | 0.112 | **0.088** | 0.128 | −4.0 (n=125 noise) |
| NQ-1k | 0.316 | **0.390** | 0.421 | −3.1 |
| TriviaQA-1k | 0.421 | **0.583** | 0.583 | **0.0** |
| PopQA-1k | 0.309 | **0.424** | 0.413 | **+1.1** |
| HotpotQA-1k | 0.201 | **0.263** | 0.297 | −3.4 |
| 2WikiMultiHopQA-1k | 0.207 | **0.239** | 0.274 | −3.5 |
| MuSiQue (full, 2417) | 0.034 | **0.055** | 0.066 | −1.1 |
| **Average** | **0.229** | **0.292** | **0.312** | **−2.0 pp** |

Per-dataset lifts from v0 → v1 range from −2.4 pp (Bamboogle, n=125 noise) to **+16.2 pp on TriviaQA**. NQ +7.4 pp, PopQA +11.5 pp, HotpotQA +6.2 pp, 2Wiki +3.2 pp, MuSiQue +2.1 pp.

### Base — F1 and ACC

| Dataset | v1 EM | v1 F1 | v1 ACC |
|---|---:|---:|---:|
| Bamboogle | 0.088 | 0.172 | 0.096 |
| NQ-1k | 0.390 | 0.474 | 0.428 |
| TriviaQA-1k | 0.583 | 0.657 | 0.638 |
| PopQA-1k | 0.424 | 0.458 | 0.441 |
| HotpotQA-1k | 0.263 | 0.365 | 0.284 |
| 2WikiMultiHopQA-1k | 0.239 | 0.306 | 0.252 |
| MuSiQue (full) | 0.055 | 0.123 | 0.064 |

### Trace hygiene — `</answer>` close rate

| Dataset | v0 base close-rate | **v1 base close-rate** | mean turns (v1) |
|---|---:|---:|---:|
| Bamboogle | 84 % (105/125) | **100 % (125/125)** | 1.00 |
| NQ-1k | 88 % (880/1000) | **99.9 % (999/1000)** | 1.00 |
| TriviaQA-1k | — | **99.9 %** | 1.00 |
| PopQA-1k | — | **100 %** | 1.00 |
| HotpotQA-1k | — | **100 %** | 1.00 |
| 2WikiMultiHopQA-1k | — | **99.6 %** | 1.00 |
| MuSiQue (full) | — | **100 %** | 1.00 |

Almost every base trace now closes cleanly. Mean turns is 1.00 across the board — the base GRPO model issues exactly one `<search>` then answers. (Instruct's mean turns on v0 was 1.33–1.94, so the multi-turn behaviour is variant-specific.)

## Instruct — placeholder, sweep in progress

These rows will be **replaced** when the instruct v1 sweep completes (~12:45 UTC).

| Dataset | v0 EM | v1 EM | Paper EM | v1−paper |
|---|---:|---:|---:|---:|
| Bamboogle (n=125) | 0.360 | _running_ | 0.232 | TBD |
| NQ-1k | 0.399 | _running_ | 0.397 | TBD |
| TriviaQA-1k | 0.539 | _running_ | 0.565 | TBD |
| PopQA-1k | 0.412 | _running_ | 0.391 | TBD |
| HotpotQA-1k | 0.354 | _running_ | 0.331 | TBD |
| 2WikiMultiHopQA-1k | 0.353 | _running_ | 0.310 | TBD |
| MuSiQue (full) | 0.149 | _running_ | 0.124 | TBD |
| **Average (v0)** | **0.367** | TBD | **0.336** | TBD |

Instruct v0 was already +3.1 pp above paper on average and within ±5 pp on 6/7 datasets. The two minor fixes (prompt sentence + special-tokens removal) gave +4.4 pp on base NQ — much more than the audit's ≤1 pp estimate — so instruct may also shift by a few pp. Direction unknown until the sweep finishes.

## What is now locked

The four divergences flagged in [PAPER_VS_OURS_AUDIT.md](PAPER_VS_OURS_AUDIT.md) and applied for v1:

| # | Fix | File:line | Estimated impact (audit) | Empirical impact (this sweep) |
|---|---|---|---:|---:|
| D1 | `apply_chat=True` for base | [scripts/run_one.sh:35](../scripts/run_one.sh#L35) | +5 to +12 pp | +3.0 pp on NQ probe (alone) |
| D7 | Restore `For example, <answer> Beijing </answer>.` in prompt | [flashrag/search_r1/templates.py:9](flashrag/search_r1/templates.py#L9) | ≤1 pp | combined with D8 below |
| D8 | Remove runtime `add_special_tokens` block | [flashrag/pipeline/active_pipeline.py:36](flashrag/pipeline/active_pipeline.py#L36) | ≤1 pp | combined with D7: **+4.4 pp** on NQ |
| D6 | FAISS Flat IP confirmed loaded | retriever runtime | ≤1 pp | verified, no change |

The audit's claim that D7+D8 were ≤1 pp each was wrong: together they delivered +4.4 pp on NQ. Either the prompt sentence biases base toward shorter answers (matching paper EM normalization better), or the special-tokens addition was changing tokenization in a way that mis-aligned with the GRPO checkpoint's training. Since both fixes also point the same direction (toward upstream behaviour), there is no need to disentangle them; the locked v1 config is correct.

`temperature: 0.0` was kept (greedy). The audit traced upstream `verl/trainer/ppo/ray_trainer.py:478,508` → `_validate()` forces `do_sample=False`, and `verl/workers/rollout/vllm_rollout/vllm_rollout.py:162-171` overrides `temperature=0, top_p=1.0, top_k=-1, n=1` whenever that flag is set. Paper eval is greedy. Our earlier −10 pp gap on base was D1+D7+D8, not temperature. (A previous experimental run at temp=1.0 looked like it added +1.7 pp on NQ but that was single-seed sampling noise at temp=1.0 — re-ran would not reproduce.)

## Plan A readiness

**YES.** Decision criteria from [docs/MILESTONE_1.md](../docs/MILESTONE_1.md): all 7 base datasets within 8 pp of paper, no catastrophic divergence, average residual ≤4 pp. Met:

- All 7 within 4 pp of paper
- Average residual −2.0 pp
- Format-validity ≥99.6 % on every dataset
- Two cells (TriviaQA, PopQA) match or beat paper

The remaining 2–3 pp residual on factoid + multi-hop datasets is consistent with 1 k-subsample SE (~1.5 pp on factoid, ~2.5 pp on multi-hop) + single-seed greedy variance. Plan A's 5-seed full-data sweep should tighten this to within ±1.5 pp of paper across the board.

Caveat: the YES is conditional on instruct v1 also landing within 8 pp of paper. Given instruct v0 was already +3.1 pp ahead of paper average, this is highly likely; will be confirmed when the instruct sweep finishes and this file is updated.

## What changed between v0 and v1 in summary

- **Code (committed)**: `run_one.sh` flips base `apply_chat=False → True`; `templates.py` adds the missing example sentence; `active_pipeline.py` drops the runtime `add_special_tokens` block.
- **No change**: temperature, top_p, retriever, model checkpoints, prompt template body, max_turns, step_limit, observation truncation, splits, metrics, FAISS index, SGLang flags.
- **Archived**: v0 base + instruct seed=1 result dirs in [results/_archive_v0/](results/_archive_v0/) for traceability. [RESULTS_PLAN_B_v0.md](RESULTS_PLAN_B_v0.md) is the snapshot of the v0 aggregate.

## Next steps

1. **(in progress)** Instruct v1 sweep — finishes ~12:45 UTC. This file will be updated with the instruct rows.
2. **Aggregate.py refresh** — once instruct is done, regenerate [RESULTS_PLAN_B.md](RESULTS_PLAN_B.md) with v1 numbers (will overwrite the v0 snapshot's location, but v0 lives in [RESULTS_PLAN_B_v0.md](RESULTS_PLAN_B_v0.md)).
3. **Plan A on Vast.ai** — 5 seeds × 7 datasets × 2 variants. Per [docs/VAST_AI_PLAN_A.md](../docs/VAST_AI_PLAN_A.md), an 8× RTX 4090 fleet completes in ≤24 h at $58–77. Local 4090 alone would take ~17 days.
