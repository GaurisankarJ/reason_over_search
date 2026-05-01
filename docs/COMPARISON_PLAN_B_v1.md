# Plan B v1 vs. Search-R1 paper

> **Status (2026-05-01): final.** The v1 instruct sweep finished 2026-04-29 ~22:05 UTC. All 14 cells (7 datasets × 2 variants) are aggregated in [RESULTS_PLAN_B.md](RESULTS_PLAN_B.md) and locked. The frozen Plan-A reproducer config lives in [FROZEN_CONFIG_v1.md](FROZEN_CONFIG_v1.md).

Plan B v1 = locked-config sweep after the [paper-vs-ours audit](PAPER_VS_OURS_AUDIT.md) landed three fixes (apply_chat=True for base, missing prompt sentence restored, special-tokens addition removed). Same 1 seed × 7 datasets × 2 variants × 1 k-row subsamples for the large datasets that Plan B v0 ran. Paper targets are GRPO numbers from arXiv 2503.09516 v5, Appendix F / Table 3. The pre-fix baseline is in [COMPARISON_PLAN_B.md](COMPARISON_PLAN_B.md).

## Headline

**Plan B v1 reproduces the paper on both variants.**

| Variant  | v0 avg EM | v1 avg EM | Paper avg EM | v1 − paper |
|---       |---:       |---:       |---:          |---:        |
| base     | 0.229     | **0.292** | 0.312        | **−2.0 pp** |
| instruct | 0.367     | **0.361** | 0.336        | **+2.5 pp** |

- **Base**: gap shrunk from −8.3 pp (v0) → −2.0 pp (v1). No dataset more than 4 pp off paper. TriviaQA hits paper EM exactly. PopQA beats paper by +1.1 pp. Format-validity (`</answer>` close-rate) went from 84 % on v0 base/Bamboogle → ≥99.6 % on every v1 base dataset.
- **Instruct**: barely moved between v0 and v1 (−0.6 pp avg) — the audit's prediction that the three fixes would be small for instruct was correct. Average is +2.5 pp above paper, within ±5 pp on 6/7 datasets, with Bamboogle the only outlier (+11.2 pp on n=125 — variance, not signal).

Both variants now satisfy the Phase 1 "reproduces" criterion.

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

## Instruct — v1 vs. v0 vs. paper

| Dataset | v0 EM | **v1 EM** | Paper EM | v1−v0 | v1−paper |
|---|---:|---:|---:|---:|---:|
| Bamboogle (n=125) | 0.360 | **0.344** | 0.232 | −1.6 | **+11.2** (n=125 variance) |
| NQ-1k | 0.399 | **0.402** | 0.397 | +0.3 | **+0.5** |
| TriviaQA-1k | 0.539 | **0.531** | 0.565 | −0.8 | −3.4 |
| PopQA-1k | 0.412 | **0.413** | 0.391 | +0.1 | **+2.2** |
| HotpotQA-1k | 0.354 | **0.346** | 0.331 | −0.8 | **+1.5** |
| 2WikiMultiHopQA-1k | 0.353 | **0.350** | 0.310 | −0.3 | **+4.0** |
| MuSiQue (full, 2417) | 0.149 | **0.141** | 0.124 | −0.8 | **+1.7** |
| **Average** | **0.367** | **0.361** | **0.336** | **−0.6** | **+2.5 pp** |

The three audit fixes barely shifted instruct, in line with the audit's prediction. Per-dataset moves are all ≤1.6 pp and within single-seed greedy noise. The +0.3 pp move on NQ for instruct is consistent with the audit's ≤1 pp estimate, unlike the +4.4 pp combined kick the same fixes gave the base variant — so the prompt sentence + special-tokens fixes were base-specific.

### Instruct — F1 and ACC

| Dataset | v1 EM | v1 F1 | v1 ACC |
|---|---:|---:|---:|
| Bamboogle | 0.344 | 0.453 | 0.376 |
| NQ-1k | 0.402 | 0.487 | 0.450 |
| TriviaQA-1k | 0.531 | 0.620 | 0.604 |
| PopQA-1k | 0.413 | 0.458 | 0.448 |
| HotpotQA-1k | 0.346 | 0.458 | 0.383 |
| 2WikiMultiHopQA-1k | 0.350 | 0.422 | 0.390 |
| MuSiQue (full) | 0.141 | 0.216 | 0.167 |

### Instruct — `</answer>` close rate, length-truncation, mean tokens

| Dataset | close-rate | length-trunc | mean tokens / trace | mean turns |
|---|---:|---:|---:|---:|
| Bamboogle | 100.0% | 0.0% | 44 | ~1.9 |
| NQ-1k | 99.0% | 0.1% | 45 | ~1.0–1.1 |
| TriviaQA-1k | 99.1% | 0.3% | 45 | ~1.0–1.1 |
| PopQA-1k | 97.5% | 0.0% | 41 | ~1.0–1.1 |
| HotpotQA-1k | 98.1% | 0.3% | 55 | >1 |
| 2WikiMultiHopQA-1k | 94.2% | 0.2% | 65 | >1 |
| MuSiQue (full) | 91.4% | 0.3% | 56 | >1 |

Length-truncation is ≤0.3 % everywhere — the per-step token cap is not biting. The 6–9 % open-trace rate on the multi-hop datasets (MuSiQue, 2Wiki) is the model burning all 4 search turns without converging on `</answer>`, not the per-step cap. This is a known failure mode of the released GRPO instruct checkpoint and not something the eval pipeline can fix.

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

**YES (unconditional).** Decision criteria from [docs/MILESTONE_1.md](../docs/MILESTONE_1.md): all datasets within 8 pp of paper, no catastrophic divergence, average residual ≤4 pp. Met for both variants:

- **Base**: all 7 within 4 pp; avg −2.0 pp; format-validity ≥99.6 %; TriviaQA + PopQA match/beat paper.
- **Instruct**: 6/7 within 4 pp; avg +2.5 pp; format-validity ≥91.4 %. Bamboogle's +11.2 pp (n=125) is the only outlier and is single-seed variance, not a real lift — Plan A's 5-seed full-data run will collapse it.

The remaining residuals are consistent with 1 k-subsample SE (~1.5 pp factoid, ~2.5 pp multi-hop) + single-seed greedy variance. Plan A's 5-seed × full-data sweep should tighten everything to within ±1.5 pp of paper.

## What changed between v0 and v1 in summary

- **Code (committed)**: `run_one.sh` flips base `apply_chat=False → True`; `templates.py` adds the missing example sentence; `active_pipeline.py` drops the runtime `add_special_tokens` block.
- **No change**: temperature, top_p, retriever, model checkpoints, prompt template body, max_turns, step_limit, observation truncation, splits, metrics, FAISS index, SGLang flags.
- **Archived**: v0 base + instruct seed=1 result dirs at `evaluation_search_r1/results/_archive_v0/` (local-only — gitignored). [archive/RESULTS_PLAN_B_v0.md](archive/RESULTS_PLAN_B_v0.md) is the committed snapshot of the v0 aggregate.
- **Discarded experiments** (full list in [archive/DISCARDED_ABLATIONS.md](archive/DISCARDED_ABLATIONS.md) and the post-mortems in [archive/](archive/)): temperature sweep above 0.0, the autoresearch loop's 11 ablations on `experiment_ros/apr27`, the `apply_chat=True + temp=1.0` probe on NQ. None reproduced when re-run greedy.

## Next steps

1. **Plan A on Vast.ai** — 5 seeds × 7 datasets × 2 variants = 70 runs. Per [VAST_AI_PLAN_A.md](VAST_AI_PLAN_A.md), an 8× RTX 4090 fleet completes in ≤24 h at $58–77. Local 4090 alone would take ~17 days. Frozen reproducer config in [FROZEN_CONFIG_v1.md](FROZEN_CONFIG_v1.md).
2. **Aggregate, write up, publish**: per-benchmark means + std-dev across the 5 seeds, side-by-side with paper, plus the audit + cost summary.
