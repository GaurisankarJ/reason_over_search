---
title: RESULTS — Plan A (1 seed) on 8×4090
tags: [eval, plan-a, results, 8gpu]
source: internal
created: 2026-05-07
updated: 2026-05-07
---

# Results — Plan A 1-seed sweep on 8×4090 (3 variants × 7 datasets)

**Date**: 2026-05-07
**Run wall-clock**: 1 h 52 min 44 s (16:13:21 → 18:03:11 UTC)
**Hardware**: 8× RTX 4090 (single host, 1 TB RAM, 192 cores)
**Index**: IVF-SQ8 (`local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index`)
**Retriever topology**: 8 paired processes on ports 3013..3020, `OMP_NUM_THREADS=8` per process, e5 encoders pinned per-GPU via `CUDA_VISIBLE_DEVICES`. The 3013..3020 range (vs the original design's 3005..3012) and the per-GPU encoder pinning were both applied during this run — see [`CODE_SETUP.md`](CODE_SETUP.md) §3a for the port-collision and encoder-stacking incidents.
**Seed**: 1
**Source script**: [`scripts/sweep_8gpu_one_seed.sh`](../../../scripts/sweep_8gpu_one_seed.sh)
**Aggregator**: [`scripts/aggregate.py`](../../../scripts/aggregate.py) writes this file directly.

> **Tables below are live.** `aggregate.py` ran at sweep end and produced 21 runs aggregated, 21 with trace health. It overwrites everything *under* the sentinel comment; the frontmatter, intro, and Variants list survive the rewrite.

## Variants

1. **`base`** — `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo` (Search-R1 GRPO from Qwen2.5-3B-Base)
2. **`instruct`** — `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo` (Search-R1 GRPO from Qwen2.5-3B-Instruct)
3. **`qwen_25_3b_instruct`** — `Qwen/Qwen2.5-3B-Instruct` (raw, non-finetuned — isolates GRPO effect)

<!-- aggregator: managed region below -->

# Search-R1 evaluation results

_Source: `/workspace/reason_over_search/evaluation_search_r1/results` (21 runs)_

## Per-seed scores

### EM

| Dataset | Variant | seed=1 | mean | std | n |
|---|---|---|---|---|---|
| bamboogle | base | 0.120 | 0.120 | — | 1 |
| bamboogle | instruct | 0.320 | 0.320 | — | 1 |
| bamboogle | qwen_25_3b_instruct | 0.160 | 0.160 | — | 1 |
| nq | base | 0.390 | 0.390 | — | 1 |
| nq | instruct | 0.394 | 0.394 | — | 1 |
| nq | qwen_25_3b_instruct | 0.197 | 0.197 | — | 1 |
| triviaqa | base | 0.576 | 0.576 | — | 1 |
| triviaqa | instruct | 0.533 | 0.533 | — | 1 |
| triviaqa | qwen_25_3b_instruct | 0.405 | 0.405 | — | 1 |
| popqa | base | 0.339 | 0.339 | — | 1 |
| popqa | instruct | 0.350 | 0.350 | — | 1 |
| popqa | qwen_25_3b_instruct | 0.235 | 0.235 | — | 1 |
| musique | base | 0.051 | 0.051 | — | 1 |
| musique | instruct | 0.124 | 0.124 | — | 1 |
| musique | qwen_25_3b_instruct | 0.043 | 0.043 | — | 1 |
| 2wikimultihopqa | base | 0.257 | 0.257 | — | 1 |
| 2wikimultihopqa | instruct | 0.330 | 0.330 | — | 1 |
| 2wikimultihopqa | qwen_25_3b_instruct | 0.177 | 0.177 | — | 1 |
| hotpotqa | base | 0.280 | 0.280 | — | 1 |
| hotpotqa | instruct | 0.335 | 0.335 | — | 1 |
| hotpotqa | qwen_25_3b_instruct | 0.178 | 0.178 | — | 1 |

### F1

| Dataset | Variant | seed=1 | mean | std | n |
|---|---|---|---|---|---|
| bamboogle | base | 0.207 | 0.207 | — | 1 |
| bamboogle | instruct | 0.402 | 0.402 | — | 1 |
| bamboogle | qwen_25_3b_instruct | 0.238 | 0.238 | — | 1 |
| nq | base | 0.468 | 0.468 | — | 1 |
| nq | instruct | 0.473 | 0.473 | — | 1 |
| nq | qwen_25_3b_instruct | 0.285 | 0.285 | — | 1 |
| triviaqa | base | 0.644 | 0.644 | — | 1 |
| triviaqa | instruct | 0.611 | 0.611 | — | 1 |
| triviaqa | qwen_25_3b_instruct | 0.487 | 0.487 | — | 1 |
| popqa | base | 0.376 | 0.376 | — | 1 |
| popqa | instruct | 0.398 | 0.398 | — | 1 |
| popqa | qwen_25_3b_instruct | 0.288 | 0.288 | — | 1 |
| musique | base | 0.121 | 0.121 | — | 1 |
| musique | instruct | 0.199 | 0.199 | — | 1 |
| musique | qwen_25_3b_instruct | 0.095 | 0.095 | — | 1 |
| 2wikimultihopqa | base | 0.314 | 0.314 | — | 1 |
| 2wikimultihopqa | instruct | 0.397 | 0.397 | — | 1 |
| 2wikimultihopqa | qwen_25_3b_instruct | 0.236 | 0.236 | — | 1 |
| hotpotqa | base | 0.369 | 0.369 | — | 1 |
| hotpotqa | instruct | 0.440 | 0.440 | — | 1 |
| hotpotqa | qwen_25_3b_instruct | 0.261 | 0.261 | — | 1 |

### ACC

| Dataset | Variant | seed=1 | mean | std | n |
|---|---|---|---|---|---|
| bamboogle | base | 0.120 | 0.120 | — | 1 |
| bamboogle | instruct | 0.328 | 0.328 | — | 1 |
| bamboogle | qwen_25_3b_instruct | 0.192 | 0.192 | — | 1 |
| nq | base | 0.419 | 0.419 | — | 1 |
| nq | instruct | 0.431 | 0.431 | — | 1 |
| nq | qwen_25_3b_instruct | 0.386 | 0.386 | — | 1 |
| triviaqa | base | 0.618 | 0.618 | — | 1 |
| triviaqa | instruct | 0.592 | 0.592 | — | 1 |
| triviaqa | qwen_25_3b_instruct | 0.522 | 0.522 | — | 1 |
| popqa | base | 0.350 | 0.350 | — | 1 |
| popqa | instruct | 0.377 | 0.377 | — | 1 |
| popqa | qwen_25_3b_instruct | 0.328 | 0.328 | — | 1 |
| musique | base | 0.059 | 0.059 | — | 1 |
| musique | instruct | 0.147 | 0.147 | — | 1 |
| musique | qwen_25_3b_instruct | 0.073 | 0.073 | — | 1 |
| 2wikimultihopqa | base | 0.271 | 0.271 | — | 1 |
| 2wikimultihopqa | instruct | 0.368 | 0.368 | — | 1 |
| 2wikimultihopqa | qwen_25_3b_instruct | 0.268 | 0.268 | — | 1 |
| hotpotqa | base | 0.298 | 0.298 | — | 1 |
| hotpotqa | instruct | 0.367 | 0.367 | — | 1 |
| hotpotqa | qwen_25_3b_instruct | 0.246 | 0.246 | — | 1 |

## Grand averages

### Grand average EM across all runs

| Variant | mean | n_runs |
|---|---|---|
| base | 0.288 | 7 |
| instruct | 0.341 | 7 |
| qwen_25_3b_instruct | 0.199 | 7 |

### Grand average F1 across all runs

| Variant | mean | n_runs |
|---|---|---|
| base | 0.357 | 7 |
| instruct | 0.417 | 7 |
| qwen_25_3b_instruct | 0.270 | 7 |

### Grand average ACC across all runs

| Variant | mean | n_runs |
|---|---|---|
| base | 0.305 | 7 |
| instruct | 0.373 | 7 |
| qwen_25_3b_instruct | 0.288 | 7 |

## Trace health

Close-rate = fraction of examples whose `final_response` contains `</answer>`. 
Length-truncated = fraction whose SGLang `stop_reason` was anything other than `stop`/`eos`/`stop_str` (typically the per-step token cap firing).

### Close-rate

| Dataset | Variant | seed=1 | mean | n_examples |
|---|---|---|---|---|
| bamboogle | base | 100.0% | 100.0% | 125 |
| bamboogle | instruct | 99.2% | 99.2% | 125 |
| bamboogle | qwen_25_3b_instruct | 95.2% | 95.2% | 125 |
| nq | base | 99.9% | 99.9% | 3610 |
| nq | instruct | 99.3% | 99.3% | 3610 |
| nq | qwen_25_3b_instruct | 93.7% | 93.7% | 3610 |
| triviaqa | base | 99.9% | 99.9% | 11313 |
| triviaqa | instruct | 99.1% | 99.1% | 11313 |
| triviaqa | qwen_25_3b_instruct | 96.2% | 96.2% | 11313 |
| popqa | base | 100.0% | 100.0% | 14267 |
| popqa | instruct | 97.8% | 97.8% | 14267 |
| popqa | qwen_25_3b_instruct | 94.8% | 94.8% | 14267 |
| musique | base | 99.9% | 99.9% | 2417 |
| musique | instruct | 91.6% | 91.6% | 2417 |
| musique | qwen_25_3b_instruct | 90.4% | 90.4% | 2417 |
| 2wikimultihopqa | base | 99.9% | 99.9% | 12576 |
| 2wikimultihopqa | instruct | 94.2% | 94.2% | 12576 |
| 2wikimultihopqa | qwen_25_3b_instruct | 92.7% | 92.7% | 12576 |
| hotpotqa | base | 99.9% | 99.9% | 7405 |
| hotpotqa | instruct | 98.3% | 98.3% | 7405 |
| hotpotqa | qwen_25_3b_instruct | 94.8% | 94.8% | 7405 |

### Length-truncation rate

| Dataset | Variant | seed=1 | mean | n_examples |
|---|---|---|---|---|
| bamboogle | base | 0.0% | 0.0% | 125 |
| bamboogle | instruct | 0.0% | 0.0% | 125 |
| bamboogle | qwen_25_3b_instruct | 1.6% | 1.6% | 125 |
| nq | base | 0.1% | 0.1% | 3610 |
| nq | instruct | 0.0% | 0.0% | 3610 |
| nq | qwen_25_3b_instruct | 0.8% | 0.8% | 3610 |
| triviaqa | base | 0.0% | 0.0% | 11313 |
| triviaqa | instruct | 0.1% | 0.1% | 11313 |
| triviaqa | qwen_25_3b_instruct | 0.8% | 0.8% | 11313 |
| popqa | base | 0.0% | 0.0% | 14267 |
| popqa | instruct | 0.1% | 0.1% | 14267 |
| popqa | qwen_25_3b_instruct | 0.1% | 0.1% | 14267 |
| musique | base | 0.1% | 0.1% | 2417 |
| musique | instruct | 0.3% | 0.3% | 2417 |
| musique | qwen_25_3b_instruct | 3.8% | 3.8% | 2417 |
| 2wikimultihopqa | base | 0.1% | 0.1% | 12576 |
| 2wikimultihopqa | instruct | 0.1% | 0.1% | 12576 |
| 2wikimultihopqa | qwen_25_3b_instruct | 0.7% | 0.7% | 12576 |
| hotpotqa | base | 0.1% | 0.1% | 7405 |
| hotpotqa | instruct | 0.1% | 0.1% | 7405 |
| hotpotqa | qwen_25_3b_instruct | 1.0% | 1.0% | 7405 |

### Mean completion tokens (whole trace, summed over turns)

| Dataset | Variant | seed=1 | mean | n_examples |
|---|---|---|---|---|
| bamboogle | base | 10 | 10 | 125 |
| bamboogle | instruct | 44 | 44 | 125 |
| bamboogle | qwen_25_3b_instruct | 108 | 108 | 125 |
| nq | base | 10 | 10 | 3610 |
| nq | instruct | 45 | 45 | 3610 |
| nq | qwen_25_3b_instruct | 108 | 108 | 3610 |
| triviaqa | base | 9 | 9 | 11313 |
| triviaqa | instruct | 46 | 46 | 11313 |
| triviaqa | qwen_25_3b_instruct | 104 | 104 | 11313 |
| popqa | base | 9 | 9 | 14267 |
| popqa | instruct | 40 | 40 | 14267 |
| popqa | qwen_25_3b_instruct | 88 | 88 | 14267 |
| musique | base | 10 | 10 | 2417 |
| musique | instruct | 55 | 55 | 2417 |
| musique | qwen_25_3b_instruct | 193 | 193 | 2417 |
| 2wikimultihopqa | base | 11 | 11 | 12576 |
| 2wikimultihopqa | instruct | 65 | 65 | 12576 |
| 2wikimultihopqa | qwen_25_3b_instruct | 134 | 134 | 12576 |
| hotpotqa | base | 10 | 10 | 7405 |
| hotpotqa | instruct | 54 | 54 | 7405 |
| hotpotqa | qwen_25_3b_instruct | 135 | 135 | 7405 |
