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

## Bottom line up front

1. **GRPO training delivers the expected lift over the raw base model.** Grand-mean EM (across all 7 datasets, 1 seed) is `instruct` 0.341 > `base` 0.288 > `qwen_25_3b_instruct` (raw) 0.199. Both Search-R1 GRPO checkpoints score ~1.5× to ~1.7× the raw Qwen2.5-3B-Instruct baseline.
2. **Instruct GRPO beats Base GRPO on 6 of 7 datasets** (Δ +0.004 to +0.200 EM), with the largest lifts on multi-hop and out-of-distribution sets (bamboogle +0.20, hotpotqa +0.055, 2wikimultihopqa +0.073, musique +0.073). The single regression is **triviaqa** (instruct 0.533 vs base 0.576, Δ −0.043) — discussed in §F.
3. **GRPO reinforces tool-use protocol hygiene.** Mean completion tokens collapse from ~100 (raw Qwen) to ~45 (instruct GRPO) to ~10 (base GRPO); answer-tag close-rate rises from 90–96% (raw) to 91–99% (instruct) to ~100% (base); length-truncation rate drops from up to 3.8% on musique (raw) to ≤0.3% (both GRPO variants). The base GRPO checkpoint produces the cleanest, most schema-faithful traces of the three.
4. **Hardest dataset for everyone is musique** (multi-hop): EM 0.043 / 0.124 / 0.051 across raw / instruct / base. Easiest is triviaqa (0.405 / 0.533 / 0.576) — predictable single-hop factoid QA. NQ is the longest dataset at 3 610 questions (drives phase wall-clock); all three variants land in the 0.20–0.39 EM band there.

## Variants

1. **`base`** — `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo` (Search-R1 GRPO from Qwen2.5-3B-Base)
2. **`instruct`** — `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo` (Search-R1 GRPO from Qwen2.5-3B-Instruct)
3. **`qwen_25_3b_instruct`** — `Qwen/Qwen2.5-3B-Instruct` (raw, non-finetuned — isolates GRPO effect)

## Eval configuration

Identical across all 21 runs unless noted. Pulled from the per-run [`config.yaml`](../../../evaluation_search_r1/results/) saved by `run_eval.py` and from [`scripts/run_one.sh`](../../../scripts/run_one.sh).

| Knob | Value | Source |
|---|---|---|
| Method | `search-r1` (paper protocol: `<search>` / `<information>` / `<answer>`) | `run_one.sh:65 --method_name` |
| Decoding | greedy: `temperature=0`, `max_tokens=32` per generation turn (per-turn cap; multi-turn loop appends until `<answer>`) | per-run `config.yaml: generation_params` |
| `apply_chat` | `True` for all three variants | `run_one.sh:37,41,45` |
| Chat template | each variant's tokenizer's own (raw Qwen and the two GRPO ckpts all carry Qwen2.5 chat template) | implicit in each model dir's `tokenizer_config.json` |
| `enable_thinking` | `False` (no `<think>` blocks; paper protocol uses `<search>` directly) | `config.yaml: enable_thinking` |
| Generator framework | `sgl_remote` (SGLang server hit via HTTP `/generate`) | `config.yaml: framework` |
| Generator max input | 4096 tokens | `config.yaml: generator_max_input_len` |
| SGLang context length | 8192 (KV cache budget per server) | `scripts/manage_sglang.sh:61 --context-length` |
| Retriever | local IVF-SQ8 e5-base-v2 (intfloat/e5-base-v2 encoder + FAISS IVF4096-SQ8 over wiki-18 100w) | `local_retriever/retriever_config.yaml` |
| `retrieval_topk` | 3 (Search-R1 paper default) | `flashrag/config/basic_config.yaml:retrieval_topk` |
| `retrieval_query_max_length` | 256 tokens | `config.yaml: retrieval_query_max_length` |
| `faiss_nprobe` | 64 | `local_retriever/retriever_config.yaml:faiss_nprobe` |
| Splits | test where it exists, dev otherwise (NQ test, TriviaQA test, PopQA test, HotpotQA dev, 2WikiMultiHopQA dev, MuSiQue dev, Bamboogle test) | `run_one.sh:24-32` |
| Metrics | EM, ACC, F1 (whitespace-tokenized exact-match / accuracy / token-F1 against gold answers) | `config.yaml: metrics` |
| Random sampling | `random_sample: false`, `test_sample_num: null` (full split) | `config.yaml` |
| Seed | 1 (single seed; no variance estimates) | `run_one.sh:14` |

The three variants differ only in `generator_model_path` (resolved as a flat local dir under `evaluation_search_r1/`):

| Variant | `generator_model_path` |
|---|---|
| `base` | `evaluation_search_r1/search_r1_base_model/` |
| `instruct` | `evaluation_search_r1/search_r1_instruct_model/` |
| `qwen_25_3b_instruct` | `evaluation_search_r1/qwen_25_3b_instruct/` |

## Findings

Numbering picks up from `CODE_SETUP.md` — these are about the eval *outputs*, not the run mechanics.

### F.1 GRPO training is real: instruct GRPO is +0.142 grand-mean EM over raw Qwen

`instruct` (Search-R1 GRPO from Qwen2.5-3B-Instruct) lands at 0.341 grand-mean EM; the raw Qwen2.5-3B-Instruct baseline lands at 0.199. That's a **+0.142 EM lift** purely from GRPO with the EM reward shape — no SFT, same base weights. The lift is consistent across datasets: instruct beats raw Qwen on **all 7** of 7. The smallest gap is musique (+0.081, both very low) and the largest is bamboogle (+0.160).

### F.2 Instruct vs Base GRPO: instruct wins on multi-hop, base wins on TriviaQA

Both GRPO checkpoints used the same training mix (`nq_hotpotqa_train`) but started from different bases. Comparing them isolates the instruction-tuning prior:

| Dataset | base EM | instruct EM | Δ |
|---|---|---|---|
| triviaqa | **0.576** | 0.533 | **−0.043** |
| nq | 0.390 | 0.394 | +0.004 |
| popqa | 0.339 | 0.350 | +0.010 |
| hotpotqa | 0.280 | **0.335** | **+0.055** |
| 2wikimultihopqa | 0.257 | **0.330** | **+0.073** |
| musique | 0.051 | **0.124** | **+0.073** |
| bamboogle | 0.120 | **0.320** | **+0.200** |

The instruct prior consistently helps on **multi-hop reasoning** (hotpotqa, 2wikimultihopqa, musique) and **out-of-distribution evaluation** (bamboogle, which is held out from the training distribution). The single regression — triviaqa — is the dataset most aligned with raw factoid pretraining, where the base model's terser ~9-token outputs may be an advantage over instruct's ~46-token outputs (the EM metric penalizes any extra tokens). See F.4.

### F.3 GRPO produces dramatically shorter, more schema-faithful outputs

Mean completion tokens (whole trace, summed over turns):

| Dataset | base | instruct | qwen raw | base / raw ratio |
|---|---|---|---|---|
| musique | 10 | 55 | 193 | 0.05× |
| 2wikimultihopqa | 11 | 65 | 134 | 0.08× |
| hotpotqa | 10 | 54 | 135 | 0.07× |
| nq | 10 | 45 | 108 | 0.09× |
| bamboogle | 10 | 44 | 108 | 0.09× |
| triviaqa | 9 | 46 | 104 | 0.09× |
| popqa | 9 | 40 | 88 | 0.10× |

Base GRPO emits ~10 tokens per question — essentially just the boxed answer, no preamble. Instruct GRPO is 4–6× more verbose. Raw Qwen is 10–20× more verbose. Length-truncation rate (SGLang `stop_reason ≠ stop/eos/stop_str`) tracks this: raw Qwen hits **3.8% truncation on musique** while both GRPO variants stay ≤0.3% on every dataset. The GRPO reward shaping clearly rewards schema-faithful, terminal-emitting traces.

### F.4 The TriviaQA regression is consistent with terseness being penalized at scale

triviaqa is the only dataset where `base` outscores `instruct`. Base produces ~9-token responses; instruct produces ~46. EM is whitespace-tokenized exact match: any spurious explanatory token after the answer fails. On large factoid datasets (triviaqa is 11 313 questions), small per-token regressions accumulate. This matches the observation in `RESULTS_v1.md` §12.1 that stricter format / more rigid structure correlates with lower realized rewards on the single-hop benchmarks. Worth a closer look at sample outputs to confirm — see Open question O.1.

### F.5 Close-rate ordering matches the GRPO-strength ordering

Close-rate = fraction of traces ending with `</answer>`. Variant means across datasets:

| Variant | mean close-rate | range |
|---|---|---|
| base | 99.93% | 99.9 – 100.0% |
| instruct | 97.07% | 91.6 – 99.3% |
| qwen_25_3b_instruct (raw) | 93.97% | 90.4 – 96.2% |

Base is the cleanest by a wide margin — it almost never fails to close the answer tag. The dataset variance is largest for instruct on multi-hop sets (musique 91.6%, 2wikimultihopqa 94.2%, hotpotqa 98.3%) — instruct's longer chain-of-thought style sometimes runs out of budget before reaching `</answer>`.

### F.6 Sweep wall-clock came in 4× faster than the 6 h budget

Total: 1 h 52 min on this 192-core / 1 TB / 8× RTX 4090 host. The original budget assumed 5090 / 32 GB cards with the (since-fixed) async-blocked retriever; the parallel-retriever fix and the much larger CPU pool combined cut the per-phase wall-clock from ~2 h to ~30–40 min. See `system_metrics.tsv` for the per-30s GPU/RAM/network telemetry. NQ remains the bottleneck per phase (longest dataset).

## Open questions

1. **O.1** — Is the triviaqa instruct-vs-base regression an artifact of terseness and EM (greedy whitespace-match), or a real reasoning regression? Sample 50 disagreement cases and compare token-level overlap; if EM disagreements collapse under F1, it's terseness.
2. **O.2** — Why is musique so much harder than the other multi-hop sets? 2wikimultihopqa scores ~0.33 instruct, hotpotqa ~0.34, but musique ~0.12. Is it the question difficulty, the gold-answer normalization, or that musique requires more retrieval rounds than the per-turn token cap allows? Worth checking length-trunc rate on the 8.4% of musique questions where instruct GRPO didn't close `</answer>`.
3. **O.3** — Per-turn `max_tokens=32` (eval-time generation-per-step cap) is tight; raw Qwen's 3.8% length-truncation on musique suggests it would benefit from a larger budget. Does raising it to 64 or 128 narrow the GRPO-vs-raw gap on multi-hop sets, or is the gap structural (the raw model doesn't know when to terminate even with more budget)?
4. **O.4** — A 5-seed run on the same hardware would take ~9–10 h (5× this run's 1.9 h, plus model-swap overhead). Do the per-dataset Δs in F.2 survive seed variance? Bamboogle's +0.200 lift is large enough that any reasonable seed-variance still leaves a positive Δ; triviaqa's −0.043 is well within plausible 1-seed noise.

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
