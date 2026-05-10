---
title: Results M4 — Qwen3.5-0.8B baseline (untrained, base + hybrid)
tags: [report, eval, m4, qwen3.5]
source: internal
created: 2026-05-08
updated: 2026-05-08
---

# Results M4: Qwen3.5-0.8B Untrained Baseline

**Status (as of 2026-05-08):** smoke runs in flight; full sweep pending. This file is the live record; numbers populate as jobs finish.

## 1. Run roster

| Variant | HF id | Local path | `prompt_mode` | `enable_thinking` |
|---|---|---|---|---|
| `qwen3.5_0.8b` (hybrid) | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | `eval/qwen3.5_0.8b/` | `qwen35` | True |
| `qwen3.5_0.8b_base` | [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | `eval/qwen3.5_0.8b_base/` | `qwen35` | False |

Pipeline: [`evaluation_qwen35/`](../../evaluation_qwen35/), [`scripts/run_m4.sh`](../../scripts/run_m4.sh), [`scripts/sbatch_m4.sh`](../../scripts/sbatch_m4.sh). Code-setup audit: [`CODE_SETUP_m4.md`](CODE_SETUP_m4.md). Milestone narrative: [`../milestone_4/MILESTONE_4.md`](../milestone_4/MILESTONE_4.md).

## 2. Eval configuration (M4)

Same shape as M3 (per CODE_SETUP_m4 §1):

| Knob | Value | Note |
|---|---|---|
| Decoding | `temperature=0.0` (greedy) | Single seed; greedy => seed-invariant |
| `apply_chat` | True | Both base and hybrid render through `tokenizer.apply_chat_template` |
| Action / observation tags | `<tool_call>` / `<tool_response>` | Qwen3.5-native vocab tokens (248058/9, 248066/7) |
| Retriever | IVF-SQ8 × 8 workers | top-5 |
| `generator_max_input_len` | 4096 | matches M3 |
| `max_search_turns` / `step_limit` / `max_obs_length` | 5 / 8192 / 256 | matches M3 |
| Datasets | bamboogle, nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique | 7 |
| Splits | test / test / test / test / dev / dev / dev | per dataset |

## 3. Smoke results (100 random items / dataset, seed=1)

Format: EM / ACC / F1.

### Hybrid (`qwen3.5_0.8b`)

Sbatch job: `<TBD — populated post-run>`. Wall-clock: TBD min on `node<TBD>`.

| Dataset | N | EM | ACC | F1 |
|---|---:|---:|---:|---:|
| bamboogle | 100 | TBD | TBD | TBD |
| nq | 100 | TBD | TBD | TBD |
| triviaqa | 100 | TBD | TBD | TBD |
| popqa | 100 | TBD | TBD | TBD |
| hotpotqa | 100 | TBD | TBD | TBD |
| 2wikimultihopqa | 100 | TBD | TBD | TBD |
| musique | 100 | TBD | TBD | TBD |
| **mean** | 700 | **TBD** | **TBD** | **TBD** |

### Base (`qwen3.5_0.8b_base`)

Sbatch job: `<TBD>`. Wall-clock: TBD min on `node<TBD>`.

| Dataset | N | EM | ACC | F1 |
|---|---:|---:|---:|---:|
| bamboogle | 100 | TBD | TBD | TBD |
| nq | 100 | TBD | TBD | TBD |
| triviaqa | 100 | TBD | TBD | TBD |
| popqa | 100 | TBD | TBD | TBD |
| hotpotqa | 100 | TBD | TBD | TBD |
| 2wikimultihopqa | 100 | TBD | TBD | TBD |
| musique | 100 | TBD | TBD | TBD |
| **mean** | 700 | **TBD** | **TBD** | **TBD** |

## 4. Full sweep (51,713 items / variant)

Pending smoke validation. Submission:

```bash
sbatch scripts/sbatch_m4.sh qwen3.5_0.8b
sbatch scripts/sbatch_m4.sh qwen3.5_0.8b_base
```

Expected wall-clock: ~150 min / variant on 1× A100-80GB (M3 reference).

| Dataset | N | hybrid EM | base EM | hybrid ACC | base ACC | hybrid F1 | base F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bamboogle | 125 | TBD | TBD | TBD | TBD | TBD | TBD |
| nq | 3,610 | TBD | TBD | TBD | TBD | TBD | TBD |
| triviaqa | 11,313 | TBD | TBD | TBD | TBD | TBD | TBD |
| popqa | 14,267 | TBD | TBD | TBD | TBD | TBD | TBD |
| hotpotqa | 7,405 | TBD | TBD | TBD | TBD | TBD | TBD |
| 2wikimultihopqa | 12,576 | TBD | TBD | TBD | TBD | TBD | TBD |
| musique | 2,417 | TBD | TBD | TBD | TBD | TBD | TBD |
| **mean** | 51,713 | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

## 5. Cross-family comparison (M3 vs M4)

Untrained-floor comparison once full sweeps are in. M3 reference is the pre-GRPO Qwen3-0.6B hybrid (`qwen3_0.6b`, [`RESULTS_m3.md`](RESULTS_m3.md) §6).

| Dataset | M3 (Qwen3-0.6B hybrid) | M4 (Qwen3.5-0.8B hybrid) | M4 (Qwen3.5-0.8B base) | Δ M4-hybrid vs M3 |
|---|---:|---:|---:|---:|
| bamboogle | 0.056 | TBD | TBD | TBD |
| nq | 0.064 | TBD | TBD | TBD |
| triviaqa | 0.220 | TBD | TBD | TBD |
| popqa | 0.094 | TBD | TBD | TBD |
| hotpotqa | 0.090 | TBD | TBD | TBD |
| 2wikimultihopqa | 0.130 | TBD | TBD | TBD |
| musique | 0.061 | TBD | TBD | TBD |
| **mean** | **0.102** | **TBD** | **TBD** | **TBD** |

(M3 mean from RESULTS_m3 §5; per-dataset values are illustrative — confirm against `RESULTS_m3.md` §6 EM table when populating.)

## 6. Findings

(To be written once smoke + full sweep land.)

## 7. Pointers

- M4 milestone narrative: [`../milestone_4/MILESTONE_4.md`](../milestone_4/MILESTONE_4.md)
- M4 code-setup deltas vs M3: [`CODE_SETUP_m4.md`](CODE_SETUP_m4.md)
- M3 results (the within-family floor): [`RESULTS_m3.md`](RESULTS_m3.md)
- Active recipe-ablation plan (drives any future M5+ Qwen3.5 GRPO training): [`../TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md)
