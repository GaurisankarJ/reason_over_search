---
title: RESULTS — Plan A (1 seed) on 8×4090
tags: [eval, plan-a, results, 8gpu]
source: internal
created: 2026-05-07
updated: 2026-05-07
---

# Results — Plan A 1-seed sweep on 8×4090 (3 variants × 7 datasets)

**Date**: TBD (filled at sweep end)
**Run wall-clock**: TBD
**Hardware**: 8× RTX 4090 (single host, ≥200 GB RAM, ≥64 cores)
**Index**: IVF-SQ8 (`local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index`)
**Retriever topology**: 8 paired processes on ports 3005..3012, `OMP_NUM_THREADS=8` per process
**Seed**: 1
**Source script**: [`scripts/sweep_8gpu_one_seed.sh`](../../../scripts/sweep_8gpu_one_seed.sh)
**Aggregator**: [`scripts/aggregate.py`](../../../scripts/aggregate.py) writes this file directly.

> **Status:** rows below are empty until the sweep finishes. `aggregate.py` overwrites this file at the end of [`sweep_8gpu_one_seed.sh`](../../../scripts/sweep_8gpu_one_seed.sh). The header sections (everything above the table) survive the rewrite if `aggregate.py` is configured to preserve frontmatter — otherwise re-paste them.

## Variants

1. **`base`** — `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo` (Search-R1 GRPO from Qwen2.5-3B-Base)
2. **`instruct`** — `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo` (Search-R1 GRPO from Qwen2.5-3B-Instruct)
3. **`qwen_25_3b_instruct`** — `Qwen/Qwen2.5-3B-Instruct` (raw, non-finetuned — isolates GRPO effect)

## Per-(variant, dataset) results

| variant | dataset | n | EM | F1 | acc | wall-clock | paper EM | Δ pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | nq | 3610 | | | | | | |
| base | triviaqa | 11313 | | | | | 0.583 | |
| base | popqa | 14267 | | | | | | |
| base | hotpotqa | 7405 | | | | | | |
| base | 2wikimultihopqa | 12576 | | | | | 0.274 | |
| base | musique | 2417 | | | | | | |
| base | bamboogle | 125 | | | | | | |
| instruct | nq | 3610 | | | | | | |
| instruct | triviaqa | 11313 | | | | | | |
| instruct | popqa | 14267 | | | | | | |
| instruct | hotpotqa | 7405 | | | | | | |
| instruct | 2wikimultihopqa | 12576 | | | | | | |
| instruct | musique | 2417 | | | | | | |
| instruct | bamboogle | 125 | | | | | | |
| qwen_25_3b_instruct | nq | 3610 | | | | | n/a | n/a |
| qwen_25_3b_instruct | triviaqa | 11313 | | | | | n/a | n/a |
| qwen_25_3b_instruct | popqa | 14267 | | | | | n/a | n/a |
| qwen_25_3b_instruct | hotpotqa | 7405 | | | | | n/a | n/a |
| qwen_25_3b_instruct | 2wikimultihopqa | 12576 | | | | | n/a | n/a |
| qwen_25_3b_instruct | musique | 2417 | | | | | n/a | n/a |
| qwen_25_3b_instruct | bamboogle | 125 | | | | | n/a | n/a |

Paper EM column is from Search-R1 Table 2 / 3. The raw `qwen_25_3b_instruct` rows have no paper baseline — that's the new datapoint this run produces.

## Headline numbers (filled at end)

| | base | instruct | qwen_25_3b_instruct (raw) | GRPO delta (instruct − raw) |
|---|---:|---:|---:|---:|
| Macro-avg EM | TBD | TBD | TBD | TBD pp |
| Macro-avg F1 | TBD | TBD | TBD | TBD pp |

The "GRPO delta" row is the headline finding of this run: how much absolute EM does Search-R1's GRPO training add on top of the raw instruct base it started from?

## Health metrics (operator notes)

- **GPU util average across all 8 GPUs**: TBD (from `nvidia-smi dmon -s u -c 720` second terminal during sweep; predecessor 4-shard run was 41% — paired-retriever target ≥75%)
- **Retriever fleet CPU sum**: TBD cores sustained (target ≈ 64 = 8 × `OMP_NUM_THREADS=8`)
- **Retriever fleet RSS sum**: TBD GB (target ≈ 136 = 8 × ~17)
- **Free RAM after fleet boot**: TBD GB (target ≥ 50)
- **Failed runs**: TBD (any (variant, dataset) without a `metric_score.txt`)

## Provenance

Run launched with:

```bash
nohup scripts/sweep_8gpu_one_seed.sh > /tmp/sweep_8gpu.log 2>&1 &
```

Raw outputs (per-rollout JSONL, metric files): `evaluation_search_r1/results/<dataset>/<dataset>_*_search_r1_<variant>_seed1/`.

## See also

- [`BOOTSTRAP.md`](BOOTSTRAP.md) — runbook for this sweep
- [`CODE_SETUP.md`](CODE_SETUP.md) — what changed in scripts vs prior
- [`SESSION_LOG.md`](SESSION_LOG.md) — running journal
- [`docs/PLAN_A_5090x4.md`](../../PLAN_A_5090x4.md) — predecessor 4-shard run for comparison
- [`docs/milestone_one/RESULTS_PLAN_B.md`](../../milestone_one/RESULTS_PLAN_B.md) — earlier 1-seed Plan B (subsampled) baseline
