# CLAUDE.md — Reason Over Search

You are a helpful research assistant with years of experience. Your job is to help me reproduce, extend, and ablate retrieval-augmented reasoning models. This file is your persistent memory: read it first whenever you restart so you can pick up where we left off.

More instructions will be added here as the research progresses. Treat this file as a living source of truth — update it (or ask me to update it) whenever a load-bearing fact changes.

## What this project is

We are reproducing and extending two papers on RL-trained retrieval-augmented LLMs:

1. **ReSearch** — https://www.alphaxiv.org/abs/2503.19470
2. **Search-R1** — https://www.alphaxiv.org/abs/2503.09516 — *current focus, Phase 1*

The Search-R1 setup: a Qwen2.5-3B policy is trained with GRPO to interleave `<search>…</search>` calls with reasoning, retrieve documents from a frozen Wikipedia FAISS index, and emit a final `<answer>…</answer>`. We are *not* training; we are *evaluating* the published GRPO checkpoints end-to-end on QA benchmarks.

The official Search-R1 repo is `https://github.com/PeterGriffinJin/Search-R1`. They do not ship an evaluation pipeline, so the one in this repo is adapted from FlashRAG/ReSearch and validated against the paper numbers.

## Phase 1 goal (from [docs/MILESTONE_1.md](../docs/MILESTONE_1.md))

Reproduce the Search-R1 3B baseline for both checkpoints (`base` and `instruct`) on 7 benchmarks: Bamboogle, 2WikiMultiHopQA, TriviaQA, PopQA, MuSiQue, NQ, HotpotQA. Run each 5×, report averages. Constraints: reproducible on Vast.ai + in-house GPUs, minimize cost/wall-clock.

Paper targets we compare against (Qwen2.5-3B EM, Search-R1 v5 Table 3):

|              | NQ    | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | Avg   |
|---           |---    |---       |---    |---       |---    |---      |---        |---    |
| GRPO base    | 0.421 | 0.583    | 0.413 | 0.297    | 0.274 | 0.066   | 0.128     | 0.312 |
| GRPO instruct| 0.397 | 0.565    | 0.391 | 0.331    | 0.310 | 0.124   | 0.232     | 0.336 |

## Repo layout (what lives where)

- [`README.md`](../README.md) — Phase 1 milestone framing.
- [`evaluation_search_r1/`](../evaluation_search_r1/) — eval pipeline (FlashRAG fork + Search-R1 prompts/parsers).
  - [`README.md`](../evaluation_search_r1/README.md) — how to run eval per dataset.
  - [`FROZEN_CONFIG_v1.md`](../docs/FROZEN_CONFIG_v1.md) — **single source of truth for the locked Plan B v1 setup**. Read first.
  - [`REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) — paper targets, the 10 divergences fixed, smoke-validation results, model sha256 verification.
  - [`EVAL_OPS.md`](../docs/EVAL_OPS.md) — three sweep plans (A full / B reduced / C one-seed), wall-clock budgets, where time goes.
  - [`RESULTS_PLAN_B.md`](../docs/RESULTS_PLAN_B.md) — auto-generated final results.
  - [`COMPARISON_PLAN_B_v1.md`](../docs/COMPARISON_PLAN_B_v1.md) — Plan B v1 vs paper, both variants. v0 baseline is at [`COMPARISON_PLAN_B.md`](../docs/COMPARISON_PLAN_B.md).
  - [`archive/`](../docs/archive/) — discarded experiments + historical snapshots ([`archive/README.md`](../docs/archive/README.md) is the index).
  - `flashrag/pipeline/active_pipeline.py` — search↔generate loop.
  - `flashrag/search_r1/parser.py` — `<search>`/`<answer>` parsing, observation truncation.
  - `flashrag/search_r1/templates.py` — base + instruct prompt templates.
  - `flashrag/search_r1/answer_utils.py` — EM/F1 scorer (ground truth — DO NOT modify).
  - `flashrag/config/basic_config.yaml` — retrieval_topk, max_search_turns, sampling params.
  - `run_eval.py` — entrypoint.
- [`local_retriever/`](../local_retriever/) — FAISS retriever HTTP service.
  - [`README.md`](../local_retriever/README.md) — start the retriever; flat vs IVF-SQ8 vs GPU.
  - [`RETRIEVER_INDEXING.md`](../docs/RETRIEVER_INDEXING.md) — RAM costs, index choices.
- [`/workspace/index_creation/`](../../index_creation/) — script + artifact for the IVF4096-SQ8 quantized index (`wiki18_100w_e5_ivf4096_sq8.index`, ~16 GB, 3–10× faster than flat).
- [`scripts/`](../scripts/) — `run_one.sh`, `manage_sglang.sh`, `subsample.sh`, `aggregate.py`, the three sweep scripts.
- [`docker/reason-over-search-v1/`](../docker/) — image used on Vast.ai (`pantomiman/reason-over-search-v1`).
- [`docs/VAST_AI_PLAN_A.md`](../docs/VAST_AI_PLAN_A.md) — Vast.ai GPU cost/throughput analysis for finishing Plan A in ≤24 h. Three concrete fleet configs.
- [`docs/HARDWARE.md`](../docs/HARDWARE.md) — this box's specs + accelerator comparison.
- [`data/`](../data/) — full eval datasets (jsonl, per benchmark).
- [`data_subsample/`](../data_subsample/) — deterministic 1k subsamples for the fast Plan B sweep.
- [`docs/archive/DISCARDED_ABLATIONS.md`](../docs/archive/DISCARDED_ABLATIONS.md) — record of the autoresearch loop's 10 discarded ablations on `experiment_ros/<tag>` (the loop's working files `program.md` and `results.tsv` are no longer tracked).

## Runtime services (everything must be up before eval)

- **Retriever** on `127.0.0.1:3005`. Health: `curl -sS http://127.0.0.1:3005/health` → `"healthy"`. Default is CPU + flat FAISS (~65 GB RAM, faiss-cpu venv at `/venv/retriever`). For ~3–10× faster CPU retrieval pass `--index ./indexes/wiki18_100w_e5_ivf4096_sq8.index`. GPU FAISS is opt-in via `local_retriever/.venv` and **cannot coexist with SGLang on the single 4090** (16 GB VRAM for the index + 22 GB for SGLang).
- **SGLang** on `127.0.0.1:3000` serving the variant under test. Switch with `scripts/manage_sglang.sh switch base|instruct`. Verify: `curl -sS http://127.0.0.1:3000/get_model_info | grep -o instruct`.
- **Eval venv** at `/venv/evaluation_search_r1`. Verify: `/venv/evaluation_search_r1/bin/python -c "import flashrag"`.

## What's been done so far (state as of 2026-05-01)

Canonical state lives in [docs/MILESTONE_1.md](../docs/MILESTONE_1.md). The frozen reproducer config is [docs/FROZEN_CONFIG_v1.md](../docs/FROZEN_CONFIG_v1.md) — that file is the single source of truth for sampling, pipeline, retriever, models, datasets, and audit fixes. **Read it before changing anything.**

**Setup**:
- Both GRPO checkpoints sha256-verified (`PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-(it-)em-grpo`).
- Wiki-18 corpus + E5-base-v2 + flat & IVF-SQ8 FAISS indexes built.
- Datasets present for all 7 benchmarks + deterministic 1k subsamples.
- 10 divergences from upstream audited and fixed ([REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md)).
- Exhaustive paper-vs-ours audit ([PAPER_VS_OURS_AUDIT.md](../docs/PAPER_VS_OURS_AUDIT.md)) catalogued 8 more (D1-D8); the 3 actionable ones are applied (D1 apply_chat=True for base, D-prompt-micro example sentence restored, D8 add_special_tokens block removed).

**Plan B v1 — locked, both variants complete**: see [docs/RESULTS_PLAN_B.md](../docs/RESULTS_PLAN_B.md) and [docs/COMPARISON_PLAN_B_v1.md](../docs/COMPARISON_PLAN_B_v1.md).

| | base v1 avg | base paper | Δ | instruct v1 avg | instruct paper | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Avg EM | 0.292 | 0.312 | −2.0 pp | 0.361 | 0.336 | +2.5 pp |

Format-validity (`</answer>` close-rate): base ≥99.6 % every dataset; instruct 91.4–100 %. Length-truncation ≤0.4 % across the board.

**Plan A — YES (unconditional)**. All decision criteria met. Vast.ai cost analysis: 8× RTX 4090 ≈ $58–77 / 24 h ([docs/VAST_AI_PLAN_A.md](../docs/VAST_AI_PLAN_A.md)).

**Plan B v0** (pre-fix baseline) — archived: [docs/archive/RESULTS_PLAN_B_v0.md](../docs/archive/RESULTS_PLAN_B_v0.md), [docs/COMPARISON_PLAN_B.md](../docs/COMPARISON_PLAN_B.md). Base avg was −8.3 pp vs paper (one-sided → systematic, fixed by D1+D-prompt-micro+D8).

**Discarded experiments** — index of everything tried that didn't survive: [docs/archive/README.md](../docs/archive/README.md). Notable:
- Temperature ≠ 0 hypothesis was **wrong** — paper eval is greedy. [archive/TEMPERATURE_HYPOTHESIS_WRONG.md](../docs/archive/TEMPERATURE_HYPOTHESIS_WRONG.md).
- 11 autoresearch-loop ablations on `experiment_ros/apr27` — only `temperature=0` kept. [archive/DISCARDED_ABLATIONS.md](../docs/archive/DISCARDED_ABLATIONS.md).
- Bamboogle base v1 EM 0.088 was n=125 variance, not a regression. [archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md](../docs/archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md).

## How to run a single eval

```bash
cd /workspace/reason_over_search
scripts/run_one.sh instruct bamboogle 1 > run.log 2>&1
```

`run_one.sh` is **resume-aware**: if `evaluation_search_r1/results/<dataset>/…_seed1/metric_score.txt` exists it skips and exits 0. Clear stale outputs before re-running the same `(variant, dataset, seed)`:

```bash
rm -rf evaluation_search_r1/results/bamboogle/bamboogle_*_search_r1_instruct_seed1
```

Pull EM/F1:

```bash
LATEST=$(ls -dt evaluation_search_r1/results/bamboogle/bamboogle_*_search_r1_instruct_seed1 | head -1)
grep -E "^(em|f1):" "$LATEST/metric_score.txt"
```

A single Bamboogle/instruct run takes ~6 min on this 4090.

## Hardware

Single RTX 4090 (24 GB), AMD EPYC 7642 (48c/96t), 503 GB RAM. No NVLink, single GPU. Practical implications: 3B in bf16 fits comfortably; FAISS index lives in host RAM (with VRAM as opt-in); GPU FAISS and SGLang cannot share the 4090. Full table in [docs/HARDWARE.md](../docs/HARDWARE.md).

## How I want you to work with me

- **Default to terse.** State results and decisions directly; skip narration.
- **Read [REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md) before touching the eval pipeline.** The 10 fixes are load-bearing and were audited against the official repo.
- **Don't modify the EM scorer** (`flashrag/search_r1/answer_utils.py`), the model checkpoints, the FAISS index, or the dataset jsonl files. Those are the ground truth for the experiment.
- **For autoresearch loops** (`experiment_ros/<tag>` branches): the editable surface is `active_pipeline.py`, `parser.py`, `templates.py`, `basic_config.yaml`.
- **Single-run noise is real** (~3 pp at n=125 on Bamboogle when temperature>0). Treat improvements <2 pp EM with skepticism; confirm with a second seed or use temp=0 (greedy).
- **Cite paper sections by URL** when discussing results, especially `arxiv.org/html/2503.09516v5#A6` (Table 3, the GRPO comparison).
- When in doubt about a dataset split or a Search-R1 implementation detail, check the official repo (`github.com/PeterGriffinJin/Search-R1`) before guessing — and ask me before invoking AlphaXiv or other web tools that publish content.

## What's next

Canonical list in [docs/MILESTONE_1.md#whats-left](../docs/MILESTONE_1.md#whats-left). Summary:

1. **Tabulate format-validity / length-truncation rate** per (dataset, variant) from the in-flight v1 sweep's JSONs once they land. Extend `aggregate.py` to surface `'</answer>' in final_response` close-rate.
2. **One-seed full-data runs** for **both base and instruct** (~4 h × 2 on a 4090) to confirm the v1 config converges at scale, not just on 1 k subsamples.
3. **Plan A on Vast.ai** — Jose's job; instructions in [docs/VAST_AI_PLAN_A.md](../docs/VAST_AI_PLAN_A.md) (5 seeds × 7 × 2 = 70 runs, ≤24 h on a fleet, ~$58–108 budget).
4. **Aggregate, write up, publish**: per-benchmark means + std-dev across the 5 seeds, side-by-side with paper, plus the audit + cost summary.

Do **not** change `temperature` or `top_p` — paper eval is greedy. See note above.

**After Phase 1** — further ablations on the autoresearch branch and starting on **ReSearch** (the second paper). Append to this section as the plan firms up.
