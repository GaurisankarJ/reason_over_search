# Milestone 1: Baseline [Search-R1](https://www.alphaxiv.org/abs/2503.09516)

## Phase 1 Context

- Retriever code is working and has already been tested.
- Evaluation code is a hybrid:
  - base evaluation flow adapted from ReSearch
  - Search-R1 prompts and evaluation logic added on top
- Search-R1 does not provide an official evaluation pipeline, so this adapted pipeline must be validated carefully.

## Goal

Reproduce the Search-R1 3B baseline for both model variants:
- base
- instruct

Target benchmarks:
- Bamboogle (first dataset for fast iteration)
- 2WikiMultiHopQA
- TriviaQA
- PopQA
- MuSiQue
- NQ (in-distribution)
- HotpotQA (in-distribution)

## Dataset Requirements

- Download missing datasets from Hugging Face.
- Use the Search-R1 paper to confirm official dataset sources.
- Verify existing local datasets against online sources.
- Confirm correct split per benchmark (test/dev/train), including whether `dev` is treated as evaluation where no test split is available.
- Use AlphaXiv AI paper assistant when split definitions are ambiguous.

## Reproducibility and Cost Constraints

- Primary objective: reproducible setup on Vast.ai and in-house GPUs.
- Secondary objective: minimize cost and wall-clock runtime.
- Track both reproducibility and cost clearly; these are important for publication quality.

## Step-by-step

1. Prepare runtime on Vast.ai.
   - Use this Docker image: [pantomiman/reason-over-search-v1](https://hub.docker.com/r/pantomiman/reason-over-search-v1)
   - Or build and push your own image using `docker/reason-over-search-v1/README.md`.
   - Create a Vast.ai custom template from that image and start an instance.
   - Storage guidance: allocate at least `150 GB` for indexes and model files.
   - GPU guidance: start testing with lower-cost options first (for example, 40 GB VRAM class) before moving to larger GPUs (for example, A100 80 GB), depending on throughput and stability.

2. Set up local retriever on the Vast instance.
   - Follow `local_retriever/README.md`.
   - If you are using the Docker image above, required environments are already installed, so you can skip environment creation.
   - Download:
     - corpus
     - indexes
     - embedding model

3. Run and validate the retriever service.
   - Start the retriever using `local_retriever/README.md`.
   - Run the health/search test calls to confirm it is serving correctly.
   - Resource note: the retriever runs on CPU and needs about `~80 GB` CPU RAM.

4. Set up evaluation models and SGLang on the Vast instance.
   - Go to `evaluation_search_r1/`.
   - Activate the evaluation environment.
   - Download both Search-R1 model variants (base and instruct).
   - Start the SGLang server for the model being evaluated.

5. Run baseline evaluations.
   - Run all target benchmarks listed in this milestone.
   - Run each benchmark `5` times per model variant.
   - Report the average score across the 5 runs for each benchmark.

## If Results Diverge from Paper

- Inspect prompt templates and answer verification logic in `evaluation_search_r1/flashrag/search_r1/`.
- Verify behavior separately for base and instruct variants.
- Cross-check implementation details against the official repository:
  - [Search-R1 GitHub](https://github.com/PeterGriffinJin/Search-R1)

## Deliverables

1. All evaluation datasets downloaded, verified, and tracked appropriately.
2. Evaluation pipeline validated and run on all listed benchmarks.
3. Aggregated results from repeated runs (5 runs per benchmark/model variant).
4. Code reviewed and cleaned up for clarity (remove unnecessary complexity/fluff).
5. Docker setup verified; documentation updated for clear reproduction.
6. Repository state suitable for easy reproduction and publication submission.

## Status (2026-04-29)

### What was done

- Adapted the FlashRAG/ReSearch eval pipeline to Search-R1 (no official eval pipeline ships upstream).
- Both GRPO checkpoints (base, instruct) sha256-verified against the upstream HF repos.
- Wiki-18 corpus + E5-base-v2 encoder + FAISS Flat IP and IVF-SQ8 indexes built.
- Exhaustive paper-vs-ours audit ([PAPER_VS_OURS_AUDIT.md](PAPER_VS_OURS_AUDIT.md)): 8 divergences catalogued, 10 earlier ones already fixed ([REPRODUCIBILITY.md](REPRODUCIBILITY.md)).
- Plan B v0 sweep — preserved as [archive/RESULTS_PLAN_B_v0.md](archive/RESULTS_PLAN_B_v0.md); v0 result dirs archived locally to `evaluation_search_r1/results/_archive_v0/` (gitignored, so only on the local machine).
- Three audit fixes applied: `apply_chat=True` for base ([run_one.sh:35](../scripts/run_one.sh#L35)), `For example, <answer> Beijing </answer>.` restored ([templates.py:10](../evaluation_search_r1/flashrag/search_r1/templates.py#L10)), `add_special_tokens` block removed ([active_pipeline.py](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py)). `temperature: 0.0` kept (paper eval is greedy per upstream `verl` `_validate()` override).
- **Plan B v1 base sweep complete** — full comparison in [COMPARISON_PLAN_B_v1.md](COMPARISON_PLAN_B_v1.md).
- **Plan B v1 instruct sweep in progress** (started 2026-04-29 06:46 UTC, ETA ~6 h).
- Vast.ai Plan-A fleet costing: 8× RTX 4090 ≈ $58–77 / 24 h ([VAST_AI_PLAN_A.md](VAST_AI_PLAN_A.md)).

### Plan B v1 — base results

All 7 base datasets within 4 pp of paper. Average residual **−2.0 pp** (was −8.3 pp in v0).

| Dataset | v0 EM | **v1 EM** | Paper EM | v1−paper |
|---|---:|---:|---:|---:|
| Bamboogle (n=125) | 0.112 | 0.088 | 0.128 | −4.0 (n=125 noise) |
| NQ-1k | 0.316 | **0.390** | 0.421 | −3.1 |
| TriviaQA-1k | 0.421 | **0.583** | 0.583 | **0.0** |
| PopQA-1k | 0.309 | **0.424** | 0.413 | **+1.1** |
| HotpotQA-1k | 0.201 | 0.263 | 0.297 | −3.4 |
| 2WikiMultiHopQA-1k | 0.207 | 0.239 | 0.274 | −3.5 |
| MuSiQue (full, 2417) | 0.034 | 0.055 | 0.066 | −1.1 |
| **Average** | **0.229** | **0.292** | **0.312** | **−2.0** |

`</answer>` close-rate ≥99.6 % on all 7 v1 base datasets (was 84 % on v0 Bamboogle base). The minor "≤1 pp" prompt + special-tokens fixes together delivered +4.4 pp on NQ — the audit underestimated; the lift held at sweep scale.

### Plan A — YES (conditional on instruct)

**Decision criteria** (set before the sweep): all 7 base datasets within 8 pp of paper, no catastrophic divergence, average residual ≤4 pp. **Met.** Max gap 4.0 pp (Bamboogle, n=125 noise), avg −2.0 pp.

The YES is conditional on instruct v1 also landing within 8 pp of paper. Instruct v0 was already +3.1 pp above paper on average and within ±5 pp on 6/7 datasets, so this is highly likely; will be locked when the in-flight instruct sweep completes (~12:45 UTC) and [COMPARISON_PLAN_B_v1.md](COMPARISON_PLAN_B_v1.md) is updated.

## What's left

- ✅ **Format-validity / length-truncation tabulation** — `aggregate.py` extended to surface `'</answer>' in final_response` close-rate, length-truncation rate (non-`stop` SGLang reasons), and mean completion tokens per (dataset, variant). Output in [RESULTS_PLAN_B.md#trace-health](RESULTS_PLAN_B.md#trace-health). Headlines on current state of `evaluation_search_r1/results/`:
  - **Bamboogle base v1: 100% close-rate** (vs 84% in v0 — apply_chat fix confirmed at the trace-health level).
  - **NQ base v1: 99.9% close-rate**.
  - Length-truncation ≤0.2% across all (dataset, variant) — the per-step cap is not biting.
  - One soft datapoint: instruct/musique close-rate 91.7% (full data, n=2417) — slightly lower than the 1k-subsample datasets, worth a glance once base/musique completes.
- ❎ **One-seed full-data runs** for both base and instruct — **NOT NEEDED.** Plan B v1 base is already within 4 pp / avg −2.0 pp of paper; the residual is consistent with 1 k-subsample SE + single-seed greedy variance. Plan A's first seed runs at full data and provides the same validation, parallelized across the Vast fleet — no reason to spend ~8 local-4090 hours on it first.
1. **Instruct v1 sweep finishes + [COMPARISON_PLAN_B_v1.md](COMPARISON_PLAN_B_v1.md) updated with instruct rows** (~12:45 UTC).
2. **Plan A on Vast.ai** — 5 seeds × 7 × 2 = 70 runs, ~517 K examples, ≤24 h on a fleet. Instructions for **Jose**: see [VAST_AI_PLAN_A.md](VAST_AI_PLAN_A.md).
3. **Aggregate, write up, publish**: per-benchmark means + std-dev across the 5 seeds, side-by-side with paper, plus the audit + cost summary.
