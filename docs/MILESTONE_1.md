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

## Status (2026-04-28)

### What was done

- Adapted the FlashRAG/ReSearch eval pipeline to Search-R1 (no official eval pipeline ships upstream).
- Both GRPO checkpoints (base, instruct) sha256-verified against the upstream HF repos.
- Wiki-18 corpus + E5-base-v2 encoder + FAISS Flat IP and IVF-SQ8 indexes built.
- Exhaustive paper-vs-ours audit ([PAPER_VS_OURS_AUDIT.md](PAPER_VS_OURS_AUDIT.md)): 8 divergences catalogued, 10 earlier ones already fixed ([REPRODUCIBILITY.md](REPRODUCIBILITY.md)).
- Plan B sweep (1 seed × 7 datasets × 2 variants; 1k subsamples for large datasets, full Bamboogle/MuSiQue) — see [RESULTS_PLAN_B.md](RESULTS_PLAN_B.md), [COMPARISON_PLAN_B.md](COMPARISON_PLAN_B.md).
- Bamboogle apply_chat=True probe on base: closed gap to paper exactly (EM 0.112→0.128, close-rate 84%→100%).
- Vast.ai Plan-A fleet costing: 8× RTX 4090 ≈ $58–77 / 24 h ([VAST_AI_PLAN_A.md](VAST_AI_PLAN_A.md)).

### Results so far

Plan B average EM vs paper (Search-R1 v5 Table 3, Qwen2.5-3B GRPO):

| Variant  | Plan B | Paper | Δ |
|---       |---:    |---:   |---:|
| Instruct | 0.367  | 0.336 | **+3.1 pp** (reproduction-grade) |
| Base     | 0.229  | 0.312 | **−8.3 pp** (below on all 7 datasets) |

- **Instruct**: within ±5 pp on 6 of 7 datasets; Bamboogle overshoots by +12.8 pp. Reproduction is essentially complete on this variant.
- **Base**: one-sided gap on every dataset → systematic, not noise. Root cause identified ([PAPER_VS_OURS_AUDIT.md D1](PAPER_VS_OURS_AUDIT.md#d1-in-detail-the-load-bearing-one)): `scripts/run_one.sh:35` hard-codes `apply_chat=False` for base, but the upstream training code applies the chat template unconditionally. Bamboogle probe confirmed the fix; full sweep pending.

## What's left

In order, gating Plan A.

**Done / in flight**:
- ✅ All three audit fixes applied: `apply_chat=True` for base ([run_one.sh:35](../scripts/run_one.sh#L35)), `For example, <answer> Beijing </answer>.` restored ([templates.py:10](../evaluation_search_r1/flashrag/search_r1/templates.py#L10)), `add_special_tokens` block removed ([active_pipeline.py](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py)).
- 🟡 Base sweep with the fixes on all 7 datasets (data_subsample, seed 1) running via `run_variant_sweep.sh` since 2026-04-28 16:35. Bamboogle complete; NQ in progress; 5 more queued.

**Open**:
1. **Inspect the Bamboogle base regression** from this sweep (EM 0.088 vs the earlier 13:05 probe's 0.128). Likely culprit: `add_special_tokens` removal or template-sentence restore. Diff the intermediate JSONs.
2. **Tabulate format-validity / length-truncation rate** per (dataset, variant) from the in-flight sweep's JSONs once they land. Extend `aggregate.py` to surface `'</answer>' in final_response` close-rate.
3. **One-seed full-NQ base run** (~4 h on a 4090) to confirm the gap closes at scale, not just on subsamples.
4. **Plan A on Vast.ai** — 5 seeds × 7 × 2 = 70 runs, ~517 K examples, ≤24 h on a fleet ([VAST_AI_PLAN_A.md](VAST_AI_PLAN_A.md)).
5. **Aggregate, write up, publish**: per-benchmark means + std-dev across the 5 seeds, side-by-side with paper, plus the audit + cost summary.
6. **Code/Docker cleanup** for deliverables (4)–(6) above.
