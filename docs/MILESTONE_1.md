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
- Plan B sweep (1 seed × 7 datasets × 2 variants; 1k subsamples for large datasets, full Bamboogle/MuSiQue) — full per-dataset numbers in [RESULTS_PLAN_B.md](RESULTS_PLAN_B.md).
- Three audit fixes applied: `apply_chat=True` for base ([run_one.sh:35](../scripts/run_one.sh#L35)), `For example, <answer> Beijing </answer>.` restored ([templates.py:10](../evaluation_search_r1/flashrag/search_r1/templates.py#L10)), `add_special_tokens` block removed ([active_pipeline.py](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py)).
- Plan B v1 base sweep (locked config, all 7 datasets, seed 1) running on local since 2026-04-28 16:35.
- Vast.ai Plan-A fleet costing: 8× RTX 4090 ≈ $58–77 / 24 h ([VAST_AI_PLAN_A.md](VAST_AI_PLAN_A.md)).

### Results so far

**Configuration is converging on NQ-1k.** The two "≤1 pp" minor fixes turned out to be much more impactful than the audit estimated:

| NQ-1k run | EM | Δ vs prior |
|---|---:|---:|
| Plan B v0 (no chat, no minor fixes) | 0.316 | — |
| Probe: base + apply_chat only | 0.346 | +3.0 pp |
| **v1 (locked): base + apply_chat + prompt sentence + no special tokens** | **0.390** | **+4.4 pp on top** |
| Paper | 0.421 | +3.1 pp residual gap |

So the two minor fixes together gave +4.4 pp on NQ — the audit underestimated. Combined with apply_chat that's +7.4 pp from v0, leaving ~3.1 pp residual to paper (within subsample SE + plausible single-seed noise).

This also resolves the Bamboogle regression worry: the EM 0.088 from the v1 sweep was n=125 variance, not a real regression. The locked config is genuinely converging. See [docs/archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md](archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md) for the post-mortem.

TriviaQA is now running (~6%); sweep ETA ~5 h to last dataset.

**Plan B (v0) average vs paper** — kept here as the pre-fix baseline:

| Variant  | Plan B v0 | Paper | Δ |
|---       |---:    |---:   |---:|
| Instruct | 0.367  | 0.336 | **+3.1 pp** (reproduction-grade) |
| Base     | 0.229  | 0.312 | **−8.3 pp** |

## What's left

- ✅ **Format-validity / length-truncation tabulation** — `aggregate.py` extended to surface `'</answer>' in final_response` close-rate, length-truncation rate (non-`stop` SGLang reasons), and mean completion tokens per (dataset, variant). Output in [RESULTS_PLAN_B.md#trace-health](RESULTS_PLAN_B.md#trace-health). Headlines on current state of `evaluation_search_r1/results/`:
  - **Bamboogle base v1: 100% close-rate** (vs 84% in v0 — apply_chat fix confirmed at the trace-health level).
  - **NQ base v1: 99.9% close-rate**.
  - Length-truncation ≤0.2% across all (dataset, variant) — the per-step cap is not biting.
  - One soft datapoint: instruct/musique close-rate 91.7% (full data, n=2417) — slightly lower than the 1k-subsample datasets, worth a glance once base/musique completes.
1. **One-seed full-data runs** for **both base and instruct** (~4 h × 2 on a 4090) to confirm the v1 config converges at scale, not just on 1 k subsamples.
2. **Plan A on Vast.ai** — 5 seeds × 7 × 2 = 70 runs, ~517 K examples, ≤24 h on a fleet. Instructions for **Jose**: see [VAST_AI_PLAN_A.md](VAST_AI_PLAN_A.md).
3. **Aggregate, write up, publish**: per-benchmark means + std-dev across the 5 seeds, side-by-side with paper, plus the audit + cost summary.
