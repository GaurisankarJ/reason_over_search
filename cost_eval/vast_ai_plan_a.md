# Plan A on Vast.ai — cost comparison (≤ 24 h target)

Goal: run **Plan A** (5 seeds × 7 datasets × 2 variants = 70 runs, ~517 K examples) end-to-end in **≤ 1 day**, as cheaply as possible.

Source docs:
- [evaluation_search_r1/EVAL_OPS.md](../evaluation_search_r1/EVAL_OPS.md) — wall-clock budgets, where time goes, speedup ranking
- [docs/HARDWARE.md](../docs/HARDWARE.md) — accelerator comparison
- [evaluation_search_r1/COMPARISON_PLAN_B.md](../evaluation_search_r1/COMPARISON_PLAN_B.md) — current state of the reproduction; gap-closing work that **must finish before launching Plan A**.

Snapshot: 2026-04-28. Vast.ai is a marketplace; prices float — re-check before booking.

## Workload model

3B model decode is **memory-bandwidth-bound**, not compute-bound. Plan A is **embarrassingly parallel** across 70 independent runs. So:

- Speed scales linearly with bandwidth (per machine) × number of machines.
- Cost scales with `$/h × hours`, where hours = `total_GPU_h_equivalent / (M × bandwidth_factor)`.
- Cost is **independent of M** for a given GPU type — only the `$/effective-throughput-h` matters. Pick the cheapest cost-per-throughput GPU and use enough of them to fit 24 h.

Baseline single-4090 wall-clock for Plan A, with the [EVAL_OPS speedup ranking](../evaluation_search_r1/EVAL_OPS.md#speedup-ranking) applied (radix-cache + overlap re-enabled, `INFERENCE_MAX_WORKERS=16`, `--num_retriever 4`, IVF-SQ8): **~7–10 days ≈ 240 GPU-h-equiv**.

For 24 h finish: need **M × k ≥ 10** 4090-equivalents.

## Throughput / cost table

Bandwidth ratios from [docs/HARDWARE.md](../docs/HARDWARE.md). Vast.ai $/h are marketplace lows seen on 2026-04-28 (sources at end).

| GPU         | Mem BW   | k = BW vs 4090 | Vast $/h (low end)  | $ / 4090-equiv-h |
|---          |---       |---:            |---:                 |---:              |
| RTX 4090    | 1.0 TB/s | 1.0×           | $0.30 – $0.55       | **$0.30 – $0.55** |
| RTX 5090    | 1.79 TB/s| 1.79×          | ~$0.50 – $1.00      | $0.28 – $0.56    |
| A100 80 GB  | 2.0 TB/s | 2.0×           | $1.64 – $1.74       | $0.82 – $0.87    |
| H100 PCIe   | 3.35 TB/s| 3.35×          | $1.49 – $1.87       | **$0.45 – $0.56** |
| H100 SXM5   | 3.35 TB/s| 3.35×          | $1.53 – $2.27       | $0.46 – $0.68    |
| H200        | 4.8 TB/s | 4.8×           | ~$2.50 – $4.00      | $0.52 – $0.83    |
| B200        | 8 TB/s   | 8×             | ~$5 – $8            | $0.62 – $1.00    |

Cheapest-per-throughput tier: **4090** and **H100 PCIe** are tied at the low end. H100 PCIe wins on host quality (RAM, NVMe, less marketplace variance).

## Three concrete configs (sorted cheapest → simplest)

### Option 1 — 8× RTX 4090 marketplace ✅ cheapest

- 8 × 1.0 = 8 effective. Wall-clock ~21–24 h.
- Cost: 8 × 24 h × $0.30–0.40 ≈ **$58–77**.
- **Constraints**:
  - Most marketplace 4090 hosts have 64–128 GB RAM, **not** 503 GB. Use the **IVF-SQ8 index** (~16 GB RAM, ~5× faster than flat with <1 % recall loss) — see [INDEXING.md](../local_retriever/INDEXING.md).
  - Disk per host: ~70–80 GB for wiki18 corpus + IVF-SQ8 index.
- **Interruptible variant** (~50–70 % of on-demand): could drop total to **$25–40**. Safe because [`run_one.sh`](../scripts/run_one.sh) is resume-aware. Smoke-test resume across instance restarts before committing.

### Option 2 — 3× H100 PCIe ✅ best balance

- 3 × 3.35 ≈ 10 effective. Wall-clock ~24 h, fits exactly.
- Cost: 3 × 24 h × $1.50 ≈ **$108**.
- Hosts typically 1–2 TB RAM, NVMe — flat FAISS works without thinking; less marketplace variance.

### Option 3 — 2–3× H200 (simplest, fewest moving parts)

- 2× H200: 2 × 4.8 = 9.6 effective. Wall-clock ~25 h — *just* over.
- 3× H200: 14.4 effective. Wall-clock ~17 h. Cost: 3 × 17 h × $3.00 ≈ **$153**.
- Use when wall-clock matters more than the ~$50 premium vs Option 2.

## Sharding plan

70 runs = 5 seeds × 2 variants × 7 datasets. Cleanest split:
- **Shard by `(variant, seed)`** → 10 shards × 7 datasets each.
- Each Vast instance gets a list of `(variant, dataset, seed)` triples and runs `scripts/run_one.sh` in a loop.
- Pin each instance to **one variant** — avoids the ~5–10 min `manage_sglang.sh switch` cost per run.
- Pull `evaluation_search_r1/results/*/*/metric_score.txt` back centrally (rclone to S3/R2 or `scp`); run `aggregate.py` locally for the final tables.

For Option 1 (8 machines): collapse to 8 shards of 8–9 runs each. For Option 2 (3 machines): 3 shards of ~23 runs each, splitting across both variants per machine accepted (one switch per machine).

## Open questions (gates before launching)

1. **Is the base-variant gap closed?** Plan A on the current config buys tighter error bars on a wrong base number. The [`apply_chat=True` + `temperature=1.0, top_p=0.95` experiments](../evaluation_search_r1/COMPARISON_PLAN_B.md#recommended-next-steps-before-plan-a) must close the gap to ~3 pp on at least NQ-1k first. **Hard gate.**
2. **Are the 4090 optimizations actually wired up?** Re-enabling radix-cache + overlap is a 1.5–2× lift. If still off in [docker/reason-over-search-v1/](../docker/reason-over-search-v1/), the 240 h estimate slides to ~410 h and the 4090/H100-PCIe economics swap (H100 PCIe wins).
3. **Does the docker image (`pantomiman/reason-over-search-v1`) bake in wiki18 + FAISS?** If not, each instance has to pull ~80 GB on init. Stage on a fast bucket (R2/B2) and download from there — typically 5–10 min on a 1 Gbps Vast host.
4. **Resume behaviour across instance restarts** — `run_one.sh` skips on existing `metric_score.txt`, but only if results dir is persisted. For interruptible 4090s, mount a persistent volume or sync results out after each run.

## Sources

- [Vast.ai pricing](https://vast.ai/pricing)
- [GPU Cloud Pricing Comparison 2026 — Spheron](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/)
- [H100 Rental Prices Compared — IntuitionLabs](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [ComputePrices — Vast](https://computeprices.com/providers/vast)
- [Rent RTX 4090 GPUs on Vast.ai](https://vast.ai/pricing/gpu/RTX-4090)
- [Rent H100 PCIE GPUs on Vast.ai](https://vast.ai/pricing/gpu/H100-PCIE)
