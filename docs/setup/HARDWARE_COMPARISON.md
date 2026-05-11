---
title: HARDWARE_COMPARISON — accelerator + provider comparison for M5.1 training
tags: [hardware, training, m5, vast, runpod, thundercompute, gpu]
source: internal
created: 2026-05-11
updated: 2026-05-11
supersedes: ../archive/HARDWARE_COMPARISON_v1.md
---

# Accelerator & provider comparison — M5.1 training

> **Purpose**: Decide which GPU (and which cloud provider) to run a single M5.1 GRPO training run on. Anchored on **live per-step measurements from the M5.1 production run** (Qwen3.5-0.8B GRPO on NeMo-RL, MuSiQue, 1× A100-80GB on Vast — see [`../report/RESULTS_SMOKE_m5.md` §6](../report/RESULTS_SMOKE_m5.md#6-m51-production-training--live)). Spec table carry-forward from [`HARDWARE_4090.md`](HARDWARE_4090.md) (historical 4090 dev-box doc).
>
> **Supersedes** [`../archive/HARDWARE_COMPARISON_v1.md`](../archive/HARDWARE_COMPARISON_v1.md), which was anchored on the steps 15-17 trough (10.6 min/step). That trough turned out to be the *minimum* of a U-shape: per-step time climbed back as the model learned to write longer (better) rollouts. v0 (step-8 anchor, 33 min/step) is at [`../archive/HARDWARE_COMPARISON_v0.md`](../archive/HARDWARE_COMPARISON_v0.md).
>
> **TL;DR (live trend, step 42, 2026-05-11 PM)**: at the **new ~24 min/step anchor** (steps 38-42 mean), the 1× A100 path lands at **~10.2 d / ~$294** (not ~4.5 d / $130 as v1 said). The cost-spread between configs has re-opened: **1× B200 ≈ 34 h / ~$201** (fastest), **2× H100 SXM TP=2 ≈ 1.7 d / ~$153** (cheapest absolute), **1× H100 SXM ≈ 4.6 d / ~$206** (cleanest swap). Sunk cost on A100 is ~$19 (15.7 h elapsed); **porting to B200 now would save ~8.8 days AND ~$74 vs letting A100 finish**.

## 0. What changed since v1 — cost-shift table

For every config below, the v2 numbers are ~2.3× higher than v1 because the live A100 anchor doubled (10.6 → 23.7 min/step). The *relative* speedups vs A100 are unchanged; only the absolute base shifted.

| Config | v1 wall (Mon AM) | **v2 wall (Mon PM)** | v1 cost | **v2 cost** | Δ |
|---|---:|---:|---:|---:|---:|
| 1× A100-80GB | 4.5 d | **10.2 d** | $130 | **$294** | +$164 |
| 1× H100 SXM | 2.0 d | **4.6 d** | $90 | **$206** | +$116 |
| 1× H200 | 1.2 d | **2.7 d** | $104 | **$233** | +$129 |
| 2× H100 SXM (TP=2) | 22 h | **41 h** | $82 | **$153** | +$71 |
| 1× B200 | 14-16 h | **34 h** | $90 | **$201** | +$111 |
| 2× H200 (TP=2) | 18 h | **36 h** | $129 | **$258** | +$129 |

**Recommendation shifted with the costs**:
- v1 said "stay on A100 (~$130, 4.5 d) — port only if you need <2 d urgently". Sunk-cost-aware.
- v2 says "**port now**: B200 saves ~9 days for less money than letting A100 finish ($201 vs $294); 2× H100 TP=2 saves ~8 days for half the money ($153 vs $294)". The U-shape recovery erased the v1 conclusion.

## 1. Live anchor — what we're measuring against

Live trajectory through step 42 (out of 622) from M5.1 production run ([`RESULTS_SMOKE_m5.md` §6](../report/RESULTS_SMOKE_m5.md#6-m51-production-training--live)). The per-step trajectory is U-shaped, not monotone:

| Window | Mean min/step | Phase |
|---|---:|---|
| Steps 1-10 | 42.9 | warmup + long rollouts (gen len 7038 → ~5100) |
| Steps 11-20 | 12.6 | "shrink" phase (gen len → 636 trough at step 17-18) |
| Steps 21-30 | 13.7 | gen len bottom + starting to climb |
| Steps 31-40 | 20.1 | "grow" phase (gen len → ~1000, reward → ~0.15) |
| **Steps 38-42** | **23.7** | **current steady-state anchor** |

Step 42 phase breakdown (1× A100-80GB, micro=1 forced):

| Phase | Time / step | Share | Scales with |
|---|---:|---:|---|
| `policy_training` | ~15.5 min | **~68%** | BF16 TFLOPS + mem-BW; **micro=1 forced** by 80 GB VRAM |
| `policy_and_reference_logprobs` | ~4.5 min | ~20% | Mem-BW (fp32 cast `[B,S,V]`); micro=1 forced same reason |
| `generation` (vLLM async) | ~2.7 min | ~12% | Mem-BW; share has grown vs v1 as rollouts re-lengthened |
| **Total (steady)** | **~23 min** | — | |
| **Elapsed so far (steps 1-42)** | **~15.7 h (0.65 d)** | — | sunk cost ~$19 |
| **Remaining (580 steps × 23.7 min)** | **~229 h ≈ 9.5 d** | — | |
| **Integrated full run** | **~245 h ≈ 10.2 d** | — | cost: ~$294 @ $1.20/h |

**Why the trend reversed**: the model is moving through two distinct learning regimes. Phase 1 (steps 1-17) was "shrink-and-improve" — model learning to stop search-loop spam. Phase 2 (step 18+) is "improve-and-grow" — model learning to reason longer when needed; rewards climbing in lock-step with gen length. Both are good. Bad news for wall-clock: every step in phase 2 costs more compute than the v1 anchor trough assumed. Detailed dynamics in [`../report/RESULTS_m5.md` §4.1](../report/RESULTS_m5.md#41-transferable-observation--rl-training-dynamic-regimes).

**The micro=1 bottleneck (unchanged from v0/v1)**: NeMo-RL v0.6.0's TP=1 path casts the full `[B,S,V]=[1,8192,248320]` logits to fp32 before chunking ([`distributed/model_utils.py:1378`](https://github.com/NVIDIA/NeMo-RL)); the chunked path only exists for TP>1. At 80 GB this OOMs at micro=2 by 0.62 GiB. Three unblock paths:
1. **More VRAM** (H200 141 GB, B200 192 GB) → fits trivially; yaml flip.
2. **TP=2** (multi-GPU) → activates the chunked path; same yaml flip.
3. **Source patch** to `model_utils.py:1378` → deferred (M5.3 candidate).

## 2. Accelerator spec reference (carry from `HARDWARE_4090.md`)

Headline FP16/BF16 dense throughput (no sparsity), memory, interconnect. Real-world throughput depends on kernels, batch size, interconnect — see §3 for per-task M5.1 estimates.

| Accelerator | Arch | Memory | Mem BW | FP16/BF16 (TFLOPS) | FP8 (TFLOPS) | Interconnect | TDP | M5.1 verdict |
|-------------|------|--------|--------|--------------------|--------------|--------------|-----|--------------|
| RTX 4090 | Ada Lovelace | 24 GB | 1.0 TB/s | ~165 | ~330 | PCIe 4 (no NVLink) | 450 W | Too small for M5.1 |
| A100 40 GB | Ampere | 40 GB HBM2e | 1.55 TB/s | 312 | — | NVLink 3 (600 GB/s) | 400 W | Too small (M5.1 needs ≥80 GB) |
| **A100 80 GB** *(live now)* | Ampere | 80 GB HBM2e | 2.0 TB/s | 312 | — | NVLink 3 (600 GB/s) | 400 W | **Baseline — ~10.2 d** |
| H100 PCIe | Hopper | 80 GB HBM3 | 2.0 TB/s | ~756 | ~1513 | PCIe 5 (NVLink bridge opt.) | 350 W | ~5.3 d; ~15% slower than SXM |
| H100 SXM | Hopper | 80 GB HBM3 | 3.35 TB/s | 989 | 1979 | NVLink 4 (900 GB/s) | 700 W | **~4.6 d**; micro=1 still forced (80 GB) |
| H100 NVL (94 GB) | Hopper | 94 GB HBM3 | 3.9 TB/s | ~835 | ~1670 | NVLink bridge (600 GB/s) | 350-400 W | ~4.0 d; 94 GB may unlock micro=2 (tight) |
| **H200 SXM** | Hopper | 141 GB HBM3e | 4.8 TB/s | 989 | 1979 | NVLink 4 (900 GB/s) | 700 W | **~2.7 d; micro=2 unlocked** |
| **B200** (Blackwell) | Blackwell | 192 GB HBM3e | 8 TB/s | ~2250 | ~4500 / 9000 (FP4) | NVLink 5 (1.8 TB/s) | 1000 W | **~34 h; micro=2 unlocked** |
| MI300X | CDNA 3 | 192 GB HBM3 | 5.3 TB/s | ~1300 | ~2600 | Infinity Fabric (896 GB/s) | 750 W | ROCm + NeMo-RL + Qwen3.5 hybrid path = uncharted; **skip** |
| TPU v5p | Google TPU | 95 GB HBM | 2.76 TB/s | 459 (BF16) | — | ICI 3D torus | ~300 W | Requires XLA/JAX port; not available |

Numbers are vendor "dense" (non-sparse) TFLOPS. NeMo-RL is not configured for FP8 on our config, so Hopper/Blackwell FP8 columns are aspirational, not realized in our wall-clock.

## 3. M5.1 wall-clock + cost estimates by hardware

Math anchored on **live 23.7 min/step** (§1, steps 38-42 mean), with three multipliers — unchanged from v1 since they describe *relative* speedups vs A100:

- **Compute/BW speedup vs A100-80GB SXM** (MLPerf + vendor docs, BF16):
  - H100 PCIe ≈ 1.9× (~15% slower than SXM due to lower TDP + PCIe interconnect)
  - H100 SXM ≈ 2.2×
  - H200 SXM ≈ 2.4-2.6× (extra mem-BW helps at vocab=248k cast)
  - B200 ≈ 5-6× over A100 (BF16; MLPerf B200/H100 ≈ 2.2-2.27×, H100/A100 ≈ 2.2×)
- **micro=2 unlock** (only ≥141 GB VRAM, or TP≥2 via chunked path): ~1.5× on training+logprobs phase (≈30% on total step). Same effect from the deferred `model_utils.py:1378` patch.
- **Multi-GPU TP=2 scaling**: ~1.7-1.9× on training+logprobs (good NVLink scaling at this model size); chunked logprobs path comes for free; generation phase doesn't TP-split in vLLM async (conservatively keep single-GPU BW).

| Config | Per-step (steady) | Total run (622 steps) | Code work | Risk |
|---|---:|---:|---|---|
| 1× A100-80GB *(live)* | ~24 min | **~10.2 d (~245 h)** | none | live |
| 1× H100-80GB PCIe | ~12.6 min | ~5.3 d | none | clean |
| **1× H100-80GB SXM** | **~10.8 min** | **~4.6 d (~110 h)** | none | clean |
| 1× H100-80GB SXM + chunk patch | ~7.2 min | ~3.1 d | NeMo-RL source mod + smoke | moderate |
| **1× H200-141GB** | **~6.3 min** | **~2.7 d (~65 h)** | yaml `micro=2` | clean |
| 2× A100-80GB (TP=2) | ~8.8 min | ~3.8 d (~91 h) | TP=2 yaml + smoke | moderate |
| **2× H100-80GB SXM (TP=2)** | **~4.0 min** | **~1.7 d (~41 h)** | TP=2 yaml + smoke re-validate | moderate |
| **1× B200-192GB** | **~2.2 min** | **~34 h** | yaml `micro=2`; verify Blackwell sm_100 image | image-compat |
| 2× H200-141GB (TP=2) | ~2.3 min | ~36 h | TP=2 yaml + smoke | moderate |
| 4× H100-80GB SXM (TP=4) | ~1.5 min | ~26 h | larger reconfig | high |

These are **steady-state extrapolations from the current "improve-and-grow" phase**. The model is at step 42; if rollout length continues climbing toward the 1024-tok/turn cap (currently 927 → 1031 oscillating), step time could rise further. If it plateaus around 1000 tokens, the 23.7 min/step anchor holds.

## 4. Provider pricing snapshot (May 2026)

Spot/marketplace lows from each provider's pricing page. **Verify before launch — prices move weekly.**

| GPU | Vast.ai | RunPod (spot) | RunPod (Secure) | Thundercompute | Notes |
|---|---:|---:|---:|---:|---|
| A100-80GB SXM | ~$0.79-1.20/h | $1.39/h | ~$1.89/h | **$0.78/h** (proto) | Thundercompute cheapest, but PCIe-only & beta |
| H100-80GB PCIe | $2.00/h | ~$2.39/h | ~$3.39/h | **$1.38/h** (proto) | Thundercompute cheapest PCIe |
| H100-80GB SXM | **$1.87/h** | $2.69/h | ~$3.89/h | not offered | Vast cheapest SXM |
| H100 NVL (94 GB) | rare | rare | ~$3.50/h | not offered | Limited inventory |
| H200-141GB | inconsistent | **$3.59/h** | ~$4.99/h | not offered | RunPod cleanest H200 source |
| B200-192GB | spotty (queue) | **$5.98/h** | ~$7.99/h | not offered | Very limited; expect queue on RunPod |
| MI300X-192GB | rare | ~$2.49/h some | — | not offered | Skip for this run (ROCm risk) |

### Thundercompute caveat (unchanged from v1)

Legit Cambridge-MA startup (founded 2024, in beta), most aggressive A100/H100-PCIe pricing in the market. PCIe-only (no SXM), virtualized-GPU stack (untested with NeMo-RL CUDA-IPC weight share), no H200/B200. Useful as the **cheapest A100 swap** if the live Vast A100 box flakes (~$170 for the remaining 580 steps × 24 min × $0.78/h vs $245 on Vast at $1.20/h). Smoke-required. Not a path to <2 d. Full pros/cons in v1 archive.

### Provider call (revised)

- **Vast.ai** still cheapest H100 SXM; reliability host-by-host. Best for the 1× or 2× H100 SXM paths in §6.
- **RunPod Secure Cloud** is the right call for B200 / H200 (multi-day reliability matters; Vast lottery doesn't).
- **Thundercompute** out of frame for this revised picture (PCIe ceiling caps wall-clock at ~5 d, no faster option on the same provider).

## 5. Cost-per-run table (all configs)

Cost = total wall × $/h. **A100 sunk cost so far: ~$19 (15.7 h × $1.20).** Compare to the "live A100" baseline (~$294 if we let it finish).

| Config | Wall | Provider | $/h | **Cost (run)** | Δ vs "let A100 finish" |
|---|---:|---|---:|---:|---:|
| **1× A100 *(let it finish)*** | 10.2 d | Vast | $1.20 | **~$294** | baseline |
| 1× A100 (swap to Thundercompute) | ~9.6 d remaining | Thundercompute | $0.78 | $19 + ~$170 = ~$189 | -$105, same wall |
| 1× H100 PCIe (Thundercompute) | 5.3 d | Thundercompute | $1.38 | $19 + ~$175 = ~$194 | -$100, save 5 d |
| **1× H100 SXM (Vast)** | 4.6 d | Vast | $1.87 | $19 + ~$206 = **~$225** | **-$69, save 5.6 d** |
| 1× H200 (RunPod spot) | 2.7 d | RunPod | $3.59 | $19 + ~$233 = ~$252 | -$42, save 7.5 d |
| 2× A100 (TP=2, Vast) | 3.8 d | Vast | $2.40 | $19 + ~$219 = ~$238 | -$56, save 6.4 d |
| **2× H100 SXM (TP=2, Vast)** | 1.7 d | Vast | $3.74 | $19 + ~$153 = **~$172** | **-$122, save 8.5 d** |
| **1× B200 (RunPod spot)** | ~34 h | RunPod | $5.98 | $19 + ~$201 = **~$220** | **-$74, save 8.8 d** |
| 1× B200 (RunPod Secure) | ~34 h | RunPod | $7.99 | $19 + ~$269 = ~$288 | -$6, save 8.8 d |
| 2× H200 (TP=2, RunPod spot) | 36 h | RunPod | $7.18 | $19 + ~$258 = ~$277 | -$17, save 8.7 d |
| 4× H100 SXM (TP=4, Vast) | 26 h | Vast | $7.48 | $19 + ~$195 = ~$214 | -$80, save 9.3 d |

**Three Pareto winners** vs the "let A100 finish" baseline:
1. **2× H100 SXM (TP=2)** — ~$172, ~1.7 d. **Cheapest AND fast**; pays TP=2 config rewrite.
2. **1× B200** — ~$220, ~34 h. **Fastest single-GPU**; pays Blackwell smoke + RunPod queue risk.
3. **1× H100 SXM** — ~$225, ~4.6 d. **Cleanest swap**; no TP, no Blackwell, simplest.

## 6. Recommendation for the current decision (live, post-trend-reversal)

The right call shifted vs v1. Three viable archetypes; **doing nothing (let A100 finish) is now Pareto-dominated**:

### A) Cheapest + fastest (2-GPU) — 2× H100 SXM (TP=2) on Vast (~$172, ~1.7 d)
- **Why pick this**: cheapest absolute *and* sub-2-day. TP=2 unlocks the chunked-logprobs path so micro=2 comes for free. Vast H100 SXM inventory is reliable. Saves 8.5 days vs A100 path.
- **Why not**: TP=2 yaml flip + smoke re-validate against M4 byte-exact prompt. Higher process risk than (C). Two-machine coordination if you've already split work across boxes.
- **Pre-commit**: 30-min smoke at `policy.parallelism.tensor_model_parallel_size=2`; verify byte-exact prompt parity. ~$2.

### B) Same-day finish (1-GPU) — 1× B200 on RunPod (~$220, ~34 h)
- **Why pick this**: fastest single-GPU; same-day-ish finish. Single-GPU = no multi-GPU rewrite; 192 GB easily fits micro=2.
- **Why not**: Blackwell sm_100 compat for the v2 worker venv (mamba-ssm / flash-attn / nv-grouped-gemm) is **untested by us**. A 30-min B200 smoke (~$3) answers this. RunPod B200 inventory is queued.
- **Pre-commit**: smoke first; if it fails fall back to (A) or (C).

### C) Cleanest swap (1-GPU, ~half wall) — 1× H100 SXM on Vast (~$225, ~4.6 d)
- **Why pick this**: simplest possible swap. yaml unchanged, image unchanged, no multi-GPU, no Blackwell risk. Saves ~5.6 days vs A100 path for a modest cost delta.
- **Why not**: ~4.6 d is still a meaningful wait. (A) and (B) are both ≤2 d and cheaper or comparable.

### Cost-only minimum — 1× A100 on Thundercompute (~$189, ~9.6 d remaining)
- Same wall-clock as letting A100 finish on Vast, $105 cheaper. Useful **only** if the live Vast A100 box becomes unstable (host loss) and we need a re-launch on the cheapest available A100 surface. Not worth the swap otherwise — wall-clock unchanged.

**My single answer**: **Option A (2× H100 SXM TP=2 on Vast)** is the new Pareto pick. Cheapest and fast; the TP=2 config work is worth $100 + 8 d saved. Take option B if same-day finish matters more than $50 cheaper; take option C if multi-GPU is off the table for any reason.

**Sequence either way**:
1. Wait until step 50 lands (~6-7 more hours on A100, ~$8) → first checkpoint, recoverable insurance.
2. Smoke the target hardware (~30 min, ~$3) to derisk the swap.
3. Launch full run on the target hardware; can resume from step-50 ckpt or restart depending on policy state.

## 7. What I'd not do

- **Don't kill A100 before step 50** (~6-7 more hours at current pace). Step-50 ckpt is the recovery insurance.
- **Don't pick H100 PCIe over SXM for the same provider**. 15% slower at often ≥50% the SXM price — bad value. PCIe only makes sense at Thundercompute where no SXM exists.
- **Skip MI300X**. ROCm + NeMo-RL + vLLM + Qwen3.5 hybrid arch is uncharted.
- **Skip FP8**. NeMo-RL v0.6.0 doesn't wire Transformer Engine FP8 through DTensor for Qwen3.5; substantial port for one run.
- **Don't extrapolate from v1's anchor.** The 10.6 min/step trough was a local minimum. Use the steps 38-42 anchor (or whatever the latest is when you read this) — the model is still evolving.

## 8. Open questions before launch

- Does NeMo-RL v0.6.0's v2 worker venv come up on Blackwell sm_100? mamba-ssm / flash-attn / nv-grouped-gemm may need a rebuild. **30-min B200 smoke answers this.**
- For multi-GPU (option A): does the M4 byte-exact prompt check still pass at TP=2? The rollout shape doesn't change but the rendering path goes through a different code branch.
- Will the per-step trend keep climbing? At step 42 gen length is ~927 (vs cap 1024/turn × ~5 turns ≈ 5000). If gen length plateaus near 1000, the 23.7 min anchor holds. If it grows toward 2000, full-run estimate creeps to ~14 d on A100. Watch [`../report/RESULTS_SMOKE_m5.md` §6.2](../report/RESULTS_SMOKE_m5.md#62-per-step-trajectory-live-refresh-as-steps-land).

## 9. Pointers

- 4090 dev-box historical snapshot: [`HARDWARE_4090.md`](HARDWARE_4090.md)
- v1 of this doc (steps 15-17 trough anchor; superseded): [`../archive/HARDWARE_COMPARISON_v1.md`](../archive/HARDWARE_COMPARISON_v1.md)
- v0 of this doc (step-8 anchor; superseded by v1): [`../archive/HARDWARE_COMPARISON_v0.md`](../archive/HARDWARE_COMPARISON_v0.md)
- Live M5.1 trajectory and bottleneck analysis: [`../report/RESULTS_SMOKE_m5.md`](../report/RESULTS_SMOKE_m5.md)
- M5.1 training-dynamics analysis (shrink-then-grow): [`../report/RESULTS_m5.md` §4.1](../report/RESULTS_m5.md#41-transferable-observation--rl-training-dynamic-regimes)
- Current Vast setup runbook: [`../vast/SETUP_VAST.md`](../vast/SETUP_VAST.md)
- M5.1 milestone: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md)
- Paper-vs-ours mapping: [`../milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md)
- Phase-2 runtime-efficiency lever menu: [`../research/RUNTIME_EFFICIENCY.md`](../research/RUNTIME_EFFICIENCY.md)
- Current catch-up TODO: [`../todo/TODO_2026-05-11.md`](../todo/TODO_2026-05-11.md)

## 10. Sources (May 2026)

- [Vast.ai GPU Pricing](https://vast.ai/pricing), [Vast H100 SXM](https://vast.ai/pricing/gpu/H100-SXM), [Vast H100 PCIe](https://vast.ai/pricing/gpu/H100-PCIE)
- [RunPod Pricing](https://www.runpod.io/pricing), [RunPod GPU breakdown — Northflank](https://northflank.com/blog/runpod-gpu-pricing)
- [Thundercompute Pricing](https://www.thundercompute.com/pricing), [Thundercompute A100 pricing](https://www.thundercompute.com/blog/nvidia-a100-pricing), [Thundercompute H100 pricing](https://www.thundercompute.com/blog/nvidia-h100-pricing), [getdeploying.com Thundercompute review](https://getdeploying.com/thunder-compute)
- [GPU Cloud Pricing Comparison 2026 — Spheron](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/), [H100 Rental Prices Compared — IntuitionLabs](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [H100 vs A100 BF16 throughput — Spheron](https://www.spheron.network/blog/nvidia-a100-vs-h100/), [bestgpusforai.com](https://www.bestgpusforai.com/gpu-comparison/a100-vs-h100)
- [B200 vs H100 — Civo](https://www.civo.com/blog/comparing-nvidia-b200-and-h100), [Lightly](https://www.lightly.ai/blog/nvidia-b200-vs-h100), [WhiteFiber LLM training infra](https://www.whitefiber.com/blog/choosing-gpu-infrastructure)
- [B200 vs H200 — Northflank](https://northflank.com/blog/b200-vs-h200)
