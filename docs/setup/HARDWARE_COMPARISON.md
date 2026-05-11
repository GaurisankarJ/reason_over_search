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
> **TL;DR (after exhaustive provider scan, 2026-05-11 PM)**: at the ~24 min/step A100 anchor, the full run is **~10.2 d / ~$294 on A100**. After surveying 20+ verified providers (§4), the **single right call** is **1× B200 on Lambda Labs at $3.49/h** → ~$138 / ~34 h (saves $156 + 8.8 d vs A100-finish). Spheron B200 spot at $2.12/h is ~$91 if you can babysit preemption. **Skip**: Vast m:71029 ($8.209/h, dominated), RTX PRO 6000 ($1.89/h trap — mem-BW worse than A100, workstation thermals), Thundercompute (PCIe-only beta, no B200). Sunk cost on A100 = $19 (15.7 h to step 42).

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

## 4. Provider pricing — exhaustive verified scan (May 2026)

Sourced from each provider's public pricing page + third-party comparison sites ([getdeploying.com B200](https://getdeploying.com/gpus/nvidia-b200) lists 22+ providers, [getdeploying.com H200](https://getdeploying.com/gpus/nvidia-h200) lists 30+, [getdeploying.com H100](https://getdeploying.com/gpus/nvidia-h100) lists 42+). **Verify before launch — prices move weekly.** Live screenshots checked this session marked ⓘ.

### B200 (192 GB) — verified providers

| Provider | Spot | On-demand | Reserved | Tier | Notes |
|---|---:|---:|---:|---|---|
| **Spheron** | **$2.12/h** | $6.02/h | — | T3 marketplace | Decentralized; per-minute, no egress fees; preemption risk on spot |
| CoreWeave | — | $5.50/h | $2.65/h (1-yr) | T1 enterprise | NASDAQ:CRWV; reserved irrelevant for 34 h run |
| **Lambda Labs** | — | **$3.49/h** | — | T1 established | Dropped from $3.79 → $3.49 in April 2026 |
| RunPod (Secure) | — | $7.99/h | — | T2 | Enterprise SLA tier |
| RunPod (Community) ⓘ | $3.59-5.49/h | $5.98/h | — | T2 marketplace | Live screenshot 2026-05-11: **$5.49/h, 1 max, Low** availability |
| Jarvislabs | — | $5.50/h | — | T2 | India HQ; per-minute |
| FluidStack | — | $4.50-5.80/h | — | T2 EU | B200 + B300 inventory |
| Modal | — | $6.25/h | — | T2 serverless | $0.001736/s; overkill for continuous run |
| DataCrunch | — | competitive | — | T2 EU | Finland; renewable energy |
| Crusoe | spot/OD/reserved | contact sales | available | T1 enterprise | Energy-focused; no public B200 price |
| Vast.ai ⓘ | varies | varies | — | T3 marketplace | Live listing m:71029 (Ohio): **$8.209/h** plus bandwidth |
| Market floor | **$2.25/h** | — | — | — | [SDB200RT index](https://www.silicondata.com/blog/b200-rental-price-march-2026-update) (April 2026) |

### H200 (141 GB) — verified providers

| Provider | Spot | On-demand | Reserved | Tier | Notes |
|---|---:|---:|---:|---|---|
| **Jarvislabs** | — | **$3.80/h** | — | T2 | Cheapest H200 on-demand ([Jarvislabs H200](https://jarvislabs.ai/blog/h200-price)) |
| CoreWeave | — | $3.89/h | — | T1 | Close second |
| **RunPod (Community)** ⓘ | **$3.59/h** | $4.99/h | — | T2 | Live screenshot 2026-05-11: **$3.99/h, 4 max, Low** availability |
| RunPod (Secure) | — | $4.99/h | — | T2 | |
| Lambda Labs | — | $4.49/h | — | T1 | |
| Spheron | — | $4.54/h | — | T3 | SXM |
| Nebius | — | $3.50-4.80/h | up to 35% off | T2 | EU + US, NASDAQ-listed |
| Hyperbolic | — | clusters | — | T3 | Pay-as-you-go |
| Crusoe | spot/OD/reserved | contact sales | available | T1 | |
| Market range | $3.50 floor | $3.72 → $10.60/h | — | | [30+ providers surveyed](https://getdeploying.com/gpus/nvidia-h200) |

### H100 SXM (80 GB) — verified providers

| Provider | Spot | On-demand | Reserved | Tier | Notes |
|---|---:|---:|---:|---|---|
| **Vast.ai** | **$1.49-1.87/h** | varies | — | T3 marketplace | Spot/marketplace lows; host quality varies |
| Vast.ai ⓘ | — | $3.723/h | — | T3 | Live listing m:92731: 8-mo max duration, 98% reliability |
| Spheron | — | $2.01/h (PCIe) | — | T3 | PCIe only |
| **Lambda Labs** | — | **$2.49/h** | — | T1 | SXM; established |
| Jarvislabs | — | $2.69/h | — | T2 | |
| CoreWeave | — | $2.69-4.00/h | — | T1 | |
| **RunPod (Community)** ⓘ | — | $2.99/h | — | T2 | Live screenshot 2026-05-11: **$2.99/h, 1 max, Low** availability |
| RunPod (Community spot/OD) | — | $2.69/h | — | T2 | Posted rate |
| Together.ai | — | $3.49/h | — | T2 | 42% premium over Spheron |
| Thundercompute | — | $1.38/h (PCIe) | — | T3 beta | PCIe + virtualized; not for production |
| Paperspace | — | $5.95/h | $2.24 (3-yr) | T2 | Expensive on-demand |
| AWS p5 | spot avail | $4.49+/h | reserved | T1 hyperscaler | Premium |
| Azure ND H100 v5 | — | $6.98/h | reserved | T1 hyperscaler | Highest of majors |

### Why we're NOT recommending RTX PRO 6000 Blackwell (96 GB) despite the price

RunPod listing shows **RTX PRO 6000 Blackwell at $1.89/h, 96 GB GDDR7, currently Unavailable**. On paper this looks like a great deal (cheapest >80 GB card by far). In practice it's **the wrong card for our workload**:

- **Memory bandwidth is actually *lower* than A100**: GDDR7 at ~1.79 TB/s vs A100's HBM2e at 2.0 TB/s vs H100's HBM3 at 3.35 TB/s. Logprobs (20%) + generation (12%) = ~32% of our step is memory-bound; on this card those phases would be **slower per token than the A100 we're trying to leave**.
- **Workstation silicon, not datacenter**: GB202 die family (same as RTX 5090), not the GB100/GB200 used in B200. NVIDIA's CUDA optimization focus is on datacenter SKUs; RL frameworks (NeMo-RL, vLLM) are tuned for H100/A100/B200.
- **Sustained-load reliability unproven**: 600 W TDP on workstation thermals isn't designed for 80+ hours of 100% utilization. Thermal throttling or instability over multi-day runs is a real risk that won't show up in a short smoke.
- **sm_120 architecture is uncharted for our v2 worker venv** — mamba-ssm, flash-attn, causal-conv1d, nv-grouped-gemm all need rebuild + validation. Same compat risk as B200 (sm_100), but with **none of the upside** since B200 has 4-5× more compute and HBM3e.
- **96 GB unlocks `micro=2` but doesn't help**: the unlock is worth ~1.5× on the compute-bound training phase; doesn't help the memory-bound phases that are already worse than A100.

**Realistic estimate** (not the optimistic $149/3.3d I first sketched): **~$190-220 / ~4-5 d, with multi-day stability risk + uncharted driver path**. Strictly worse than Lambda B200 ($138 / 34 h) on every axis. **Skip it.**

### Thundercompute caveat (unchanged from v1)

Legit Cambridge-MA startup (founded 2024, beta), most aggressive A100/H100-PCIe pricing. PCIe only, virtualized stack (NeMo-RL CUDA-IPC untested), no H200/B200. Useful as the **cheapest A100 swap** if Vast box flakes. Not a path to <2 d.

### Reliability tier legend

- **T1** — established hyperscalers / enterprise GPU clouds (CoreWeave, Lambda, Crusoe, AWS, Azure, GCP). 1+ year track record, public uptime SLAs, premium pricing.
- **T2** — well-funded mid-size (RunPod Secure, Jarvislabs, Nebius, DataCrunch, FluidStack, Modal). Transparent pricing, reliable for multi-day runs.
- **T3** — newer / marketplace (Spheron, Vast.ai, Hyperbolic, Thundercompute). Cheapest, but **preemption real on spot tier**.

### Provider call (revised)

- **Lambda Labs** is the new top recommendation for B200 — clean on-demand at $3.49/h, T1 reliability, no preemption risk.
- **Spheron B200 spot at $2.12/h** is the cost-min play if you're willing to babysit preemption (save_period=50 means ≤4 h lost if reclaimed; relaunch from ckpt).
- **CoreWeave** is the enterprise-SLA pick if "this absolutely cannot fail" matters.
- **RunPod (live screenshot)** — B200 at $5.49/h is mid-pack; H200 at $3.99/h is fine; H100 SXM at $2.99/h is on-demand, not their lowest. Convenient if already using RunPod.
- **Vast.ai** still has the cheapest H100 SXM **at spot/marketplace lows**, but the on-demand listings (like m:92731 at $3.723) are *more* expensive than Lambda H100. Always sort by price.

## 5. Cost-per-run table — actual verified rates (May 2026)

Cost = total wall × $/h. **A100 sunk cost so far: ~$19 (15.7 h × $1.20).** Compare to the "let A100 finish" baseline (~$294). Ordered cheapest → most expensive total run.

| Config | Wall | Provider | $/h | **Cost (run)** | Δ vs A100-finish |
|---|---:|---|---:|---:|---:|
| **1× B200** | ~34 h | **Spheron** (spot) | $2.12 | $19 + ~$72 = **~$91** | **−$203, save 8.8 d** ⚠ preempt risk |
| **1× B200** | ~34 h | **Lambda Labs** | $3.49 | $19 + ~$119 = **~$138** | **−$156, save 8.8 d** ⭐ |
| 1× B200 | ~34 h | RunPod (Community) ⓘ | $5.49 | $19 + ~$187 = ~$206 | −$88, save 8.8 d |
| 1× B200 | ~34 h | CoreWeave | $5.50 | $19 + ~$187 = ~$206 | −$88, save 8.8 d (T1 SLA) |
| 1× B200 | ~34 h | Jarvislabs | $5.50 | $19 + ~$187 = ~$206 | −$88, save 8.8 d |
| 1× B200 | ~34 h | RunPod (Secure) | $7.99 | $19 + ~$272 = ~$291 | −$3, save 8.8 d |
| **1× B200** | ~34 h | Vast m:71029 | **$8.209** | $19 + ~$279 = ~$298 | +$4, save 8.8 d — **dominated** |
| 1× H200 | ~65 h | **Jarvislabs** | **$3.80** | $19 + ~$247 = **~$266** | −$28, save 7.5 d |
| 1× H200 | ~65 h | CoreWeave | $3.89 | $19 + ~$253 = ~$272 | −$22, save 7.5 d |
| 1× H200 | ~65 h | RunPod (spot) | $3.59 | $19 + ~$233 = ~$252 | −$42, save 7.5 d |
| 1× H200 | ~65 h | RunPod (Community) ⓘ | $3.99 | $19 + ~$259 = ~$278 | −$16, save 7.5 d |
| RTX PRO 6000 (96 GB) ❌ | ~4-5 d realistic | RunPod (when avail.) | $1.89 | $19 + ~$190-220 = ~$210-240 | mem-BW < A100; workstation thermals; **don't pick** |
| 1× H100 SXM | ~110 h | Vast (spot, if avail.) | $1.87 | $19 + ~$206 = ~$225 | −$69, save 5.6 d |
| 1× H100 SXM | ~110 h | **Lambda Labs** | $2.49 | $19 + ~$274 = ~$293 | −$1, save 5.6 d |
| 1× H100 SXM | ~110 h | RunPod (Community) ⓘ | $2.99 | $19 + ~$329 = ~$348 | +$54, save 5.6 d |
| 1× H100 SXM | ~110 h | Vast m:92731 | $3.723 | $19 + ~$410 = ~$429 | +$135, save 5.6 d — **dominated** |
| 2× H100 SXM (TP=2) | 41 h | Vast (2× spot) | $3.74 | $19 + ~$153 = ~$172 | −$122, save 8.5 d |
| 2× H100 SXM (TP=2) | 41 h | Lambda (2× $2.49) | $4.98 | $19 + ~$204 = ~$223 | −$71, save 8.5 d |
| 2× H200 (TP=2) | 36 h | Jarvislabs (2× $3.80) | $7.60 | $19 + ~$274 = ~$293 | −$1, save 8.7 d |
| **1× A100 *(let it finish)*** | 245 h remaining | Vast | $1.20 | **~$294** | baseline |
| 1× A100 (swap to Thundercompute) | ~9.6 d remaining | Thundercompute | $0.78 | $19 + ~$170 = ~$189 | −$105, same wall |

**Top 2 Pareto winners** (best ratio of $ saved + wall saved):

1. **🥇 Lambda Labs B200 — ~$138, ~34 h**. Best risk-adjusted: T1 reliability, on-demand (no preempt), ~50% below the Vast listing. **My pick for production.**
2. **🥈 Spheron B200 (spot) — ~$91, ~34 h**. Cheapest full-stop ($47 below Lambda). Preemption risk on T3 marketplace; mitigated by step-50 checkpoint resume. Worth the babysitting if cost-min matters.

The Vast m:71029 B200 at $8.209/h is **dominated by 6 cheaper B200 options on the same wall-clock**. The "best" pick on Vast specifically is to sort their listings by `$/h` and re-evaluate; the m:71029 host is overpriced relative to the market.

**Cheaper-on-paper traps we're avoiding**:
- RTX PRO 6000 at $1.89/h: GDDR7 mem-BW is *worse than A100*, workstation thermals not built for multi-day sustained load — see §4 "Why we're NOT recommending RTX PRO 6000" for the full reasoning.
- Vast on-demand listings: spot/marketplace lows ($1.49-1.87/h) only beat Lambda *if available*; on-demand listings can be 2× the Lambda price.

## 6. Recommendation for the current decision (live, post-exhaustive-survey)

Exhaustive provider scan (May 2026) changed the picture again. The best on-demand B200 rate is **Lambda Labs at $3.49/h**, not RunPod or Vast. New ranking with single-GPU paths winning across the board:

### A) Best risk-adjusted — **1× B200 on Lambda Labs** (~$138, ~34 h) ⭐
- **Why pick this**: cheapest on-demand B200 in the market; T1-reliability provider (a16z + NVIDIA backed; long-standing ML cloud); no preemption; single-GPU = no multi-GPU config rewrite; 192 GB unlocks `micro=2` trivially.
- **Why not**: still need the 30-min Blackwell sm_100 smoke (~$2 on Lambda) to verify the v2 worker venv (mamba-ssm / flash-attn / nv-grouped-gemm) builds on sm_100.
- **Cost vs A100-finish**: saves ~$156 AND ~8.8 d. Clear win.

### B) Cost-min play — **1× B200 spot on Spheron** (~$91, ~34 h, preempt risk)
- **Why pick this**: cheapest M5.1 run of any verified option ($91 total). Marketplace + per-minute billing, no egress fees. Save $47 vs Lambda.
- **Why not**: Spheron is T3 (newer marketplace; preemption real). For our 34 h run, expected ≤4 h work lost per preemption (save_period=50 means a checkpoint every ~3.3 h at B200 steady state). 1-2 preemptions over 34 h is realistic; relaunch from ckpt each time. Net wall could stretch to 40-50 h with bad luck.
- **Caveat**: smoke-required, same as (A); add preemption-handling logic to `run.sh` (auto-resume from latest ckpt on relaunch).

### C) Same-day finish on screenshotted RunPod — **1× B200** at $5.49/h (~$206, ~34 h)
- **Why pick this**: if you're already in the RunPod console with this listing in front of you, click-rent path. $68 more than Lambda but zero account-switching friction. Cheaper than the Vast m:71029 listing at $8.209.
- **Why not**: $68 more than Lambda for the same outcome.

### D) Cleanest 1-GPU swap if Blackwell smoke fails — **1× H100 SXM on Lambda** (~$293, ~4.6 d)
- **Why pick this**: fallback path. T1 reliability, established Blackwell-free toolchain (mature on Hopper), no compat unknowns. ~$1 cheaper than letting A100 finish, saves 5.6 d.
- **Why not**: $155 more than Lambda B200 for 4 d slower. Only makes sense if B200 smoke fails and you need to ship.

### E) TP=2 multi-GPU plays (situational)

- **2× H100 SXM on Vast (TP=2)**: ~$172, ~1.7 d. Cheapest *and* fast IF the Vast spot lows ($1.87/h) are available right now. Pays TP=2 config tax + M4 byte-exact prompt re-validate. Worth checking Vast inventory before committing.
- **2× H200 on Jarvislabs (TP=2)**: ~$293, ~1.5 d. T2 reliability. Probably not worth the TP=2 tax over single-B200 paths.

### Skipped: RTX PRO 6000 ($1.89/h)

96 GB at $1.89/h looks like a bargain. It's not — see §4 "Why we're NOT recommending RTX PRO 6000". The headline reasons: (1) memory bandwidth is *worse than A100*; (2) workstation thermals aren't built for 80+ hours of 100% utilization; (3) sm_120 driver path uncharted; (4) the cheap hourly is more than wiped out by the slower wall-clock and stability risk. Skip.

### Doing nothing (let A100 finish) — Pareto-dominated

- 245 h remaining × $1.20 = ~$294 on A100. **Strictly worse than Lambda B200** ($138 / 8.8 d faster) and **strictly worse than Spheron B200 spot** ($91 / 8.8 d faster). Only stay on A100 if (a) Lambda B200 smoke fails AND (b) you can't afford H100 swap risk.

### Recommended sequence

1. **Wait until step 50 lands** on A100 (~6-7 more hours, ~$8) → first checkpoint = swap-insurance.
2. **Rent 1× B200 on Lambda Labs** (~$3.49/h on-demand). Boot the standard image, clone repo, `bash training/scripts/bootstrap.sh`.
3. **30-min compat smoke** (~$2): `cd training_m5_1 && bash scripts/smoke.sh`. Watch for sm_100 kernel errors in mamba-ssm / flash-attn / causal-conv1d / nv-grouped-gemm.
4. **If smoke green** (most likely; Lambda has been running Blackwell since GA): flip `train_micro_batch_size: 1 → 2` in `m5_1_research_paper.yaml`, launch full run. ETA ~34 h, total cost ~$138 + ~$8 A100 wait = **~$146**.
5. **If smoke red**: fall back to (E) Lambda H100 SXM (~$293, ~4.6 d) or wait for the chunked-fp32 patch (M5.3).
6. **Once swap is healthy**: kill the A100 run on Vast.
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
