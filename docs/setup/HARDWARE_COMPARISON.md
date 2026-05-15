---
title: HARDWARE_COMPARISON v3 — accelerator + provider comparison for M5.1 training
tags: [hardware, training, m5, vast, runpod, spheron, gpu, h100, h200, b200, a100]
source: internal
created: 2026-05-11
updated: 2026-05-15
supersedes: ../archive/HARDWARE_COMPARISON_v2.md
---

# Accelerator & provider comparison — M5.1 training (v3)

> **Purpose**: decide which GPU (and which cloud provider) to run a single M5.1 GRPO training run on. Anchored on **live per-step measurements from the M5.1 b200-a3 production run** (Qwen3.5-0.8B GRPO on NeMo-RL, MuSiQue, 1× B200-192GB on Spheron; crashed at step 56 after 9.29 h; W&B run `reason_over_search_b200/h68uskz6`).
>
> **Supersedes** [`../archive/HARDWARE_COMPARISON_v2.md`](../archive/HARDWARE_COMPARISON_v2.md), which anchored on a *projected* B200 wall-clock of 2.2 min/step (5-6× over A100). Live measurement showed **8.9 min/step (2.66× over A100)**, ~4× slower than v2's projection. Root cause: Blackwell sm_100 kernel immaturity in NeMo-RL / vLLM 0.17 for BF16-dense small-batch workloads. The big B200 numbers (~2250 TFLOPS BF16, 8 TB/s mem-BW) include FP4 + sparsity headroom that doesn't apply to our config. Hopper (H100, H200) and Ampere (A100) projections in v2 are closer to accurate.
>
> **TL;DR (2026-05-15 PM)**: at the **8.9 min/step live B200 anchor**, the paper-faithful 2-epoch run is ~92 h. Live a4 relaunched on H200 with persistent volume. **2× H200 SXM5 (TP=2) on Spheron is the cost+wall-clock Pareto pick** for a new launch (~36-52 h / ~$137-200 at $1.56/GPU/h). **8× H200 SXM5 (DP=8) on Spheron** (`CONFIGURATION LOCKED` at $15.15/h = $1.89/GPU/h, verified screenshot) is the only single-box config that fits 2 epochs in ≤24 h, but is overkill for a 0.8B model unless multiple ablations queue behind. **Stay-on-B200** is dominated; the calibration shock makes single B200 ~3× more expensive than a 2× H200 equivalent.

## 0. What changed since v2 — the B200 calibration shock

The v2 doc assumed 1× B200 ≈ 5-6× compute speedup over A100 in BF16, plus 1.5× from unlocking `train_micro_batch_size=2` at seq=8192, for a combined ~10.8× → 2.2 min/step / 34 h full run. Live data on b200-a3 (W&B `h68uskz6`, 55 steps before crash):

| Metric | v2 projected | v3 measured |
|---|---:|---:|
| Steady-state per-step | 2.2 min | **8.9 min** (532 s trailing-20 at step 38; 612 s real wall over run lifetime) |
| Full 622-step run | ~34 h | **~92 h** |
| Speedup vs 1× A100 (23.7 min/step v2 anchor) | 10.8× | **2.66×** |
| Cost on Spheron B200 ($2.25/h) | ~$77 | **~$207** |

**Why so much slower than projected**:
1. **Blackwell sm_100 BF16-dense kernels in NeMo-RL/vLLM are immature**. The B200's headline 2250 TFLOPS is for FP4 (with sparsity), the figure that's actually quoted on most spec pages. For dense BF16 the effective figure is closer to ~1100 TFLOPS, and at small batch sizes (our micro=2 forward pass) kernel-launch overhead dominates anyway.
2. **TP=1 path is unchunked**. The same `model_utils.py:1378` fp32 logit cast that bit A100 still bites B200; we just had memory headroom to hide it, not compute headroom to outrun it.
3. **vLLM 0.17 colocation on Blackwell** wasn't a first-class target during the 0.17 release window; some kernels fall back to less-optimized paths.

**What this means for the H100 / H200 / A100 projections in v2**: Hopper has had a mature CUDA stack for 18+ months and the BF16-dense kernels are well-tuned. Those numbers should be ~accurate, but I'm applying a 10-15% conservative haircut anyway since they were anchored on the same v2 model that overshot on B200.

## 1. Live anchor — b200-a3 trajectory (now frozen at crash)

Run died at step 56 / 9.29 h on Spheron 1× B200 with `state=crashed` (post-mortem pending; not the metric_name bug from a1, not the [Not Found] misdiagnosis from a2). The trajectory through step 55 is the cleanest single-GPU anchor we have:

| Window | Mean min/step | Phase |
|---|---:|---|
| Steps 1-5 | 17.4 | warmup + long rollouts (resp_len ~1400, truncation ~60%) |
| Steps 10-17 | 8.0 | "shrink" phase (resp_len → 940, reward 0.10 → 0.15) |
| Steps 18-30 | 7.6 | gen-len bottom (resp_len ~1000, reward 0.13 → 0.17) |
| **Steps 38-55** | **8.9** | **steady-state anchor** (resp_len ~1050, reward 0.13-0.20, truncation 4-9%) |
| Step 55 final | 8.0 | reward 0.16, kl_penalty 0.071, grad_norm 0.66 |

Step 55 phase breakdown (1× B200, micro=2 already on, seq=8192):

| Phase | Time / step | Share | Scales with |
|---|---:|---:|---|
| `policy_training` | ~5.0 min | **~60%** | BF16 TFLOPS + mem-BW |
| `policy_and_reference_logprobs` | ~1.5 min | ~17% | mem-BW (fp32 cast `[B,S,V]`) |
| `generation` (vLLM async) | ~1.9 min | ~21% | mem-BW; share is bigger on B200 than A100 because compute-bound train shrunk more than mem-bound gen |
| **Total (steady)** | **~8.9 min** | — | |

**Phase share is stable at 60/21/17 (train/gen/logprob).** That's different from v2's 68/12/20 mix because we're now running micro=2 (which shrinks train) and Blackwell happens to under-realize on compute (which prevents train from shrinking *more*).

a4 relaunched 2026-05-15 ~11:18 UTC on 1× H200 Spheron with persistent volume; transitioned to prod 2026-05-15 ~14:57 UTC. Trajectory too early to anchor on yet.

## 2. Accelerator spec reference

Headline FP16/BF16 dense throughput (no sparsity), memory, interconnect. Vendor numbers; what we *realize* on the M5.1 workload is in §3.

| Accelerator | Arch | Memory | Mem BW | BF16 dense (TFLOPS) | Interconnect | TDP | M5.1 verdict |
|-------------|------|--------|--------|---------------------|--------------|-----|--------------|
| RTX 4090 | Ada | 24 GB | 1.0 TB/s | ~165 | PCIe 4 (no NVLink) | 450 W | Too small |
| A100 40 GB | Ampere | 40 GB HBM2e | 1.55 TB/s | 312 | NVLink 3 (600 GB/s) | 400 W | Too small (seq=8192 OOM) |
| **A100 80 GB** | Ampere | 80 GB HBM2e | 2.0 TB/s | 312 | NVLink 3 (600 GB/s) | 400 W | **v2 baseline 23.7 min/step** |
| H100 PCIe | Hopper | 80 GB HBM3 | 2.0 TB/s | ~756 | PCIe 5 / opt. NVLink bridge | 350 W | ~12.6 min/step; ~15% slower than SXM |
| H100 SXM | Hopper | 80 GB HBM3 | 3.35 TB/s | 989 | NVLink 4 (900 GB/s) | 700 W | ~10.8 min/step; micro=1 forced (80 GB) |
| **H200 SXM** | Hopper | 141 GB HBM3e | 4.8 TB/s | 989 | NVLink 4 (900 GB/s) | 700 W | **~7-8 min/step; micro=2 unlocked** |
| **B200** (Blackwell) | Blackwell | 192 GB HBM3e (180 GB usable) | 8 TB/s | ~1100 (BF16 dense; the 2250 figure is FP4) | NVLink 5 (1.8 TB/s) | 1000 W | **~8.9 min/step measured**; kernel maturity gap |
| GH200 | Hopper + Grace | 96 GB HBM3 + 480 GB LPDDR5X | 4 TB/s | 989 | NVLink-C2C 900 GB/s | 1000 W | ARM CPU; **skip** (our stack is x86) |
| MI300X | CDNA 3 | 192 GB HBM3 | 5.3 TB/s | ~1300 | Infinity Fabric (896 GB/s) | 750 W | ROCm + NeMo-RL + Qwen3.5 hybrid uncharted; **skip** |
| RTX PRO 6000 Blackwell | Blackwell workstation | 96 GB GDDR7 | **1.79 TB/s** | — | PCIe 5 | 600 W | mem-BW *worse than A100*; **skip** |

## 3. M5.1 wall-clock + cost estimates by hardware (recalibrated)

Math anchored on the **8.9 min/step live B200 measurement** for Blackwell, and v2's per-system speedup math (with a 10-15% conservative haircut) for Hopper/Ampere since their kernel maturity is established. Multi-GPU scaling uses standard FSDP/DTensor DP scaling (~1.7-1.9× per added GPU at this model size) and chunked-logprob TP-path unlock.

| Config | Per-step (steady) | Total run (622 steps) | Code work | Risk |
|---|---:|---:|---|---|
| 1× A100-80GB (v2 anchor) | ~23.7 min | ~245 h (10.2 d) | none | live previously |
| 1× H100 PCIe | ~12.6 min | **~131 h (5.5 d)** | none | clean |
| **1× H100-80GB SXM** | **~10.8 min** | **~112 h (4.7 d)** | none | clean; micro=1 forced |
| **1× H200-141GB** | **~7.5 min** | **~78 h (3.2 d)** | yaml `micro=2` | clean |
| **1× B200-192GB (measured)** | **~8.9 min** | **~92 h (3.8 d)** | yaml `micro=2`; sm_100 surprises | image-compat + kernel maturity |
| 2× A100-80GB (TP=2) | ~13.0 min | ~135 h | TP=2 yaml + smoke | moderate |
| **2× H100 SXM (TP=2)** | **~5.5 min** | **~57 h (2.4 d)** | TP=2 yaml + M4 byte-exact smoke | moderate |
| **2× H200 (TP=2)** | **~4.2 min** | **~44 h (1.8 d)** | TP=2 yaml + smoke | moderate |
| 2× B200 (TP=2) | ~5.0 min (extrapolation) | ~52 h | TP=2 yaml + Blackwell smoke | image-compat |
| 4× A100-80GB (TP=2 + DP=2) | ~6.5 min | ~67 h (2.8 d) | TP=2+DP=2 yaml + smoke | moderate |
| 4× H100 SXM (TP=2 + DP=2) | ~2.8 min | ~29 h | larger reconfig | high |
| **8× H200 SXM5 (DP=8)** | **~1.5-2.5 min** | **~16-26 h (0.7-1.1 d)** | DP=8 yaml + smoke | moderate (DP=8 untested on this codebase) |

**Where the v3 numbers diverge from v2**:
- v2 said 1× B200 = 2.2 min/step / 34 h. **v3 measured 8.9 min/step / 92 h.** A ~4× correction.
- v2 said 2× H100 TP=2 = 4.0 min/step / 41 h. **v3 = 5.5 min/step / 57 h.** Hopper kernels are mature, so most of v2 holds; the haircut covers our optimism on TP=2 NVLink scaling.
- v2 had no 2× H200 / 8× H200 entries; v3 adds them based on the Spheron screenshot offers.

### 3.1 The 60% train-share matters for hardware choice

Phase share at the b200 anchor is **60% train / 21% gen / 17% logprob**. The hardware levers that move the needle:

- **Compute (BF16 dense TFLOPS)** drives the 60% train phase. H200 = H100 (989 TFLOPS); B200 dense ≈ 1100 TFLOPS but kernel-immature.
- **Memory bandwidth** drives the 17% logprob phase (fp32 cast on `[B, S, V=248K]`) and the 21% gen phase. H200 = 4.8 TB/s, H100 = 3.35 TB/s, A100 = 2.0 TB/s, B200 = 8 TB/s. **H200's mem-BW advantage over H100 is real and visible in the wall-clock.**
- **Aggregate at TP=2/DP=2**: 2× H200 = 9.6 TB/s aggregate, which actually beats 1× B200's 8 TB/s. This is why 2× H200 TP=2 lands at 4.2 min/step (faster than 1× B200's 8.9), not slower.

### 3.2 The 8× H200 SXM5 Spheron offer (live screenshot)

Spheron offers a "CONFIGURATION LOCKED" 8× H200 SXM5 instance: **$15.15/h = $1.89/GPU/h**, 128 vCPU, 1.6 TB RAM, 512 GB storage, NVLink enabled, EU North 1, Ubuntu 22.04. Per-GPU rate is competitive (vs RunPod Community single H200 at $3.99/h, Lambda doesn't sell single H200, CoreWeave HGX H200 = 8-pod only at $6.30/GPU/h).

For Qwen3.5-0.8B the right topology is **TP=1 + DP=8** (each GPU runs full model + colocated vLLM on its slice of 40 trajectories per step). TP>2 on 0.8B has diminishing returns (communication dominates tiny per-shard compute).

Expected scaling:

| DP scaling | Per-step | 622 steps (2 epochs) | 311 steps (1 epoch) |
|---|---:|---:|---:|
| Optimistic (6.5×) | ~1.4 min | **~15 h** | **~7.5 h** |
| Realistic (5×) | ~1.8 min | **~19 h** | **~9.5 h** |
| Conservative (3.5×) | ~2.5 min | **~26 h** | **~13 h** |
| Pessimistic (mostly using 2 of 8) | ~5 min | ~52 h | ~26 h |

Cost at $15.15/h: realistic 2-epoch run = **~$288**; realistic 1-epoch = **~$144**. Plus ~$30 first-launch debug overhead.

**When 8× H200 is the right call**:
- Both 2 epochs AND finish-in-one-workday (≤12h goal). Only single-box config that fits.
- Multiple back-to-back ablations on the same rental (M5.5 + M5.6 + M5.2 sweep). Box pays for itself across runs.
- Scaling trajectories-per-step back up toward paper's 256 (memory headroom on 8× 141 GB = 1.1 TB makes 4× the per-step batch trivial).

**When 8× H200 is overkill**:
- Single experiment with no immediate follow-ups queued. **2× H200 TP=2 at $3.12/h is the right size** for a 0.8B model — ~2 day wall, ~$140.

**Three risks before clicking rent on 8× H200**:
1. DP=8 has never been smoke-tested on this codebase; Ray-orchestration + colocated-vLLM × 8 may surface bugs. Budget 2-3 h for first launch to be debug-heavy.
2. EU North 1 region may add 20-30 min for first HF / index downloads if your HF repos are US-East.
3. $15.15/h compounds fast on debug time. Smoke a 10-step run before launching the full 2-epoch.

### 3.3 Two-GPU comparison (the user's question from 2026-05-15)

For "I want a step up from 1× B200 but not an 8-GPU box", the three sensible options:

| Config | Per-GPU specs | Aggregate (TP=2) | Wall (622 steps) | Cost @ Spheron rate |
|---|---|---|---:|---:|
| 2× H100 SXM | 989 TFLOPS, 3.35 TB/s, 80 GB | 1978 TFLOPS, 6.7 TB/s, 160 GB | ~57 h | $2.66/h × 57 = **~$152** |
| **2× H200 SXM** | **989 TFLOPS, 4.8 TB/s, 141 GB** | **1978 TFLOPS, 9.6 TB/s, 282 GB** | **~44 h** | **$3.12/h × 44 = ~$137** |
| 4× A100 80GB | 312 TFLOPS, 2.0 TB/s, 80 GB | 1248 TFLOPS, 8.0 TB/s, 320 GB | ~67 h | $2.88/h × 67 = **~$193** |

**2× H200 dominates on both wall-clock and $/run.** 9.6 TB/s aggregate mem-BW beats 1× B200's 8 TB/s; 141 GB per GPU means `micro=4` + `gpu_mem_util=0.92` + drop `activation_checkpointing` are all trivial. Mature Hopper kernels (no Blackwell sm_100 surprises). The 2× H100 path is the same compute but 30% less aggregate mem-BW — slower by ~25%. 4× A100 has the most VRAM but 37% less compute — wall-clock takes the hit.

## 4. Provider pricing (verified May 2026)

Rows marked ✓ verified directly from provider pricing page in May 2026; ⓘ from user screenshots; ⚠ third-party.

### H200 SXM5 (141 GB)

| Provider | $/GPU/h | Units | Verified | Notes |
|---|---:|---:|---|---|
| **Spheron** | **$1.56/h** (single) | 1 | ✓ [spheron.network/pricing](https://www.spheron.network/pricing/) | Cheapest verified single H200 |
| **Spheron 8× H200 SXM5 (locked)** | **$1.89/h** | 8 (locked) | ⓘ user screenshot | $15.15/h pod; NVLink enabled; 1.6 TB RAM; 512 GB storage; EU North 1 |
| RunPod (Community) | $3.99/h | 1 | ⓘ user screenshot | T2, "Low" availability |
| Lambda Labs | — | — | ✓ verified absent | No single-GPU H200 on the page |
| Jarvislabs | $3.80/h | 1 | ⚠ third-party | Not verified |
| CoreWeave HGX H200 | $6.30/GPU/h | 8 (pod only) | ✓ pricing page | No single-GPU SKU |

### H100 SXM (80 GB)

| Provider | $/GPU/h | Units | Verified | Notes |
|---|---:|---:|---|---|
| **Spheron** | **$1.33/h** | 1 | ✓ [spheron.network/pricing](https://www.spheron.network/pricing/) | Cheapest verified |
| RunPod (Community) | $2.99/h | 1 | ⓘ user screenshot | T2 |
| Lambda Labs | $4.29/h | 1 | ✓ [lambda.ai/pricing](https://lambda.ai/pricing) | T1 premium |
| Vast.ai (spot lows) | $1.49-1.87/h | 1 | ⚠ third-party | Marketplace |

### B200 (192 GB)

Carrying forward from v2 since the offers haven't moved. **Cost-per-run on B200 is now 4× higher than v2 suggested** because wall-clock is 4× longer.

| Provider | $/GPU/h | Verified | v3 cost for 1× B200 622-step run (~92h) |
|---|---:|---|---:|
| Spheron | $2.25/h | ✓ | ~$207 |
| Lambda Labs | $6.99/h | ✓ | ~$643 |
| RunPod Community | $5.49/h | ⓘ | ~$505 |

### A100 80GB

| Provider | $/GPU/h | Verified | Notes |
|---|---:|---|---|
| Spheron | $0.72/h | ✓ | Fallback if cheaper Hopper unavailable |
| Vast.ai | $1.20/h | ⓘ | v2 baseline rate |
| Thundercompute | $0.78/h | ✓ | PCIe + virtualized; not for prod |

## 5. Cost-per-run table — recalibrated (May 2026)

Ordered cheapest → most expensive total run, using v3 wall-clock estimates. **Sunk cost on b200-a3 = ~$21 (9.3 h × $2.25/h Spheron)** before the crash; new launch starts fresh.

| Config | Wall | Provider | $/h | **Cost (run)** | Δ vs A100-fresh ($294) |
|---|---:|---|---:|---:|---:|
| **2× H200 TP=2 (Spheron)** | ~44 h | Spheron 2× $1.56 | **$3.12** | **~$137** | **−$157, save 8.4 d** |
| 2× H100 TP=2 (Spheron) | ~57 h | Spheron 2× $1.33 | $2.66 | ~$152 | −$142, save 7.8 d |
| **8× H200 DP=8 (Spheron, realistic 5× scaling, 2 epochs)** | **~19 h** | Spheron 8× $1.89 | **$15.15** | **~$288** | **−$6, save 9.4 d** |
| 8× H200 DP=8 (Spheron, 1 epoch realistic) | ~9.5 h | Spheron | $15.15 | ~$144 | −$150, save 9.8 d but only 1 epoch |
| 1× H200 (Spheron) | ~78 h | Spheron | $1.56 | ~$122 | −$172, save 7.0 d |
| 1× H100 SXM (Spheron) | ~112 h | Spheron | $1.33 | ~$149 | −$145, save 5.5 d |
| 1× B200 (Spheron) | ~92 h | Spheron | $2.25 | ~$207 | −$87, save 6.4 d |
| 4× A100 TP=2+DP=2 (Spheron) | ~67 h | Spheron 4× $0.72 | $2.88 | ~$193 | −$101, save 7.4 d |
| 2× A100 TP=2 | ~135 h | Spheron 2× $0.72 | $1.44 | ~$194 | −$100, save 4.6 d |
| 1× A100 fresh on Spheron | ~245 h | Spheron | $0.72 | ~$176 | −$118, same wall as Vast A100 |

**Top 3 Pareto winners (v3)**:

1. **🥇 2× H200 TP=2 on Spheron — ~$137 / ~44 h.** Best $/run AND second-fastest single-box wall-clock under 2 days. Hopper kernel maturity, no Blackwell surprises. **Recommended for one-shot M5.1 runs.**
2. **🥈 8× H200 SXM5 DP=8 on Spheron — ~$288 / ~19 h (2 epochs realistic).** Only single-box config that fits 2 epochs in <1 day. Right call if multiple ablations are queued behind. DP=8 is untested on the codebase; smoke first.
3. **🥉 2× H100 TP=2 on Spheron — ~$152 / ~57 h.** Slightly cheaper hardware than 2× H200, but mem-BW gap costs ~30% on wall. Still beats single B200.

**Stay-on-1× B200 is now Pareto-dominated** by both 2× H200 and 2× H100 paths after the calibration shock.

## 6. Recommendation (post-calibration, post-a3-crash)

The live a4 run on 1× H200 Spheron is fine for now (smoke just transitioned to prod ~2026-05-15 14:57 UTC). If it survives to step 50 ckpt and the recipe is stable, **don't migrate** — single-H200 wall at ~78h is acceptable for one paper-faithful run.

**For the next new launch (M5.5 or M5.6 ablations after thesis)**:
1. **Default**: 2× H200 TP=2 on Spheron. Cheapest, fastest single-box, mature kernel stack.
2. **If multiple ablations queue back-to-back**: 8× H200 SXM5 DP=8 on Spheron. Worth the $15.15/h *only* if 3+ runs are queued. Smoke DP=8 for one hour before committing.
3. **If on a free academic grant (SURF Snellius `gpu_h100`)**: that's the right call — see [`COMPUTE_GRANTS.md`](COMPUTE_GRANTS.md). 4× H100 nodes are natively available; TP=2 + DP=2 on a single node is ~29 h wall and free.

**Skip**:
- 1× B200 (anywhere) for paper-faithful 2-epoch runs. Kernel-maturity gap doubles cost vs 2× H200.
- LUMI (AMD) — ROCm + NeMo-RL + Qwen3.5 hybrid is uncharted.
- RTX PRO 6000 — mem-BW worse than A100; workstation thermals not for multi-day.
- GH200 — ARM stack rebuild is ≥1 week.
- 40 GiB A100 anywhere (MeluXina, Karolina, Vega, Discoverer on EuroHPC; Lambda single A100) — won't fit M5.1 prod yaml.

## 7. What I'd NOT do (mostly unchanged from v2, plus B200 caveat)

- **Don't pick 1× B200 over 2× H200 just because B200 sounds newer.** The calibration shock is real. B200's FP4 + sparsity edges don't apply to our BF16-dense small-batch workload.
- **Don't pick H100 PCIe over SXM for the same provider.** 15% slower at ≥50% of SXM price.
- **Don't extrapolate from v2 wall-clock numbers.** Use the v3 anchors in §3 (Hopper holds, Blackwell ~4× corrected).
- **Don't kill a4 mid-run** unless it's clearly broken. The cost calculus has changed; staying on H200 is fine.

## 8. Open questions before each launch type

- **For 2× H200 TP=2**: does the M4 byte-exact prompt check still pass at TP=2? Rollout shape doesn't change but the rendering path goes through a different code branch. Smoke validates this in ~30 min.
- **For 8× H200 DP=8**: never smoke-tested on this codebase. Ray-orchestration on 8 workers, colocated vLLM × 8, gradient allreduce at 0.8B-scale all need a 10-step verification. Budget 1-2 h.
- **For SURF / EuroHPC**: queue wait on `gpu_h100` (SURF) vs Leonardo (EuroHPC) is unknown. Multi-day waits eat into SBU-effective throughput. See [`COMPUTE_GRANTS.md`](COMPUTE_GRANTS.md).

## 9. Pointers

- v2 archived: [`../archive/HARDWARE_COMPARISON_v2.md`](../archive/HARDWARE_COMPARISON_v2.md)
- v1 archived: [`../archive/HARDWARE_COMPARISON_v1.md`](../archive/HARDWARE_COMPARISON_v1.md)
- v0 archived: [`../archive/HARDWARE_COMPARISON_v0.md`](../archive/HARDWARE_COMPARISON_v0.md)
- 4090 dev-box historical: [`HARDWARE_4090.md`](HARDWARE_4090.md)
- Spheron H200 runbook: [`../spheron/SETUP_SPHERON.md`](../spheron/SETUP_SPHERON.md)
- Grants routes (SURF + EuroHPC): [`COMPUTE_GRANTS.md`](COMPUTE_GRANTS.md)
- b200-a3 W&B run (anchor): [reason_over_search_b200/h68uskz6](https://wandb.ai/gaurisankarj1996-leiden-university/reason_over_search_b200/runs/h68uskz6) — crashed at step 56 / 9.29 h
- Live a4 status: [`../report/RESULTS_M5_1_H200.md`](../report/RESULTS_M5_1_H200.md)
- M5.1 milestone: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md)
- Paper-vs-ours mapping: [`../milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md)
- Runtime-efficiency menu: [`../research/RUNTIME_EFFICIENCY.md`](../research/RUNTIME_EFFICIENCY.md)

## 10. Sources (verified May 2026)

- [Vast.ai GPU Pricing](https://vast.ai/pricing)
- [Spheron pricing](https://www.spheron.network/pricing/)
- [RunPod Pricing](https://www.runpod.io/pricing)
- [Lambda Labs Pricing](https://lambda.ai/pricing)
- [Thundercompute Pricing](https://www.thundercompute.com/pricing)
- [CoreWeave Pricing](https://www.coreweave.com/pricing)
- [B200 spec — Snel](https://www.snel.com/managed-servers/gpu-servers/nvidia-b200-gpu/) — 192 GB total / 180 GB usable
- [B200 spec — Jarvislabs](https://jarvislabs.ai/gpu/nvidia-b200)
- [B200 vs H100 — Vast](https://vast.ai/article/nvidia-h200-vs-b200-comparing-datacenter-grade-accelerators)
- [B200 vs H200 — Northflank](https://northflank.com/blog/b200-vs-h200)
- [GPU Cloud Pricing Comparison 2026 — Spheron](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/)
- [H100 Rental Prices — IntuitionLabs](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [H100 vs A100 BF16 throughput — Spheron](https://www.spheron.network/blog/nvidia-a100-vs-h100/)
