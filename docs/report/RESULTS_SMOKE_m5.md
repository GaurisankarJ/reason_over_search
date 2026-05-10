---
title: Results M5 — smoke iteration log (Qwen3.5-0.8B GRPO training, NeMo-RL)
tags: [report, training, m5, m5.1, qwen3.5, smoke]
source: internal
created: 2026-05-09
updated: 2026-05-10
---

# Results M5 smoke — iteration log

Live record of smoke iterations on `training_m5_1/`. Pipeline: [`training_m5_1/`](../../training_m5_1/). Milestone: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md). Wall-clock anchor: [`../training/SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md) (M2 Qwen3.5-2B at 57 s/step on 1× A100-80GB).

## 1. Run roster

| Version | Phase | When | Pipeline / config change |
|---|---|---|---|
| v1-v3 | M5 smoke — exploratory | 2026-05-10 | Scaffold + memory/IPC iterations; failed (1: nv-grouped-gemm CUDA init, 2: /tmp exhaustion, 3: v1 DTensor hardcodes `model.model.layers` which Qwen3.5 lacks). |
| v4 | M5 smoke — v2 worker | 2026-05-10 | v2 DTensor venv extracted; step 1 succeeded (145.58s), step 2 OOM by 0.31 GB at `log_softmax` (micro=4 × seq=4096 × vocab=248,320 × 2 B = 8.14 GB). |
| v5 | M5 smoke — alloc env | 2026-05-10 | Tried `PYTORCH_ALLOC_CONF=expandable_segments` → broke CUDA-IPC weight share (`pidfd_getfd: Operation not permitted`). Reverted. |
| **v6** | **M5 smoke — Group C + memory** | **2026-05-10** | **All 10 steps clean.** micro=4→2, `vllm_cfg.gpu_memory_utilization=0.7→0.5`. Group-C paper-faithful flips applied (LOO off, warmup 0, max_obs_chars 1024, max_new_tokens 1024). |
| v7 | M5.2 baseline | 2026-05-10 | Production shape × 10 steps; measures true per-step at `m5_1_research_paper.yaml` config (no system gains). |
| v8+ | M5.2 system-gain iterations | 2026-05-10 | One lever per smoke: R1 prefix-cache, O1 fused AdamW, R2 async vLLM, O5 torch.compile, R4 gpu_mem_util bump. |

All runs: 1× A100-80GB on Vast (`pantomiman/reason-over-search-v1:v1`), `Qwen/Qwen3.5-0.8B`, MuSiQue train, IVF-SQ8 retriever × 8 workers, vLLM bf16, seed 42.

## 2. v6 — M5 smoke (pipeline validation, smoke shape) — SUCCESS

### 2.1 Config

[`training_m5_1/configs/m5_smoke.yaml`](../../training_m5_1/configs/m5_smoke.yaml):
- `num_prompts_per_step=4`, `num_generations_per_prompt=5` → **20 traj/step**
- `max_num_steps=10`, `max_rollout_turns=10`
- `max_total_sequence_length=4096`, `max_new_tokens=1024`
- `train_micro_batch_size=2`, `logprob_batch_size=2`
- `vllm_cfg.gpu_memory_utilization=0.5`
- `use_leave_one_out_baseline=false` (Group C, paper default)
- LR warmup: 0 (Group C; LinearLR `total_iters=1` start=end=1.0 → no-op)
- `dtensor_cfg._v2=true`, `enable_thinking=true`

### 2.2 Per-step timing

| Step | Total | policy_training | logprobs | generation | prepare | transfer |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 111.10 | 56.17 (50.6%) | 19.69 (17.7%) | 25.38 (22.8%) | 5.88 (5.3%) | 0.66 |
| 2 | 94.76 | 53.92 (56.9%) | 15.90 (16.8%) | 8.99 (9.5%) | 9.57 (10.1%) | 0.48 |
| 3 | 105.94 | 58.60 (55.3%) | 21.08 (19.9%) | 9.85 (9.3%) | 10.62 (10.0%) | 0.38 |
| 4 | 96.57 | 53.24 (55.1%) | 19.46 (20.2%) | 11.63 (12.0%) | 7.21 (7.5%) | 0.37 |
| 5 | 83.96 | 47.64 (56.7%) | 16.74 (19.9%) | 9.11 (10.9%) | 5.19 (6.2%) | 0.37 |
| 6 | 90.56 | 52.59 (58.1%) | 17.88 (19.7%) | 10.33 (11.4%) | 5.07 (5.6%) | 0.37 |
| 7 | 79.27 | 47.98 (60.5%) | 14.11 (17.8%) | 8.46 (10.7%) | 4.80 (6.1%) | 0.39 |
| 8 | 92.97 | 55.95 (60.2%) | 16.92 (18.2%) | 10.92 (11.7%) | 4.84 (5.2%) | 0.39 |
| 9 | 98.66 | 61.55 (62.4%) | 17.12 (17.4%) | 9.24 (9.4%) | 6.40 (6.5%) | 0.56 |
| 10 | 95.16 | 57.62 (60.5%) | 17.03 (17.9%) | 10.50 (11.0%) | 4.59 (4.8%) | 0.42 |

**Mean (steps 2–10, ex-warmup): 93.1 s/step**. Step 1 includes vLLM engine init + first weight-update warmup; the 22.8% generation share there is one-off.

### 2.3 Phase shares (mean of steps 2-10)

| Phase | Time | Share |
|---|---:|---:|
| `policy_training` | 54.3 s | **58.4%** |
| `policy_and_reference_logprobs` | 17.4 s | 18.7% |
| `generation` (vLLM rollout) | 9.9 s | 10.6% |
| `prepare_for_generation/total` | 6.5 s | 7.0% |
| `transfer_and_update_weights` | 0.4 s | 0.5% |

**Training-dominated** at smoke shape (small generation batch fits in vLLM continuous batch). Note this reverses at production shape: 64 prompts × G=5 × seq=8192 means rollout's KV-cache + per-turn prefill grows ~32× while training scales linearly.

### 2.4 Other metrics (smoke; not learning signal)

| Metric | Value |
|---|---:|
| Avg reward (mean of 10 steps) | 0.011 |
| Mean generation length (mean of 10 steps) | 692.7 tokens |
| Loss (final step) | step 1 = 0.0735; reset each step |
| Generation KL error | 0.0006 (step 1) |
| Peak VRAM | ~71 GB (vLLM at 0.5 mem_util + training peak) |

Reward stays low (0/0/0/0/0.051/0/0/0.033/0/0.017) because 10 steps × 4 prompts = 40 unique prompts is far below the learning floor. Smoke is for pipeline validation only.

### 2.5 What we learned

- v2 DTensor + nemo_automodel handles Qwen3.5 hybrid architecture; v1's hard-coded `model.model.layers` does not.
- `PYTORCH_ALLOC_CONF=expandable_segments` is incompatible with NeMo-RL's CUDA-IPC weight share (`pidfd_getfd`). Stay on the default allocator.
- log_softmax peak (`batch × seq × vocab × 2 B`) drives logprob memory; `train_micro_batch_size=2` is the sweet spot at seq=4096 on 1× A100.
- `/tmp` must point to a roomy mount: mamba-ssm + flash-attn + causal-conv1d compile temps blew up 30 GB. We now export `TMPDIR=/workspace/tmp_build RAY_TMPDIR=/workspace/ray_tmp TORCHINDUCTOR_CACHE_DIR=/workspace/torchinductor_cache`.

---

## 3. v7 — M5.2 baseline (production-shape, no system gains) — IN PROGRESS

### 3.1 Config

[`training_m5_1/configs/m5_1_research_paper.yaml`](../../training_m5_1/configs/m5_1_research_paper.yaml), overridden to 10 steps for measurement:

- `num_prompts_per_step=64`, `num_generations_per_prompt=5` → **320 traj/step**
- `max_num_steps=10` (override)
- `max_total_sequence_length=8192`, `max_new_tokens=1024`
- `train_micro_batch_size=2`, `logprob_batch_size=2`
- `vllm_cfg.gpu_memory_utilization=0.5`
- Everything else: as smoke v6.

### 3.2 Result — TODO

Filled by the M5.2 baseline smoke. Becomes the per-step anchor for Phase 1 projection.

---

## 4. Phase 1 wall-clock projection — paper-faithful, no system gains

### 4.1 Methodology

Project from **smoke v6 per-step time × scale factor** while a measured production-shape baseline (v7) is in flight. Two sets of numbers:

1. **Naive linear scaling** from smoke v6 — gives an upper bound (overestimates because larger batch improves GPU utilization).
2. **Measured from v7** — replaces the naive scaling once v7 lands.

### 4.2 Scale factors (smoke v6 → production)

| Phase | Smoke v6 (s) | Linear scale | Production (s) | Adjustment | Realistic (s) |
|---|---:|---|---:|---|---:|
| `policy_training` | 54.3 | × 16 (traj) × 2 (seq) = 32 | 1738 | larger batch → ~1.4× GPU util gain | ~1240 |
| `logprobs` | 17.4 | × 32 | 557 | same | ~400 |
| `generation` (vLLM) | 9.9 | × 8-16 (vLLM continuous batching sublinear) | 80-160 | mostly KV-bound | ~120 |
| `prepare` | 6.5 | × ~1.5 | ~10 | fixed costs | ~10 |
| **Total** | **93.1** | | **~2300** | | **~1770 (~29 min)** |

### 4.3 Projection — paper-faithful M5.1 full run

| Quantity | Naive scaling | Realistic | Source |
|---|---:|---:|---|
| Per-step wall-clock | ~38 min | **~29 min** | §4.2 |
| Schedule | 622 steps (2 epochs × 19,938 prompts / 64) | 622 steps | [`m5_1_research_paper.yaml:43-49`](../../training_m5_1/configs/m5_1_research_paper.yaml#L43-L49) |
| Total wall-clock | ~393 h (16.4 d) | **~300 h (12.5 d)** | per-step × steps |
| Cost @ \$1.50/h on Vast 1× A100 | ~\$590 | **~\$450** | Vast pricing |

**Caveat: these are pre-measurement estimates.** The v7 baseline smoke replaces these numbers with measured per-step time. The realistic column assumes typical 1.4× GPU-utilization gain from bigger batches, which has not been verified at this scale.

### 4.4 Sensitivity to `num_prompts_per_step`

At fixed 2 epochs of MuSiQue, total trajectories = 199,380 regardless of batch size. **Wall-clock is fundamentally throughput-limited by 1× A100 token-bandwidth**, so smaller-batch options don't save time — they just trade per-step wall for more steps:

| `num_prompts_per_step` | Steps | Per-step est. | Total est. |
|---:|---:|---:|---:|
| 32 | 1,246 | ~16 min | ~330 h (13.7 d) |
| **64 (locked)** | **622** | **~29 min** | **~300 h (12.5 d)** |
| 128 | 312 | ~52 min | ~270 h (11.3 d) |
| 256 (paper) | 156 | OOMs (>80 GB) | n/a on 1× A100 |

128 is marginally faster (better GPU util) but doubles the logprob memory peak and risks OOM. **64 is the safer pick** until v7 measures real headroom.

---

## 5. Phase 2 — M5.2 system gains plan

Iterate one lever at a time against the v7 baseline. Each iteration = 10-step smoke at production shape; record per-step delta. Levers chosen from [`../research/RUNTIME_EFFICIENCY.md`](../research/RUNTIME_EFFICIENCY.md), filtered to those that **don't change training math or paper-faithful values**:

| Iter | Lever | Source | Expected delta | Risk |
|---|---|---|---:|---|
| **v7** | (baseline, no gains) | — | — | — |
| v8 | R1: vLLM `enable_prefix_caching: true` | RUNTIME_EFFICIENCY §R1 | 1.5–2.0× on rollout (~5% step) | very low |
| v9 | O1: AdamW `fused: true` | RUNTIME_EFFICIENCY §O1 | 1.03–1.08× on step | none |
| v10 | R2: vLLM `async_engine: true` | RUNTIME_EFFICIENCY §R2 | 1.3–1.5× on rollout | low |
| v11 | O5: `torch.compile` on policy | RUNTIME_EFFICIENCY §O5 | 1.10–1.25× on train (60% of step!) | 5–10 min compile cost |
| v12 | R4: bump `gpu_memory_utilization` 0.5 → 0.6 | RUNTIME_EFFICIENCY §R4 | small on rollout; enables R5 | OOM (touchy under colocation) |
| v13 | Stack winners | — | combined | re-validate non-OOM |

**Skipped per user directive 2026-05-10**:
- G3 (G=5 → 4): touches GRPO group baseline statistics — non-paper-faithful.
- M1 (max_total_sequence_length 4096 → 3072): increases truncation rate.
- G1/G2 (drop KL / DAPO dynamic sampling): touches training math.
- O3 LoRA: capacity ceiling at <2B per Plasticity-vs-Rigidity.
- R3 async GRPO, R7 EAGLE-3, O2 8-bit AdamW, O4 drop activation ckpt, O6 sequence packing: PR-level effort or blocked.
- C1 decolocate: requires 2× A100.

### 5.1 Phase 2 expected combined speedup

Conservative compound (lever effects compose but on different phases):
- R1 + O1 + O5 stacked: ~1.15–1.25× on full step (training-side dominates).
- + R2: ~1.20–1.35× on full step.
- + R4/R5 if headroom: ~1.25–1.45×.

**Target: cut 12.5 d → 9–10 d on 1× A100.**

---

## 6. Phase 2 results — TODO

Filled as the iterations run.

| Iter | Mean s/step (steps 2-10) | Δ vs baseline | Notes |
|---:|---:|---:|---|
| v7 (baseline) | TODO | — | |
| v8 (R1 prefix) | TODO | TODO | |
| v9 (R1+O1) | TODO | TODO | |
| v10 (R1+O1+R2) | TODO | TODO | |
| v11 (R1+O1+R2+O5) | TODO | TODO | |
| v12 (R1+O1+R2+O5+R4) | TODO | TODO | |

### 6.1 Final projection — paper-faithful + best system gains

| Quantity | Value |
|---|---:|
| Per-step wall-clock (M5.2 winner) | TODO |
| Schedule | 622 steps |
| Total wall-clock | TODO |
| Cost @ \$1.50/h | TODO |

---

## 7. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-09 | M5 starts in NeMo-RL, not verl | Verl does not support Qwen3.5 (M2 finding); patching is more work than porting the recipe knob-by-knob. |
| 2026-05-09 | M5 train rollout byte-aligned to M4 eval rollout | Avoids M3-style 14-fix train/eval-drift audit by construction. |
| 2026-05-09 | M5.1 reward = F1-only on `<answer>…</answer>` | F1 is more discriminative than EM at small scale; format-reward partial-credit floor masks the tool-use signal. |
| 2026-05-09 | M5.1 answer wrap = plain `<answer>X</answer>` (no `\boxed{}`) | Carry from M4; eval scorer accepts both shapes. |
| 2026-05-09 | M5.1 dataset = MuSiQue only | Hardest of 4 paper benchmarks; largest single-dataset headroom (M1 EM 0.124 baseline); simplest recipe. |
| 2026-05-10 | Group C resolved (LOO off, warmup 0, max_obs_chars 1024, max_new_tokens 1024, max_num_steps 10 for smoke) | Paper-faithfulness audit (see PAPER_VS_OURS_M5 §8). |
| 2026-05-10 | num_prompts_per_step = 64 for production | Memory ceiling on 1× A100-80GB at seq=8192; 256 OOMs, 32 over-pessimistic, 128 risky. 64 keeps GRPO baseline well-conditioned. |
| 2026-05-10 | M5.2 system gains restricted to non-paper levers | User directive: paper-faithful config locked, only orthogonal throughput levers. |
| 2026-05-10 | `research_v2_a` branch cut from M5.1 paper-faithful baseline (no system gains) | Preserves clean reproducer of paper-faithful state regardless of how M5.2 evolves. |

---

## 8. Pointers

- M5 milestone narrative: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md)
- M5 code-setup audit: [`CODE_SETUP_m5.md`](CODE_SETUP_m5.md)
- M5.1 paper-vs-ours mapping: [`../milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md)
- Runtime-efficiency menu (Phase 2 lever source): [`../research/RUNTIME_EFFICIENCY.md`](../research/RUNTIME_EFFICIENCY.md)
- M2 smoke wall-clock anchor: [`../training/SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md)
- ReSearch paper: [arXiv:2503.19470](https://arxiv.org/abs/2503.19470)
