---
title: Results M5 — smoke iteration log (Qwen3.5-0.8B GRPO training, NeMo-RL)
tags: [report, training, m5, m5.1, qwen3.5, smoke]
source: internal
created: 2026-05-09
updated: 2026-05-11
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

## 3. v7 — M5.2 baseline (production-shape) — MEASURED 2026-05-11

### 3.1 First attempt — OOM at step 2 (`train_micro_batch_size=2`)

Same config as smoke v6 except production shape:
- `num_prompts_per_step=64`, `G=5` → **320 traj/step**
- `max_total_sequence_length=8192` (doubled from smoke's 4096)
- `train_micro_batch_size=2`, `vllm_cfg.gpu_memory_utilization=0.5`

Step 1: **2022.67 s (33.7 min)** clean. Step 2: OOM in training's `prepare_loss_input` → `log_softmax` (tried to allocate 13.73 GiB, 13.11 GiB free; off by 0.62 GiB). Trace: `automodel/train.py:442 → forward_with_post_processing_fn` → `model_utils.py:1378` (`next_token_logits.to(torch.float32)` on full `[B,S,V] = [2,8192,248320]` = 16.3 GiB before chunking).

**Root cause**: NeMo-RL v0.6.0 TP=1 logprob path **does not chunk** the fp32 cast of logits. Only the TP>1 path (`ChunkedDistributedLogprobWithSampling`) supports memory-bounded chunking. The `logprob_chunk_size` knob is wired to `LogprobsPostProcessor` (logprob phase) but `LossPostProcessor` (training phase) calls a different path that ignores it.

**Fix paths**:
1. (Taken) `train_micro_batch_size=2 → 1`. Halves the cast peak.
2. Patch `distributed/model_utils.py:1378` to chunk the cast. Unblocks micro=2 → ~halves training wall-clock. **DEFERRED to a follow-up** (would require source mod + re-validation; not within overnight budget).

### 3.2 Second attempt — `train_micro_batch_size=1` (success)

| Step | Total | policy_training | logprobs | generation | other |
|---:|---:|---:|---:|---:|---:|
| 1 (warmup) | 3334.25 s (55.6 min) | 2457 s (73.7%) | 703 s (21.1%) | 161 s (4.8%) | ~13 s |
| 2 (steady) | 3543.32 s (59.1 min) | 2645 s (74.6%) | 754 s (21.3%) | 126 s (3.6%) | ~18 s |

**Steady-state: ~57-60 s/step at micro=1 production shape.**

Phase mix flips vs smoke: **training is now 75% of step** (was 58% at smoke). Generation drops to 4% — vLLM continuous batching scales superbly. The micro=1 → micro=2 unlock would mostly attack the dominant 75% slice.

### 3.3 What we learned (v7)

- The OOM headroom on 1× A100-80GB at seq=8192 is **0.62 GiB** — essentially zero. micro=1 is forced.
- micro=1 doubles the kernel-launch overhead in training: per-step doesn't double in compute but does balloon ~1.7× vs theoretical micro=2 estimate (29 → 58 min).
- Generation at production shape is fast (~2 min/step) — vLLM is not the bottleneck. Training is.
- A NeMo-RL patch (chunked fp32 cast in `model_utils.py:1378`) would meaningfully change M5.1 economics: 25 d → ~13 d. Deferred to a future iteration.

### 3.4 Decisions taken

- Skip v8 (separate-measurement smoke for O1+R2): predicted gain ~1-3% of step, dwarfed by the micro=1 reality. Fold both into the production yaml directly.
- Skip R4 (gpu_mem_util 0.5 → 0.6): the OOM was already 0.62 GiB into the red; no headroom.
- Production yaml updated to micro=1 + O1 (`fused: true`) + R2 (`async_engine: true`).

---

## 4. Phase 1 wall-clock projection — MEASURED from v7

### 4.1 v7 supersedes pre-measurement estimates

The smoke-v6-linear-scaling estimate (29 min/step, 12.5 d full run) assumed micro=2 would fit at production shape. It doesn't (§3.1). With micro=1 forced, real numbers are ~2× higher.

### 4.2 Measured projection — paper-faithful M5.1 full run

| Quantity | Pre-measurement estimate | **v7-measured reality** | Source |
|---|---:|---:|---|
| Per-step wall-clock | ~29 min (micro=2) | **~58 min (micro=1)** | §3.2 |
| Schedule | 622 steps (2 epochs × 19,938 prompts / 64) | 622 steps | [m5_1_research_paper.yaml](../../training_m5_1/configs/m5_1_research_paper.yaml) |
| Total wall-clock | ~300 h (12.5 d) | **~600 h (~25 d)** | per-step × steps |
| Cost @ \$1.50/h on Vast 1× A100 | ~\$450 | **~\$900** | Vast pricing |

This is the operational reality. The 12.5 d projection was based on a configuration that doesn't fit; the 25 d is what runs on the GPU we have.

### 4.3 What could change the answer

| Lever | Wall-clock impact | Effort | Status |
|---|---|---|---|
| Patch `model_utils.py:1378` to chunk fp32 cast | 25 d → ~13 d (unlocks micro=2) | NeMo-RL source mod + smoke re-validate | **Deferred** — out of overnight budget |
| 1 epoch instead of 2 (Search-R1 paper, paper-divergent) | 25 d → 12.5 d | yaml flip + decision | Deferred to user |
| 2× A100 decolocation | 25 d → ~15-18 d | new instance + yaml | Deferred |
| H100 instead of A100 | 25 d → ~12-15 d | hardware swap | Deferred |

### 4.4 Sensitivity to `num_prompts_per_step`

At fixed 2 epochs of MuSiQue, total trajectories = 199,380 regardless of batch size. **Wall-clock is fundamentally bound by total tokens processed**, so smaller batch sizes don't save time — they just trade per-step wall for more steps:

| `num_prompts_per_step` | Steps | Per-step est. (micro=1) | Total est. |
|---:|---:|---:|---:|
| 32 | 1,246 | ~29 min | ~600 h (25 d) |
| **64 (locked)** | **622** | **~58 min** | **~600 h (25 d)** |
| 128 | 312 | ~116 min | ~600 h (25 d) |
| 256 (paper) | 156 | OOMs (>80 GB) | n/a on 1× A100 |

The micro=1 ceiling dominates. 64 is locked.

---

## 5. Phase 2 — M5.2 system gains (truncated)

Original plan: 5 iterations (v8–v12), one lever per smoke, ~5h each. After v7 measured 58 min/step at micro=1, the iteration budget would consume most of the night without delivering full training. Pivoted to a single bundled decision:

| Lever | Status | Rationale |
|---|---|---|
| **O1 — fused AdamW** | **Applied in prod yaml** | Textbook-safe PyTorch shipped feature. ~1.03-1.08× on step. No iteration needed to validate. |
| **R2 — vLLM async_engine** | **Applied in prod yaml** | Shipped vLLM feature. ~1.3-1.5× on rollout — but rollout is only 4% of step at v7-measured production shape, so full-step delta is ~1-2%. Still worth taking. |
| R1 — vLLM prefix_caching | **Already on** | NeMo-RL defaults `enable_prefix_caching=true` for A100 cc≥8 (`vllm_worker.py:549-550`). Already in effect. |
| R4 — gpu_memory_utilization bump | **Skipped** | v7 OOM was 0.62 GiB into the red at the current 0.5; no headroom to give back to vLLM. |
| O5 — torch.compile | **Not available** | NeMo-RL DTensor backend doesn't expose the knob (only SGLang config has `enable_torch_compile`). Would require source mod. |
| Other levers (G1/G2/G3/M1/O3/O4/O6/R3/R7/C1) | **Skipped per user directive** | Either touch training math (paper-faithful locked), are PR-level effort, or require 2× A100. |

**Net M5.2 gain folded into production yaml: ~1-3% per-step. Saved iteration time: ~10h.**

The one lever that would meaningfully change the answer — patching NeMo-RL `model_utils.py:1378` to chunk the pre-fp32-cast logits → unlock micro=2 → halve training-phase wall-clock — is a source modification deferred to a future M5.3.

---

## 6. M5.1 production training — LIVE

### 6.1 Launch state (2026-05-11 ~01:05 UTC, pid 178440)

- Config: [m5_1_research_paper.yaml @ db0852b](../../training_m5_1/configs/m5_1_research_paper.yaml) — paper-faithful + M5.2 system gains (O1 fused AdamW, R2 vLLM async_engine) + validation disabled (no MuSiQue dev split; eval out-of-band).
- Schedule: `max_num_steps=622`, `max_num_epochs=2`. Display shows `Step N/311` (per-epoch); transition to epoch 2 at step 311.
- Checkpoint: every 50 steps to `results/grpo/m5_prod/seed42/`. First save at step 50. `train/loss/mean` metric, `keep_top_k=0` (keep all).
- W&B run: `qwen3.5-0.8b-musique-m5_prod-seed42` on project `reason_over_search_m5_1`.

### 6.2 Per-step trajectory (live, refresh as steps land)

| Step | Wall (min) | r_mean (F1) | r=0% | r=1% | tok_mean | trunc% | Loss | KL err |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 57.9 | 0.020 | 95.6% | 1.6% | 7038 | **68.4%** | 0.013 | 0.0006 |
| 2 | 55.9 | 0.071 | 90.0% | 5.3% | 6738 | 62.2% | 0.025 | 0.0006 |
| 3 | 55.8 | 0.023 | 95.0% | 1.2% | 6830 | 62.5% | 0.007 | 0.0006 |
| 4 | 52.9 | 0.073 | 86.6% | 4.1% | 6439 | 52.8% | 0.025 | 0.0006 |
| 5 | 47.4 | 0.077 | 87.8% | 4.7% | 5911 | 46.2% | -0.019 | 0.0006 |
| 6 | 38.2 | 0.095 | 80.6% | 5.3% | 5069 | 27.2% | 0.027 | 0.0006 |
| 7 | 37.3 | 0.060 | 86.6% | 2.2% | 4900 | 25.0% | 0.018 | 0.0006 |
| 8 | 32.8 | 0.075 | 86.9% | 5.0% | 4478 | **15.9%** | 0.012 | 0.0006 |

### 6.3 Health signals (steps 1-8)

- **Reward**: trending up (0.02 → 0.09 peak; noisy step-to-step at 320 traj/step).
- **r=0% (no reward)**: dropping 95.6% → 86.9%. Model finding rewards more often.
- **r=1% (perfect F1)**: climbing 1.6% → 5.0%.
- **Mean generation length**: 7038 → 4478 tokens. Model learning to **answer faster, stop sooner**.
- **Truncation rate**: 68.4% → 15.9%. The strongest learning signal — most rollouts now finish before hitting the 8192 cap. Confirms the model is escaping the "search forever" failure mode.
- **KL error to reference**: stable at 0.0006 (β=0.001 KL penalty doing its job — no policy collapse).
- **Loss**: oscillating in [-0.019, 0.027]. Normal for clipped-PG with γ=ε=0.2.
- **Per-step time**: trending faster (57.9 → 32.8 min). vLLM async engine + dropping gen length compound. **Steady-state estimate now ~33-38 min/step.**

### 6.4 ETA — revised

| Estimate vintage | Per-step | Full run (622 steps) |
|---|---:|---:|
| Pre-measurement (smoke v6 × scale) | ~29 min | ~12.5 d |
| v7 baseline measured (steps 1-2) | ~58 min | **~25 d** |
| **Live (steps 6-8 trend)** | **~33-38 min** | **~14-16 d** |

The improvement vs v7 baseline projection (which used only 2 steady-state samples) comes from: model learning to produce shorter rollouts → vLLM gen phase shrinks; vLLM async batching settling in; possibly some warmup amortization.

### 6.5 Live timing breakdown (step 8)

| Phase | Time | Share |
|---|---:|---:|
| `policy_training` | 23.5 min | **71.6%** |
| `policy_and_reference_logprobs` | 6.7 min | 20.5% |
| `generation` (vLLM async) | 2.3 min | 6.9% |
| `prepare_for_generation/total` | 0.1 min | 0.3% |

Training is still the dominant cost; vLLM async + multi-turn + chunked prefill keeps generation under 7%. The deferred `model_utils.py:1378` chunking patch (would unlock micro=2) is still the biggest available speedup if pursued in M5.3.

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
| 2026-05-11 | `train_micro_batch_size: 2 → 1` | v7 step-2 OOM at production seq=8192. NeMo-RL TP=1 path casts full [B,S,V] logits to fp32 before any chunking (model_utils.py:1378). 0.62 GiB short. |
| 2026-05-11 | Phase-2 iteration plan (v7→v8→…) truncated to bundled O1+R2 commit | Each iteration ~5h at production shape; user-headed-to-bed budget couldn't fit five iterations AND launch full training. O1/R2 are textbook-safe shipped features; no measurement needed. |
| 2026-05-11 | Validation disabled at runtime | First launch crashed: `val_at_end: true` + `data.validation: null`. No MuSiQue dev parquet (`prep_musique.py` only emits train split). Eval final ckpt out-of-band via evaluation_qwen35. |
| 2026-05-11 | Checkpoint metric switched to `train/loss/mean` | Was `val:accuracy`; needed a value to compare for `keep_top_k`. With val disabled, switched to a training metric + `keep_top_k=0` (keep all 12 saves). |

---

## 8. Pointers

- M5 milestone narrative: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md)
- M5 code-setup audit: [`CODE_SETUP_m5.md`](CODE_SETUP_m5.md)
- M5.1 paper-vs-ours mapping: [`../milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md)
- Runtime-efficiency menu (Phase 2 lever source): [`../research/RUNTIME_EFFICIENCY.md`](../research/RUNTIME_EFFICIENCY.md)
- M2 smoke wall-clock anchor: [`../training/SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md)
- ReSearch paper: [arXiv:2503.19470](https://arxiv.org/abs/2503.19470)
