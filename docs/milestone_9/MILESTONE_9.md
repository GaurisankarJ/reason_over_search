---
title: MILESTONE 9 — M5.1 GRPO checkpoint evaluation against the 7-benchmark suite
tags: [milestone, evaluation, m9, m5.1, qwen3.5-0.8b, alice]
source: internal
created: 2026-05-17
updated: 2026-05-17
---

# Milestone 9: Evaluate M5.1 checkpoints

## Context

[M5.1](../milestone_5/MILESTONE_5.md) trained Qwen3.5-0.8B with GRPO on MuSiQue under the ReSearch-paper recipe with F1-only reward. The run paused at **step_180** (58 % of one epoch); see [`docs/report/RESULTS_M5_1_H200.md` §9.6](../report/RESULTS_M5_1_H200.md) for the hold rationale.

[M4](../milestone_4/MILESTONE_4.md) established the untrained Qwen3.5-0.8B floor on the same 7 benchmarks using `prompt_mode=qwen35_minimal` (locked 2026-05-09 in M4.2). M9 evaluates the M5.1 GRPO checkpoints against the same 7 benchmarks so we can call the actual EM/F1 lift over the untrained floor.

## Scope (coarse pass while CloudRift access is being sorted)

5 checkpoints sampled across the M5.1 trajectory:

| Checkpoint | Step | Cadence | Notes |
|---|---:|---:|---|
| `step_10` | 10 | C1 | Earliest learnable signal |
| `step_50` | 50 | C5 | Just past the warm-up; reward ~0.20 |
| `step_100` | 100 | C10 | Mid-run; reward ~0.23, just before C11 jump |
| `step_150` | 150 | C15 | Recovery cadence after C14 over-search peak |
| `step_180` | 180 | C18 | Final (paused) checkpoint |

7 benchmarks: NQ (3,610 test), TriviaQA (11,313 test), PopQA (14,267 test), HotpotQA (7,405 dev), 2WikiMultiHopQA (12,576 dev), MuSiQue (2,417 dev), Bamboogle (125 test) = **51,713 rows / checkpoint**.

5 × 51,713 = 258,565 generations total. Per M3 calibration (Qwen3-0.6B greedy, A100-80GB, ~2.5 h / variant), expect ~3-4 h per checkpoint with Qwen3.5-0.8B (slightly slower per-token but same retrieval contract). Total wall: ~15-20 h on a single A100; in parallel across multiple ALICE allocations the calendar time is bounded by partition queue depth.

## Pipeline + alignment

- **Eval pipeline**: [`evaluation_qwen35/`](../../evaluation_qwen35/) (M4's pipeline, branch `eval_1_alice`).
- **Prompt mode**: `m5_qwen35_train` (NEW; added in commit 72ccc91 on `eval_1_alice`). Byte-for-byte mirror of [`training_m5_1/src/prompts/m5_qwen35_user.txt`](../../training_m5_1/src/prompts/m5_qwen35_user.txt). The checkpoints were trained against this exact user-message prompt; evaluating with a different prompt would test OOD policy behaviour.
- **Tool schema**: `tools=[QWEN35_SEARCH_TOOL]` auto-injected (same Python object the training rollout imports; no drift possible).
- **No system message** (`system_prompt_file: null` at training; user-locus prompt at eval).
- **`enable_thinking=True`** (training rolled out with the open `<think>\n` generation prefix).
- **Retriever**: Wiki-18 corpus + E5-base-v2 + IVF-SQ8 FAISS at `indexes/wiki18_100w_e5_ivf4096_sq8.index` (symlink to the M3/M4 prebuilt index), 8-worker CPU service on port 3005.
- **Decoding**: greedy (temperature=0) per M3/M4 convention.

Audit of train-vs-eval alignment is captured in [`docs/report/CODE_SETUP_m9.md`](../report/CODE_SETUP_m9.md) (one-page summary).

## Checkpoint conversion (NeMo-RL to HF)

Our HF-uploaded checkpoints store NeMo-RL's *consolidated* format: single shard `step_<N>/policy/weights/model/shard-00001-model-00001-of-00001.safetensors` with 473 `model.language_model.*` and `model.visual.*` keys (Qwen3.5 VL-namespace) and NeMo-RL shard naming. SGLang needs HF-native naming plus a complete state dict.

[`scripts/convert_m5_ckpt_to_hf.py`](../../scripts/convert_m5_ckpt_to_hf.py) overlays the trained tensors onto the base Qwen3.5-0.8B HF state dict, keeping the 15 base `mtp.*` (speculative-decoding head) tensors that NeMo-RL training doesn't update. Output: a complete HF dir at `eval/m9/step_<N>_hf/` with the base safetensors filename so the base's `model.safetensors.index.json` still resolves.

## Status

- Branch: `eval_1_alice` (forked from `experiment_1_h200`; merge back after M9 wraps).
- Workspace on ALICE: `/zfsstore/user/s4374886/omega/reason_over_search_m9/`.
- 5 checkpoints downloaded + converted (each ~3.3 GB HF format, 16.5 GB total).
- First sbatch job (step_10) submitted as smoke 2026-05-17 ~18:05 UTC; job 2409771 on gpu-short A100.

Results land in [`docs/report/RESULTS_m9.md`](../report/RESULTS_m9.md) (auto-aggregated).
