---
title: Results M5 — smoke iteration log (Qwen3.5-0.8B GRPO training, NeMo-RL)
tags: [report, training, m5, m5.1, qwen3.5, smoke]
source: internal
created: 2026-05-09
updated: 2026-05-09
---

# Results M5 smoke — iteration log

Live record of the smoke iterations on `training_m5_1/` (M5 pipeline-validation smoke + M5.1 production-shape smoke). Mirrors the [`RESULTS_SMOKE_m4.md`](RESULTS_SMOKE_m4.md) format. Status: **spine only**; sections marked **TODO** populate as the smokes run.

Pipeline: [`training_m5_1/`](../../training_m5_1/). Milestone: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md). Wall-clock anchor: [`../training/SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md) (M2 Qwen3.5-2B, 57 s/step at 20 traj/step on 1× A100-80GB).

## 1. Run roster

| Version | Phase | When | Pipeline change | Config change |
|---|---|---|---|---|
| v1 | M5 smoke | TODO | `training_m5_1/` scaffold copied from `training/`, MuSiQue dataset adapter, parser re-exported from `evaluation_qwen35` | `m5_smoke.yaml`: 20 traj/step, 50 steps, 200-row subsample |
| v2 | M5.1 smoke | TODO | `src/reward.py` swapped to F1-only on `<answer>…</answer>` (M5.1 divergence #1) | `m5_1_research_paper.yaml`: paper-shape (TBD), ~100 steps |

All runs: 1× A100-80GB on Vast.ai (`pantomiman/reason-over-search-v1:v1`), `Qwen/Qwen3.5-0.8B` hybrid, MuSiQue train split, IVF-SQ8 retriever × 8 workers, vLLM bf16, single seed.

## 2. v1 — M5 smoke (pipeline validation) — TODO

### 2.1 Setup

- `training_m5_1/configs/m5_smoke.yaml`: 20 trajectories/step, 50 steps total, MuSiQue 200-row subsample, no validation, 8× vLLM dp.
- `policy.model_name=Qwen/Qwen3.5-0.8B` (hybrid).
- Reward: still EM-only (carried from M2's `qa_em.py` port at smoke time; F1-only is M5.1, not M5).
- Action format / tool-response wrap: byte-aligned to M4 ([`evaluation_qwen35/flashrag/pipeline/active_pipeline.py`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py) qwen35 branch).

### 2.2 Result — TODO

| Metric | Value |
|---|---:|
| Per-step wall-clock (mean) | TODO |
| Per-step wall-clock (p50) | TODO |
| Per-step wall-clock (p95) | TODO |
| reward/mean (final 5 steps) | TODO |
| `tool_call_counts/mean` | TODO |
| `had_valid_answer` rate | TODO |
| Generation tokens (mean / p95 / max) | TODO / TODO / TODO |
| Gradient-norm trajectory | TODO |
| Clip-ratio mean | TODO |
| Total wall-clock | TODO |
| Peak VRAM | TODO |

### 2.3 What we learned — TODO

(Free-form bullets after the smoke runs.)

### 2.4 Verification — TODO

- [ ] Rendered prompt parity: `tokenizer.apply_chat_template(...)` output from `training_m5_1/src/retrieval_env.py:render_initial_prompt` `diff` empty against `evaluation_qwen35/flashrag/pipeline/active_pipeline.py` qwen35-branch first-turn prompt on a hand-picked MuSiQue question.
- [ ] Tool-response wrap parity: byte-for-byte match on a hand-picked turn.
- [ ] No vLLM rollout-engine crashes / `nan` rewards / stuck rollouts.

---

## 3. v2 — M5.1 smoke (production shape) — TODO

### 3.1 Setup

- `training_m5_1/configs/m5_1_research_paper.yaml`: ReSearch-paper-aligned shape (group size G, KL coef, response length, schedule — all from paper / `Agent-RL/ReSearch`).
- Reward: **F1-only on `<answer>…</answer>`** (M5.1 divergence #1).
- Answer wrap: plain `<answer>X</answer>` (M5.1 divergence #2 vs paper's `\boxed{}`).
- ~100 steps on full MuSiQue train split (no subsample).

### 3.2 Result — TODO

| Metric | Value |
|---|---:|
| Per-step wall-clock (mean) | TODO |
| Per-step wall-clock (p50) | TODO |
| Per-step wall-clock (p95) | TODO |
| reward/mean (final 5 steps) | TODO |
| reward/mean (first 5 vs final 5 — early-step learning signal) | TODO |
| `tool_call_counts/mean` | TODO |
| `had_valid_answer` rate (`<answer>` tag emitted) | TODO |
| Generation tokens (mean / p95 / max) | TODO / TODO / TODO |
| Truncation rate (response hit `max_response_length`) | TODO |
| Gradient-norm trajectory | TODO |
| Clip-ratio mean | TODO |
| Total smoke wall-clock | TODO |
| Peak VRAM | TODO |

### 3.3 Wall-clock projection (full M5.1 training run) — TODO

| Quantity | Value | Source |
|---|---:|---|
| Per-step wall-clock (M5.1 smoke) | TODO s | §3.2 |
| Paper schedule | TODO steps | [ReSearch paper §B](https://arxiv.org/abs/2503.19470) + `Agent-RL/ReSearch` config |
| Total wall-clock | TODO h | per-step × steps |
| Cost @ \$1.20/h on 1× A100-80GB | \$ TODO | Vast.ai pricing |
| Fits ≤10 h target? | TODO (Y/N) | [`../TODO_2026-05-04.md`](../TODO_2026-05-04.md) |
| If N: cheapest knob to cut | TODO | typically `max_response_length` or `num_prompts_per_step` |

### 3.4 What we learned — TODO

(Free-form bullets after the smoke runs.)

### 3.5 Verification — TODO

- [ ] Reward path on 5 hand-picked rollouts (correct / partial-overlap / wrong / empty / format-broken). Expected F1 = 1.0 / 0<F1<1 / 0 / 0 / 0.
- [ ] Rendered-prompt parity check repeated against M4.
- [ ] `wandb` panels show non-degenerate gradient norm + clip ratio (no `inf` / `nan`).
- [ ] First 5 steps' reward/mean ≥ untrained-floor on a 50-row MuSiQue val subset (sanity: the model can already get some F1 even before training; if 0, something upstream is broken).

---

## 4. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-09 | M5 starts in NeMo-RL, not verl | Verl does not support Qwen3.5 (M2 finding); patching is more work than porting the recipe knob-by-knob. |
| 2026-05-09 | M5 train rollout byte-aligned to M4 eval rollout | Avoids the M3-style 14-fix train/eval-drift audit by construction. |
| 2026-05-09 | M5.1 reward = F1-only on `<answer>…</answer>` | F1 is more discriminative than EM at small scale; format-reward partial-credit floor masks the tool-use signal (Phase-1 finding from [`RESULTS_m0_a.md`](RESULTS_m0_a.md) §learning #4). |
| 2026-05-09 | M5.1 answer wrap = plain `<answer>X</answer>` (no `\boxed{}`) | Carry from M4; eval scorer accepts both shapes. |
| 2026-05-09 | M5.1 dataset = MuSiQue only (single-dataset training) | Hardest of the four paper benchmarks; largest single-dataset headroom in M1 numbers (EM 0.124 baseline); simplest reproducible recipe. |
| 2026-05-09 | Folder layout = `training_m5_1/`, `training_m5_2/` (separate per experiment) | Hard isolation while one experiment full-trains; second experiment can mutate any code path without touching the active run. |

---

## 5. Pointers

- M5 milestone narrative: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md)
- M5 code-setup audit: [`CODE_SETUP_m5.md`](CODE_SETUP_m5.md)
- M5.1 paper-vs-ours mapping: [`../milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md) (TODO)
- M2 smoke (wall-clock anchor; Qwen3.5-2B at 57 s/step): [`../training/SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md)
- M4 eval pipeline (the rollout-shape source of truth): [`../milestone_4/MILESTONE_4.md`](../milestone_4/MILESTONE_4.md), [`CODE_SETUP_m4.md`](CODE_SETUP_m4.md), [`RESULTS_SMOKE_m4.md`](RESULTS_SMOKE_m4.md)
- ReSearch paper: [arXiv:2503.19470](https://arxiv.org/abs/2503.19470), notes at [`../papers/2503.19470_research.md`](../papers/2503.19470_research.md)
- ReSearch official codebase: [`Agent-RL/ReSearch`](https://github.com/Agent-RL/ReSearch)
- Active recipe-ablation plan: [`../TODO_2026-05-04.md`](../TODO_2026-05-04.md)
