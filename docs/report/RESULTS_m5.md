---
title: Results M5 — Qwen3.5-0.8B GRPO-trained on MuSiQue (ReSearch recipe)
tags: [report, training, eval, m5, m5.1, qwen3.5]
source: internal
created: 2026-05-11
updated: 2026-05-11
status: training-in-flight
---

# Results M5: Qwen3.5-0.8B GRPO Training (M5 + M5.1)

**Status (2026-05-11):** M5.1 production training is running on Vast 1× A100-80GB (pid 178440, W&B run `uwbodqgt`). This doc holds the final results + transferable observations once training completes; smoke / iteration history lives at [`RESULTS_SMOKE_m5.md`](RESULTS_SMOKE_m5.md).

## 1. Run roster

| Variant | Base model | Config | Branch @ commit | W&B run |
|---|---|---|---|---|
| `qwen3.5_0.8b_grpo_musique` | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) (hybrid) | [`m5_1_research_paper.yaml`](../../training_m5_1/configs/m5_1_research_paper.yaml) | `research_v2 @ db0852b` | `uwbodqgt` on [`reason_over_search_m5_1`](https://wandb.ai/gaurisankarj1996-leiden-university/reason_over_search_m5_1) |

Pipeline: [`training_m5_1/`](../../training_m5_1/). Code-setup audit: [`CODE_SETUP_m5.md`](CODE_SETUP_m5.md). Milestone narrative: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md). Paper-vs-ours: [`../milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md).

## 2. Training configuration (M5.1)

| Knob | Value | Note |
|---|---|---|
| Base model | `Qwen/Qwen3.5-0.8B` (hybrid) | M4 eval target — train and eval share rollout shape |
| Dataset | MuSiQue train (`data/training/musique/train.parquet`, 19,938 rows) | Hardest single-dataset paper benchmark |
| Algorithm | GRPO | Paper-faithful |
| Group size G | 5 | Paper-faithful |
| `num_prompts_per_step` | 64 | Paper has 256; 1× A100 ceiling forced 64 (see PAPER_VS_OURS_M5 §8) |
| `train_micro_batch_size` | 1 | v7 OOM at micro=2; see RESULTS_SMOKE_m5 §3.1 |
| `max_total_sequence_length` | 8192 | Paper-faithful |
| `max_new_tokens` per turn | 1024 | Paper-faithful (Group C) |
| `max_rollout_turns` | 10 | Safety cap (paper has no explicit) |
| KL coefficient | 0.001, k3 | Paper-faithful |
| PPO clip ε | 0.2 (sym) | Paper-faithful |
| LR | 1e-6 constant (no warmup) | Paper-faithful (Group C) |
| Reward | F1 only on `<answer>` content | M5 divergence — paper has F1 + 0.1 floor + format gate; ours matches M4 eval scorer |
| `use_leave_one_out_baseline` | false | Paper default (Group C) |
| `max_obs_chars` | 1024 | Safety cap; paper has no per-obs cap |
| Schedule | 2 epochs × 311 steps = **622 steps total** | Paper: 2 epochs × ~78 steps = 156 (same data, smaller batch) |
| System gains | O1 (fused AdamW) + R2 (vLLM async_engine) + R1 (prefix caching, default on A100) | M5.2 — non-paper, orthogonal to training math |
| Validation | disabled (no MuSiQue dev parquet) | Eval out-of-band via `evaluation_qwen35` on final ckpt |
| Checkpoint | every 50 steps, `train/loss/mean`, `keep_top_k=0` | ~12 saves over the run |

## 3. Smoke results

See [`RESULTS_SMOKE_m5.md`](RESULTS_SMOKE_m5.md) for the full v1-v6 + v7 + M5.1 iteration log. Headline:

| Version | Mean s/step (ex-warmup) | Phase mix | Outcome |
|---|---:|---|---|
| **v6** (smoke shape, 20 traj/step, seq=4096) | 93.1 s | train 58% / logprob 19% / gen 11% | Pipeline validated. Baseline. |
| **v7** (production shape, 320 traj/step, seq=8192, `train_micro_batch_size=2`) | OOM at step 2 | — | Forced micro=1; root-caused at `model_utils.py:1378` |
| **v7** (production shape, micro=1) | 3340 s (~55-59 min) | train 75% / logprob 21% / gen 4% | Established the 25-d baseline ETA |
| **M5.1** (production, live) | Trending down from 58 → 10 min (steps 1-16) | as v7 | See §4 |

## 4. M5.1 training — observations (live, will close at run end)

See [`RESULTS_SMOKE_m5.md` §6](RESULTS_SMOKE_m5.md#6-m51-production-training--live) for the live per-step trace + health-signal commentary; that section will be summarized here once training completes.

### 4.1 Transferable observation — RL-training dynamic regimes

A pattern surfaced in the M5.1 training run that's worth recording as a transferable finding (independent of the M5.1 final numbers). Two distinct RL-training trajectories appear in the literature:

| Regime | Length over training | Tool calls | Reward | Examples |
|---|---|---|---|---|
| "Aha moment" / long-CoT | grows ↑↑ | stable | grows ↑ | DeepSeek-R1, OpenAI o1 — **math / reasoning** |
| **"Efficient agent"** | **shrinks ↓** | **stabilizes at task complexity** | **grows ↑** | Search-R1, **ReSearch** — **multi-hop QA** |

**Why the difference matters**:

- **Long-CoT tasks** (math proofs, multi-step deductive reasoning) reward longer chains because verification of intermediate steps is part of the answer. Length growing signals the model discovering richer reasoning patterns. The "aha moment" in DeepSeek-R1 papers refers to exactly this — sudden length jumps coincide with capability emergence.
- **Efficient-agent tasks** (multi-hop retrieval QA) reward *correct* search-then-answer in the minimum hops needed. Length growth here typically signals **failure modes**: looping, confusion, hitting the rollout cap. The right learning signal is length DOWN to the natural complexity of the task (e.g., 2-4 hops for MuSiQue), with reward UP.

**M5.1's trajectory (steps 1-16, ~5% of full run)** lands firmly in the efficient-agent regime:
- `tc_mean` (tool calls per rollout): **8.96 → 3.47** (model converged on ~3-4 calls — matches MuSiQue's 2-4 hop complexity).
- `tok_mean` (rollout length): **7038 → 2183 tokens** (3.2× compression).
- Truncation rate: **68.4% → 0%** by step 15 (the "search forever" failure mode disappeared).
- Reward (F1 mean): **0.020 → 0.132 peak** (6.6× from baseline; 3-step rolling mean 0.039 → 0.120).

The "shrink-and-improve" pattern is the textbook signature of GRPO + retrieval on multi-hop QA — paper authors don't dwell on length plots because the dynamic is expected. If we saw length GROW *with* reward growth, that would suggest the model was discovering reasoning patterns the recipe didn't anticipate (interesting but inconsistent with the clipped-PG + format-light reward design).

**What was being "shrunk away"** in the first 16 steps:
1. Rambling `<think>` blocks (instruction-tuned Qwen3.5 overthinks).
2. Literal `<function=example_function_name>` template copies from the prompt — the model emitted the example placeholder verbatim for the first 1-2 tool calls before self-correcting.
3. Hitting `max_turns=10` truncation in 68% of rollouts (the model didn't know how to commit to an answer).

The right interpretation: GRPO is unwinding bad rollout habits inherited from the instruction-tuned base, then refining toward a 2-4-hop pattern that matches the task. Reward and efficiency move together.

**Watch points for the rest of the run**:
- If length plateaus near 600 tokens (model's "minimum viable rollout" for a 2-4 hop question) and reward keeps climbing → healthy.
- If length keeps shrinking below 400 tokens while reward stalls → model has converged to "guess fast", potential reward hacking.
- If length suddenly starts GROWING again while reward grows (epoch 2?) → the model is finding a long-CoT mode for harder multi-hop questions. Would be the most interesting outcome to study.

Will re-check the dynamic at step 50 (first ckpt), step 311 (epoch boundary), and step 622 (run end).

## 5. Full-run results — TODO

To fill in once training completes (~5-15 days from launch on 2026-05-11 ~01:05 UTC).

| Quantity | Value |
|---|---:|
| Final checkpoint | TODO |
| Total wall-clock | TODO |
| MuSiQue dev EM (M4 eval pipeline) | TODO |
| MuSiQue dev F1 | TODO |
| Cross-benchmark EM (NQ / TriviaQA / PopQA / HotpotQA / 2WikiMultiHopQA / Bamboogle) — same eval as M4 | TODO |
| vs M4 untrained hybrid baseline (mean EM 0.057) | TODO |

### 5.1 Sub-checkpoint evaluations

Each 50-step checkpoint will be evaluable via `scripts/run_m4.sh` swapped to point at `results/grpo/m5_prod/seed42/step_N/`. Track:

| Step | Wall-clock from start | reward/mean (final 5 steps before ckpt) | r=1% | tc_mean | tok_mean | MuSiQue dev EM | MuSiQue dev F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 100 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 150 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 200 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 311 (end epoch 1) | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 400 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 500 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 622 (end epoch 2) | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## 6. Pointers

- M5 milestone narrative: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md)
- M5 code-setup audit: [`CODE_SETUP_m5.md`](CODE_SETUP_m5.md)
- M5 smoke + training iteration log (live snapshot during M5.1): [`RESULTS_SMOKE_m5.md`](RESULTS_SMOKE_m5.md)
- M5.1 paper-vs-ours mapping: [`../milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md)
- M5.3 (training-time efficiency) follow-up: [`../milestone_5/MILESTONE_5_3.md`](../milestone_5/MILESTONE_5_3.md)
- M4 untrained baseline (comparison anchor): [`RESULTS_m4.md`](RESULTS_m4.md)
- ReSearch paper: [arXiv:2503.19470](https://arxiv.org/abs/2503.19470)
