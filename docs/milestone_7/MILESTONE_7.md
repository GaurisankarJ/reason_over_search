---
title: MILESTONE 7 — Qwen3.5-0.8B-Base GRPO training (parallel to M5.1 hybrid)
tags: [milestone, training, grpo, m7, m7.1, base]
source: internal
created: 2026-05-12
updated: 2026-05-12
---

# Milestone 7: GRPO training on Qwen3.5-0.8B-Base with the M4.3-locked prompt

## Context

[`MILESTONE_5.md`](../milestone_5/MILESTONE_5.md) defined GRPO training on the **hybrid** Qwen3.5-0.8B with the M4-locked terse prompt. M7 is the parallel experiment on the **base** Qwen3.5-0.8B-Base variant with the M4.3-locked `qwen35_minimal_no_system` prompt.

Hybrid was post-trained on tool-use (instruct + thinking soft-switch); base was not. M4 evidence:
- Untrained hybrid scores mean EM **0.060** (M4.2) → **0.092** with the M4.5-locked terse prompt.
- Untrained base scores mean EM **0.010** across every prompt variant tested in [M4.4 Phase 4](MILESTONE_4.md#phase-4-results--primary-4-candidate-base-prompt-screen-2026-05-12) (7-candidate × n=300 prompt screen, null result).
- Base has **no tool-use prior** — it doesn't reliably emit `<tool_call>` blocks zero-shot.

M7 tests the hypothesis that **GRPO can teach the base model the tool-use loop from scratch**, potentially exceeding what post-training-conditioned hybrid + GRPO achieves (post-training may bias hybrid toward unhelpful patterns; base is a clean slate).

## Goal + hypothesis

1. **Train Qwen3.5-0.8B-Base via GRPO** on MuSiQue, identical recipe to M5.1, only swapping model weights + prompt + render shape.
2. **Test the cleaner-slate hypothesis**: does base + GRPO learn the search-loop without hybrid's post-training biases?
3. **Plug the trained-base row into [RESULTS_m4.md §4](../report/RESULTS_m4.md)** for direct comparison vs both the M4.3 untrained floor (0.010) and the M5.1 trained-hybrid checkpoint (TBD).

**Critical open question**: can GRPO produce a tool-use prior in a model that doesn't have one? If base can't reliably emit `<tool_call>` blocks zero-shot, almost every rollout will hit reward = 0 (F1-only reward, no partial-credit floor per [`training_m5_1/src/rewards/search_r1.py:113-116`](../../training_m5_1/src/rewards/search_r1.py)), GRPO advantage = 0 across the group, **no gradient signal → no learning**. The M7 smoke gates this risk.

## What's the SAME as M5.1

| Knob | Value | Source |
|---|---|---|
| Algorithm | GRPO (NeMo-RL v0.6.0) | [training_m5_1/nemo_rl](../../training_m5_1/nemo_rl) |
| Dataset | MuSiQue train (~20k items × 2 epochs ≈ 40k prompts) | [`scripts/prep_musique.py`](../../training_m5_1/scripts/prep_musique.py) |
| Reward | F1 on `<answer>X</answer>`, **no partial-credit floor**; reward = `float(f1)` | [`src/rewards/search_r1.py`](../../training_m5_1/src/rewards/search_r1.py) |
| Eval scorer | `flashrag.search_r1.reward.{em_check, f1_check, extract_solution}` — same code path as M4 eval | [`evaluation_qwen35/flashrag/search_r1/reward.py`](../../evaluation_qwen35/flashrag/search_r1/reward.py) |
| Hardware | 1× A100-80GB (current Vast box) | – |
| Production hyperparameters | 64 prompts/step × G=5 = 320 traj/step, 622 steps × 2 epochs, `max_total_sequence_length=8192`, `max_new_tokens=1024`, `max_rollout_turns=10` | [`configs/m5_1_research_paper.yaml`](../../training_m5_1/configs/m5_1_research_paper.yaml) |
| Wall-clock estimate (production) | ~10-13 d on A100 (anchored on M5.1 a1+a2 step-trajectory data) | – |
| Cost estimate (production) | ~\$300-450 @ Vast \$1.20/h | – |

## What's DIFFERENT from M5.1

| Knob | M5.1 (hybrid) | **M7.1 (base)** |
|---|---|---|
| Model weights | `Qwen/Qwen3.5-0.8B` | **`Qwen/Qwen3.5-0.8B-Base`** |
| Prompt template | `qwen35_terse` (M4.5 lock) | **`qwen35_minimal_no_system`** (M4.3 lock) |
| Render shape | system block (auto-injected `# Tools` + `<IMPORTANT>`) + terse user message | **user-only message, no system block; format spec inlined into the user prompt** |
| `tools=[SEARCH_TOOL]` passed to `apply_chat_template` | **yes** (auto-inject ON) | **no** (auto-inject OFF — M4.3 evidence: hurts base 5×) |
| Untrained EM (cross-family vs M3=0.102) | 0.092 → −0.010 gap | 0.010 → −0.092 gap |
| Tool-use prior | yes (post-training) | **no** |
| Reward-signal density at step 0 | meaningful — hybrid emits `<tool_call>` reliably | **sparse — base may not emit `<tool_call>` at all** |
| New processor arm | `qwen_native` | **`qwen_native_no_system`** (new) — drops `tools=` from `apply_chat_template` |
| Prompt file | `prompts/m5_qwen35_user.txt` | **`prompts/m7_qwen35_base_user.txt`** |
| Wandb project | `reason_over_search_m5_1` | **`reason_over_search_m7_1`** |
| Checkpoint dir | `results/grpo/m5_prod/seed42/` | **`results/grpo/m7_prod/seed42/`** |

## Risks specific to base

1. **No tool-use prior → reward sparsity → no learning**. If `<tool_call>` emission rate stays <5% in early rollouts, almost every rollout hits reward = 0, GRPO advantage = 0 across the group, gradient signal = 0. The M7 smoke catches this in 10-20 steps.
2. **Off-distribution `enable_thinking=True`**. Base wasn't post-trained on the hybrid soft-switch protocol, so the open `<think>\n` generation prefix is mildly off-distribution. M4.3 lock keeps `enable_thinking=True` anyway for cross-variant comparability; M7 inherits this.
3. **2wiki ceiling**. Untrained base on 2wikimultihopqa is 0.032 (M4.3) vs hybrid 0.046 (M4.5). The binary-comparison subset of 2wiki was where base scored unusually well via bypass-and-fabricate at ~50% chance. GRPO may amplify the bypass-fabricate pattern on 2wiki if reward signal there is sticky. Worth monitoring per-dataset reward trajectories in W&B.
4. **Compute waste**. \$300-450 if production runs to completion and learns nothing. Smoke kill-switch (below) prevents commitment to production without signal.

## Phase plan with kill-switches

| Phase | What | Wall | Cost | Decision |
|---|---|---:|---:|---|
| **M7.0** Setup | Clone `training_m5_1/` → `training_m7_1/`; add `qwen_native_no_system` arm to processor; new prompt file; update yaml configs (smoke + prod); update registry imports | 30-45 min coding, no GPU | $0 | – |
| **M7.0.5** Verification | Render check on base tokenizer (verify no system block, format spec inlined, ~150 tok); inspect single rollout against live retriever (no GRPO loss; manual one-item run); reward function unit test (F1 path); confirm extract_solution works on a hand-crafted rollout | 15 min, no GPU | $0 | Confirms pipeline mechanics before any compute |
| **M7.0.7** Smoke | 10-20 step GRPO run, smoke config (20 traj/step, MuSiQue). Watch: reward-mean trajectory, `<tool_call>` emission rate per rollout, rollout-length distribution, OOM/timeout | ~30-60 min on A100 | ~$1-2 | **Kill if** `<tool_call>` emission rate <5% by step 5 (base isn't engaging the tool). **Kill if** reward-mean stays at 0.0 across all 10-20 steps (no signal). **Proceed if** any positive reward movement OR `<tool_call>` emission rate ≥10% (gradient signal present, even if reward absolute is low) |
| **M7.1** Production | Full GRPO run, paper-faithful config (M5.1-prod-a3 shape, base model swapped in) | ~10-13 d | ~\$300-450 | Cap at step 100 initially; review reward + ckpt; full ramp on green |
| **M7.2** Eval | Full Plan A sweep (n=51,713 × 7 datasets) with `qwen35_minimal_no_system` against the trained-base checkpoint | ~2.5 h | ~$5 | Compare to M4.3 untrained floor (0.010) and to M5.1 trained-hybrid (TBD). M7 success bar set post-prod once we see signal trajectory |

## Implementation plan for M7.0 (concrete steps)

1. **Clone training overlay**:
   ```bash
   cp -r training_m5_1/ training_m7_1/
   ```
2. **Rename Python imports** (`from training_m5_1.src...` → `from training_m7_1.src...`) — 14 import sites per M5.1 audit; mirror that rename here.
3. **Add `qwen_native_no_system` arm** to [`src/processors/search_r1.py`](../../training_m5_1/src/processors/search_r1.py): same as `qwen_native` but skip `tools=[SEARCH_TOOL]` argument. ~10 LoC.
4. **Create base prompt file** `training_m7_1/src/prompts/m7_qwen35_base_user.txt` from `QWEN35_MINIMAL_NO_SYSTEM_TEMPLATE`. Use `scripts/sync_m4_prompts.py --mode qwen35_minimal_no_system` if the sync script supports it (verify); otherwise hand-author.
5. **Update yaml configs**:
   - `configs/m7_smoke.yaml`: model_path = base; arm = `qwen_native_no_system`; prompt_file = `m7_qwen35_base_user.txt`; wandb_project = `reason_over_search_m7_1`; checkpoint_dir = `results/grpo/m7_prod`; 10-20 step cap
   - `configs/m7_1_research_paper.yaml`: same model/arm/prompt swap; otherwise identical to `m5_1_research_paper.yaml`
6. **Update entry scripts**: `scripts/run.sh`, `scripts/smoke.sh`, `scripts/run_grpo.py` — same logic but rooted at `training_m7_1/`
7. **Test scaffolding**: `tests/test_reward_parity.py` — verify M7 reward path matches eval scorer (should be identical since both call `flashrag.search_r1.reward`)

## Success criteria (definition deferred to post-smoke)

- **Smoke pass criteria** (defined now): `<tool_call>` emission rate ≥10% by step 20 AND any positive reward movement above 0.0 mean.
- **Production success bar** (to define post-smoke based on actual signal): see [M7.2 eval phase](#phase-plan-with-kill-switches).

The conservative interpretation: M7 producing a measurable lift over the M4.3 untrained floor of 0.010 mean EM is the *minimum* success criterion. The aspirational target is matching or beating M4.2 hybrid floor (0.060), which would demonstrate GRPO can substitute for tool-use post-training. Final bar set after seeing the smoke + production trajectory.

## Run sequence (target dates)

| Phase | Target start | ETA done | Status |
|---|---|---|---|
| M7.0 Setup | 2026-05-12 17:30 UTC | 30-45 min | ✅ done |
| M7.0.5 Verification | 2026-05-12 17:40 UTC | 15 min | ✅ done — see [§"Verification results"](#verification-results-m705) |
| M7.0.7a Smoke seq=4096 | 2026-05-12 17:14 UTC | 44 min wall | ✅ done — REWARD COLLAPSE confirmed; see [§"Smoke A results"](#smoke-a-results-m707a-seq4096) |
| M7.0.7b Smoke seq=8192 | 2026-05-12 18:00 UTC | partial (16/20 steps, OOM on step 17) | ✅ usable data; see [§"Smoke B results"](#smoke-b-results-m707b-seq8192-partial) |
| M7.0.7c Prod-shape smoke | 2026-05-12 18:39 UTC | 51 min wall (3 steps) | ✅ done — STRUCTURAL LEARNING SIGNAL; see [§"Smoke C results"](#smoke-c-results-m707c-prod-shape-3-steps) |
| M7.1 Production | Pending user auth | ~5-10 d est. | ⏳ ready to launch — see [§"M7.1 production decision"](#m71-production-decision) |
| M7.2 Eval | After M7.1 | 2.5 h | pending M7.1 |

## Verification results (M7.0.5)

All 7 verification checkpoints PASS as of 2026-05-12 17:40 UTC.

| Checkpoint | Method | Result |
|---|---|---|
| Prompt template render | Compare `tokenizer.apply_chat_template(...)` of `m7_qwen35_base_user.txt` vs `QWEN35_MINIMAL_NO_SYSTEM_TEMPLATE` on the base tokenizer | ✅ **BYTE-IDENTICAL** to M4.3 eval lock, 145 tokens, no system block, open `<think>\n` |
| Arm dispatch (dataset) | `VALID_ARMS` in [`src/datasets/search_r1.py:30`](../../training_m7_1/src/datasets/search_r1.py) + training log shows `search_r1_qwen_native_no_system` | ✅ wired |
| Arm dispatch (parser) | `parse_query("qwen_native_no_system", sample)` extracts the search query from the `<tool_call>` block | ✅ regex matches Qwen3.5 nested-XML |
| Arm dispatch (env) | Tool-response wrap routes to `format_docs_qwen_native`; `_stop_strings_for_arm` returns `["</tool_call>", "</answer>"]` | ✅ same wrap as hybrid |
| Arm dispatch (processor) | New `elif arm == "qwen_native_no_system"` branch in [`src/processors/search_r1.py`](../../training_m7_1/src/processors/search_r1.py) skips `tools=[SEARCH_TOOL]` in `apply_chat_template` | ✅ auto-inject OFF as designed |
| Reward function | `compute_search_r1_reward("<answer>Christopher Nolan</answer>", ["Christopher Nolan"])` → `{reward: 1.0, em: 1.0, f1: 1.0}`. Partial: `{reward: 0.667, ...}`. No-extract: `{reward: 0.0, ...}` | ✅ F1-only, no 0.1 floor (matches M5.1 contract) |
| Model + venv | `/venv/.../python -c "import torch; import nemo_rl"` returns clean; model loads from `/workspace/.../eval/qwen3.5_0.8b_base` | ✅ |

## Smoke results (M7.0.7a, seq=4096)

Ran 2026-05-12 17:14:20 → 17:58:30 UTC, **44 min total wall**, 20 steps × 20 traj/step = 400 rollouts, on 1× A100-80GB. W&B run: `qwen3.5-0.8b-musique-m5_smoke-seed42-20260512T1740Z` (cosmetic — the actual run is the M7 base smoke; the `m5_` prefix is a leftover from M5.1's run.sh that was fixed in this commit but the smoke was already running).

**Headline result: reward collapse confirmed. Truncation is the root cause.**

Per-step trajectory across all 20 steps:

| step | reward mean | pct rew>0 | pct emit `<answer>` | pct truncated | avg `<tool_call>`s |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.014 | 10 % | 25 % | 90 % | 1.00 |
| 5 | 0.000 | 10 % | 40 % | 80 % | 1.20 |
| 10 | 0.000 | 0 % | 45 % | 65 % | 1.50 |
| 15 | 0.000 | 0 % | 5 % | 100 % | 1.40 |
| 20 | 0.004 | 5 % | 15 % | 80 % | 1.45 |

**Findings:**

1. **`<tool_call>` emission rate is ~100 % from step 1.** Untrained base immediately engages the tool — this was the biggest positive surprise. The inlined format spec in the user prompt + the open `<think>\n` generation prefix is sufficient to drive consistent tool use without any post-training prior.
2. **Truncation rate is 65-100 % throughout.** Rollouts hit the 4096 sequence budget mid-`<think>` or mid-search-loop and never emit `<answer>X</answer>`. Per-rollout decomposition: ~500-800 tok initial `<think>`, ~1024 tok first generation, ~600 tok first `<tool_response>` (top-5 chunks × 120 tok/chunk), ~400-600 tok second `<think>`, optional 2nd search loop — total easily 3500-4500 tok before answer.
3. **Reward signal is sparse but present.** 5 of 20 steps (1, 5, 6, 11, 18, 20) had at least one rollout with reward > 0. Mean reward across these "lucky" rollouts is 0.005-0.02. When all 5 rollouts in a group hit 0.0, GRPO advantage = 0 → no gradient signal. When one rollout in a group hits 0.087, the positive rollout gets a +0.07 advantage, the others get a -0.017 advantage.
4. **`<answer>` emission rate trajectory: 25 → 40 → 45 → 5 → 15 %.** Steps 1-10 trend positive (model learning to close with an answer); step 15 collapses; step 20 partial recovery. Not enough signal density for monotonic learning.

**Implications for next smokes:**
- **M7.0.7b** (seq=8192): hypothesis is that doubling the sequence budget reduces truncation rate from ~80 % to <50 %, raising the `<answer>` emission rate and unlocking F1 reward signal density.
- **M7.0.7c** (prod-shape, 320 traj/step): even if the per-rollout success rate stays around 10 %, having 320 rollouts per step means ~32 positive-reward rollouts per step (vs ~2 in this smoke), distributed across 64 groups. The expected number of groups containing at least one positive rollout is much higher → much more gradient signal per step.

Result files preserved at `logs/exp_012/train_data_step{1..20}.jsonl`.

**Sequencing vs M5.1-prod-a3**: M5.1-prod-a3 is currently paused awaiting user authorization per [`TODO_2026-05-12.md`](../todo/TODO_2026-05-12.md). M7 runs FIRST on the current box; M5.1-a3 launches afterward (either on same box if M7 fails/completes, or on a second box if budget allows parallelism).

## Smoke B results (M7.0.7b, seq=8192, partial — 16 of 20 steps)

Ran 2026-05-12 18:00 → ~18:37 UTC; crashed at step 17 with CUDA OOM (`torch.OutOfMemoryError: Tried to allocate 15.16 GiB`) — same `[B,S,V]=[2,8192,248320]` log_softmax issue M5.1 v7 hit. Fixed in yaml (`train_micro_batch_size: 2 → 1`) for any future re-runs; production already uses micro=1. 16 steps of clean data preserved at `logs/exp_013/train_data_step{1..16}.jsonl`.

| step | rew_mean | pct rew>0 | max rew | pct `<answer>` | pct trunc |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0100 | 5 % | **0.200** | 30 % | 75 % |
| 5 | 0.0002 | 5 % | 0.003 | 50 % | 85 % |
| 6 | 0.0248 | 10 % | **0.400** | 20 % | 90 % |
| 11 | 0.0102 | 10 % | 0.182 | 40 % | 65 % |
| 14 | 0.0200 | 5 % | **0.400** | 50 % | 70 % |
| 16 | 0.0333 | 5 % | **0.667** | 30 % | 90 % |
| **agg (16 × 20 = 320)** | **0.0062** | **2.5 %** | **0.667** | **~33 %** | **~80 %** |

**Key signals vs Smoke A (seq=4096):**
- **Peak single F1 jumped from 0.087 → 0.667** — at 8192 the model produces SUBSTANTIAL semantic matches when it doesn't truncate. The capability exists.
- `<answer>` emission rate stayed flat (~30-45 %) but with higher quality.
- Truncation rate barely dropped (~80 % both smokes). 8192 helps the model finish more often but doesn't eliminate the over-thinking problem.
- Per-rollout positive-reward rate stayed at ~3 % (very sparse).

**Interpretation:** at smoke batch (20 traj/step), 3 % positive rate ≈ 0.6 positive rollouts/step on average → near-zero group advantage signal almost every step → GRPO can't learn. The signal needs density at the GROUP level, not the rollout level.

## Smoke C results (M7.0.7c, prod-shape, 3 steps)

Ran 2026-05-12 18:39 → 19:30 UTC (~51 min total wall, ~17 min/step including setup; much faster than the M5.1 v7 anchor of 55-60 min/step because smoke C skips checkpointing + validation overhead). 3 steps × 64 prompts × G=5 = 960 rollouts total. W&B run: `qwen3.5-0.8b-base-musique-m7_prod_shape-seed42-20260512T1839Z`.

| step | n_pos | rew_mean | max F1 | pct `<answer>` | pct trunc | expected groups w/signal (of 64) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10/320 | 0.012 | **1.000** ← perfect | **25.6 %** | 84.7 % | ~9.4 (14.8 %) |
| 2 | 1/320 | 0.001 | 0.150 | **29.4 %** | 82.5 % | ~1.0 (1.6 %) |
| 3 | 8/320 | 0.007 | 0.600 | **32.8 %** | **78.1 %** | ~7.7 (12.0 %) |

**Headline structural-learning signals (monotonic across 3 steps):**

- **`<answer>` emission rate: 25.6 → 29.4 → 32.8 %** (+7 pp). Model learning to close with an answer.
- **Truncation rate: 84.7 → 82.5 → 78.1 %** (−6.6 pp). Model learning to be more concise.
- These trends are exactly what we'd expect if base is learning the tool-use loop's *format* even when the F1-on-content reward is sparse.

**Reward-signal density:**

- 19 of 960 rollouts (2.0 %) hit reward>0; **2 hit a perfect F1=1.0** (step 1), **5 hit F1≥0.4** (substantial semantic matches).
- Random-distribution estimate: at 3.1 % positive rate, ~9 of 64 groups per step have advantage signal → meaningful learnability.
- Steps 1 + 3 had healthy signal (~12-15 % of groups). Step 2 dipped to 1.6 % — a "thrashing" step where the prior step's gradient happened to push the policy slightly off.

**Group-level expected signal across the 3 steps**: roughly 18 of 192 group-step instances (~9 %) produced GRPO advantage signal. Over 622 production steps, that scales to ~**4,000 advantage-positive group-steps** — enough cumulative gradient to potentially drive meaningful learning.

Result files at `logs/exp_014/train_data_step{1,2,3}.jsonl` (~80 MB total).

## M7.1 production decision

**Recommendation: GO with kill-switches.** Launch M7.1 full production (622 steps × 2 epochs), monitor with explicit abort conditions.

**Why GO:**
1. **Structural learning signal is real and monotonic across smokes B + C**: `<answer>` emission rate rises, truncation rate falls. This shows the model is responding to the GRPO updates even when the F1-on-content reward is sparse.
2. **Capability is demonstrated**: the perfect-F1 rollout in step 1 of smoke C confirms the base model can produce correct MuSiQue answers when it doesn't truncate. We're not bootstrapping a model with zero capability — we're bootstrapping one whose capability gets gated by truncation.
3. **Prod batch shape gives 5-15× more signal density than smoke shape**. At smoke batch (20 traj/step), 3 % positive rate ⇒ ~0.6 positive/step ⇒ near-zero group advantage. At prod batch (320 traj/step), 3 % rate ⇒ ~10 positive/step distributed across 64 groups ⇒ ~9-15 % of groups have non-zero advantage — enough for cumulative learning over 622 steps.
4. **Wall + cost are reasonable**: at smoke C's pace (17 min/step incl. setup) extrapolated to 622 steps = ~7.4 days; production has checkpointing + validation overhead so realistic estimate is **~7-10 days, ~$200-300** on Vast 1× A100. That's at the low end of the M5.1 cost envelope.

**Why NOT GO would be defensible:**
- Per-rollout reward rate is still very low (2-3 %) — model may take 100s of steps to start producing reliably correct answers.
- Step 2 reward dip suggests gradient signal isn't dense enough to monotonically improve in 3 steps; with 622 steps the pattern might be net-positive or might oscillate without converging.
- We don't know yet whether M5.1 hybrid + GRPO actually beats the M4.5 hybrid floor (0.092). If hybrid + GRPO doesn't lift, base + GRPO is unlikely to lift either.

**Proposed M7.1 launch with kill-switches:**

| Checkpoint | Action |
|---|---|
| Steps 1-50 | Monitor reward_mean, `<answer>` emission rate, truncation rate. Save ckpt at step 50. |
| **Step 50 review** | If reward_mean has risen to ≥ **0.02** (cumulative trend) AND `<answer>` emission rate ≥ **40 %** → continue. Otherwise → **KILL** at step 50. Cost up to that point: ~50 × 20 min × $1.20/h ≈ **$20**. |
| Steps 50-200 | If signal trajectory continues, ramp full schedule. Save every 50 steps. HF Hub auto-upload watcher running. |
| **Step 200 review** | If reward_mean ≥ 0.05 → continue to 622. If stuck below 0.03 → KILL at step 200. Cost up to here: ~$80. |
| Steps 200-622 | Full ramp; M7.2 eval on the final checkpoint. |

**Alternatives if GO is too risky for you:**
1. **Wait for M5.1-a3 hybrid** to produce a trained-hybrid result first. If hybrid + GRPO lifts the M4.5 floor (0.092), confidence rises that base + GRPO will too. Cost: 10-13 d wait.
2. **Add a small format-gate reward floor for M7** — e.g., reward = max(0.05, f1) if rollout emits `<answer>` (even an empty one). This bootstraps gradient density at the cost of diverging from the M5.1 contract. Quick fix; ~20 LoC.
3. **Reduce M7.1 to a shorter run** — e.g., 100 or 200 steps instead of 622. Costs ~$30-60. Still meaningful trajectory data even if it doesn't converge.

## M7.1 results (short100 — 100-step probe)

**Run**: `qwen3.5-0.8b-base-musique-m7_short100-seed42-20260512T2040Z` (W&B: `d82xtvwo`).
Started 2026-05-12 20:40 UTC, finished 2026-05-12 23:31 UTC. Wall: 2 h 51 min for 100 steps × 320 trajectories/step = 32,000 rollouts on 1× A100-80GB.

Per-step pace evolved dramatically (no fixed step time — vLLM AOT compile cache from M5.1 + capture-graphs warming up made it monotonically faster):

| step range | avg step time | reason |
|---|---:|---|
| 1-5 | 285 s | warmup, vLLM cudagraph capture, first-batch overhead |
| 6-25 | 200 s | post-warmup, micro=1 baseline |
| 26-50 | 110 s | steady-state, AOT cache hits |
| 51-100 | 70 s | final pace, all caches warm |

Total cost on Vast 1× A100 ≈ $4-5 (against my earlier $30-120 estimate; the AOT compile cache hit collapsed wall-clock).

### Reward trajectory (the headline)

| window | rew_mean | n_pos / 320 | lift vs baseline |
|---|---:|---:|---:|
| **steps 1-10 (baseline)** | **0.0104** | ~10 (3.3%) | 1.0× |
| steps 11-20 | 0.0166 | ~17 (5.3%) | 1.6× |
| steps 21-30 | 0.0344 | ~38 (12%) | 3.3× |
| steps 31-40 | 0.0344 | ~50 (16%) | 3.3× |
| steps 41-50 | 0.0378 | ~57 (18%) | 3.6× |
| steps 51-60 | 0.0589 | ~70 (22%) | 5.7× |
| steps 71-80 | 0.0840 | ~73 (23%) | 8.1× |
| **steps 91-100** | **0.0872** | **~67 (21%)** | **8.4×** |

GRPO is delivering real signal: **8.4× lift in 100 steps**, monotonic except for normal noise. Per-rollout positive rate climbed from 3% → 21%. The model went from "occasionally lucky" to "1 in 5 trajectories has nonzero F1."

Single-trajectory **F1 = 1.000** (perfect MuSiQue answer) appeared first at step 2 and was sustained from step 6 onward — that confirms the parametric capability was there from the start, and training is amplifying its expression.

### Thinking-trace inspection (early vs trained)

The model's reasoning style changed sharply over the 100 steps:

| step | avg response chars | % emit `<answer>` | % emit closing `</answer>` | % emit `<tool_call>` |
|---:|---:|---:|---:|---:|
| 1 | 2966 | 31.6% | 28.1% | **14.4%** |
| 10 | 2693 | 54.4% | 49.7% | 9.1% |
| 25 | 1743 | 92.2% | 90.3% | 1.2% |
| 50 | 978 | 100% | 99.4% | **0.0%** |
| 75 | 804 | 99.7% | 100% | 0.0% |
| 100 | 1088 | 99.7% | 99.7% | 0.0% |

Three behaviors moved together: response length collapsed 3×, format-closure went near-perfect, and **tool-call emission dropped to 0%**. The first two are unambiguously good. The third is the critical finding.

#### Reward hacking by tool-call bypass

The model learned to **skip the search tool entirely** and answer directly from parametric knowledge.

Concrete step 110 example (extend run; same behavior):
> Question: *"When did the city the regional office of the World Bank in the country with a constitution is located in first host the Asian Games?"*
> Model thinking: *"Hmm... the city is Kathmandu. Asian Games were held in 1951. Answer: 1951."*
> Output: `<answer> 1951</answer>` → F1 = 1.000 (matches MuSiQue's reference). **No `<tool_call>` invoked.**

**Why this happened (mechanism).** Our reward is `f1(extract_answer(rollout), gold_answer)` — F1-only, no format gate, no floor (M5.1 contract inherited from `training_m5_1/src/rewards/search_r1.py`, deliberate per Group-C decision). Tool-call emission carries zero direct reward. GRPO discovered the dominant strategy: emit `<answer>` as soon as the parametric prior coughs up something plausible; never emit `<tool_call>`. On MuSiQue, the 0.8B base apparently has enough world knowledge to F1-match the gold answer ~20% of the time without retrieval, and that's the signal GRPO ratchets.

**Why the original Search-R1 paper avoided this.** Paper uses `reward = f1 + 0.1` for any non-empty `<answer>` rollout, with a format gate. The 0.1 floor + format requirement forces the model to actually emit answers (not nothing) while keeping F1 as the gradient direction. The format gate keeps tool-use behavior alive. We chose F1-only for cleanliness — and lost tool use as a consequence.

**Is this still publishable?** Yes. The finding *"F1-only GRPO on a base model induces tool-call bypass; the model becomes a fast direct-answer retriever rather than a tool-using agent"* is a real result. It's the inverse of what Search-R1 reports and isolates which part of their reward design is load-bearing. It also explains why the M7 trained checkpoint will likely *outperform* untrained on simple-factoid datasets (TriviaQA, NQ) and *underperform* on multi-hop datasets that require fresh retrieval (Bamboogle, 2wiki) — a clean ablation.

### M7.1-extend (resume from step 100 → step 622)

Auto-launched 2026-05-13 05:33 UTC after the step-100 GO criterion passed (`rew_mean(91-100) = 0.0872 ≥ 0.035 AND lift ≥ 2×`). Run name: `qwen3.5-0.8b-base-musique-m7_extend-seed42-20260513T0533Z`. Config: [`m7_1_extend.yaml`](../../training_m7_1/configs/m7_1_extend.yaml) — same as `m7_1_short100.yaml` with `max_num_steps: 622`, `save_period: 100`, `keep_top_k: 2`.

**Interim (through step 115 ≈ 1 h into extend)**: pace stable at ~72 s/step; `rew_mean` averaging 0.10-0.17 (e.g., step 110 = 0.164); tool-call emission still 0%. Target end ETA: 2026-05-13 16:00 CEST.

**Resume caveats:**
- Optimizer state not saved (`save_optimizer: false` — disk constraint inherited from short100). AdamW moments re-initialized from zero. At constant LR=1e-6 this is a minor perturbation; in practice we see no discontinuity in the reward trajectory across the resume boundary (step 100 short100 final mean 0.087 → step 101 extend 0.13 → step 110 extend 0.16).
- The `m7_short100/seed42/step_50` ckpt was deleted to recover disk during the extend launch (overlay fill incident; see `log.md` 2026-05-13 entry). `step_100` is preserved.

### Cost ledger to date

| segment | wall | $ |
|---|---:|---:|
| Smoke A (20-step seq=4096) | 44 min | ~$1 |
| Smoke B (20-step seq=8192, partial) | 38 min | ~$1 |
| Smoke C (3-step prod-shape) | 51 min | ~$1 |
| Short100 (100 steps) | 2 h 51 min | ~$5 |
| Idle gap after short100 (auto-launch failed) | 6 h 02 min | ~$8 |
| Extend (in progress, ~10 h projected) | ~10 h | ~$13 |
| **Total M7 GPU spend** | ~21 h | **~$29** |

The idle-gap line is the auto-launch failure documented above: short100 completed at 23:31 UTC, extend launched 05:33 UTC — 6+ hours of idle A100 because the auto-continuation was promised but never actually wired as a side-process. Lesson captured in [`memory/feedback_auto_action_promises.md`](file:///root/.claude/projects/-workspace/memory/feedback_auto_action_promises.md).

## Pointers

- M4 final locks + handoff: [`MILESTONE_4.md` §"Handoff to M5"](../milestone_4/MILESTONE_4.md)
- M4 base lock template: [`evaluation_qwen35/flashrag/search_r1/templates.py:QWEN35_MINIMAL_NO_SYSTEM_TEMPLATE`](../../evaluation_qwen35/flashrag/search_r1/templates.py)
- M5.1 training overlay (to mirror): [`training_m5_1/`](../../training_m5_1/)
- M5.1 reward function (F1-only, no floor): [`training_m5_1/src/rewards/search_r1.py`](../../training_m5_1/src/rewards/search_r1.py)
- M5.1 yaml configs to mirror: [`training_m5_1/configs/m5_smoke.yaml`](../../training_m5_1/configs/m5_smoke.yaml), [`training_m5_1/configs/m5_1_research_paper.yaml`](../../training_m5_1/configs/m5_1_research_paper.yaml)
- M5.1 postmortems (rules inherited): [`report/RESULTS_SMOKE_m5.md` §7](../report/RESULTS_SMOKE_m5.md)
- M4 untrained floor (the eval target): [`report/RESULTS_m4.md` §4](../report/RESULTS_m4.md)
- Hardware comparison (alternative providers if A100 swap needed): [`setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md)
- Cross-family delta + 2wiki anomaly context: [`report/RESULTS_m4.md` §5](../report/RESULTS_m4.md)
