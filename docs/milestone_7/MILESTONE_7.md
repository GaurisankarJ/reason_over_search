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
| M7.0 Setup | 2026-05-12 (this commit) | 30-45 min | ⏳ next |
| M7.0.5 Verification | After M7.0 | 15 min | pending |
| M7.0.7 Smoke | After M7.0.5 | 30-60 min on A100 | pending |
| M7.1 Production | Conditional on smoke pass | 10-13 d | conditional |
| M7.2 Eval | After M7.1 | 2.5 h | conditional |

**Sequencing vs M5.1-prod-a3**: M5.1-prod-a3 is currently paused awaiting user authorization per [`TODO_2026-05-12.md`](../todo/TODO_2026-05-12.md). M7 runs FIRST on the current box; M5.1-a3 launches afterward (either on same box if M7 fails/completes, or on a second box if budget allows parallelism).

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
