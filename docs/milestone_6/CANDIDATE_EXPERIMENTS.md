---
title: M6 Phase 2: Candidate experiment short-list
tags: [milestone, m6, phase2, experiments, planning]
source: internal
created: 2026-05-16
updated: 2026-05-16
---

# Candidate Experiment Short-list

> Phase 2 deliverable of [`MILESTONE_6.md`](MILESTONE_6.md). Five candidates with cost / wall-clock / success criterion / failure-mode / conflict-with-concurrent-work. Each candidate is informative under both positive and negative outcomes (per the Phase-3 selection criterion).
>
> Inputs: [`LANDSCAPE_TABLE_2026-05.md`](LANDSCAPE_TABLE_2026-05.md), [`PHASE_1_SALVAGE.md`](PHASE_1_SALVAGE.md), [`DATA_AUDIT_PHASE_1.md`](DATA_AUDIT_PHASE_1.md), [`LITERATURE_GAP_AUDIT_2026-05-16.md`](LITERATURE_GAP_AUDIT_2026-05-16.md). Hardware costs read from [`../setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md). Budget envelope: ≤ \$400 combined, ≤ 18 days combined wall-clock (per [`CONVERSATION_CONTEXT.md` §5](CONVERSATION_CONTEXT.md#5-compute-and-budget-m6-relevant)).

**Status**: frozen 2026-05-16. Selection of the pair is in [`PICKED_PAIR.md`](PICKED_PAIR.md).

---

## Summary table

| # | Name | Closes which Phase-1 gap | Cost (1× A100) | Wall-clock | Novelty | Defensibility under negative outcome |
|---|---|---|---:|---|---|---|
| **C** | Reward-shape ablation (F1+0.1 / F1-only / EM-only) on M5.1 recipe at 0.8B | Finding 1 (0.1 floor masks tool-use signal) | ~\$110-160 | ~9-13 d total | Medium (post-audit; partial scoop by [arXiv:2602.19526](https://arxiv.org/html/2602.19526v1)) | "F1-only matches F1+0.1 at 0.8B" is a publishable finding either way |
| **R** | 2-seed Qwen3.5-0.8B prompt-pair replication (`p3_decide_no_ex` vs `p1_basic_w_ex`) | Finding 2 (no-example + decision-rules pareto) + Cross-cutting gap #1 (no multi-seed anywhere) | ~\$80-120 | ~5-8 d total | Low-Medium | Pareto-flip on Qwen3.5 is a 1-page finding; pareto-confirm is a 1-section finding |
| **M** | MC-GRPO at G=2 (median baseline) on M5.1 recipe | None directly; addresses LANDSCAPE row C3 method-transfer | ~\$150-220 | ~10-14 d | Medium-High (genuine gap per [`CONVERSATION_CONTEXT.md` §3](CONVERSATION_CONTEXT.md#3-working-assumption-from-2026-05-11-critical-review)) | Positive: efficiency contribution; negative: "median doesn't help on retrieval reward" is still a finding |
| **D** | E2H curriculum on M5.1 (NQ → HotpotQA → MuSiQue, ~200 steps/stage) | None directly | ~\$130-190 | ~9-12 d | Low (post-audit; [Rethinking E2H](https://arxiv.org/html/2603.27226) is the new framing) | "E2H critique generalises (or not) to RAG" is a publishable finding under both outcomes |
| **S** | Scale-up: M5.1 recipe at Qwen3.5-2B for the last 200 steps of a checkpoint resumed from M5.1 | None directly; addresses LANDSCAPE Axis C reviewer objection | ~\$240-340 | ~12-16 d | Low | "Recipe transfers (or not) to 2B" is the reviewer-killer answer to "does this scale" |

## Detailed entries

### Candidate C: Reward-shape ablation

**Question**: at 0.8B on the M5.1 recipe, does removing the ReSearch 0.1 partial-credit floor change tool-use behaviour and held-out EM, in either direction?

**Setup**: three runs at fixed prompt × fixed seed × reward ∈ {F1 + 0.1-floor (paper-faithful), F1-only (M5.1 control), EM-only (strict)}. M5.1 recipe otherwise unchanged. Each run ≤ 622 steps (paper schedule) or earlier on plateau-detected stop. Eval at the M3 protocol on the 7-benchmark suite + the 51,713-item full-data hold-out.

**What this closes**: [`PHASE_1_SALVAGE.md` Finding 1](PHASE_1_SALVAGE.md#finding-1--the-01-partial-credit-floor-masks-tool-use-signal) and [`DATA_AUDIT_PHASE_1.md` §1](DATA_AUDIT_PHASE_1.md#1-finding-1--01-partial-credit-floor-masks-tool-use-signal). The "floor masks signal" claim becomes ablation grade.

**Success criterion**: directional separation ≥ 3 pp EM on at least one held-out benchmark *across* the three reward variants. Cleanest positive: EM-only > F1-only > F1+0.1, supporting the "floor masks tool-use" mechanism. Cleanest negative: all three within 1 pp; concludes "at 0.8B on MuSiQue training the reward-shape lever is dominated by other factors" (still publishable).

**Failure-mode interpretation**: even the negative outcome adds the first ablation grade evidence on this question at sub-1B; [arXiv:2602.19526](https://arxiv.org/html/2602.19526v1) ablates at 3B+ with action-penalty variants, not the partial-credit floor. The negative result narrows the claim space; the positive result fills the literature gap.

**Conflict with concurrent work**: partial scoop by [arXiv:2602.19526](https://arxiv.org/html/2602.19526v1) "How to Train Your Deep Research Agent?" (Feb 2026; per [`LITERATURE_GAP_AUDIT_2026-05-16.md` §"Tier 1 #1"](LITERATURE_GAP_AUDIT_2026-05-16.md#1-how-to-train-your-deep-research-agent--arxiv260219526-feb-2026)). They ablate EM vs F1 vs F1+action-penalty in Search-R1 at 3B+; M5.1's ablation is at 0.8B without format-reward and isolates the 0.1 partial-credit floor. **Reframed novelty**: small-model, single-GPU, no-format-reward complementary point; Phase-1 Finding 1's "0.1 floor masks tool-use signal" mechanism story stays the project's own.

**Cost**: three runs × ~3-5 days each on 1× A100 SXM at the M5.1 step-time profile (~30-40 min/step at steady state per [`../report/RESULTS_M5_1_H200.md`](../report/RESULTS_M5_1_H200.md) but ~25 % faster on A100 given the H200/A100 ratio from [`../setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md)). At \$1.20-1.50 / h on Vast: ~\$110-160 total.

**Threats**: (i) reward-shape effect could be confounded by which checkpoints are evaluated (mid-training vs converged); mitigation: eval at fixed step counts. (ii) the EM-only variant could fail to bootstrap (reward sparsity); mitigation: pre-flight 50-step smoke before committing to the full run.

### Candidate R: Replication of the v0 prompt-pair on Qwen3.5-0.8B with 2 seeds

**Question**: does the v0 finding ([`PHASE_1_SALVAGE.md` Finding 2](PHASE_1_SALVAGE.md#finding-2--the-no-example--decision-rules-prompt-is-a-pareto-improvement); `p3_decide_no_ex` pareto-dominates `p1_basic_w_ex` on Qwen3-0.6B) transfer to Qwen3.5-0.8B, with a variance bar from a second seed?

**Setup**: four runs total: 2 prompts × 2 seeds. Same M5.1 recipe otherwise. Eval at the M4 protocol on the 7-benchmark suite.

**What this closes**: Finding 2's cross-family transfer gap + the project's load-bearing **no-multi-seed-anywhere** gap ([`DATA_AUDIT_PHASE_1.md` §5 #1](DATA_AUDIT_PHASE_1.md#5-cross-cutting-gaps-that-affect-everything)). A second seed on any one experiment changes the project's variance claim from "single-seed observation" to "two-seed indicative".

**Success criterion**: a pareto-direction agreement (EM, ACC, F1) between Qwen3-0.6B (M3.1) and Qwen3.5-0.8B (this experiment) at p < 0.05 on a sign test across 7 benchmarks. Cleanest positive: pareto-win on 0.8B at half response budget. Cleanest negative: pareto reverses on 0.8B, which is then evidence of cross-family non-transfer of the v0 prompt finding.

**Failure-mode interpretation**: either outcome is publishable as a 1-page finding ("prompt design transfers across model family") or 1-section finding ("prompt design is family-specific at small scale"). Negative outcome is also a useful negative result against the prompt-engineering-uber-alles narrative.

**Conflict with concurrent work**: none. Prompt-design replication at sub-1B is unexplored in the 2026 literature scan.

**Cost**: four runs × ~1.5-2 d each at the M5.1 step-time profile, 622 steps each = ~6-8 d wall-clock total if serial; halve if two A100s available. \$1.20-1.50 / h × ~50-80 h = ~\$80-120.

**Threats**: (i) prompt-mode misalignment between Phase-1 v0 prompts and the M5.1 `qwen35_minimal` lock; mitigation: re-prompt both seeds with the v0 prompts directly, do not use the M4.2 minimal lock. (ii) Bamboogle n=125 noise (~3 pp SE) muddying a single-benchmark conclusion; mitigation: report the 7-benchmark sign test, not the per-benchmark gap.

### Candidate M: MC-GRPO at G=2 on M5.1 recipe

**Question**: does the median-baseline of MC-GRPO ([arXiv:2601.22582](https://arxiv.org/abs/2601.22582)) close the small-G performance gap on retrieval-induced reward variance (as opposed to math-correctness variance)?

**Setup**: two runs: vanilla M5.1 at G=2 vs MC-GRPO (G+1=3 rollouts, median of G=2 used as baseline, pivot excluded from policy update) at G=2. Otherwise the M5.1 recipe.

**What this closes**: no Phase-1 gap directly. Addresses the LANDSCAPE row C3 method-transfer claim ("MC-GRPO has not been applied to search-tool RL") with a concrete result.

**Success criterion**: MC-GRPO at G=2 reaches the M5.1-at-G=5 reward floor within 80 % of the wall-clock of M5.1-at-G=5. Cleanest positive: efficiency claim with a mechanism (reward variance is the bottleneck at small G in search-tool RL). Cleanest negative: "MC doesn't help on retrieval reward variance"; still a finding, narrows the small-G scope.

**Failure-mode interpretation**: either outcome is publishable. The positive result is the highest-novelty outcome in the candidate list (NeurIPS/ICLR territory under [`CONVERSATION_CONTEXT.md` §3](CONVERSATION_CONTEXT.md#3-working-assumption-from-2026-05-11-critical-review)'s honest acceptance ranges). The negative result is a useful narrowing finding.

**Conflict with concurrent work**: Tree-GRPO (LANDSCAPE C1) is the natural reviewer comparator on the rollout-efficiency axis. Tree-GRPO uses prefix-sharing, not median-baseline; the two are orthogonal and stackable on paper. Without a head-to-head against Tree-GRPO in the same paper, reviewer 2 will hammer. Mitigation in the framing: "MC-GRPO is the small-G mechanism for retrieval reward variance; Tree-GRPO is the prefix-sharing mechanism for rollout reuse; we ablate the former".

**Cost**: two runs × ~5-7 d each on 1× A100 at G=2 (faster per step than G=5 but the run needs more steps to converge). ~\$150-220 total.

**Threats**: (i) M5.1 at G=2 may not converge; mitigation: pre-flight 100-step smoke. (ii) MC-GRPO requires G+1=3 rollouts even at G=2, so the per-step cost is 1.5× vanilla G=2; the "efficiency" claim depends on convergence-step reduction outweighing the per-step overhead. (iii) the project does not yet have a Tree-GRPO implementation in NeMo-RL; a head-to-head against Tree-GRPO requires porting their framework which busts the wall-clock budget.

### Candidate D: E2H curriculum on M5.1 (NQ → HotpotQA → MuSiQue)

**Question**: does easy-to-hard curriculum help search-augmented RL at 0.8B, in light of the [Rethinking E2H](https://arxiv.org/html/2603.27226) negative result on deductive reasoning?

**Setup**: one run with curriculum stages of ~200 steps each on NQ → HotpotQA → MuSiQue (E2H, [arXiv:2506.06632](https://arxiv.org/abs/2506.06632) ordering), vs the M5.1 MuSiQue-only baseline. Wall-clock controlled equal.

**What this closes**: no Phase-1 gap directly. Addresses the LANDSCAPE Axis A reviewer question on curriculum.

**Success criterion**: ≥ 3 pp EM lift on the 7-benchmark held-out at equal compute, OR a clear negative (no lift, with the trajectory showing curriculum-induced stage friction). Both outcomes are publishable.

**Failure-mode interpretation**: post-audit reframing per [`LITERATURE_GAP_AUDIT_2026-05-16.md` §"Tier 3 #17"](LITERATURE_GAP_AUDIT_2026-05-16.md#tier-3--additive-citations-cheap-upgrades-to-related-work): the experiment becomes "does the [Rethinking E2H](https://arxiv.org/html/2603.27226) critique generalise to search-augmented RAG?" rather than "does E2H help?". A negative result here is a publishable confirmation of the critique on a new task; a positive result is a publishable counter-example.

**Conflict with concurrent work**: directly engaged with [arXiv:2603.27226](https://arxiv.org/html/2603.27226). Less novelty than C / R / M because the framing now follows their critique rather than leading.

**Cost**: one run + comparing to existing M5.1 baseline = ~5-6 d wall-clock. ~\$130-190.

**Threats**: (i) NQ rollouts may be too easy to give a learning signal (saturate at high reward immediately); mitigation: pre-flight 50-step smoke. (ii) curriculum schedule introduces a new hyperparameter (stage length, fade pattern); mitigation: lock at ~200 steps / stage with no fade for first run.

### Candidate S: Scale-up to Qwen3.5-2B from an M5.1 checkpoint

**Question**: does the M5.1 recipe transfer to 2B at half the run-from-scratch budget by warm-starting from the converged 0.8B checkpoint?

**Setup**: load M5.1 final checkpoint, do model surgery (Qwen3.5-0.8B → Qwen3.5-2B initialised from the 0.8B weights via standard architecture-extrapolation), continue training for 200 steps on the same data. Eval at the M3 protocol.

**What this closes**: no Phase-1 gap directly. Addresses the LANDSCAPE Axis C reviewer objection "does this scale to 7B?" with a 2B data point.

**Success criterion**: ≥ 5 pp held-out EM lift over the M5.1 0.8B final at the 2B-final-vs-0.8B-final comparison. Cleanest negative: 2B does not improve over 0.8B at equal training compute; this would be a striking finding (recipe not bottlenecked by scale at this regime).

**Failure-mode interpretation**: positive outcome answers the scale-up question. Negative outcome flips the "small-scale-as-strength" framing into "the bottleneck at this regime is not scale; here are the candidates", which is a useful negative result.

**Conflict with concurrent work**: LiteResearcher (LANDSCAPE C6) is at 4B; this is the closest published reference. The 2B point sits between the project's 0.8B and LiteResearcher's 4B and would tighten the small-model-search-agent landscape.

**Cost**: one M5.1-shape run at 2B is ~2.5× the per-step cost (parameter count) and may need 30-40 % more steps; ~10-14 d wall-clock at A100, longer if also fitting the 200 % memory bump. ~\$240-340. **Largest single candidate**.

**Threats**: (i) model surgery for Qwen3.5-0.8B → Qwen3.5-2B is not standard; the standard approach is from-scratch GRPO on Qwen3.5-2B with no warm-start. From-scratch at 2B for 622 steps under the project's compute lane is infeasible. (ii) checkpoint-extrapolation methods (e.g. weight tiling, depth-up) introduce their own ablation. (iii) the experiment is engineering-heavy and eats time that could go into Phase-3 framing.

## Pair construction

Per [`MILESTONE_6.md` Phase-3 selection criteria](MILESTONE_6.md#phase-3-pick-the-pair--publication-framing-brief-2026-05-16--2026-05-18), the picked pair must:

1. Each experiment informative under both outcomes.
2. Together support one mechanism-named takeaway sentence.
3. Combined ≤ \$400 / ≤ 18 d.
4. Address the strongest reviewer objection (currently "head-to-head against Tree-GRPO" per [`CONVERSATION_CONTEXT.md` §3](CONVERSATION_CONTEXT.md#3-working-assumption-from-2026-05-11-critical-review)).
5. At least one experiment must not require new infrastructure beyond M5.1.

Pair options and their fit:

| Pair | Combined cost | Combined wall-clock | Mechanism-named takeaway | Addresses Tree-GRPO objection | Requires new infra |
|---|---:|---|---|---|---|
| **C + R** (reward-shape + prompt-replication) | ~\$190-280 | ~14-21 d | "At 0.8B, F1-only is a Pareto-defensible reward design; the v0 prompt finding transfers cross-family" | No (Tree-GRPO addressed in related-work only) | None |
| **C + M** (reward-shape + MC-GRPO) | ~\$260-380 | ~19-27 d | "At 0.8B on retrieval reward variance, the partial-credit floor and the small-G mean-baseline both compress signal: minimal recipe + median baseline is the small-model frontier" | Partial (MC-GRPO ≠ Tree-GRPO but engages the efficiency axis) | None |
| **C + D** (reward-shape + E2H) | ~\$240-350 | ~18-25 d | "At 0.8B, neither reward-shape nor curriculum is load-bearing: the bottleneck is upstream" | No | None |
| **R + M** (prompt-replication + MC-GRPO) | ~\$230-340 | ~15-22 d | (Two unrelated findings; weak takeaway) | Partial | None |
| **C + S** (reward-shape + scale-up) | ~\$350-500 | ~21-29 d | "At 0.8B, F1-only matches F1+0.1; at 2B, recipe lifts (or doesn't): small-scale is the right regime" | No | Model surgery for warm-start |

**Constraints filter**: C + S violates the budget envelope; R + M lacks a mechanism-named takeaway; C + D is post-audit dominated by C + R (C + R closes two project gaps; C + D engages a Tier-3 paper). Two candidate pairs survive: **C + R (defensive)** and **C + M (ambitious)**.

The choice between C + R and C + M is venue-dependent and is made in [`PICKED_PAIR.md`](PICKED_PAIR.md).

## Cross-references

- Sources: [`LANDSCAPE_TABLE_2026-05.md`](LANDSCAPE_TABLE_2026-05.md), [`PHASE_1_SALVAGE.md`](PHASE_1_SALVAGE.md), [`DATA_AUDIT_PHASE_1.md`](DATA_AUDIT_PHASE_1.md), [`LITERATURE_GAP_AUDIT_2026-05-16.md`](LITERATURE_GAP_AUDIT_2026-05-16.md).
- Hardware cost lanes: [`../setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md).
- M5.1 baseline being built on: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md), [`../report/RESULTS_M5_1_H200.md`](../report/RESULTS_M5_1_H200.md).
- Downstream: [`PICKED_PAIR.md`](PICKED_PAIR.md), [`PUBLICATION_FRAMING.md`](PUBLICATION_FRAMING.md).
