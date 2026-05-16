---
title: M6 Phase 3a: Picked pair
tags: [milestone, m6, phase3, picked-pair, decision]
source: internal
created: 2026-05-16
updated: 2026-05-16
---

# Picked Pair

> Phase 3a deliverable of [`MILESTONE_6.md`](MILESTONE_6.md). The two experiments to run after M5.1 lands, with defended rationale. Companion: [`PUBLICATION_FRAMING.md`](PUBLICATION_FRAMING.md).
>
> Input: [`CANDIDATE_EXPERIMENTS.md`](CANDIDATE_EXPERIMENTS.md) §"Pair construction" filtered the pair space down to two viable options: **C + R (defensive)** and **C + M (ambitious)**.

**Status**: provisional 2026-05-16, pending one user decision (defensive vs ambitious; see §3 below). All other commitments frozen.

---

## 1. The pair

### Pick #1 (frozen): **Candidate C: Reward-shape ablation**

Three runs at fixed prompt × fixed seed × reward ∈ {F1 + 0.1-floor, F1-only, EM-only} on the M5.1 recipe at Qwen3.5-0.8B. Full spec in [`CANDIDATE_EXPERIMENTS.md` §"Candidate C"](CANDIDATE_EXPERIMENTS.md#candidate-c--reward-shape-ablation).

**Why this is locked**:
- Closes [`PHASE_1_SALVAGE.md` Finding 1](PHASE_1_SALVAGE.md#finding-1--the-01-partial-credit-floor-masks-tool-use-signal): the project's strongest mechanism-attached observation. Without this ablation, Finding 1 is observation grade only.
- Post-audit, the experiment is reframed (per [`LITERATURE_GAP_AUDIT_2026-05-16.md` §"Tier 1 #1"](LITERATURE_GAP_AUDIT_2026-05-16.md#1-how-to-train-your-deep-research-agent--arxiv260219526-feb-2026)) as the **small-model (0.8B), single-GPU-budget, no-format-reward complementary point** to [arXiv:2602.19526](https://arxiv.org/html/2602.19526v1). The 0.1 partial-credit floor specifically is not ablated in their work; the mechanism story (Phase-1 Finding 1) is the project's own.
- Both positive and negative outcomes are publishable. Positive: ablation grade evidence that the 0.1 floor masks tool-use signal at 0.8B. Negative: "at 0.8B on MuSiQue training, reward-shape is dominated by other factors" narrows the claim space.

### Pick #2 (decision pending): **Candidate R (defensive)** or **Candidate M (ambitious)**

Both candidates pair coherently with Candidate C. The choice is venue-dependent, per the Phase-3 selection criterion 4 ("address the strongest reviewer objection") and per the [`CONVERSATION_CONTEXT.md` §3 row 4-6](CONVERSATION_CONTEXT.md#3-working-assumption-from-2026-05-11-critical-review-confirmed-2026-05-16) honest venue ranges.

**Option R: 2-seed Qwen3.5-0.8B prompt-pair replication** ([`CANDIDATE_EXPERIMENTS.md` §"Candidate R"](CANDIDATE_EXPERIMENTS.md#candidate-r--replication-of-the-v0-prompt-pair-on-qwen35-08b-with-2-seeds)):
- Closes Finding 2's cross-family transfer gap + the project's load-bearing **no-multi-seed-anywhere** gap.
- Combined cost with C: **~\$190-280 / 14-21 d**. Fits budget envelope with margin.
- Publication-defensibility: high (closes two project gaps; gives variance bars on one experiment in the chapter).
- Venue fit: **Findings, Workshop, ICLR blogpost** ranges; harder to land at NeurIPS / ICLR / ACL main (no efficiency or algorithmic contribution; relies on the small-scale-regime characterisation framing carrying the paper).
- Risk profile: low. No new infrastructure; prompts already exist in the v0 record.

**Option M: MC-GRPO at G=2 on M5.1 recipe** ([`CANDIDATE_EXPERIMENTS.md` §"Candidate M"](CANDIDATE_EXPERIMENTS.md#candidate-m--mc-grpo-at-g2-on-m51-recipe)):
- No Phase-1 gap directly. The hook is novelty against [LANDSCAPE row C3](LANDSCAPE_TABLE_2026-05.md#2a-direct-competitors-must-engage-in-related-work) ("MC-GRPO not yet applied to search-tool").
- Combined cost with C: **~\$260-380 / 19-27 d**. At the top of the envelope; 27 d wall-clock is over the 18-d combined budget unless H200/B200 lane is used.
- Publication-defensibility: medium-high. Positive outcome is the highest-novelty candidate. Negative outcome is a useful narrowing.
- Venue fit: positive outcome plausibly **ICLR / ACL / EMNLP main 15-25 %** with the right framing; negative outcome maps onto Findings or Workshop.
- Risk profile: medium-high. M5.1 at G=2 may not converge; pre-flight smoke required. The reviewer objection (head-to-head vs Tree-GRPO) is not closed by this pair alone; it must be addressed in the related-work paragraph.

## 2. Combined takeaway sentence (the reviewer-quotable line)

Both pair options support a different takeaway. **Picking the pair fixes the takeaway, not the other way around**: the user must pick before this sentence freezes.

| Pair | One-sentence takeaway |
|---|---|
| **C + R** | "At sub-1B scale on retrieval-augmented multi-hop QA, the F1-only reward is Pareto-defensible against the ReSearch partial-credit floor, and the no-example + decision-rules prompt design transfers across model family, both at half the response budget of the standard recipe." |
| **C + M** | "At sub-1B scale on retrieval-augmented multi-hop QA, the partial-credit reward floor and the small-G mean-baseline both compress training signal; removing the floor and switching to median-baseline GRPO together define the small-model frontier for search-augmented RL." |

## 3. Recommendation

**Default to C + R unless the user explicitly opts for the ambitious target.**

Reasoning:
1. **Venue honesty**: the [`CONVERSATION_CONTEXT.md` §3 row 6](CONVERSATION_CONTEXT.md#3-working-assumption-from-2026-05-11-critical-review-confirmed-2026-05-16) NeurIPS-main estimate is 5-10 %. C + M shifts that to ~15-25 % under a positive outcome only; C + R sits at Findings/Workshop/blogpost ranges (50 %+). C + R clears the thesis bar (which is the load-bearing constraint to 2026-06-15) more reliably.
2. **Cross-cutting gap repair**: R closes the no-multi-seed-anywhere gap on at least one experiment, addressing the load-bearing Reviewer 2 concern across the rest of the project.
3. **Risk control**: M requires G=2 to converge, requires a pre-flight smoke, and requires a head-to-head against Tree-GRPO in related work. R requires neither.
4. **Time to thesis**: C + R fits in 14-21 d wall-clock; M5.1 lands around 2026-05-22 to 2026-05-25 on the H200 substrate per [`../report/RESULTS_M5_1_H200.md`](../report/RESULTS_M5_1_H200.md). C + R could close by 2026-06-08 to 2026-06-15 (right at the 06-10 experimentation cutoff). C + M could slip past 2026-06-10.

**Reasoning to override and pick C + M**: if the user's priority is the post-thesis publication (NeurIPS 2026 / ICLR 2027) rather than the thesis itself, the higher-novelty C + M is worth the risk. The thesis chapter can still be written from C alone if M fails to converge; the negative result is a chapter section either way.

## 4. Pre-flight requirements (regardless of pick #2)

- **M5.1 must land** with reward floor ≥ 0.20 on MuSiQue training (otherwise the picked pair is a remediation experiment, not a follow-up: see [`CONVERSATION_CONTEXT.md` §6 threat #1](CONVERSATION_CONTEXT.md#6-threats-and-unknowns)).
- **EM-only smoke** (50 steps, single seed, M5.1 prompt) before committing to the full EM-only run. Confirm reward signal is not too sparse to bootstrap at 0.8B. If the smoke fails, drop EM-only and run C as a 2-variant ablation (F1+0.1 vs F1-only) only.
- **Rollout JSONL extraction pipeline** built before run #1 so that per-rollout reward histograms (currently a [`DATA_AUDIT_PHASE_1.md` §5 #2](DATA_AUDIT_PHASE_1.md#5-cross-cutting-gaps-that-affect-everything) gap) are produced as a side effect of the runs, not as a separate effort.
- **2-seed protocol locked** if Option R is picked (single-seed of the same prompt is the wrong control; explicit seeds = {42, 1337} per Phase-1 convention).
- **Pre-flight smoke for M5.1-at-G=2** if Option M is picked. 100 steps; reward must show a non-zero advantage signal by step 50.

## 5. Threats to validity (consolidated)

| # | Threat | Mitigation |
|---|---|---|
| 1 | M5.1 final reward below 0.20 → pair becomes remediation | Wait for M5.1 to land before committing rented hours; if remediation needed, run pure-M5.1-debug then re-plan |
| 2 | Single-seed of pick #1 (C) on the F1-only variant → variance unbounded | (a) Use Option R to get 2 seeds elsewhere in the chapter; (b) note as a limit in the threats section of the paper |
| 3 | EM-only fails to bootstrap at 0.8B → C reduces to 2-variant | Pre-flight smoke (§4); document the drop honestly |
| 4 | [arXiv:2602.19526](https://arxiv.org/html/2602.19526v1) gets cited by reviewer as full scoop | Reframing in [`PUBLICATION_FRAMING.md` §"Related work"](PUBLICATION_FRAMING.md) makes the differentiation explicit |
| 5 | Reviewer 2 demands head-to-head vs Tree-GRPO | Acknowledge as future work; the chapter framing is "characterisation of the small-rollout regime", not "rollout efficiency contribution" |
| 6 | Bamboogle n=125 noise muddies single-benchmark conclusions in R | Report 7-benchmark sign test, not per-benchmark gap; cite the [`../archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md`](../archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md) postmortem |

## 6. Decision needed from the user

Pick one:

- **Option A: C + R (defensive, Findings-grade)**. Frozen rationale above; defaults here unless overridden.
- **Option B: C + M (ambitious, ICLR-main-grade-under-positive-outcome)**. Pick this if post-thesis publication priority outweighs thesis-deadline risk.

Once the user picks, the picked pair freezes and [`PUBLICATION_FRAMING.md`](PUBLICATION_FRAMING.md) §"Venue and probability" is updated to reflect the choice. [`MILESTONE_6.md` §"Phase 3"](MILESTONE_6.md#phase-3-pick-the-pair--publication-framing-brief-2026-05-16--2026-05-18) task list is then marked complete.

## Cross-references

- Pair filter: [`CANDIDATE_EXPERIMENTS.md` §"Pair construction"](CANDIDATE_EXPERIMENTS.md#pair-construction).
- Findings that motivate the pair: [`PHASE_1_SALVAGE.md`](PHASE_1_SALVAGE.md), [`DATA_AUDIT_PHASE_1.md`](DATA_AUDIT_PHASE_1.md).
- Competitive landscape: [`LANDSCAPE_TABLE_2026-05.md`](LANDSCAPE_TABLE_2026-05.md), [`LITERATURE_GAP_AUDIT_2026-05-16.md`](LITERATURE_GAP_AUDIT_2026-05-16.md).
- Downstream: [`PUBLICATION_FRAMING.md`](PUBLICATION_FRAMING.md).
- M5.1 baseline: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md), [`../report/RESULTS_M5_1_H200.md`](../report/RESULTS_M5_1_H200.md).
