---
title: M6 Phase 3b: Publication framing brief
tags: [milestone, m6, phase3, publication, venue]
source: internal
created: 2026-05-16
updated: 2026-05-17
---

# Publication Framing Brief

> Phase 3b deliverable of [`MILESTONE_6.md`](MILESTONE_6.md). Names: the one-sentence takeaway, the realistic venue and acceptance probability, the 200-word related-work paragraph, and the three threats-to-validity a reviewer will hammer. Companion: [`PICKED_PAIR.md`](PICKED_PAIR.md).
>
> Conditional on the user's pick-#2 decision in [`PICKED_PAIR.md` §6](PICKED_PAIR.md#6-decision-needed-from-the-user). Both variants laid out below; merge to one once the pick is made.

**Status**: provisional 2026-05-16; venue ranges sharpened 2026-05-17 with M5.1-landed evidence (lean-drift-lean cycling + empirical chain-flip data add a fourth contribution). Freezes on user pick.

---

## 1. Title (working)

**If C + R**: *"Pareto-defensible minimalism: an outcome-only reward and prompt-design transfer for sub-1B retrieval-augmented RL"*.

**If C + M**: *"The small-model frontier of search-augmented RL: removing the partial-credit floor and switching to median-baseline GRPO"*.

## 2. One-sentence reviewer-quotable takeaway

(Same as [`PICKED_PAIR.md` §2](PICKED_PAIR.md#2-combined-takeaway-sentence-the-reviewer-quotable-line); repeated here as the load-bearing artifact of the framing.)

| Pair | Takeaway |
|---|---|
| **C + R** | "At sub-1B scale on retrieval-augmented multi-hop QA, the F1-only reward is Pareto-defensible against the ReSearch partial-credit floor, and the no-example + decision-rules prompt design transfers across model family, both at half the response budget of the standard recipe." |
| **C + M** | "At sub-1B scale on retrieval-augmented multi-hop QA, the partial-credit reward floor and the small-G mean-baseline both compress training signal; removing the floor and switching to median-baseline GRPO together define the small-model frontier for search-augmented RL." |

## 3. Venue and acceptance probability

Anchored to the [`CONVERSATION_CONTEXT.md` §3 row 6](CONVERSATION_CONTEXT.md#3-working-assumption-from-2026-05-11-critical-review-confirmed-2026-05-16) honest ranges (NeurIPS main 5-10 %, ICLR / ACL / EMNLP main 15-25 %, Findings / Workshop 50 %+, ICLR blogpost very plausible if the message is "simple recipe + characterisation").

### If C + R (defensive) (sharpened 2026-05-17)

| Venue | Probability | Why |
|---|---|---|
| **ICLR blogpost** (like JustRL) | very plausible (~85-92 %) | M5.1-landed evidence gives three blogpost-friendly figures (trajectory matrix, lean-drift-lean cycling, chain-flip-vs-reward scatter); JustRL template fit |
| **Workshop** (NeurIPS SoLaR / NeurIPS R0-FoMo / ICLR Reasoning) | 70-82 % | Lean-drift-lean cycling is workshop-receptive training-dynamics content |
| **Findings of ACL / EMNLP** | 50-65 % | Empirical chain-flip rate + trajectory matrix clear the Findings bar; lean-drift-lean cycling is a venue-positive secondary finding |
| **ICLR / ACL / EMNLP main** | 22-32 % | Mechanism-named takeaway + concrete trajectory + self-stabilisation finding; still hampered by head-to-head and seed-variance gaps |
| **NeurIPS main** | 10-18 % | The added self-stabilisation finding helps; head-to-head + scale + seed gaps still dominate. Do not optimise for NeurIPS main |

**Recommended target**: **ICLR blogpost** first, with workshop as the fallback. Findings is a stretch but worth a Cycle 2 attempt if the blogpost rejection comes back with positive reviews.

### If C + M (ambitious) (sharpened 2026-05-17)

| Venue | Probability | Why |
|---|---|---|
| **ICLR / ACL / EMNLP main** | 22-32 % (positive M outcome) | Mechanism-named takeaway + concrete efficiency claim + closes the MC-GRPO-on-search literature gap + the M5.1 self-stabilisation finding strengthens the training-dynamics section |
| **NeurIPS main** | 10-18 % (positive outcome only) | Compute / scale still insufficient for NeurIPS main competitiveness even with positive outcome |
| **Workshop** (NeurIPS R0-FoMo / Agentic-RL workshops) | 65-78 % | Positive outcome is clear novelty; negative outcome is still useful narrowing |
| **Findings of ACL / EMNLP** | 42-58 % | Either outcome fits Findings |
| **ICLR blogpost** | 55-68 % | Less natural fit than C + R; blogposts favour single-takeaway. Still plausible after the M5.1 lean-drift-lean addition |

**Recommended target**: **ICLR / ACL main** first (under positive M outcome) with **Workshop or Findings** as fallback. NeurIPS is dishonest as a primary target.

## 4. Related-work paragraph (200 words, paper-ready)

> *Retrieval-augmented RL post-training emerged from Search-R1 (Jin et al., 2025) and ReSearch (Chen et al., 2025), which interleave `<search>` tool calls with GRPO at 3B-32B. R1-Searcher (Song et al., 2025) shows base-model cold-start needs SFT below 7B. By 2026 the field bifurcates along three axes. **Dense reward shaping** (IGPO, TIPS, IG-Search, Search-P1) replaces outcome-only reward with per-turn or per-search-step signals; we instead defend outcome-only F1 as deliberate minimalism, motivated by our Phase-1 finding that the ReSearch partial-credit floor masks tool-use signal at sub-1B. **Rollout efficiency** (Tree-GRPO at ICLR 2026, DIVA-GRPO, AERO, BAPO) attacks wasted rollouts; we [characterise the small-rollout regime / address it via the MC-GRPO median baseline]. **Reward-design ablations** are rare: concurrent work (Wang et al., 2026, arXiv:2602.19526) studies EM vs F1 vs F1+penalties in Search-R1 at 3B+ and finds F1+action-penalties wins; we ablate at 0.8B without format reward, isolating the partial-credit floor. At sub-1B specifically, DGPO (Liu et al., 2025) reports pure RL collapses search behaviour, motivating distillation guidance; we show the shrink-and-improve regime is consistent with their failure mode but not predictive of it under the right reward.*

The bracketed clause is the only pair-conditional sentence; the rest is shared.

## 5. Three threats-to-validity the reviewer will hammer

### Threat 1: Small-scale generalisation

*"How do you know this transfers to 7B?"*

**Response**: we don't, and we don't claim it. The paper's scope is sub-1B retrieval-augmented RL under realistic resource constraints (single A100, ~\$300 / experiment, no SFT cold-start budget). Three pieces of evidence anchor the scope as a legitimate regime, not an apology:
1. DGPO (Liu et al., 2025; [arXiv:2508.20324](https://arxiv.org/abs/2508.20324)) reports 0.5-1B agentic RAG fails under pure RL and proposes distillation guidance.
2. Our Phase-1 base-model bootstrap evidence ([`PHASE_1_SALVAGE.md` §4b](PHASE_1_SALVAGE.md), [`../report/RESULTS_m0_b.md`](../report/RESULTS_m0_b.md)) confirms 5/5 base-model attempts at 0.6B fail to bootstrap without SFT.
3. The M5.1 shrink-and-improve trajectory at 0.8B ([`PHASE_1_SALVAGE.md` Finding 3](PHASE_1_SALVAGE.md#finding-3--multi-hop-tool-use-produces-a-shrink-and-improve-rl-regime), [`../report/RESULTS_SMOKE_m5.md` §6](../report/RESULTS_SMOKE_m5.md)) inverts the long-CoT regime familiar from math reasoning RL.

**Limit**: the paper does not run a 2B confirmatory; that is acknowledged future work. The 2B candidate (S) was dropped from the picked pair for budget reasons ([`CANDIDATE_EXPERIMENTS.md` §"Candidate S"](CANDIDATE_EXPERIMENTS.md#candidate-s--scale-up-to-qwen35-2b-from-an-m51-checkpoint)).

### Threat 2: Head-to-head with concurrent work

*"Why no Tree-GRPO comparison?"*

**Response**: Tree-GRPO (Yu et al., 2025; [arXiv:2509.21240](https://arxiv.org/abs/2509.21240)) is on a different axis: prefix-sharing across siblings to reduce rollout budget. The two are orthogonal and stackable on paper. A fair head-to-head requires porting Tree-GRPO's tree-search rollout to NeMo-RL (verl-side currently), which the wall-clock budget does not accommodate before 2026-06-10. We acknowledge this as a limitation and identify the head-to-head as the natural follow-up.

*"Why no comparison with [arXiv:2602.19526](https://arxiv.org/html/2602.19526v1)?"*: they ablate at 3B+ with format and action-penalty rewards; our setup is 0.8B with no format reward and isolates the 0.1 partial-credit floor specifically. The two ablations are complementary points; we cite them directly.

### Threat 3: Novelty bound by method-transfer framing

*"This is just method-transfer from math reasoning RL to retrieval."*

**Response**: the paper does not claim a new algorithm. The contribution is **characterisation** of a regime that has not been characterised: sub-1B retrieval-augmented RL at single-A100 budget with outcome-only reward. The mechanism story (the 0.1 floor masks tool-use signal; F1-only is Pareto-defensible at small scale; the shrink-and-improve regime inverts long-CoT) is the project's own and grounded in Phase-1 measurements. The picked-pair experiments turn the strongest Phase-1 observation into an ablation. Method-transfer is not a weakness when the contribution is characterising a regime, not inventing a method.

**Backup framing if reviewer doesn't accept this**: re-pitch the paper as a *negative-result-plus-recipe* ICLR blogpost (the JustRL template); single-finding format does not need method novelty.

## 6. Cross-references

- Pair: [`PICKED_PAIR.md`](PICKED_PAIR.md).
- Landscape and related-work scaffolding: [`LANDSCAPE_TABLE_2026-05.md`](LANDSCAPE_TABLE_2026-05.md), [`LITERATURE_GAP_AUDIT_2026-05-16.md`](LITERATURE_GAP_AUDIT_2026-05-16.md).
- Project measurements the paper cites in §1 (motivation): [`PHASE_1_SALVAGE.md`](PHASE_1_SALVAGE.md), [`DATA_AUDIT_PHASE_1.md`](DATA_AUDIT_PHASE_1.md).
- M5.1 baseline: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md), [`../report/RESULTS_M5_1_H200.md`](../report/RESULTS_M5_1_H200.md).
- Last supervisor brief (predecessor to the M0-to-6 story brief): [`../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md).
