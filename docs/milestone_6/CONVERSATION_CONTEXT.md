---
title: Conversation Context — Milestone 6 (literature review + experiment planning)
tags: [milestone, m6, conversation-context, snapshot]
source: internal
created: 2026-05-11
updated: 2026-05-16
---

# Conversation Context: Milestone 6

> Living snapshot of M6 — literature landscape pass, candidate experiment short-list, picked pair, publication framing. Update when something material changes; freeze older state below the line.
>
> Sibling snapshots: [`../report/CONVERSATION_CONTEXT.md`](../report/CONVERSATION_CONTEXT.md) (thesis writing), [`../research/CONVERSATION_CONTEXT.md`](../research/CONVERSATION_CONTEXT.md) (frozen 2026-05-04 literature arm + algorithm/systems), [`../training/CONVERSATION_CONTEXT.md`](../training/CONVERSATION_CONTEXT.md) (training pipeline).

**Last updated**: 2026-05-16 (Phase 1, 2, 3 artifacts frozen; supervisor brief + presentation outline authored; one user decision pending on pick #2)

---

## 1. Status (one paragraph)

M6 is the planning-and-positioning milestone created on 2026-05-11 to run in parallel with M5.1's wall-clock (~10-13 d projected per [`../log.md` 2026-05-11](../log.md)). M5.1 is the **ReSearch-paper recipe** (GRPO, F1-only reward on `<answer>X</answer>`, MuSiQue, G=5) ported to NeMo-RL and applied to **Qwen3.5-0.8B on 1× A100-80GB** ([`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md)). Once M5.1 lands, M6 needs to have picked the next two experiments **and** the publication framing already, otherwise the budget and timeline (~\$1000 total, experimentation must close 2026-06-10) cannot accommodate a separate planning phase after M5.1. The objective is therefore: (a) a frozen 2026 landscape table, (b) a frozen candidate experiments short-list, (c) a picked pair with rationale, (d) a publication-framing brief. M6 produces **no GPU work**; it is read-and-write only. The triggering insight is the 2026-05-11 critical review (this conversation): the framing the user reached for instinctively — *"basic paper recreation (Search-R1 + ReSearch) + MC-GRPO efficiency improvement"* — is partly factually wrong (M5 is ReSearch-paper alone, not a Search-R1 + ReSearch combination) and partly under-defended for NeurIPS (concurrent ICLR 2026 work in Tree-GRPO + JustRL covers adjacent ground). M6 exists to close both gaps before any further GPU is rented.

## 2. Active question

> **What two experiments, run after M5.1 lands, would produce a finding mechanistically defensible enough to publish at NeurIPS / ICLR / ACL/EMNLP main, and within a combined \$400 / 18 days?**

Sub-questions M6 must answer:
- What is the *mechanism-named* one-sentence takeaway the work supports? (Not a recipe stack.)
- Does the picked pair survive head-to-head against Tree-GRPO ([arXiv:2509.21240](https://arxiv.org/abs/2509.21240), ICLR 2026) and the JustRL "tricks may hurt" thesis ([arXiv:2512.16649](https://arxiv.org/abs/2512.16649))?
- What is the realistic venue with honest acceptance probability — and is that venue worth optimising the experiments for?

## 3. Working assumption from 2026-05-11 critical review (confirmed 2026-05-16)

Captured here as the starting point, not as a settled answer. All six rows survived the 2026-05-16 literature gap audit ([`LITERATURE_GAP_AUDIT_2026-05-16.md`](LITERATURE_GAP_AUDIT_2026-05-16.md)) unchanged; one row added below (#7) for Candidate C exposure to [arXiv:2602.19526](https://arxiv.org/html/2602.19526v1).

| Claim | Verdict (2026-05-11) | Implication |
|---|---|---|
| "M5 = Search-R1 + ReSearch combined" | **Inaccurate**. M5 is ReSearch-paper recipe alone (GRPO, F1, MuSiQue, G=5); Search-R1's role is the 7-dataset eval. | Rewrite the framing in any draft and supervisor brief. |
| "MC-GRPO has not been applied to search-tool RL" | **True**. [MC-GRPO](https://arxiv.org/abs/2601.22582) abstract + reported experiments are math (Qwen3-1.7B on GSM8K). | Genuine gap. But the mechanism for *why* search-tool should benefit (retrieval reward variance, longer trajectories, smaller G under memory pressure) needs to be made explicit. |
| "JustRL works as a control for the recipe" | **Methodologically true, but not a contribution**. JustRL itself ([arXiv:2512.16649](https://arxiv.org/abs/2512.16649)) is the published "simple beats complex" finding; we can use it as a control, not as a novelty hook. | Don't pitch the control as the paper. |
| "Two efficiency improvements → faster training, same result = good paper" | **Insufficient** for NeurIPS main. DIVA-GRPO already claims 2.55× step + 1.76× wall-clock; Tree-GRPO claims 1/4 rollout budget at ICLR 2026. AERO ([2602.14338](https://arxiv.org/abs/2602.14338)) claims 48 % compute reduction at 1.5B; BAPO ([2602.20722](https://arxiv.org/html/2602.20722v1)) adds replay-buffer angle (both Feb 2026, found in 2026-05-16 audit). | Either head-to-head one of them or shift the contribution from "efficiency" to "characterisation of small-rollout regime for search-tool RL". |
| "0.8B scale is a strength (reproducibility)" | **Mixed → strengthened**. DGPO ([2508.20324](https://arxiv.org/abs/2508.20324), v4 = Apr 2026) explicitly argues 0.5–1B models can't do agentic search via RL alone — gives mechanistic support to the small-scale scope and to Phase-1 Finding 3's shrink-and-improve regime. | Pick the lane; defend in the introduction; cite DGPO as small-scale-mechanism prior. |
| "NeurIPS main is the realistic target" | **No**. Realistic ranges: NeurIPS main 5-10 %, ICLR / ACL / EMNLP main 15-25 %, Findings / Workshop 50 %+, ICLR blogpost (like JustRL) very plausible if the message is "simple recipe + characterisation". | Pick the venue before picking the experiments. |
| **"Candidate C reward-shape ablation is novel"** (added 2026-05-16) | **Partial scoop**. [arXiv:2602.19526](https://arxiv.org/html/2602.19526v1) "How to Train Your Deep Research Agent?" (Feb 2026) directly studies EM vs F1 reward in Search-R1; finds naive F1 less stable than EM, F1+action-penalties wins. Their work is 3B+; ablates F1 vs EM, not the 0.1 partial-credit floor; doesn't carry Phase-1 Finding 1's 0.1-floor-masks-signal mechanism. | Rewrite Candidate C as the **small-model (0.8B), single-GPU-budget, no-format-reward** complementary point. Phase-1 Finding 1 remains the project's own. |

## 4. Decisions captured (M6-strand)

| Decision | When | Rationale |
|---|---|---|
| Create M6 as a planning-only milestone parallel to M5.1 wall-clock | 2026-05-11 | Budget + timeline doesn't allow planning after M5.1; positioning must be ready when M5.1 lands |
| Three frozen artifacts: landscape table, candidate experiments, picked pair + publication-framing brief | 2026-05-11 | Each phase produces a defensible artifact; later phases cannot be defended without earlier ones |
| Stop saying "Search-R1 + ReSearch combined" | 2026-05-11 | Inaccurate; M5 is ReSearch alone |
| Don't anchor on NeurIPS main as the realistic target | 2026-05-11 | 5-10 % acceptance probability with current plan; Findings / Workshop / ICLR blogpost are honest targets |
| Candidate C must engage with [arXiv:2602.19526](https://arxiv.org/html/2602.19526v1) before any GPU is rented | 2026-05-16 | Partial scoop of EM-vs-F1 ablation; rewrite as small-model / single-GPU / no-format-reward complementary point |
| Candidate D rationale (if picked) must engage with [arXiv:2603.27226](https://arxiv.org/html/2603.27226) | 2026-05-16 | Negative-result paper on E2H for deductive reasoning; reframes E2H-for-RAG ablation as "testing whether the critique generalises" |
| Pick #1 frozen: **Candidate C reward-shape ablation** at the M5.1 recipe | 2026-05-16 | Closes Phase-1 Finding 1's gap; both positive and negative outcomes publishable; rewritten post-audit as small-model / single-GPU / no-format-reward complementary point. See [`PICKED_PAIR.md`](PICKED_PAIR.md) |
| Pick #2 default: **Candidate R 2-seed prompt-pair replication** (defensive); user opt-in for Candidate M (ambitious) | 2026-05-16 | C + R fits 14-21 d wall-clock and clears the 2026-06-10 cutoff with 3-6 d margin; C + M slips to cutoff. Targets: C + R = ICLR-blogpost / Workshop / Findings; C + M = ICLR / ACL main 15-25 % under positive outcome only |
| Frame the paper as a **characterisation of the small-rollout regime for search-tool RL at sub-1B**, not as a recipe-stack | 2026-05-16 | Mechanism-named takeaway; turns method-transfer objection into a non-objection (the contribution is the regime, not a new algorithm) |

## 5. Compute and budget (M6-relevant)

M6 itself uses **no GPU compute**. The constraints on what M6's picked pair can prescribe:

- **Remaining budget**: ~\$500-600 USD after M5.1 lands (assumes M5.1 costs ~\$400-500 on A100, per the live 10-13 d ETA at \$1.50/h).
- **Remaining wall-clock window**: 2026-05-11 → 2026-06-10 = ~30 days. M5.1 consumes ~13 of those; M6 planning is parallel; the picked pair has ~17 days of wall-clock for both experiments together.
- **Hardware lanes** (per [`../setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md)): 1× A100 / H100 SXM (~2× faster than A100) / B200 (~6-7× faster than A100, but Blackwell sm_100 + v2 worker venv compat caveat unresolved). The picked pair should specify which lane.

## 6. Threats and unknowns

1. **M5.1 might not converge to expected level** — if final M5.1 reward is well below ReSearch's 0.40-0.47 floor, the picked pair must include a remediation experiment (not a novelty experiment), changing the M6 conclusion entirely.
2. **Tree-GRPO infrastructure cost is unknown** — if a fair head-to-head against Tree-GRPO requires porting their tree-search rollout to NeMo-RL, the engineering cost may exceed the wall-clock budget. M6 needs a cost estimate before committing.
3. **The "right venue" is moving** — NeurIPS 2026 deadlines, ICLR 2027 deadlines, EMNLP 2026 deadlines all sit in the post-thesis window. The picked pair must produce a finding usable for a venue 3-6 months after thesis defense, not just for the thesis itself.

## 7. Doc inventory (M6 folder)

| File | Status |
|---|---|
| [`MILESTONE_6.md`](MILESTONE_6.md) | Authored 2026-05-11; milestone definition; external sources list refreshed 2026-05-16 |
| [`CONVERSATION_CONTEXT.md`](CONVERSATION_CONTEXT.md) | This file; living |
| [`LOG.md`](LOG.md) | Authored 2026-05-11; append-only; 2026-05-16 entry added |
| [`PHASE_1_SALVAGE.md`](PHASE_1_SALVAGE.md) | Authored 2026-05-11; three Phase-1 findings as paper motivation, with file:line cites + limits |
| [`DATA_AUDIT_PHASE_1.md`](DATA_AUDIT_PHASE_1.md) | Authored 2026-05-11; deep-dive on what is measured cleanly vs what is missing |
| [`LITERATURE_GAP_AUDIT_2026-05-16.md`](LITERATURE_GAP_AUDIT_2026-05-16.md) | Authored 2026-05-16; 18 papers (2025-11 → 2026-05) missing from the frozen 2026-05-04 pass; 5 Tier-1 (must-engage), 4 Tier-2 (dense-credit cluster), 9 Tier-3 (additive) |
| [`LANDSCAPE_TABLE_2026-05.md`](LANDSCAPE_TABLE_2026-05.md) | **Frozen 2026-05-16**; 25 papers, 3-axis positioning analysis, 200-word related-work paragraph drafted |
| [`CANDIDATE_EXPERIMENTS.md`](CANDIDATE_EXPERIMENTS.md) | **Frozen 2026-05-16**; five candidates filtered to two viable pairs (C + R defensive, C + M ambitious) |
| [`PICKED_PAIR.md`](PICKED_PAIR.md) | **Frozen 2026-05-16 except pick #2**; C locked, R is default with user opt-in for M |
| [`PUBLICATION_FRAMING.md`](PUBLICATION_FRAMING.md) | **Provisional 2026-05-16**; both pair variants drafted; freezes on user pick #2 |
| [`PRESENTATION_OUTLINE.md`](PRESENTATION_OUTLINE.md) | Authored 2026-05-16; terse 17-slide outline for talk + thesis defense |
| [`STORYLINE.md`](STORYLINE.md) | **Living** 2026-05-16; the running narrative, fact-checked, with the math example for the F1+0.1 floor + standardised step-1000 plots + critical NeurIPS assessment |
| [`../report/SUPERVISOR_MEETING_2026-05-16_m0_to_6.md`](../report/SUPERVISOR_MEETING_2026-05-16_m0_to_6.md) | Authored 2026-05-16; M0-to-M6 narrative; successor to the 2026-05-07 brief |

## 9. Salvage summary (from 2026-05-11 deep dive)

Three Phase-1 findings survive critical scrutiny and can serve as the introduction's motivation paragraph (full detail in [`PHASE_1_SALVAGE.md`](PHASE_1_SALVAGE.md), gap analysis in [`DATA_AUDIT_PHASE_1.md`](DATA_AUDIT_PHASE_1.md)):

| # | Finding | Status | Gap to publication |
|---|---|---|---|
| 1 | 0.1 partial-credit floor masks tool-use signal (3-6 pp tool/no-tool gap vs 9 pp prompt-driven swing) | Observation across 9 cross-prompt runs | **High** — needs A/B reward-shape ablation. Closes with M6 Candidate C (3 runs × ~3-5 d each). |
| 2 | No-example + decision-rules prompt pareto-dominates with-example (held-out EM +9 %, ACC +12 %, F1 +14 %, half budget) | Single-seed Qwen3-0.6B + 51,713-item held-out eval | **Medium-High** — needs 2 seeds + Qwen3.5-0.8B replication. Closes with 2× ~5-day runs on H100. |
| 3 | "Shrink-and-improve" RL regime: length/calls 3× compress while reward grows (inverts long-CoT regime) | Single-run M5.1 mid-training (step 10-16 of 622) | **Medium** — needs M5.1 completion (free) + ideally a long-CoT baseline (1× M5.1 cost). |

**The audit elevates Candidate C (reward-shape ablation) to likely pick #1** because it is the only candidate that (a) closes an unfilled-by-the-literature gap, (b) is supported by a Phase-1 motivation paragraph the project already owns, and (c) directly addresses ReSearch's own "biggest gap to exploit" admission (per [`../papers/2503.19470_research.md` §"Takeaways"](../papers/2503.19470_research.md#L77)). Pick #2 deferred to Phase 3.

**Three findings *not* salvageable** for a contribution but useful for related-work / methods chapters: M3 +52 % EM lift (replication of ReSearch at smaller scale), M1 ±2.5 pp Search-R1 reproduction (replication), Qwen3.5-0.8B < Qwen3-0.6B cross-family degradation (observation without mechanism — could become a 1-page finding with a tokenizer/template diagnostic, see [`DATA_AUDIT_PHASE_1.md` §4a](DATA_AUDIT_PHASE_1.md#4a-qwen35-08b-hybrid--qwen3-06b-hybrid-on-untrained-tool-use-results_m4md-5)).

## 10. Critical gaps (cross-cutting, from [`DATA_AUDIT_PHASE_1.md` §5](DATA_AUDIT_PHASE_1.md#5-cross-cutting-gaps-that-affect-everything))

Affect *all* salvage findings; the picked pair should close at least one:

1. **No multi-seed runs anywhere** in Phase-1. Single-seed Bamboogle (n=125) ~3 pp SE is the load-bearing example.
2. **No per-rollout-level analysis** — all metrics are batch means; rollout JSONLs exist but aren't extracted.
3. **No 0.8B / Qwen3.5 prompt sweep** — the Phase-1 prompt finding doesn't yet transfer cross-family.
4. **No format-OK-but-F1=0 rate by run** — 0.1-floor share is back-calculated, not directly observed.
5. **No baseline that converges to a different regime** — shrink-and-improve "signature" claim is one-sided without a long-CoT contrastive.

## 8. Cross-references

- M5.1 baseline: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md), [`../milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md), live trajectory [`../report/RESULTS_SMOKE_m5.md` §6](../report/RESULTS_SMOKE_m5.md).
- Frozen literature arm (M6 extends): [`../research/SURVEY_FOCUSED.md`](../research/SURVEY_FOCUSED.md), [`../research/PARADIGM_REVIEW.md`](../research/PARADIGM_REVIEW.md), [`../research/RUNTIME_EFFICIENCY.md`](../research/RUNTIME_EFFICIENCY.md), [`../research/INTEGRATION_GUIDE.md`](../research/INTEGRATION_GUIDE.md).
- Last canonical chapter brief: [`../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](../report/SUPERVISOR_MEETING_2026-05-07_m0_to_3.md).
- Hardware + cost lanes: [`../setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md).
- Active project TODO: [`../todo/TODO_2026-05-11.md`](../todo/TODO_2026-05-11.md).

---

## Historical (frozen)

*(Nothing frozen yet — M6 was just created.)*
