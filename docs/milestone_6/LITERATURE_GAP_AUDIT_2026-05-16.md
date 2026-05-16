---
title: Literature Gap Audit — 2025-11 → 2026-05 papers missing from the 2026-05-04 frozen pass
tags: [milestone, m6, phase1, literature, gap-audit]
source: internal
created: 2026-05-16
updated: 2026-05-16
---

# Literature Gap Audit (2025-11 → 2026-05)

> Deep-search audit dated 2026-05-16, covering papers published between 2025-11 and 2026-05 that the frozen 2026-05-04 literature pass missed and that materially affect M6's framing. Companion docs: [`PHASE_1_SALVAGE.md`](PHASE_1_SALVAGE.md) (what we own), [`DATA_AUDIT_PHASE_1.md`](DATA_AUDIT_PHASE_1.md) (what is missing from our data), this doc (what is missing from our literature).
>
> Seeds the Phase-1 `LANDSCAPE_TABLE_2026-05.md` ([`MILESTONE_6.md` §"Method"](MILESTONE_6.md#phase-1-literature-landscape-pass-2026-05-11--2026-05-13)) — does not replace it. The landscape table requires reading each paper end-to-end; this audit identifies *which* papers must be on the read list.

## Method

Four parallel agent-driven web searches (2026-05-16), each scoped to one axis:

1. Agentic search RL / retrieval-augmented RL / multi-hop QA with RL (search-tool angle).
2. GRPO algorithm variants / reward design / sample efficiency (algorithm angle).
3. Small-LM RL post-training / resource-constrained RLVR / base-model dependence (scale angle).
4. Curriculum / data scheduling / RL training infrastructure / multi-turn agent systems (engineering angle).

Each agent was given the 2026-05-04 frozen citation list as "already-known" and instructed to flag only papers (a) published 2025-11 or later, (b) not in the frozen list, (c) materially affecting M6 positioning. Cap: 8–12 papers per axis.

Results were deduplicated across axes (some papers — e.g. **2604.17931 (LiteResearcher)** — appeared in multiple agent reports). Final list below: 18 papers, grouped by impact on M6.

## Bottom-line verdict

The frozen pass is **broadly current**. No paper found fully scoops the project's framing ("ReSearch-paper recipe ported to NeMo-RL on Qwen3.5-0.8B with F1-only reward + small-model RAG-RL recipe ablation"). The Milestone 6 critical review's own observation that "the 2026 competitive landscape on agentic-RL is denser than the 2026-05-04 frozen survey captured" ([`LOG.md` 2026-05-11](LOG.md)) is now quantified: **18 missing entries**, of which **1 partially scoops a planned ablation** and **9 require explicit related-work engagement** before submission.

Phase-1 Salvage findings ([`PHASE_1_SALVAGE.md`](PHASE_1_SALVAGE.md)) are **not at risk** — none of the new papers measures the same things on the same prompt-sweep data.

## Tier 1 — Direct overlap with the planned picked pair (must engage)

### 1. How to Train Your Deep Research Agent — [arXiv:2602.19526](https://arxiv.org/html/2602.19526v1) (Feb 2026)

**Why it matters**: directly studies **EM vs F1 reward in Search-R1**. Finds naive F1 training less stable than EM; F1 + lightweight action-level penalties beats EM. This is the single most consequential omission.

**Impact**: **PARTIAL SCOOP of Candidate C** ([`MILESTONE_6.md` Phase 2 table row C](MILESTONE_6.md#phase-2-candidate-experiment-short-list-2026-05-13--2026-05-15) — reward-shape ablation: F1 vs F1+format vs partial-credit). The project's planned F1+0.1 vs F1-only vs EM-only ablation must now be re-pitched as **the small-model, single-GPU-budget, no-format-reward complementary point** — not as a novel ablation. The mechanism story (Phase-1 Finding 1 — "the 0.1 floor masks tool-use signal", [`PHASE_1_SALVAGE.md` §"Finding 1"](PHASE_1_SALVAGE.md#finding-1--the-01-partial-credit-floor-masks-tool-use-signal)) still holds and remains the project's own.

**Action**: must-cite. Rewrite Candidate C rationale to differentiate on (a) 0.8B scale (their work is 3B+), (b) F1+0.1 ablation specifically (they ablate F1 vs EM, not the partial-credit floor), (c) the cross-prompt 3-6 pp tool/no-tool gap from Phase 1 as confirming the load-bearing nature of the floor.

### 2. AERO — Adaptive Efficient Rollout Optimization — [arXiv:2602.14338](https://arxiv.org/abs/2602.14338) (Feb 2026)

**Why it matters**: GRPO with adaptive rollout, selective rejection, Bayesian posterior to avoid zero-advantage dead zones. Reports **48% compute reduction on 1.5B**, 47% on 7B.

**Impact**: **COMPETE on the small-rollout-budget framing**. The project's "characterise the small-rollout regime" framing ([`CONVERSATION_CONTEXT.md` §3](CONVERSATION_CONTEXT.md#3-working-assumption-from-2026-05-11-critical-review) row 4) remains defensible but must explicitly differentiate from AERO's *adaptive budget* mechanism. Closely related to the project's MC-GRPO interest (median-baseline for small G).

**Action**: must-cite. Add to the related-work paragraph as "AERO addresses the same wasted-rollout problem from an adaptive-budget angle; we characterise the regime instead of altering the budget."

### 3. BAPO — Buffer Matters: Off-Policy RL for LLM Reasoning — [arXiv:2602.20722](https://arxiv.org/html/2602.20722v1) (Feb 2026)

**Why it matters**: off-policy RLVR with dynamic batch re-evaluation of historically hard samples; lower-bound improvement guarantee. Replay-buffer angle on the same "wasted rollout" problem.

**Impact**: **PARTIAL COMPETE on small-rollout framing**. Together with AERO, BAPO covers much of the small-rollout-regime conceptual space from a "fix the budget" angle the project doesn't currently address.

**Action**: cite-and-distinguish. Either adopt the replay angle in the picked pair (if cheap) or position the project's "characterise the regime first, alter it second" as the deliberate scoping choice.

### 4. DGPO — Can Compact LMs Search Like Agents? — [arXiv:2508.20324](https://arxiv.org/abs/2508.20324) (Aug 2025, v4 = Apr 2026)

**Why it matters**: explicitly targets **0.5–1B agentic RAG**; argues **pure RL collapses small-model search behaviors**, proposes distillation-guided GRPO to preserve them.

**Impact**: **MECHANISTIC SUPPORT for Phase-1 Finding 3** (shrink-and-improve regime — [`PHASE_1_SALVAGE.md` §"Finding 3"](PHASE_1_SALVAGE.md#finding-3--multi-hop-tool-use-produces-a-shrink-and-improve-rl-regime)). The "small models can't do agentic search via RL alone" failure mode is exactly the kind of mechanism paper M6's brief asks for. Validates the project's small-scale scope as a *regime with its own dynamics*, not as a low-resource apology.

**Action**: must-cite as motivation paragraph anchor. Phase-1 Finding 3's "shrink-and-improve" regime can be contextualised as the *favourable* small-model dynamic that DGPO's distillation-guidance also targets — different lever, same regime.

### 5. LiteResearcher — Scalable Agentic-RL Framework for Deep Research — [arXiv:2604.17931](https://arxiv.org/abs/2604.17931) (Apr 2026)

**Why it matters**: closest published reference to the project's goal. 4B search-agent via lightweight agentic-RL framework reaches 71.3% GAIA / 78.0% Xbench.

**Impact**: **PARTIAL OVERLAP** in the small-model search-agent niche. Just above the 2B ceiling; their framework / training-cost numbers are the closest published anchor.

**Action**: must-cite. Differentiate on (a) Qwen3.5-0.8B vs their 4B (4× smaller), (b) NeMo-RL substrate vs their custom framework, (c) single-A100 budget vs their unstated infra, (d) MuSiQue training vs their broader deep-research mix.

## Tier 2 — Dense-credit-assignment cluster (positioning shift)

These four together establish "dense potential-based reward shaping" as a 2026 trend the project's F1-only reward is **implicitly arguing against**. None scoop the project, but together they require the F1-only choice to be defended as **deliberate minimalism** rather than an unconsidered default.

### 6. IGPO — Information Gain-based Policy Optimization — [arXiv:2510.14967](https://arxiv.org/abs/2510.14967) (Oct 2025, ICLR 2026 Poster)

Per-turn dense reward = marginal increase in policy's probability of producing the correct answer. Direct competitor to F1-only outcome reward.

### 7. TIPS — Turn-Level Information-Potential Reward Shaping — [arXiv:2603.22293](https://arxiv.org/html/2603.22293v1) (Mar 2026)

Dense per-turn reward = teacher-model log-likelihood gain. +11.8% EM / +13.6% F1 over PPO on Qwen-2.5-7B across 7 QA benchmarks.

### 8. IG-Search — Step-Level Information Gain for Search-Augmented Reasoning — [arXiv:2604.15148](https://arxiv.org/abs/2604.15148) (Apr 2026)

Per-search-step IG measured as confidence improvement vs random-doc counterfactual; per-token advantage modulation in GRPO. Evaluates on HotpotQA / 2Wiki / MuSiQue / Bamboogle — overlaps the project's eval suite exactly.

### 9. Search-P1 — Path-Centric Reward Shaping for Stable Agentic RAG — [arXiv:2602.22576](https://arxiv.org/html/2602.22576v1) (Feb 2026, Tencent)

Order-agnostic step coverage + dual-track scoring vs reference planners; +7.7 avg accuracy over Search-R1 on Qwen2.5-3B/7B. Closest published baseline-to-Search-R1 paper.

**Tier-2 action**: add as a single "outcome-vs-dense reward" axis in `LANDSCAPE_TABLE_2026-05.md`. Position the F1-only choice as deliberate minimalism (Phase-1 Finding 1 — the 0.1 floor compresses signal at 0.8B, so adding more reward components is unlikely to help and risks reward hacking).

## Tier 3 — Additive citations (cheap upgrades to related-work)

### Algorithm / theory

| # | Paper | Date | Why cite |
|---|---|---|---|
| 10 | **[Balanced Aggregation / 2605.04077](https://arxiv.org/abs/2605.04077)** | May 2026 | Token-vs-sequence aggregation bias; validated on **Qwen3-1.7B** (project's size class). Affects how loss-mean should be reported. |
| 11 | **[EqLen / 2604.17328](https://arxiv.org/html/2604.17328v1)** | Apr 2026 | Identifies length-inconsistency failure modes (truncation-induced entropy collapse, learning-tax). Directly informs small-rollout-regime characterisation. |
| 12 | **[Why GRPO Needs Normalization / 2601.23135](https://arxiv.org/abs/2601.23135)** | Jan 2026 | First rigorous theory for GRPO std-normalization. Pairs with existing shrinkage cite (2511.03710). |
| 13 | **[EBPO / 2602.05165](https://arxiv.org/html/2602.05165)** | Feb 2026 | Empirical-Bayes posterior-shrinkage extension of 2511.03710. One-sentence cite. |
| 14 | **[REAL / 2602.05630](https://arxiv.org/abs/2602.05630)** | Feb 2026 | Reframes verifiable rewards as classification labels; +6.7% Pass@1 over DAPO on 1.5B. Touches the partial-credit conceptual ground from a different angle. |
| 15 | **[Path Not Taken / 2511.08567](https://arxiv.org/abs/2511.08567)** | Nov 2025 | Three-Gate Theory — geometric explanation for why RLVR updates **off principal SVD directions**. Mechanistic foundation for the project's PEFT-for-RLVR (2512.23165) and LoRA-plasticity (2601.06677) cites. |

### Curriculum / data

| # | Paper | Date | Why cite |
|---|---|---|---|
| 16 | **[Learning from Less / 2604.18381](https://arxiv.org/abs/2604.18381)** | Apr 2026 (MLSys 2026 oral) | **5× sample efficiency** from mixed-complexity training. Maps directly onto the project's ~$1000-budget framing. |
| 17 | **[Rethinking Easy-to-Hard / 2603.27226](https://arxiv.org/html/2603.27226)** | Mar 2026 | **Negative result on E2H** for deductive reasoning. Strict upgrade to Candidate D ([`MILESTONE_6.md` Phase 2 table row D](MILESTONE_6.md#phase-2-candidate-experiment-short-list-2026-05-13--2026-05-15)) — reframes the picked-pair ablation as "testing whether the E2H critique generalises to search-augmented RAG", not "demonstrating curriculum helps". |

### Infrastructure (free wins for the M5.1 substrate)

| # | Paper | Date | Why cite |
|---|---|---|---|
| 18 | **[Speculative Decoding in NeMo-RL+vLLM / 2604.26779](https://arxiv.org/html/2604.26779v1)** | Apr 2026 (NVIDIA) | EAGLE-3 + MTP speculative decoding **inside NeMo-RL with vLLM backend**. **1.8× rollout throughput at 8B, projected ~2.5× end-to-end.** Free throughput win on the M5.1 substrate; should be noted as future-work / orthogonal-optimisation lever. |

## What survived the audit unchanged

- **Phase-1 Salvage findings** ([`PHASE_1_SALVAGE.md`](PHASE_1_SALVAGE.md)): all three remain unique to the project's data (no paper measures the same things on the same v0 prompt sweep).
- **Milestone 6 framing decisions** ([`CONVERSATION_CONTEXT.md` §3](CONVERSATION_CONTEXT.md#3-working-assumption-from-2026-05-11-critical-review)): all six survive: "stop saying Search-R1+ReSearch combined", "NeurIPS main 5–10%", "pick venue before experiments", "characterise small-rollout regime not method-transfer", "MC-GRPO is a genuine gap on search-tool", "JustRL is a control not a contribution".
- **The framing of the picked pair** (Candidate C reward-shape + a second pick TBD): Candidate C requires the rewrite noted under Tier 1 #1 (2602.19526) but the core question — does dropping the 0.1 floor help tool-use signal at 0.8B specifically — remains unanswered in the new literature.

## Recommended updates to other M6 docs

1. **[`MILESTONE_6.md` §"Sources of truth" §7 "External"](MILESTONE_6.md#sources-of-truth-in-order)**: append the 5 Tier-1 papers and the 4 Tier-2 cluster papers.
2. **[`CONVERSATION_CONTEXT.md` §3](CONVERSATION_CONTEXT.md#3-working-assumption-from-2026-05-11-critical-review)**: add a row reflecting that 2602.19526 partially scoops Candidate C.
3. **[`LOG.md`](LOG.md)**: append 2026-05-16 entry (this audit; outcome).
4. **Phase 1 deliverable `LANDSCAPE_TABLE_2026-05.md`**: when authored, must include all 18 entries from this audit as table rows.
5. **Phase 2 deliverable `CANDIDATE_EXPERIMENTS.md`**: when authored, the Candidate C rationale must engage with 2602.19526; Candidate D rationale must engage with 2603.27226.

## Cross-references

- M6 milestone definition: [`MILESTONE_6.md`](MILESTONE_6.md).
- M6 conversation snapshot: [`CONVERSATION_CONTEXT.md`](CONVERSATION_CONTEXT.md).
- M6 log: [`LOG.md`](LOG.md).
- Phase-1 salvage (the data the project owns): [`PHASE_1_SALVAGE.md`](PHASE_1_SALVAGE.md).
- Phase-1 data audit (what is missing from the data): [`DATA_AUDIT_PHASE_1.md`](DATA_AUDIT_PHASE_1.md).
- Frozen 2026-05-04 literature arm (what this audit extends): [`../research/SURVEY.md`](../research/SURVEY.md), [`../research/SURVEY_FOCUSED.md`](../research/SURVEY_FOCUSED.md), [`../research/LITERATURE_REVIEW.md`](../research/LITERATURE_REVIEW.md), [`../research/PARADIGM_REVIEW.md`](../research/PARADIGM_REVIEW.md), [`../research/RUNTIME_EFFICIENCY.md`](../research/RUNTIME_EFFICIENCY.md), [`../research/INTEGRATION_GUIDE.md`](../research/INTEGRATION_GUIDE.md).
