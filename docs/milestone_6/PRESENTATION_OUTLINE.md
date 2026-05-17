---
title: M6: terse presentation outline (talk + thesis defense)
tags: [milestone, m6, presentation, talk, outline]
source: internal
created: 2026-05-16
updated: 2026-05-17
---

# Presentation Outline (terse)

> Slide-by-slide outline for a 20-25 min talk + 10 min Q&A. Doubles as the thesis defense outline. Companion: [`../report/SUPERVISOR_MEETING_2026-05-16_m0_to_6.md`](../report/SUPERVISOR_MEETING_2026-05-16_m0_to_6.md) is the full narrative; this file is the speaker's deck.
>
> Convention: one slide per `##` heading. Bullets are speaker notes, not slide text. Slide text in **bold**.

---

## Slide 1: Title

**Reproducing and extending retrieval-augmented RL post-training at sub-1B scale**

- Master's thesis, Leiden University.
- Supervisors track via [`../report/`](../report/).
- Project repo: `reason_over_search`.

## Slide 2: Question

**Can a 0.8B language model be post-trained to Search-R1-level retrieval-augmented multi-hop QA on a single A100 within \$1000?**

- Reframed 2026-05-04 from RLVR-via-tool-use to recipe-search under realistic resource constraints.
- Hard deadlines: experimentation cutoff 2026-06-10, thesis submission 2026-06-15.

## Slide 3: One-slide story

**Three observations from Phase-1 prompt-sweep + M5.1 training:**

1. The ReSearch 0.1 partial-credit reward floor masks tool-use signal at 0.6B (3-6 pp tool/no-tool gap vs 9 pp prompt swing).
2. No-example + decision-rules prompt pareto-dominates with-example: held-out EM +9 % rel, F1 +14 %, half the response budget.
3. At 0.8B GRPO on multi-hop QA, length compresses 3× while reward grows. **Inverts the long-CoT regime of math reasoning RL**.

→ The picked-pair experiments turn these into ablations.

## Slide 4: What's been done (M0 → M3 recap)

**Four closed milestones**:

- **M0** (Phase-1): 29 Qwen3-0.6B ALICE runs; reward 0.16-0.22 band; prompt design dominates reward design at sub-1B.
- **M1**: Search-R1 eval reproduction within ±2.5 pp on Qwen2.5-3B.
- **M2**: training pipeline ported to NeMo-RL (verl does not support Qwen3.5).
- **M3 + M3.1**: untrained Qwen3-0.6B → GRPO-trained, EM 0.102 → 0.169 (+66 % rel), 5/7 datasets improved, held-out generalisation rules out memorisation.

## Slide 5: What's been done (M4 → M5.1)

**Two new milestones since the 2026-05-07 supervisor brief**:

- **M4** (closed 2026-05-09): untrained Qwen3.5-0.8B baselines. Avg EM hybrid 0.057, base 0.034. The "untrained floor" any trained Qwen3.5 checkpoint must beat.
- **M5.1 LANDED 2026-05-17 ~08:18 UTC at step_180** (HOLD, not crash; 58 % of one MuSiQue epoch). First GRPO-trained Qwen3.5-0.8B on the ReSearch-paper recipe in NeMo-RL; 18 cadences × 10 steps.
  - Peak window-mean **0.280 at cadence 11**. Run-high single step **0.394 at step 49**.
  - Tool calls 8.96 → ~3.4 (matches MuSiQue ground-truth ~3-hop chain length); token mean 7038 → ~2200 (3.2× compression).
  - **F1 ceiling structural**: 130 steps in 0.20-0.28 cadence band, no monotone climb.

## Slide 6: The shrink-and-improve regime + lean-drift-lean cycling

**Plot 1**: reward window-mean vs step (climbs 0.028 → 0.280) overlaid with token mean (compresses 7038 → 2200).
**Plot 2**: tool_median vs step showing two cycles (C12-C16: 3 → 6 → 3; C17-C18: 3 → 5 in flight) with chain-flip-rate overlay.

- **Shrink-and-improve** inverts the long-CoT regime of math reasoning RL.
- **Lean-drift-lean cycling** (NEW 2026-05-17): GRPO self-stabilises. Cycle 1 peaked at tool_med 6 + len_med 28.8 K + flip rate 58 %; cycle 2 peaked at tool_med 5 + len_med 23.7 K + flip rate 53 % (**damped** on every metric).
- DGPO ([arXiv:2508.20324](https://arxiv.org/abs/2508.20324)): 0.5-1B agentic RAG fails under pure RL in their setup; ours succeeds with the right reward (no partial-credit floor).
- This is the **regime claim**: small-scale is the paper's scope, not an apology.

## Slide 7: M6: the planning milestone

**Goal**: pick the two follow-up experiments + publication framing **before** any further GPU is rented.

- M6 runs parallel to M5.1 wall-clock; no GPU work.
- Five artifacts produced 2026-05-11 to 2026-05-16: salvage, data audit, literature gap audit (18 papers), landscape table, candidate experiments, picked pair, publication framing.
- One user decision still pending (defensive vs ambitious pick #2).

## Slide 8: Critical review (what we caught on paper)

**Three problems with the instinctive framing: caught before any GPU**:

1. "Search-R1 + ReSearch combined" is inaccurate; M5 is ReSearch alone.
2. "MC-GRPO applied to search-tool" competes with Tree-GRPO at ICLR 2026.
3. "Two efficiency improvements" is insufficient for NeurIPS main.

**Honest venue ranges**: NeurIPS main 5-10 %, ICLR / ACL / EMNLP main 15-25 %, Findings / Workshop 50 %+, ICLR blogpost very plausible.

## Slide 9: Literature gap audit (2026-05-16)

**18 papers** published 2025-11 → 2026-05 missing from the project's frozen 2026-05-04 literature pass.

- 5 Tier-1 (must-engage): [arXiv:2602.19526](https://arxiv.org/html/2602.19526v1), AERO, BAPO, DGPO, LiteResearcher.
- 4 Tier-2 (dense-credit cluster): IGPO, TIPS, IG-Search, Search-P1.
- 9 Tier-3 (additive citations).

**Key result**: arXiv:2602.19526 partially scoops the reward-shape ablation. Rewrite as small-model / single-GPU / no-format-reward complementary point. **All six framing decisions from 2026-05-11 survive the audit unchanged.**

## Slide 10: Picked pair (pre-flight unblocked 2026-05-17)

**Pick #1 (locked + F1-only series already published)**: **reward-shape ablation** at the M5.1 recipe (Qwen3.5-0.8B). The F1-only run *is* M5.1; 18 checkpoints on HF Hub at [pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42-f1-only](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42-f1-only). Remaining: 2 more runs (F1+0.1, EM-only) at same recipe + same 180-step horizon. Total picked-pair eval matrix: **3 rewards × 18 ckpts × 7 benchmarks × 1-2 seeds = 378-756 data points**.

**Pick #2 (one decision pending)**:
- **Option A (defensive, default)**: 2-seed Qwen3.5-0.8B prompt-pair replication. Closes Finding 2 + the no-multi-seed gap. ~\$80-120, ~5-8 d. ICLR-blogpost / Workshop / Findings target.
- **Option B (ambitious)**: MC-GRPO at G=2 on M5.1 recipe. Engages the literature gap on median-baseline-for-search-tool. ~\$150-220, ~10-14 d. ICLR / ACL main 22-32 % under positive outcome.

## Slide 11: Takeaway sentences (per pair)

**C + R (defensive)**:

> "At sub-1B scale on retrieval-augmented multi-hop QA, the F1-only reward is Pareto-defensible against the ReSearch partial-credit floor, and the no-example + decision-rules prompt design transfers across model family, both at half the response budget of the standard recipe."

**C + M (ambitious)**:

> "At sub-1B scale on retrieval-augmented multi-hop QA, the partial-credit reward floor and the small-G mean-baseline both compress training signal; removing the floor and switching to median-baseline GRPO together define the small-model frontier for search-augmented RL."

## Slide 12: Threats-to-validity

**Three a reviewer will hammer**:

1. **Scale generalisation** ("does this transfer to 7B?"). Response: the paper's scope is sub-1B; three anchor points (DGPO, Phase-1 base-bootstrap failure, M5.1 shrink-and-improve trajectory).
2. **Head-to-head with Tree-GRPO**. Response: orthogonal axis (prefix-sharing); fair comparison requires NeMo-RL port of their framework, busts the budget; named future work.
3. **Method-transfer framing**. Response: the contribution is *characterisation* of a regime, not a new algorithm. Backup framing: re-pitch as a JustRL-template ICLR blogpost.

## Slide 13: Where it goes (timeline)

**~17-day implementation window after M5.1 lands**:

- M5.1 ETA: 2026-05-17 to 2026-05-22.
- Pick-#1 runs (3 reward variants): ~9-13 d.
- Pick-#2 runs (defensive R or ambitious M): ~5-14 d.
- Eval pipeline runs: ~1 d total.
- Experimentation cutoff (firm): **2026-06-10**.
- Thesis submission (firm): **2026-06-15**.

**Risk**: C + M slips up to cutoff with no margin. C + R closes 3-6 days early.

## Slide 14: Chapter outline (provisional)

1. Introduction (three-finding motivation paragraph).
2. Related work (200-word paragraph, drafted at [`LANDSCAPE_TABLE_2026-05.md` §4](LANDSCAPE_TABLE_2026-05.md#4-positioning-paragraph-200-words-for-the-papers-related-work-section)).
3. Setup: 0.8B + NeMo-RL + single-A100 + ReSearch-paper recipe with intentional divergences.
4. Phase-1 motivation experiments.
5. M5.1 baseline (shrink-and-improve + F1 ceiling).
6. Picked-pair experiments (6a + 6b).
7. Discussion + limitations.
8. Future work.

## Slide 15: Future work (one slide for the defense)

**Three follow-ups logged but not pursued in the thesis**:

1. **Head-to-head against Tree-GRPO** (named threat-to-validity #2; requires NeMo-RL port).
2. **Cross-family Qwen3 → Qwen3.5 degradation diagnostic** (M4 surfaced; 1-page negative result if mechanism attaches).
3. **Model-based GRPO via learned latent rollouts** ([`research/QUESTIONS.md` Q2](../research/QUESTIONS.md); MuZero-for-LLMs lineage; ~9-month research project, not a thesis extension).

## Slide 16: Summary

**Three things to remember**:

1. **The regime is real**: shrink-and-improve at sub-1B on multi-hop retrieval. Documented in M5.1 cadence 1-9.
2. **The Phase-1 findings ladder into the paper**: 0.1-floor masks signal → reward-shape ablation; decision-rules substitute for example → prompt replication; length compresses → regime characterisation.
3. **The picked pair is mechanism-driven, not method-transfer**: every experiment closes a Phase-1 gap.

**One sentence**: small-scale retrieval-augmented RL is a characterised regime with its own dynamics; the project's contribution is the characterisation, not an algorithm.

## Slide 17: Acknowledgements + thanks

- Supervisors at Leiden.
- Alstom internship parallel strand (separate brief).
- Compute: ALICE (Phase 1), Vast.ai (M1-M2), Spheron 1× H200 (M5.1 a4).
- Frameworks: NeMo-RL (NVIDIA), verl (Phase 1), FAISS + E5.

---

## Speaker timing (target 22 min talk, 8 min Q&A)

| Section | Slides | Time |
|---|---|---:|
| Title + question | 1-2 | 1:30 |
| Story + recap | 3-5 | 4:30 |
| M5.1 live | 6 | 2:30 |
| M6 critical review + audit | 7-9 | 4:30 |
| Picked pair + takeaways | 10-11 | 3:30 |
| Threats + timeline | 12-13 | 2:30 |
| Chapter + future work | 14-15 | 2:00 |
| Summary | 16-17 | 1:00 |
| **Total** | **17** | **22:00** |

## Q&A preparation (likely questions)

1. **"Why MuSiQue only for training?"** → Hardest of the four ReSearch benchmarks; the recipe should generalise (M3.1 held-out result confirms cross-benchmark transfer at 0.6B); single-dataset matches paper.
2. **"What if M5.1 plateaus below 0.20?"** → The picked pair becomes remediation; the framing shifts but the experiments are still informative (the plateau cause analysis at [`RESULTS_M5_1_H200.md` §9.5](../report/RESULTS_M5_1_H200.md#95-f1-reward-ceiling-the-structural-plateau-cause--chain-quality-reward-designs-added-2026-05-16-post-cadence-9) is itself a chapter section).
3. **"Why no SFT cold-start?"** → R1-Searcher needs SFT below 7B but Qwen3.5-0.8B is hybrid (post-trained with instruct + reasoning data); bootstraps tool-use from the format alone. M5.1 cadence 1-9 confirms.
4. **"Cost of 1× A100 vs H100 vs H200?"** → See [`setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md); 1× B200 turned out 2.66× over A100, not 11× as initially projected.
5. **"Why not Tree-GRPO head-to-head?"** → Threats-to-validity slide; orthogonal axis; named future work.
6. **"Is the chapter publishable as-is?"** → Honestly: ICLR blogpost or Workshop, yes. ICLR / ACL main, only under C + M positive outcome. NeurIPS main, no.

## Cross-references

- Full narrative: [`../report/SUPERVISOR_MEETING_2026-05-16_m0_to_6.md`](../report/SUPERVISOR_MEETING_2026-05-16_m0_to_6.md).
- M6 source-of-truth docs: see [`MILESTONE_6.md` §"Deliverables"](MILESTONE_6.md#deliverables-this-folder).
