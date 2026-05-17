---
title: Supervisor Meeting 2026-05-16: story so far (M0 to M6) and where it goes
tags: [report, supervisor, m4, m5, m6, story]
source: internal
created: 2026-05-16
updated: 2026-05-16
---

# Supervisor Meeting (2026-05-16): story so far and where it goes

Consolidated brief covering M0 to M6. Successor to [`SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) (M0 to M3, frozen). M4 (untrained-baseline eval) closed; M5.1 (Qwen3.5-0.8B GRPO training, ReSearch-paper recipe) is live; M6 (literature + experiment-planning milestone) has frozen four artifacts on 2026-05-16 and decides the picked pair after one user-side choice.

## 0. TL;DR (six bullets)

1. **The story closes a chapter and opens one.** M0-M3 (closed in the previous brief) established that GRPO on Qwen3-0.6B is stable but capacity-bounded and that prompt design dominates reward design at sub-1B. **M4 (closed 2026-05-09)** ran the untrained Qwen3.5-0.8B floor; **M5.1 (live since 2026-05-11)** is the first trained Qwen3.5-0.8B checkpoint at the ReSearch-paper recipe, ported to NeMo-RL. **M6 (planning milestone, 2026-05-11 → 2026-05-18)** picks the two follow-up experiments and the publication framing.

2. **M5.1 is climbing.** Through cadence 9 (steps 81-93, 2026-05-16 ~12:30 UTC, H200 dedicated tier; per [`RESULTS_M5_1_H200.md` §8](RESULTS_M5_1_H200.md#8-live-trajectory)): reward mean 0.110 at step 13 → 0.228 at step 93 (~8× lift from cold start); fraction reward > 0 climbed from 8 % at step 1 to ~46 % at step 93; multi-hop chains correctly resolved on ~150 / 320 prompts per step. The trajectory exhibits the **shrink-and-improve regime**: rollout length compresses 3× (token mean 7038 → 2183) while reward grows, *inverting* the long-CoT regime familiar from math reasoning RL. Step wall-clock dropped from 18:22 to 6:04 in the first cadence and is now ~6-7 min/step.

3. **M5.1 plateau is structural, not capacity-bounded.** Cadence 5-9 reward sits in the 0.20-0.24 band. Trace analysis at [`RESULTS_M5_1_H200.md` §9.5](RESULTS_M5_1_H200.md#95-f1-reward-ceiling-the-structural-plateau-cause--chain-quality-reward-designs-added-2026-05-16-post-cadence-9) identifies the cause: F1-only reward gives identical scalar credit to chain-correct rollouts and token-aligned-by-luck rollouts; the optimiser cannot disambiguate. This is **the same mechanism Phase-1 Finding 1 named at 0.6B** ([`PHASE_1_SALVAGE.md` Finding 1](../milestone_6/PHASE_1_SALVAGE.md#finding-1--the-01-partial-credit-floor-masks-tool-use-signal)) reappearing at 0.8B under a different floor. It is the load-bearing observation that motivates M6's picked-pair experiment #1 (reward-shape ablation).

4. **M6 reframed the thesis question's competitive position.** The 2026-05-11 critical review caught three problems with the instinctive framing ("Search-R1 + ReSearch recipe + MC-GRPO efficiency"): the M5 setup is ReSearch-paper alone (Search-R1 is the eval, not the training), the MC-GRPO-on-search hook competes with Tree-GRPO at ICLR 2026, and the "two efficiency wins" pitch is insufficient for NeurIPS main. The 2026-05-16 literature gap audit found **18 papers** published 2025-11 → 2026-05 missing from the project's frozen 2026-05-04 literature pass, one of which ([arXiv:2602.19526](https://arxiv.org/html/2602.19526v1), Feb 2026) **partially scoops** the planned reward-shape ablation. All six framing decisions from 2026-05-11 survive the audit unchanged; the picked-pair pitch shifts to the small-model / single-GPU / no-format-reward complementary point. Full audit at [`LITERATURE_GAP_AUDIT_2026-05-16.md`](../milestone_6/LITERATURE_GAP_AUDIT_2026-05-16.md).

5. **The picked pair is frozen except for one user choice.** Pick #1: **reward-shape ablation** (F1+0.1 vs F1-only vs EM-only at the M5.1 recipe; closes Phase-1 Finding 1's gap; informative under both outcomes; ~\$110-160, ~9-13 d on 1× A100). Pick #2 is venue-dependent: **defensive R** (2-seed Qwen3.5-0.8B prompt-pair replication; closes Finding 2 + the project's no-multi-seed gap; ICLR-blogpost / Workshop / Findings target; ~\$80-120, ~5-8 d) or **ambitious M** (MC-GRPO at G=2; addresses the literature gap on median-baseline-for-search-tool; ICLR / ACL main 15-25 % under positive outcome; ~\$150-220, ~10-14 d). Default recommendation: **C + R**. Full breakdown at [`PICKED_PAIR.md`](../milestone_6/PICKED_PAIR.md).

6. **Honest venue ranges**: NeurIPS main 5-10 %, ICLR / ACL / EMNLP main 15-25 %, Findings / Workshop 50 %+, **ICLR blogpost very plausible** (the JustRL template). For the thesis itself, the picked pair lands in either case before the 2026-06-10 experimentation cutoff. The post-thesis publication is what the user-side decision in §5 is really choosing about.

---

## 1. Recap M0 to M3 (short; full detail in the 2026-05-07 brief)

See [`SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) for the source-of-truth narrative. Headline numbers:

- **M0 Phase-1** (29 Qwen3-0.6B ALICE runs across v0+v1 blocks, Apr 3 → Apr 19): reward bands 0.16-0.22; 5/5 base-model attempts fail to bootstrap tool-use at 0.6B; prompt design dominates reward; the 0.1 partial-credit floor masks the tool-use signal.
- **M1** (Search-R1 eval reproduction, Qwen2.5-3B base + instruct, completed 2026-04-23): avg EM 0.292 (base; −2.0 pp vs paper) / 0.361 (instruct; +2.5 pp); 6/7 datasets within ±5 pp.
- **M2** (NeMo-RL training pipeline, Phase-1 build): Search-R1 GRPO loop ported to NeMo-RL because verl does not support Qwen3.5; 15 reward-parity tests pass; smoke-tested on 1× A100.
- **M3 + M3.1** (eval of v0 GRPO checkpoints vs untrained Qwen3-0.6B hybrid, 2026-05-07 / 2026-05-08): avg EM 0.102 → 0.155 (M3, +52 % rel) → 0.169 (M3.1, +66 % rel over pre-GRPO). 5/7 datasets improved over M3 on M3.1. Held-out generalisation rules out memorisation.

The reframed RQ from 2026-05-04 (carried forward unchanged): *"is it feasible to post-train a small LM under realistic resource constraints, and what is the optimised training recipe?"*

---

## 2. M4: untrained Qwen3.5-0.8B baseline (closed 2026-05-09)

The Qwen3.5-family equivalent of the M3 eval pipeline. Establishes the "untrained floor" any Qwen3.5-trained checkpoint must beat. Pipeline at [`evaluation_qwen35/`](../../evaluation_qwen35/), a copy of [`evaluation_research/`](../../evaluation_research/) with Qwen3.5-native tool-use tags (`<tool_call>` / `<tool_response>`, vocab ids 248058 / 248059 / 248066 / 248067) replacing Search-R1's invented `<search>` / `<result>`.

### Numbers (n=100 / dataset / variant; greedy decode; M4.2 final lock)

| Variant | Avg EM | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B hybrid (instruct) | **0.057** | 0.06 | 0.21 | 0.05 | 0.05 | 0.04 | 0.01 | 0.00 |
| Qwen3.5-0.8B Base | 0.034 | 0.04 | 0.12 | 0.04 | 0.02 | 0.02 | 0.00 | 0.00 |

For comparison, Qwen3-0.6B hybrid on the same protocol scored **0.102 EM** ([`RESULTS_m4.md` §5](RESULTS_m4.md#L67)). M4 thus surfaces an **uncomfortable cross-family observation**: at the untrained floor, Qwen3.5-0.8B is **uniformly below** Qwen3-0.6B (all 7 datasets, no crosses, mean Δ −0.042 EM, −41 % rel). Possible mechanisms not ruled out: tokenizer drift (151K → 248K vocab), chat-template drift, training-distribution drift, prompt-mode misalignment. This is logged in [`DATA_AUDIT_PHASE_1.md` §4a](../milestone_6/DATA_AUDIT_PHASE_1.md#4a-qwen35-08b-hybrid--qwen3-06b-hybrid-on-untrained-tool-use-results_m4md-5) as a candidate 1-page negative result if a cheap diagnostic can attach mechanism.

### M4.1 and M4.2 sub-phases

- **M4.1** (2026-05-08): swapped flat `<tool_call>X</tool_call>` form for Qwen3.5's canonical nested-XML form (the format Qwen3.5 was post-trained on; flat was off-distribution).
- **M4.2** (2026-05-09): locked `prompt_mode=qwen35_minimal` as canonical after smoke iteration. The v3 system-message protocol with auto-inject was drowning the 0.8B model in scaffolding (~330 pre-question words); minimal lifted hybrid mean EM from 0.0086 to 0.057 (6.6×).

Full doc trail: [`MILESTONE_4.md`](../milestone_4/MILESTONE_4.md), [`RESULTS_m4.md`](RESULTS_m4.md), [`CODE_SETUP_m4.md`](CODE_SETUP_m4.md), [`RESULTS_SMOKE_m4.md`](RESULTS_SMOKE_m4.md).

---

## 3. M5.1: Qwen3.5-0.8B GRPO training on the ReSearch-paper recipe (live)

### Setup and motivation

M5.1 is the training-side counterpart to M4. GRPO-trains Qwen3.5-0.8B on the existing [`training/`](../../training/) NeMo-RL scaffold (M2 work), with the rollout / prompt / tag scheme byte-aligned to the M4 eval pipeline so the trained checkpoint is directly evaluable without a re-alignment audit. **Verl is not used** (Qwen3.5 not supported); M5.1 ports the [ReSearch paper](https://arxiv.org/abs/2503.19470) recipe knob-by-knob into NeMo-RL using [`Agent-RL/ReSearch`](https://github.com/Agent-RL/ReSearch) as the concrete-config source of truth.

Two **intentional divergences** from the paper:
- **F1-only reward** on `<answer>X</answer>` content (no format reward, no partial-credit floor). Phase-1 Finding 4 (the 0.1 floor masks tool-use signal at 0.6B) made this the explicit recipe choice.
- **No `\boxed{}` answer wrapper** (carry from M4's `qwen35_minimal` prompt-mode lock).

Training data: **MuSiQue only** (single-dataset; hardest of the four ReSearch benchmarks; matches the ReSearch paper's reported training set). Doc trail: [`MILESTONE_5.md`](../milestone_5/MILESTONE_5.md), [`CODE_SETUP_m5.md`](CODE_SETUP_m5.md), [`PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md), live results at [`RESULTS_M5_1_H200.md`](RESULTS_M5_1_H200.md).

### Substrate history (incident chain → resilience)

The run moved between three substrates after two incidents:

1. **a1 on Vast 1× A100** (launched 2026-05-11). Smoke OK; full-run plateaued in compute-cost terms.
2. **a3 on B200** (launched 2026-05-14): crashed at step 56 / 9.29 h after Blackwell sm_100 BF16-dense kernel immaturity surfaced in NeMo-RL/vLLM 0.17 ([`log.md` 2026-05-15](../log.md#2026-05-15)). 1× B200 measured at 2.66× over A100, not the 11× projected by HARDWARE_COMPARISON v2.
3. **a4 on Spheron 1× H200 with persistent volume `miletone5`** (launched 2026-05-15). Runbook frozen at [`spheron/SETUP_SPHERON.md`](../spheron/SETUP_SPHERON.md) with 11 named pitfalls (P1-P11); checkpoint upload watcher → HF Hub `pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42-f1-only`; v3-v11 prod incident chain (vLLM 0.17 FlashInfer GDN prefill kernel deadlocks on H200 sm_90 under sync engine at prod-batch sizes; fixed by patching `qwen3_next.py:156` from `is_device_capability(90)` to `False`).

H200 ETA at step 311 (50 % of paper schedule): **~17:00 UTC 2026-05-17** (~28.6 h remaining at 2026-05-16 cadence-9 pace).

### Live trajectory (through cadence 9, 2026-05-16 ~12:30 UTC)

| Cadence | Steps | Reward mean (peak) | rew > 0 | tc mean | tok mean | Step wall (min) |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1-13 | 0.028 → 0.110 | 8 → 20 % | ~7 | 7038 → 4500 | 18:22 → 6:04 |
| 2 | 14-22 | 0.119 → 0.132 | 22 % | 5.2 → 4.0 | 4200 → 2700 | 5:36 |
| 3-4 | 23-50 | 0.140 → 0.180 | 27 → 35 % | 3.8 → 3.5 | 2400 → 2200 | 5:50 |
| 5 | 51-60 | 0.202 | 38 % | 3.5 | 2200 | 6:00 |
| 6 | 61-70 | **0.224** | 40 % | 3.5 | 2200 | 6:10 |
| 7 | 71-80 | 0.202 | 39 % | 3.5 | 2200 | 6:30 |
| 8 | 81-90 | 0.221 | 43 % | 3.4 | 2200 | 6:30 |
| 9 | 91-93 | **0.228** | 46 % | 3.4 | 2150 | 6:40 |

Full trajectory + per-rollout mechanical examples (best / worst / mean): [`RESULTS_M5_1_H200.md` §8](RESULTS_M5_1_H200.md#8-live-trajectory).

### Findings on M5.1 (provisional, mid-training)

1. **Shrink-and-improve regime holds at 0.8B**. Tool calls compressed 8.96 → ~3.4 (close to MuSiQue's ~3-hop ground truth); token mean compressed 7038 → ~2200 (3.2×); reward grew 8× from cold start. The regime **inverts the long-CoT regime** familiar from math reasoning RL (where length grows with reward). DGPO ([arXiv:2508.20324](https://arxiv.org/abs/2508.20324) v4) gives mechanistic support: 0.5-1B agentic RAG "fails under pure RL" in their setup; ours succeeds, suggesting the shrink-and-improve regime is a favourable small-model dynamic when the reward design doesn't fight it.

2. **F1-only reward has a structural ceiling at ~0.22-0.24**. Cadence 5-9 plateau is not a capacity limit; it is the F1 metric being unable to disambiguate chain-correct rollouts from token-aligned-by-luck rollouts. Trace example at [`RESULTS_M5_1_H200.md` §9.5](RESULTS_M5_1_H200.md#95-f1-reward-ceiling-the-structural-plateau-cause--chain-quality-reward-designs-added-2026-05-16-post-cadence-9): "Fox Island / Pan-African Conference" (step 93 idx 10) earns reward 1.0 with `<answer>United Kingdom</answer>` matching MuSiQue gold despite the chain stating "Fox Island is in United States of America" then silently flipping. Identical scalar reward to a chain-clean trace. The optimiser cannot tell them apart. **This is the load-bearing observation for M6's picked-pair experiment C.**

3. **Multi-hop generalisation works for non-Ghana bridges by cadence 9**. The 4-hop+ chains correctly resolved climb steadily through training. Cadence 6 first showed 4-hop chains with all three intermediate retrievals supporting the final answer.

---

## 4. M6: literature + experiment planning milestone (2026-05-11 → 2026-05-18; live)

M6 runs **in parallel with M5.1 wall-clock** as a planning-only milestone (no GPU work). Objective: pick the two follow-up experiments after M5.1 lands, and the publication-framing brief, before any further GPU is rented. The bottleneck is **decision compounding** (the next two experiments eat 30-40 % of the remaining \$1000 budget and 60-70 % of the remaining wall-clock window before 2026-06-10) and **positioning** (NeurIPS / ICLR / ACL / Findings / Workshop / blogpost decisions shape what the next two experiments need to demonstrate).

### Critical review (2026-05-11)

The instinctive framing: *"basic paper recreation (Search-R1 + ReSearch) + MC-GRPO efficiency improvement"*: had three problems caught on paper before any GPU was rented (full table at [`CONVERSATION_CONTEXT.md` §3](../milestone_6/CONVERSATION_CONTEXT.md#3-working-assumption-from-2026-05-11-critical-review-confirmed-2026-05-16)):

1. **"Search-R1 + ReSearch combined" is inaccurate**. M5 is ReSearch-paper recipe alone (GRPO, F1, MuSiQue, G=5); Search-R1's role is the 7-dataset eval pipeline. Will be caught in review.
2. **"MC-GRPO applied to search-tool" is real**, but Tree-GRPO at ICLR 2026 already addresses the *efficient rollouts for agent RL* axis. The mechanistic story for *why* median-baselining helps search-tool specifically (retrieval reward variance, longer trajectories, smaller G under memory pressure) needs to be made explicit, or the work reads as method-transfer.
3. **"Two efficiency improvements stacked"** is competitive with DIVA-GRPO (2.55× step + 1.76× wall-clock) and Tree-GRPO (1/4 rollout budget). Without a head-to-head against at least one of them, NeurIPS-tier reviewers will reject for incomplete comparison.

The honest venue ranges that fall out: **NeurIPS main 5-10 %, ICLR / ACL / EMNLP main 15-25 %, Findings / Workshop 50 %+, ICLR blogpost very plausible**.

### Phase-1 salvage (2026-05-11)

Three Phase-1 findings survive critical scrutiny and serve as the introduction's motivation paragraph (full detail at [`PHASE_1_SALVAGE.md`](../milestone_6/PHASE_1_SALVAGE.md)):

1. **The 0.1 partial-credit floor masks tool-use signal** (3-6 pp tool/no-tool gap vs 9 pp prompt swing). Observation grade across 9 cross-prompt runs; M5.1 cadence-9 plateau is the same mechanism reappearing under F1-only at 0.8B.
2. **No-example + decision-rules prompt is a pareto improvement** (held-out EM +9 % rel, ACC +12 %, F1 +14 %, half budget). Mechanism named (decision-rules scaffolding substitutes for in-context example).
3. **Shrink-and-improve regime** at multi-hop tool-use (length compresses 3× while reward grows). M5.1 confirms at 0.8B.

Findings *not* salvageable (replication-grade): M3 +52 % rel EM lift, M1 ±2.5 pp Search-R1 reproduction, the 14 train/eval alignment fixes, Qwen3.5 < Qwen3 cross-family degradation (negative observation, no mechanism attached).

### Data audit (2026-05-11)

Five cross-cutting gaps documented at [`DATA_AUDIT_PHASE_1.md` §5](../milestone_6/DATA_AUDIT_PHASE_1.md#5-cross-cutting-gaps-that-affect-everything): no multi-seed runs anywhere, no per-rollout-level analysis, no 0.8B / Qwen3.5 prompt-sweep replication, no direct format-OK-but-F1=0 rate measurement, no baseline that converges to a different regime. The picked pair should close at least one.

### Literature gap audit (2026-05-16)

Four-axis deep-search audit dated 2026-05-16 surfaced **18 papers published 2025-11 → 2026-05** missing from the frozen 2026-05-04 literature pass ([`LITERATURE_GAP_AUDIT_2026-05-16.md`](../milestone_6/LITERATURE_GAP_AUDIT_2026-05-16.md)). 5 Tier-1 (must-engage), 4 Tier-2 (dense-credit cluster), 9 Tier-3 (additive).

Key findings:

- **[arXiv:2602.19526](https://arxiv.org/html/2602.19526v1)** (Feb 2026) **partially scoops the planned reward-shape ablation**: directly ablates EM vs F1 reward in Search-R1 at 3B+ and finds F1 + action-penalties wins. Our work re-pitches as the small-model (0.8B), single-GPU-budget, no-format-reward complementary point; Phase-1 Finding 1's 0.1-floor mechanism remains the project's own.
- **DGPO ([arXiv:2508.20324](https://arxiv.org/abs/2508.20324) v4, Apr 2026)** gives mechanistic support to Phase-1 Finding 3 (shrink-and-improve regime at sub-1B agentic search).
- **LiteResearcher ([arXiv:2604.17931](https://arxiv.org/abs/2604.17931), Apr 2026)** is the closest published reference at 4B; differentiate on (a) 0.8B vs 4B, (b) NeMo-RL substrate, (c) single-A100 budget, (d) MuSiQue training.
- **Dense-credit cluster (IGPO, TIPS, IG-Search, Search-P1)** establishes outcome-only F1 as a 2026 *minority position*. F1-only must now be defended as deliberate minimalism (Phase-1 Finding 1 is the wedge).
- **Rethinking E2H ([arXiv:2603.27226](https://arxiv.org/html/2603.27226))** is a negative result on curriculum; reframes the curriculum candidate.
- **NeMo-RL speculative decoding ([arXiv:2604.26779](https://arxiv.org/html/2604.26779v1))** is a free 1.8× rollout throughput win on the M5.1 substrate; future-work mention.

All six framing decisions from 2026-05-11 **survive the audit unchanged**. No paper fully scoops the project's framing.

### Phase 1 landscape table and positioning (2026-05-16)

Frozen at [`LANDSCAPE_TABLE_2026-05.md`](../milestone_6/LANDSCAPE_TABLE_2026-05.md). Three axes the picked pair takes a defensible position on:

- **Axis A: Reward shape**. F1-only is the minority position vs the 2026 dense-credit trend (D1-D4 + C5). Defend with mechanism (Phase-1 Finding 1) + positive ablation evidence (Candidate C).
- **Axis B: Rollout efficiency**. Tree-GRPO, DIVA-GRPO, AERO, BAPO compete here. Position: "characterise the small-rollout regime for search-tool RL at sub-1B; defer altering it to follow-up". Makes the contribution **characterisation**, not **efficiency**, and earns the small-scale scope.
- **Axis C: Scale**. DGPO and LiteResearcher are the only sub-2B reference points. M5.1 lands in this gap.

The 200-word related-work paragraph for the paper introduction is drafted at [`LANDSCAPE_TABLE_2026-05.md` §4](../milestone_6/LANDSCAPE_TABLE_2026-05.md#4-positioning-paragraph-200-words-for-the-papers-related-work-section).

### Phase 2 candidate experiments (2026-05-16)

Five candidates filtered to two viable pairs ([`CANDIDATE_EXPERIMENTS.md`](../milestone_6/CANDIDATE_EXPERIMENTS.md)):

- **C + R** (defensive): reward-shape ablation + 2-seed Qwen3.5-0.8B prompt replication. ~\$190-280, ~14-21 d, ICLR-blogpost / Workshop / Findings target.
- **C + M** (ambitious): reward-shape ablation + MC-GRPO at G=2. ~\$260-380, ~19-27 d, ICLR / ACL / EMNLP main 15-25 % target under positive outcome only.

C + D (curriculum) is post-audit dominated by C + R; C + S (scale-up) busts the budget; R + M lacks a mechanism-named takeaway.

### Phase 3 picked pair (2026-05-16; one user choice pending)

**Pick #1 (frozen): reward-shape ablation**. Three runs at fixed prompt × fixed seed × reward ∈ {F1+0.1, F1-only, EM-only} on the M5.1 recipe at Qwen3.5-0.8B. Closes Finding 1's gap. Both positive and negative outcomes publishable.

**Pick #2 (decision pending; default C + R)**: 2-seed prompt replication (defensive) or MC-GRPO at G=2 (ambitious). Recommendation default to C + R unless the user explicitly opts for the ambitious target; rationale at [`PICKED_PAIR.md` §3](../milestone_6/PICKED_PAIR.md#3-recommendation).

Full pre-flight requirements and threats-to-validity consolidated at [`PICKED_PAIR.md` §4-5](../milestone_6/PICKED_PAIR.md#4-pre-flight-requirements-regardless-of-pick-2). Publication-framing brief at [`PUBLICATION_FRAMING.md`](../milestone_6/PUBLICATION_FRAMING.md) (both pair variants drafted; freezes on user pick).

---

## 5. The thesis story (what M6 lets us write)

Restated for the chapter outline:

> *Small language models post-trained with RL on retrieval-augmented multi-hop QA enter a* shrink-and-improve *regime that inverts the long-CoT regime familiar from math reasoning RL. At sub-1B scale on a single A100 within a ~\$1000 budget, the F1-only outcome reward is Pareto-defensible against the ReSearch partial-credit reward floor [Candidate C]; the prompt-design lever dominates the reward-shape lever at this scale [Candidate R / Finding 2]. The shrink-and-improve regime characterises a sub-1B retrieval-augmented RL training regime distinct from both the long-CoT regime of math reasoning RL and the failure mode DGPO (2025) reports for pure RL at 0.5-1B agentic search.*

This frame turns the small-scale scope from an apology into a *characterised regime*. The picked-pair experiments are mechanism-driven follow-ups to Phase-1 observations the project already owns; they are not method-transfer.

### Chapter outline (provisional)

1. **Introduction**. Motivation paragraph citing the three Phase-1 salvage findings ([`PHASE_1_SALVAGE.md` §"How these three findings ladder into the paper"](../milestone_6/PHASE_1_SALVAGE.md#how-these-three-findings-ladder-into-the-paper)).
2. **Related work**. 200-word paragraph at [`LANDSCAPE_TABLE_2026-05.md` §4](../milestone_6/LANDSCAPE_TABLE_2026-05.md#4-positioning-paragraph-200-words-for-the-papers-related-work-section).
3. **Setup**. The 0.8B / NeMo-RL / single-A100 / ReSearch-paper recipe; intentional divergences (F1-only, no boxed wrapper).
4. **Phase-1 motivation experiments**. M0 prompt sweep + M3 / M3.1 held-out evaluation; the 9-prompt paired comparison and the +9 % EM held-out lift.
5. **M5.1 baseline**. The shrink-and-improve trajectory; the F1-ceiling plateau; trace evidence of chain-quality blindness.
6. **Picked-pair experiments**.
   - 6a. Reward-shape ablation (Candidate C).
   - 6b. [Pick #2: prompt replication / MC-GRPO].
7. **Discussion**. Position vs Tree-GRPO, dense-credit cluster, DGPO. Limitations: single-seed on most experiments, no 7B confirmation, no head-to-head with Tree-GRPO.
8. **Future work**. The model-based-GRPO question logged at [`research/QUESTIONS.md` Q2](../research/QUESTIONS.md); the cross-family Qwen3 → Qwen3.5 degradation diagnostic; the long-CoT contrastive.

---

## 6. What's left + decision points

### What is left under M6 itself

| Item | Owner | Blocker | ETA |
|---|---|---|---|
| **User picks Option A (C+R) or Option B (C+M)** for pick #2 | User |: | 2026-05-17 |
| Freeze [`PUBLICATION_FRAMING.md`](../milestone_6/PUBLICATION_FRAMING.md) to the picked variant | Sankar | (1) | 2026-05-18 |
| Update [`research/CONVERSATION_CONTEXT.md`](../research/CONVERSATION_CONTEXT.md) §"Active question" and [`report/CONVERSATION_CONTEXT.md`](CONVERSATION_CONTEXT.md) to reflect M6 conclusions | Sankar | (1)(2) | 2026-05-18 |
| Supervisor briefing using this doc + the [`PRESENTATION_OUTLINE.md`](../milestone_6/PRESENTATION_OUTLINE.md) terse outline | Sankar | (1)(2)(3) | 2026-05-19 |

### What is left after M6 closes (the ~17-day implementation window)

| Item | Compute | Wall-clock | Cost | Dependencies |
|---|---|---|---|---|
| M5.1 final + cadence-aware analysis (per-rollout JSONL extraction included) | (running) | M5.1 ETA 2026-05-17 to 2026-05-22 | (running) |: |
| Pick-#1 run: F1+0.1 variant | 1× A100 | ~3-5 d | ~\$50-75 | M5.1 land |
| Pick-#1 run: F1-only variant | 1× A100 | ~3-5 d | ~\$50-75 | M5.1 land |
| Pick-#1 run: EM-only variant (after smoke) | 1× A100 | ~3-5 d | ~\$50-75 | M5.1 land + 50-step smoke |
| **If pick #2 = R**: 4 prompt-pair runs × 2 seeds × 2 prompts | 1× A100 | ~6-8 d | ~\$80-120 | M5.1 land |
| **If pick #2 = M**: 2 MC-GRPO runs | 1× A100 | ~10-14 d | ~\$150-220 | M5.1 land + 100-step smoke |
| Eval pipeline runs (M3 / M4 protocol on each new checkpoint) | 1× A100 | ~1 d total | ~\$15 | Pick-#1 + pick-#2 |
| Thesis chapter writing |: | continuous |: | All of the above |

### Hard timeline

| Date | Milestone |
|---|---|
| ~2026-05-17 | M5.1 lands (full schedule) |
| 2026-05-18 | M6 closes |
| ~2026-05-22 → 2026-06-08 | Picked-pair experiments execute |
| **2026-06-10** | **Experimentation cutoff (firm)** |
| **2026-06-15** | **Thesis submission (firm)** |
| ~2026-07-15 | Defense |

The picked pair (C + R variant) closes ~3-6 days before the experimentation cutoff. The C + M variant slips up to the cutoff with no margin; this is the load-bearing reason the recommendation defaults to C + R.

---

## 7. Pointers

### Source-of-truth M6 docs

- [`MILESTONE_6.md`](../milestone_6/MILESTONE_6.md): milestone definition.
- [`CONVERSATION_CONTEXT.md`](../milestone_6/CONVERSATION_CONTEXT.md): living M6 snapshot.
- [`LOG.md`](../milestone_6/LOG.md): append-only M6 log.
- [`PHASE_1_SALVAGE.md`](../milestone_6/PHASE_1_SALVAGE.md): three Phase-1 findings reusable as paper motivation.
- [`DATA_AUDIT_PHASE_1.md`](../milestone_6/DATA_AUDIT_PHASE_1.md): what is in the data, what is not.
- [`LITERATURE_GAP_AUDIT_2026-05-16.md`](../milestone_6/LITERATURE_GAP_AUDIT_2026-05-16.md): 18-paper audit (2025-11 → 2026-05).
- [`LANDSCAPE_TABLE_2026-05.md`](../milestone_6/LANDSCAPE_TABLE_2026-05.md): Phase 1 frozen.
- [`CANDIDATE_EXPERIMENTS.md`](../milestone_6/CANDIDATE_EXPERIMENTS.md): Phase 2 frozen.
- [`PICKED_PAIR.md`](../milestone_6/PICKED_PAIR.md): Phase 3a (one user choice pending).
- [`PUBLICATION_FRAMING.md`](../milestone_6/PUBLICATION_FRAMING.md): Phase 3b (freezes on user choice).

### Source-of-truth M5.1 docs

- [`MILESTONE_5.md`](../milestone_5/MILESTONE_5.md): milestone definition.
- [`RESULTS_M5_1_H200.md`](RESULTS_M5_1_H200.md): live results (cadence 1-9 through 2026-05-16).
- [`CODE_SETUP_m5.md`](CODE_SETUP_m5.md): locked recipe.
- [`PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md): clause-by-clause paper-to-YAML mapping.

### Predecessor brief

- [`SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](SUPERVISOR_MEETING_2026-05-07_m0_to_3.md): M0 to M3 narrative.

### Hardware / cost

- [`setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md): current accelerator + provider comparison.
- [`spheron/SETUP_SPHERON.md`](../spheron/SETUP_SPHERON.md): H200 runbook with 11 named pitfalls.
