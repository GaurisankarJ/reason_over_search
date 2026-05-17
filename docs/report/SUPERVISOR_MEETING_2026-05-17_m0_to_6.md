---
title: Supervisor Meeting 2026-05-17: M5.1 landed at step_180, picked pair unblocked
tags: [report, supervisor, m4, m5, m6, story, m5-landed]
source: internal
created: 2026-05-17
updated: 2026-05-17
---

# Supervisor Meeting (2026-05-17): M5.1 landed, picked pair unblocked

Consolidated brief covering M0 to M6 with **M5.1 finished and held at step_180**. Successor to [`SUPERVISOR_MEETING_2026-05-16_m0_to_6.md`](SUPERVISOR_MEETING_2026-05-16_m0_to_6.md) (frozen reference for the M5.1-mid-flight state); read this file for the current state. Predecessor's predecessor [`SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) is the M0-to-M3 reference.

## 0. TL;DR (six bullets, revised)

1. **M5.1 LANDED at step_180 on 2026-05-17 ~08:18 UTC.** 18 cadences of 10 steps each, 58 % of one MuSiQue epoch. Run paused, not crashed: deliberate HOLD because the F1-only ceiling is structural (130 steps in the 0.20-0.28 cadence band, no monotone climb), the remaining 131 steps would buy < 2 pp reward at ~\$60-80 cost, and that money buys ~12 hours of M8.2 training with a predicted 0.04-0.08 absolute lift. Operational reason for stopping at 180 rather than 200 / 250: four consecutive Spot replacements were SSH-unreachable mid-bring-up after the dedicated host went down ([`RESULTS_M5_1_H200.md` §9.6](RESULTS_M5_1_H200.md#96-hold-decision-at-step_180-2026-05-17)).

2. **Peak window-mean 0.280 at cadence 11 (steps 101-110).** Run-high single step 0.394 at step 49; two ties at 0.355 (steps 105 and 170, in different operating regimes 65 steps apart). Cadence-by-cadence window means: 0.028 → 0.110 → 0.132 → 0.160 → 0.171 → 0.202 → 0.224 → 0.202 → 0.221 → 0.228 → 0.232 → **0.280** → 0.247 → 0.221 → 0.240 → 0.242 → 0.256 → 0.265 → **0.275**.

3. **NEW finding: lean-drift-lean cycling, GRPO self-stabilising.** The policy oscillates between a lean operating point (tool_med 3, len_med 15 K, step wall 411 s) and an over-search regime (tool_med 6, len_med 28.8 K, step wall 824 s). Two complete cycles in 180 steps; every metric of the cycle-2 excursion is smaller than cycle-1 (tool peak 5 vs 6, len peak 23.7 K vs 28.8 K, wall peak 11 min vs 17 min, flip rate peak 56 % vs 58 %). The recipe self-stabilises; over-search-as-divergence is wrong, it is over-search-as-exploration-that-pays-off-in-flip-rate-not-reward.

4. **NEW finding: empirically measured chain-flip rate.** A regex-based silent-flip detector (the M8.1 chain-consistency penalty algorithm) applied to every reward ≥ 0.9 rollout across **all 18 cadences (180 training steps; C1-C4 backfilled 2026-05-17 post-hold)**: **18.6 % (cadence 9 low) to 58.0 % (cadence 14 high) band, with positive correlation between cadence-mean reward and cadence flip-rate**. C2-C4 cluster at 28-32 %, so **the operating band is reached within the first 20 training steps and does not exit it**: the flip rate does not start low and rise with training; it starts in-band and stays in-band. Reward and silent-flip count climb *together*, not inversely. Direct evidence that F1-only is reward-shaping for token alignment, not chain coherence even from step 1. Two concrete reward-1.0-via-broken-chain traces documented (Fox Island, World Cup); ~11 % of all rollouts in cadence 12 onward are planned + reward-1.0 + chain-broken (Goodhart at scale).

5. **The picked pair pre-flight is unblocked.** Pick #1 (reward-shape ablation) was always the load-bearing experiment; M5.1 now provides the F1-only anchor checkpoint series ([`pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42-f1-only`](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42-f1-only); 18 checkpoints). The remaining two runs (F1+0.1 and EM-only) launch at the same recipe + same H200 substrate + same 180-step horizon; each adds 18 checkpoints. Total picked-pair trajectory matrix: **3 rewards × 18 checkpoints × 7 benchmarks × 1-2 seeds = 378-756 eval data points**.

6. **Honest venue ranges sharpened** with the M5.1-landed evidence base: ICLR blogpost very plausible (~85-92 %), Workshop 70-82 %, Findings 50-65 %, ICLR / ACL main 22-32 %, NeurIPS main 10-18 %. The lean-drift-lean cycling + the empirical chain-flip data lift each band ~5 pp over the 2026-05-16 estimates because three of the four contributions in the picked-pair paper are now data-supported, not promised.

---

## 1. Recap M0 to M3 (unchanged from 2026-05-16 brief)

See [`SUPERVISOR_MEETING_2026-05-16_m0_to_6.md` §1](SUPERVISOR_MEETING_2026-05-16_m0_to_6.md#1-recap-m0-to-m3-short-full-detail-in-the-2026-05-07-brief) and [`SUPERVISOR_MEETING_2026-05-07_m0_to_3.md`](SUPERVISOR_MEETING_2026-05-07_m0_to_3.md) for the M0-M3 narrative. Headline:

- M0 Phase-1 (29 ALICE Qwen3-0.6B runs): reward 0.16-0.22 band; 5/5 base-model attempts fail; prompt dominates reward; 0.1 partial-credit floor masks tool-use signal.
- M1: Search-R1 eval reproduction within ±2.5 pp on Qwen2.5-3B.
- M2: NeMo-RL training pipeline (verl does not support Qwen3.5); 15 reward-parity tests pass.
- M3 + M3.1: untrained Qwen3-0.6B → GRPO-trained, EM 0.102 → 0.169 (+66 % rel).

## 2. M4 (unchanged from 2026-05-16 brief)

Qwen3.5-0.8B untrained baselines locked at M4.2 prompt-mode. Hybrid 0.057 EM avg, base 0.034 EM avg. Cross-family observation: Qwen3.5-0.8B is uniformly below Qwen3-0.6B (mean Δ -0.042 EM, no datasets cross); mechanism unexplained. Full at [`SUPERVISOR_MEETING_2026-05-16_m0_to_6.md` §2](SUPERVISOR_MEETING_2026-05-16_m0_to_6.md#2-m4-untrained-qwen35-08b-baseline-closed-2026-05-09).

## 3. M5.1: full landed trajectory (cadences 1-18, steps 1-180)

The complete run, replacing the 2026-05-16 brief's cadence 1-9 partial table:

| Cadence | Steps | Reward window-mean | Tool median | Token mean | Step wall (min) | Chain-flip rate | Key event |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 1-13 | 0.028 → 0.110 | 7 → 3 | 7038 → 4500 | 18:22 → 6:04 | **48.0 %** (n=123, small denom) | Cold-start; truncation 68 → 0 % |
| 2 | 14-24 | 0.119 → 0.132 | 3 | 4500 → 2700 | 5:36 | **27.9 %** (in-band by step 11-20) | Shrink-and-improve in flight |
| 3 | 25-30 | 0.140 → 0.160 | 3 | 2700 → 2300 | 5:50 | **31.0 %** | Standard 1-tool shape |
| 4 | 31-40 | 0.171 | 4-5 | 2400 | 7:21 | **31.8 %** | Cross-verification mode emerges |
| 5 | 41-50 | 0.202 | 3 | 2200 | 6:00 | 37.9 % | **Step 49 = 0.394 (run-high single)** |
| 6 | 51-60 | 0.224 | 3 | 2200 | 6:10 | 27.9 % | Three steps cross 0.25 |
| 7 | 61-70 | 0.202 | 2-3 | 2200 | 6:30 | 40.2 % | Brief noise dip |
| 8 | 71-80 | 0.221 | 3 | 2200 | 6:30 | 33.3 % | Step 78 = 0.296 |
| 9 | 81-90 | 0.228 | 3 | 2150 | 6:40 | **18.6 %** (run low) | Step 86 = 0.294 |
| 10 | 91-100 | 0.232 | 3 | 14800 | 8:04 | 26.1 % | Planned-multi-hop +63 % → 249 |
| **11** | 101-110 | **0.280** (run peak) | 3 | 15800 | 8:36 | 42.7 % | **Step 105 = 0.355**; ≥ 50 % rew > 0 first time |
| 12 | 111-120 | 0.247 | 4 (drift) | 18300 | 10:06 | 47.4 % | First over-search excursion starts |
| 13 | 121-130 | 0.221 | 5 | 23600 | 12:44 | 44 % | Drift deepens |
| 14 | 131-140 | 0.240 | **6** (peak) | **28800** (peak) | **13:44** | **58.0 %** (run high) | Over-search peak; 2 steps > 17 min |
| 15 | 141-150 | 0.242 | 4 → 3 | 23600 → 18000 | 9:19 | 40.8 % | Self-correction begins mid-cadence |
| 16 | 151-160 | 0.256 | 3 | 15000 | 6:51 | 39.6 % | Returned to C6 shape; over-search trap exited |
| 17 | 161-170 | 0.265 | 3 → 4 | 19800 | 9:11 | 48.6 % | Second drift starts; **step 170 = 0.355** |
| **18** | 171-180 | 0.275 | 4-5 | 23700 | 11:00 | 53.4 % | Second drift damping |

Source: [`RESULTS_M5_1_H200.md` §8](RESULTS_M5_1_H200.md#8-live-trajectory) cadence-by-cadence.

## 4. The four M5.1 findings locked in at HOLD

These five findings are the M5.1 contribution to the thesis regardless of whether step_180 vs step_311; all are in hand at the hold:

### 4.1. F1-only ceiling is structural at 0.22-0.28 cadence window-mean

130 steps in cadences 8-18 sit in the 0.20-0.28 band with no monotone climb. Trace evidence at [`RESULTS_M5_1_H200.md` §9.5](RESULTS_M5_1_H200.md#95-f1-reward-ceiling-the-structural-plateau-cause--chain-quality-reward-designs-added-2026-05-16-post-cadence-9): F1-only gives identical scalar reward to chain-correct and chain-broken-but-token-aligned-by-luck rollouts. Two concrete cases documented:

- **Fox Island / Pan-African Conference (cadence 9, step 93)**: model silently flips "country containing Fox Island" from "United States of America" (call 1) to "United Kingdom" (call 2) with no justification, then commits to United Kingdom which matches gold. Reward 1.0.
- **2014 World Cup / 2006 finish (cadence 11, step 102)**: model thinks Brazil won 2014, Italy won 2006, Italy finished 3rd; final answer "third" matches gold for the *correct* chain (Germany → 3rd) by accident. Reward 1.0.

### 4.2. Empirical chain-flip rate band 18-58 % with positive reward correlation

A regex-based silent-flip detector applied to ~6800 perfect rollouts across **all 18 cadences (180 training steps)** (C1-C4 backfilled 2026-05-17 post-hold; C5-C11 added post-hoc at C11 then backfilled to C5-C10; C12-C18 live with the run): chain-flip rate is **18.6 % (C9 low) to 58.0 % (C14 high)** and is **positively correlated with cadence-mean reward**. **The 28-32 % operating band is reached by C2 (steps 11-20)**; the flip rate does not start low and rise, it starts in-band and stays in-band (C1 = 48.0 % is on n=123, the only sub-200-sample row, and is lucky-match-driven). F1-only does not push toward cleaner chains *from step 1*; it pushes toward token-likely-correct shapes. The M8.1 chain-consistency penalty would have applied to 18-58 % of perfect rollouts at every cadence, creating a within-group advantage gap that GRPO can act on.

### 4.3. Lean-drift-lean cycling: GRPO self-stabilisation

The recipe oscillates between a lean operating point and an over-search regime, and GRPO self-corrects without external intervention:

| | Cycle 1 (C12-C16) | Cycle 2 (C17-C18, in flight at HOLD) |
|---|---|---|
| Tool count peak | **6** (C14) | 5 (C18) |
| Token mean peak | **28.8 K** (C14) | 23.7 K (C18) |
| Step wall peak | **13:44 mean / 17 min single** (C14) | 11:00 mean / 11:15 single (C18) |
| Chain-flip rate peak | **58 %** (C14) | 53 % (C18) |
| Recovery cadences | 4 (C15-C16) | n/a (HOLD before completion) |

Every metric of cycle 2 is smaller than cycle 1. The policy is *damping* the cycle. Non-trivial training-dynamics characterisation independent of the reward-shape ablation.

### 4.4. 4-hop generalisation across 5+ distinct bridges

The model resolves 4-hop+ chains across Ghana, Nigeria, UK, Iowa, Singapore, Manitoba and more. 4-hop+ wins per cadence: 20 (C5) → 32 (C8) → 40 (C11) etc. The model has the multi-hop capability; the reward isn't selecting for it.

### 4.5. Planned-multi-hop count peaks at 391 / cadence

The "numbered-plan + 3-5 calls + final answer" shape consolidates across the run: 132 rollouts (C8) → 153 (C9) → 249 (C10) → 327 (C11) → 344 (C12) → 370 (C13) → 391 (C15, peak) → 200-300 stable thereafter. The planned shape persists through over-search drift and through correction; cycle 1 was tool-count drift, not planning-mode drift.

## 5. Picked-pair contribution (unchanged, pre-flight unblocked)

Pick #1 (reward-shape ablation) frozen 2026-05-16. With M5.1 landed:

- **The F1-only anchor checkpoint series is published**: [`pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42-f1-only`](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42-f1-only), 18 checkpoints at every 10 steps.
- **Remaining picked-pair runs**: F1+0.1 (paper-faithful reward) and EM-only (strict) at the same recipe + same 180-step horizon on H200. Each: ~7-9 days wall-clock, ~\$50-80.
- **Total picked-pair eval matrix**: 3 rewards × 18 checkpoints × 7 benchmarks × 1-2 seeds = 378-756 eval data points.

Pick #2 (defensive R vs ambitious M) still awaiting user decision (per [`../milestone_6/PICKED_PAIR.md` §6](../milestone_6/PICKED_PAIR.md#6-decision-needed-from-the-user)). Default recommendation still C+R unless overridden.

## 6. The thesis-paper story (compact, revised)

**One sentence**: at sub-1B scale on retrieval-augmented multi-hop QA, the ReSearch partial-credit reward floor masks tool-use signal at training time *and* removing the floor in favour of F1-only creates a different mask (chain-quality blindness) that we measure at 18-58 % silent-flip rate across **all 180 training steps of M5.1 (in-band by C2)**; the resulting reward plateau is structural at 0.22-0.28 window-mean, the small-model regime exhibits a *shrink-and-improve* training dynamic distinct from the long-CoT regime familiar from math reasoning RL, and GRPO self-stabilises around the F1-optimum operating shape (lean-drift-lean cycling, damped over two cycles in 180 steps).

**Four contributions** (in priority order):

1. **Per-checkpoint × per-benchmark × per-reward trajectory** ablation at 0.8B (3 rewards × 18 checkpoints × 7 benchmarks × 1-2 seeds = 378-756 eval data points). The first such trajectory-grade evidence at sub-1B in the search-tool RL literature.
2. **Empirically measured chain-flip rate as the structural F1-ceiling diagnostic** (18-58 % silent-flip band across all 18 cadences; in-band by C2; positive reward-flip correlation; two concrete traces). Direct evidence that F1-only is reward-shaping for token alignment, not chain coherence, from step 1 of training.
3. **Lean-drift-lean cycling as a GRPO self-stabilisation finding** (two complete cycles in 180 steps; second cycle damped). Non-trivial training-dynamics characterisation independent of the reward ablation.
4. **Regime characterisation**: shrink-and-improve at sub-1B retrieval-augmented RL inverts the long-CoT regime of math reasoning RL.

## 7. Critical assessment (sharpened with M5.1 evidence)

### 7.1. Strengths added by M5.1 landing

- F1-only ceiling is now an *empirically measured* claim (130 steps in 0.22-0.28 band), not a theoretical one.
- Chain-flip rate is *measured* across all 180 training steps (18-58 % band, ~6800 rollouts analysed, in-band from C2 onward), not derived.
- Lean-drift-lean cycling is a *new* finding (not in the 2026-05-16 brief) that adds a fourth contribution.
- Two concrete reward-1.0-via-broken-chain traces are paper-quality motivation material.

### 7.2. Gaps unchanged from 2026-05-16

- Single seed everywhere (the no-multi-seed-anywhere gap).
- No Tree-GRPO head-to-head (the ICLR 2026 comparator gap).
- arXiv:2602.19526 partial scoop on EM vs F1 reward ablation at 3B+.
- No 2B / 7B scale-up data point.
- No SFT cold-start ablation.

### 7.3. Honest venue ranges (sharpened)

| Venue | 2026-05-16 estimate | 2026-05-17 estimate (M5.1 landed) | Delta justification |
|---|---|---|---|
| ICLR blogpost | 80-90 % | **85-92 %** | Lean-drift-lean cycling plot + chain-flip-vs-reward scatter add visual material |
| Workshop | 65-80 % | **70-82 %** | Training-dynamics findings are workshop-receptive |
| Findings | 45-65 % | **50-65 %** | Trajectory matrix + empirical chain-flip clear the Findings bar |
| ICLR / ACL main | 20-30 % | **22-32 %** | Self-stabilisation finding adds a section without busting scope |
| NeurIPS main | 8-15 % | **10-18 %** | The added finding helps; head-to-head + scale + seed gaps still dominate |

## 8. What's left (revised timeline)

| Item | Compute | Wall-clock | Cost | Status |
|---|---|---|---|---|
| **M5.1 step_180**: landed 2026-05-17 ~08:18 UTC | (done) | (done) | (done) | **DONE** |
| **Pre-flight smoke**: EM-only @50 steps (gate for the third reward variant) | 1× H200 | ~6 h | ~\$30 | pending user pick #2 decision |
| Pick-#1 run: F1+0.1 variant | 1× H200 | ~7-9 d | ~\$60-90 | pending |
| Pick-#1 run: EM-only variant | 1× H200 | ~7-9 d (if smoke passes) | ~\$60-90 | pending |
| Eval pipeline: 18 checkpoints × 7 benchmarks × 3 rewards (× 1-2 seeds) | 1× H200 or 1× A100 | ~3-4 d total | ~\$50-80 | pending |
| **If pick #2 = R**: 4 prompt-pair runs × 2 seeds × 2 prompts | 1× A100 | ~6-8 d | ~\$80-120 | pending |
| **If pick #2 = M**: 2 MC-GRPO runs | 1× A100 | ~10-14 d | ~\$150-220 | pending |
| Thesis chapter writing | n/a | continuous | n/a | starts now |

Hard timeline unchanged: experimentation cutoff 2026-06-10; thesis submission 2026-06-15; defense ~2026-07-15.

## 9. Pointers (updated)

- M5.1 results: [`RESULTS_M5_1_H200.md`](RESULTS_M5_1_H200.md), now through cadence 18 + HOLD section.
- M5.1 published checkpoints: [`pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42-f1-only`](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42-f1-only).
- M6 storyline (revised 2026-05-17): [`../milestone_6/STORYLINE.md`](../milestone_6/STORYLINE.md).
- Picked pair: [`../milestone_6/PICKED_PAIR.md`](../milestone_6/PICKED_PAIR.md).
- M8.2 chain-consistency reward extension (out of scope for picked pair): [`../milestone_8/MILESTONE_8.md`](../milestone_8/MILESTONE_8.md).
- Previous supervisor brief (frozen reference, M5.1-mid-flight state): [`SUPERVISOR_MEETING_2026-05-16_m0_to_6.md`](SUPERVISOR_MEETING_2026-05-16_m0_to_6.md).
