---
title: CONVERSATION CONTEXT
tags: []
source: internal
created: 2026-05-04
updated: 2026-05-06
---

# Conversation Context

> Living snapshot of project status, decisions, and pointers to the canonical docs. Update this when something material changes; keep historical sections frozen below.

**Last updated**: 2026-05-06

> **Doc reorganisation 2026-05-06**: The literature-review strand (`LITERATURE_REVIEW.md`, `SURVEY.md`, `SURVEY_FOCUSED.md`, `SURVEY_OVERFLOW.md`) moved to [`docs/research/`](../research/) so this folder is purely thesis-writing. See [`docs/research/CONVERSATION_CONTEXT.md`](../research/CONVERSATION_CONTEXT.md) for the research-strand snapshot. The supervisor-meeting brief is now date-suffixed: `SUPERVISOR_MEETING_2026-05-07.md`.

---

## 1. Status (one paragraph)

Phase-1 of the thesis (Qwen3-0.6B prompt + base-model ablations on 1× A100, ALICE) is closed and documented in `RESULTS_v0.md` and `RESULTS_v1.md`. The Search-R1 evaluation baseline is reproduced (`milestone_one/COMPARISON_PLAN_B_v1.md`, ±2.5 pp of paper). The Phase-2 NeMo-RL training pipeline for Qwen3.5-2B is built and launch-ready. Wall-clock is observed at 11 to 17 days per run on 1× A100-80GB, which kills the original reward-function-ablation plan. The thesis question is being reframed from "extending RLVR via tool-use" to "what is the optimised training recipe for a small LM under a single-A100 budget"; proposed recipe stacks E2H curriculum + S-GRPO + MC-GRPO on a Search-R1 GRPO baseline. Net is captured in `SUPERVISOR_MEETING_2026-05-07.md`.

## 2. Hard timeline

| Date | Milestone |
|---|---|
| 2026-05-07 | Supervisor meeting (`SUPERVISOR_MEETING_2026-05-07.md` is the brief) |
| 2026-06-10 | Experimentation must be finished |
| 2026-06-15 | Thesis submission |
| ~2026-07-15 | Defense |

## 3. Compute and constraints (confirmed)

- **GPU**: 1× A100-80GB (rented; ALICE retired going forward).
- **Budget**: ~$1000 USD for training.
- **Implication**: 2 to 3 full Qwen3.5-2B GRPO runs feasible at observed wall-clock (11 to 17 d / run on 1× A100; 5 to 8.5 d on 1× H100). No reward-ablation sweep affordable.

## 4. Decisions captured

| Decision | When | Rationale |
|---|---|---|
| Use hybrid Qwen3 (not base) | After v0 | Base could not produce structured tool_call format from cold start; 5/5 v1 base attempts had 0 tool calls |
| Switch to `<tool_call>` JSON tags | v1 | In-distribution for Qwen3 chat template; same reward at equal step count as paper `<search>` tags |
| Drop reward-function ablation | After Phase-2 wall-clock | 11 to 17 d / run on 1× A100; multiple paired runs not affordable |
| Pivot from verl to NeMo-RL | Phase-2 setup | verl does not support Qwen3.5 |
| FAISS Flat IP → IVF-SQ8 for training | Phase-2 smoke | Flat IP times out under rollout HTTP load |
| Reframe thesis question | 2026-05-04 | Original RQs (reward modeling, meta-reasoning, curriculum) require a sweep that is not affordable; reframed to "is it feasible + what is the optimised recipe" |
| Proposed training recipe | 2026-05-04 | E2H curriculum + S-GRPO + MC-GRPO on Search-R1 GRPO baseline; all drop-in, all from [`docs/research/SURVEY_FOCUSED.md`](../research/SURVEY_FOCUSED.md) |

## 5. Working memory

Things that are easy to lose track of:

- **Two repos**: `reason_over_search` (thesis) and `research` (training infrastructure, Qwen3-0.6B port). Both reference each other.
- **Two W&B projects**: `research` (v0, 26 runs) and `research_revamp` (v1, 15 runs). All Qwen3-0.6B; analysis frozen.
- **CSVs of all 29 v0/v1 runs** are saved at `results_v0_assets/csv/` and `results_v1_assets/csv/` so the analysis can be regenerated without W&B access.
- **Reward function gotcha**: Search-R1's GitHub ships `qa_em.py` (paper-faithful EM-only) and `qa_em_format.py` (shaped 6-tier with non-zero defaults). Earlier project docs conflated them; caught in `training/SMOKE_RESULTS_2026-05-06.md` (renamed from _V4). Phase-2 NeMo-RL port uses EM-only.
- **`base_breakthrough` (v1, b8vv0qe2)** showing reward 0.7 is a reward-function-code change artifact, not learning. Configs are identical to `base_state_machine_a` which scored 0.0. Treat as instrumented, not earned.

## 6. Doc inventory

See `README.md` in this directory for the per-file index. The supervisor-facing summary is `SUPERVISOR_MEETING_2026-05-07.md`. Literature/survey strand lives in [`docs/research/`](../research/).

---

## Historical (frozen)

### Original vision (Plan A and Plan B, Feb 2026)

Three threads in `ORIGINAL_PLAN_A.md`: reward model design, meta-reasoning tags, curriculum + transfer learning. Core question: "Does domain adaptation mean learning a way to *interact with* the domain (e.g. tool use) rather than internalising the domain?"

Plan B (`ORIGINAL_PLAN_B.md`) RQs:
- RQ1: Current approaches for domain expansion?
- RQ2: How to model reward functions?
- RQ3: Can meta-reasoning improve training?
- RQ4: Is curriculum-based training feasible?

Original timeline: Feb (lit) → Mar (pipeline) → Apr (experiments) → May (analysis) → Jun (writing). Slipped: pipeline + experiments compressed into Apr to early May. The Phase-2 compute reality (Section 3 above) forced the reframing in Section 4.

### The pivot to ReSearch + Search-R1

Picked as the entry papers because they are the cleanest concrete instance of "tool-use as the surrogate objective" in the lit review. ReSearch flagged as "EXACTLY WHAT I WANTED TO DO" in [`docs/research/LITERATURE_REVIEW.md`](../research/LITERATURE_REVIEW.md). The "non-verifiable domains" framing collapsed into "exact-match QA over Wikipedia" (verifiable via tool); the reframing in Section 4 above resolves this loose end by changing the question from "extend RLVR" to "find the optimised recipe".
