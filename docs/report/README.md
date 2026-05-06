---
title: README
tags: []
source: internal
created: 2026-05-04
updated: 2026-05-06
---

# `docs/report/`

Thesis-facing record. Files at the top of this list are the canonical entry points; everything below is supporting detail.

## Start here

| File | Purpose |
|---|---|
| [`SUPERVISOR_MEETING_2026-05-07.md`](./SUPERVISOR_MEETING_2026-05-07.md) | Supervisor-facing summary for the 2026-05-07 meeting. § 1 closes Phase-1 (Qwen3-0.6B); § 2 proposes the next-steps recipe (E2H curriculum + S-GRPO + MC-GRPO). Glanceable. PDF alongside. |
| [`CONVERSATION_CONTEXT.md`](./CONVERSATION_CONTEXT.md) | Living snapshot of project status, decisions, compute, and a doc inventory. Update when something material changes. |

## Phase-1 results (Qwen3-0.6B, ALICE, frozen)

| File | What |
|---|---|
| [`RESULTS_v0.md`](./RESULTS_v0.md) | First ablation block (W&B project `research`). 14 runs: prompt sensitivity on hybrid model + base-model attempts. 17 PNGs embedded; full prompt strings recovered from W&B notes. CSVs at `results_v0_assets/csv/`. |
| [`RESULTS_v1.md`](./RESULTS_v1.md) | Second ablation block (W&B project `research_revamp`). 15 runs: new in-distribution `<tool_call>` prompt format, fresh base-model attempts, reward-function probing. 21 PNGs embedded; CSVs at `results_v1_assets/csv/`. |

Storyline plot used in the supervisor doc lives at `supervisor_assets/storyline_v0_v1.png`.

## Setup and codebase

| File | What |
|---|---|
| [`CODE_SETUP_v0.md`](./CODE_SETUP_v0.md) | What changed in the ReSearch port vs. paper code: Qwen2.5-7B → Qwen3-0.6B, verl pin → verl_latest, faiss-gpu → faiss-cpu, 4-GPU → 1-GPU profiles, prompt and reward rewrites. Hyperparameter matrices side-by-side with paper. |
| [`CODE_SETUP_v1.md`](./CODE_SETUP_v1.md) | What changed from v0 to the M2 NeMo-RL pipeline: Qwen3.5-2B, NeMo-RL replaces verl, EM-only reward, two prompt arms, 6 smoke-run bugs fixed, Qwen3.5 architecture constraints (sequence packing off, enable_thinking), IVF-SQ8 retriever default, bootstrap.sh env setup. Quick-summary format. |

## Original plan (frozen, historical)

| File | What |
|---|---|
| [`ORIGINAL_PLAN_A.md`](./ORIGINAL_PLAN_A.md) | Abstract / introduction draft (Feb 2026): three research threads. |
| [`ORIGINAL_PLAN_B.md`](./ORIGINAL_PLAN_B.md) | Problem statement: extend RLVR to non-verifiable domains via tool-use. RQ1 to RQ4. |

The reframing of the research questions is in `SUPERVISOR_MEETING_2026-05-07.md` § 1 ("Proposed reframing").

## Literature (moved)

The literature-review and survey strand (`LITERATURE_REVIEW.md`, `SURVEY.md`, `SURVEY_FOCUSED.md`, `SURVEY_OVERFLOW.md`) lives in [`docs/research/`](../research/) alongside the algorithm + systems research. See [`docs/research/README.md`](../research/README.md) and [`docs/research/CONVERSATION_CONTEXT.md`](../research/CONVERSATION_CONTEXT.md).

## Assets

- `results_v0_assets/`: 17 PNGs (3 combined + 14 per-run) and 14 CSVs.
- `results_v1_assets/`: 21 PNGs (6 combined + 15 per-run) and 15 CSVs.
- `supervisor_assets/`: 1 storyline PNG.

All 29 W&B run histories are saved as CSV under `results_v0_assets/csv/` and `results_v1_assets/csv/` so analyses are reproducible without W&B access.
