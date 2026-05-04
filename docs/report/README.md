# `docs/report/`

Thesis-facing record. Files at the top of this list are the canonical entry points; everything below is supporting detail.

## Start here

| File | Purpose |
|---|---|
| [`SUPERVISOR_MEETING.md`](./SUPERVISOR_MEETING.md) | Supervisor-facing summary. § 1 closes Phase-1 (Qwen3-0.6B); § 2 proposes the next-steps recipe (E2H curriculum + S-GRPO + MC-GRPO). Glanceable. |
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

## Original plan (frozen, historical)

| File | What |
|---|---|
| [`ORIGINAL_PLAN_A.md`](./ORIGINAL_PLAN_A.md) | Abstract / introduction draft (Feb 2026): three research threads. |
| [`ORIGINAL_PLAN_B.md`](./ORIGINAL_PLAN_B.md) | Problem statement: extend RLVR to non-verifiable domains via tool-use. RQ1 to RQ4. |

The reframing of the research questions is in `SUPERVISOR_MEETING.md` § 1 ("Proposed reframing").

## Literature

| File | What |
|---|---|
| [`LITERATURE_REVIEW.md`](./LITERATURE_REVIEW.md) | Original RLVR + tool-use lit review. Working notes, paper list, tagged "important" papers. |
| [`SURVEY.md`](./SURVEY.md) | Comprehensive survey, deep version. |
| [`SURVEY_FOCUSED.md`](./SURVEY_FOCUSED.md) | Focused subset with paper cards. The next-steps recipe (E2H, S-GRPO, MC-GRPO) is sourced from § 2 and § 5 here. |
| [`SURVEY_OVERFLOW.md`](./SURVEY_OVERFLOW.md) | Out-of-scope-but-relevant references. |

## Assets

- `results_v0_assets/`: 17 PNGs (3 combined + 14 per-run) and 14 CSVs.
- `results_v1_assets/`: 21 PNGs (6 combined + 15 per-run) and 15 CSVs.
- `supervisor_assets/`: 1 storyline PNG.

All 29 W&B run histories are saved as CSV under `results_v0_assets/csv/` and `results_v1_assets/csv/` so analyses are reproducible without W&B access.
