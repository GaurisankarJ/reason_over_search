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
| [`LITERATURE_REVIEW.md`](./LITERATURE_REVIEW.md) | Project-personal working notebook. Thesis framing, introduction outline, per-paper annotations with `==notes==` and `⭐` markers, and five appendices preserving thinking-as-it-happened. Drove the v0 / v1 decisions. |
| [`SURVEY.md`](./SURVEY.md) | Reference-style survey of the RLVR field. 14 thematic sections (Foundations, Verifier Design, Policy Optimization, Exploration, Reward Hacking, Self-Verification, Multi-Task, Open-Ended, Tool-Use, Mechanism Studies, Theoretical Foundations, Measurement, Open Challenges, Bibliography). Each section ends with paper cards in standardised Summary / Problem / Method / Result / Takeaway / ELI5 format. 96 cards total. |
| [`SURVEY_FOCUSED.md`](./SURVEY_FOCUSED.md) | Project-specific subset of `SURVEY.md`: only the papers relevant to Qwen3.5-2B / single-A100 / search-augmented. Source of the next-steps recipe (E2H curriculum + S-GRPO + MC-GRPO) used in § 2 of `SUPERVISOR_MEETING.md`. |
| [`SURVEY_OVERFLOW.md`](./SURVEY_OVERFLOW.md) | 15 adjacent papers (foundational priors, surveys, alternative directions) kept for completeness. |

## Assets

- `results_v0_assets/`: 17 PNGs (3 combined + 14 per-run) and 14 CSVs.
- `results_v1_assets/`: 21 PNGs (6 combined + 15 per-run) and 15 CSVs.
- `supervisor_assets/`: 1 storyline PNG.

All 29 W&B run histories are saved as CSV under `results_v0_assets/csv/` and `results_v1_assets/csv/` so analyses are reproducible without W&B access.
