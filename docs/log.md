---
title: Log
tags: []
source: internal
created: 2026-05-06
updated: 2026-05-07
---

# Wiki log

Append-only chronological record of ingest events, decisions, and learnings. New entries go on top (newest first). Each day gets a single `## YYYY-MM-DD` heading; entries within a day are bullets.

Format:

```
## YYYY-MM-DD
- Ingested: <source path or URL> -> <pages touched>
- Learned: <one-liner>, filed in <page>
- Decision: <one-liner>, recorded in <page>
- Lint: <summary of lint cycle output and what was fixed>
```

Conventions:

- Cite sources by relative path (e.g. `../raw/2026-05-06_foo.pdf`) or by full URL.
- Cite wiki pages by relative path from this file (e.g. `report/CONVERSATION_CONTEXT.md`).
- Keep each bullet to one line. Synthesis goes in the wiki page, not here.
- Don't rewrite past entries; if a fact turns out wrong, add a correcting bullet on the day the correction was made.

---

## 2026-05-08

- Decision: launched **M3.1** — second Phase-1 checkpoint evaluation. Target: `p3_decide_no_ex_el6s2d2h` (Phase-1 v0 highest-reward run: end reward 0.215 / +43 % rel over 2280 steps; vs M3's z7kcxfof at 0.190 / +28 % rel over 1046 steps). Settles whether the +0.025 rollout-reward gap translates to held-out EM, or is mostly the partial-credit-floor artifact. Sbatch job **2134645** queued (gpu-short, 4 h, ALICE A100-80GB). Setup is additive-only (no new alignment fixes; the M3 14-fix audit holds): added `P3_DECIDE_NO_EX_TEMPLATE` to `evaluation_research/flashrag/search_r1/templates.py` (verbatim from training, recovered from `report/RESULTS_v0.md` §`p3_decide_no_ex (el6s2d2h)`); added `QWEN3_TEMPLATES` registry keyed on `prompt_mode`; switched `prompt_mode == 'qwen3'` checks in `active_pipeline.py` and `run_eval.py` to `prompt_mode.startswith('qwen3')` so qwen3 family shares retrieval format / budgets / `enable_thinking`. Checkpoint converted verl-FSDP → HF via `python -m verl.model_merger merge --backend fsdp …` (1.5 GB safetensors at `eval/qwen_3_0.6b_v0_no_ex/`). Doc trail: [`milestone_three/MILESTONE_3.1.md`](milestone_three/MILESTONE_3.1.md), [`report/CODE_SETUP_v2.md`](report/CODE_SETUP_v2.md) §13.5, [`report/RESULTS_v2.md`](report/RESULTS_v2.md) §14, [`report/SUPERVISOR_MEETING_2026-05-07.md`](report/SUPERVISOR_MEETING_2026-05-07.md) §4.8, [`report/PROGRESS_REPORT_01.md`](report/PROGRESS_REPORT_01.md).
- Decision: bumped SGLang `/health`-readiness wait in `scripts/sbatch_m3.sh` from 300 s to 600 s after sbatch **2134645** failed at the cliff (M3 reference: 260 s; this run: cold cache on `node875`, would have needed ~310 s). The retriever + checkpoint + pipeline were all healthy at the failure point; only the wait window was too tight. Re-submitted as sbatch **2134663** (same partition, same flags, only the wait window bumped). 600 s is the same budget the retriever wait already uses, so the two waits are now symmetric.
- Decision: published the M3.1 checkpoint to HuggingFace as [`pantomiman/Qwen3-0.6B-v0.1`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0.1) — parallel to the M3 checkpoint at [`pantomiman/Qwen3-0.6B-v0`](https://huggingface.co/pantomiman/Qwen3-0.6B-v0); the `.1` minor-version suffix distinguishes the no-example variant trained from the same base + algorithm + reward + data, varying only the prompt. Model card documents run id, training-time signature, and the `v0` ↔ `v0.1` relationship.

## 2026-05-07

- Decision: M3 closed. First evaluation of the v0 GRPO checkpoint (`p1_basic_w_ex_z7kcxfof`, 1046 / 9968 steps, 23h 47m 30s wall on 1× A100-40GB) vs. the untrained Qwen3-0.6B hybrid on the 7 paper QA benchmarks. Headline: average EM 0.102 → 0.155 (+0.053 absolute, +52 % relative) across full Plan A (51,713 items / variant); 6 / 7 datasets improved; held-out generalisation rules out memorisation. The eval pipeline is now pinned and reusable for Phase-2 NeMo-RL evaluation. Full record: [`report/RESULTS_v2.md`](report/RESULTS_v2.md); consolidated brief: [`report/SUPERVISOR_MEETING_2026-05-07.md`](report/SUPERVISOR_MEETING_2026-05-07.md) § 4; closed milestone: [`milestone_three/MILESTONE_3.md`](milestone_three/MILESTONE_3.md).
- Decision: ReSearch action-format attribution corrected. The published ReSearch paper uses `<search>` / `<result>` tags (verified upstream `re-search` commit `51d98e1`); the `<tool_call>` JSON variant in our local `re-search/` checkout is a separate ablation we introduced (commit `2c32dd3`, 2026-04-12) and tested in the v1 block. Earlier docs that called `<tool_call>` "ReSearch's format" were misattributing our own ablation. Fixed in `report/SUPERVISOR_MEETING_2026-05-07.md` § 4.5 and `report/RESULTS_v2.md` § 10.7.
- Decision: heavy rebuild of `report/SUPERVISOR_MEETING_2026-05-07.md` into a 6-section canonical chapter-closing summary (§ 0 TL;DR · § 1 Phase-1 · § 2 M1 baseline · § 3 M2 NeMo-RL pivot · § 4 M3 results · § 5 compute reality + reframed RQ · § 6 next-steps recipe). Cross-references in downstream docs bumped from old § 1 / § 2 / § 3 to the new section numbers.
- Created: [`research/QUESTIONS.md`](research/QUESTIONS.md), running register for open research questions and "register this as a question" entries. Threaded into [`../claude/CLAUDE.md`](../claude/CLAUDE.md) (repo layout + new "register a question" rule under "How I want you to work with me"), [`research/README.md`](research/README.md) (intro + doc inventory), and [`research/CONVERSATION_CONTEXT.md`](research/CONVERSATION_CONTEXT.md) (doc inventory). First question filed: Q1 sequence packing in verl and NeMo-RL, works with Qwen3 but not Qwen3.5.

## 2026-05-06

- Added: E2H curriculum recipe entry in [`research/RECIPES.md`](research/RECIPES.md) (status: planned, item #5 in active ablation list). Lays out the data-side curriculum mechanics, hop-count difficulty mapping for retrieval QA (NQ=0/HotpotQA=1/MuSiQue=2), borrow-worthy tricks in priority order (hop-count bucketing → stage-based fade → Gaussian schedule), and the JustRL counter-evidence caveat that the C-minimal control must validate against. No runs yet.
- Ingested: https://arxiv.org/abs/2506.06632 (E2H Reasoner, ICLR 2026) -> [`papers/2506.06632_e2h.md`](papers/2506.06632_e2h.md) (deep ingest), [`research/DATASET_ANALYSIS.md`](research/DATASET_ANALYSIS.md) (curriculum application section). PDF capture pending; should land at `raw/papers/2026-05-06_e2h.pdf`. Existing brief card in [`research/SURVEY_FOCUSED.md §5.4`](research/SURVEY_FOCUSED.md) left in place.
- Created: [`research/DATASET_ANALYSIS.md`](research/DATASET_ANALYSIS.md) - covers all 7 Search-R1 benchmarks (NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle) with hops, sizes, structure, real example pulled from our `data/<dataset>/test.jsonl`, and v1 base EM vs paper. Maps E2H difficulty ordinals to NQ (0) / HotpotQA (1) / MuSiQue (2); the other 4 stay held-out OOD.
- Learned: E2H paper does **not** evaluate on retrieval-augmented multi-hop QA (only closed-book MATH / Countdown / Blocksworld). Applying it to our setting is a thesis novelty; convergence rates may differ from paper's pp gains. Filed in [`papers/2506.06632_e2h.md`](papers/2506.06632_e2h.md) "Limitations" + "Takeaways for us".
- Created: [`research/RECIPES.md`](research/RECIPES.md) - living catalog of tested and referenced recipes with analysis, bad/good breakdown, our results with actual run numbers, and 8 cross-cutting learnings (L1-L8) grounded in the v0/v1/smoke result files.
- Synthesised: [`research/RECIPE_COMPARISON.md`](research/RECIPE_COMPARISON.md) - grounded side-by-side of all three RL+search papers (Search-R1, R1-Searcher, ReSearch). Every row sourced from the paper or scripts; rows we could not verify are marked "not stated" or "not in our extract".
- Ingested: [`raw/papers/2026-05-06_search-r1.pdf`](raw/papers/2026-05-06_search-r1.pdf) (Search-R1, arXiv 2503.09516v5) -> [`papers/2503.09516_search-r1.md`](papers/2503.09516_search-r1.md). This is our M1 reproduction target (Plan B v1 within ±2.5 pp paper avg, see [`milestone_one/COMPARISON_PLAN_B_v1.md`](milestone_one/COMPARISON_PLAN_B_v1.md)). Stated compute: 8 H100, single node, 500 steps; wall-clock not stated. Headline GRPO group size G not in our extract (ablated 1/3/5 in Appendix H).
- Decision: enforce the no-inference rule on paper notes. Paper-note fields that are not stated are now marked "not stated" rather than inferred. Strengthened in [`papers/README.md`](papers/README.md) under "No inference rule"; existing R1-Searcher and ReSearch notes scrubbed of GPU-type and wall-clock guesses; ReSearch's "0.46 EM at 7B" mistake corrected (that figure is the 32B-Instruct row).
- Ingested: [`raw/papers/2026-05-06_r1-searcher.pdf`](raw/papers/2026-05-06_r1-searcher.pdf) (R1-Searcher, arXiv 2503.05592v2) -> [`papers/2503.05592_r1-searcher.md`](papers/2503.05592_r1-searcher.md). Headline takeaways: 2-stage curriculum + asymmetric `+0/-2` format penalty are the borrow-able tricks; G=16 / 29k generate-max not affordable at our scale (8 GPUs, type not stated).
- Ingested: [`raw/papers/2026-05-06_research-rl.pdf`](raw/papers/2026-05-06_research-rl.pdf) (ReSearch, arXiv 2503.19470v3, NeurIPS 2025) -> [`papers/2503.19470_research.md`](papers/2503.19470_research.md). This is the recipe behind our [v0](report/RESULTS_v0.md) and [v1](report/RESULTS_v1.md) Phase-1 Qwen3-0.6B results. 64×H800 paper config; the 0.1 partial-credit reward is the most ablation-worthy line in the loss and the paper does not ablate it.
- Decision: stand up [`papers/`](papers/) as a wiki section for deep, single-paper notes (companion to `research/SURVEY.md`). Convention captured in [`papers/README.md`](papers/README.md); template at [`templates/paper.md`](templates/paper.md).
- Decision: adopt Karpathy's wiki workflow on top of `docs/`. Recorded in [`SCHEMA.md`](SCHEMA.md), pointer added to [`../claude/CLAUDE.md`](../claude/CLAUDE.md).
- Ingested: [Karpathy gist on personal wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) -> [`SCHEMA.md`](SCHEMA.md), [`raw/README.md`](raw/README.md), this file.
- Lint: post-papers-ingest run, 96 files total. Same 3 pre-existing broken links as before (none introduced by paper ingestion); orphans = 8 sankar/* (CV strand, expected) + 2 templates/* (`daily.md`, `zettel.md`; invoked by Templater plugin, not linked). New files [`papers/README.md`](papers/README.md), [`papers/2503.05592_r1-searcher.md`](papers/2503.05592_r1-searcher.md), [`papers/2503.19470_research.md`](papers/2503.19470_research.md), [`templates/paper.md`](templates/paper.md) all link-resolve cleanly.
- Lint: first lint cycle run. 3 pre-existing broken links flagged (not introduced by wiki conversion): `archive/training/SMOKE_RESULTS_2026-05-02_smoke_shape.md` -> `local_retriever/README_v1.md`; `milestone_one/MILESTONE_1.1_QWEN_BASELINES.md` -> `RESULTS_QWEN3_BASELINE.md` (file not yet authored, Jose-owned task); `retriever/INDEX_ARCHITECTURE.md` -> `../../../index_creation/README.md` (assumes `/workspace/...` host path). 8 sankar/* orphans (CV strand, expected; only some are linked from `sankar/00_CONVERSATION_CONTEXT.md`). 1 frontmatter-incomplete (`sankar/03_RESUME_LEGACY.md` has Obsidian-style frontmatter with different keys).
- Learned: paper batch shape is 2560 traj/step (512 prompts × 5 gen + 10 grad updates); ours is 510 traj/step (1/5th paper); per-step time is rollout-dominated, so paper config on 1× A100 is ~5× slower per step (~55-85 d / run). Filed in [`training/PAPER_VS_OURS_TRAINING.md`](training/PAPER_VS_OURS_TRAINING.md), [`report/SUPERVISOR_MEETING_2026-05-07.md`](report/SUPERVISOR_MEETING_2026-05-07.md), [`report/PROGRESS_REPORT_01.md`](report/PROGRESS_REPORT_01.md).

## 2026-05-05

- Decision: pivot the thesis question from "extend RLVR via tool-use" to "is post-training a small LM to Search-R1 results feasible under realistic resource constraints, and what is the optimised recipe?" Recorded in [`report/SUPERVISOR_MEETING_2026-05-07.md`](report/SUPERVISOR_MEETING_2026-05-07.md).
- Decision: candidate recipe = E2H curriculum + S-GRPO + MC-GRPO with a JustRL plain-GRPO control. Recorded in [`report/SUPERVISOR_MEETING_2026-05-07.md`](report/SUPERVISOR_MEETING_2026-05-07.md), expanded in [`TODO_2026-05-04.md`](TODO_2026-05-04.md).

## 2026-05-04

- Learned: Phase-1 Qwen3-0.6B partial-credit reward creates a 0.1 floor that masks the tool-use signal (3-6 pp gap between tool-using and no-tool runs). Filed in [`report/RESULTS_v0.md`](report/RESULTS_v0.md), [`report/RESULTS_v1.md`](report/RESULTS_v1.md).
- Learned: Qwen3-0.6B base model cannot bootstrap the tool-call format from cold-start (5/5 v1 base attempts stayed at 0 tool calls, longest 2300 steps). Filed in [`report/RESULTS_v1.md`](report/RESULTS_v1.md). Decision: don't retry without SFT warm-start.
