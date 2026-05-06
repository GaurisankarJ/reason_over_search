# CLAUDE.md — Reason Over Search

You are a helpful research assistant with years of experience. Your job is to help me reproduce, extend, and ablate retrieval-augmented reasoning models. This file is your persistent memory: read it first whenever you restart so you can pick up where we left off.

This file is a living source of truth; update it (or ask me to) whenever a load-bearing fact changes. **Style**: avoid em-dashes (`—`, `--`) in any prose you write for me; they read as LLM-generated. Use semicolons, colons, parentheses, or "X to Y" instead.

**If you only read one thing after this file**: [docs/TODO_2026-05-04.md](../docs/TODO_2026-05-04.md) is the canonical catch-up doc with the active ablation path. State-of-the-world snapshots: [docs/report/CONVERSATION_CONTEXT.md](../docs/report/CONVERSATION_CONTEXT.md) (thesis status), [docs/internship/CONVERSATION_CONTEXT.md](../docs/internship/CONVERSATION_CONTEXT.md) (Alstom internship side), and [docs/training/CONVERSATION_CONTEXT.md](../docs/training/CONVERSATION_CONTEXT.md) (Phase-2 training pipeline; load this when working on training). [docs/report/SUPERVISOR_MEETING.md](../docs/report/SUPERVISOR_MEETING.md) is the two-page story so far for the supervisor (also as PDF in `docs/report/SUPERVISOR_MEETING.pdf` and `docs/report/PROGRESS_REPORT_01.pdf`).

## What this project is

We are reproducing and extending two papers on RL-trained retrieval-augmented LLMs:

1. **ReSearch** — https://www.alphaxiv.org/abs/2503.19470
2. **Search-R1** — https://www.alphaxiv.org/abs/2503.09516 — *current focus, Phase 1*

The Search-R1 setup: a Qwen2.5-3B policy is trained with GRPO to interleave `<search>…</search>` calls with reasoning, retrieve documents from a frozen Wikipedia FAISS index, and emit a final `<answer>…</answer>`.

Two milestones run in parallel:
- **M1 (eval reproduction)**: evaluate the published Search-R1 GRPO checkpoints end-to-end on 7 QA benchmarks. Reproduced within ±2.5 pp of paper; see [docs/milestone_one/COMPARISON_PLAN_B_v1.md](../docs/milestone_one/COMPARISON_PLAN_B_v1.md).
- **M2 (our training)**: port the Search-R1 GRPO loop to NeMo-RL and train Qwen3.5-2B from scratch on 1× A100, ablating a recipe-stack (E2H + S-GRPO + MC-GRPO + JustRL control). Pipeline built and smoke-tested; current frontier.

The official Search-R1 repo is `https://github.com/PeterGriffinJin/Search-R1`. They do not ship an evaluation pipeline, so the M1 one in this repo is adapted from FlashRAG/ReSearch and validated against the paper numbers; M2 ports the training loop to NeMo-RL because verl does not support Qwen3.5.

## Thesis context (snapshot 2026-05-05)

This work is a Master's thesis (Leiden University; supervisors track in `docs/report/`). The Alstom internship runs in parallel and shares the post-training story (see [docs/internship/CONVERSATION_CONTEXT.md](../docs/internship/CONVERSATION_CONTEXT.md)).

**Hard timeline**:

| Date | Milestone |
|---|---|
| ~2026-05-10 | Supervisor meeting (`SUPERVISOR_MEETING.md` is the brief) |
| 2026-06-10 | Experimentation must be finished |
| 2026-06-15 | Thesis submission |
| ~2026-07-15 | Defense |

**Compute and budget**: 1× A100-80GB rented on Vast.ai (ALICE retired going forward). ~$1000 USD total training budget. Observed wall-clock for a full Search-R1-shape Qwen3.5-2B GRPO run is **11 to 17 days on 1× A100** (5 to 8.5 d on 1× H100, 6.5 to 9.5 d on 2× A100). That gives **2 to 3 full runs**, so reward-ablation sweeps are off the table.

**Reframed thesis question** (after Phase-2 wall-clock reality landed): from "extend RLVR via tool-use" to **"is it feasible to post-train a small LM to Search-R1-level results under realistic resource constraints, and what is the optimised training recipe?"**. The candidate answer is a stack of three drop-in additions on a Search-R1 GRPO baseline run: **E2H curriculum + S-GRPO + MC-GRPO**, with a **JustRL plain-GRPO control** alongside (per [arxiv 2512.16649](https://arxiv.org/abs/2512.16649) "tricks may hurt"). Full proposal: [docs/report/SUPERVISOR_MEETING.md § 2](../docs/report/SUPERVISOR_MEETING.md).

**Two repos** (this is easy to lose track of):
- `reason_over_search` (this repo) — thesis writeup, eval reproduction (M1), training pipeline (M2 NeMo-RL port).
- `research` (sibling) — Phase-1 Qwen3-0.6B work on ALICE: the from-scratch port + the v0 / v1 ablation blocks. Both repos cross-reference.

**Two W&B projects** (Phase-1 frozen):
- `research` (v0, 26 Qwen3-0.6B runs in W&B including noise/aborted; 14 focus runs analysed in `RESULTS_v0.md`, paper `<search>`/`<result>` tags). CSVs at `docs/report/results_v0_assets/csv/`.
- `research_revamp` (v1, 15 Qwen3-0.6B runs, in-distribution `<tool_call>` tags). CSVs at `docs/report/results_v1_assets/csv/`.

For the Qwen3.5-2B Phase-2 block use a new project name (e.g. `reason_over_search_2b_v1`).

## Phase 1 goal (from [docs/MILESTONE_1.md](../docs/milestone_one/MILESTONE_1.md))

Reproduce the Search-R1 3B baseline for both checkpoints (`base` and `instruct`) on 7 benchmarks: Bamboogle, 2WikiMultiHopQA, TriviaQA, PopQA, MuSiQue, NQ, HotpotQA. Run each 5×, report averages. Constraints: reproducible on Vast.ai + in-house GPUs, minimize cost/wall-clock.

Paper targets we compare against (Qwen2.5-3B EM, Search-R1 v5 Table 3):

|              | NQ    | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | Avg   |
|---           |---    |---       |---    |---       |---    |---      |---        |---    |
| GRPO base    | 0.421 | 0.583    | 0.413 | 0.297    | 0.274 | 0.066   | 0.128     | 0.312 |
| GRPO instruct| 0.397 | 0.565    | 0.391 | 0.331    | 0.310 | 0.124   | 0.232     | 0.336 |

## Repo layout (what lives where)

- [`README.md`](../README.md) — Phase 1 milestone framing.
- **Catch-up + active ablation path**: [`docs/TODO_2026-05-04.md`](../docs/TODO_2026-05-04.md). The single best entry point for a new contributor (or a fresh agent session). Contains the lean reading path, the active target (≤10 h per run on 1× A100), and the prioritised ablation list.
- **Thesis writing / reporting**: [`docs/report/`](../docs/report/) — `CONVERSATION_CONTEXT.md` (status), `SUPERVISOR_MEETING.md` + `.pdf` (two-page story), `PROGRESS_REPORT_01.md` + `.pdf`, `RESULTS_v0.md` + `RESULTS_v1.md` (Phase-1 Qwen3-0.6B ablation blocks from sibling `research` repo), `SURVEY.md` / `SURVEY_FOCUSED.md` / `SURVEY_OVERFLOW.md` (paper survey; `SURVEY_FOCUSED` is the one to read for this project, with §8 Key Results + §9 TLDR), `LITERATURE_REVIEW.md`, `ORIGINAL_PLAN_A.md` / `ORIGINAL_PLAN_B.md`.
- **Algorithm + systems research**: [`docs/research/`](../docs/research/) — `INTEGRATION_GUIDE.md` (decision tree, start here), `PARADIGM_REVIEW.md` (algorithmic levers, v1→v2→v3 with JustRL counter-evidence), `RUNTIME_EFFICIENCY.md` (systems levers R1–R7 + C1 + G1–G3 + O1–O6 + M1–M3 with measured speedups).
- **Internship strand**: [`docs/internship/`](../docs/internship/) — `CONVERSATION_CONTEXT.md`, `6_MONTH_PLAN.md` (rewritten 2026-05-04: M1–M2 done, M3 in progress, M4–M6 TBA), `TECH_IMRAD_MARCH_v1.md` + `TECH_IMRAD_APRIL_v0.md` (living technical reports, both holistically rewritten 2026-05-04 to reflect the recipe-search pivot), `BUSINESS_REVIEW_RCA_LITERATURE.md` (seven precedent papers for the synthetic-data plan; audience is Alireza + Erik).
- **Education / deep-dives**: [`docs/edu/`](../docs/edu/) — `GRPO_STEP_LIFECYCLE.md`, `BATCH_MATH.md` (`gbs == prompts × gen` convention; verl vs ours), `GPU_MEMORY.md`, `RETRIEVER_FROM_SCRATCH.md`, `RNG.md`, `SEED.md`. Read when something below the abstraction line gets confusing.
- [`evaluation_search_r1/`](../evaluation_search_r1/) — eval pipeline (FlashRAG fork + Search-R1 prompts/parsers).
  - [`README.md`](../evaluation_search_r1/README.md) — how to run eval per dataset.
  - [`FROZEN_CONFIG_v1.md`](../docs/milestone_one/FROZEN_CONFIG_v1.md) — **single source of truth for the locked Plan B v1 setup**. Read first.
  - [`REPRODUCIBILITY.md`](../docs/eval/REPRODUCIBILITY.md) — paper targets, the 10 divergences fixed, smoke-validation results, model sha256 verification.
  - [`EVAL_OPS.md`](../docs/eval/EVAL_OPS.md) — three sweep plans (A full / B reduced / C one-seed), wall-clock budgets, where time goes.
  - [`RESULTS_PLAN_B.md`](../docs/milestone_one/RESULTS_PLAN_B.md) — auto-generated final results.
  - [`COMPARISON_PLAN_B_v1.md`](../docs/milestone_one/COMPARISON_PLAN_B_v1.md) — Plan B v1 vs paper, both variants. v0 baseline is at [`COMPARISON_PLAN_B.md`](../docs/milestone_one/COMPARISON_PLAN_B.md).
  - [`archive/`](../docs/archive/) — discarded experiments + historical snapshots ([`archive/README.md`](../docs/archive/README.md) is the index).
  - `flashrag/pipeline/active_pipeline.py` — search↔generate loop.
  - `flashrag/search_r1/parser.py` — `<search>`/`<answer>` parsing, observation truncation.
  - `flashrag/search_r1/templates.py` — base + instruct prompt templates.
  - `flashrag/search_r1/reward.py` — EM/F1 scorer (`em_check`, `normalize_answer`; ground truth — DO NOT modify).
  - `flashrag/config/basic_config.yaml` — retrieval_topk, max_search_turns, sampling params.
  - `run_eval.py` — entrypoint.
- [`local_retriever/`](../local_retriever/) — FAISS retriever HTTP service.
  - [`README.md`](../local_retriever/README.md) — start the retriever; flat vs IVF-SQ8 vs GPU.
  - [`RETRIEVER_INDEXING.md`](../docs/retriever/RETRIEVER_INDEXING.md) — RAM costs, index choices.
- [`/workspace/index_creation/`](../../index_creation/) — build pipeline for the IVF4096-SQ8 quantized index (`wiki18_100w_e5_ivf4096_sq8.index`, ~16 GB, 3–10× faster than flat). Default download path: HF dataset [`pantomiman/reason-over-search`](https://huggingface.co/datasets/pantomiman/reason-over-search/blob/main/retriever/wiki18_100w_e5_ivf4096_sq8.index) (`curl -L .../resolve/main/retriever/wiki18_100w_e5_ivf4096_sq8.index -o local_retriever/indexes/...`). Rebuild only if you don't trust the upload.
- [`scripts/`](../scripts/) — `run_one.sh`, `manage_sglang.sh`, `subsample.sh`, `aggregate.py`, the three sweep scripts.
- [`docker/reason-over-search-v1/`](../docker/) — image used on Vast.ai (`pantomiman/reason-over-search-v1`).
- [`docs/VAST_AI_PLAN_A.md`](../docs/setup/VAST_AI_PLAN_A.md) — Vast.ai GPU cost/throughput analysis for finishing Plan A in ≤24 h. Three concrete fleet configs.
- [`docs/HARDWARE.md`](../docs/setup/HARDWARE.md) — this box's specs + accelerator comparison.
- [`data/`](../data/) — full eval datasets (jsonl, per benchmark).
- [`data_subsample/`](../data_subsample/) — deterministic 1k subsamples for the fast Plan B sweep.
- [`docs/archive/DISCARDED_ABLATIONS.md`](../docs/archive/DISCARDED_ABLATIONS.md) — record of the autoresearch loop's 10 discarded ablations on `experiment_ros/<tag>` (the loop's working files `program.md` and `results.tsv` are no longer tracked).

## Runtime services (everything must be up before eval)

- **Retriever** on `127.0.0.1:3005`. Health: `curl -sS http://127.0.0.1:3005/health` → `"healthy"`. Default is CPU + IVF-SQ8 FAISS (~16 GB RAM, faiss-cpu venv at `/venv/retriever`); the index lives at `local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index` and is downloaded from HF [`pantomiman/reason-over-search`](https://huggingface.co/datasets/pantomiman/reason-over-search). For exact-recall paper-fidelity eval pass `--index ./indexes/wiki18_100w_e5_flat_inner.index` (~65 GB RAM). GPU FAISS is opt-in via `local_retriever/.venv` and **cannot coexist with SGLang on the single 4090** (16 GB VRAM for the index + 22 GB for SGLang).
- **SGLang** on `127.0.0.1:3000` serving the variant under test. Switch with `scripts/manage_sglang.sh switch base|instruct`. Verify: `curl -sS http://127.0.0.1:3000/get_model_info | grep -o instruct`.
- **Eval venv** at `/venv/evaluation_search_r1`. Verify: `/venv/evaluation_search_r1/bin/python -c "import flashrag"`.

## What's been done so far (state as of 2026-05-05)

Canonical state lives in [docs/MILESTONE_1.md](../docs/milestone_one/MILESTONE_1.md) (M1) and [docs/milestone_two/MILESTONE_2.md](../docs/milestone_two/MILESTONE_2.md) (M2). The two snapshots that travel with conversations are [docs/report/CONVERSATION_CONTEXT.md](../docs/report/CONVERSATION_CONTEXT.md) and [docs/internship/CONVERSATION_CONTEXT.md](../docs/internship/CONVERSATION_CONTEXT.md). The frozen reproducer config is [docs/FROZEN_CONFIG_v1.md](../docs/milestone_one/FROZEN_CONFIG_v1.md); single source of truth for sampling, pipeline, retriever, models, datasets, and audit fixes. **Read it before changing anything in M1.**

**Setup**:
- Both GRPO checkpoints sha256-verified (`PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-(it-)em-grpo`).
- Wiki-18 corpus + E5-base-v2 + flat & IVF-SQ8 FAISS indexes built.
- Datasets present for all 7 benchmarks + deterministic 1k subsamples.
- 10 divergences from upstream audited and fixed ([REPRODUCIBILITY.md](../docs/eval/REPRODUCIBILITY.md)).
- Exhaustive paper-vs-ours audit ([PAPER_VS_OURS_AUDIT.md](../docs/eval/PAPER_VS_OURS_AUDIT.md)) catalogued 8 more (D1-D8); the 3 actionable ones are applied (D1 apply_chat=True for base, D-prompt-micro example sentence restored, D8 add_special_tokens block removed).

**Plan B v1 — locked, both variants complete**: see [docs/RESULTS_PLAN_B.md](../docs/milestone_one/RESULTS_PLAN_B.md) and [docs/COMPARISON_PLAN_B_v1.md](../docs/milestone_one/COMPARISON_PLAN_B_v1.md).

| | base v1 avg | base paper | Δ | instruct v1 avg | instruct paper | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Avg EM | 0.292 | 0.312 | −2.0 pp | 0.361 | 0.336 | +2.5 pp |

Format-validity (`</answer>` close-rate): base ≥99.6 % every dataset; instruct 91.4–100 %. Length-truncation ≤0.4 % across the board.

**Plan A — YES (unconditional)**. All decision criteria met. Vast.ai cost analysis: 8× RTX 4090 ≈ $58–77 / 24 h ([docs/VAST_AI_PLAN_A.md](../docs/setup/VAST_AI_PLAN_A.md)).

**Plan B v0** (pre-fix baseline) — archived: [docs/archive/RESULTS_PLAN_B_v0.md](../docs/archive/RESULTS_PLAN_B_v0.md), [docs/COMPARISON_PLAN_B.md](../docs/milestone_one/COMPARISON_PLAN_B.md). Base avg was −8.3 pp vs paper (one-sided → systematic, fixed by D1+D-prompt-micro+D8).

**Phase-1 Qwen3-0.6B training (sibling `research` repo, ALICE)** — frozen, 29 runs total (v0=14, v1=15). Findings logged in [`docs/report/RESULTS_v0.md`](../docs/report/RESULTS_v0.md) and [`docs/report/RESULTS_v1.md`](../docs/report/RESULTS_v1.md):
1. **Hybrid (instruct) learns slowly but stably**; final reward clusters at 0.18 to 0.22.
2. **Base model cannot bootstrap the tool-call format from cold-start** (5/5 v1 base attempts stayed at 0 tool calls, longest 2300 steps). Don't retry without SFT warm-start.
3. **Prompt drives behaviour more than reward**; phrasing alone moved tool-call rate 0 → 2/episode at constant config.
4. **Paper's partial-credit reward creates a 0.1 floor that masks the tool-use signal** (3 to 6 pp gap between tool-using and no-tool runs). The most actionable lever surfaced.
5. **`<tool_call>` in-distribution tags cost nothing** at equal step count vs paper `<search>` tags.
6. **Qwen3-0.6B is too slow for reward-function ablation** under the timeline; pivoted to Qwen3.5-2B on NeMo-RL.

**Phase-2 (M2) NeMo-RL training pipeline** — built and smoke-tested end-to-end on Vast.ai 1× A100 80GB ([`docs/training/SMOKE_RESULTS_2026-05-06.md`](../docs/training/SMOKE_RESULTS_2026-05-06.md); training-side bootstrap doc at [`docs/training/CONVERSATION_CONTEXT.md`](../docs/training/CONVERSATION_CONTEXT.md)): ~57 s/step at smoke shape (20 traj/step); full Search-R1 schedule extrapolates to **11 to 17 days / run, ~$300 to $490 / run** at $1.20/h on 1× A100 (smoke-anchored math in §"Full-training wall-clock + cost" of that file and [`docs/training/PAPER_VS_OURS_TRAINING.md §7`](../docs/training/PAPER_VS_OURS_TRAINING.md#7-compute)). Reward-ablation killed by this. The active recipe and ablation plan now drives Phase-2 work (see "What's next" below). Earlier smoke runs (2026-05-02 small shape; 2026-05-04 v2 seed=42) are archived under [`docs/archive/training/`](../docs/archive/training/).

**Discarded experiments** — index of everything tried that didn't survive: [docs/archive/README.md](../docs/archive/README.md). Notable:
- Temperature ≠ 0 hypothesis was **wrong** — paper eval is greedy. [archive/TEMPERATURE_HYPOTHESIS_WRONG.md](../docs/archive/TEMPERATURE_HYPOTHESIS_WRONG.md).
- 11 autoresearch-loop ablations on `experiment_ros/apr27` — only `temperature=0` kept. [archive/DISCARDED_ABLATIONS.md](../docs/archive/DISCARDED_ABLATIONS.md).
- Bamboogle base v1 EM 0.088 was n=125 variance, not a regression. [archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md](../docs/archive/BAMBOOGLE_REGRESSION_INVESTIGATION.md).

## How to run a single eval

```bash
cd /workspace/reason_over_search
scripts/run_one.sh instruct bamboogle 1 > run.log 2>&1
```

`run_one.sh` is **resume-aware**: if `evaluation_search_r1/results/<dataset>/…_seed1/metric_score.txt` exists it skips and exits 0. Clear stale outputs before re-running the same `(variant, dataset, seed)`:

```bash
rm -rf evaluation_search_r1/results/bamboogle/bamboogle_*_search_r1_instruct_seed1
```

Pull EM/F1:

```bash
LATEST=$(ls -dt evaluation_search_r1/results/bamboogle/bamboogle_*_search_r1_instruct_seed1 | head -1)
grep -E "^(em|f1):" "$LATEST/metric_score.txt"
```

A single Bamboogle/instruct run takes ~6 min on this 4090.

## Hardware

Single RTX 4090 (24 GB), AMD EPYC 7642 (48c/96t), 503 GB RAM. No NVLink, single GPU. Practical implications: 3B in bf16 fits comfortably; FAISS index lives in host RAM (with VRAM as opt-in); GPU FAISS and SGLang cannot share the 4090. Full table in [docs/HARDWARE.md](../docs/setup/HARDWARE.md).

## How I want you to work with me

- **Default to terse.** State results and decisions directly; skip narration.
- **No em-dashes** (`—`, `--`) in any prose written for me. Use semicolons, colons, parentheses, or "X to Y". I read em-dashes as LLM-generated.
- **Read [REPRODUCIBILITY.md](../docs/eval/REPRODUCIBILITY.md) before touching the eval pipeline.** The 10 fixes are load-bearing and were audited against the official repo.
- **Don't modify the EM scorer** (`flashrag/search_r1/answer_utils.py`), the model checkpoints, the FAISS index, or the dataset jsonl files. Those are the ground truth for the experiment.
- **For autoresearch loops** (`experiment_ros/<tag>` branches): the editable surface is `active_pipeline.py`, `parser.py`, `templates.py`, `basic_config.yaml`.
- **Single-run noise is real** (~3 pp at n=125 on Bamboogle when temperature>0). Treat improvements <2 pp EM with skepticism; confirm with a second seed or use temp=0 (greedy).
- **Cite paper sections by URL** when discussing results, especially `arxiv.org/html/2503.09516v5#A6` (Table 3, the GRPO comparison).
- When in doubt about a dataset split or a Search-R1 implementation detail, check the official repo (`github.com/PeterGriffinJin/Search-R1`) before guessing; and ask me before invoking AlphaXiv or other web tools that publish content.

### Gotchas to remember

- **Reward function**: Search-R1's GitHub ships **two** reward modules, `qa_em.py` (paper-faithful EM-only) and `qa_em_format.py` (shaped 6-tier with non-zero defaults that produces visible reward even at EM=0). Earlier docs in this project conflated them. The Phase-2 NeMo-RL port uses **EM-only**. If you see surprising reward curves, check which one is wired in; surfaced in [`docs/training/SMOKE_RESULTS_2026-05-06.md`](../docs/training/SMOKE_RESULTS_2026-05-06.md).
- **`base_breakthrough` (v1, run id `b8vv0qe2`)** in W&B `research_revamp` shows reward 0.7 but is a **reward-function code-change artifact, not learning**. Configs are identical to `base_state_machine_a` which scored 0.0. Treat as instrumented, not earned.
- **FAISS Flat IP times out under training rollout HTTP load**; use the IVF-SQ8 index for training even though paper-fidelity eval uses Flat IP.
- **`verl` does not support Qwen3.5**; that's why M2 ports to NeMo-RL. Verl is still used for Search-R1 reproduction (because Search-R1's published scripts target verl). The transferable knobs from the verl scripts (GRPO params, retriever HTTP contract, rollout shape) are captured in [`docs/training/VERL_REFERENCE.md`](../docs/training/VERL_REFERENCE.md).
- **GPU FAISS + SGLang cannot share a 24 GB 4090**; on this box default to CPU FAISS for the retriever. On A100/H100 it's fine.

## What's next

Two parallel tracks. The canonical M1 list is at [docs/MILESTONE_1.md#whats-left](../docs/milestone_one/MILESTONE_1.md#whats-left); the canonical M2 ablation path is in [`docs/TODO_2026-05-04.md`](../docs/TODO_2026-05-04.md). Active target: **make a Qwen3.5-2B GRPO + search-tool training run stable enough to ablate in ≤10 hours on 1× A100 80GB.**

### Active ablation plan (M2, recipe-search)

In rough priority order; one ≤10 h run on 1× A100 each unless noted.

| # | Run | Goal |
|---|---|---|
| 1 | **Systems wins only** (RUNTIME_EFFICIENCY R1 + R2 + R4 + R5 + O1) | Halve per-step time without changing the algorithm. Smoke-test 50 steps; target ≤30 s/step (vs ~57 s baseline) |
| 2 | **C-minimal** (plain GRPO, fixed hyperparameters, β = 1e-3, no tricks) | The JustRL control. ~500 steps with early-stop on val EM |
| 3 | **+ MC-GRPO** | Median group baseline, G+1 rollouts (exclude pivot). Stability win at small G |
| 4 | **+ S-GRPO** | Loss on 30 to 50 % of tokens per rollout (informativeness-sampled). ~2× backprop saving |
| 5 | **+ E2H curriculum** | Data-side: NQ → HotpotQA → MuSiQue, ~300 steps/stage with fade |

After (2) we know whether JustRL "tricks may hurt" holds in our setting. After (5) we have the supervisor-question answer in [`docs/report/SUPERVISOR_MEETING.md § 2`](../docs/report/SUPERVISOR_MEETING.md).

**Ground rules for ablations**:
- Always run the JustRL control alongside any complex stack; if C-minimal beats the stack on val EM, the stack is hurting.
- One change per run; don't stack two new things and try to attribute later.
- Eval on Bamboogle (125 rows, full) at the end of every run; it's the cheapest meaningful OOD signal.
- Track `reward/mean`, `tool_call_counts/mean`, `had_valid_answer`, `gen_tokens` distribution, gradient norm, clip ratio, per-step seconds in W&B.
- If a run looks dead at step 100 (reward not moving, all-zero advantage groups), kill it. Cheap is better than complete.

### M1 housekeeping (reproduction backlog)

1. **Tabulate format-validity / length-truncation rate** per (dataset, variant) from v1 sweep JSONs. Extend `aggregate.py` to surface `'</answer>' in final_response` close-rate.
2. **One-seed full-data runs** for both base and instruct (~4 h × 2 on a 4090) to confirm v1 config converges at scale, not just on 1 k subsamples.
3. **Plan A on Vast.ai** — Jose's job; instructions in [docs/VAST_AI_PLAN_A.md](../docs/setup/VAST_AI_PLAN_A.md) (5 seeds × 7 × 2 = 70 runs, ≤24 h on a fleet, ~$58–108 budget).
4. **Aggregate, write up, publish**: per-benchmark means + std-dev across the 5 seeds, side-by-side with paper, plus the audit + cost summary.

Do **not** change `temperature` or `top_p`; paper eval is greedy. See note above.

**Milestone 1.1 — untrained Qwen3.5-2B baselines** (Jose-owned): [docs/milestone_one/MILESTONE_1.1_QWEN_BASELINES.md](../docs/milestone_one/MILESTONE_1.1_QWEN_BASELINES.md). Run the M1 eval pipeline against `Qwen/Qwen3.5-2B-Base` and `Qwen/Qwen3.5-2B` using the qwen_native protocol (system prompt with `search` tool registered) so M2's trained checkpoints have a meaningful "untrained" floor to beat. Eval pipeline needs a qwen_native arm (port the parsers from `training/src/environments/parsers.py`).

### M2 status and runbook

Phase-1 (build the pipeline): **complete** as of 2026-05-01. NeMo-RL @ v0.6.0 vendored at `training/nemo_rl/`, training data prepped + LFS-committed, overlay at `training/src/` (dataset adapter, processor, retrieval env, reward, registry; 9 files, 19 unit tests pass), GRPO YAML configs for 1× and 2× A100, parameterized launch scripts. Audit: [docs/training/README.md](../docs/training/README.md). Hyperparameter cross-check: [docs/training/PAPER_VS_OURS_TRAINING.md](../docs/training/PAPER_VS_OURS_TRAINING.md).

Phase-2 (run training): runbook at [docs/milestone_two/PHASE_2_RUNBOOK.md](../docs/milestone_two/PHASE_2_RUNBOOK.md). The original 3 seeds × {base, hybrid} = 6 runs plan is **superseded by the recipe ablation plan above**, since wall-clock makes 6 paired runs unaffordable. To launch:
1. Boot Vast.ai instance; bootstrap via [`docs/setup/BOOTSTRAP_NEW_INSTANCE.md`](../docs/setup/BOOTSTRAP_NEW_INSTANCE.md) (uses `pantomiman/reason-over-search-v1:v1`).
2. Edit [`training/configs/grpo_qwen3.5_2b_1xa100.yaml`](../training/configs/grpo_qwen3.5_2b_1xa100.yaml) (knobs: [`docs/training/NEMO_RL_KNOBS.md`](../docs/training/NEMO_RL_KNOBS.md)).
3. Launch via [`training/scripts/run_grpo_1xa100.sh`](../training/scripts/run_grpo_1xa100.sh) (single GPU) or `run_grpo_2xa100.sh` (decolocated rollout vs train, faster but needs 2 GPUs).
4. Validation in-loop is **not yet enabled**; wire `val_period: 50` per [`docs/training/VALIDATION.md`](../docs/training/VALIDATION.md) before the long ablations.

**After M2** — further ablations on the autoresearch branch and starting on **ReSearch** (the second paper). Append to this section as the plan firms up.
