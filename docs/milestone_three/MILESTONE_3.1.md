---
title: MILESTONE 3.1
tags: []
source: internal
created: 2026-05-08
updated: 2026-05-08
---

# Milestone 3.1: Evaluate the v0-best (no example) GRPO checkpoint

## Context

[MILESTONE_3](MILESTONE_3.md) closed 2026-05-07 by evaluating `p1_basic_w_ex_z7kcxfof` — the *with-example* run that converged on **heavy-tool 2-call / 4-turn** behaviour over 1046 steps, end-of-run rollout reward 0.190 (+28 % rel over 0.148 first-decile). That gave avg EM 0.102 → 0.155 on the 7 paper benchmarks (+0.053 abs, +52 % rel; full Plan A 51,713 items / variant).

But that run **was not the highest-reward run of the v0 block**. Per [`docs/report/RESULTS_v0.md`](../report/RESULTS_v0.md) §11, the highest-reward Phase-1 run was `p3_decide_no_ex_el6s2d2h`:

| Run | Prompt | Steps | First-decile | Last-decile | Δ rel | Behaviour |
|---|---|---:|---:|---:|---:|---|
| `z7kcxfof` (M3) | `p1_basic_w_ex` (with example) | 1046 | 0.148 | **0.190** | +28 % | heavy-tool 2/4, ~2050 tok |
| `el6s2d2h` (this milestone) | `p3_decide_no_ex` (no example, decision rules) | 2280 | 0.151 | **0.215** | **+43 %** | standard 1/3, ~1117 tok |

The *qualitative* finding from the Phase-1 ablations was that **decision-rule scaffolding can substitute for the few-shot example** (the only v0 prompt where removing the example was harmless). The *quantitative* question M3.1 answers: does that higher rollout reward translate to higher held-out EM, and if so, by how much vs. M3?

## Hypothesis

If the rollout-reward gap (0.215 vs 0.190) is meaningful, el6s2d2h should beat z7kcxfof on at least the single-hop datasets (where both checkpoints have headroom and the lift in M3 was concentrated). If the rollout reward gap is mostly the partial-credit floor (cf. `RESULTS_v0.md` finding 4), the gap may close at eval time. Either result is informative.

## Goal

Run the M3 evaluation pipeline against `eval/qwen_3_0.6b_v0_no_ex/` (HF-converted from `docs/archive/verl_runs/v0/p3_decide_no_ex_el6s2d2h/global_step_2000/actor`) on all 7 paper QA benchmarks at full Plan A; compare against the existing M3 numbers (pre-GRPO + z7kcxfof) populated in [`RESULTS_v2.md`](../report/RESULTS_v2.md).

## Setup

Reuses the M3 pipeline byte-for-byte except for the **prompt template**: `p3_decide_no_ex` (no Hamlet example; two extra decision-rule sentences). The rest of the alignment fixes ([`CODE_SETUP_v2.md`](../report/CODE_SETUP_v2.md) §3 14-fix audit) carry over unchanged.

| Setting | M3 (z7kcxfof) | M3.1 (el6s2d2h) |
|---|---|---|
| Checkpoint | `eval/qwen_3_0.6b_v0/` (1046 steps) | `eval/qwen_3_0.6b_v0_no_ex/` (2000 steps; converted via `verl.model_merger merge --backend fsdp`) |
| Prompt template | `QWEN3_0_6B_TEMPLATE` (= `p1_basic_w_ex`) | **`P3_DECIDE_NO_EX_TEMPLATE`** (new; `evaluation_research/flashrag/search_r1/templates.py`) |
| `prompt_mode` | `qwen3` | **`qwen3_p3_decide_no_ex`** (new; falls in the `qwen3*` family — same retrieval format, same budgets, same `enable_thinking`) |
| Action format | `<search>` / `<result>` | same |
| Retrieval text | raw `{contents}\n\n` joined+stripped | same |
| `top_n` | 5 | 5 |
| `max_search_turns` | 5 | 5 |
| `step_limit` | 8192 (no per-step cap) | same |
| `max_obs_length` | 256 | same |
| `generator_max_input_len` | 4096 | 4096 |
| `enable_thinking` | True | True |
| Datasets | bamboogle, NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue (full Plan A, 51,713 items) | same |
| Hardware | ALICE 1× A100-80GB | same |
| Decoding | greedy (temperature=0.0) | same |
| Seed | 1 | 1 |

The new prompt template (verbatim from training, recovered from [`RESULTS_v0.md`](../report/RESULTS_v0.md) §10 `p3_decide_no_ex`):

```text
You are a helpful assistant who can answer questions using a Wikipedia search tool.
You can call the search tool by writing: <search> your query </search>
You will receive the result in: <result> your search result </result>
Use the search tool to obtain the information needed for the answer.
Use the information in the search results to determine the final answer.
After each search result, decide whether another search is needed or whether you can provide the final answer.
If a search result is incomplete, search again for the missing information.
You may use the search tool multiple times if needed before giving the final answer.
Provide the final answer in the format: <answer>The final answer is \[ \boxed{answer here} \]</answer>.
```

Difference vs `p1_basic_w_ex`: rules section has the two extra decision-guidance sentences ("After each search result, decide whether another search is needed…" / "If a search result is incomplete, search again…"); **no Hamlet example**.

## Pipeline changes (committed)

- [`evaluation_research/flashrag/search_r1/templates.py`](../../evaluation_research/flashrag/search_r1/templates.py): added `P3_DECIDE_NO_EX_TEMPLATE` and a `QWEN3_TEMPLATES` dict keyed on `prompt_mode`.
- [`evaluation_research/flashrag/pipeline/active_pipeline.py`](../../evaluation_research/flashrag/pipeline/active_pipeline.py): replaced `prompt_mode == 'qwen3'` checks with `prompt_mode.startswith('qwen3')` (so all qwen3 modes share retrieval format / budgets / `enable_thinking`); template selection now lookup-driven.
- [`evaluation_research/run_eval.py`](../../evaluation_research/run_eval.py): same `startswith('qwen3')` change for `retrieval_topk=5` gate; help text updated.
- [`scripts/run_m3.sh`](../../scripts/run_m3.sh): added `qwen3_0.6b_v0_no_ex` variant case → `prompt_mode=qwen3_p3_decide_no_ex`.
- [`scripts/sbatch_m3.sh`](../../scripts/sbatch_m3.sh): added `qwen3_0.6b_v0_no_ex` variant case (same SGLang flags as the other variants).

## How to run

```bash
# On ALICE login node:
cd /zfsstore/user/s4374886/omega/reason_over_search
sbatch scripts/sbatch_m3.sh qwen3_0.6b_v0_no_ex
```

Same `gpu-short` partition, 4 h time limit, IVF-SQ8 × 8 retriever workers, 32 inference workers as M3. Expected wall-clock: ~2.5 h per variant (matches M3's 2 h 26 m for the v0 sbatch).

## Status

**2026-05-08 (queued)**: sbatch job **2134645** submitted, gpu-short partition, 4 h limit, pending Priority. Will start retriever + SGLang on `eval/qwen_3_0.6b_v0_no_ex/`, run all 7 datasets, then cleanup. Logs at `logs/m3_2134645_*.{out,err,log}`. Per-dataset results land at `evaluation_research/results/<dataset>/<dataset>_*_m3_qwen3_0.6b_v0_no_ex_seed1/`.

Numerical results will be populated in [`docs/report/RESULTS_v2.md`](../report/RESULTS_v2.md) §M3.1 once the job completes.

## Deliverables

1. [x] Convert `el6s2d2h` global_step_2000 actor → HF (1.5 GB safetensors at `eval/qwen_3_0.6b_v0_no_ex/`).
2. [x] Add `P3_DECIDE_NO_EX_TEMPLATE` and `QWEN3_TEMPLATES` registry to `evaluation_research/flashrag/search_r1/templates.py`.
3. [x] Wire `prompt_mode='qwen3_p3_decide_no_ex'` through `active_pipeline.py` and `run_eval.py`; switch `qwen3` checks to family-prefix matching.
4. [x] Add `qwen3_0.6b_v0_no_ex` variant case to `scripts/run_m3.sh` and `scripts/sbatch_m3.sh`.
5. [x] Submit sbatch (job 2134645).
6. [ ] Job completes successfully (all 7 datasets).
7. [ ] Per-dataset EM / ACC / F1 added to [`docs/report/RESULTS_v2.md`](../report/RESULTS_v2.md) §M3.1.
8. [ ] Side-by-side M3 vs M3.1 comparison table (pre-GRPO / z7kcxfof / el6s2d2h) added to RESULTS_v2 + supervisor brief.
9. [ ] Training-plot comparison panel (with-example z7kcxfof vs no-example el6s2d2h: reward, tool calls, num_turns, response_length curves) — already-trained data, only the visualisation is new. Tracked in [`docs/report/SUPERVISOR_MEETING_2026-05-07.md`](../report/SUPERVISOR_MEETING_2026-05-07.md) §4.8 plan.
