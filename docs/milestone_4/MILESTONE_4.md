---
title: MILESTONE 4 — Qwen3.5-0.8B baseline eval pipeline
tags: [milestone, eval, qwen3.5, m4]
source: internal
created: 2026-05-08
updated: 2026-05-08
---

# Milestone 4: Eval pipeline + baselines for Qwen3.5-0.8B (untrained)

## Context

M3 closed on 2026-05-07 with a tried-and-tested eval pipeline for the Qwen3-0.6B family ([`evaluation_research/`](../../evaluation_research/), 14 alignment fixes audited in [`report/CODE_SETUP_v2.md`](../report/CODE_SETUP_v2.md)). M4 stands up the equivalent pipeline for the **Qwen3.5-0.8B** family so that any future GRPO checkpoint we train on Qwen3.5 has an "untrained floor" to be compared against. Qwen3.5-0.8B is the first model from the new Qwen3.5 small-model family (0.8B / 2B / 4B / 9B; the M2 NeMo-RL training pipeline targets the 2B; we start at 0.8B for cheap iteration per [`TODO_2026-05-04.md`](../TODO_2026-05-04.md)).

The two snapshots evaluated:

| Snapshot | HF id | Local path on ALICE | Description |
|---|---|---|---|
| `qwen3.5_0.8b` | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | `eval/qwen3.5_0.8b/` | hybrid (instruct + thinking soft-switch) |
| `qwen3.5_0.8b_base` | [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | `eval/qwen3.5_0.8b_base/` | base (pretrained only) |

Both are downloaded once via [`scripts/m4_download_models.sh`](../../scripts/m4_download_models.sh) into the project-root `eval/` directory (1.7 GB each, bf16 safetensors).

## Action format: `<tool_call>` / `<tool_response>` (Qwen3.5-native)

The Qwen3-0.6B M3 prompt taught the model to use the Search-R1 invented `<search>` / `<result>` tags. Qwen3.5 already ships those native tool-use tags in its vocab (`<tool_call>`=248058, `</tool_call>`=248059, `<tool_response>`=248066, `</tool_response>`=248067; see [`docs/training/CHAT_TEMPLATE.md`](../training/CHAT_TEMPLATE.md) §2), so the M4 prompt swaps to those — keeping the rest of the prose identical to the M3 `p1_basic_w_ex` template, only `<search>` ↔ `<tool_call>` and `<result>` ↔ `<tool_response>`. Phase-1 v1 finding 5 ([`RESULTS_v1.md`](../report/RESULTS_v1.md)) showed in-distribution `<tool_call>` tags cost nothing at equal step count vs paper `<search>` tags; M4 puts the eval side of that on the same footing.

Final M4 system message (in [`evaluation_qwen35/flashrag/search_r1/templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py) as `QWEN35_0_8B_TEMPLATE`):

```text
You are a helpful assistant who can answer questions using multiple Wikipedia search tool calls.
You can call the search tool by writing: <tool_call> your query </tool_call>
You will receive the result in: <tool_response> your search result </tool_response>
Use the search tool to obtain the information needed for the answer.
Answers should be based on the search results.
You may use the search tool multiple times if needed before giving the final answer.
Provide the final answer in the format: <answer>The final answer is \[ \boxed{answer here} \]</answer>.
For example:
Question: What is the nationality of the author of Hamlet?
<tool_call>Hamlet</tool_call>
<tool_response>The Tragedy of Hamlet was written by William Shakespeare.</tool_response>
<tool_call>William Shakespeare</tool_call>
<tool_response>William Shakespeare was an English playwright.</tool_response>
<answer>The final answer is \[ \boxed{English} \]</answer>
```

The flat `<tool_call>X</tool_call>` form is a deliberate choice over Qwen3.5's nested XML form (`<tool_call><function=search><parameter=query>X</parameter></function></tool_call>`; CHAT_TEMPLATE.md §2). The flat form keeps the prompt literal-identical to the M3 prose (one-line search query, one-line tool response) and avoids the auto-injected tools schema preamble that would otherwise prepend ~600 tokens of tool description on every prompt.

## Pipeline layout — `evaluation_qwen35/`

Full copy of [`evaluation_research/`](../../evaluation_research/) (which itself is a copy of [`evaluation_search_r1/`](../../evaluation_search_r1/)). The M4 surface area is six files:

| File | What changed vs M3 |
|---|---|
| [`flashrag/search_r1/templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py) | + `QWEN35_0_8B_TEMPLATE` and `QWEN35_TEMPLATES` registry |
| [`flashrag/search_r1/parser.py`](../../evaluation_qwen35/flashrag/search_r1/parser.py) | + `extract_tool_call_query` (flat `<tool_call>X</tool_call>` parser) |
| [`flashrag/pipeline/active_pipeline.py`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py) | family-based dispatch (qwen3 / qwen35 / search_r1) for action stop tokens, parser, and result-wrapper |
| [`flashrag/utils/utils.py`](../../evaluation_qwen35/flashrag/utils/utils.py) | `get_generator` short-circuits on `framework=sgl_remote` before the multimodal-detection check (Qwen3.5 has `vision_config` in config.json so the existing check would mis-route to `HFMultiModalGenerator`) |
| [`run_eval.py`](../../evaluation_qwen35/run_eval.py) | + `qwen35` / `qwen35_p1_basic_w_ex` prompt_modes; `--test_sample_num` / `--random_sample` / `--seed` for the 100-item quick eval; deterministic `random.seed(config[seed])` before subsampling |
| [`setup.py`](../../evaluation_qwen35/setup.py) | distribution name `evaluation-qwen35` (avoids collision when `evaluation_research` is editable-installed in the same conda env) |

Tag dispatch table (in `active_pipeline.SearchR1Pipeline.run_item`):

| Mode family | Prompt template | Action stop | Parser | Result wrapper |
|---|---|---|---|---|
| `qwen3*` (M3) | `QWEN3_TEMPLATES[mode]` | `</search>` | `extract_search_tag_query` | `" <result>\n{X}\n</result>"` |
| `qwen35*` (M4) | `QWEN35_TEMPLATES[mode]` | `</tool_call>` | `extract_tool_call_query` | `" <tool_response>\n{X}\n</tool_response>"` |
| `search_r1` (M1) | `SEARCH_R1_TEMPLATE.format(prompt=…)` | `</search>` | `extract_search_tag_query` | `"\n\n<information>{X}</information>\n\n"` |

Per-mode budgets (qwen3 + qwen35 share the M3 shape; search_r1 stays at paper):

| Knob | qwen3* / qwen35* | search_r1 |
|---|---|---|
| `max_search_turns` | 5 | 4 |
| `step_limit` | 8192 (no per-step cap; bounded by `remain_length`) | 500 |
| `max_obs_length` | 256 tokens | 500 tokens |
| `retrieval_topk` | 5 | 3 |

## Variant dispatch

| Variant | Path | `enable_thinking` | `prompt_mode` |
|---|---|---|---|
| `qwen3.5_0.8b` (hybrid) | `eval/qwen3.5_0.8b/` | True | `qwen35` |
| `qwen3.5_0.8b_base` | `eval/qwen3.5_0.8b_base/` | False | `qwen35` |

Hybrid runs with `enable_thinking=True` so the chat template emits `<think>\n` (open block, model fills it). Base runs with `enable_thinking=False` so the chat template emits `<think>\n\n</think>\n\n` (closed empty block) — base wasn't post-trained on the hybrid soft-switch protocol; emitting an open `<think>` would derail it.

## Goal

1. **Quick eval**: 100 random items / dataset (deterministic via `seed=1`), 7 datasets × 2 variants = 14 sub-runs, ≤ 30 min wall on 1× A100-80GB. Used to validate the pipeline mechanically and surface tag-format issues on the first pass.
2. **Full sweep**: full Plan A test/dev sets (51,713 items / variant), 7 datasets × 2 variants = 14 sub-runs, expected ~150 min / variant on 1× A100-80GB (M3 reference: 146 min for v0).

Both runs use greedy decode (`temperature=0.0`), single seed (greedy => seed-invariant past `random_sample`), bf16 SGLang inference, IVF-SQ8 retriever × 8 workers.

## Run

Quick eval (smoke):

```bash
sbatch --time=01:00:00 scripts/sbatch_m4.sh qwen3.5_0.8b 100
sbatch --time=01:00:00 scripts/sbatch_m4.sh qwen3.5_0.8b_base 100
```

Full sweep:

```bash
sbatch scripts/sbatch_m4.sh qwen3.5_0.8b
sbatch scripts/sbatch_m4.sh qwen3.5_0.8b_base
```

Single-(variant, dataset, seed) launch via [`scripts/run_m4.sh`](../../scripts/run_m4.sh). Smoke results land in `evaluation_qwen35/results/<dataset>/<dataset>_*_m4_<variant>_seed1_n100/metric_score.txt`; full results drop the `_n100` suffix.

## What's left

| # | Task |
|---|---|
| 1 | Quick smoke (100 / dataset) for both variants — validate the pipeline end-to-end |
| 2 | Full sweep (51,713 items / variant) — produce `RESULTS_v3.md` with the M4 baseline numbers |
| 3 | Cross-family comparison: M3 (Qwen3-0.6B hybrid) vs M4 (Qwen3.5-0.8B hybrid + base) on the same 7 datasets — establishes the "untrained floor" for any future M5+ GRPO trained Qwen3.5 checkpoint |
| 4 | M4.1 (placeholder): pick + run the same eval against the first GRPO-trained Qwen3.5-0.8B checkpoint once the recipe-search Phase-2 work produces one |

## Pointers

- M3 narrative: [`../milestone_three/MILESTONE_3.md`](../milestone_three/MILESTONE_3.md), [`../milestone_three/MILESTONE_3.1.md`](../milestone_three/MILESTONE_3.1.md)
- M3 alignment audit (the 14 fixes M4 inherits unchanged): [`../report/CODE_SETUP_v2.md`](../report/CODE_SETUP_v2.md) §3
- M3 results table: [`../report/RESULTS_v2.md`](../report/RESULTS_v2.md)
- Active recipe-ablation plan (drives M5+ training on Qwen3.5): [`../TODO_2026-05-04.md`](../TODO_2026-05-04.md)
- Phase-2 NeMo-RL training pipeline (smoke-tested on 1× A100 for Qwen3.5-2B): [`../training/SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md)
- Qwen3.5 chat template (verbatim): [`../training/CHAT_TEMPLATE.md`](../training/CHAT_TEMPLATE.md) §2
