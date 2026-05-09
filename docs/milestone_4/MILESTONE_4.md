---
title: MILESTONE 4 (and M4.1) — Qwen3.5-0.8B baseline eval pipeline
tags: [milestone, eval, qwen3.5, m4, m4.1]
source: internal
created: 2026-05-08
updated: 2026-05-08
---

# Milestone 4: Eval pipeline + baselines for Qwen3.5-0.8B (untrained)

## Context

M3 closed on 2026-05-07 with a tried-and-tested eval pipeline for the Qwen3-0.6B family ([`evaluation_research/`](../../evaluation_research/), 14 alignment fixes audited in [`report/CODE_SETUP_m3.md`](../report/CODE_SETUP_m3.md)). M4 stands up the equivalent pipeline for the **Qwen3.5-0.8B** family so that any future GRPO checkpoint we train on Qwen3.5 has an "untrained floor" to be compared against. Qwen3.5-0.8B is the first model from the new Qwen3.5 small-model family (0.8B / 2B / 4B / 9B; the M2 NeMo-RL training pipeline targets the 2B; we start at 0.8B for cheap iteration per [`TODO_2026-05-04.md`](../TODO_2026-05-04.md)).

The two snapshots evaluated:

| Snapshot | HF id | Local path on ALICE | Description |
|---|---|---|---|
| `qwen3.5_0.8b` | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | `eval/qwen3.5_0.8b/` | hybrid (instruct + thinking soft-switch) |
| `qwen3.5_0.8b_base` | [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | `eval/qwen3.5_0.8b_base/` | base (pretrained only) |

Both are downloaded once via [`scripts/m4_download_models.sh`](../../scripts/m4_download_models.sh) into the project-root `eval/` directory (1.7 GB each, bf16 safetensors).

## Action format: canonical Qwen3.5 nested-XML tool use (M4.1)

> **Note**: an earlier draft of M4 used a flat `<tool_call>X</tool_call>` form to keep the prose literal-identical to the M3 `p1_basic_w_ex` Qwen3 template. **M4.1 (2026-05-08)** replaces that with Qwen3.5's canonical nested-XML form (the format Qwen3.5 was post-trained on; the flat form was off-distribution). See §M4.1 below for the design and rationale.

Qwen3.5 ships native tool-use tags in its vocab (`<tool_call>`=248058, `</tool_call>`=248059, `<tool_response>`=248066, `</tool_response>`=248067; see [`docs/training/CHAT_TEMPLATE.md`](../training/CHAT_TEMPLATE.md) §2). The canonical post-training format is **nested XML**:

```text
<tool_call>
<function=search>
<parameter=query>
QUERY TEXT
</parameter>
</function>
</tool_call>
```

That format is **auto-injected** by Qwen3.5's chat template when `tools=[QWEN35_SEARCH_TOOL]` is passed to `tokenizer.apply_chat_template`. The template emits a `# Tools` block (function signature) + a verbatim format example + an `<IMPORTANT>` reminder, all before our system content. So our system prompt only needs the role intro + 3 brief process steps; the format spec is free.

Final M4.1 system message (in [`evaluation_qwen35/flashrag/search_r1/templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py) as `QWEN35_NATIVE_TEMPLATE`, registered as `QWEN35_TEMPLATES["qwen35"]` and `QWEN35_TEMPLATES["qwen35_native"]`):

```text
You are a helpful assistant. Answer the user's question by using the `search` tool when you need external knowledge; you may call it multiple times. When you have enough information, give the final answer inside <answer> and </answer>. For example, <answer>Beijing</answer>.
```

User message:

```text
Question: {question}
```

Tool-response wrap (env-side, mirrors training-side `format_docs_qwen_native`):

```text
<|im_end|>
<|im_start|>user
<tool_response>
{retrieved docs}
</tool_response><|im_end|>
<|im_start|>assistant
```

Final-answer format: plain `<answer>X</answer>` (the M4-placeholder `\boxed{}` wrapper was inherited from M3 by accident; the EM scorer normalizes either form, but the plain form is shorter and matches what training will produce).

## Pipeline layout — `evaluation_qwen35/`

Full copy of [`evaluation_research/`](../../evaluation_research/) (which itself is a copy of [`evaluation_search_r1/`](../../evaluation_search_r1/)). The M4 surface area is six files:

| File | What changed vs M3 |
|---|---|
| [`flashrag/search_r1/templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py) | + `QWEN35_NATIVE_TEMPLATE` (3-line role + protocol), + `QWEN35_SEARCH_TOOL` (OpenAI-style schema, mirror of training-side `chat_template/tools.py:SEARCH_TOOL`), + `QWEN35_TEMPLATES` registry |
| [`flashrag/search_r1/parser.py`](../../evaluation_qwen35/flashrag/search_r1/parser.py) | + `extract_tool_call_query` (canonical Qwen3.5 nested-XML `<tool_call><function=search><parameter=query>X</parameter></function></tool_call>` parser; mirror of training-side `parsers.py:_RE_QWEN_QUERY`) |
| [`flashrag/pipeline/active_pipeline.py`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py) | family-based dispatch (qwen3 / qwen35 / search_r1); `qwen35` branch passes `tools=[QWEN35_SEARCH_TOOL]` to `apply_chat_template`, builds `Question: {q}` user message, uses turn-bounded tool-response wrapping |
| [`flashrag/utils/utils.py`](../../evaluation_qwen35/flashrag/utils/utils.py) | `get_generator` short-circuits on `framework=sgl_remote` before the multimodal-detection check (Qwen3.5 has `vision_config` in config.json so the existing check would mis-route to `HFMultiModalGenerator`) |
| [`run_eval.py`](../../evaluation_qwen35/run_eval.py) | + `qwen35` / `qwen35_native` prompt_modes; `--test_sample_num` / `--random_sample` / `--seed` for the 100-item quick eval; deterministic `random.seed(config[seed])` before subsampling |
| [`setup.py`](../../evaluation_qwen35/setup.py) | distribution name `evaluation-qwen35` (avoids collision when `evaluation_research` is editable-installed in the same conda env) |

Tag dispatch table (in `active_pipeline.SearchR1Pipeline.run_item`):

| Mode family | Prompt template | Tools schema | Action stop | Parser | Result wrapper |
|---|---|---|---|---|---|
| `qwen3*` (M3) | `QWEN3_TEMPLATES[mode]` (system role) + bare question (user) | (none) | `</search>` | `extract_search_tag_query` | `" <result>\n{X}\n</result>"` |
| `qwen35*` (M4.1) | `QWEN35_NATIVE_TEMPLATE` (system role) + `Question: {q}` (user) | `tools=[QWEN35_SEARCH_TOOL]` (auto-injects nested-XML format spec) | `</tool_call>` | `extract_tool_call_query` (nested-XML) | `<\|im_end\|>\n<\|im_start\|>user\n<tool_response>\n{X}\n</tool_response><\|im_end\|>\n<\|im_start\|>assistant\n` |
| `search_r1` (M1) | `SEARCH_R1_TEMPLATE.format(prompt=…)` (user-only) | (none) | `</search>` | `extract_search_tag_query` | `"\n\n<information>{X}</information>\n\n"` |

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

## M4.1 — prompt redesign (canonical Qwen3.5 tool use)

**Status (2026-05-08): code change applied; pending smoke validation.**

### Why we revisited the M4 prompt

The M4 placeholder template was a literal port of the M3 `p1_basic_w_ex` prose with `<search>` ↔ `<tool_call>` and `<result>` ↔ `<tool_response>` substitution, plus the M3-inherited `\boxed{}` answer wrapper. Two problems with that:

1. **Off-distribution tool format.** Qwen3.5 was post-trained on the **nested-XML** form (CHAT_TEMPLATE.md §2). The flat `<tool_call>X</tool_call>` form we'd written is not in the model's post-training distribution; using it asks the base model to ignore its strongest tool-use prior.
2. **Train/eval mismatch (forward-looking).** The training-side qwen_native arm (CHAT_TEMPLATE.md §1a) already uses the nested-XML form via `tools=[SEARCH_TOOL]`. When M5+ produces a GRPO-trained Qwen3.5 checkpoint, evaluating it with a flat-form prompt would have been a known-divergence mismatch we'd need to undo anyway.

### What changed

| Concern | M4 placeholder | M4.1 (canonical) |
|---|---|---|
| Tool-call format | flat `<tool_call>X</tool_call>` (off-distribution) | nested-XML (in Qwen3.5 post-training distribution) |
| Tools schema | not registered | `tools=[QWEN35_SEARCH_TOOL]` passed to `apply_chat_template`; chat template auto-injects the `# Tools` block + format example + `<IMPORTANT>` reminder |
| System message | 14-line prose + 5-step Hamlet example | 3-line role + 3-step protocol; no tool-call example needed (chat template emits one) |
| User message | bare question | `Question: {question}` |
| Final-answer wrap | `<answer>The final answer is \[ \boxed{X} \]</answer>` | `<answer>X</answer>` |
| Tool-response wrap | `" <tool_response>\n{X}\n</tool_response>"` (continuation, leading space) | `<\|im_end\|>\n<\|im_start\|>user\n<tool_response>\n{X}\n</tool_response><\|im_end\|>\n<\|im_start\|>assistant\n` (turn-bounded; mirrors training-side `format_docs_qwen_native`) |
| Corrective on invalid action | "...put the query between `<tool_call>` and `</tool_call>`..." | "...call the `search` tool..." (chat template's `# Tools` block carries the spec, no need to dictate format here) |

### Why mirror the training-side qwen_native arm rather than introduce something new

The training-side already has a tested `qwen_native` arm (CHAT_TEMPLATE.md §1a, §7a) that registers `tools=[SEARCH_TOOL]`, uses turn-bounded `<tool_response>` blocks, and ends with `<answer>X</answer>`. Aligning M4.1 to that arm gives:

- Byte-identical tool-call / tool-response shapes between training and eval.
- A plausible "untrained floor" measurement: the same prompt that the trained checkpoint will see, evaluated on the untrained checkpoint, with the model's behaviour driven by its post-training prior alone.

The only intentional divergence from the training arm: M4.1 puts the brief protocol in the **system** message and uses `Question: {q}` as the **user** message; the training arm puts the protocol in the user message. That divergence is deliberate (the user prompt should be minimal, just the question; the protocol guidance is system-level scaffolding for an untrained model). Future training arms can match this layout if cross-comparability matters, but the trade-off is on the training side, not eval.

### Verification before smoke

Before launching the smoke, sanity-check the rendered prompt on a sample question (Bamboogle's first item) on ALICE:

```bash
ssh alice 'cd /zfsstore/user/s4374886/omega/reason_over_search-m4 && \
  /home/s4374886/.conda/envs/evaluation_search_r1/bin/python -c "
from transformers import AutoTokenizer
from evaluation_qwen35.flashrag.search_r1.templates import QWEN35_NATIVE_TEMPLATE, QWEN35_SEARCH_TOOL
tok = AutoTokenizer.from_pretrained(\"eval/qwen3.5_0.8b\")
print(tok.apply_chat_template(
    [{\"role\":\"system\",\"content\":QWEN35_NATIVE_TEMPLATE},
     {\"role\":\"user\",\"content\":\"Question: Who directed Inception?\"}],
    tools=[QWEN35_SEARCH_TOOL], tokenize=False,
    add_generation_prompt=True, enable_thinking=True))
"'
```

Expect: a `<|im_start|>system` block containing the auto-injected `# Tools` + format example + `<IMPORTANT>` reminder + our 3-line role intro; then `<|im_start|>user\nQuestion: Who directed Inception?<|im_end|>`; then `<|im_start|>assistant\n<think>\n` (hybrid generation prefix).

## What's left

| # | Task |
|---|---|
| 1 | Quick smoke (100 / dataset) for both variants on the M4.1 prompt — validate the pipeline end-to-end |
| 2 | Full sweep (51,713 items / variant) — produce `RESULTS_m4.md` with the M4.1 baseline numbers |
| 3 | Cross-family comparison: M3 (Qwen3-0.6B hybrid) vs M4 (Qwen3.5-0.8B hybrid + base) on the same 7 datasets — establishes the "untrained floor" for any future M5+ GRPO trained Qwen3.5 checkpoint |
| 4 | M5 (was M4.1-placeholder; renamed): pick + run the same eval against the first GRPO-trained Qwen3.5-0.8B checkpoint once the recipe-search Phase-2 work produces one |

## Pointers

- M3 narrative: [`../milestone_3/MILESTONE_3.md`](../milestone_3/MILESTONE_3.md), [`../milestone_3/MILESTONE_3.1.md`](../milestone_3/MILESTONE_3.1.md)
- M3 alignment audit (the 14 fixes M4 inherits unchanged): [`../report/CODE_SETUP_m3.md`](../report/CODE_SETUP_m3.md) §3
- M3 results table: [`../report/RESULTS_m3.md`](../report/RESULTS_m3.md)
- Active recipe-ablation plan (drives M5+ training on Qwen3.5): [`../TODO_2026-05-04.md`](../TODO_2026-05-04.md)
- Phase-2 NeMo-RL training pipeline (smoke-tested on 1× A100 for Qwen3.5-2B): [`../training/SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md)
- Qwen3.5 chat template (verbatim): [`../training/CHAT_TEMPLATE.md`](../training/CHAT_TEMPLATE.md) §2
