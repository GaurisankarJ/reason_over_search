---
title: MILESTONE 4 (M4.1 / M4.2 / M4.3 / M4.4) — Qwen3.5-0.8B baseline eval pipeline
tags: [milestone, eval, qwen3.5, m4, m4.1, m4.2, m4.3, m4.4]
source: internal
created: 2026-05-08
updated: 2026-05-10
---

# Milestone 4: Eval pipeline + baselines for Qwen3.5-0.8B (untrained)

## Context

M3 closed on 2026-05-07 with a tried-and-tested eval pipeline for the Qwen3-0.6B family ([`evaluation_research/`](../../evaluation_research/), 14 alignment fixes audited in [`report/CODE_SETUP_m3.md`](../report/CODE_SETUP_m3.md)). M4 stands up the equivalent pipeline for the **Qwen3.5-0.8B** family so that any future GRPO checkpoint we train on Qwen3.5 has an "untrained floor" to be compared against. Qwen3.5-0.8B is the first model from the new Qwen3.5 small-model family (0.8B / 2B / 4B / 9B; the M2 NeMo-RL training pipeline targets the 2B; we start at 0.8B for cheap iteration per [`TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md)).

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
You are a helpful assistant with access to a `search` tool that retrieves Wikipedia passages.

The user will give you a question in the form: Question: <question>

Steps:
- Call `search` with a focused query in the format described above.
- The search result will appear as <tool_response>...</tool_response>.
- If you have enough information, write the final answer inside <answer> and </answer> and stop.
- Otherwise, refine your query and call `search` again.
```

Loop semantics only (search → observe → reason → answer-or-loop). The format spec is auto-injected by the chat template (the `# Tools` block + the `<IMPORTANT>` reminder, both verbatim from the model's [`tokenizer_config.json:chat_template`](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/tokenizer_config.json)) — we don't repeat it. **No few-shot example**: we want to see whether Qwen3.5-0.8B's post-training prior alone is sufficient to drive the loop on this size.

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

| Variant | Path | `enable_thinking` | `prompt_mode` (default) |
|---|---|---|---|
| `qwen3.5_0.8b` (hybrid) | `eval/qwen3.5_0.8b/` | True | `qwen35_minimal` (M4.2; auto-inject in system) |
| `qwen3.5_0.8b_base` | `eval/qwen3.5_0.8b_base/` | True | `qwen35_minimal_no_system` (M4.3; no system block) |

**Both variants run with `enable_thinking=True`** so the chat template emits an open `<think>\n` generation prefix and the model reasons before each tool call. This is mildly off-distribution for the base variant (which wasn't post-trained on the hybrid soft-switch protocol), but giving the base model space to reason before each tool call is worth more than the small cost of seeing an open think block; using the same render shape across variants also makes the hybrid-vs-base comparison directly comparable. (Earlier draft had base on `enable_thinking=False`; flipped 2026-05-09.)

**M4.2 / M4.3 (2026-05-09)** lock asymmetric per-variant defaults after smoke iteration. Hybrid does best with the auto-injected `# Tools` + `<IMPORTANT>` block in the system role (mean EM 0.057, n=100/dataset, 6.6× over v3); base does best WITHOUT it (mean EM 0.016 vs 0.003, 5×). Mechanism: the auto-inject's `<IMPORTANT>` reminder drives search loops on hybrid (in-distribution for tool-use post-training); on base it's pure scaffolding noise that crowds out the answer. Full smoke iteration log + cross-comparison with M3 baseline at [`../report/RESULTS_SMOKE_m4.md`](../report/RESULTS_SMOKE_m4.md) (§6 hybrid M4.2, §7 M4.3 + asymmetric lock-in).

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

## M4.4 — prompt search to close the M3 cross-family gap

**Status (2026-05-10): Phase 1 partially executed — first candidate (`qwen35_recall_port`, ReCall-prose port, system-role + reasoning framing) ran and FAILED the acceptance bar (mean EM 0.0510 vs A=0.0594, Δ −0.0084). Four candidates (B / C / D / E) still pending. M4.2 baseline + M3-vs-M4 cross-family comparison are closed (see [`RESULTS_m4.md`](../report/RESULTS_m4.md)); M4.4 is an additive prompt-search sub-phase that may revise the locked prompt before M5 commits training compute, but does NOT block M5 scaffold work.**

### Why we revisit the M4.2 lock

M4.2 hybrid `qwen35_minimal` lands at mean EM **0.0594** (n=1000, [`RESULTS_SMOKE_m4.md` §9](../report/RESULTS_SMOKE_m4.md)) / **0.0571** (n=100, §6) / **~0.060** (full Plan A, §10.2 partial). The M3 untrained Qwen3-0.6B hybrid — a *smaller* model from a previous family — reaches **0.102** on the same 7 benchmarks at full Plan A. A 1.7× cross-family gap in favour of the smaller / older model implicates the prompt, not the parameters: Qwen3.5 was post-trained on different tool-use shapes than Qwen3, and we have not yet tested the prompt shapes that the literature (Search-R1, ReSearch / ReCall) and our own M3 / M3.1 ablations identified as load-bearing.

What we have NOT yet tested on Qwen3.5-0.8B:

1. **Search-R1 verbatim `<search>` / `<information>` tag scheme.** M4.2 ported the *prose* but swapped tags to `<tool_call>` / `<tool_response>` to leverage Qwen3.5's `tools=[]` auto-injection. The M3 finding "tag schema is interchangeable" ([`MILESTONE_3.1.md` §findings](../milestone_3/MILESTONE_3.1.md)) suggests the prose matters more than the literal tag tokens. Search-R1's exact paper template drove Qwen2.5-3B to 0.292 EM in our M1 reproduction — never tried directly on Qwen3.5.
2. **M3.1's winning shape** (system-role + decision-rule scaffolding + no example). [`P3_DECIDE_NO_EX_TEMPLATE`](../../evaluation_qwen35/flashrag/search_r1/templates.py#L32-L42) is the v0-block training-time winner that delivered EM 0.169 in M3.1 — a tested shape for the Qwen3 family that we have not ported across.
3. **`enable_thinking=False`.** [`RESULTS_SMOKE_m4.md` §6.5](../report/RESULTS_SMOKE_m4.md) flagged that the model emits `<think>` 100 % of the time and entity hallucinations inside it anchor downstream queries. Closing `<think>\n\n</think>` removes that failure path.
4. **Additive decision rules on top of M4.2.** Smallest delta — keeps the M4.2 user-message structure but appends the two M3.1 decision-rule sentences.

### Phase 1 — broad screen (5 candidates × 7 datasets × n=300)

All candidates run greedy / `seed=1` on the M4-perf optimised stack (multi-block `<tool_response>` per chunk where applicable, per-chunk cap 120 tok, `generator_max_input_len=8192`, retriever IVF-SQ8 × 16 + `asyncio.to_thread` + `INFERENCE_MAX_WORKERS=128`). Final-answer wrap is plain `<answer>X</answer>` for all (no `\boxed{}`).

| # | `prompt_mode` | Locus | Tag scheme | `tools=[]` auto-inject | `enable_thinking` | Decision rules | Example | Source / rationale |
|---|---|---|---|---|---|---|---|---|
| **A** | `qwen35_minimal` (control) | user | `<tool_call>` / `<tool_response>` | yes | True | no | no | M4.2 lock; baseline = 0.0594 |
| **B** | `qwen35_searchr1_pure` | user | `<search>` / `<information>` | no | True | no | no | Verbatim port of [`SEARCH_R1_TEMPLATE`](../../evaluation_qwen35/flashrag/search_r1/templates.py#L175-L186); M1 paper config |
| **C** | `qwen35_p3_decide_no_ex` | system | `<search>` / `<result>` | no | True | yes | no | Verbatim port of [`P3_DECIDE_NO_EX_TEMPLATE`](../../evaluation_qwen35/flashrag/search_r1/templates.py#L32-L42); M3.1 winner shape |
| **D** | `qwen35_minimal_decide` | user | `<tool_call>` / `<tool_response>` | yes | True | yes | no | A's user message + the two M3.1 decision sentences |
| **E** | `qwen35_minimal_nothink` | user | `<tool_call>` / `<tool_response>` | yes | **False** | no | no | A unchanged, `<think>\n\n</think>` closed |

Notes on B and C:

- **B**: byte-identical to the Search-R1 paper template that drove the M1 0.292 EM Qwen2.5-3B reproduction. Empty system, single user message wrapping `SEARCH_R1_TEMPLATE.format(prompt=question)`, retrieval wrap `\n\n<information>{X}</information>\n\n` (search_r1 mode shape). Tests the hypothesis that the prose / tag scheme that worked at the paper's home base also works on Qwen3.5.
- **C**: system role contains the M3.1 winner; user message is `Question: {q}` (matches M4.1 layout); retrieval wrap `" <result>\n{X}\n</result>"` (qwen3 mode shape). Drop the `\boxed{}` wrapper — M4 EM scorer normalises and the plain form is shorter / matches what M5 training will emit.

Wall-clock estimate: ~25 min / config × 5 configs ≈ **2 h** on the optimised stack (n=300/dataset × 7 datasets = 2100 items / config).

### Acceptance bar

A candidate replaces A only if **mean EM ≥ A + 0.025** at n=300. Wilson 95 % CI half-width at p=0.06, n=300 is ~2.7 pp; +0.025 is the floor for "real lift" rather than noise. Below that, lock M4.2 (A) unchanged for the full sweep.

### Phase 2 — refine the Phase-1 winner (conditional, n=300)

If Phase 1 produces a candidate above the bar, ablate one knob to test additivity. ~25 min wall.

| Phase-1 winner | Phase-2 ablation |
|---|---|
| B (Search-R1 verbatim) | B + decision rules from C ("Search-R1 prose + M3.1 rules") |
| C (M3.1 port) | C + 1-shot Hamlet example (the M3 `p1_basic_w_ex` shape) |
| D (M4.2 + rules) | D + `enable_thinking=False` (combine D and E) |
| E (M4.2 no-think) | E + decision rules (combine E and D) |

A Phase-2 winner needs **+0.04** over A to commit to Phase 3 (full sweep).

### Phase 3 — lock + full sweep

Replace the in-flight musique-hybrid cell with the locked-best `prompt_mode`. Re-run all 7 hybrid cells at n=51,713 with the locked prompt; keep the 6 already-finished hybrid cells **only if** the lock equals A (unchanged baseline). Wall-clock ~7 h on the optimised stack. Base variant stays at `qwen35_minimal_no_system` per the M4.3 lock — base lacks the tool-use prior, and M4.3 already locked the right shape there.

### Implementation cost

| Component | Files touched | Effort |
|---|---|---|
| Templates B–E | [`evaluation_qwen35/flashrag/search_r1/templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py) | ~80 LoC, 4 new constants + registry entries |
| Pipeline branches | [`evaluation_qwen35/flashrag/pipeline/active_pipeline.py`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py) | B and C re-use the existing qwen3 `<search>`/`</search>` action stop + `<information>` / `<result>` env wraps; E needs `enable_thinking` plumbed per-mode |
| Parser | [`evaluation_qwen35/flashrag/search_r1/parser.py`](../../evaluation_qwen35/flashrag/search_r1/parser.py) | `extract_search_tag_query` already exists from the qwen3 branch — re-used unchanged for B and C |
| Driver | [`scripts/run_m4.sh`](../../scripts/run_m4.sh) | accept `PROMPT_MODE` for any of the 5 modes; `save_note` tags collide-free |
| Orchestrator | new `scripts/orchestrate_m4_4.sh` | sequential 5-config × 7-dataset run with skip-aware resumption; identical structure to [`orchestrate_C_then_A.sh`](../../scripts/orchestrate_C_then_A.sh) |

Estimated implementation wall: **3–4 h coding + 1× n=100 sanity-check rendering** before launching the n=300 screen in earnest.

### Risks and stop conditions

- **Noise floor**: at n=300, +0.025 is barely outside CI. Some "lifts" will be noise. The bar is a screening floor; for a Phase-2 winner we want +0.04 before committing to the full sweep.
- **Diminishing returns**: this is the 4th prompt iteration on M4 (v1 → v2 → v3 → minimal / no-system → M4.4). If Phase 1 produces no candidate above the bar, **ship M4.2 unchanged** and move to M5 — the higher-leverage next step is training, not prompt golf.
- **Don't expand Phase 1**: if 5 candidates aren't enough, the gap is probably not closeable with prompting alone — that's an evaluative finding, not a "try more prompts" signal.

### Phase 1 results — `qwen35_recall_port` (ran first, 2026-05-10)

Drafted as a Phase-2 backup but executed first while Phase 1 implementation work (B / C / D / E) was still pending; it lifts directly into the existing M4.1 system-role + `tools=[]` render branch with zero pipeline-code changes (template-only delta). Adapts the verbatim Agent-RL/ReCall `re_call_template_sys` prose ([`Agent-RL/ReCall/src/verl/utils/dataset/template.py`](https://github.com/Agent-RL/ReCall/blob/main/src/verl/utils/dataset/template.py)) onto the M4.2 scaffolding — keeps `<tool_call>` / `<tool_response>` tags via Qwen3.5 nested-XML auto-inject, swaps `\boxed{}` for `<answer>X</answer>`, drops the redundant `{func_schemas}` + JSON-args reminder blocks (auto-inject covers both), replaces ReCall's weather example with a search-style Ted Turner / CNN example.

Constant: [`QWEN35_RECALL_PORT_TEMPLATE`](../../evaluation_qwen35/flashrag/search_r1/templates.py) (~95 words, 456 prompt tokens vs A's 418). Render verified against `Qwen/Qwen3.5-0.8B` chat template before launch.

Run: hybrid only, n=300 / dataset (bamboogle capped at full split = 125), greedy / `seed=1`, optimised stack (16-worker retriever + `asyncio.to_thread` + 128 client workers). Wall-clock 17 min 19 s for all 7 datasets (1039 s total; per-dataset 79–180 s).

| Dataset | M4.2 baseline (n=1000, §9.1) | M4.4 `qwen35_recall_port` (n=300) | Δ |
|---|---:|---:|---:|
| bamboogle (n=125) | 0.040 | 0.024 | −0.016 |
| nq | 0.065 | 0.053 | −0.012 |
| triviaqa | 0.122 | 0.123 | +0.001 |
| popqa | 0.071 | 0.067 | −0.004 |
| hotpotqa | 0.067 | 0.070 | +0.003 |
| 2wikimultihopqa | 0.041 | 0.017 | **−0.024** |
| musique | 0.010 | 0.003 | −0.007 |
| **mean** | **0.0594** | **0.0510** | **−0.0084** |

**Verdict: fails the acceptance bar by 3.3 pp.** 5 / 7 datasets regress; the largest single drop is 2wikimultihopqa (−0.024), a multi-hop dataset where M4.2 was already weak. Below A on aggregate at p < 0.10 (binomial sign test on per-dataset Δ).

**Mechanism (likely):** moving the protocol prose from user role (M4.2) into system role and re-framing it around ReCall's "<think> reasoning + <tool_call> action" schema *adds* scaffolding tokens (+38 prompt tokens) without changing what the model already knew from the auto-injected `# Tools` + `<IMPORTANT>` block. The auto-inject is the load-bearing piece for hybrid (per [`RESULTS_SMOKE_m4.md` §7.3](../report/RESULTS_SMOKE_m4.md)); piling more system-role prose on top crowds it without adding signal, and on multi-hop datasets the extra system-role text appears to displace search-loop attention.

**Implication for the rest of Phase 1**: candidate D (M4.2 + M3.1 decision rules in the *user* message — no system change) and candidate E (M4.2 + `enable_thinking=False`) become the highest-priority remaining tests, since both keep the M4.2 user-role + auto-inject base intact and modify only one knob each. Candidate C (full M3.1 system-role port) is now lower priority — `qwen35_recall_port` is a milder "system-role prose" intervention than C and already failed.

Result files: `evaluation_qwen35/results/<dataset>/<dataset>_*_m4_qwen3.5_0.8b_qwen35_recall_port_seed1_n300/metric_score.txt` (all 7 preserved).

## What's left

| # | Task | Status |
|---|---|---|
| 1 | Quick smoke (100 / dataset) for both variants on the M4.1 prompt | ✅ done; M4.2 / M4.3 asymmetric per-variant defaults locked ([`RESULTS_SMOKE_m4.md` §6 / §7](../report/RESULTS_SMOKE_m4.md)) |
| 2 | Full sweep 51,713 items / variant — base + hybrid baselines (M4.2 lock) | ✅ done 2026-05-10 ([`RESULTS_m4.md` §4](../report/RESULTS_m4.md)); hybrid mean EM 0.060, base 0.010 |
| 3 | Cross-family comparison M3 (Qwen3-0.6B) vs M4.2 (Qwen3.5-0.8B) baseline | ✅ done 2026-05-10 ([`RESULTS_m4.md` §5](../report/RESULTS_m4.md)); avg Δ −0.042 EM motivates M4.4 |
| 4 | **M4.4 prompt search at n=300** — close (or rule out) the M3 cross-family gap before committing M5 training compute | 🟡 Phase 1 partial: `qwen35_recall_port` ran 2026-05-10, mean EM 0.0510 vs A=0.0594 (FAILED bar by 3.3 pp); B / C / D / E still pending |
| 5 | Cross-family re-comparison M3 vs M4 post-M4.4 lock — refresh [`RESULTS_m4.md`](../report/RESULTS_m4.md) §5 if M4.4 changes the lock | ⏳ blocked on #4 |
| 6 | M5 (separate milestone): GRPO training on Qwen3.5-0.8B with the M4.4-locked prompt as the byte-aligned eval / train shape — see [`MILESTONE_5.md`](../milestone_5/MILESTONE_5.md) | ⏳ scaffold pending |

## Pointers

- M3 narrative: [`../milestone_3/MILESTONE_3.md`](../milestone_3/MILESTONE_3.md), [`../milestone_3/MILESTONE_3.1.md`](../milestone_3/MILESTONE_3.1.md)
- M3 alignment audit (the 14 fixes M4 inherits unchanged): [`../report/CODE_SETUP_m3.md`](../report/CODE_SETUP_m3.md) §3
- M3 results table: [`../report/RESULTS_m3.md`](../report/RESULTS_m3.md)
- Active recipe-ablation plan (drives M5+ training on Qwen3.5): [`../TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md)
- Phase-2 NeMo-RL training pipeline (smoke-tested on 1× A100 for Qwen3.5-2B): [`../training/SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md)
- Qwen3.5 chat template (verbatim): [`../training/CHAT_TEMPLATE.md`](../training/CHAT_TEMPLATE.md) §2
