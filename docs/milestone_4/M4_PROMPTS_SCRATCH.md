---
title: M4 prompts — scratch file for comments
tags: [milestone, eval, m4, scratch]
source: internal
created: 2026-05-10
updated: 2026-05-10
---

# M4 prompts — scratch file for comments

> Temporary scratch — feel free to scribble inline. Source-of-truth lives in
> [`evaluation_qwen35/flashrag/search_r1/templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py)
> and the variant lock is in [`MILESTONE_4.md`](MILESTONE_4.md) §"Variant dispatch".

## Variant dispatch (M4.2 / M4.3 lock-in, 2026-05-09)

| Variant | `prompt_mode` | System message | `tools=[…]` auto-inject | Rationale |
|---|---|---|---|---|
| `qwen3.5_0.8b` (hybrid) | `qwen35_minimal` | empty | **yes** (`# Tools` + format spec + `<IMPORTANT>`) | hybrid was post-trained on tool-use; auto-inject's `<IMPORTANT>` reminder drives search loops (in-distribution). Mean EM 0.057, 6.6× over v3. |
| `qwen3.5_0.8b_base` | `qwen35_minimal_no_system` | none at all | **no** | base lacks tool-use prior; auto-inject is scaffolding noise that crowds out the answer. Mean EM 0.016 vs 0.003 (5×). |

Both variants use `enable_thinking=True` (open `<think>\n` generation prefix) and greedy decode (`temperature=0.0`).

<!-- comment: -->

---

## Variant 1 — hybrid (`qwen35_minimal`)

**Locus**: protocol prose in user message; system is empty; `tools=[QWEN35_SEARCH_TOOL]` triggers Qwen3.5 chat-template auto-inject.

### Template constant (templates.py:134)

```python
QWEN35_SEARCH_R1_LIKE_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call the `search` tool in the "
    "format described above and it will return the top searched results inside <tool_response> and </tool_response>. "
    "You can search as many times as you want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "Question: {prompt}\n"
)
```

<!-- comment: -->

### What the model actually sees after `apply_chat_template`

```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search Wikipedia for passages relevant to the query. Returns the top-K most relevant chunks. Call this whenever the question requires factual knowledge you do not already have.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query. Be specific."}}, "required": ["query"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT><|im_end|>
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call the `search` tool in the format described above and it will return the top searched results inside <tool_response> and </tool_response>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: <QUESTION GOES HERE>
<|im_end|>
<|im_start|>assistant
<think>
```

The system block (everything between `<|im_start|>system` and `<|im_im_end|>`) is **entirely auto-generated** by Qwen3.5's chat template from `tools=[QWEN35_SEARCH_TOOL]` — we do not write it. Our `QWEN35_SEARCH_R1_LIKE_TEMPLATE` only fills the user turn.

<!-- comment: -->

---

## Variant 2 — base (`qwen35_minimal_no_system`)

**Locus**: everything in the user message. **No** system block. **No** `tools=[…]` arg → no auto-inject. Format spec is inlined verbatim into the user message as a fallback anchor.

### Template constant (templates.py:159)

```python
QWEN35_MINIMAL_NO_SYSTEM_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call the search tool by writing:\n"
    "<tool_call>\n<function=search>\n<parameter=query>\nyour query\n</parameter>\n</function>\n</tool_call>\n"
    "The result will be returned inside <tool_response> and </tool_response>. "
    "You can search as many times as you want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. "
    "Question: {prompt}\n"
)
```

<!-- comment: -->

### What the model actually sees after `apply_chat_template`

```
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call the search tool by writing:
<tool_call>
<function=search>
<parameter=query>
your query
</parameter>
</function>
</tool_call>
The result will be returned inside <tool_response> and </tool_response>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: <QUESTION GOES HERE>
<|im_end|>
<|im_start|>assistant
<think>
```

Total prompt token count: ~150 vs ~424 for hybrid (the `# Tools` + `<IMPORTANT>` block is ~220 words alone).

<!-- comment: -->

---

## Side-by-side diff

| Aspect | Hybrid `qwen35_minimal` | Base `qwen35_minimal_no_system` |
|---|---|---|
| System block | auto-injected (~220 words) | **absent** |
| `tools=[QWEN35_SEARCH_TOOL]` passed? | yes | **no** |
| User-message format spec | "in the format described above" (refers to auto-inject) | inlined `<tool_call>…</tool_call>` block |
| `<IMPORTANT>` reminder | yes (auto-injected) | **no** |
| Few-shot example | `<answer> Beijing </answer>` only | `<answer> Beijing </answer>` only |
| Token count | ~424 | ~150 |
| `enable_thinking` | True | True |
| Greedy decode | True | True |

<!-- comment: -->

---

## Open question — M4.4 prompt search (status: pending)

M4.2 hybrid lands at mean EM 0.0594; M3 untrained Qwen3-0.6B (smaller, older family) reaches 0.102 on the same 7 benchmarks. 1.7× cross-family gap → prompt headroom is the prime suspect. Phase-1 plan in [`MILESTONE_4.md` §M4.4](MILESTONE_4.md#L195) tests 5 candidates at n=300/dataset; if no candidate beats M4.2 by ≥ +1 pt mean EM, ship M4.2 unchanged.

<!-- comment: -->
