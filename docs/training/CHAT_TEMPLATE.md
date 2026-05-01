# Chat Template Decision

## TL;DR

Search-R1 invents its own `<search>` / `<information>` tags and teaches the model them via RL. **Qwen3.5 was already pre-trained on `<tool_call>` / `<tool_response>` tags** with XML-nested function/parameter elements ([verbatim from Qwen3.5-2B's `tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/tokenizer_config.json)). We therefore use **Qwen3.5's native `<tool_call>` template as the baseline**: keep the action format inside the pre-training distribution so RL spends compute on *behaviour* (when to search, what to query), not on learning new structural tokens.

> **Correction note**: an earlier draft of this doc embedded the Qwen3 (3.0) tool format (Hermes-style JSON inside `<tool_call>`). Qwen3.5 changed the format — it now wraps a function call in **XML elements** with `<function=name>` and `<parameter=name>` tags. The verbatim jinja below is from Qwen3.5-2B specifically.

---

## 1. Paper template (Search-R1)

Source: [`evaluation_search_r1/flashrag/search_r1/templates.py`](../../evaluation_search_r1/flashrag/search_r1/templates.py) (canonical, copied from upstream Search-R1).

**Base model — single-turn template:**

```
Answer the given question. You must conduct reasoning inside <think> and </think>
first every time you get new information. After reasoning, if you find you lack
some knowledge, you can call a search engine by <search> query </search> and it
will return the top searched results between <information> and </information>.
You can search as many times as your want. If you find no further external
knowledge needed, you can directly provide the answer inside <answer> and
</answer>, without detailed illustrations. For example, <answer> Beijing </answer>.
Question: {prompt}
```

**Instruct model — system prompt:**

```
You are a helpful assistant that solves the question step by step with a search
tool.

Use this protocol exactly:
1) Think with <think>...</think>
2) Search with <search>query</search> when needed
3) Consume evidence from <information>...</information>
4) End with <answer>final answer</answer>

Do not output tool JSON, and do not use <tool_call> or <tool_response> tags.
```

**Tag inventory:**

| Tag | Purpose |
|---|---|
| `<think> ... </think>` | Reasoning blocks |
| `<search> query </search>` | Retrieval call (Search-R1 invention, not in any base model's pre-training distribution) |
| `<information> ... </information>` | Retrieved documents injected into the trace (Search-R1 invention) |
| `<answer> ... </answer>` | Final answer; close-rate is the format-validity metric |

---

## 2. Qwen3.5-2B native template (verbatim)

Source: [`Qwen/Qwen3.5-2B/tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/tokenizer_config.json), `chat_template` field.

**Special-token IDs (relevant subset):**

| ID | Token | Marked `special` in tokenizer? |
|---|---|---|
| 248045 | `<\|im_start\|>` | yes |
| 248046 | `<\|im_end\|>` | yes |
| 248058 / 248059 | `<tool_call>` / `</tool_call>` | **no** (regular tokens) |
| 248066 / 248067 | `<tool_response>` / `</tool_response>` | **no** |
| 248068 / 248069 | `<think>` / `</think>` | **no** |

> Implication: `<tool_call>` and `<think>` are *vocabulary tokens* but not *special tokens* — they are emitted as text by the model and must not be `add_special_tokens`-stripped during training rollouts. The same lesson Milestone 1's audit (D8) flagged for the eval pipeline.

**Message wrapping (verbatim):**

```
<|im_start|>user
{user content}<|im_end|>
<|im_start|>assistant
{assistant content}<|im_end|>
```

**Generation prompt — thinking-mode toggle (verbatim jinja):**

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is true %}
        {{- '<think>\n' }}
    {%- else %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
```

So Qwen3.5-2B always emits `<think>...</think>` (empty when thinking is off). The hybrid variant we train uses `enable_thinking=true` so the assistant fills it.

**Tool call format — XML-style with nested `<function>` and `<parameter>` (verbatim jinja from Qwen3.5-2B):**

```jinja
{%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
    {%- for tool_call in message.tool_calls %}
        {%- if tool_call.function is defined %}
            {%- set tool_call = tool_call.function %}
        {%- endif %}
        {%- if loop.first %}
            {%- if content|trim %}
                {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
            {%- else %}
                {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
            {%- endif %}
        {%- else %}
            {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
        {%- endif %}
        {%- if tool_call.arguments is defined %}
            {%- for args_name, args_value in tool_call.arguments|items %}
                {{- '<parameter=' + args_name + '>\n' }}
                {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
                {{- args_value }}
                {{- '\n</parameter>\n' }}
            {%- endfor %}
        {%- endif %}
        {{- '</function>\n</tool_call>' }}
    {%- endfor %}
{%- endif %}
```

**Rendered tool call for a single search query:**

```
<tool_call>
<function=search>
<parameter=query>
who directed Inception
</parameter>
</function>
</tool_call>
```

**Tool response format (verbatim jinja):**

```jinja
{%- elif message.role == "tool" %}
    {%- if loop.previtem and loop.previtem.role != "tool" %}
        {{- '<|im_start|>user' }}
    {%- endif %}
    {{- '\n<tool_response>\n' }}
    {{- content }}
    {{- '\n</tool_response>' }}
    {%- if not loop.last and loop.nextitem.role != "tool" %}
        {{- '<|im_end|>\n' }}
    {%- elif loop.last %}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
```

**Rendered tool response:**

```
<|im_start|>user
<tool_response>
Doc 1: Inception is a 2010 science fiction film written and directed by Christopher Nolan...
Doc 2: ...
</tool_response><|im_end|>
```

**Tag inventory:**

| Tag | Purpose | Origin |
|---|---|---|
| `<\|im_start\|>` / `<\|im_end\|>` | Message delimiters | Qwen2.5-era special tokens, kept |
| `<think> ... </think>` | Reasoning (always emitted; empty when thinking off) | Qwen3-era, vocab token (not "special") |
| `<tool_call>\n<function=NAME>\n<parameter=ARG>\nVAL\n</parameter>\n</function>\n</tool_call>` | Tool call (XML-nested) | **Qwen3.5-era — new format vs. Qwen3** |
| `<tool_response>\n...\n</tool_response>` (in synthetic user turn) | Tool result | Qwen3-era, kept |

---

## 3. Side-by-side

| Concern | Search-R1 (paper) | Qwen3.5 native (ours) |
|---|---|---|
| Reasoning | `<think>...</think>` (free text) | `<think>...</think>` (always present, empty when thinking off) |
| Trigger retrieval | `<search>query</search>` | `<tool_call><function=search><parameter=query>QUERY</parameter></function></tool_call>` |
| Inject retrieved docs | `<information>...</information>` (in-stream, same turn) | `<tool_response>...</tool_response>` wrapped in a synthetic `user` turn (`<\|im_start\|>user ... <\|im_end\|>`) |
| Final answer | `<answer>...</answer>` | `<answer>...</answer>` (kept; the EM scorer expects it) |
| Pre-trained on these tags? | No — model learns them during GRPO | Yes — already in the post-training distribution |
| System-prompt registration of the tool? | No (free-text protocol) | Yes (Qwen3.5 expects an OpenAI-style tool schema in the system prompt; the jinja walks it) |
| Tool format | implicit (just the `<search>` tag) | Explicit XML: `<function=NAME>` + `<parameter=ARG>VAL</parameter>` |

---

## 4. Decision rationale

We use **Qwen3.5's native `<tool_call>` / `<tool_response>` template** as the Milestone 2 baseline. Reasons:

1. **In-distribution action format.** Qwen3.5 was post-trained on Hermes-XML tool use. The action format (call the tool, read the response, decide next step) is something the base policy already does competently, so RL can focus on improving the *retrieval policy* — when to search, what to search for, when to stop — rather than on teaching new structural tokens.
2. **No wasted reward signal on format compliance.** Search-R1 reports format-validity ≥99 % for base after training, but the early-training trajectory pays a tax to *learn* the tag protocol. With Qwen3.5's native tags we expect cleaner format compliance from step 0 (subject to verification — log close-rate from the start).
3. **Search-R1's own prompt explicitly forbids `<tool_call>`.** That choice made sense for Qwen2.5 where Hermes-style tool use was less mature, but for Qwen3.5 the calculus is reversed: the native tags *are* the path of least resistance.
4. **`<answer>` tag preserved.** The Milestone 1 EM scorer in [`flashrag/search_r1/reward.py`](../../evaluation_search_r1/flashrag/search_r1/reward.py) extracts answers from `<answer>...</answer>`. We keep that tag in the final assistant message so the existing eval pipeline works on the trained checkpoint without modification.

**Risk we are taking on** (track in W&B): if Qwen3.5's strong tool-use prior locks the policy into a behaviour that is hard to update via RL (mode collapse onto its pre-trained tool patterns), reward will plateau early. Mitigation: log per-step retrieval count and answer length distributions; if degenerate, fall back to the paper's `<search>` template as an ablation.

---

## 5. Dataset implication (read this before writing the conversion script)

The `prompt` field in [`PeterJinGo/nq_hotpotqa_train`](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train) **already contains Search-R1's `<search>`-tag instructions** (verified by inspecting the dataset). If we use Qwen3.5's `<tool_call>` template, we must **rewrite the system prompt during conversion** rather than passing the dataset's prompt through verbatim.

Two-step conversion in [`TRAINING_DATA.md`](TRAINING_DATA.md):

1. **Drop or replace the dataset's `prompt[0].content`** with our Qwen3.5-aware prompt (registering `search` as an OpenAI-style tool in a system message).
2. **Keep `question`, `golden_answers`, `reward_model.ground_truth.target`, `data_source`** as-is — those drive the EM reward and aren't tied to the prompt format.

---

## 6. Implementation pointer

The chat template lives in `training/src/chat_template/` (to be created). Two interchangeable implementations:

- `qwen_native.py` — the baseline. Builds prompts via the tokenizer's `apply_chat_template` with a registered `search` tool schema (function name = `search`, parameter = `query: str`). **Default for Milestone 2.**
- `paper.py` — port of [`evaluation_search_r1/flashrag/search_r1/templates.py`](../../evaluation_search_r1/flashrag/search_r1/templates.py). Available as an ablation arm.

Both must implement the same interface so the training loop can switch via config without touching core code.
