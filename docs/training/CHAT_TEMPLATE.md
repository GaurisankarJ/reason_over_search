# Chat Template Decision

## TL;DR

Search-R1 invents its own `<search>` / `<information>` tags and teaches the model them via RL. **Qwen3.5 was already pre-trained on `<tool_call>` / `<tool_response>` tags** with XML-nested function/parameter elements ([verbatim from Qwen3.5-2B's `tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/tokenizer_config.json)). We therefore use **Qwen3.5's native `<tool_call>` template as the baseline**: keep the action format inside the pre-training distribution so RL spends compute on *behaviour* (when to search, what to query), not on learning new structural tokens.

> **Correction note**: an earlier draft of this doc embedded the Qwen3 (3.0) tool format (Hermes-style JSON inside `<tool_call>`). Qwen3.5 changed the format — it now wraps a function call in **XML elements** with `<function=name>` and `<parameter=name>` tags. The verbatim jinja below is from Qwen3.5-2B specifically.

---

## 1. Paper template (Search-R1)

Source: [`evaluation_search_r1/flashrag/search_r1/templates.py`](../../evaluation_search_r1/flashrag/search_r1/templates.py) (canonical, copied from upstream Search-R1).

Search-R1 uses **a single prompt template** for both base and instruct variants. Upstream `make_prefix` ([`Search-R1/scripts/data_process/qa_search_test_merge.py:26-39`](https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/data_process/qa_search_test_merge.py)) only implements `template_type == 'base'` and raises `NotImplementedError` otherwise; the four sibling data-prep scripts (`qa_search_train_merge.py`, `nq_search.py`, `nq_rag.py`) are identical. The same string is fed as a `user`-role message in both cases — for instruct it is additionally rendered through the model's chat template, but no separate system prompt is added.

**Single template (base + instruct):**

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

**Tag inventory:**

| Tag | Purpose |
|---|---|
| `<think> ... </think>` | Reasoning blocks |
| `<search> query </search>` | Retrieval call (Search-R1 invention, not in any base model's pre-training distribution) |
| `<information> ... </information>` | Retrieved documents injected into the trace (Search-R1 invention) |
| `<answer> ... </answer>` | Final answer; close-rate is the format-validity metric |

### 1a. Our `qwen_native` arm — paper-equivalent system prompt

Our default arm uses Qwen3.5's native `<tool_call>` / `<tool_response>` machinery, but we keep the *protocol* the paper teaches: reason in `<think>`, search when needed, deliver the answer in `<answer>...</answer>`. The `<search>` / `<information>` tag instructions are dropped because Qwen3.5 already knows how to call tools via `<tool_call>` and read their results via `<tool_response>` (post-training distribution covers both).

Source: [`training/src/prompts/search_r1_qwen_native_system.txt`](../../training/src/prompts/search_r1_qwen_native_system.txt). Loaded by the processor as a **system message** (verbatim, not a Python format string), wired via `data.default.system_prompt_file` in [`grpo_qwen3.5_2b_{1,2}xa100.yaml`](../../training/configs/).

**System prompt:**

```
You are a helpful assistant. Answer the user's question by using the `search` tool when you need external knowledge.

You must conduct reasoning inside <think> and </think> first every time you get new information. You may call `search` as many times as needed. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.
```

**User message:** the bare question (no template wrapping). The processor passes `messages[0].content` straight through.

**What Qwen3.5 auto-injects on top.** Because we pass `tools=[SEARCH_TOOL]` to `tokenizer.apply_chat_template`, Qwen3.5's chat template walks the [`SEARCH_TOOL` schema](../../training/src/chat_template/tools.py) and renders the `search` function signature into the system area on its own — so the model has both (a) the protocol from our system prompt and (b) the tool's argument schema from the auto-injected description. Together they're the qwen_native equivalent of the paper's verbose user-message template.

**Diff vs the paper template** — what we kept, what we dropped, why:

| Paper protocol element | qwen_native equivalent | Notes |
|---|---|---|
| "Answer the given question." | "Answer the user's question." | kept (same intent) |
| "conduct reasoning inside `<think>` and `</think>`" | identical wording | kept — `<think>` is a Qwen3.5 vocab token but not auto-emitted; we still need to instruct the model to use it |
| "you can call a search engine by `<search> query </search>` ..." | dropped | Qwen3.5's chat template emits `<tool_call><function=search><parameter=query>...</parameter></function></tool_call>` automatically when the model decides to call a registered tool |
| "it will return the top searched results between `<information>` and `</information>`" | dropped | Tool results come back as `<tool_response>...</tool_response>` (rendered by our env's `format_docs_qwen_native`) — already in Qwen3.5's training distribution |
| "search as many times as your want" | "call `search` as many times as needed" | kept (paraphrased) |
| "provide the answer inside `<answer>` and `</answer>`, without detailed illustrations" | identical | kept — `<answer>` is the EM scorer's anchor in [`reward.py:extract_solution`](../../training/src/rewards/search_r1.py); must be in the output |
| "For example, `<answer> Beijing </answer>`." | identical | kept — M1 audit (D-prompt-micro) found this micro-detail materially affects single-hop QA accuracy |
| "Question: {prompt}" | (omitted; the bare question goes in the user role) | the user role IS the question in chat-template land |

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
3. **Search-R1 invented its own tags before Qwen tool-use matured.** The paper introduces `<search>` / `<information>` and trains Qwen2.5 on them from scratch. That made sense in 2024 when Hermes-style tool use on Qwen was less mature, but for Qwen3.5 the calculus is reversed: the native tags *are* the path of least resistance.
4. **`<answer>` tag preserved.** The Milestone 1 EM scorer in [`flashrag/search_r1/reward.py`](../../evaluation_search_r1/flashrag/search_r1/reward.py) extracts answers from `<answer>...</answer>`. We keep that tag in the final assistant message so the existing eval pipeline works on the trained checkpoint without modification.

**Risk we are taking on** (track in W&B): if Qwen3.5's strong tool-use prior locks the policy into a behaviour that is hard to update via RL (mode collapse onto its pre-trained tool patterns), reward will plateau early. Mitigation: log per-step retrieval count and answer length distributions; if degenerate, fall back to the paper's `<search>` template as an ablation.

---

## 5. Dataset implication

The `prompt` field in [`PeterJinGo/nq_hotpotqa_train`](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train) **ships with Search-R1's `<search>`-tag instructions baked in**. Passing it through verbatim would force the policy to emit `<search>` tags regardless of which chat template the run config selects, and would require a per-arm dataset re-conversion to swap templates.

Our prep script ([`training/scripts/prepare_dataset.py`](../../training/scripts/prepare_dataset.py)) **strips the prebaked template** at download time — `prompt[0].content` becomes the bare `question`. The dataset is **template-agnostic**.

The chat template is applied at **rollout time** by the training loop (a config knob, not a dataset transformation):

- **`qwen_native` (default).** Bare question goes in the user role. Our paper-equivalent system prompt at [`training/src/prompts/search_r1_qwen_native_system.txt`](../../training/src/prompts/search_r1_qwen_native_system.txt) (see §1a above) is loaded via `data.default.system_prompt_file` and prepended as a `system` message. `tools=[SEARCH_TOOL]` from [`chat_template/tools.py`](../../training/src/chat_template/tools.py) is passed to `tokenizer.apply_chat_template` so Qwen3.5's jinja auto-renders the search tool's schema into the system area. Tool calls / responses follow §2's XML format.
- **`paper` (ablation arm).** Wrap the bare question with upstream `make_prefix` — the bash wrapper sets `data.default.prompt_file=training/src/prompts/search_r1_paper.txt` and clears `system_prompt_file`. The processor uses `task_data_spec.prompt.format(question)` to build the user-role content.

Switching arms is a config flip — same dataset (committed via LFS), different runtime template. `question`, `golden_answers`, `reward_model.ground_truth.target`, `data_source` are preserved verbatim by the prep script (those drive the EM reward and aren't tied to the prompt format).

---

## 6. Implementation pointer

| Component | Where |
|---|---|
| Tool schema (qwen_native arm) | [`training/src/chat_template/tools.py`](../../training/src/chat_template/tools.py) — `SEARCH_TOOL` |
| qwen_native system prompt | [`training/src/prompts/search_r1_qwen_native_system.txt`](../../training/src/prompts/search_r1_qwen_native_system.txt) |
| Paper user-message template | [`training/src/prompts/search_r1_paper.txt`](../../training/src/prompts/search_r1_paper.txt) |
| Arm dispatch (processor) | [`training/src/processors/search_r1.py`](../../training/src/processors/search_r1.py) — reads `task_data_spec.prompt` / `task_data_spec.system_prompt`, branches on `arm` recovered from `task_name` |
| Env's tool-call parser + tool-response formatter | [`training/src/environments/parsers.py`](../../training/src/environments/parsers.py) |
| Config defaults (qwen_native) | [`training/configs/grpo_qwen3.5_2b_{1,2}xa100.yaml`](../../training/configs/) — `data.default.system_prompt_file` |
| Config override (paper) | [`training/scripts/run_grpo_{1,2}xa100.sh`](../../training/scripts/) — when `--arm paper`, sets `prompt_file` and clears `system_prompt_file` |
