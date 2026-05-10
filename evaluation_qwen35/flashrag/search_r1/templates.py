# ─── Qwen3 hybrid system-message templates (Phase-1 ALICE training) ──────────
# Each constant is byte-for-byte the system-message used at training time;
# evaluating with a different prompt than training produces off-distribution
# behaviour (see CODE_SETUP_m3 §3 for the alignment audit).

# `p1_basic_w_ex` (W&B id z7kcxfof): basic rules + 2-search Hamlet example.
# Anchored the model on heavy-tool 2-call / 4-turn behaviour (response length
# ~2050 tokens). End-of-run rollout reward 0.190 over 1046 steps. Evaluated in
# M3 (RESULTS_m3 §4–§9).
QWEN3_0_6B_TEMPLATE = (
    "You are a helpful assistant who can answer questions using multiple Wikipedia search tool calls.\n"
    "You can call the search tool by writing: <search> your query </search>\n"
    "You will receive the result in: <result> your search result </result>\n"
    "Use the search tool to obtain the information needed for the answer.\n"
    "Answers should be based on the search results.\n"
    "You may use the search tool multiple times if needed before giving the final answer.\n"
    "Provide the final answer in the format: <answer>The final answer is \\[ \\boxed{answer here} \\]</answer>.\n"
    "For example:\n"
    "Question: What is the nationality of the author of Hamlet?\n"
    "<search>Hamlet</search>\n"
    "<result>The Tragedy of Hamlet was written by William Shakespeare.</result>\n"
    "<search>William Shakespeare</search>\n"
    "<result>William Shakespeare was an English playwright.</result>\n"
    "<answer>The final answer is \\[ \\boxed{English} \\]</answer>"
)

# `p3_decide_no_ex` (W&B id el6s2d2h): decision-rule scaffolding, NO example.
# Best end-of-run rollout reward of the v0 block (0.215, +43% rel over the
# first-decile mean) over 2280 steps. Tool-use survives example removal here
# because the rules section has explicit per-step decision guidance — the only
# v0 prompt for which removing the example was harmless. Evaluated in M3.1.
P3_DECIDE_NO_EX_TEMPLATE = (
    "You are a helpful assistant who can answer questions using a Wikipedia search tool.\n"
    "You can call the search tool by writing: <search> your query </search>\n"
    "You will receive the result in: <result> your search result </result>\n"
    "Use the search tool to obtain the information needed for the answer.\n"
    "Use the information in the search results to determine the final answer.\n"
    "After each search result, decide whether another search is needed or whether you can provide the final answer.\n"
    "If a search result is incomplete, search again for the missing information.\n"
    "You may use the search tool multiple times if needed before giving the final answer.\n"
    "Provide the final answer in the format: <answer>The final answer is \\[ \\boxed{answer here} \\]</answer>."
)

# Registry: prompt_mode -> system-message template. All `qwen3*` modes share
# the same retrieval-text format / budgets / enable_thinking; only the system
# message differs.
QWEN3_TEMPLATES = {
    "qwen3": QWEN3_0_6B_TEMPLATE,
    "qwen3_p1_basic_w_ex": QWEN3_0_6B_TEMPLATE,  # explicit alias
    "qwen3_p3_decide_no_ex": P3_DECIDE_NO_EX_TEMPLATE,
}

# ─── Qwen3.5 system-message template (M4 / M4.1) ─────────────────────────────
# M4.1 prompt design (replaces the M4 placeholder flat `<tool_call>X</tool_call>`
# template): canonical Qwen3.5 nested-XML tool use, minimal system message,
# `Question: {q}` user message, plain `<answer>X</answer>` (no `\boxed{}`).
#
# We pass `tools=[QWEN35_SEARCH_TOOL]` to `tokenizer.apply_chat_template`, so
# Qwen3.5's chat template auto-injects (a) the `# Tools` block with the search
# function signature, (b) the verbatim format spec
# (`<tool_call>\n<function=NAME>\n<parameter=ARG>\nVAL\n</parameter>\n</function>\n</tool_call>`),
# and (c) the `<IMPORTANT>` reminder block — verbatim text from the model's
# `tokenizer_config.json:chat_template` field on HF (see CHAT_TEMPLATE.md §2,
# verified against `https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/tokenizer_config.json`).
# Auto-injection is in Qwen3.5's post-training distribution, so leveraging it
# costs the model zero adaptation vs. the off-distribution flat form.
#
# This template (system role) carries only what the auto-injected block does
# NOT cover: the loop semantics — the user gives a question, the model decides
# whether to search, observes the result, decides whether to answer or search
# again. The format spec itself is auto-injected, so we don't repeat it. No
# few-shot example: we want to see whether the post-training prior alone is
# enough to drive the loop on this 0.8B model.
QWEN35_NATIVE_TEMPLATE = (
    "You are a helpful assistant with access to a `search` tool that retrieves Wikipedia passages. "
    "Always call `search` at least once before answering; do not answer from prior knowledge.\n"
    "\n"
    "The user will give you a question in the form: Question: <question>\n"
    "\n"
    "Steps:\n"
    "- Call `search` with a focused query in the format described above.\n"
    "- Each search call returns several Wikipedia passages, one per "
    "<tool_response>...</tool_response> block. Some passages may be off-topic; "
    "use only the relevant ones.\n"
    "- If the relevant passages contain the answer, write it inside <answer> and </answer> and stop.\n"
    "- Otherwise, refine your query using facts and entities from those relevant passages "
    "(not from prior knowledge) and call `search` again."
)

# OpenAI-style tool schema rendered into the system area by Qwen3.5's chat
# template. Mirror of training/src/chat_template/tools.py:SEARCH_TOOL so the
# eval prompt is byte-identical to the training rollout's qwen_native arm
# (CHAT_TEMPLATE.md §1a). Keep schema small: one `query: str` is all retrieval
# needs, and shallow schemas reduce the `# Tools` token overhead.
QWEN35_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search Wikipedia for passages relevant to the query. Returns the "
            "top-K most relevant chunks. Call this whenever the question "
            "requires factual knowledge you do not already have."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific.",
                },
            },
            "required": ["query"],
        },
    },
}

QWEN35_TEMPLATES = {
    "qwen35": QWEN35_NATIVE_TEMPLATE,
    "qwen35_native": QWEN35_NATIVE_TEMPLATE,  # explicit alias
}

# ─── M4.2 Search-R1-style minimal user-message prompt for Qwen3.5 ────────────
# v3 hit EM 0.0086 mean across 7 datasets while M3 untrained Qwen3-0.6B got 0.102
# on the same prompts (RESULTS_SMOKE_m4 §4.3); the gap implicates the ~330-word
# pre-question scaffolding (auto-injected `# Tools` + `<IMPORTANT>` + our system
# message). This template ports Search-R1's compact paper prompt to Qwen3.5
# tags: protocol prose lives in the USER message (Search-R1 style — dense,
# single-paragraph, one-word `<answer> Beijing </answer>` example), and we
# still pass `tools=[QWEN35_SEARCH_TOOL]` so the chat template auto-injects
# the `# Tools` format spec into the system role (in-distribution for
# Qwen3.5's tool-use post-training; CHAT_TEMPLATE.md §2). System message
# itself is empty — model sees just the auto-injected block, then our user
# message containing the protocol + question.
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

QWEN35_TEMPLATES["qwen35_minimal"] = QWEN35_SEARCH_R1_LIKE_TEMPLATE
QWEN35_TEMPLATES["qwen35_searchr1"] = QWEN35_SEARCH_R1_LIKE_TEMPLATE  # alias

# ─── M4.3 — fully-minimal: no system message, no `tools=[]` auto-inject ──────
# Strips the `tools=[QWEN35_SEARCH_TOOL]` auto-injected `# Tools` schema +
# `<IMPORTANT>` reminder block (~220 words) by NOT passing tools= to
# apply_chat_template. The model now sees ONLY the user message, no system
# block at all. To compensate we inline the verbatim nested-XML format spec
# (lifted from the chat template's auto-inject) so the model still has a
# format anchor; this is technically off-distribution for Qwen3.5's tool-use
# post-training (the auto-inject is what the model was trained on), but it
# tests whether further scaffolding reduction lifts EM. Total prompt
# token count: ~150 (vs ~424 for qwen35_minimal which keeps the auto-inject).
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

QWEN35_TEMPLATES["qwen35_minimal_no_system"] = QWEN35_MINIMAL_NO_SYSTEM_TEMPLATE

# ─── M4.4 — ReCall-port system message (Agent-RL/ReCall prose, M4.2 scaffolding) ───
# Verbatim ReCall `re_call_template_sys` (Agent-RL/ReCall, src/verl/utils/dataset/template.py)
# adapted to our M4 lock:
#   - keep `<tool_call>` / `<tool_response>` tags via Qwen3.5 nested-XML (auto-injected by
#     `tools=[QWEN35_SEARCH_TOOL]`); drop ReCall's JSON-args format reminder (auto-inject covers it)
#   - swap `\boxed{}` for `<answer>X</answer>` (M4 lock)
#   - drop the `{func_schemas}` placeholder block (the auto-inject already emits the function signature)
#   - replace ReCall's weather/Beijing example with a search-style example (Ted Turner / CNN)
#   - say `search` not "tools" (we have one tool; less abstract for the model)
# System-role: in-distribution for Qwen3.5's hybrid post-training.
# Pipeline routes this through the M4.1 branch (active_pipeline._is_qwen35, system + tools=[]).
QWEN35_RECALL_PORT_TEMPLATE = (
    "In this environment you have access to a `search` tool that retrieves Wikipedia passages. "
    "You may perform multiple rounds of search calls.\n"
    "\n"
    "In your response, first think about the reasoning process inside <think> </think>, "
    "then conduct function calling inside <tool_call> </tool_call> in the format described above. "
    "The results of each search call will be returned inside <tool_response> </tool_response>. "
    "You can continue to call `search` until you have enough information to answer the user's question. "
    "When you have the answer, enclose it inside <answer> </answer> and stop calling functions. "
    "For example: <think> Based on the response from the search call, the founder of CNN is Ted Turner. </think> "
    "<answer> Ted Turner </answer>."
)

QWEN35_TEMPLATES["qwen35_recall_port"] = QWEN35_RECALL_PORT_TEMPLATE

# ─── Search-R1 paper user-message template (M1 baseline reproduction) ────────
SEARCH_R1_TEMPLATE = (
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call a search engine by "
        "<search> query </search> and it will return the top searched results between "
        "<information> and </information>. "
        "You can search as many times as your want. "
        "If you find no further external knowledge needed, you can directly provide the answer inside "
        "<answer> and </answer>, without detailed illustrations. "
        "For example, <answer> Beijing </answer>. "
        "Question: {prompt}\n"
        )
