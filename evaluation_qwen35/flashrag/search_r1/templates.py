# ─── Qwen3 hybrid system-message templates (Phase-1 ALICE training) ──────────
# Each constant is byte-for-byte the system-message used at training time;
# evaluating with a different prompt than training produces off-distribution
# behaviour (see CODE_SETUP_v2 §3 for the alignment audit).

# `p1_basic_w_ex` (W&B id z7kcxfof): basic rules + 2-search Hamlet example.
# Anchored the model on heavy-tool 2-call / 4-turn behaviour (response length
# ~2050 tokens). End-of-run rollout reward 0.190 over 1046 steps. Evaluated in
# M3 (RESULTS_v2 §4–§9).
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

# ─── Qwen3.5 system-message template (M4) ────────────────────────────────────
# Same prose as `p1_basic_w_ex`, with `<search>` ↔ `<tool_call>` and
# `<result>` ↔ `<tool_response>`. The Qwen3.5 family ships these tags as
# in-distribution vocab tokens (CHAT_TEMPLATE.md §2), so we expect lower
# bootstrap cost than Qwen3-0.6B (Phase-1 v1 finding 5: `<tool_call>` cost
# nothing at equal step count vs paper `<search>` tags).
#
# Used for the M4 baseline eval of Qwen3.5-0.8B (base + hybrid) before any
# GRPO training; the eventual M4 trained checkpoint will use the same prompt
# verbatim, so this eval becomes the meaningful "untrained floor".
QWEN35_0_8B_TEMPLATE = (
    "You are a helpful assistant who can answer questions using multiple Wikipedia search tool calls.\n"
    "You can call the search tool by writing: <tool_call> your query </tool_call>\n"
    "You will receive the result in: <tool_response> your search result </tool_response>\n"
    "Use the search tool to obtain the information needed for the answer.\n"
    "Answers should be based on the search results.\n"
    "You may use the search tool multiple times if needed before giving the final answer.\n"
    "Provide the final answer in the format: <answer>The final answer is \\[ \\boxed{answer here} \\]</answer>.\n"
    "For example:\n"
    "Question: What is the nationality of the author of Hamlet?\n"
    "<tool_call>Hamlet</tool_call>\n"
    "<tool_response>The Tragedy of Hamlet was written by William Shakespeare.</tool_response>\n"
    "<tool_call>William Shakespeare</tool_call>\n"
    "<tool_response>William Shakespeare was an English playwright.</tool_response>\n"
    "<answer>The final answer is \\[ \\boxed{English} \\]</answer>"
)

QWEN35_TEMPLATES = {
    "qwen35": QWEN35_0_8B_TEMPLATE,
    "qwen35_p1_basic_w_ex": QWEN35_0_8B_TEMPLATE,  # explicit alias
}

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
