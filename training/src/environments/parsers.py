"""Pure-Python parsing & formatting helpers for the Search-R1 env.

Split out so unit tests can import these without dragging in `torch`, `ray`, or
`nemo_rl.*`. The env actor (search_r1_env.py) re-exports and uses these.
"""
from __future__ import annotations

import re
from typing import Optional

VALID_ARMS = ("qwen_native", "paper")

# DOTALL so newlines inside content blocks match `.`.
_RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_RE_PAPER_SEARCH = re.compile(r"<search>(.*?)</search>", re.DOTALL)
# Permissive parser for Qwen3.5 tool calls — only `<parameter=query>` matters.
_RE_QWEN_QUERY = re.compile(
    r"<tool_call>.*?<parameter=query>\s*(.*?)\s*</parameter>.*?</tool_call>",
    re.DOTALL,
)


def parse_query(arm: str, assistant_text: str) -> Optional[str]:
    """Pull the latest search query out of an assistant message; None if absent."""
    if arm == "qwen_native":
        m = _RE_QWEN_QUERY.search(assistant_text)
    elif arm == "paper":
        m = _RE_PAPER_SEARCH.search(assistant_text)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    if not m:
        return None
    q = m.group(1).strip()
    return q or None


def format_docs_paper(docs: list[str]) -> str:
    """Wrap retrieved docs in the paper's `<information>` envelope."""
    body = "\n".join(f"Doc {i + 1}: {d}" for i, d in enumerate(docs))
    return f"\n<information>\n{body}\n</information>\n"


def format_docs_qwen_native(docs: list[str]) -> str:
    """Hand-craft the Qwen3.5 chat-template markers for a tool response.

    The previous assistant turn ended at `</tool_call>` (vLLM stop string),
    no `<|im_end|>` was emitted. We close the assistant turn, open a synthetic
    user turn carrying `<tool_response>`, close it, and re-open the assistant
    so the next generation continues in-distribution.
    """
    body = "\n".join(f"Doc {i + 1}: {d}" for i, d in enumerate(docs))
    return (
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<tool_response>\n"
        f"{body}\n"
        "</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def retriever_failed_message(arm: str, reason: str) -> str:
    """Wrap a retrieval failure as a tool/info response so the LM can recover."""
    msg = f"Retriever failed: {reason}"
    if arm == "qwen_native":
        return format_docs_qwen_native([msg])
    return format_docs_paper([msg])
