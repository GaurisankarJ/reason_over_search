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


# Per-observation char cap, mirroring verl's `max_obs_length=500` (tokens) from
# Search-R1's v0.2 yaml. We use a char proxy (~4 chars/token for English Wiki
# text) instead of a real tokenizer so parsers.py stays pure-Python and unit-
# testable without dragging in transformers. ~500 tokens × 4 ≈ 2000 chars.
DEFAULT_MAX_OBS_CHARS = 2000


def _truncate_body(body: str, max_chars: int | None) -> str:
    if max_chars is None or len(body) <= max_chars:
        return body
    # Truncate hard, append a marker so the LM knows it's been cut. The marker
    # itself fits inside the cap (subtract its length).
    marker = "\n…[truncated]"
    keep = max(0, max_chars - len(marker))
    return body[:keep] + marker


def format_docs_paper(docs: list[str], max_chars: int | None = DEFAULT_MAX_OBS_CHARS) -> str:
    """Wrap retrieved docs in the paper's `<information>` envelope.

    `max_chars` caps the docs body (numbered Doc lines) before wrapping —
    matches the *intent* of verl's `max_obs_length=500` token cap, using a
    char proxy. Set to None to disable.
    """
    body = "\n".join(f"Doc {i + 1}: {d}" for i, d in enumerate(docs))
    body = _truncate_body(body, max_chars)
    return f"\n<information>\n{body}\n</information>\n"


def format_docs_qwen_native(
    docs: list[str], max_chars: int | None = DEFAULT_MAX_OBS_CHARS
) -> str:
    """Hand-craft the Qwen3.5 chat-template markers for a tool response.

    The previous assistant turn ended at `</tool_call>` (vLLM stop string),
    no `<|im_end|>` was emitted. We close the assistant turn, open a synthetic
    user turn carrying `<tool_response>`, close it, and re-open the assistant
    so the next generation continues in-distribution.

    `max_chars` caps the docs body — see `format_docs_paper` for the rationale.
    """
    body = "\n".join(f"Doc {i + 1}: {d}" for i, d in enumerate(docs))
    body = _truncate_body(body, max_chars)
    return (
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<tool_response>\n"
        f"{body}\n"
        "</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def retriever_failed_message(
    arm: str, reason: str, max_chars: int | None = DEFAULT_MAX_OBS_CHARS
) -> str:
    """Wrap a retrieval failure as a tool/info response so the LM can recover."""
    msg = f"Retriever failed: {reason}"
    if arm == "qwen_native":
        return format_docs_qwen_native([msg], max_chars=max_chars)
    return format_docs_paper([msg], max_chars=max_chars)
