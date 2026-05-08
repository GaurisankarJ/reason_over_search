import re
from typing import Tuple


def extract_search_tag_query(text: str) -> Tuple[str, str]:
    # Match Search-R1 generation.py:postprocess_predictions — first match, not last.
    match = re.search(r"<search>(.*?)</search>", text, re.DOTALL)
    if not match:
        return "", "no_search_tag"
    query = match.group(1).strip()
    if not query:
        return "", "empty_search_query"
    return query, "valid_search_tag"


def extract_tool_call_query(text: str) -> Tuple[str, str]:
    """Pull the first `<tool_call>X</tool_call>` body (M4 Qwen3.5 flat scheme).

    Mirrors `extract_search_tag_query` for the Qwen3.5 prompt where the model
    is instructed to emit `<tool_call> query </tool_call>` directly. Does NOT
    parse the Qwen3.5 native nested XML form (`<function=...><parameter=...>`)
    because the M4 prompt explicitly uses the flat scheme; see
    `QWEN35_0_8B_TEMPLATE` in `templates.py`.
    """
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not match:
        return "", "no_tool_call_tag"
    query = match.group(1).strip()
    if not query:
        return "", "empty_tool_call_query"
    return query, "valid_tool_call_tag"
