import re
from typing import Tuple


def extract_search_tag_query(text: str) -> Tuple[str, str]:
    matches = list(re.finditer(r"<search>(.*?)</search>", text, re.DOTALL))
    if not matches:
        return "", "no_search_tag"
    query = matches[-1].group(1).strip()
    if not query:
        return "", "empty_search_query"
    return query, "valid_search_tag"

