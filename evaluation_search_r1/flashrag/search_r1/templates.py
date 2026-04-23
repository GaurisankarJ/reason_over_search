SEARCH_R1_TEMPLATE = """Answer the given question.
You must conduct reasoning inside <think> and </think>.
After reasoning, if you need external knowledge, call search using <search> query </search>.
Retrieved content will be returned between <information> and </information>.
You may iterate think-search-information multiple times.
When ready, provide the final answer inside <answer> and </answer>.
Question: {prompt}
Assistant:"""


SEARCH_R1_TEMPLATE_SYS = """You are a helpful assistant that solves the question step by step with a search tool.

Use this protocol exactly:
1) Think with <think>...</think>
2) Search with <search>query</search> when needed
3) Consume evidence from <information>...</information>
4) End with <answer>final answer</answer>

Do not output tool JSON, and do not use <tool_call> or <tool_response> tags."""

