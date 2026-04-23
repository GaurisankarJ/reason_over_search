from flashrag.search_r1.reward import compute_search_r1_reward
from flashrag.search_r1.answer_utils import extract_answer, last_boxed_only_string, remove_boxed


def test_search_r1_reward_valid_and_correct():
    text = (
        "<think>Need info</think>"
        "<search>capital of france</search>"
        "<information>Paris is the capital of France.</information>"
        "<think>Done</think>"
        "<answer>France</answer>"
    )
    result = compute_search_r1_reward(text, ["France"])
    assert result["reward"] == 1.0
    assert result["format_valid"] is True
    assert result["retrieval_hit"] is True
    assert result["extracted_answer"] == "France"


def test_search_r1_reward_format_fallback_score():
    text = "<answer>wrong</answer>"
    result = compute_search_r1_reward(text, ["France"])
    assert result["format_valid"] is False
    assert result["reward"] == 0.1


def test_answer_boxed_helpers_keep_correctness_path():
    text = "<answer> The final answer is \\[ \\boxed{Paris} \\] </answer>"
    answer_part = extract_answer(text)
    assert answer_part is not None
    boxed = last_boxed_only_string(answer_part)
    assert boxed == "\\boxed{Paris}"
    assert remove_boxed(boxed) == "Paris"

