from flashrag.search_r1.templates import SEARCH_R1_TEMPLATE, SEARCH_R1_TEMPLATE_SYS
from flashrag.search_r1.answer_utils import remove_boxed, last_boxed_only_string, extract_answer
from flashrag.search_r1.reward import compute_search_r1_reward
from flashrag.search_r1.parser import extract_search_tag_query

