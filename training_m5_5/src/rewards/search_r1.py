"""M5.5 F1+format reward — ReSearch paper's 3-tier shaping ported to Qwen3.5-0.8B.

Reward formula:

    reward(rollout) =
        0.0            if not is_valid_sequence(rollout)            # format broken
        FORMAT_BONUS   if extract_solution(rollout) is None         # format ok, no <answer>
        FORMAT_BONUS   if f1(answer, gold) == 0                     # format ok, answer wrong
        f1             if f1(answer, gold) > 0                      # format ok, partial / full

`FORMAT_BONUS = 0.1` matches the ReSearch paper's choice and the Phase-1 v0
observation that this floor masks the tool-use signal. The floor is NOT
additive (F1=0.5 -> reward=0.5, not 0.6); it only applies when F1==0.

Format validation reuses `is_valid_sequence` from
`training/src/rewards/search_r1.py` (the M2 paper-faithful module),
which is the canonical state-machine format walker for the
<think>/<search>/<information>/<answer> tag schema. F1 and `extract_solution`
are re-exported from the M4 eval pipeline so the M5 train-time reward and the
M4 eval-time scorer share the same code path (same precedent as M5.1).

API kept stable for `training_m5_5/src/environments/search_r1_env.py`:
`compute_search_r1_reward(solution_str, gold)`, `em_check`, `extract_solution`
exported under the same names as M5.1's F1-only reward.

Companion doc: docs/milestone_5/MILESTONE_5_5.md.
"""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

# Re-use the M4 eval's extract_solution + normalize_answer. Same precedent as
# M5.1: train and eval code paths must share string normalisation to avoid
# M3 14-fix-style train/eval drift.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EVAL_QWEN35_ROOT = _REPO_ROOT / "evaluation_qwen35"
if str(_EVAL_QWEN35_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_QWEN35_ROOT))

from flashrag.search_r1.reward import (  # noqa: E402
    extract_solution,
    normalize_answer,
)


def _load_m2_reward_module():
    """Load training/src/rewards/search_r1.py by absolute path.

    Bypass sys.path / package machinery to avoid a name collision: this very
    file IS `rewards.search_r1` under PYTHONPATH=training_m5_5/src, so a plain
    `from rewards.search_r1 import is_valid_sequence` resolves to ourselves
    (circular import).
    """
    m2_path = _REPO_ROOT / "training" / "src" / "rewards" / "search_r1.py"
    spec = importlib.util.spec_from_file_location("_m2_search_r1", str(m2_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load M2 reward module from {m2_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_m2 = _load_m2_reward_module()
is_valid_sequence = _m2.is_valid_sequence

# 0.1 partial-credit floor from the ReSearch paper / Phase-1 v0 observation.
# This is the load-bearing knob the ablation targets.
FORMAT_BONUS: float = 0.1

__all__ = [
    "compute_search_r1_reward",
    "em_check",
    "extract_solution",
    "f1_check",
    "is_valid_sequence",
    "normalize_answer",
    "FORMAT_BONUS",
]


def em_check(prediction: str, golden_answers) -> float:
    """SQuAD-style EM. Kept for telemetry parity with M5.1."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1.0
    return 0.0


def f1_check(prediction: str, golden_answers) -> float:
    """SQuAD-style token-overlap F1; best over the gold list (multi-answer).

    Same implementation as M5.1; duplicated here so this module is
    self-contained.
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for golden in golden_answers:
        gold_tokens = normalize_answer(golden).split()
        if not pred_tokens and not gold_tokens:
            best = max(best, 1.0)
            continue
        if not pred_tokens or not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best:
            best = f1
    return best


def compute_search_r1_reward(solution_str: str, golden_answers) -> dict:
    """M5.5 F1+format reward. Return shape matches M5.1's contract.

    `extracted_answer`, `f1`, `em` populated for telemetry; `format_valid`
    populated for the W&B trajectory plot (the headline signal of the
    ablation is whether format-valid rollouts cluster above the 0.1 floor
    or stay pinned to it).
    """
    is_valid, _ = is_valid_sequence(solution_str)
    if not is_valid:
        return {
            "reward": 0.0,
            "extracted_answer": None,
            "f1": 0.0,
            "em": 0.0,
            "format_valid": False,
        }
    answer = extract_solution(solution_str)
    if answer is None:
        return {
            "reward": FORMAT_BONUS,
            "extracted_answer": None,
            "f1": 0.0,
            "em": 0.0,
            "format_valid": True,
        }
    f1 = f1_check(answer, golden_answers)
    em = em_check(answer, golden_answers)
    reward = f1 if f1 > 0.0 else FORMAT_BONUS
    return {
        "reward": float(reward),
        "extracted_answer": answer,
        "f1": float(f1),
        "em": float(em),
        "format_valid": True,
    }
