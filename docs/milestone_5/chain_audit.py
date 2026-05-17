#!/usr/bin/env python3
"""Chain-flip audit on M5.1 rollouts (per M8.1 spec in docs/milestone_8/MILESTONE_8.md).

Reproduces the per-cadence "Measured chain-flip rate" table in
docs/report/RESULTS_M5_1_H200.md §9.5.

Usage:
    1. Pull rollouts from HF (each ~110 MB):
         for i in $(seq 1 180); do
           curl -sSL -o rollouts/train_data_step${i}.jsonl \\
             https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42-f1-only/resolve/main/rollouts/train_data_step${i}.jsonl &
           (( $(jobs -r | wc -l) >= 5 )) && wait -n
         done; wait
    2. Set ROLL_DIR below and run.

Rollout schema: each .jsonl line is one rollout with
    content = [[turn_0_str, turn_1_str, ..., turn_N_str]]  (list-of-list of chat turns)
    rewards = [scalar_reward]
"""
import json
import re
from pathlib import Path

ROLL_DIR = Path("/tmp/m5_rollouts/rollouts")

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
_TOOL_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
_BRIDGE_RE = re.compile(
    r"(?:country|city|state|nation|place|location)\s+(?:is|=|:|containing|of)?\s*[\"']?([A-Z][A-Za-z ]{2,40})[\"']?",
    re.I,
)


def has_silent_flip(rollout: str) -> bool:
    """True iff at least one silent entity flip detected."""
    thinks = _THINK_RE.findall(rollout)
    tools = _TOOL_RE.findall(rollout)
    if len(thinks) < 2:
        return False
    bridges = []
    for t in thinks:
        cands = _BRIDGE_RE.findall(t)
        bridges.append(cands[-1].strip() if cands else None)
    for i in range(1, len(bridges)):
        prev, curr = bridges[i - 1], bridges[i]
        if prev is None or curr is None or prev == curr:
            continue
        intervening = tools[i - 1:i] if i - 1 < len(tools) else []
        if not any(curr.lower() in tr.lower() for tr in intervening):
            return True
    return False


def audit_step(step: int) -> tuple[int, int]:
    """Return (perfect_count, silent_flip_count) for one step.

    Schema per HF rollout JSONL: one line = one rollout.
    `content` = [[turn_0_str, turn_1_str, ..., turn_N_str]] (list-of-list of chat turns)
    `rewards` = [scalar_reward]
    """
    path = ROLL_DIR / f"train_data_step{step}.jsonl"
    perfect = 0
    flips = 0
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rewards = rec.get("rewards", [])
            contents = rec.get("content", [])
            if not rewards or not contents:
                continue
            reward = rewards[0]
            turns = contents[0] if isinstance(contents[0], list) else [contents[0]]
            text = "\n".join(turns)
            if reward < 0.9:
                continue
            perfect += 1
            if has_silent_flip(text):
                flips += 1
    return perfect, flips


def main():
    # All 18 cadences. Each cadence covers 10 steps; C1 = steps 1-10, ..., C18 = 171-180.
    cadences = [(f"C{i+1}", range(1 + 10 * i, 11 + 10 * i)) for i in range(18)]
    print(f"{'Cadence':<6} {'Steps':<10} {'Perfect':>8} {'Flipped':>8} {'Rate':>7}")
    print("-" * 45)
    for name, rng in cadences:
        tot_p, tot_f = 0, 0
        for step in rng:
            try:
                p, f = audit_step(step)
            except FileNotFoundError:
                continue
            tot_p += p
            tot_f += f
        if tot_p == 0:
            continue  # missing rollouts for this cadence
        rate = tot_f / tot_p * 100
        s, e = min(rng), max(rng)
        print(f"{name:<6} {s}-{e:<6} {tot_p:>8} {tot_f:>8} {rate:>6.1f}%")


if __name__ == "__main__":
    main()
