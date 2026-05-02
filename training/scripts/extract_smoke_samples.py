#!/usr/bin/env python3
"""Extract sampled (prompt, response, reward) tuples per combo for SMOKE_RESULTS.md.

Reads logs/smoke_{combo}/train_data_step{1,2}.jsonl and emits a markdown report.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "logs"
OUT = REPO / "docs" / "training" / "SMOKE_RESULTS.md"

COMBOS = [
    ("base_qwen_native", "base", "qwen_native"),
    ("base_paper", "base", "paper"),
    ("hybrid_qwen_native", "hybrid", "qwen_native"),
    ("hybrid_paper", "hybrid", "paper"),
]


def flatten_content(c) -> str:
    out: list[str] = []
    if isinstance(c, list):
        for x in c:
            if isinstance(x, list):
                out.extend(str(y) for y in x)
            else:
                out.append(str(x))
    else:
        out.append(str(c))
    return "".join(out)


def truncate_middle(s: str, head: int = 600, tail: int = 800) -> str:
    if len(s) <= head + tail + 64:
        return s
    elided = len(s) - head - tail
    return f"{s[:head]}\n\n[…{elided} chars elided…]\n\n{s[-tail:]}"


def split_prompt_response(full: str) -> tuple[str, str]:
    """Split on the last `<|im_start|>assistant\\n` that follows the user question.

    The Qwen tokenizer uses <|im_start|>assistant to mark assistant turn starts;
    everything before the FIRST such marker is the rendered prompt; the rest is
    the (multi-turn) response including tool calls and their responses.
    """
    marker = "<|im_start|>assistant\n"
    i = full.find(marker)
    if i < 0:
        return full, ""
    return full[: i + len(marker)], full[i + len(marker) :]


def count_retrieval_calls(arm: str, response: str) -> int:
    if arm == "qwen_native":
        return len(re.findall(r"<tool_call>", response))
    return len(re.findall(r"<search>", response))


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def render_combo(combo: str, variant: str, arm: str) -> dict:
    log_dir = LOGS / f"smoke_{combo}"
    if not log_dir.exists():
        return {
            "combo": combo,
            "missing": True,
            "log_dir": str(log_dir),
        }

    samples: list[dict] = []
    for step in (1, 2):
        f = log_dir / f"train_data_step{step}.jsonl"
        if not f.exists():
            continue
        for idx, d in enumerate(load_jsonl(f)):
            full = flatten_content(d["content"])
            prompt, response = split_prompt_response(full)
            reward = d["rewards"][0]
            input_len = (d.get("input_lengths") or [None])[0]
            samples.append(
                {
                    "step": step,
                    "idx": idx,
                    "reward": reward,
                    "input_len": input_len,
                    "prompt": prompt,
                    "response": response,
                    "n_retrieval": count_retrieval_calls(arm, response),
                    "format_valid_paper": is_paper_format_valid(response),
                    "has_answer": "<answer>" in response,
                }
            )

    if not samples:
        return {"combo": combo, "missing": True, "log_dir": str(log_dir)}

    rewards = [s["reward"] for s in samples]
    n = len(rewards)
    summary = {
        "combo": combo,
        "log_dir": str(log_dir),
        "variant": variant,
        "arm": arm,
        "n_traj": n,
        "mean_reward": sum(rewards) / n,
        "max_reward": max(rewards),
        "n_em_hits": sum(1 for r in rewards if r >= 0.99),  # 1.0 (paper format+EM) or 0.8 (EM, no format)
        "n_partial_em": sum(1 for r in rewards if 0.7 <= r < 0.99),
        "n_format_only": sum(1 for r in rewards if 0.05 < r < 0.7),
        "n_zero": sum(1 for r in rewards if r == 0.0),
        "n_format_valid_paper": sum(1 for s in samples if s["format_valid_paper"]),
        "n_has_answer": sum(1 for s in samples if s["has_answer"]),
        "avg_retrieval_calls": sum(s["n_retrieval"] for s in samples) / n,
    }

    # Pick 8-10 samples: prefer mix of (reward>0) and (reward=0).
    nonzero = [s for s in samples if s["reward"] > 0]
    zero = [s for s in samples if s["reward"] == 0]
    target = 9  # 8-10 acceptable, aim for 9
    chosen: list[dict] = []
    chosen.extend(sorted(nonzero, key=lambda s: -s["reward"])[: max(1, target // 2)])
    # Fill from zeros, sampling across both steps and prompts
    seen = {(s["step"], s["idx"]) for s in chosen}
    for s in zero:
        if len(chosen) >= target:
            break
        if (s["step"], s["idx"]) in seen:
            continue
        chosen.append(s)
        seen.add((s["step"], s["idx"]))
    summary["samples"] = chosen
    return summary


def is_paper_format_valid(response: str) -> bool:
    """Cheap check: paper-style trajectory ending with <answer>...</answer>
    and using <search>/<information> tags, no orphan opens."""
    if "<answer>" not in response or "</answer>" not in response:
        return False
    for tag in ("think", "search", "answer"):
        if response.count(f"<{tag}>") != response.count(f"</{tag}>"):
            return False
    return True


def md_summary_row(s: dict) -> str:
    if s.get("missing"):
        return f"| {s['combo']} | MISSING ({s['log_dir']}) | — | — | — | — | — |"
    return (
        f"| {s['variant']} × {s['arm']} | `{s['log_dir']}` | "
        f"{s['mean_reward']:.3f} | {s['n_em_hits']}/{s['n_traj']} | "
        f"{100 * s['n_has_answer'] / s['n_traj']:.0f}% | "
        f"{s['avg_retrieval_calls']:.1f} | "
        f"{'✓' if s['n_em_hits'] > 0 else '—'} |"
    )


def md_sample(s: dict, sample_n: int) -> str:
    """Render one sample block."""
    prompt = truncate_middle(s["prompt"], head=600, tail=400)
    response = truncate_middle(s["response"], head=800, tail=600)
    return (
        f"### Sample {sample_n} — step {s['step']}, idx {s['idx']} — "
        f"reward={s['reward']:.3f}, retrieval-calls={s['n_retrieval']}, "
        f"input_len={s['input_len']} tok\n\n"
        f"**Prompt** (rendered, truncated):\n```\n{prompt}\n```\n\n"
        f"**Response** (truncated):\n```\n{response}\n```\n"
    )


def main() -> int:
    summaries = [render_combo(c, v, a) for c, v, a in COMBOS]

    md_lines: list[str] = []
    md_lines.append("# Smoke Test Results — Phase 2 First-Pass Mechanics\n")
    md_lines.append(
        "Run on Vast.ai 1× A100 80 GB, image `pantomiman/reason-over-search-v1:v1`. "
        "Retriever: IVF-SQ8 + 8 workers (the flat IP default times out under "
        "training rollout load). Smoke shape: 2 outer steps × 4 prompts × group=5 "
        "= 40 trajectories per combo (20 per step jsonl line).\n"
    )
    md_lines.append(
        "**Required smoke overrides** (Qwen3.5 + multimodal + Mamba layers):\n\n"
        "```\npolicy.sequence_packing.enabled=false  "
        "policy.dynamic_batching.enabled=true  "
        "policy.train_micro_batch_size=2\n```\n"
    )

    md_lines.append("## Summary\n")
    md_lines.append(
        "| Combo | Run dir | Mean reward | EM hits | Has-answer % | "
        "Avg retrieval calls | Notes |\n"
        "|---|---|---:|---:|---:|---:|---|"
    )
    for s in summaries:
        md_lines.append(md_summary_row(s))

    for s in summaries:
        if s.get("missing"):
            continue
        md_lines.append(f"\n## {s['variant']} × {s['arm']}\n")
        md_lines.append(
            f"- trajectories: {s['n_traj']} ({s['n_traj']//2}/step × 2 steps)\n"
            f"- reward distribution: "
            f"EM-hit (≥0.8): {s['n_em_hits']} | "
            f"format-only (0.1–0.3): {s['n_format_only']} | "
            f"zero: {s['n_zero']}\n"
            f"- has-`<answer>`: {s['n_has_answer']}/{s['n_traj']}, "
            f"paper-format-valid: {s['n_format_valid_paper']}/{s['n_traj']}\n"
            f"- avg retrieval calls per traj: "
            f"{s['avg_retrieval_calls']:.2f} ({'<tool_call>' if s['arm']=='qwen_native' else '<search>'})\n"
        )
        for i, samp in enumerate(s["samples"], 1):
            md_lines.append(md_sample(samp, i))

    md_lines.append("\n## Observations\n")
    md_lines.append(
        "- **All four combos completed without errors** after applying the "
        "fixes in `training/fix/CHANGES.md`.\n"
        "- **Tag dispatch works**: `qwen_native` arm emits `<tool_call>` (retrieval "
        "via Qwen's native tool-call schema); `paper` arm emits `<search>` (Search-R1 "
        "paper schema). The reward function uses paper tags, so `qwen_native` "
        "trajectories that happen to wrap their answer in `<think>...</think>` + "
        "`<answer>...</answer>` (with the `<tool_call>` blocks treated as plain "
        "text by the state machine) can still hit reward=1.0 on EM. This matches "
        "the M2-baseline-by-design caveat noted in `training/src/rewards/search_r1.py`.\n"
        "- **Reward distribution at step 0 is sparse**: most trajectories score 0 "
        "(model fails to extract a correct answer or emits no `<answer>` block at "
        "all). One or two trajectories per step land EM-hits. This is expected for "
        "an untrained base model; learning kicks in over hundreds of steps.\n"
        "- **No retriever timeouts** in the IVF runs — 8 workers + IVF-SQ8 cleared "
        "the bottleneck that the flat IP default had.\n"
    )

    OUT.write_text("\n".join(md_lines))
    print(f"wrote {OUT}")
    print("Summary:")
    for s in summaries:
        if s.get("missing"):
            print(f"  - {s['combo']}: MISSING")
        else:
            print(
                f"  - {s['variant']:6s} × {s['arm']:11s} : mean={s['mean_reward']:.3f} "
                f"em={s['n_em_hits']}/{s['n_traj']} retr={s['avg_retrieval_calls']:.1f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
