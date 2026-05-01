#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.24",
#   "pyarrow>=15",
# ]
# ///
"""Download the Search-R1 RL training corpus and strip the prebaked template.

Source : https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train
Output : data/training/nq_hotpotqa_train/{train,test}.parquet (repo-root relative)

Why we strip: upstream `make_prefix` bakes the paper's `<search>` / `<information>`
instruction string into `prompt[0].content`. Our Milestone 2 baseline uses
Qwen3.5's native `<tool_call>` template instead (see docs/training/CHAT_TEMPLATE.md),
so we keep the dataset template-agnostic and let the training loop apply
whichever chat template the run config selects at rollout time.

Conversion is a single field rewrite: `prompt[0].content := question`. Everything
else (`question`, `golden_answers`, `data_source`, `reward_model.ground_truth`,
`extra_info`) is preserved verbatim.

We read the upstream parquets directly with pyarrow rather than via
`datasets.load_dataset`, because the latter tries to unify the per-row
`extra_info` schema across the NQ + HotpotQA mixture and fails (the two
sub-datasets carry different per-row metadata).

Idempotent. Re-runs skip if the output parquets exist; pass `--force` to override.

Run:
  training/scripts/prepare_dataset.py [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

HF_REPO = "PeterJinGo/nq_hotpotqa_train"
SPLITS = ("train", "test")
EXPECTED_ROWS = {"train": 169_615, "test": 51_713}

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "training" / "nq_hotpotqa_train"


def rewrite_prompt_column(table: pa.Table) -> pa.Table:
    """Replace each row's `prompt` with `[{"role": "user", "content": question}]`."""
    questions = table.column("question").to_pylist()
    new_prompt = [[{"role": "user", "content": q}] for q in questions]
    prompt_idx = table.schema.get_field_index("prompt")
    return table.set_column(prompt_idx, "prompt", pa.array(new_prompt, type=table.schema.field("prompt").type))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--force", action="store_true", help="re-download even if outputs exist")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_paths = {s: OUT_DIR / f"{s}.parquet" for s in SPLITS}

    if not args.force and all(p.exists() for p in out_paths.values()):
        print(f"[skip] outputs already present at {OUT_DIR} (use --force to redo)")
        for s, p in out_paths.items():
            print(f"  {s}: {p} ({p.stat().st_size / 1e6:.1f} MB)")
        return 0

    sample_before: str | None = None
    sample_after: str | None = None
    sample_question: str | None = None

    for split in SPLITS:
        print(f"[fetch] {HF_REPO}:{split}.parquet")
        src = hf_hub_download(HF_REPO, f"{split}.parquet", repo_type="dataset")
        table = pq.read_table(src)
        n = table.num_rows
        if n != EXPECTED_ROWS[split]:
            print(f"[warn] {split}: got {n} rows, expected {EXPECTED_ROWS[split]}")

        if split == "train":
            sample_before = table.column("prompt")[0].as_py()[0]["content"]

        rewritten = rewrite_prompt_column(table)

        if split == "train":
            sample_after = rewritten.column("prompt")[0].as_py()[0]["content"]
            sample_question = rewritten.column("question")[0].as_py()

        pq.write_table(rewritten, out_paths[split])
        print(f"[write] {split}: {out_paths[split]} ({n} rows, {out_paths[split].stat().st_size / 1e6:.1f} MB)")

    print("\n[verify] row 0 prompt[0].content")
    print(f"  before  : {sample_before[:120]}{'…' if len(sample_before) > 120 else ''}")
    print(f"  after   : {sample_after!r}")
    print(f"  question: {sample_question!r}")
    if sample_after != sample_question:
        print("[error] post-convert content does not match question", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
