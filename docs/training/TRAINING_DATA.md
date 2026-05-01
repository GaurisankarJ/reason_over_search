# Training Data

The Search-R1 RL training corpus, lifted directly from the paper authors' HF distribution. Used for both [Milestone 2](../milestone_two/MILESTONE_2.md) variants (base + hybrid).

## Source

[`PeterJinGo/nq_hotpotqa_train`](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train) — official Search-R1 RL training mixture.

## Schema

Format: parquet (auto-converted from the upstream).

| Column | Type | Example |
|---|---|---|
| `id` | string | `train_0` |
| `question` | string (already `.strip() + '?'` normalized upstream) | `"total number of death row inmates in the us?"` |
| `golden_answers` | list[string] | `["2,718"]` |
| `data_source` | string (one of `nq`, `hotpotqa`) | `"nq"` |
| `prompt` | list[dict] (length 1) | upstream: paper template baked in; **after our prep script: `[{"role": "user", "content": question}]`** |
| `ability` | string | `"fact-reasoning"` |
| `reward_model` | dict | `{"ground_truth": {"target": ["2,718"]}, "style": "rule"}` |
| `extra_info` | dict | NQ + HotpotQA carry different per-row shapes; preserved verbatim |
| `metadata` | dict | `null` for NQ rows |

## Splits

| Split | Rows |
|---|---|
| `train` | 169,615 |
| `test` | 51,713 |

The `test` split is the in-loop validation set ([`VALIDATION.md`](VALIDATION.md)) — same as the paper.

## Download + prep

[`training/scripts/prepare_dataset.py`](../../training/scripts/prepare_dataset.py) — uv inline-script, idempotent. Pulls the upstream parquets via `hf_hub_download`, strips the prebaked Search-R1 template (replaces `prompt[0].content` with the bare `question`), writes to `data/training/nq_hotpotqa_train/{train,test}.parquet`. Re-runs skip unless `--force`.

```bash
training/scripts/prepare_dataset.py
```

> **Why `huggingface_hub` + pyarrow, not `datasets.load_dataset`:** the upstream parquet has heterogeneous `extra_info` schemas across the NQ + HotpotQA mixture, so the `datasets` library's cross-row schema unification fails with a cast error. Reading the parquet files directly bypasses this entirely.

## Why we strip the prebaked template

Upstream `make_prefix` ([`Search-R1/scripts/data_process/qa_search_test_merge.py:26-39`](https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/data_process/qa_search_test_merge.py)) bakes the paper's full `<think>` / `<search>` / `<information>` / `<answer>` instruction string into `prompt[0].content` before publishing. Keeping that text would (a) force the rollout to emit `<search>` tags regardless of which chat template the run config selects, and (b) make a per-arm dataset re-conversion necessary to swap templates.

Stripping to the bare question keeps the dataset **template-agnostic**. The chat template — Qwen3.5 native `<tool_call>` (default) or the paper's `<search>` (ablation arm) — is applied at **rollout time**, not at dataset prep time. See [`CHAT_TEMPLATE.md`](CHAT_TEMPLATE.md) §5.

## Open questions

1. **NeMo-RL's exact data schema** for GRPO. The example configs reference data sources but don't fully document the row-level shape. Check `examples/` in the NeMo-RL clone and update this doc with the verified mapping before wiring the data loader. If NeMo-RL expects a `messages` field (or anything other than `prompt: list[{role,content}]`), add a final reshape pass to the prep script.
2. **Test-split usage during training.** The Search-R1 paper uses these test sets for both in-loop validation *and* final eval. We split that role: 1k subsamples for in-loop val (paths in [`VALIDATION.md`](VALIDATION.md)), full sets for post-training Milestone 1-style eval. Verify the upstream verl config doesn't do anything sneakier (e.g. sampling without replacement across val rounds) when we audit the NeMo-RL port.
