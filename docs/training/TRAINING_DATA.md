# Training Data

The Search-R1 RL training corpus, lifted directly from the paper authors' HF distribution. Used for both [Milestone 2](../milestone_two/MILESTONE_2.md) variants (base + hybrid).

## Source

[`PeterJinGo/nq_hotpotqa_train`](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train) — official Search-R1 RL training mixture.

## Schema

Format: parquet (auto-converted from the upstream).

| Column | Type | Example |
|---|---|---|
| `id` | string | `train_0` |
| `question` | string | `"total number of death row inmates in the us?"` |
| `golden_answers` | list[string] | `["2,718"]` |
| `data_source` | string (one of `nq`, `hotpotqa`) | `"nq"` |
| `prompt` | list[dict] (length 1) | `[{"content": "Answer the given question. You must conduct reasoning inside <think>..."}]` |
| `ability` | string | `"fact-reasoning"` |
| `reward_model` | dict | `{"ground_truth": {"target": ["2,718"]}, "style": "rule"}` |
| `extra_info` | dict | `{"index": 0, "split": "train"}` |
| `metadata` | dict | `null` |

## Splits

| Split | Rows |
|---|---|
| `train` | 170k |
| `test` | 51.7k |

The `test` split is the in-loop validation set ([`VALIDATION.md`](VALIDATION.md)) — same as the paper.

## Quick load

```python
from datasets import load_dataset

ds = load_dataset("PeterJinGo/nq_hotpotqa_train")
print(ds)
# DatasetDict({
#   train: Dataset({features: [...], num_rows: 170000})
#   test:  Dataset({features: [...], num_rows: 51719})
# })

print(ds["train"][0])
```

## Important: prompt field already contains Search-R1's `<search>` template

The `prompt[0].content` is **literally the Search-R1 paper's instruction string** with `<think>` / `<search>` / `<information>` / `<answer>` tags baked in. Since we are using Qwen3.5's native `<tool_call>` template ([`CHAT_TEMPLATE.md`](CHAT_TEMPLATE.md)), the dataset's prompt **must be rewritten during conversion** — passing it through as-is would force the model to emit `<search>` tags despite our chat template registering `search` as a `<tool_call>` function.

## Conversion to NeMo-RL format

Milestone 2 step 3 (after step 2's download). Steps:

1. Pull the parquet from HF → land at `data/training/nq_hotpotqa_train/{train,test}.parquet`.
2. **Rewrite `prompt`** per row:
   - **For the `qwen_native` (default) chat template:** drop the dataset's `prompt[0].content` and rebuild it from `question` via the tokenizer's `apply_chat_template` with a registered `search` tool. The new prompt is the `messages` list:
     ```python
     [
       {"role": "system", "content": "You are a helpful assistant. Use the `search` tool to retrieve evidence, then place your final answer inside <answer>...</answer>."},
       {"role": "user", "content": question},
     ]
     ```
     plus the OpenAI-style tool schema:
     ```python
     tools = [{
       "type": "function",
       "function": {
         "name": "search",
         "description": "Search Wikipedia for passages relevant to the query.",
         "parameters": {
           "type": "object",
           "properties": {"query": {"type": "string"}},
           "required": ["query"],
         },
       },
     }]
     ```
   - **For the `paper` ablation arm:** pass `prompt[0].content` through unchanged.
3. **Preserve `golden_answers`, `data_source`, `reward_model.ground_truth.target`, `extra_info`** — these drive the EM reward and dataset bookkeeping.
4. **Map to NeMo-RL's expected schema** — TBD (see Open Questions below). NeMo-RL's `grpo_math_1B.yaml` example uses the OpenAI Math dataset with a different shape; the exact mapping requires inspecting `examples/grpo_math/data.py` (or equivalent).

## Open questions

1. **NeMo-RL's exact data schema** for GRPO. The example configs reference data sources but don't fully document the row-level shape. Check `examples/` in the NeMo-RL clone (Milestone 2 step 4) and update this doc with the verified mapping before running step 3.
2. **Streaming vs. local cache.** 170k rows is small enough to fit in memory; default to `load_dataset(..., split="train")` (full materialise) rather than streaming.
3. **Test-split usage during training.** The Search-R1 paper uses these test sets for both in-loop validation *and* final eval. We split that role: 1k subsamples for in-loop val (paths in [`VALIDATION.md`](VALIDATION.md)), full sets for post-training Milestone 1-style eval. Verify the upstream verl config doesn't do anything sneakier (e.g. sampling without replacement across val rounds) when we audit the NeMo-RL port.
