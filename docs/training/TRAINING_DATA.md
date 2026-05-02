# Training Data

The Search-R1 RL training corpus, lifted directly from the paper authors' HF distribution. Used for both [Milestone 2](../milestone_two/MILESTONE_2.md) variants (base + hybrid).

## Source

[`PeterJinGo/nq_hotpotqa_train`](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train) — official Search-R1 RL training mixture.

## Schema

Format: parquet (auto-converted from the upstream).

After our prep script reshapes upstream into the NeMo-RL row schema:

| Column | Type | Example |
|---|---|---|
| `id` | string | `train_0` |
| `question` | string (already `.strip() + '?'` normalized upstream) | `"total number of death row inmates in the us?"` |
| `golden_answers` | list[string] | `["2,718"]` |
| `data_source` | string (one of `nq`, `hotpotqa`) | `"nq"` |
| `messages` | list[struct{role,content}] (length 1) | `[{"role": "user", "content": question}]` |
| `ability` | string | `"fact-reasoning"` |
| `reward_model` | struct | `{"ground_truth": {"target": ["2,718"]}, "style": "rule"}` |
| `extra_info` | struct | NQ + HotpotQA carry different per-row shapes; preserved verbatim |
| `metadata` | struct | `null` for NQ rows; HotpotQA carries `supporting_facts` + `context` paragraphs |

> Upstream's `prompt` column is dropped (redundant with `messages` after the strip). Upstream's HF schema for the `prompt` column (`list<struct<content,role>>`) is reused verbatim for `messages`.

## Splits

| Split | Rows |
|---|---|
| `train` | 169,615 |
| `test` | 51,713 |

The `test` split is the in-loop validation set ([`VALIDATION.md`](VALIDATION.md)) — same as the paper.

## Download + prep

[`training/scripts/prepare_dataset.py`](../../training/scripts/prepare_dataset.py) — uv inline-script, idempotent. Pulls the upstream parquets via `hf_hub_download`, applies two transforms in one pass, writes to `data/training/nq_hotpotqa_train/{train,test}.parquet`:

1. **Strip the prebaked Search-R1 template** — `prompt[0].content := question` (template-agnostic; runtime applies the chat template).
2. **Rename `prompt` → `messages`** — match NeMo-RL's `ResponseDataset` row schema.

Output is committed via Git LFS ([`.gitattributes`](../../.gitattributes) `data/**/*.parquet`). Re-runs skip unless `--force`.

```bash
training/scripts/prepare_dataset.py
```

> **Why `huggingface_hub` + pyarrow, not `datasets.load_dataset`:** the upstream parquet has heterogeneous `extra_info`/`metadata` schemas across the NQ + HotpotQA mixture, so the `datasets` library's cross-row schema unification fails with a cast error. Reading the parquet files directly bypasses this entirely.

## Why we strip the prebaked template

Upstream `make_prefix` ([`Search-R1/scripts/data_process/qa_search_test_merge.py:26-39`](https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/data_process/qa_search_test_merge.py)) bakes the paper's full `<think>` / `<search>` / `<information>` / `<answer>` instruction string into `prompt[0].content` before publishing. Keeping that text would (a) force the rollout to emit `<search>` tags regardless of which chat template the run config selects, and (b) make a per-arm dataset re-conversion necessary to swap templates.

Stripping to the bare question keeps the dataset **template-agnostic**. The chat template — Qwen3.5 native `<tool_call>` (default) or the paper's `<search>` (ablation arm) — is applied at **rollout time**, not at dataset prep time. See [`CHAT_TEMPLATE.md`](CHAT_TEMPLATE.md) §5.

## Mapping to NeMo-RL's row schema

Verified against the vendored NeMo-RL @ v0.6.0 source.

**Routing rule** ([`response_dataset.py:64`](../../training/nemo_rl/nemo_rl/data/datasets/response_datasets/response_dataset.py#L64)): `ResponseDataset` checks for a `messages` column. If present, it just adds `task_name` (preserving every other column); if absent, it re-maps from `input_key` / `output_key` and **drops everything else**. We need the preserve branch so `golden_answers`, `data_source`, etc. survive into the processor.

**Length-1 messages** is intentional. Math-side processors ([`processors.py:454-456`](../../training/nemo_rl/nemo_rl/data/processors.py#L454-L456)) read `messages[0].content` for the user prompt and `messages[1].content` for the ground truth. Search-R1 has no reference rollout — only a list of gold strings — and packing `list[str]` into a single message content would force JSON-encoding round-trips. The custom processor we'll write in M2 step 4 reads `messages[0].content` for the question and pulls `golden_answers` from its own column, so a single user message is enough.

**No pre-baked `task_name`** — `add_column` in [`response_dataset.py:71`](../../training/nemo_rl/nemo_rl/data/datasets/response_datasets/response_dataset.py#L71) errors on column-name conflicts. The M2-step-4 dataset adapter sets `self.task_name = "search_r1"` so the processor registry routes our env correctly.

**Chat template stays at runtime.** `math_hf_data_processor` ([`processors.py:467-477`](../../training/nemo_rl/nemo_rl/data/processors.py#L467-L477)) reads `messages[0].content`, wraps it via `task_data_spec.prompt.format(...)` from the configured `prompt_file`, prepends an optional `system_prompt`, and finally renders through `tokenizer.apply_chat_template`. Our custom processor will follow the same pattern, with two arms selected by config:

- **`qwen_native`**: no `prompt_file`; pass `tools=[search]` to `apply_chat_template`. Qwen3.5's chat template emits the `<tool_call>` / `<tool_response>` XML format from §2 of [`CHAT_TEMPLATE.md`](CHAT_TEMPLATE.md).
- **`paper`**: `prompt_file = training/src/prompts/search_r1.txt` (the paper's instruction string); no tool registration.

## Open questions

1. **Test-split usage during training.** The Search-R1 paper uses these test sets for both in-loop validation *and* final eval. We split that role: 1k subsamples for in-loop val (paths in [`VALIDATION.md`](VALIDATION.md)), full sets for post-training Milestone 1-style eval. Verify the upstream verl config doesn't do anything sneakier (e.g. sampling without replacement across val rounds) when we audit the NeMo-RL port.
