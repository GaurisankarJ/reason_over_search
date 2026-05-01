# Training docs

This directory documents the training-side reproduction of [Search-R1](https://www.alphaxiv.org/abs/2503.09516) on **Qwen3.5-2B** via [NeMo-RL](https://github.com/NVIDIA-NeMo/RL). Owned by [Milestone 2](../milestone_two/MILESTONE_2.md).

> Code lives at [`training/`](../../training/); these docs explain the *why*.

---

## Index

| File | What it covers | Read when |
|---|---|---|
| [TRAINING_DATA.md](TRAINING_DATA.md) | `PeterJinGo/nq_hotpotqa_train` schema, our reshape (strip prebaked template, `prompt → messages`), NeMo-RL row-schema mapping | working on the dataset prep script or wondering about `messages[0].content` |
| [CHAT_TEMPLATE.md](CHAT_TEMPLATE.md) | Two chat-template arms — Qwen3.5 native `<tool_call>` (default) vs. paper's `<search>` (ablation) — with verbatim Qwen3.5 jinja | wiring the tokenizer / prompt; choosing an arm |
| [PAPER_VS_OURS_TRAINING.md](PAPER_VS_OURS_TRAINING.md) | The canonical hyperparameter audit — every knob in our run vs. the upstream verl yaml that produced the paper's published checkpoints | before launching a training run |
| [VERL_REFERENCE.md](VERL_REFERENCE.md) | verl-side references porting to NeMo-RL: HTTP retriever contract, KL/GRPO mappings, FSDP→DTensor translations, JSONL log shape to mirror | reading verl scripts; resolving a NeMo-RL config knob |
| [VALIDATION.md](VALIDATION.md) | In-loop validation plan — datasets, cadence, metrics, sampling | wiring W&B; deciding val_period |
| [NEMO_RL_KNOBS.md](NEMO_RL_KNOBS.md) | Memory + throughput + algorithm knobs with our recommended starting values for A100 80GB; concrete starting yaml | tuning a config |

---

## End-to-end view

```
                  PeterJinGo/nq_hotpotqa_train (HF)
                          │
                          ▼  prepare_dataset.py
        data/training/nq_hotpotqa_train/{train,test}.parquet  (Git LFS)
                          │
                          ▼  SearchR1Dataset    (training/src/datasets/)
                  task_name = search_r1_{arm}
                          │
                          ▼  search_r1_processor  (training/src/processors/)
            apply_chat_template + extract golden_answers
                          │
                          ▼  GRPO rollout loop  (vendored NeMo-RL)
                  policy generates  ──→  SearchR1Environment
                                          │   (training/src/environments/)
                                          │
                          ┌───────────────┴──────────────┐
                          ▼                              ▼
                  parse_query() per arm             extract_solution()
                          │                              │
                          ▼                              ▼
                  POST /batch_search                EM via reward
                  (local_retriever:3005)            (training/src/rewards/)
                          │                              │
                          ▼                              ▼
                  format_docs_*  ──→  observation     reward + terminated
                          │                              │
                          └───────────────┬──────────────┘
                                          ▼
                             next-turn generation
```

Each layer is decoupled — chat template lives in the processor + env, reward lives in `rewards/`, retrieval contract lives in the env. Swapping arms is a config flip; swapping the reward (M3 ablation) is a one-file change.

---

## Step-5 audit summary

Lock-in mapping between [`Search-R1/scripts/nq_hotpotqa/v0.2/train_grpo.sh`](https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/nq_hotpotqa/v0.2/train_grpo.sh) (the EM-only baseline that produced the published GRPO checkpoints we evaluated in M1) and our NeMo-RL setup.

| Knob | verl value (v0.2 yaml) | NeMo-RL key | Our value | Status |
|---|---|---|---|---|
| Optimizer LR | `1e-6` | `optimizer.kwargs.lr` | `1e-6` | match |
| Warmup ratio | `0.285` | `optimizer` LinearLR `total_iters` | computed from `0.285 × 1005 ≈ 286` | match |
| Total steps | `total_training_steps=1005` | `grpo.max_num_steps` | `1005` | match (paper text says 500; verl says 1005 — published checkpoints came from 1005-step runs) |
| Save cadence | `save_freq=100` | `checkpointing.save_period` | `100` | match |
| Val cadence | `test_freq=100` | `grpo.val_period` | `100` | match |
| Val at start | `val_before_train=true` | `grpo.val_at_start` | `true` | match |
| KL coef (β) | `kl_loss_coef=0.001` | `loss_fn.reference_policy_kl_penalty` | `0.001` | match |
| **KL estimator** | `kl_loss_type=low_var_kl` | `loss_fn.reference_policy_kl_type=k3` | NeMo-RL default | **byte-identical** — both compute Schulman 2020 k3 |
| **State masking** | `state_masking=true` (mask `<information>` from policy gradient) | role-based `token_loss_mask` ([grpo.py:1685-1693](../../training/nemo_rl/nemo_rl/algorithms/grpo.py#L1685-L1693)) | automatic | **equivalent, no config knob** — env emits `role: tool`, gradient zero-masks it |
| Clip ratio (ε) | `0.2` | `loss_fn.ratio_clip_min` / `ratio_clip_max` | `0.2` | match |
| Group size G | `n_agent=5` | `grpo.num_generations_per_prompt` | `5` | match |
| Train batch | `train_batch_size=512` | `policy.train_global_batch_size` | `512` | match |
| Max prompt len | `4096` | `policy.max_total_sequence_length` | `4096` | match |
| Max response len | `500` | `policy.generation.max_new_tokens` | `500` | match |
| Max search turns | `max_turns=4` | `env.search_r1.max_turns` (overlay) | `4` | match |
| Topk retrieved | `retriever.topk=3` | `env.search_r1.top_n` (overlay) | `3` | match |
| Rollout temp | `1.0` | `policy.generation.temperature` | `1.0` | match |
| Rollout top_p | `1.0` | `policy.generation.top_p` | `1.0` | match |
| **Max obs len** | `max_obs_length=500` (per-`<information>` block tokens) | `env.search_r1.max_obs_chars` (overlay; ~4 char/tok proxy) | `2000` | **char-proxy** (no tokenizer in env actor by design) |
| Reward function | EM-based (`flashrag/search_r1/reward.py`) | byte-identical port at `training/src/rewards/search_r1.py` | match | verified by 15 tests |

**Knowing divergences from the paper** (recorded in [PAPER_VS_OURS_TRAINING.md §9](PAPER_VS_OURS_TRAINING.md#9-divergence-summary-one-place-to-glance)): model family (Qwen3.5-2B vs Qwen2.5-3B), chat template (qwen_native vs paper), variant naming (hybrid vs instruct), hardware (1×–2× A100 vs 8× H100), framework (NeMo-RL vs verl). Everything else matches.

---

## training/src/ overlay architecture

NeMo-RL is vendored at [`training/nemo_rl/`](../../training/nemo_rl/) at pinned `v0.6.0`. We **never edit it directly** — Search-R1-specific behavior lives as a pure overlay under [`training/src/`](../../training/src/) that registers itself with NeMo-RL's pluggable registries at startup.

| Overlay file | Role | Plugged into |
|---|---|---|
| [`rewards/search_r1.py`](../../training/src/rewards/search_r1.py) | Byte-identical port of M1 EM scorer (`compute_search_r1_reward`, `em_check`, `extract_solution`, ...) | imported by env actor |
| [`prompts/search_r1_paper.txt`](../../training/src/prompts/search_r1_paper.txt) | Paper instruction string (`{}` placeholder) | loaded as `task_data_spec.prompt` for `paper` arm only |
| [`chat_template/tools.py`](../../training/src/chat_template/tools.py) | OpenAI-style `search` tool schema (qwen_native arm) | passed as `tools=[SEARCH_TOOL]` to `tokenizer.apply_chat_template` |
| [`datasets/search_r1.py`](../../training/src/datasets/search_r1.py) | `SearchR1Dataset(RawDataset)` — loads our parquet, sets `task_name = f"search_r1_{arm}"` | `DATASET_REGISTRY["search_r1"]` (monkey-patched — no `register_dataset` upstream) |
| [`processors/search_r1.py`](../../training/src/processors/search_r1.py) | `search_r1_processor` — reads `messages[0].content` + `golden_answers`, dispatches qwen_native vs paper arms | `register_processor("search_r1_processor", ...)` |
| [`environments/parsers.py`](../../training/src/environments/parsers.py) | Pure-Python `parse_query`, `format_docs_*`, `retriever_failed_message` (testable without torch/ray/nemo_rl) | re-exported by env |
| [`environments/search_r1_env.py`](../../training/src/environments/search_r1_env.py) | `SearchR1Env` (testable plain class) + `SearchR1Environment = ray.remote(SearchR1Env)` | `register_env("search_r1", "training.src.environments.search_r1_env.SearchR1Environment")` |
| [`registry.py`](../../training/src/registry.py) | Single import-side-effect module that populates DATASET / PROCESSOR / ENV registries idempotently | imported once by the launch script |

**Wiring contract** — the launch script does:

```python
import training.src.registry  # populates DATASET_REGISTRY, PROCESSOR_REGISTRY, ENV_REGISTRY
from examples.run_grpo import main
main()
```

After that, the training loop sees `dataset_name: search_r1`, `processor: search_r1_processor`, `env_name: search_r1` as if they were built-in.

**Tests** — [`training/tests/`](../../training/tests/) covers reward parity (15 tests, byte-identical to the M1 eval pipeline), parser dispatch (per-arm regex behavior), format helpers (Qwen3.5 chat-template marker construction), env-step against a mocked retriever (search/answer/exhausted paths), and dataset adapter (column preservation, `task_name` correctness, validation split). Pure-Python tests run anywhere; tests needing torch/ray/nemo_rl skip cleanly outside the training venv.

---

## Hardware + runtime expectations

Per [PAPER_VS_OURS_TRAINING.md §7](PAPER_VS_OURS_TRAINING.md#7-compute) — projected:
- 1× A100 80GB: ~30–50 h / run, $30–100 / run.
- 2× A100 80GB: ~18–28 h / run, $36–112 / run.
- Total Phase 2 budget for 6 runs (3 seeds × {base, hybrid}): ~$180–600 on 1× A100.

Retriever: separate process on `127.0.0.1:3005`, ~65 GB host RAM (Wiki-18 FAISS-flat). See [`local_retriever/README.md`](../../local_retriever/README.md).

Concrete configs + launch scripts: [`training/configs/`](../../training/configs/) and [`training/scripts/`](../../training/scripts/).

## Running training

The how-to-launch view:
- **Operator-side** (what to run): [`training/README.md`](../../training/README.md)
- **Phase 2 runbook** (Vast.ai sequence, smoke run, monitoring, eval gate): [`../milestone_two/PHASE_2_RUNBOOK.md`](../milestone_two/PHASE_2_RUNBOOK.md)
- **Milestone scope + status**: [`../milestone_two/MILESTONE_2.md`](../milestone_two/MILESTONE_2.md)
