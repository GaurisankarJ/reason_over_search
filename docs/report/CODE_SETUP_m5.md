---
title: Code Setup M5 — Qwen3.5-0.8B GRPO training on NeMo-RL (M5 + M5.1)
tags: [report, training, m5, m5.1, qwen3.5, nemo-rl]
source: internal
created: 2026-05-09
updated: 2026-05-09
---

# Code Setup M5: Qwen3.5-0.8B GRPO Training Pipeline (M5 + M5.1)

**Status**: spine only; sections marked **TODO** populate as the work lands.
**Date**: 2026-05-09 (M5 + M5.1 launch).
**Scope**: documents what changed from the M2 NeMo-RL training scaffold ([`training/`](../../training/), Qwen3.5-2B target, paper-default reward + tag scheme) to the M5 / M5.1 pipeline ([`training_m5_1/`](../../training_m5_1/)) for **Qwen3.5-0.8B** training with the **ReSearch-paper recipe** modulo two intentional divergences (F1-only reward, no `\boxed{}` answer wrapper). Train rollout is byte-aligned to the [M4 eval pipeline](../milestone_4/MILESTONE_4.md) so the trained checkpoint is directly evaluable without re-aligning.
**Cluster**: 1× A100-80GB on Vast.ai.
**Source paths**: [`training_m5_1/`](../../training_m5_1/) (M5 + M5.1), [`training_m5_1/configs/m5_smoke.yaml`](../../training_m5_1/configs/m5_smoke.yaml) (M5 smoke), [`training_m5_1/configs/m5_1_research_paper.yaml`](../../training_m5_1/configs/m5_1_research_paper.yaml) (M5.1 production), [`training_m5_1/scripts/`](../../training_m5_1/scripts/), milestone narrative at [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md).

---

## 1. Headline Diff vs the M2 Pipeline (`training/`)

| Dimension | M2 (`training/`, Qwen3.5-2B target, smoke-tested) | M5 / M5.1 (this doc, `training_m5_1/`) |
|---|---|---|
| **Target model** | Qwen3.5-2B | **Qwen3.5-0.8B** |
| **Training data** | Search-R1 mix (NQ + HotpotQA train splits) | **MuSiQue train split** (single-dataset; multi-hop, hardest of the four paper benchmarks) |
| **Reward** | Search-R1 `qa_em.py` EM-only (M2 default per `training/src/reward.py`) | **F1-only on `<answer>…</answer>`** (M5.1 divergence #1; paper format reward dropped) |
| **Answer wrap** | `<answer>X</answer>` (already plain in M2) | identical (M5.1 divergence #2 vs paper, which uses `\boxed{}` inside `<answer>`) |
| **Action format** | Qwen3.5 nested-XML `<tool_call>` (M2 qwen_native arm) | identical (carried from M4) |
| **Tool-response wrap** | turn-bounded `<\|im_end\|>\n<\|im_start\|>user\n<tool_response>\n…\n</tool_response><\|im_end\|>\n<\|im_start\|>assistant\n` | identical (must match M4 byte-for-byte) |
| **`max_search_turns`** | 5 | 5 |
| **`max_obs_length` (per chunk)** | 256 tokens (M2 default) | **120 tokens** (M4 v3 lock; per-chunk, no total cap) |
| **`retrieval_topk`** | 5 | 5 |
| **`generator_max_input_len`** | 4096 | 4096 (smoke) / **TODO M5.1** (paper-aligned) |
| **`enable_thinking`** | True | True |
| **GRPO group size** | TODO M5.1 (paper: G=5; ours likely match, confirm against `Agent-RL/ReSearch`) | TODO M5.1 |
| **KL coefficient** | TODO M5.1 | TODO M5.1 (paper-aligned) |
| **Schedule** | NeMo-RL `grpo.max_num_steps: 1005` (M2 v0.2-yaml-aligned, paper-extrapolated from 0.6-epoch budget) | TODO M5.1 (paper-faithful for the recipe; smoke first, then schedule) |
| **W&B project** | `reason_over_search_2b_v1` (M2 placeholder) | **`reason_over_search_m5_1`** |

---

## 2. What's Unchanged (M2 + M4 audit holds)

The M2 NeMo-RL setup decisions catalogued in [`docs/training/PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md) and [`docs/training/VERL_REFERENCE.md`](../training/VERL_REFERENCE.md) transfer to M5 unchanged:

- NeMo-RL `v0.6.0` vendored install, `uv sync --extra vllm` from the pre-warmed wheel cache in `pantomiman/reason-over-search-v1:v1`.
- `policy.generation.vllm_cfg.*` shape (`tensor_parallel_size`, `gpu_memory_utilization`, `enforce_eager`, `kv_cache_dtype`) inherits from `training/configs/grpo_qwen3.5_2b_1xa100.yaml` adapted for 0.8B (smaller model → can raise `gpu_memory_utilization` and `max_num_seqs`).
- `kl_type=k3` (Schulman 2020) — NeMo-RL default matches verl `low_var_kl`.
- State masking — NeMo-RL `token_loss_mask` via role-based masking; tool-response role ≠ "assistant" → loss=0 on retrieved docs (paper equivalent of verl `state_masking=true`).
- Retriever HTTP contract: same as M2 / M4 (POST `/batch_search` with `{queries, topk, return_scores}`; CPU FAISS IVF-SQ8 default, GPU FAISS opt-in).
- Tokenizer / chat template invocation — identical to M4 ([`evaluation_qwen35/flashrag/search_r1/templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py)); rendered-prompt byte-check is part of the M5 smoke deliverables.

---

## 3. M5 Critical Changes (smoke pipeline)

### 3.1 Folder layout (committed 2026-05-09)

`training_m5_1/` is a self-contained copy of `training/` (motivation in [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md) §"Folder layout"). Future experiments are sibling dirs `training_m5_2/`, `training_m5_3/`, …; no shared `src/` or `configs/` between experiments.

```text
training_m5_1/
  nemo_rl/                            # SYMLINK -> ../training/nemo_rl/  (disk: 23 G; symlink until disk allows a real copy)
  src/                                # overlay; copied from training/src/, then 14 import sites + several files edited
    rewards/search_r1.py              # CHANGED: F1-only on <answer>...</answer> (re-exports normalize_answer + extract_solution from evaluation_qwen35)
    environments/parsers.py           # CHANGED: qwen_native parse_query delegates to evaluation_qwen35 extract_tool_call_query
    environments/search_r1_env.py     # CHANGED: docstring rewrite, em_check fallback -> f1_check, em_hit_rate -> near_em_rate, max_chars kwarg fix
    datasets/search_r1.py             # unchanged structure; data_path now points at MuSiQue parquet
    processors/search_r1.py           # docstring comment updated for MuSiQue
    chat_template/tools.py            # CHANGED: re-export QWEN35_SEARCH_TOOL from evaluation_qwen35
    prompts/m5_qwen35_user.txt        # NEW: pre-staged from M4.2 canonical (qwen35_minimal); regenerate via sync_m4_prompts.py
    prompts/_archive_m2/              # M2 prompt files archived here
    registry.py                       # CHANGED: all imports renamed training. -> training_m5_1.
  configs/
    m5_smoke.yaml                     # NEW: 20 traj/step, 50 steps, MuSiQue, qwen_native, 1xA100
    m5_1_research_paper.yaml          # TODO (M5.1 step 6): paper-faithful production config
    _archive_m2/                      # grpo_qwen3.5_2b_{1,2}xa100.yaml archived here
  scripts/
    run.sh                            # NEW: --mode smoke|prod, --seed N, [-- extra hydra overrides]
    smoke.sh                          # NEW: thin alias for `run.sh --mode smoke`
    run_grpo.py                       # CHANGED: REPO_ROOT/training_m5_1/nemo_rl path; import training_m5_1.src.registry
    prep_musique.py                   # NEW: downloads RUC-NLPIR/FlashRAG_datasets musique/train.jsonl -> data/training/musique/train.parquet
    sync_m4_prompts.py                # NEW: materialises QWEN35_TEMPLATES[mode] into src/prompts/m5_*.txt
    bootstrap.sh, bootstrap_alice.sh  # still M2-shaped; rewrite when M5.1 launches on a fresh Vast box
    _archive_m2/                      # run_grpo_{1,2}xa100.sh + prepare_dataset.py archived here
  tests/
    test_reward_parity.py             # CHANGED: rewritten for F1 semantics; M2 byte-parity assertion dropped
    test_parser_dispatch.py           # unchanged (still pinning qwen_native + paper arm regex behaviour)
    test_format_helpers.py            # unchanged
    test_env_step.py, test_dataset_adapter.py  # unchanged
  setup.sh                            # mirror of training/setup.sh; comments still mention training/ (cosmetic)
  README.md                           # CHANGED: M5.1-specific narrative + run sequence
```

### 3.2 Overlay deltas vs `training/src/` (M5; status 2026-05-10)

| File (M5 layout) | M2 (`training/src/`) | M5 (`training_m5_1/src/`) | Status | Why |
|---|---|---|---|---|
| `rewards/search_r1.py` | EM-with-shaping (Search-R1 `qa_em.py` port) | F1-only on `<answer>...</answer>`; re-exports `normalize_answer` + `extract_solution` from `flashrag.search_r1.reward`; adds `f1_check` | **done** | M5.1 divergence #1; same scorer code path as M4 eval |
| `environments/parsers.py` | local `_RE_QWEN_QUERY` regex | qwen_native arm delegates to `flashrag.search_r1.parser.extract_tool_call_query`; paper arm regex kept local | **done** | Single source of truth for the action parser (M3 14-fix precedent) |
| `environments/search_r1_env.py` | turn-bounded `<tool_response>` wrap; `em_check` fallback; `em_hit_rate` metric | identical wrap + byte-for-byte M4 alignment; `f1_check` fallback under truncation; `near_em_rate` metric (F1 ≥ 0.8); fixed `max_chars_per_chunk` kwarg dispatch bug (was `max_chars=`, would TypeError on first qwen_native search turn) | **done** | Hard alignment + M5.1 reward consistency + latent bug fix |
| `chat_template/tools.py` | local `SEARCH_TOOL` dict | re-export `QWEN35_SEARCH_TOOL` (aliased as `SEARCH_TOOL`) from `flashrag.search_r1.templates` | **done** | Locks training schema = eval schema |
| `prompts/m5_qwen35_user.txt` | n/a (M2 used `search_r1_qwen_native_user.txt`) | pre-staged from M4.2 canonical (`qwen35_minimal`); written by `scripts/sync_m4_prompts.py --mode <key>` | **done; re-run after M4.4 lock** | Dynamic sync avoids hard-coding the prompt text in training overlay |
| `datasets/search_r1.py` | reads NQ + HotpotQA parquet | unchanged code; new `data_path: data/training/musique/train.parquet` in m5_smoke.yaml | **done** (dataset-agnostic adapter; only the config path changes) | M5.1 dataset choice (MuSiQue only) |
| `processors/search_r1.py` | docstring mentions NQ+HotpotQA only | docstring updated to mention both NQ+HotpotQA (M2) and MuSiQue (M5.1) | **done** (no functional change; `data_source` field optional) | Documentation only |
| `registry.py` | `from training.src...` imports | `from training_m5_1.src...` (14 import sites renamed across src/, tests/, scripts/) | **done** | Package isolation between sibling experiments |
| **Tests** | M2 byte-parity assertions | rewritten `test_reward_parity.py` with re-export identity checks + F1 semantics + 5 hand-picked rollouts (milestone doc M5.1 step 6 requirement) | **done** (23 pure-Python tests pass) | M5.1 invariants |

### 3.3 Smoke config (`configs/m5_smoke.yaml`) — TODO

Smoke goal: validate the pipeline end-to-end and produce a per-step time number on 1× A100-80GB.

| Knob | Smoke value | Production (M5.1) | Notes |
|---|---|---|---|
| `policy.model_name` | `Qwen/Qwen3.5-0.8B` | same | hybrid; base variant is a separate sub-run |
| `policy.train_global_batch_size` | TODO | TODO | M5.1 paper-aligned |
| `policy.train_micro_batch_size` | TODO (likely 4 — bigger than M2's 2 since 0.8B is smaller) | TODO | OOM check at smoke |
| `grpo.num_prompts_per_step` | 20 (smoke) | TODO M5.1 | paper: 512 prompts × 5 generations = 2560 traj/step; ours target ≤ 510 / step (per `BATCH_MATH.md`) |
| `grpo.num_generations_per_prompt` | 5 | 5 | paper-aligned |
| `grpo.max_num_steps` | 50 (smoke) | TODO M5.1 | from paper schedule |
| `policy.generation.vllm_cfg.max_model_len` | 4096 | TODO | smoke matches M4 |
| `policy.generation.max_response_length` | 4096 | TODO M5.1 | paper: 500 |
| `data.dataset_name` | `musique` | `musique` | single-dataset training |
| `data.train_dataset.subsample` | 200 (smoke) | full | smoke is fast |
| `logger.wandb.project` | `reason_over_search_m5_1_smoke` | `reason_over_search_m5_1` | per-experiment isolation |
| `validation.enabled` | False (smoke) | True (M5.1, `val_period: 50` per [`docs/training/VALIDATION.md`](../training/VALIDATION.md)) | wire validation before M5.1 launches |

### 3.4 Verification (M5)

Two byte-level checks before declaring M5 smoke green:

1. **Rendered prompt parity**. On 1 hand-picked MuSiQue question, capture `tokenizer.apply_chat_template(...)` output from (a) `training_m5_1/src/retrieval_env.py:render_initial_prompt` and (b) `evaluation_qwen35/flashrag/pipeline/active_pipeline.py` qwen35-branch first-turn prompt. `diff -u` must be empty.
2. **Reward path on 5 hand-picked rollouts**:
   - "Right entity, exact match" → F1 = 1.0
   - "Right entity, extra words" → 0 < F1 < 1
   - "Wrong entity" → F1 = 0
   - "Empty `<answer>`" → F1 = 0
   - "Format-broken (no `<answer>` tag)" → F1 = 0 (no format penalty added — this is the M5.1 divergence #1)

### 3.5 What gets logged (M5 smoke) — Partial (2026-05-10)

**Step 1 ground truth from the 4th smoke attempt** ([wandb run `jeedpsjq` in project `reason_over_search_m5_1`](https://wandb.ai/gaurisankarj1996-leiden-university/reason_over_search_m5_1/runs/jeedpsjq)):
- Setup wall-clock: **73.0 s** (vLLM init 44.1 s + Policy init 9.0 s + other 15.7 s; both worker venvs reused from disk cache)
- **Step 1 wall-clock: 145.58 s** at 20 trajectories (4 prompts × 5 group)
- Step 1 rollout produced a 5-turn dialogue on most prompts (env.step exercised the full search→retrieve→continue→answer path)

**Step 2 OOM'd**: `log_softmax` over 248,320 vocab needed 15.15 GB, only 14.84 GB free; off by 0.31 GB. Caused by vLLM's sleep-mode resident footprint (6.74 GB) + PyTorch allocator fragmentation. Fixed by:
- `train_micro_batch_size: 4 → 2` (halves the log_softmax tensor per microbatch)
- `gpu_memory_utilization: 0.7 → 0.5` (vLLM gives back more memory on sleep)
- (`PYTORCH_ALLOC_CONF=expandable_segments:True` was tried but breaks NeMo-RL's CUDA IPC weight sharing — rejected)

**Group C resolutions applied to m5_smoke.yaml** (see [`../milestone_5/PAPER_VS_OURS_M5.md §8`](../milestone_5/PAPER_VS_OURS_M5.md) for the locked decisions):
- `use_leave_one_out_baseline: true → false` (paper uses group-mean)
- `max_new_tokens: 500 → 1024` (paper budget is 8192 total)
- `max_obs_chars: 480 → 1024` (paper has no per-obs cap)
- `lr_warmup: 14 steps → 0` (paper uses no warmup)
- `max_num_steps: 50 → 10` (smoke validation only needs ~5 stable steps)

**Remaining smoke deliverables** (from the next launch with the above fixes):
- per-step wall-clock mean / p50 / p95 over steps 1-10
- reward/mean trajectory + `near_em_rate` (F1 ≥ 0.8 rate)
- generation-token-count distribution
- `tool_call_counts/mean` per rollout
- gradient-norm + clip-ratio trajectories

---

## 4. M5.1 Critical Changes (ReSearch-paper-aligned config) — TODO

This section populates from the paper-vs-ours mapping work; spine here.

### 4.1 Paper-vs-ours mapping (PAPER_VS_OURS_M5.md) — TODO

Companion file at [`../milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md) (to be written): clause-by-clause mapping of every concrete number in the [ReSearch paper](https://arxiv.org/abs/2503.19470) and every concrete config value in [`Agent-RL/ReSearch`](https://github.com/Agent-RL/ReSearch) to our `configs/m5_1_research_paper.yaml`. Two divergences flagged in red:

- **Reward**: paper = format reward + EM partial credit; ours = F1-only on `<answer>…</answer>` content.
- **Answer wrap**: paper = `<answer>The final answer is \[ \boxed{X} \]</answer>`; ours = `<answer>X</answer>`.

### 4.2 Hyperparameters — TODO (after paper read)

| Knob | Paper | Ours (`m5_1_research_paper.yaml`) | Note |
|---|---|---|---|
| Algorithm | GRPO | GRPO | identical |
| KL type | k1 / k2 / k3 — TBD from paper | k3 (NeMo-RL default; matches verl `low_var_kl`) | confirm paper choice |
| KL coefficient | TBD | TBD | confirm paper |
| Group size G | TBD (likely 5) | TBD | from `Agent-RL/ReSearch` config |
| Prompts per step | TBD | TBD (≤ 510 / step on 1× A100; see [`docs/edu/BATCH_MATH.md`](../edu/BATCH_MATH.md)) | rollout-shape decision |
| Mini-batch / micro-batch | TBD | TBD | OOM-check at smoke |
| Max response length | TBD (paper: 500 in main script) | TBD | M4 eval has 4096 cap; mismatch is acceptable on the rollout side if response truncation is rare |
| Learning rate | TBD | TBD | from paper |
| Warmup | TBD | TBD | from paper |
| Schedule | TBD steps | TBD | paper-faithful |

### 4.3 Reward implementation — TODO

`training_m5_1/src/reward.py` exposes `compute_reward(rollout: dict) -> float`. Sketch:

```python
def compute_reward(rollout):
    pred = extract_solution(rollout["response"])    # first <answer>…</answer> block
    if pred is None:
        return 0.0                                   # no format penalty (M5.1 divergence #1)
    return f1_score(pred, rollout["gold_answer"])   # token-level F1, lowercase + strip + punct-norm
```

`extract_solution` and `f1_score` re-export from `evaluation_qwen35/flashrag/search_r1/answer_utils.py` so train and eval use the **same scorer code path**. Sanity-check on 5 hand-picked rollouts (§3.4 above).

### 4.4 Verification (M5.1) — TODO

Repeat §3.4 after the M5.1 config lands. Add:

3. **Wall-clock projection**: per-step time × paper schedule → total hours. Compare to [`docs/TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md) ≤10 h target. If over, identify the cheapest knob to cut (typically `max_response_length` or `num_prompts_per_step`) and re-smoke.

---

## 5. M5 Run sequence

```bash
# Bootstrap fresh Vast.ai instance per docs/vast/SETUP_VAST.md
cd /workspace/reason_over_search
cd training_m5_1
bash setup.sh                           # uv sync --extra vllm against vendored nemo_rl/

# M5 smoke (50 steps, MuSiQue 200-row subsample, 1× A100-80GB)
bash scripts/smoke.sh                   # ≤30 min wall, target ≤25 s/step

# M5.1 smoke at production shape (~100 steps)
CONFIG=configs/m5_1_research_paper.yaml bash scripts/smoke.sh

# M5.1 full run
CONFIG=configs/m5_1_research_paper.yaml bash scripts/run.sh
```

Per-experiment W&B project naming: `reason_over_search_m5_<N>` (smoke uses `…_smoke` suffix). Avoids cross-experiment metric pollution.

---

## 6. Pointers

- M5 milestone narrative: [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md)
- M5 / M5.1 smoke results: [`RESULTS_SMOKE_m5.md`](RESULTS_SMOKE_m5.md)
- M5.1 paper-vs-ours mapping (companion to this doc): [`../milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md) (TODO)
- M2 NeMo-RL paper-to-NeMo mapping (the foundation): [`../training/PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md), [`../training/VERL_REFERENCE.md`](../training/VERL_REFERENCE.md), [`../training/NEMO_RL_KNOBS.md`](../training/NEMO_RL_KNOBS.md)
- M2 smoke (the wall-clock anchor): [`../training/SMOKE_RESULTS_2026-05-06.md`](../training/SMOKE_RESULTS_2026-05-06.md)
- M4 eval pipeline (the rollout-shape source of truth): [`CODE_SETUP_m4.md`](CODE_SETUP_m4.md), [`../milestone_4/MILESTONE_4.md`](../milestone_4/MILESTONE_4.md)
- ReSearch paper: [arXiv:2503.19470](https://arxiv.org/abs/2503.19470), notes at [`../papers/2503.19470_research.md`](../papers/2503.19470_research.md)
- ReSearch official codebase: [`Agent-RL/ReSearch`](https://github.com/Agent-RL/ReSearch)
- Active recipe-ablation plan: [`../TODO_2026-05-04.md`](../todo/TODO_2026-05-04.md)
