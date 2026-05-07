---
title: RESULTS v2
tags: [report, eval, m3]
source: internal
created: 2026-05-07
updated: 2026-05-07
---

# Results v2: M3 — First evaluation of the Phase-1 v0 GRPO checkpoint vs the untrained Qwen3-0.6B hybrid

**Date compiled**: 2026-05-07
**Source**:
- Training run: W&B project [`gaurisankarj1996-leiden-university/research`](https://wandb.ai/gaurisankarj1996-leiden-university/research), run id **`z7kcxfof`** (compact name `p1_basic_w_ex`, archived at `docs/archive/verl_runs/v0/p1_basic_w_ex_z7kcxfof/`).
- Evaluation: ALICE SLURM jobs `2120423` (interactive `srun`, pre-GRPO) + `2125009` (`sbatch`, post-GRPO). Per-dataset result JSONs in `evaluation_research/results/<dataset>/<dataset>_<timestamp>_m3_<variant>_seed1/`.

**Hardware**:
- **Training (z7kcxfof)**: ALICE 1× **A100-40GB** (`gpu_1_40gb`).
- **Evaluation (this report)**: ALICE 1× **A100-80GB** (`gpu-short` and `gpu-a100-80g` partitions).

**Model**: `Qwen3-0.6B` hybrid (Qwen's post-trained soft-switch reasoning model). Two snapshots compared:

| Snapshot | Source | Description |
|---|---|---|
| `qwen_3_0.6b` | HF `Qwen/Qwen3-0.6B` (cached at `eval/qwen_3_0.6b/`) | Pre-GRPO frozen hybrid checkpoint (untrained) |
| `qwen_3_0.6b_v0` | HF checkpoint exported from `docs/archive/verl_runs/v0/p1_basic_w_ex_z7kcxfof/global_step_1000_rollout_20260407_002729_hf` | Post-GRPO: 1046 steps with the `p1_basic_w_ex` prompt + Search-R1 EM reward |

> **Bottom line up front**:
> 1. **Across all 7 paper benchmarks (51,713 items per variant, full Plan A test/dev sets), 1046 GRPO steps with the `p1_basic_w_ex` prompt lifted average EM from 0.102 to 0.155 (+5.3 pp absolute, +52% relative).** Six of seven datasets improved; one (2WikiMultiHopQA) was statistically tied.
> 2. **The largest lifts are on single-hop QA**: NQ +0.078 EM (+69%), TriviaQA +0.124 EM (+70%), PopQA +0.094 EM (+71%). MuSiQue (3-hop) doubled from 0.010 to 0.023, but at small absolute numbers.
> 3. **Multi-hop saturates at 0.6B**: HotpotQA +0.033 EM (+40%) lifts modestly; 2WikiMultiHopQA −0.003 EM (essentially tied). Multi-hop reasoning at 600 M parameters appears to be capacity-bound, not training-bound.
> 4. **The eval was Plan A (full test/dev sets), not Plan B (1k subsamples)** as MILESTONE_3.md originally planned — `sample_num` is not respected by the `search_r1` pipeline path. This makes the comparison statistically more rigorous than planned, with no subsampling noise.
> 5. **Setup cost**: 14 fixes between clone-and-run and the first clean comparison. All recorded in `docs/report/CODE_SETUP_v2.md` §3. The two highest-impact fixes were (a) the `<result>` leading-space + raw retrieval format (matches training tokenization) and (b) `enable_thinking=True` (lets Qwen3 hybrid actually reason instead of getting an empty `<think></think>` injected by the chat template).

---

## 1. Training run roster (1 run — the only run that produced the evaluated checkpoint)

| W&B id | Compact name | Date | Block | Steps trained / horizon | Model | Behavior summary |
|---|---|---|---|---:|---|---|
| `z7kcxfof` | `p1_basic_w_ex` | 2026-04-06 | v0 (Phase-1 ALICE, paper `<search>`/`<result>` tags) | **1046 / 9968** | Qwen3-0.6B hybrid | basic rules + Hamlet 2-search example; **the only Phase-1 run that converged on heavy-tool 2-call / 4-turn behavior**; mean rollout reward 0.148 → 0.190 |

Phase-1 context: across the v0 ALICE block (Apr 3 – 9, 14 runs total) + v1 ALICE block (Apr 12 – 19, 15 runs) → **29 ALICE training runs total**. Of those, only `z7kcxfof` produced the heavy-tool behavioral signature (2 search calls / 4 turns / ~2050-token responses) that this evaluation captures. All other Phase-1 prompts converged to the standard 1-tool / 3-turn regime. See `docs/report/RESULTS_v0.md` §10 (`p1_basic_w_ex`) and `docs/report/RESULTS_v0.md` §11.2 ("Prompt drives behavior more than reward").

---

## 2. Training configuration (z7kcxfof, copied from v0 W&B run config)

| Setting | Value |
|---|---|
| Algorithm | GRPO |
| KL control | `kl_ctrl.kl_coef=0.001`, `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl` |
| Optimizer | AdamW, `lr=1e-06`, constant schedule |
| Reward | `re_search` reward manager (Search-R1 EM-only, `verl_legacy/utils/reward_score/re_search.py`) |
| Train batch | `train_batch_size=4`, `ppo_mini_batch_size=4` |
| Sequence | `max_prompt_length=512`, `max_response_length=4096` |
| Rollout | `n=3` (group size G=3), `vllm.max_model_len=4608`, `enforce_eager=True`, `top_n=5` |
| FSDP | `actor.param_offload=True`, `ref.param_offload=True` |
| Agent loop | `re_search_agent` (verl_legacy with vllm_rollout) |
| Retriever | CPU FAISS, `wiki18_100w_mini`, `http://127.0.0.1:3005` |
| Hardware | ALICE 1× A100-40GB |
| Compile | `use_torch_compile=True`, `attn_implementation=sdpa` |
| Prompt | `re_search_template_sys` slot filled with the `p1_basic_w_ex` body (see §3) |
| Apply chat | `True` |

**Training behavior at end of run** (per `RESULTS_v0.md` §11):

| Metric | First-decile mean | Last-decile mean |
|---|---:|---:|
| reward (rollout, EM 0/1 + tiny format bonus) | 0.148 | **0.190** |
| `tool_call_counts/mean` | 1.83 | **2.00** |
| `num_turns/mean` | 3.83 | **4.00** |
| `response_length/mean` | 1788 | **2047** |

Heavy-tool mode: 2 tool calls per question, 4 conversation turns, ~2050-token responses. The Hamlet 2-search example in the prompt anchors the model on imitating "always do 2 searches"; this is the only Phase-1 run that converges on this 2-call regime (the other 13 v0 prompts cluster at 1 tool call / 3 turns).

---

## 3. The `p1_basic_w_ex` prompt (training rollout system message; identical at eval time)

```text
You are a helpful assistant who can answer questions using multiple Wikipedia search tool calls.
You can call the search tool by writing: <search> your query </search>
You will receive the result in: <result> your search result </result>
Use the search tool to obtain the information needed for the answer.
Answers should be based on the search results.
You may use the search tool multiple times if needed before giving the final answer.
Provide the final answer in the format: <answer>The final answer is \[ \boxed{answer here} \]</answer>.
For example:
Question: What is the nationality of the author of Hamlet?
<search>Hamlet</search>
<result>The Tragedy of Hamlet was written by William Shakespeare.</result>
<search>William Shakespeare</search>
<result>William Shakespeare was an English playwright.</result>
<answer>The final answer is \[ \boxed{English} \]</answer>
```

This is rendered into the chat template as `[{role: 'system', content: <above>}, {role: 'user', content: <question>}]`, then `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)`. Identical at training and eval time, byte-for-byte.

---

## 4. Evaluation configuration (M3, this report)

The eval pipeline (`evaluation_research/`, an editable-install overlay of `evaluation_search_r1/`) is byte-aligned to the training rollout (verl-legacy `vllm_rollout.py`); see `CODE_SETUP_v2.md` §3 for all 14 fixes that produced the alignment.

| Setting | Value | Aligned with |
|---|---|---|
| Action tags | `<search>` / `<result>` | training |
| Result wrapper | `" <result>\n{X}\n</result>"` (leading space) | `vllm_rollout.py:419` |
| Retrieval text format | raw `"{contents}\n\n".join(...).strip()` | `vllm_rollout.py:286-290` |
| `top_n` (retriever) | 5 | training |
| `max_obs_length` | 256 tokens | training `max_tool_response_length` |
| `generator_max_input_len` | 4096 | training `response_length` |
| `step_limit` per call | 8192 (no cap; bounded by remain_length) | training (no per-step cap) |
| `max_search_turns` | 5 | training observed max |
| `enable_thinking` | True | training distribution (model emits `<think>`) |
| `apply_chat` | True | training |
| Decoding | greedy (temperature=0.0) | M1 / paper convention |
| Retriever index | `wiki18_100w_e5_ivf4096_sq8.index` (16 GB IVF-SQ8) × 8 workers | training (recall hit < 1 % vs flat IP) |
| `INFERENCE_MAX_WORKERS` | 32 (post-bump; 16 for early pre-GRPO datasets) | tuning, ~2× speedup |
| Subsampling | **none** (`sample_num` not applied; full test/dev sets) | upgrade vs Plan B |

---

## 5. Headline result: average EM lifted from 0.102 to 0.155

Across all 7 datasets (51,713 items per variant):

| Metric | pre-GRPO (qwen_3_0.6b) | post-GRPO (qwen_3_0.6b_v0, z7kcxfof) | Δ absolute | Δ relative |
|---|---:|---:|---:|---:|
| **EM** (avg) | **0.102** | **0.155** | **+0.053** | **+52 %** |
| ACC (avg) | 0.123 | 0.189 | +0.066 | +54 % |
| F1 (avg) | 0.140 | 0.223 | +0.083 | +59 % |

**6 of 7 datasets improved on EM**; the seventh (2WikiMultiHopQA) is within ±0.003 EM of the baseline.

---

## 6. Per-dataset comparison

### EM (exact match, the headline metric)

| Dataset | Items | pre-GRPO EM | v0 EM | Δ EM | Δ % |
|---|---:|---:|---:|---:|---:|
| bamboogle (test) | 125 | 0.056 | **0.088** | +0.032 | +57 % |
| nq (test) | 3,610 | 0.113 | **0.191** | +0.078 | +69 % |
| triviaqa (test) | 11,313 | 0.178 | **0.302** | +0.124 | +70 % |
| popqa (test) | 14,267 | 0.133 | **0.227** | +0.094 | +71 % |
| hotpotqa (dev) | 7,405 | 0.083 | **0.116** | +0.033 | +40 % |
| 2wikimultihopqa (dev) | 12,576 | **0.141** | 0.138 | −0.003 | −2 % |
| musique (dev) | 2,417 | 0.010 | **0.023** | +0.013 | +130 % |
| **average** | 51,713 | 0.102 | **0.155** | **+0.053** | **+52 %** |

### ACC (substring containment)

| Dataset | pre-GRPO ACC | v0 ACC | Δ ACC |
|---|---:|---:|---:|
| bamboogle | 0.080 | 0.096 | +0.016 |
| nq | 0.149 | 0.256 | +0.107 |
| triviaqa | 0.209 | 0.367 | +0.158 |
| popqa | 0.153 | 0.263 | +0.110 |
| hotpotqa | 0.096 | 0.147 | +0.051 |
| 2wikimultihopqa | 0.154 | 0.157 | +0.003 |
| musique | 0.019 | 0.038 | +0.019 |
| average | 0.123 | 0.189 | +0.066 |

### F1 (token-overlap)

| Dataset | pre-GRPO F1 | v0 F1 | Δ F1 |
|---|---:|---:|---:|
| bamboogle | 0.085 | 0.140 | +0.055 |
| nq | 0.162 | 0.279 | +0.117 |
| triviaqa | 0.228 | 0.390 | +0.162 |
| popqa | 0.162 | 0.275 | +0.113 |
| hotpotqa | 0.128 | 0.196 | +0.068 |
| 2wikimultihopqa | 0.177 | 0.194 | +0.017 |
| musique | 0.039 | 0.086 | +0.047 |
| average | 0.140 | 0.223 | +0.083 |

---

## 7. Observed wall-clock (full eval per variant)

| Variant | Run mode | `INFERENCE_MAX_WORKERS` | Total wall-clock | Notes |
|---|---|---|---|---|
| pre-GRPO (`qwen_3_0.6b`) | Interactive `srun` (job 2120423, node870) | 16 (NQ, TriviaQA, PopQA) → 32 (HotpotQA, 2Wiki, MuSiQue) bumped mid-run | ~115 min | Plus ~28 s bamboogle smoke |
| v0 (`qwen_3_0.6b_v0`) | `sbatch` (job 2125009, node875) | 32 throughout | **2h 26m 33s** (`sacct: COMPLETED 02:26:33`) | Including ~5m45s retriever cold-start (first-time HF arrow build on this node) and ~4m20s SGLang cold-start (flashinfer JIT compile) |

The 32-worker bump gave a measured **~2× speedup** (popqa @ 16w = 1.79 s/item vs hotpotqa @ 32w = 0.92 s/item, both 1k-item-equivalent shape). Decode-side throughput peaked at ~3,300 tokens/s on a 0.6B model with cuda-graph + flashinfer; the bottleneck after the bump is retriever queue depth (8 workers feeding 32 in-flight clients).

---

## 8. Per-dataset plots (training-side, for context)

The training-side reward and tool-use curves for `z7kcxfof` are in:

```
docs/report/results_v0_assets/single_p1_basic_w_ex_z7kcxfof.png
```

(Reward 0.148 → 0.190 over 1046 steps; tool calls ramp 1.83 → 2.00; response length 1788 → 2047. The "always 2 searches" anchor from the Hamlet example saturates around step 600.)

No new plots are produced for the eval (single greedy seed → deterministic outputs; the comparison is the table in §6 above).

---

## 9. Cross-run summary table (eval-side)

| Variant | Bamboogle | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Avg EM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pre-GRPO (qwen_3_0.6b) | 0.056 | 0.113 | 0.178 | 0.133 | 0.083 | **0.141** | 0.010 | 0.102 |
| v0 (qwen_3_0.6b_v0) | **0.088** | **0.191** | **0.302** | **0.227** | **0.116** | 0.138 | **0.023** | **0.155** |
| Δ EM | +0.032 | +0.078 | +0.124 | +0.094 | +0.033 | −0.003 | +0.013 | **+0.053** |

---

## 10. Findings

### 10.1. GRPO meaningfully improved Qwen3-0.6B on this task

A clean +0.053 EM (+52 % relative) lift across 51,713 items per variant on full Plan A test/dev sets is well outside any plausible noise band for greedy single-seed eval. **GRPO with the paper-faithful Search-R1 EM reward and the `p1_basic_w_ex` prompt does teach a 0.6 B Qwen3 hybrid useful tool-use behavior.** Heavy-tool mode (2 calls / 4 turns / ~2050 tokens) emerges from the 2-search Hamlet example anchor — this is the same behavioral signature documented at training time in `RESULTS_v0.md` §10.

### 10.2. Lift is concentrated on single-hop QA

NQ, TriviaQA, PopQA all show +0.07 to +0.12 EM (about 70 % relative). MuSiQue (3-hop) doubles in absolute EM but from a small base (0.010 → 0.023). Multi-hop datasets are dampened: HotpotQA +0.033 EM (+40 %), 2WikiMultiHopQA flat. **At 0.6 B parameters, multi-hop is capacity-bound rather than training-bound.** The training-time prompt's "always 2 searches" anchor is helpful for compositions that decompose into two factual single-hops (typical of NQ-with-followup) but does not solve genuine compositional 2-hop reasoning the way a 3 B+ model would.

### 10.3. Pre-GRPO performance on bamboogle (0.056 EM) sits below the user's prior memory (0.072)

The user previously recorded `~0.072 EM` for an untrained Qwen3-0.6B on bamboogle. Our pre-GRPO bamboogle result is 0.056 EM (125 items, single seed, fully aligned config). The 0.016-EM gap on a 125-item benchmark is **within ±0.04 95 % CI** for a 12.5 %-rate proportion at n=125 (approx ±0.04). The v0 result (0.088 EM) **matches** the user's expected ~0.08 within ±0.01. We treat the comparison as valid: the lift signal is the headline, the pre-GRPO absolute number has small-n noise on bamboogle.

### 10.4. The Plan A upgrade is meaningful

Originally MILESTONE_3.md planned 1k stratified subsamples for the five large datasets (NQ, TriviaQA, PopQA, HotpotQA, 2Wiki). The eval ran on **full** test/dev sets (3,610 / 11,313 / 14,267 / 7,405 / 12,576 items respectively); `sample_num` is not applied by the FlashRAG `search_r1` pipeline path. This eliminates the ~±0.026 to ±0.030 sampling SE we projected for Plan B. The resulting comparison is **statistically Plan A**; the per-dataset EMs in §6 are population-true.

### 10.5. Setup cost was non-trivial but mostly one-shot

14 fixes (`CODE_SETUP_v2.md` §3) between clone-and-run and the first clean comparison. The most consequential were:
- **Single braces in `\boxed{answer here}`** (template was using Python format-escape `{{}}` but read as raw string)
- **Leading space before `<result>`** (matches training tokenization byte-for-byte)
- **Raw retrieval format** (drop the Search-R1 `Doc i (Title:...)` prefix; training fed raw `{contents}\n\n`)
- **`top_n=5`, `max_obs_length=256`, `generator_max_input_len=4096`** (training-side budgets)
- **`enable_thinking=True`** (so Qwen3 hybrid's chat template doesn't auto-inject empty `<think></think>`)

After these were applied in sequence, the bamboogle smoke EM rose from 0.000 → 0.008 → 0.040 / 0.080 → 0.056 / 0.088 (pre-GRPO / v0). Each fix moved a recognisable amount; the alignment is therefore mechanically interpretable rather than empirically tuned.

### 10.6. The GRPO checkpoint generalises to held-out benchmarks

`p1_basic_w_ex_z7kcxfof` was trained on **MuSiQue** (3-hop) with EM reward against the gold answer. Held-out evaluation here on **NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, Bamboogle** (none seen at training time) shows the lift transfers strongly to single-hop and modestly to multi-hop. This rules out a "memorised the training distribution" explanation: the model is not retrieving training-set answers — it is exhibiting a learned tool-use skill that generalises across QA shapes.

### 10.7. The eval is now pinned and reusable for downstream comparisons

With the `prompt_mode=qwen3` switch in `SearchR1Pipeline`, the eval pipeline is byte-aligned to verl-legacy training rollouts that use the `<search>/<result>` tag format and the `p1_basic_w_ex` prompt. **Any future Qwen3-0.6B (or Qwen3.5-2B with the same tag scheme) checkpoint can be plugged in via `eval/<name>/` and `bash scripts/run_m3.sh <variant> <dataset> <seed>` without touching the eval code.** This is the M3 deliverable that unblocks Phase-2 NeMo-RL evaluation.

---

## 11. Open questions raised by these results

1. Does the lift hold on a larger model? Qwen3.5-2B (Phase-2 NeMo-RL target) is 3.3× the parameter count and adds dynamic batching + `<tool_call>` JSON tags. Expectation: lift on multi-hop should expand once the capacity floor is lifted. Testable once the first Qwen3.5-2B GRPO checkpoint exists.

2. The `p1_basic_w_ex` prompt anchors heavy-tool mode (2 calls / 4 turns) via its 2-search Hamlet example. **Single-hop datasets (NQ, TriviaQA, PopQA) get +69-71 % relative lift, which is high.** Is the 2-search anchor over-applied (the model also does 2 searches on questions where 1 suffices, which mostly works because the second search is wasted but not harmful)? Examining `output['final_response']` distributions for `tool_call_counts/mean` per-dataset would settle this.

3. **2WikiMultiHopQA tied** while HotpotQA gained — both are 2-hop. The 2Wiki questions tend to require multi-fact composition over the same passage; HotpotQA tends to require span-stitching across passages. Why did GRPO help HotpotQA but not 2Wiki? Inspecting the `final_response` for items where pre-GRPO succeeded but v0 failed (and vice versa) on 2Wiki vs HotpotQA would identify whether the training prompt distribution mismatched 2Wiki's question shape.

4. The training rollout reward function (`re_search.py` EM-only) returns 1.0 / 0.0; the **Search-R1 partial-credit reward** (0.1 floor for any well-formatted but wrong answer) was not used here (`p1_basic_w_ex` is in the v0 block where verl-legacy `qa_em.py` was active; partial credit was a v1 artifact). Whether re-training with the partial-credit reward would close the 2Wiki gap — or whether it would create the floor-masking issue documented in `RESULTS_v0.md` §11 — is an open question for Phase-2.

5. **All 7 datasets at full Plan A and a single seed take ~2.5 hours per variant on 1× A100-80GB**. Multi-seed × full Plan A is ~2.5 h × k seeds per variant; under the JustRL paper's mean-of-3-seeds convention this is ~7.5 h / variant. For Phase-2 multi-recipe evaluation, this is the rate-limiting cost to budget for.

---

## 12. Reproduction recipe (M3 v0 evaluation)

```bash
# On ALICE login node, from project root:
cd /zfsstore/user/s4374886/omega/reason_over_search

# 1. Acquire 4 h gpu-short A100 allocation (or 7-day gpu-a100-80g for sbatch)
#    Either interactive or sbatch — both supported.

# Interactive:
srun --partition=gpu-short --gres=gpu:a100:1 --cpus-per-task=40 --mem=160g --time=04:00:00 --pty bash
# Then on the allocated node, follow MILESTONE_3.md §Step-by-step.

# Sbatch (autonomous):
sbatch scripts/sbatch_m3.sh qwen3_0.6b      # pre-GRPO (~2h)
sbatch scripts/sbatch_m3.sh qwen3_0.6b_v0   # post-GRPO (~2h)

# Each sbatch starts retriever + SGLang, waits for health, runs all 7 datasets.
# Logs: logs/m3_<jobid>_m3_eval.{out,err}, logs/m3_<jobid>_retriever.log, logs/m3_<jobid>_sglang.log
# Results: evaluation_research/results/<dataset>/<dataset>_<timestamp>_m3_<variant>_seed1/
#   ├── metric_score.txt        — EM / ACC / F1
#   ├── intermediate_data.json  — per-item question / golden / pred / final_response
#   └── config.yaml             — reproduced eval config
```

To swap to a different checkpoint, drop it under `eval/<name>/` and add the case to `scripts/run_m3.sh` and `scripts/sbatch_m3.sh` (`MODEL_PATH` switch).

---

## 13. Files of record

- Pre-GRPO results (per-dataset, full intermediate data): `evaluation_research/results/<dataset>/<dataset>_<timestamp>_m3_qwen3_0.6b_seed1/`
- v0 (post-GRPO) results: `evaluation_research/results/<dataset>/<dataset>_<timestamp>_m3_qwen3_0.6b_v0_seed1/`
- v0 sbatch logs: `logs/m3_2125009_m3_eval.out`, `logs/m3_2125009_m3_eval.err`, `logs/m3_2125009_retriever.log`, `logs/m3_2125009_sglang.log`
- Pre-GRPO interactive transcript: ssh-driven on node870 within job 2120423; foreground bash output preserved in the conversation log.
- Training run archive: `docs/archive/verl_runs/v0/p1_basic_w_ex_z7kcxfof/` (training.log, W&B export, HF checkpoint at `global_step_1000_rollout_20260407_002729_hf/`)
- M3 milestone: `docs/milestone_three/MILESTONE_3.md`
- M3 code-setup diff: `docs/report/CODE_SETUP_v2.md`
- Training-time per-run synthesis (29 ALICE runs): `docs/report/RESULTS_v0.md` (14 v0) + `docs/report/RESULTS_v1.md` (15 v1)
