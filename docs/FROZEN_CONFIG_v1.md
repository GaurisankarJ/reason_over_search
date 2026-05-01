# Plan B v1 — frozen reproducer config

> **Status (2026-05-01): locked.** This is the single source of truth for the Plan B v1 setup that produced [RESULTS_PLAN_B.md](RESULTS_PLAN_B.md) and serves as the configuration Plan A will scale to 5 seeds × full datasets. Anything that contradicts this file in another doc is wrong; fix the other doc.
>
> Pinning: this file describes the repo state at commit on `retriver-updates` after this freeze (see git log). Re-running on a fresh checkout of that commit, on the hardware described below, against the same model checkpoints (sha256 below) and FAISS index, must reproduce [RESULTS_PLAN_B.md](RESULTS_PLAN_B.md) exactly under greedy decoding.

## Headline numbers (single seed × 1 k subsamples; full Bamboogle and MuSiQue)

| Dataset | base v1 | base paper | base Δ | instruct v1 | instruct paper | instruct Δ |
|---|---:|---:|---:|---:|---:|---:|
| Bamboogle (n=125) | 0.088 | 0.128 | −4.0 | 0.344 | 0.232 | +11.2 (n=125 var) |
| NQ-1k | 0.390 | 0.421 | −3.1 | 0.402 | 0.397 | +0.5 |
| TriviaQA-1k | 0.583 | 0.583 | 0.0 | 0.531 | 0.565 | −3.4 |
| PopQA-1k | 0.424 | 0.413 | +1.1 | 0.413 | 0.391 | +2.2 |
| HotpotQA-1k | 0.263 | 0.297 | −3.4 | 0.346 | 0.331 | +1.5 |
| 2WikiMultiHopQA-1k | 0.239 | 0.274 | −3.5 | 0.350 | 0.310 | +4.0 |
| MuSiQue (full, 2417) | 0.055 | 0.066 | −1.1 | 0.141 | 0.124 | +1.7 |
| **Avg EM** | **0.292** | **0.312** | **−2.0** | **0.361** | **0.336** | **+2.5** |

Paper targets: arXiv 2503.09516 v5, Appendix F / Table 3 (`arxiv.org/html/2503.09516v5#A6`).

## Models (sha256-pinned)

| Variant | HF repo | local path | shard-1 LFS sha256 | total | eos_token_id |
|---|---|---|---|---:|---:|
| base | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo` | `evaluation_search_r1/search_r1_base_model/` | `7ac54e1b…36a9dabf` | 13.6 GB / 3 shards | 151643 (`<\|endoftext\|>`) |
| instruct | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo` | `evaluation_search_r1/search_r1_instruct_model/` | `3d787062…68ccd35` | 13.6 GB / 3 shards | 151645 (`<\|im_end\|>`) |

`config.json` `_name_or_path` reads `Qwen/Qwen2.5-3B(-Instruct)` for both — that's stamped from the trainer init, **not** an identity check. Use the LFS sha256.

## Retriever

- Encoder: `intfloat/e5-base-v2` (mean pooling, fp16).
- Corpus: `wiki18_100w` (`local_retriever/corpus/wiki18_100w.jsonl`).
- Index: **FAISS Flat Inner-Product**, `local_retriever/indexes/wiki18_100w_e5_flat_inner.index` (~65 GB, exact float32). Recall = 100 % by definition.
- Service: `retriever_serving.py` on `127.0.0.1:3005`. Default config: [`local_retriever/retriever_config.yaml`](../local_retriever/retriever_config.yaml).
- `retrieval_topk: 3`, `retrieval_query_max_length: 256`, `retrieval_use_fp16: True`.
- Health check: `curl -sS http://127.0.0.1:3005/health` → `"healthy"`.
- IVF-SQ8 alternative (`wiki18_100w_e5_ivf4096_sq8.index`, ~16 GB, 3–10× faster CPU retrieval) is supported via `--index` but is **NOT** the frozen config — flat IP is the reproducer index.

## Generator (SGLang)

- Server: `127.0.0.1:3000`, started by [`scripts/manage_sglang.sh switch <variant>`](../scripts/manage_sglang.sh).
- Flags: `--tp 1 --context-length 8192 --dtype bfloat16 --trust-remote-code`.
- Verify variant: `curl -sS http://127.0.0.1:3000/get_model_info | grep -o instruct`.

### Sampling — greedy

| Parameter | Value | Source |
|---|---:|---|
| `temperature` | **0.0** | [`basic_config.yaml:110`](../evaluation_search_r1/flashrag/config/basic_config.yaml#L110) |
| `top_p` | 1.0 (default) | implicit |
| `top_k` | -1 (default) | implicit |
| `do_sample` | implicit False | greedy when temp=0 |
| `max_tokens` (per step) | 32 (overridden per step in pipeline) | [`basic_config.yaml:109`](../evaluation_search_r1/flashrag/config/basic_config.yaml#L109) |
| Per-step token cap | 500 | [`active_pipeline.py:58`](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py#L58) (`step_limit`) |
| Stop tokens | `</search>`, `</answer>`, `<\|im_end\|>`, `<\|endoftext\|>` | [`active_pipeline.py:62`](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py#L62) |

**Do not change `temperature` or `top_p`.** Paper eval is greedy via verl `_validate()` → `vllm_rollout.py:162-171` overrides `do_sample=False` to `(temp=0, top_p=1.0, top_k=-1, n=1)`. See [archive/TEMPERATURE_HYPOTHESIS_WRONG.md](archive/TEMPERATURE_HYPOTHESIS_WRONG.md).

## Pipeline knobs

In [`evaluation_search_r1/flashrag/pipeline/active_pipeline.py`](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py):

| Knob | Value | Line |
|---|---:|---|
| `max_search_turns` | 4 | [`:57`](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py#L57) |
| `step_limit` (per-step max new tokens) | 500 | [`:58`](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py#L58) |
| `max_obs_length` (retrieval observation truncation) | 500 tokens | [`:59`](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py#L59) |
| `generator_max_input_len` | 4096 | [`basic_config.yaml:105`](../evaluation_search_r1/flashrag/config/basic_config.yaml#L105) |
| Passage formatting | `Doc i(Title: <title>) <text>\n` (then strip + truncate to 500 tok) | [`:111`](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py#L111) |
| `<information>` whitespace | `\n\n<information>{stripped}</information>\n\n` | [`:120`](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py#L120) |
| Question normalization | `.strip()` + ensure trailing `?` | [`:40-42`](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py#L40) |
| `<search>` regex | first-match | [`flashrag/search_r1/parser.py`](../evaluation_search_r1/flashrag/search_r1/parser.py) |
| `add_special_tokens` block | **removed** (D8 fix) | not present in pipeline; observation tokens are encoded with `add_special_tokens=False` for length counting only |

## Prompt

[`evaluation_search_r1/flashrag/search_r1/templates.py`](../evaluation_search_r1/flashrag/search_r1/templates.py) `SEARCH_R1_TEMPLATE`. Byte-identical to upstream `make_prefix(template_type='base')`, including the restored `For example, <answer> Beijing </answer>.` example sentence ([`:10`](../evaluation_search_r1/flashrag/search_r1/templates.py#L10)).

Both variants use this template body. The chat-template wrapper (`tokenizer.apply_chat_template(...)` at [`active_pipeline.py:45`](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py#L45)) is applied **for both base and instruct** (D1 fix; [`run_one.sh:35,39`](../scripts/run_one.sh#L35) hard-codes `apply_chat=True` for both). The base GRPO checkpoint's `tokenizer_config.json` ships with the Qwen2.5 chat template, so this is well-defined.

## Datasets and splits

Frozen as a dataset → split mapping in [`scripts/run_one.sh:22-31`](../scripts/run_one.sh#L22):

| Dataset | Split | Subsample? | n |
|---|---|---|---:|
| bamboogle | test | full | 125 |
| nq | test | 1k subsample | 1000 |
| triviaqa | test | 1k subsample | 1000 |
| popqa | test | 1k subsample | 1000 |
| hotpotqa | dev | 1k subsample | 1000 |
| 2wikimultihopqa | dev | 1k subsample | 1000 |
| musique | dev | full | 2417 |

Subsamples are deterministic and live at `data_subsample/<dataset>/<split>.jsonl` (built by `scripts/subsample.sh`). Plan A on Vast.ai will run **full** splits on every dataset — the v1 reduced-data sweep is for the locked-config validation only.

## Metrics

- Scorer: [`flashrag/search_r1/answer_utils.py`](../evaluation_search_r1/flashrag/search_r1/answer_utils.py) — SQuAD-canonical normalize → exact equality. **DO NOT modify.**
- Reported metrics: `em`, `acc` (sub-EM), `f1` (token-level F1).
- Trace-health (close-rate / length-truncation / mean tokens) computed by [`scripts/aggregate.py`](../scripts/aggregate.py) from each run's `intermediate_data.json`.

## Audit fixes applied (load-bearing)

| # | Fix | File:line |
|---|---|---|
| D1 | `apply_chat=True` for **base** (was False) | [`scripts/run_one.sh:35`](../scripts/run_one.sh#L35) |
| D-prompt-micro | Restore `For example, <answer> Beijing </answer>.` in prompt | [`templates.py:10`](../evaluation_search_r1/flashrag/search_r1/templates.py#L10) |
| D8 | Remove runtime `add_special_tokens` block | [`active_pipeline.py`](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py) (block removed) |

D1 alone delivers +3.0 pp on NQ base; D-prompt-micro + D8 together deliver another +4.4 pp on NQ base. On instruct, all three combined moved <1 pp.

The 10 earlier audit fixes are listed in [REPRODUCIBILITY.md](REPRODUCIBILITY.md#divergences-fixed) and summarized as: passage formatting `Doc i(Title: …) <text>`, `retrieval_topk` 5 → 3, `<information>` whitespace `\n\n` outer / none inner, `max_search_turns` 8 → 4, observation truncation 500 tokens, question `.strip()` + trailing `?`, `retrieval_query_max_length` 128 → 256, invalid-search corrective wording, first-match `<search>` regex, per-step token limit 512 → 500.

## Hardware

Single RTX 4090 (24 GB), AMD EPYC 7642 (48 c / 96 t), 503 GB RAM, no NVLink. Full table: [HARDWARE.md](HARDWARE.md). Implications:

- Qwen2.5-3B in bf16 fits in ~22 GB on the 4090.
- Flat-IP FAISS lives in host RAM (~65 GB).
- GPU FAISS and SGLang **cannot** share the 4090 (16 GB index + 22 GB SGLang > 24 GB VRAM).
- Single-4090 wall-clock for one full v1 sweep: ~17 h (1 seed × 7 datasets × 2 variants); per-dataset Bamboogle/instruct ~6 min.

## Software environments

- Eval venv: `/venv/evaluation_search_r1` (FlashRAG + Search-R1 fork). Verify: `/venv/evaluation_search_r1/bin/python -c "import flashrag"`.
- Retriever venv: `/venv/retriever` (faiss-cpu). GPU FAISS optionally at `local_retriever/.venv` (faiss-gpu-cu12 + torch+cu130, ~6.3 GB).
- Docker image used on Vast.ai: [`pantomiman/reason-over-search-v1`](https://hub.docker.com/r/pantomiman/reason-over-search-v1).

## Reproducing one run end-to-end

Pre-flight (services up):

```bash
# 1) Retriever (host RAM ~65 GB; flat IP)
/venv/retriever/bin/python local_retriever/retriever_serving.py &
curl -sS http://127.0.0.1:3005/health   # → "healthy"

# 2) SGLang (variant under test)
scripts/manage_sglang.sh switch instruct
curl -sS http://127.0.0.1:3000/get_model_info | grep -o instruct
```

Run:

```bash
cd /workspace/reason_over_search
scripts/run_one.sh instruct bamboogle 1 > run.log 2>&1
```

`run_one.sh` is **resume-aware**: if `metric_score.txt` exists for `(variant, dataset, seed)` it skips. Force re-run by clearing the matching dir first.

Pull EM / F1:

```bash
LATEST=$(ls -dt evaluation_search_r1/results/bamboogle/bamboogle_*_search_r1_instruct_seed1 | head -1)
grep -E "^(em|f1|acc):" "$LATEST/metric_score.txt"
```

Aggregate after a sweep:

```bash
/venv/evaluation_search_r1/bin/python scripts/aggregate.py --output docs/RESULTS_PLAN_B.md
```

## What is NOT in the frozen config (do not change without a new audit)

- Sampling: `temperature`, `top_p`, `top_k`, `do_sample`, `n`.
- Retrieval: `retrieval_topk`, `retrieval_query_max_length`, `max_obs_length`, the FAISS index type (flat IP), the encoder, the corpus.
- Pipeline: `max_search_turns`, `step_limit`, the prompt template body, the chat-template wrapper, the question-normalization step, the stop-token list, the passage format string, the `<information>` whitespace, the invalid-search corrective text, the `<search>` first-match regex.
- Models: the two GRPO checkpoints (sha256-pinned), the `Qwen/Qwen2.5-3B(-Instruct)` underlying base.
- Metrics: the SQuAD-canonical EM scorer, the F1 / ACC definitions, the gold-answer fields in the dataset jsonl.

Anything you might be tempted to change — see if it's listed in [archive/DISCARDED_ABLATIONS.md](archive/DISCARDED_ABLATIONS.md) first. The autoresearch loop already tried `topk 5`, `max_obs_length 750/833`, `max_search_turns 5`, multi-query retrieval, query expansion, system-message dropout, `repetition_penalty 1.05`, and serialized inference. None of them lifted EM.
