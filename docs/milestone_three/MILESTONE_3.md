---
title: MILESTONE 3
tags: []
source: internal
created: 2026-05-06
updated: 2026-05-06
---

# Milestone 3: Evaluate [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3) Hybrid

## Context

During the Qwen3-0.6B GRPO ablation block (W&B project `research`, archived in [docs/report/RESULTS_v0.md](../report/RESULTS_v0.md)), all training runs used the **hybrid** checkpoint (`Qwen3-0.6B`, the post-trained soft-switch reasoning model). The best-documented run in that block is **`p1_basic_w_ex` (W&B id `z7kcxfof`)** — the one that uniquely converged on heavy-tool behaviour (2 search calls / 4 turns, ~2050 token responses, end-of-run reward 0.190 after 1046 steps).

Two model snapshots are available for evaluation:

| Snapshot | Location | Description |
|---|---|---|
| `qwen_3_0.6b` | [`eval/qwen_3_0.6b/`](../../eval/qwen_3_0.6b/) | Qwen3-0.6B hybrid (pre-GRPO; the frozen post-trained checkpoint Qwen releases) |
| `qwen_3_0.6b_v0` | [`eval/qwen_3_0.6b_v0/`](../../eval/qwen_3_0.6b_v0/) | Qwen3-0.6B after 1046 GRPO steps with the p1_basic_w_ex prompt (HF checkpoint from `docs/archive/verl_runs/v0/p1_basic_w_ex_z7kcxfof/`) |

The archive run directory is preserved unchanged at `docs/archive/verl_runs/v0/p1_basic_w_ex_z7kcxfof/`.

## Goal

Run Plan B evaluation on both snapshots using **the prompt from the `p1_basic_w_ex_z7kcxfof` training run**, across all 7 benchmarks, on the ALICE cluster.

This answers two questions:
1. What is the zero-shot baseline of Qwen3-0.6B hybrid on these benchmarks (before any GRPO)?
2. Does 1046 steps of GRPO training with the p1_basic_w_ex prompt lift EM on the same benchmarks?

## The p1_basic_w_ex prompt (system message)

Used verbatim as the system message in the eval pipeline:

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

## Plan B settings (carry-over from M1)

All settings from [docs/milestone_one/FROZEN_CONFIG_v1.md](../milestone_one/FROZEN_CONFIG_v1.md) apply unchanged, except:

| Setting | M1 value | M3 value | Reason |
|---|---|---|---|
| Model | Search-R1 Qwen2.5-3B checkpoints | `eval/qwen_3_0.6b/` and `eval/qwen_3_0.6b_v0/` | Different model |
| Prompt format | `SEARCH_R1_TEMPLATE` (user message) | p1_basic_w_ex (system message) | Match training distribution |
| Retrieval response wrapper | `<information>` | `<result>` | Match training distribution |
| Hardware | Vast.ai RTX 4090 24GB | ALICE A100 80GB | In-house cluster |
| `apply_chat` | `True` for both variants | `True` (Qwen3 hybrid) | Same rule |
| Retriever index | Flat IP (`--num_retriever 2`) | IVF-SQ8 × 8 workers | Throughput under concurrent eval |
| Eval pipeline source | `evaluation_search_r1/` | `evaluation_research/` | M3-adapted fork |

**Unchanged from M1**: greedy decoding (`temperature=0.0`), `max_search_turns=4`, `step_limit=500`, `max_obs_length=500 tokens`, `retrieval_topk=3`, 7 datasets, 1k subsamples (bamboogle=125, musique=full 2417).

## Statistical justification

M3 uses **1 seed × Plan B** (subsampled: 1k each for nq/triviaqa/popqa/hotpotqa/2wiki; full for bamboogle=125 and musique=2417). This section justifies that against the alternative — Plan A (full test sets) with multiple seeds — and quantifies the residual uncertainty.

### Greedy decoding makes multi-seed Plan A redundant

Generation uses `temperature=0.0` (greedy). Output is fully determined by input; seeds in the eval pipeline only control *which* items are subsampled, not the model. Under Plan A (all items, no subsampling):

```
k × Plan A ≡ 1 × Plan A   for any k ≥ 1, under greedy decoding
```

Multi-seed Plan A is k× the compute for the same number. The "k seeds for robustness" framing only adds information when seeds change *something* — sampling temperature, subsample selection, augmentation. None apply to Plan A as defined.

### Plan B as an unbiased estimator of the Plan A truth

If subsampling is uniform-random, Plan B's empirical EM is an unbiased estimator of Plan A's population EM. With finite-population correction:

```
SE(p̂) = sqrt((N − n) / (N − 1)) × sqrt(p̂(1 − p̂) / n)
```

For 1 seed × Plan B at the M1 dataset sizes, with realistic p̂:

| Dataset | N (full) | n (Plan B) | p̂ | SE | 95% CI |
|---|---:|---:|---:|---:|---:|
| NQ | 3,610 | 1,000 | 0.40 | 0.013 | ±0.026 |
| TriviaQA | 11,313 | 1,000 | 0.55 | 0.015 | ±0.030 |
| PopQA | 14,267 | 1,000 | 0.40 | 0.015 | ±0.030 |
| HotpotQA | 7,405 | 1,000 | 0.30 | 0.013 | ±0.026 |
| 2WikiMultiHopQA | 12,576 | 1,000 | 0.30 | 0.014 | ±0.028 |
| MuSiQue | 2,417 | 2,417 | 0.10 | 0 | exact |
| Bamboogle | 125 | 125 | 0.10 | 0 | exact |

Each subsampled dataset's EM is known to within **±0.03 EM at 95% confidence** of what Plan A would report.

### What multiple Plan B seeds would buy

Pooling k seeds of n items each (independent draws), expected unique coverage of the full pool is:

```
n_unique = N × (1 − (1 − n/N)^k)
```

For 5 seeds × 1k on TriviaQA (N=11,313): n_unique ≈ 4,640 (41% of full). For NQ (N=3,610): n_unique ≈ 2,890 (80%). SE shrinks by ≈√k_eff, tightening per-dataset CI to ±0.01–0.013 — close to the Plan A truth at a fraction of the compute.

So **5 × Plan B is a reasonable random-sample approximation of 1 × Plan A** (Plan A 3× being a no-op under greedy):

```
P(|p̂_5×PlanB − p_PlanA| < 0.013) ≈ 0.95   (worst case across datasets)
```

### What matters for the M3 comparison question

The deliverable is a paired model comparison: does `qwen3_0.6b_v0` (post-GRPO) lift EM over `qwen3_0.6b` (pre-GRPO)? The unpaired difference statistic:

```
Δ = EM_v0 − EM_base
SE(Δ) ≤ √2 × SE(EM)        ≈ ±0.04 worst case at 1 × Plan B
```

A tighter, paired analysis on the same subsample uses **McNemar's test** on the discordant pairs (b = base wrong & v0 right; c = base right & v0 wrong):

```
McNemar χ² = (b − c)² / (b + c)    under H₀: no model difference
```

Paired comparison removes item-difficulty variance. At n=1k discordants, McNemar detects Δ ≥ 0.02 EM at p<0.05; lifts in the expected range (0.05–0.10 EM if GRPO worked) are detected with high power.

### Conclusion

| Claim | Verdict |
|---|---|
| 5 × Plan B ≈ Plan A truth, within ±0.01–0.013 EM at 95% | Yes (random-sample estimator) |
| 3 × Plan A is more rigorous than 1 × Plan A under greedy | No (identical results) |
| 1 × Plan B is sufficient to detect a real GRPO effect (Δ ≥ 0.05 EM) | Yes (McNemar p<0.01 at this n) |
| Upgrade to 3–5 seeds × Plan B if Δ lands marginal (≈0.02–0.03) | Recommended fallback |

M3 runs at 1 seed × Plan B for both variants. If a marginal result emerges, the same pipeline produces 3–5 seeds at ~5× the compute — still cheaper than 1 × Plan A.

## Environments

| Env name | Python | Purpose |
|---|---|---|
| `retriever` | 3.10 | FAISS retriever service |
| `evaluation_search_r1` | 3.11 | SGLang + evaluation_research eval pipeline |

Both envs are created and verified on the ALICE login node (2026-05-06). The `flashrag` package editable install in `evaluation_search_r1` now points to `evaluation_research/` (contains M3 code changes).

## IVF-SQ8 × 8 workers: resource analysis

The retriever runs the IVF4096-SQ8 index with 8 parallel workers (matching the training-time v1 default). **This is not suitable for paper-fidelity reproduction** (use `--index .../wiki18_100w_e5_flat_inner.index --num_retriever 2` for that), but is fast enough for the M3 comparative eval.

### Index

`local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index` — 16 GB on disk, downloaded 2026-05-06.

### CPU requirements (exact derivation)

| Thread pool | Count | Notes |
|---|---|---|
| FAISS OMP threads | 8 workers × 4 threads = **32** | Diminishing returns beyond 4/worker at nprobe=64 |
| E5-base-v2 encoder | 8 concurrent | Runs single-threaded per call; absorbed in OMP budget |
| FastAPI/uvicorn | ~4 | Event-loop + worker threads |
| SGLang (tokeniser + scheduler) | ~4 | |
| **Total** | **~40** | |

Request: **`--cpus-per-task 40`**. Setting `OMP_NUM_THREADS=4` before launching the retriever ensures FAISS respects the per-worker thread budget.

### RAM requirements

| Component | RAM |
|---|---|
| 8 × IVF-SQ8 index | 8 × 16 GB = 128 GB (measured: 134 GB RSS at v1 launch) |
| 8 × E5-base-v2 encoder | 8 × 440 MB ≈ 3.5 GB |
| Qwen3-0.6B (SGLang, bf16) | ~1.5 GB (VRAM only; negligible CPU footprint) |
| Corpus jsonl + Python overhead | ~10 GB |
| **Total** | **~150 GB** |

Request: **`--mem=160g`** (comfortable headroom; A100 nodes have ≥ 256 GB host RAM).

## Wall-clock estimate (initial; update after validation)

These are pre-run estimates based on model size, hardware specs, and M1 baseline timing. **Update this table after the bamboogle short-partition validation run.**

| Dataset | Items | Est. time (Qwen3-0.6B, A100) |
|---|---:|---|
| bamboogle | 125 | 3–6 min |
| nq | 1 000 | 10–20 min |
| triviaqa | 1 000 | 10–20 min |
| popqa | 1 000 | 10–20 min |
| hotpotqa | 1 000 | 10–20 min |
| 2wikimultihopqa | 1 000 | 10–20 min |
| musique | 2 417 | 25–45 min |
| **Total per variant** | **7 542** | **~1.5–2.5 h** |
| **Both variants** | 15 084 | **~3–5 h** |

Rationale: Qwen3-0.6B (600 M params, 1.2 GB at bf16) on A100 (2 TB/s MBW) is roughly 10× faster per token than Qwen2.5-3B on RTX 4090. Each item averages ~2 search turns × ~300–500 tokens generated + retrieval latency. At INFERENCE_MAX_WORKERS=16 and IVF latency ~0.3 s/call, throughput is dominated by retrieval until the model becomes the bottleneck at larger batch sizes.

## Step-by-step

### 1. Get a SLURM allocation

**For short-partition validation (bamboogle only, one variant):**

```bash
srun \
  --partition=gpu-a100-80g \
  --gres=gpu:a100:1 \
  --cpus-per-task=16 \
  --mem=80g \
  --time=00:30:00 \
  --pty bash
```

16 CPUs and 80 GB are sufficient for IVF × 8 (the index loads but not all 8 workers run at peak simultaneously during bamboogle's 125 items). The 30-minute time limit puts this job in the backfill window.

**For full 7-dataset runs, use sbatch (see §5).**

### 2. Load environment and start retriever

```bash
module purge
module load ALICE/default
module load Miniconda3/24.7.1-0
module load CUDA/12.4.0

cd /zfsstore/user/s4374886/omega/reason_over_search

export OMP_NUM_THREADS=4
/home/s4374886/.conda/envs/retriever/bin/python local_retriever/retriever_serving.py \
  --config local_retriever/retriever_config.yaml \
  --num_retriever 8 \
  --index "$(pwd)/local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index" \
  --port 3005 &

# Wait ~60 s for 8 workers to load index (~134 GB RSS on first launch)
sleep 60
curl -sS http://127.0.0.1:3005/health   # → {"status":"healthy","retrievers":{"total":8,"available":8}}
```

For the flat IP fallback (paper-fidelity or if IVF not available):
```bash
/home/s4374886/.conda/envs/retriever/bin/python local_retriever/retriever_serving.py \
  --config local_retriever/retriever_config.yaml \
  --num_retriever 2 \
  --index /zfsstore/user/s4374886/omega/re-search/assets/indexes/wiki18_100w_e5_flat_inner.index \
  --port 3005 &
```

### 3. Start the SGLang server

```bash
# qwen_3_0.6b (pre-GRPO)
/home/s4374886/.conda/envs/evaluation_search_r1/bin/python -m sglang.launch_server \
  --model-path eval/qwen_3_0.6b \
  --host 127.0.0.1 --port 3000 \
  --tp 1 --context-length 8192 --dtype bfloat16 --trust-remote-code &

# verify
curl -sS http://127.0.0.1:3000/health
```

Swap to `eval/qwen_3_0.6b_v0` for the v0 checkpoint run. Restart SGLang between variants.

### 4. Short-partition validation (bamboogle, one variant)

```bash
bash scripts/run_m3.sh qwen3_0.6b bamboogle 1
```

Check the results JSON: confirm search turns average ~1–2, tokens per response reasonable, EM non-zero. Update the wall-clock table above with the observed time.

### 5. Full run via sbatch

```bash
mkdir -p logs
sbatch scripts/sbatch_m3.sh qwen3_0.6b
sbatch scripts/sbatch_m3.sh qwen3_0.6b_v0
```

`sbatch_m3.sh` starts the retriever and SGLang automatically, runs all 7 datasets, then cleans up. It falls back to the flat IP index if the IVF index is missing. Logs go to `logs/m3_<jobid>_m3_eval.{out,err}`.

Time limit: `--time=04:00:00`. If the estimate proves too tight after validation, bump to `06:00:00`.

### 6. Aggregate results

```bash
/home/s4374886/.conda/envs/evaluation_search_r1/bin/python scripts/aggregate.py \
  --output docs/milestone_three/RESULTS_M3.md
```

## Expected results and success criteria

No prior EM numbers exist for Qwen3-0.6B on these benchmarks. Use M1 instruct results as a rough ceiling and M1 base as a rough floor:

| Dataset | M1 base | M1 instruct | Expected 0.6B range |
|---|---:|---:|---|
| Bamboogle | 0.088 | 0.344 | unknown (multi-hop is hard at 0.6B) |
| NQ-1k | 0.390 | 0.402 | lower than 3B |
| TriviaQA-1k | 0.583 | 0.531 | lower than 3B |
| PopQA-1k | 0.424 | 0.413 | lower than 3B |
| HotpotQA-1k | 0.263 | 0.346 | lower than 3B |
| 2WikiMultiHopQA-1k | 0.239 | 0.350 | lower than 3B |
| MuSiQue (2417) | 0.055 | 0.141 | very low |

**Success criterion**: both variants complete all 7 datasets; results are recorded and reproducible. If `qwen_3_0.6b_v0` exceeds `qwen_3_0.6b` on average EM, training helped. If not, document which datasets regressed.

## Deliverables

1. [x] `eval/qwen_3_0.6b/` and `eval/qwen_3_0.6b_v0/` populated and accessible on ALICE. ✓ (done 2026-05-06)
2. [x] `retriever` and `evaluation_search_r1` conda envs created on ALICE. ✓ (done 2026-05-06)
3. [x] IVF-SQ8 index downloaded to `local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index` (16 GB). ✓ (done 2026-05-06)
4. [x] `evaluation_research/` pipeline fork created with M3 code changes (QWEN3_0_6B_TEMPLATE, `prompt_mode=qwen3`, `<result>` tag swap); editable install verified in `evaluation_search_r1` env. ✓ (done 2026-05-06)
5. [x] `scripts/run_m3.sh` and `scripts/sbatch_m3.sh` created. ✓ (done 2026-05-06)
6. [ ] Short-partition validation: bamboogle (qwen3_0.6b) passes; wall-clock table updated.
7. [ ] Both variants evaluated on all 7 benchmarks via sbatch.
8. [ ] Results aggregated to `docs/milestone_three/RESULTS_M3.md`.
9. [ ] Summary table and findings written up here under §Status.

## Status

**2026-05-06**: Models staged (`eval/qwen_3_0.6b/`, `eval/qwen_3_0.6b_v0/`). Both conda envs created and verified. IVF-SQ8 index downloaded (16 GB, `local_retriever/indexes/`). `evaluation_research/` fork created from `evaluation_search_r1/` with three targeted changes: `QWEN3_0_6B_TEMPLATE` added to `templates.py`; `prompt_mode='qwen3'` wired through `active_pipeline.py` and `run_eval.py` (system-message chat format + `<result>` observation wrapper); editable install verified. `scripts/run_m3.sh` and `scripts/sbatch_m3.sh` created. Next: short-partition validation (bamboogle, 30 min srun), then submit sbatch for both variants.
