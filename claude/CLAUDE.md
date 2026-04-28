# CLAUDE.md — Reason Over Search

You are a helpful research assistant with years of experience. Your job is to help me reproduce, extend, and ablate retrieval-augmented reasoning models. This file is your persistent memory: read it first whenever you restart so you can pick up where we left off.

More instructions will be added here as the research progresses. Treat this file as a living source of truth — update it (or ask me to update it) whenever a load-bearing fact changes.

## What this project is

We are reproducing and extending two papers on RL-trained retrieval-augmented LLMs:

1. **ReSearch** — https://www.alphaxiv.org/abs/2503.19470
2. **Search-R1** — https://www.alphaxiv.org/abs/2503.09516 — *current focus, Phase 1*

The Search-R1 setup: a Qwen2.5-3B policy is trained with GRPO to interleave `<search>…</search>` calls with reasoning, retrieve documents from a frozen Wikipedia FAISS index, and emit a final `<answer>…</answer>`. We are *not* training; we are *evaluating* the published GRPO checkpoints end-to-end on QA benchmarks.

The official Search-R1 repo is `https://github.com/PeterGriffinJin/Search-R1`. They do not ship an evaluation pipeline, so the one in this repo is adapted from FlashRAG/ReSearch and validated against the paper numbers.

## Phase 1 goal (from [README.md](../README.md))

Reproduce the Search-R1 3B baseline for both checkpoints (`base` and `instruct`) on 7 benchmarks: Bamboogle, 2WikiMultiHopQA, TriviaQA, PopQA, MuSiQue, NQ, HotpotQA. Run each 5×, report averages. Constraints: reproducible on Vast.ai + in-house GPUs, minimize cost/wall-clock.

Paper targets we compare against (Qwen2.5-3B EM, Search-R1 v5 Table 3):

|              | NQ    | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | Avg   |
|---           |---    |---       |---    |---       |---    |---      |---        |---    |
| GRPO base    | 0.421 | 0.583    | 0.413 | 0.297    | 0.274 | 0.066   | 0.128     | 0.312 |
| GRPO instruct| 0.397 | 0.565    | 0.391 | 0.331    | 0.310 | 0.124   | 0.232     | 0.336 |

## Repo layout (what lives where)

- [`README.md`](../README.md) — Phase 1 milestone framing.
- [`evaluation_search_r1/`](../evaluation_search_r1/) — eval pipeline (FlashRAG fork + Search-R1 prompts/parsers).
  - [`README.md`](../evaluation_search_r1/README.md) — how to run eval per dataset.
  - [`REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) — paper targets, the 10 divergences fixed, smoke-validation results, model sha256 verification. **Read this before changing anything in the eval pipeline.**
  - [`EVAL_OPS.md`](../docs/EVAL_OPS.md) — three sweep plans (A full / B reduced / C one-seed), wall-clock budgets, where time goes.
  - [`RESULTS_PLAN_B.md`](../docs/RESULTS_PLAN_B.md) — auto-generated final results.
  - [`COMPARISON_PLAN_B.md`](../docs/COMPARISON_PLAN_B.md) — Plan B vs paper gap analysis; base-variant deviation discussion + ranked next-step experiments.
  - `flashrag/pipeline/active_pipeline.py` — search↔generate loop.
  - `flashrag/search_r1/parser.py` — `<search>`/`<answer>` parsing, observation truncation.
  - `flashrag/search_r1/templates.py` — base + instruct prompt templates.
  - `flashrag/search_r1/answer_utils.py` — EM/F1 scorer (ground truth — DO NOT modify).
  - `flashrag/config/basic_config.yaml` — retrieval_topk, max_search_turns, sampling params.
  - `run_eval.py` — entrypoint.
- [`local_retriever/`](../local_retriever/) — FAISS retriever HTTP service.
  - [`README.md`](../local_retriever/README.md) — start the retriever; flat vs IVF-SQ8 vs GPU.
  - [`RETRIEVER_INDEXING.md`](../docs/RETRIEVER_INDEXING.md) — RAM costs, index choices.
- [`/workspace/index_creation/`](../../index_creation/) — script + artifact for the IVF4096-SQ8 quantized index (`wiki18_100w_e5_ivf4096_sq8.index`, ~16 GB, 3–10× faster than flat).
- [`scripts/`](../scripts/) — `run_one.sh`, `manage_sglang.sh`, `subsample.sh`, `aggregate.py`, the three sweep scripts.
- [`docker/reason-over-search-v1/`](../docker/) — image used on Vast.ai (`pantomiman/reason-over-search-v1`).
- [`docs/VAST_AI_PLAN_A.md`](../docs/VAST_AI_PLAN_A.md) — Vast.ai GPU cost/throughput analysis for finishing Plan A in ≤24 h. Three concrete fleet configs.
- [`docs/HARDWARE.md`](../docs/HARDWARE.md) — this box's specs + accelerator comparison.
- [`data/`](../data/) — full eval datasets (jsonl, per benchmark).
- [`data_subsample/`](../data_subsample/) — deterministic 1k subsamples for the fast Plan B sweep.
- [`program.md`](../program.md) — playbook for the autonomous-research experiment loop on `experiment_ros/<tag>` branches.
- `results.tsv` — per-experiment EM/F1 log for the autoresearch loop (untracked, do not commit).

## Runtime services (everything must be up before eval)

- **Retriever** on `127.0.0.1:3005`. Health: `curl -sS http://127.0.0.1:3005/health` → `"healthy"`. Default is CPU + flat FAISS (~65 GB RAM, faiss-cpu venv at `/venv/retriever`). For ~3–10× faster CPU retrieval pass `--index ./indexes/wiki18_100w_e5_ivf4096_sq8.index`. GPU FAISS is opt-in via `local_retriever/.venv` and **cannot coexist with SGLang on the single 4090** (16 GB VRAM for the index + 22 GB for SGLang).
- **SGLang** on `127.0.0.1:3000` serving the variant under test. Switch with `scripts/manage_sglang.sh switch base|instruct`. Verify: `curl -sS http://127.0.0.1:3000/get_model_info | grep -o instruct`.
- **Eval venv** at `/venv/evaluation_search_r1`. Verify: `/venv/evaluation_search_r1/bin/python -c "import flashrag"`.

## What's been done so far (state as of 2026-04-28)

Phase 1 reproduction is essentially complete on the **instruct** variant; the **base** variant has an unresolved one-sided gap.

**Setup**:
- Both GRPO checkpoints downloaded and **identity-verified by LFS sha256** against the upstream HF repos (`PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-(it-)em-grpo`). Underlying base is `Qwen/Qwen2.5-3B`. Note: `config.json` `_name_or_path` reads the underlying Qwen path (same as on the HF GRPO repo), not the GRPO repo name — sha256 is the real identity check. Details in [REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).
- Wiki-18 corpus + E5-base-v2 encoder + flat and IVF-SQ8 FAISS indexes downloaded/built and live on this box.
- Datasets present for all 7 benchmarks + deterministic 1k subsamples.
- 10 divergences from the official Search-R1 repo audited and fixed (passage formatting, retrieval_topk 5→3, `<information>` whitespace, max_search_turns 8→4, observation truncation to 500 tokens, question normalization, retrieval_query_max_length 128→256, invalid-search corrective text, first-match `<search>` regex, per-step token cap 512→500). Full table in [REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).

**Smoke + Plan B results** (1 seed × 7 datasets × 2 variants; large datasets 1k-subsampled, Bamboogle/MuSiQue full — see [COMPARISON_PLAN_B.md](../docs/COMPARISON_PLAN_B.md)):

- **Instruct**: average +3.1 pp vs paper (0.367 vs 0.336). Within ±5 pp on 6/7 datasets; Bamboogle is the only outlier (+12.8 pp / ~3.4 σ on n=125). Reproduction-grade; the small positive bias is consistent with cleaner `</answer>` closure than the paper's training-time rollouts.
- **Base**: average **−8.3 pp** vs paper (0.229 vs 0.312). **Below paper on all 7 datasets** (−1.6 to −16.2 pp). One-sided across the board → not subsample noise, not a sampling lottery — likely a systematic config issue.

**Top suspects for the base gap** (ranked by cost / information ratio in [COMPARISON_PLAN_B.md](../docs/COMPARISON_PLAN_B.md)):
1. **Apply_chat=False on base** (probe-confirmed on Bamboogle): `run_one.sh:35` hard-codes `apply_chat=False` for base, but the base GRPO checkpoint ships with a Qwen2.5 chat template. Re-running Bamboogle base with `apply_chat=True` closed the gap to paper exactly (EM 0.112→0.128, paper 0.128) and pushed `</answer>` close-rate 84%→100%. NQ/TriviaQA/PopQA/HotpotQA likely benefit too. **Probe sweep on NQ-1k + TriviaQA-1k is the in-flight next step.**
2. **Prompt template missing one sentence**: our [`templates.py:1-11`](../evaluation_search_r1/flashrag/search_r1/templates.py) drops upstream's `For example, <answer> Beijing </answer>.` from the base prompt. Small but real divergence; one-line fix. (LOW severity per [PAPER_VS_OURS_AUDIT.md](../docs/PAPER_VS_OURS_AUDIT.md#prompt-template-micro-divergence-negligible).)
3. **Special-token additions at runtime** ([`active_pipeline.py:37-42`](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py#L37-L42)): we add `<search>`, `</search>`, etc. as additional special tokens after tokenizer load; upstream does not. Changes runtime IDs vs what the model was GRPO-trained on. LOW severity, easy to remove. See [PAPER_VS_OURS_AUDIT.md D8](../docs/PAPER_VS_OURS_AUDIT.md#d8-special-token-additions).
4. **Base trace hygiene visibility**: Plan B doesn't surface per-dataset truncation rate. Likely silently undercounting base on factoid datasets the same way it was on Bamboogle pre-apply_chat.
5. **Single seed at greedy**: tokenizer/KV-cache nondeterminism still gives ~1–2 pp drift; need 2–3 seeds to bracket per-dataset variance before Plan A.

**Note on temperature**: Earlier in this session I hypothesized the paper uses `temperature=1.0`. **That was wrong.** [PAPER_VS_OURS_AUDIT.md D3](../docs/PAPER_VS_OURS_AUDIT.md) shows verl's `_validate()` hard-codes `do_sample=False` and `vllm_rollout.py` overrides to `temperature=0, top_p=1.0` for validation. Paper eval is greedy. Our `temperature: 0.0` is **correct**. Post-mortem: [docs/archive/TEMPERATURE_HYPOTHESIS_WRONG.md](../docs/archive/TEMPERATURE_HYPOTHESIS_WRONG.md).

Recommendation: **don't launch Plan A (~17 days) until the base gap closes to ~3 pp on at least one dataset**; otherwise it just buys tighter error bars on a wrong number.

**Autoresearch loop** (see [program.md](../program.md)) — running on branch `experiment_ros/apr27`. Current `results.tsv` shows instruct/Bamboogle baseline at EM 0.336 (greedy, temp=0) with 11 ablations tried; only `temperature=0` (greedy) was kept. Discarded include `topk 3→5`, `max_obs_length 500→750`, `max_search_turns 4→5`, multi-query retrieval, query expansion, dropping the default chat-template system message, repetition_penalty 1.05, and serializing inference. Full log in [`results.tsv`](../results.tsv); per-ablation reasoning in [docs/archive/DISCARDED_ABLATIONS.md](../docs/archive/DISCARDED_ABLATIONS.md).

**Plan A prep** — `eval_final` branch (off `experiment_ros/apr27`) is the working branch for the Plan A launch. Cost analysis for Vast.ai in [docs/VAST_AI_PLAN_A.md](../docs/VAST_AI_PLAN_A.md) (cheapest: 8× RTX 4090 marketplace ≈ $58–77; balanced: 3× H100 PCIe ≈ $108).

## How to run a single eval

```bash
cd /workspace/reason_over_search
scripts/run_one.sh instruct bamboogle 1 > run.log 2>&1
```

`run_one.sh` is **resume-aware**: if `evaluation_search_r1/results/<dataset>/…_seed1/metric_score.txt` exists it skips and exits 0. Clear stale outputs before re-running the same `(variant, dataset, seed)`:

```bash
rm -rf evaluation_search_r1/results/bamboogle/bamboogle_*_search_r1_instruct_seed1
```

Pull EM/F1:

```bash
LATEST=$(ls -dt evaluation_search_r1/results/bamboogle/bamboogle_*_search_r1_instruct_seed1 | head -1)
grep -E "^(em|f1):" "$LATEST/metric_score.txt"
```

A single Bamboogle/instruct run takes ~6 min on this 4090.

## Hardware

Single RTX 4090 (24 GB), AMD EPYC 7642 (48c/96t), 503 GB RAM. No NVLink, single GPU. Practical implications: 3B in bf16 fits comfortably; FAISS index lives in host RAM (with VRAM as opt-in); GPU FAISS and SGLang cannot share the 4090. Full table in [docs/HARDWARE.md](../docs/HARDWARE.md).

## How I want you to work with me

- **Default to terse.** State results and decisions directly; skip narration.
- **Read [REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md) before touching the eval pipeline.** The 10 fixes are load-bearing and were audited against the official repo.
- **Don't modify the EM scorer** (`flashrag/search_r1/answer_utils.py`), the model checkpoints, the FAISS index, or the dataset jsonl files. Those are the ground truth for the experiment.
- **For autoresearch loops** (`experiment_ros/<tag>` branches): the editable surface is `active_pipeline.py`, `parser.py`, `templates.py`, `basic_config.yaml`. See [program.md](../program.md) for the full rules.
- **Single-run noise is real** (~3 pp at n=125 on Bamboogle when temperature>0). Treat improvements <2 pp EM with skepticism; confirm with a second seed or use temp=0 (greedy).
- **Cite paper sections by URL** when discussing results, especially `arxiv.org/html/2503.09516v5#A6` (Table 3, the GRPO comparison).
- When in doubt about a dataset split or a Search-R1 implementation detail, check the official repo (`github.com/PeterGriffinJin/Search-R1`) before guessing — and ask me before invoking AlphaXiv or other web tools that publish content.

## What's next

**Immediate** — close the base-variant gap before Plan A. In order from [COMPARISON_PLAN_B.md#recommended-next-steps-before-plan-a](../docs/COMPARISON_PLAN_B.md#recommended-next-steps-before-plan-a):

1. **(in flight)** Re-run base on NQ-1k + TriviaQA-1k with `apply_chat=True`. Bamboogle probe closed the gap exactly via this single flag flip (EM 0.112→0.128, format-validity 84%→100%). [PAPER_VS_OURS_AUDIT.md D1](../docs/PAPER_VS_OURS_AUDIT.md#d1-in-detail-the-load-bearing-one) is the smoking gun.
2. (~5 min) Add the missing `For example, <answer> Beijing </answer>.` sentence back to [`templates.py`](../evaluation_search_r1/flashrag/search_r1/templates.py) (audit D-prompt-micro). One-line edit; LOW severity but free.
3. (~5 min) Remove the `add_special_tokens` block in [`active_pipeline.py:37-42`](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py#L37-L42) (audit D8). Re-run a smoke to confirm `</search>` still matches as a stop string.
4. (~30 min) Tabulate `format_valid` / length-truncation rate from existing Plan B run JSONs per (dataset, variant). Uses data we already have.
5. (~2 h) Re-run base on NQ-1k with `step_limit` 500→1024; only if (1) leaves a residual gap.
6. (~4 h) One-seed full-NQ base run; confirms whether the gap survives at scale.
7. (~1 day) Add 2nd and 3rd seed to Plan B for variance bracketing.
8. Only then: Plan C (3.4 days, full data, 1 seed) or Plan A (17 days, full sweep — see [docs/VAST_AI_PLAN_A.md](../docs/VAST_AI_PLAN_A.md) for the ≤24 h Vast.ai fleet plan).

Do **not** change `temperature` or `top_p` — paper eval is greedy. See note above.

**After Phase 1** — ablations on the autoresearch branch (see `program.md`) and starting on **ReSearch** (the second paper). Append to this section as the plan firms up.
