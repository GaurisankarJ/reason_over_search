---
title: SETUP CLAUDE
tags: []
source: internal
created: 2026-05-02
updated: 2026-05-06
---

# SETUP_CLAUDE.md — fresh-Vast bring-up, one-step

> **For Claude (the agent):** this doc is your runbook. The user has just
> spun up a Vast.ai instance on `pantomiman/reason-over-search-v1:v1`,
> cloned this repo into `/workspace/reason_over_search`, and invoked you
> with this file. Your job is to get them from "fresh box" to "training
> step printing rewards" in as few interactive turns as possible.
>
> Follow the steps in order. Where the doc says **ASK THE USER**, use the
> `AskUserQuestion` tool — don't make the choice yourself.

---

## 0. Sanity (≈10 s)

Run:

```bash
cd /workspace/reason_over_search && \
  pwd && which uv && conda env list 2>/dev/null | head && \
  nvidia-smi --query-gpu=name --format=csv,noheader && \
  df -h /workspace
```

Expect: cwd is the repo root; `uv` resolves; conda envs include
`retriever`, `evaluation_search_r1`, `main`; one A100/H100; ≥ 30 GB free.

If anything is missing, **STOP and tell the user**. Don't try to repair the
image — that's a docker-rebuild job, not a runtime fix.

## 1. Run bootstrap (≈30–45 min cold, < 1 min warm)

This is one command. It is idempotent — re-running just prints "already
done" for steps that are.

```bash
bash training/scripts/bootstrap.sh
```

Watch for the final line `▶ Bootstrap complete.`. If it errors out, read
the error and stop — bootstrap.sh checks for the conditions it needs and
fails loud.

What it does, in order (so you know what to expect):
1. Sanity (envs, disk, RAM, GPU) — instant.
2. Git LFS pull if `data/training/nq_hotpotqa_train/train.parquet` is missing.
3. Download Qwen3.5-2B-Base + Qwen3.5-2B to `$HF_HOME=/workspace/hf_cache`
   if not cached (~4 min, 8 GB).
4. `uv sync --extra vllm` to materialize `training/nemo_rl/.venv`
   (~2 min from the pre-warmed cache).
5. Download the pre-built v2/automodel uv venv (~5 GB) from
   `pantomiman/reason-over-search-v1-venvs` on HuggingFace Hub and extract it
   to `training/nemo_rl/venvs/.../DTensorPolicyWorkerV2/` (~3 min; fast path).
   Falls back to host-shell compile (~25 min) if HF download fails. **This step
   cannot be moved to the Ray actor that NeMo-RL would normally use, because
   `nv-grouped-gemm`'s setup.py calls `torch.cuda.init()` at install time
   and the actor has no GPU.**
6. Start the IVF-SQ8 retriever with 8 workers on port 3005, wait until
   `Uvicorn running` lands in `/tmp/retriever.log`, smoke-check the
   `/health` endpoint.

If the user has a `WANDB_API_KEY` they want to use, ensure it's in
`training/.env` before launching training. If they don't, prepend
`WANDB_MODE=disabled` to the launch command in step 3.

## 2. ASK THE USER which combo to run

Use `AskUserQuestion` — exact question and options below. Don't guess
based on context; the user picks.

Question text:

> Which training combo do you want to run? (Smoke = 2 outer steps × 4
> prompts × group=5; full = 1005 steps × 102 prompts.)

Options (single-select):
- `smoke base × qwen_native` — Qwen3.5-2B-Base, qwen-native chat template
- `smoke base × paper` — Qwen3.5-2B-Base, paper-style `<search>` template
- `smoke hybrid × qwen_native` — Qwen3.5-2B (hybrid soft-switch), qwen-native
- `smoke hybrid × paper` — Qwen3.5-2B (hybrid), paper template
- `smoke all 4` — run all four sequentially (~30 min total)
- `full base × qwen_native` — 1005 steps; **11–17 d** on this 1× A100 (see docs/training/SMOKE_RESULTS_2026-05-06.md "Full-training wall-clock + cost")
- `full custom` — they'll tell you variant + arm + seed + steps

Map their answer:

| Choice | variant | arm | extra Hydra args |
|---|---|---|---|
| `smoke base × qwen_native` | base | qwen_native | smoke knobs (below) |
| `smoke base × paper` | base | paper | smoke knobs |
| `smoke hybrid × qwen_native` | hybrid | qwen_native | smoke knobs |
| `smoke hybrid × paper` | hybrid | paper | smoke knobs |
| `smoke all 4` | loop | loop | smoke knobs, all 4 in sequence |
| `full base × qwen_native` | base | qwen_native | (no smoke knobs) |
| `full custom` | ASK FOLLOW-UP | ASK FOLLOW-UP | (compose) |

**Smoke knobs** (mandatory for any smoke):
```
grpo.max_num_steps=2 grpo.num_prompts_per_step=4 policy.train_global_batch_size=20
```

**Always-required Qwen3.5 overrides** (smoke OR full):
```
policy.sequence_packing.enabled=false
policy.dynamic_batching.enabled=true
policy.train_micro_batch_size=2
```
(Without these, GatedDeltaNet/Mamba layers crash with `CUDA illegal memory
access` during `get_logprobs`. See `training/fix/CHANGES.md`.)

## 3. Launch training

Single combo:

```bash
cd /workspace/reason_over_search
export HF_HOME=/workspace/hf_cache
ulimit -n 65536
rm -rf /tmp/ray

bash training/scripts/run_grpo_1xa100.sh \
  --variant <V> --seed 42 --arm <A> \
  -- \
  policy.sequence_packing.enabled=false \
  policy.dynamic_batching.enabled=true \
  policy.train_micro_batch_size=2 \
  <SMOKE_KNOBS_IF_SMOKE>
```

Run in foreground if the user wants to watch live. Run in background and
tail the log if the user prefers a quieter session. Either way, tell them
the W&B URL once it appears — `wandb: 🚀 View run at https://wandb.ai/...`
in the launcher output.

For `smoke all 4`, run the four combos sequentially (each one takes ~5 min
once the v2 venv exists). After each, `mv logs/exp_NNN
logs/smoke_<variant>_<arm>` so the trace dirs are named.

## 4. After training starts

- Successful step looks like: `========================= Step N/M =========================`
  followed by `Logged data to logs/exp_NNN/train_data_step{N}.jsonl`.
- For smokes, after step 2 the run exits cleanly with `EXIT=0` printed via
  the launcher.
- For full runs (1005 steps), checkpointing is **disabled** in the current
  config (`checkpointing.enabled=false`). Re-enable before kicking off a
  long run if the user wants to resume — see
  `docs/training/NEMO_RL_KNOBS.md`.
- W&B has the live curves; the per-step JSONL on disk has the raw
  trajectories.

## 5. Sample extraction (smoke only)

If the user ran the four smoke combos and wants the consolidated report:

```bash
python3 training/scripts/extract_smoke_samples.py
# writes /workspace/reason_over_search/docs/training/SMOKE_RESULTS.md
# (rename it to docs/training/SMOKE_RESULTS_<UTC-DATE>.md before committing
#  and update CONVERSATION_CONTEXT.md to point at it; previous dated runs live
#  under docs/archive/training/)
```

The script picks 8–9 samples per combo (mix of correct + incorrect),
truncates over-long trajectories, and tabulates reward distribution +
retrieval-call counts.

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| bootstrap.sh: `Conda env 'retriever' missing` | wrong docker image | Confirm `pantomiman/reason-over-search-v1:v1` (or rebuilt v1+) |
| bootstrap.sh: `nvidia-smi not found` | not a GPU instance | Vast template selection issue; pick a GPU machine |
| Retriever times out during rollouts | flat IP fallback by mistake | Confirm `local_retriever/retriever_config.yaml`'s `index_path` is the IVF; restart with `--num_retriever 8` (`bootstrap.sh` does this) |
| `KeyError: 'task_name'` in DataLoader | running pre-fix code | `git pull` — fix is in `training/src/datasets/search_r1.py` |
| `No actor environment registered for ... SearchR1Environment` | running pre-fix registry | `git pull` — fix is in `training/src/registry.py` |
| `unknown arg: --` from launcher | running pre-fix wrapper | `git pull` — fix is in `training/scripts/run_grpo_1xa100.sh` |
| `RuntimeError: No CUDA GPUs are available` (during uv install) | v2 venv being built inside Ray actor | bootstrap.sh handles this from host shell; if you bypass bootstrap, run step 3c from `training/fix/CHANGES.md` manually |
| `AttributeError: 'Qwen3_5Model' object has no attribute 'layers'` | `_v2: false` was overridden | Use the YAML default (`_v2: true`); ensure the v2 venv exists |
| `CUDA error: an illegal memory access` in `torch_chunk_gated_delta_rule` | sequence packing enabled with Qwen3.5 | Add `policy.sequence_packing.enabled=false` (always required) |
| Ray cluster startup timeout | stale `/tmp/ray` | `rm -rf /tmp/ray` and relaunch |

## Time + cost expectations (so you can level-set the user)

- Bootstrap on a fresh Vast box (HF download path): **~10 min** (Qwen weights ~4 min + v2 venv download/extract ~3 min + retriever cold start ~1 min).
- Bootstrap on a fresh Vast box (compile fallback, HF unavailable): **~35 min** (v2 venv compile is ~25 min).
- Bootstrap on a re-used box: **< 1 min**.
- Smoke combo (2 steps × 4 prompts × group 5): **~5 min** once v2 venv exists.
- Full Phase-2 run (1005 steps × 510 trajectories) on this 1× A100: **~17 d** linear, ~11 d sub-linear; ~$300–490 / run at $1.20/h.
- Recommended hardware for full runs: **1× H100 80 GB SXM** (~5–8.5 d, ~$240–410/run). See [`docs/training/SMOKE_RESULTS_2026-05-06.md` "Full-training wall-clock + cost"](SMOKE_RESULTS_2026-05-06.md#full-training-wall-clock--cost-phase-2-real-config).
