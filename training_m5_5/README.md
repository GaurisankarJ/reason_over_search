# training_m5_5

GRPO training scaffold for **milestone M5.5 — F1+format reward ablation** on Qwen3.5-0.8B / MuSiQue. This is the **F1 + 0.1 partial-credit floor + format-gate** variant of the reward-shape ablation triad:

- **M5.1** ([`training_m5_1/`](../training_m5_1/)) — F1-only baseline (the existing in-flight run).
- **M5.5** (this directory) — F1 + 0.1 floor when format valid but F1=0 (paper-faithful 3-tier shaping).
- **M5.6** ([`training_m5_6/`](../training_m5_6/)) — EM-only sibling.

Every knob except the reward function is byte-identical to M5.1 — this is a mechanism ablation, not a "which is best" race. Narrative + design decisions: [docs/milestone_5/MILESTONE_5_5.md](../docs/milestone_5/MILESTONE_5_5.md). Paper-vs-ours mapping: [docs/milestone_5/PAPER_VS_OURS_M5.md](../docs/milestone_5/PAPER_VS_OURS_M5.md). Audit of what differs from M2: [docs/report/CODE_SETUP_m5.md](../docs/report/CODE_SETUP_m5.md).

## What this milestone answers

Does the ReSearch paper's 0.1 partial-credit floor (`reward=0.1` when format valid but F1=0) change end-of-training behaviour and held-out EM/F1 relative to F1-only and EM-only, on Qwen3.5-0.8B / MuSiQue? Motivation: [PHASE_1_SALVAGE.md Finding 1](../docs/milestone_6/PHASE_1_SALVAGE.md#finding-1--the-01-partial-credit-floor-masks-tool-use-signal) observed the floor compresses tool-use signal into a 3-6 pp reward band; the paper never ablates it.

## Reward spec

```
reward(rollout) =
  0.0    if not is_valid_format(rollout)            # format broken
  0.1    if extract_solution(rollout) is None       # format ok, no <answer>
  0.1    if f1(answer, gold) == 0                   # format ok, answer wrong
  f1     if f1(answer, gold) > 0                    # format ok, partial / full
```

The floor is NOT additive (F1=0.5 → reward=0.5, not 0.6); it only applies when F1=0. `is_valid_format` is the ReSearch state-machine walker ported to Qwen3.5 nested-XML schema with `<answer>` (not `\boxed{}`). Full spec + ReSearch source-of-truth mapping in [src/rewards/search_r1.py](src/rewards/search_r1.py).

## What's different from M2

Same as M5.1: byte-aligned with the M4 eval pipeline ([`evaluation_qwen35/`](../evaluation_qwen35/)) so a trained checkpoint is directly evaluable.

| | M2 ([`training/`](../training/)) | M5.5 (this directory) |
|---|---|---|
| Target model | Qwen3.5-2B | **Qwen3.5-0.8B** |
| Training data | NQ + HotpotQA mix | **MuSiQue only** |
| Reward | EM + format shaping (Search-R1 paper-faithful) | **F1 + 0.1 floor (format-gated)** on `<answer>X</answer>`; no `\boxed{}` wrap |
| Parsers / scorer / tool schema | local copies of FlashRAG functions | **re-exported from `evaluation_qwen35`** so train and eval can't drift |
| Prompts | local `.txt` files (M2) | materialised from `evaluation_qwen35.flashrag.search_r1.templates` via `scripts/sync_m4_prompts.py` |
| W&B project | `reason-over-search` (shared) | `reason_over_search_m5_5` (per-experiment isolation) |

## What's different from M5.1

Per [MILESTONE_5_5.md §3](../docs/milestone_5/MILESTONE_5_5.md), every knob except the reward is byte-identical to [`training_m5_1/configs/m5_1_research_paper.yaml`](../training_m5_1/configs/m5_1_research_paper.yaml). The diff is confined to [src/rewards/search_r1.py](src/rewards/search_r1.py) (`FORMAT_BONUS = 0.1` + `is_valid_format` walker) and the W&B project name.

## Environment setup

`nemo_rl/` here is a **symlink** to `../training/nemo_rl/` (disk-constrained; the upstream vendor is 23 GB and `/workspace` only has ~25 GB free). The `setup.sh` flow follows the symlink and creates the venv inside it, which means **M5.5 and M2 share the same venv state**. That's fine while we're on the same NeMo-RL version (v0.6.0); if you ever bump NeMo-RL in one experiment, you must break the symlink first (otherwise the bump leaks to the other experiment).

```bash
cd training_m5_5
bash setup.sh                       # uv sync --extra vllm against ../training/nemo_rl/
source nemo_rl/.venv/bin/activate   # resolves through the symlink
```

The pre-warmed wheel cache at `/.uv/cache/` makes this fast on Vast (`pantomiman/reason-over-search-v1`). For ALICE HPC see [scripts/bootstrap_alice.sh](scripts/bootstrap_alice.sh) (Apptainer SIF flow).

## Data prep

```bash
# 1. MuSiQue train split (~20k rows, ~10 MB parquet).
python scripts/prep_musique.py
#   ↳ writes data/training/musique/train.parquet at the repo root.

# 2. M4 prompt template. Currently pre-staged at the M4.2 canonical
#    (`qwen35_minimal`). After M4.4 locks a new winner, re-run with --mode.
python scripts/sync_m4_prompts.py --list
python scripts/sync_m4_prompts.py --mode qwen35_minimal
#   ↳ writes src/prompts/m5_qwen35_user.txt (+ optional m5_qwen35_system.txt).
```

Both scripts are idempotent: re-running skips if outputs are already there.

## Configure W&B (one-time per instance)

```bash
cp training_m5_5/.env.example training_m5_5/.env
# edit training_m5_5/.env, fill WANDB_API_KEY (+ optional CHECKPOINT_DIR_BASE).
```

`training_m5_5/.env` is gitignored.

## Run

**Pre-flight**: retriever live at `127.0.0.1:3005` (see [`local_retriever/README.md`](../local_retriever/README.md)); `data/training/musique/train.parquet` present; `src/prompts/m5_qwen35_user.txt` present; `nemo_rl/.venv/` materialized.

```bash
# 50-step end-to-end smoke (20 traj/step). Target: <= 25 s/step on 1x A100-80GB.
bash scripts/smoke.sh
bash scripts/smoke.sh --seed 7

# Production run (M5.5 paper-faithful recipe; identical knobs to M5.1 except reward).
# configs/m5_5_research_paper.yaml is committed (paper-faithful + M5.2 system gains
# O1 fused AdamW + R2 vLLM async_engine). Schedule: 622 steps × 320 trajectories.
# Wall-clock: ~10-13 days on 1x A100-80GB per the live M5.1 trajectory
# (RESULTS_m5.md §4.1). Submitted to ALICE `gpu-a100-80g` partition at the 7-day
# cap via scripts/sbatch_m5_5.sh; resume-from-latest-ckpt across sbatch slots.
bash scripts/run.sh --mode prod --seed 42

# ALICE sbatch submission:
sbatch scripts/sbatch_m5_5.sh         # 1x A100-80GB
sbatch scripts/sbatch_m5_5_2xa100.sh  # 2x A100-80GB variant

# Hydra overrides:
bash scripts/smoke.sh -- policy.train_micro_batch_size=8 grpo.max_num_steps=20
```

W&B run name: `qwen3.5-0.8b-musique-m5_5_<mode>-seed<N>-<TS>`. Checkpoints land under `${CHECKPOINT_DIR_BASE:-results/grpo}/m5_5_<mode>/seed<N>/` every 50 steps (not timestamped, so resumes work).

## Folder layout

```
training_m5_5/
├── README.md                       # this file
├── setup.sh                        # uv venv + uv sync against nemo_rl/ symlink
├── .env, .env.example, .gitignore
├── nemo_rl/ -> ../training/nemo_rl/   # SYMLINK; shared with M2
├── src/                            # overlay; see docs/report/CODE_SETUP_m5.md §3.2 for the per-file diff
│   ├── chat_template/tools.py      # re-exports QWEN35_SEARCH_TOOL from evaluation_qwen35
│   ├── datasets/search_r1.py       # dataset-agnostic parquet adapter
│   ├── environments/parsers.py     # qwen_native parse_query delegates to evaluation_qwen35
│   ├── environments/search_r1_env.py # F1 fallback + near_em_rate metric + kwarg-dispatch fix
│   ├── processors/search_r1.py     # docstring updated; dataset-agnostic
│   ├── prompts/m5_qwen35_user.txt  # pre-staged from M4.2 canonical
│   ├── prompts/_archive_m2/        # M2 prompt files preserved for ablation
│   ├── rewards/search_r1.py        # **M5.5: F1 + 0.1 partial-credit floor + is_valid_format walker** (the ablation knob)
│   ├── parallel_plan_qwen35.py     # DTensor parallel plan for Qwen3.5 hybrid arch
│   └── registry.py                 # `training_m5_5.src.*` import paths
├── configs/
│   ├── m5_smoke.yaml               # M5 pipeline-validation smoke (20 traj/step × 50 steps)
│   ├── m5_smoke_2xa100.yaml        # 2x A100 smoke variant
│   ├── m5_5_research_paper.yaml    # M5.5 production config (LIVE; 622 steps, 320 traj/step)
│   ├── m5_5_research_paper_2xa100.yaml # 2x A100 production variant
│   └── _archive_m2/                # M2 2B configs
├── scripts/
│   ├── run.sh                      # --mode smoke|prod --seed N
│   ├── smoke.sh                    # alias for `run.sh --mode smoke`
│   ├── run_grpo.py                 # imports training_m5_5.src.registry, hands off to NeMo-RL examples/run_grpo.py
│   ├── prep_musique.py             # MuSiQue train -> data/training/musique/train.parquet
│   ├── sync_m4_prompts.py          # materialises M4 prompt mode into src/prompts/
│   ├── bootstrap.sh                # Vast bootstrap (M2-shape; rewrite for M5.5 when needed)
│   ├── bootstrap_alice.sh          # ALICE HPC Apptainer SIF bootstrap
│   ├── sbatch_m5_5.sh              # ALICE sbatch wrapper (1x A100-80GB, 7-day cap)
│   ├── sbatch_m5_5_2xa100.sh       # ALICE sbatch wrapper (2x A100-80GB)
│   ├── upload_ckpts_watcher.sh     # polling HF Hub uploader for new step_N/ ckpts
│   ├── extract_smoke_samples.py    # extract (prompt, response, reward) tuples per combo for SMOKE_RESULTS.md
│   └── _archive_m2/                # M2 wrappers preserved
├── tests/                          # pure-Python tests (reward + parser + format-helper)
└── fix/CHANGES.md                  # M2 changelog; M5 changes summarized in docs/report/CODE_SETUP_m5.md
```

`docs/training/CONVERSATION_CONTEXT.md` is the per-strand bootstrap doc for the training side.

## Sibling experiments (the reward-shape ablation triad)

M5.5 is the **F1+format** leg of a three-experiment ablation:

```bash
training_m5_1/   # F1-only baseline (in-flight on B200 Spheron, see experiment_1_b200)
training_m5_5/   # F1 + 0.1 floor + format gate (this directory)
training_m5_6/   # EM-only sibling
```

The W&B project key in each copy's yaml (`logger.wandb.project: reason_over_search_m5_<N>`) auto-isolates the runs. See [docs/milestone_5/MILESTONE_5_5.md §"Parallel experiments"](../docs/milestone_5/MILESTONE_5_5.md) for the design rationale; [docs/milestone_5/MILESTONE_5_6.md](../docs/milestone_5/MILESTONE_5_6.md) for the EM-only sibling.

## Pointers

- M5.5 milestone narrative: [docs/milestone_5/MILESTONE_5_5.md](../docs/milestone_5/MILESTONE_5_5.md)
- M5.6 sibling (EM-only): [docs/milestone_5/MILESTONE_5_6.md](../docs/milestone_5/MILESTONE_5_6.md)
- M5.1 baseline (F1-only, in flight): [docs/milestone_5/MILESTONE_5.md](../docs/milestone_5/MILESTONE_5.md), [training_m5_1/](../training_m5_1/)
- M5 paper-vs-ours mapping: [docs/milestone_5/PAPER_VS_OURS_M5.md](../docs/milestone_5/PAPER_VS_OURS_M5.md)
- M5 code-setup audit: [docs/report/CODE_SETUP_m5.md](../docs/report/CODE_SETUP_m5.md)
- M5 smoke results log: [docs/report/RESULTS_SMOKE_m5.md](../docs/report/RESULTS_SMOKE_m5.md)
- Motivation for the 0.1-floor ablation: [PHASE_1_SALVAGE.md Finding 1](../docs/milestone_6/PHASE_1_SALVAGE.md#finding-1--the-01-partial-credit-floor-masks-tool-use-signal)
- M2 NeMo-RL operational guide (still the foundation): [training/README.md](../training/README.md), [docs/training/README.md](../docs/training/README.md)
- M4 eval pipeline (the rollout shape we're aligned to): [docs/milestone_4/MILESTONE_4.md](../docs/milestone_4/MILESTONE_4.md), [docs/report/CODE_SETUP_m4.md](../docs/report/CODE_SETUP_m4.md)
- ReSearch paper: [arXiv:2503.19470](https://arxiv.org/abs/2503.19470); official code: [Agent-RL/ReSearch](https://github.com/Agent-RL/ReSearch)
