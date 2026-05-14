---
title: RESULTS — M5.1-prod-a3 (Qwen3.5-0.8B GRPO on MuSiQue, B200 Spheron)
tags: [results, m5.1, b200, production, a3]
source: internal
created: 2026-05-14
updated: 2026-05-14
status: live
---

# M5.1-prod-a3 — Production training on Spheron B200

> **Status**: Launch in progress (2026-05-14 ~15:00 UTC). Following 3 prior losses (a1 ckpt crash, a2 misdiagnosis kill, a1 rollout-corpus deletion); this is the verified-fix relaunch.
>
> **Live state**: this doc is the canonical narrative. Updated every 25 training steps; auto-committed to `experiment_1_b200` and pushed to GitHub + HF Hub.
>
> **External pointers**: W&B run, HF repo, smoke baseline, hardware comparison are listed at the bottom.

## 1. Run identity

| Field | Value |
|---|---|
| Run name | `qwen3.5-0.8b-musique-b200-a3-seed42` |
| W&B project | `reason_over_search_b200` |
| HF Hub repo | [`pantomiman/qwen3.5-0.8b-grpo-musique-b200-a3-seed42`](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-b200-a3-seed42) |
| Git branch | `experiment_1_b200` (locally + remote) |
| Started commit | `f0f960e` |
| Launch time (UTC) | 2026-05-14T15:03:36Z |
| W&B run | [`qwen3.5-0.8b-musique-m5_prod-seed42-20260514T1503Z`](https://wandb.ai/gaurisankarj1996-leiden-university/reason_over_search_b200/runs/h68uskz6) (run id `h68uskz6`) |
| Process pids | wrapper 7942, training 7961, uploader 7919 |
| Container | `m5_b200_a3` on host `46.243.145.4` |
| Seed | 42 |
| Hardware | 1× NVIDIA B200 SXM6, 192 GB VRAM (Spheron ES, ME West 1, $3.83/h spot) |
| Driver / CUDA host | 580.126.09 / 13.0 |
| Container | `pantomiman/reason-over-search-v1:v2` (CUDA 12.9 worker stack) |
| Image entrypoint | `/bin/bash` (Vast-specific boot bypassed) |
| Training framework | NeMo-RL v0.6.0 (vendored at `training/nemo_rl/`) |
| Model | `Qwen/Qwen3.5-0.8B` (hybrid; GatedDeltaNet + attention) |

## 2. Config — five B200 fixes applied vs `experiment_1_alice`

Branch `experiment_1_b200` diverges from `experiment_1_alice` only in `training_m5_1/configs/m5_1_research_paper.yaml`. Each fix is documented inline in that file.

| # | Knob | A100 value | B200 value | Why |
|---|---|---|---|---|
| 1 | `policy.train_micro_batch_size` | 1 | **2** | 192 GB fits log_softmax peak that OOM'd at 80 GB; ~halves training-phase wall |
| 2 | `policy.generation.vllm_cfg.gpu_memory_utilization` | 0.5 | 0.7 → **0.8** | vLLM sleep mode releases ~90% during training; 0.8 → ~154 GB allocated, ~15 GB resident. More KV cache for rollout batching. |
| 3 | `logger.wandb.project` (& swanlab, mlflow) | `reason_over_search_m5_1` | `reason_over_search_b200` | New W&B project to keep B200 runs separated from prior A100 attempts |
| 4 | `logger.wandb.name` (& swanlab, mlflow) | `qwen3.5-0.8b-musique-m5_prod-seed42` | `qwen3.5-0.8b-musique-b200-a3-seed42` | Reflects hardware + attempt number |
| 5 | (same as #2, two-stage bump) | — | — | — |

Unchanged (paper-faithful):
- `num_prompts_per_step: 64`, `num_generations_per_prompt: 5` (320 traj/step)
- `max_num_steps: 622` (2 epochs at 64 prompts/step from 19,938 MuSiQue rows)
- `max_total_sequence_length: 8192`
- `precision: bfloat16`
- `activation_checkpointing: true`
- All loss / KL / clip hyperparameters
- `save_period: 50`, `metric_name: null`, `save_optimizer: false`, `keep_top_k: null`

## 3. Checkpoint plan

Saves every 50 steps via NeMo-RL atomic save (writes `tmp_step_N/`, renames to `step_N/` on flush).

| Step | Save lands at (estimate, see §6 for live anchor) |
|---|---|
| 50 | (filled in) |
| 100 | (filled in) |
| 150 | (filled in) |
| 200 | (filled in) |
| 250 | (filled in) |
| 300 | (filled in) |
| 350 | (filled in) |
| 400 | (filled in) |
| 450 | (filled in) |
| 500 | (filled in) |
| 550 | (filled in) |
| 600 | (filled in) |

**Note on final step (per user decision Q1=b)**: `max_num_steps=622` is not a multiple of `save_period=50`, so the last save is at step 600. Steps 601-622 still train but their state is not preserved. The **step-600 checkpoint is the canonical "final" artifact** for evaluation. Trade-off accepted: ~3.5% of 2-epoch trajectory budget after the last save isn't checkpointed.

Each saved ckpt is ~3.2 GB (consolidated fp32 safetensors, no optimizer state per `save_optimizer: false`). Total local disk: 12 × 3.2 GB = 38.4 GB. Plenty of headroom (~400 GB free).

## 4. HF Hub backup — single-repo structure

The upload watcher ([scripts/upload_a3_to_hf.sh](../../training_m5_1/scripts/upload_a3_to_hf.sh)) is a new script (not the existing `upload_ckpts_watcher.sh` which targets per-step repos). Polls every 60 s, uploads new artifacts to a single HF model repo.

Target repo layout:

```
pantomiman/qwen3.5-0.8b-grpo-musique-b200-a3-seed42/
├── README.md                # this doc, mirrored
├── config_snapshot.yaml     # the prod yaml at launch
├── step_50/  ... step_600/  # NeMo-RL checkpoint dirs (1:1 with disk)
├── logs/
│   ├── prod.log             # full stdout/stderr from run.sh (overwritten each cycle)
│   ├── train_data/          # per-step rollout JSONLs
│   └── wandb_run.txt        # W&B run id + URL
├── timings.csv              # per-step wall + phase breakdown
└── final_metrics.json       # populated at run-complete
```

Watcher state file at `/workspace/.uploaded_artifacts`; idempotent across restarts.

## 5. Auto-restart wrapper

Training is launched via [scripts/run_prod_a3_resilient.sh](../../training_m5_1/scripts/run_prod_a3_resilient.sh), not directly via `run.sh`. The wrapper:

- Detects exit != 0 from `run.sh`
- Waits 60 s (lets GPU memory + Ray actors fully release)
- Relaunches; NeMo-RL auto-resumes from the latest `step_N/` ckpt in `checkpoint_dir`
- Caps at MAX_RESTARTS=5 to prevent infinite loops on persistent failure
- All restart events logged to `/workspace/prod.log` with `[wrapper]` tag

Designed for unattended multi-day runs where Spheron spot preemption + the rare framework exception both trigger restart.

## 6. Live trajectory

Per-step times will be logged here as they land. Anchor for extrapolation: smoke step 3 = 43 s at smoke shape (seq=4096, 20 traj/step, micro=2); prod shape (seq=8192, 320 traj/step, micro=2) is ~15× larger per step → target **~10-11 min/step** if smoke→prod scaling holds. Run wall projection: 622 × 10 min ≈ 100 h ≈ 4.2 d (could be 3-6 d depending on rollout-length dynamics, per the M5 §4.1 shrink-then-grow observation).

### 6.1 Step log

| Step | Wall (s) | Reward mean | Gen length | Tool calls | KL | Notes |
|------|---:|---:|---:|---:|---:|---|
| 1 | TBD | TBD | TBD | TBD | TBD | (filled at first step) |

### 6.2 Cadence checkpoints

Following user spec (Q6 = every 25 steps): full analysis update + git commit + HF push at step 25, 50, 75, 100, ... Each entry below is added live.

#### 2026-05-14 15:03 UTC — Launch ✓

| Item | Value |
|---|---|
| Wrapper started | 2026-05-14T15:03:36Z (via `run_prod_a3_resilient.sh`, attempt 1/6 fresh) |
| Training process | pid 7961, `python run_grpo.py --config m5_1_research_paper.yaml grpo.seed=42 ...` |
| W&B project | `reason_over_search_b200` ✓ |
| W&B run id | `h68uskz6` |
| W&B run name (auto-set by run.sh) | `qwen3.5-0.8b-musique-m5_prod-seed42-20260514T1503Z` (overrides yaml's `b200-a3-seed42` — known cosmetic issue, distinguishable via project + timestamp) |
| Training samples loaded | **311** ✓ (= floor(19,938 / 64), 1 epoch; max_num_steps=622 = 2 epochs) |
| Compute cluster | Ray, 1 node, colocated mode |
| Worker init mode | sequential (colocated; vLLM and DTensor share GPU via sleep/wake) |
| Setup phase observed at this snapshot | Building `VllmAsyncGenerationWorker` venv (~5 min one-time install of 468 packages on first prod use; cached after) |

**Pre-launch artifacts pushed**:
- HF repo created (private): https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-b200-a3-seed42
- README.md (this doc) + config_snapshot.yaml uploaded to HF
- prod.log initialized at /workspace/prod.log with launch metadata
- Upload watcher (pid 7919) polling every 60 s

**Watch-points for first step** (per smoke baseline):
- vLLM async engine setup: should complete in ~5-8 min on first prod use (smoke used `VllmGenerationWorker`, prod uses `VllmAsyncGenerationWorker` — additional one-time venv build)
- Total setup time on smoke was 182.7 s; prod will likely be 5-10 min total because the async worker venv builds + larger model context init
- First step wall: paper-shape (320 traj × seq=8192 × micro=2) extrapolates from smoke step 3 (43 s at 20 traj × seq=4096) by ~15× → **~10-12 min estimated**. Real number lands within the hour.

## 7. Spend tracker

Spheron B200 spot at $3.83/h on Spheron ES (verified from console).

| Phase | Hours | Cost |
|---|---:|---:|
| Setup + bootstrap + smoke (pre-launch) | TBD | TBD |
| Production training (live) | TBD | TBD |
| **Cumulative** | TBD | TBD |

Sunk cost on the first (broken) box: ~$2 (destroyed by user 2026-05-14).

## 8. The 3 prior losses being recovered from

Recap (full details in [`RESULTS_SMOKE_m5.md` §7 / §7.8 / §7.8.1](RESULTS_SMOKE_m5.md)):

| # | When | Cost | Cause | Fix verified by |
|---|---|---|---|---|
| a1 step-50 ckpt crash | 2026-05-11 | ~$30 | `metric_name: "train/loss/mean"` violated NeMo-RL colon-prefix assertion | smoke-ckpt-verify1 + smoke-ckpt-verify2 + the M5.1-prod-a3 smoke (this session) — all 4 saves landed cleanly |
| a1 rollout-corpus deletion | 2026-05-11 | 196 MB data permanently lost | Bundled `rm -rf logs/exp_010 logs/exp_011` (exp_010 was prod) | HF Hub upload rule (new): every per-step JSONL pushed to HF means no single-machine deletion can destroy it |
| a2 zombie GPU misdiagnosis kill | 2026-05-12 | ~$21 | Misread `[Not Found]` in `nvidia-smi --query-compute-apps` | New no-go rule: only authorize kills on confirmed evidence |

Total prior loss: ~$51 + 196 MB. M5.1-prod-a3 should bring the canonical first artifact home.

## 9. Pointers

- W&B run: (filled at launch — `https://wandb.ai/gaurisankarj1996-leiden-university/reason_over_search_b200/runs/<id>`)
- HF Hub repo: [`pantomiman/qwen3.5-0.8b-grpo-musique-b200-a3-seed42`](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-b200-a3-seed42)
- Smoke baseline (2026-05-14): smoke W&B + smoke ckpts at `/workspace/results/grpo/m5_smoke/seed42/` (step_2, step_4 — kept for reference)
- Prod config: [`training_m5_1/configs/m5_1_research_paper.yaml`](../../training_m5_1/configs/m5_1_research_paper.yaml)
- Upload watcher: [`training_m5_1/scripts/upload_a3_to_hf.sh`](../../training_m5_1/scripts/upload_a3_to_hf.sh)
- Resilient launcher: [`training_m5_1/scripts/run_prod_a3_resilient.sh`](../../training_m5_1/scripts/run_prod_a3_resilient.sh)
- Sibling iteration log: [`RESULTS_SMOKE_m5.md`](RESULTS_SMOKE_m5.md)
- Hardware comparison: [`../setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md)
- Catch-up TODO at launch time: [`../todo/TODO_2026-05-12.md`](../todo/TODO_2026-05-12.md)
