---
title: ALICE hardware fixes — M5.1/M5.5/M5.6 sbatch hardening session
tags: [milestone-5, alice, sbatch, hardware, postmortem]
source: live working session 2026-05-12 (experiment_1_alice branch)
created: 2026-05-12
updated: 2026-05-12
---

# Hardware fixes 2026-05-12 — getting the M5 sbatches to actually work on ALICE

Single-session log of the bugs caught + fixed before the 6 M5 production
sbatches (M5.1 / M5.5 / M5.6 × {1× A100, 2× A100}) were resubmitted. Sticks
to facts, not aspiration: each fix is paired with the symptom that surfaced
it and the verification status.

The doc lives in [`docs/milestone_5/`](.) alongside the M5 narrative; see
also [`../report/RESULTS_SMOKE_m5.md`](../report/RESULTS_SMOKE_m5.md) for the
research_v2 postmortem that seeded this work.

## 1. Context (why 8 bugs in one day)

The M5.1 / M5.5 / M5.6 sbatches were authored on Vast.ai 1× A100 + 24 GB local
NVMe; they passed there. ALICE has a different stack:

- Apptainer SIF instead of Docker (read-only rootfs, different bind semantics)
- `/zfsstore` NFS-style storage (~40 MB/s reads, vs Vast.ai's ~1 GB/s NVMe)
- 252 GB RAM nodes (vs Vast.ai's 64 GB)
- 64 CPUs per node (vs Vast.ai's 16-32)
- SLURM SBATCH headers (Vast.ai used bash directly)

Most fixes are environment-portability issues that the Vast.ai validation
didn't expose. The remaining ones are correctness bugs that production-shape
testing would have hit on either platform.

## 2. Bugs caught + fixes applied (in order discovered)

| # | Bug | Surfaced via | Fix | Commit |
|---|---|---|---|---|
| 1 | Venv pre-check `[ -x .../python ]` follows symlink into `/.uv/python_install/...` (container-only), fails on host | first in-srun smoke (rc=1 in 1 s) | `-d` on dir instead of `-x` on python | `8b8525d` |
| 2 | M5.5/M5.6 2x sbatch pre-checks referenced `m5_1_research_paper_2xa100.yaml` (wrong filename) | grep audit during deep audit | rename refs to `m5_5_…` / `m5_6_…` | `8b8525d` |
| 3 | Ckpt config crash: `metric_name: "train/loss/mean"` doesn't start with `train:`/`val:`; `keep_top_k: 0` would delete every save | already documented in research_v2 (a1 prod crashed at step-50 first save, ~19.5h compute lost) | `metric_name: null`, `keep_top_k: null`, `save_optimizer: false` (3.2 GB/save vs 8.9 GB) | `3aa04ff` |
| 4 | HF watcher had no per-experiment namespace → all 3 experiments would push to the same HF repo names | code review while porting watcher to M5.5/M5.6 | repo name = `${HF_REPO_PREFIX}-m5_X_${MODE}-seed${SEED}-step${N}` per experiment | `3aa04ff` |
| 5 | Smoke yamls had `enabled: false` for ckpt → smoke didn't actually exercise the save round-trip (the load-bearing fix from research_v2) | code review while applying fix #3 | smoke yamls flipped to `enabled: true`, `save_period: 2`, `max_num_steps: 4` | `3aa04ff` |
| 6 | Retriever `/health` timeout hard-coded 1800 s → not enough for cold zfsstore (8 retrievers × ~7 min first-cold-read = >30 min) | in-srun smoke timeout at exactly 30 min | bump default to 5400 s; override via env var | `8363b79`, `456171f` |
| 7 | HF watcher's apptainer exec didn't forward `CHECKPOINT_DIR_BASE` → watcher reads `results/grpo` while trainer writes `results/grpo_m5_5` (for M5.5/M5.6) | deep audit before re-fire | `--env CHECKPOINT_DIR_BASE=…` in watcher apptainer exec | `9296123` |
| 8 | `--mem=160g` (1x) / `200g` (2x) too small: `faiss.read_index()` loads full 16 GB index per retriever → 8 × 17.5 GB = 140 GB + 60 GB trainer + 15 GB overhead = ~215 GB | user caught my wrong "17 GB total" reading; verified at "Initializing retriever 7/8" with RSS=105 GB | bump `--mem=240g` for all 6 sbatches (node has 252 GB total) | `eb2e6b1` |
| 9 | Ray's default `_temp_dir=/scratchdata` is read-only inside the apptainer SIF rootfs → `OSError: Read-only file system: '/scratchdata'` during `ray.init()` | smoke verification rc=1 in <1 min, before training start | bind-mount `${REPO_ROOT}/logs/ray_scratch_${RUN_ID}` → `/scratchdata` in apptainer | `e6ddd9d` |
| 10 | DTensor `tensor_parallel_size: 1` in all 3 2xa100 yamls (DDP across 2 GPUs, not TP=2) → skipped ~70% of the projected 2× A100 speedup | user reading HARDWARE_COMPARISON.md §3, vs current yaml | flip DTensor TP=1 → 2 in all 3 2xa100 prod yamls; kept micro=1 pending 2-GPU smoke | `68f6bd4` |
| 11 | DTensorPolicyWorkerV2 venv empty on disk (skeleton from May 2 only, no payload) → `uv run --locked --extra automodel` fails because the env-builder Ray actor is GPU-less and nv-grouped-gemm's setup.py calls `torch.cuda.init()` | re-fired smoke; "DTensorPolicyWorkerV2/lib/python3.13/..." not found | download `dtensor_policy_worker_v2.tar.gz` (6.5 GB) from `pantomiman/reason-over-search-v1-venvs` and extract (~12 GB) via SLURM `cpu-short` job | runtime fix; no commit |
| 12 | Sbatch arg parser didn't accept `--` for Hydra overrides → couldn't pass `grpo.max_num_steps=1` for a prod-shape OOM check | designing the 2x verification stage 4 | add `--) shift; EXTRA_OVERRIDES=("$@"); break ;;` case in sbatch_m5_1_2xa100, forward to run.sh | `596afe6` |

## 3. The CPU + RAM math (now sized correctly)

A 0.8B model's compute footprint is small; the load-bearing memory is the
retriever's FAISS index and the trainer's DTensor activation peak.

**Per-process RAM**:
- Retriever: ONE Python process with N async workers (semaphore-bounded). `faiss.read_index()` loads the full 16 GB IVF-SQ8 index into RAM **per retriever wrapper**. 8 retrievers = 8 × ~17.5 GB = ~140 GB.
- DTensor trainer: ~60 GB peak (micro=1, seq=8192, 320 traj/step) — measured in research_v2's v7 smoke.
- vLLM + Ray driver + Python overhead: ~10-15 GB.
- **Total peak**: ~215 GB.

**CPU**:
- Retriever: 8 async workers × `OMP_NUM_THREADS=4` = up to 32 concurrent FAISS threads (semaphore-bounded).
- Trainer (vLLM + DTensor + Ray): ~10 CPUs in flight during a step.
- **Total peak**: ~42 cores.

**Node capacity** (gpu-a100-80g):
- 64 CPUs, 252 GB RAM, 2-3 A100s per node.
- 1× A100 allocation: 40 CPUs + 240 GB → tight on CPU at peak, comfortable on RAM (12 GB OS buffer).
- 2× A100 allocation: 48 CPUs + 240 GB → comfortable on both.

## 4. zfsstore is the throughput bottleneck

Measured (in-srun on node875 via `dd` skip-512-count-128 8 MB blocks):
- zfsstore: **~40 MB/s sequential read** (cold cache).
- /tmp on node875: ~200 MB/s sequential write; reads faster from Linux page cache.

Implications:
- First retriever's FAISS load: 16 GB / 40 MB/s = **~7 min** cold.
- Subsequent retrievers benefit from warm page cache: ~1-2 min each.
- 8-retriever boot: **~7 + 7×~2 = ~21 min worst-case** cold; ~5-8 min on a hot node.
- Bumped `RETRIEVER_HEALTH_TIMEOUT_S` to **5400 s (90 min)** as 50 % margin.

Further workarounds NOT applied yet (defer until needed):
- Pre-warm via `dd if=index of=/dev/null` before retriever launch (deterministic ~7 min).
- Copy index to compute-node `/tmp` (199 GB free on node875) → all workers read from local fast storage.
- FAISS mmap mode (`IO_FLAG_MMAP`) — requires editing `retriever_serving.py`, not done.

## 5. Verification matrix (what we know works vs trust)

| Verification | Status | What it proves | What it does NOT prove |
|---|---|---|---|
| **Plan B**: in-srun 1x smoke, num_retriever=4, 120 GB allocation | RUNNING (m5_1 ✅ PASSED rc=0 with ckpt + WANDB + HF upload of step_2 to `pantomiman/qwen3.5-0.8b-grpo-musique-m5_1_smoke-seed42-step2`; m5_5/m5_6 in progress) | sbatch scaffold + ckpt + HF + WANDB + cleanup trap | 8-retriever shape; --mem=240g allocation |
| **sbatch 2274508**: 1x A100 + 240g + 8 retrievers, gpu-short, fires `m5_verify_seq.sh` | QUEUED, StartTime 2026-05-13 08:23 CEST | Full 8-retriever + 240g shape (closes Plan B gap) | 2x shape |
| **sbatch 2275683**: 2x A100 + 240g, gpu-short, fires `m5_verify_seq_2xa100.sh` (3 smoke_2xa100 + stage 4 m5_1 prod_2xa100 1-step OOM check) | QUEUED, StartTime 2026-05-13 13:50 CEST | DTensor TP=2 + full prod batch shape (320 traj, seq=8192, micro=1) doesn't OOM | micro=2 unlock (not flipped yet) |
| **srun 2275986**: 2x A100 + 240g, interactive shell, 4h walltime | QUEUED, StartTime 2026-05-13 17:50 CEST | Hands-on poking after sbatch 2275683 completes | n/a |

## 6. Wall-clock projection for 2× A100 prod (does the run fit a single 7-day sbatch?)

Anchors from HARDWARE_COMPARISON.md §3:
- 1× A100 anchor: ~23.7 min/step (a1 step-38-42, "improve-and-grow" phase)
- 2× A100 TP=2 projection: ~8.8 min/step with micro=2 unlock; ~15-17 min/step at micro=1

| Config | Per-step | 622 steps | Fits 7-d walltime cap (168 h)? |
|---|---:|---:|---|
| 1× A100 (anchor, micro=1) | 23.7 min | ~246 h (10.2 d) | **NO** — needs 2 resumes |
| 2× A100 TP=2, **micro=1** (current 2xa100 prod yaml) | ~15.8 min | ~164 h (6.8 d) | **MARGINAL** — ~4 h buffer |
| 2× A100 TP=2, **micro=2** (if 2-GPU smoke confirms no OOM) | ~8.8 min | ~91 h (3.8 d) | **COMFORTABLE** — ~77 h buffer |

So the 2× A100 prod runs **probably fit one srun**, but only with confidence after either:
- Tomorrow's stage-4 OOM check at micro=1 passes AND step-time stays ≤16 min/step at steady state; OR
- A follow-up smoke at micro=2 demonstrates no OOM, after which we flip the prod yaml.

The 5-7 day firm-StartTime buffer on the 2× A100 prod sbatches gives time
to do the micro=2 smoke and flip if we want the comfortable budget.

## 7. Sensitivity to a3 step-time (rollout-length drift)

a1's anchor was during a phase where rollout length grew toward ~1000 tokens.
If a3's steady state is shorter (more terminal answers, fewer search turns)
the per-step time drops; if longer, it grows. Scaling table (linear in step
time at fixed batch shape):

| 1× A100 anchor | 2× A100 TP=2 wall (micro=2) | 1-srun fit on 2× A100? |
|---|---:|---|
| 15 min/step | ~2.4 d / ~58 h | YES, big buffer |
| 20 min/step | ~3.2 d / ~77 h | YES, comfortable |
| 23.7 min/step (a1 anchor) | ~3.8 d / ~91 h | YES, 77 h buffer |
| 30 min/step | ~4.8 d / ~115 h | YES, 53 h buffer |

So even at the high end (30 min/step), a single 7-day 2× A100 srun completes
the 622-step run with the micro=2 unlock. **Without** micro=2 we lose the
buffer; whether one srun suffices then depends on whether a3 stays under
~23 min/step at steady state.

## 8. Open issues (not blocking today's resubmit)

- **HF watcher container-lifetime**: in Plan B m5_1 smoke, the watcher uploaded step_2 to HF Hub successfully, then squashfuse_ll (apptainer's SIF mount process) timed out before the step_4 upload cycle. In prod (12 saves over 25 days), only the first 1-2 ckpts would upload before the container dies. Fix: run watcher OUTSIDE apptainer with system python + `huggingface_hub`, OR keep the SIF mount alive via heartbeat. Tracking before May 15.
- **micro=2 unlock**: kept conservative micro=1 in 2xa100 prod yaml. A 2-GPU smoke at micro=2 would unlock the comfortable 1-srun budget. Deferrable; the firm StartTimes (May 15+) leave time.

## 9. Files touched today

```
training_m5_1/configs/m5_1_research_paper.yaml         # ckpt fix (already on research_v2, merged in)
training_m5_1/configs/m5_1_research_paper_2xa100.yaml  # ckpt + DTensor TP=2
training_m5_1/configs/m5_smoke.yaml                    # ckpt-enabled, save_period=2, max_num_steps=4
training_m5_1/configs/m5_smoke_2xa100.yaml             # NEW (Option C-1)
training_m5_1/scripts/sbatch_m5_1.sh                   # venv check, timeout, /scratchdata, --mem=240g, HF watcher wired, cleanup trap
training_m5_1/scripts/sbatch_m5_1_2xa100.sh            # all of the above + -- pass-through for Hydra overrides
training_m5_1/scripts/run.sh                           # new smoke_2xa100 mode case
training_m5_1/scripts/upload_ckpts_watcher.sh          # per-experiment repo namespace (-m5_1_ prefix)
training_m5_1/.env                                     # populated WANDB + HF_TOKEN + HF_REPO_PREFIX (gitignored, on ALICE)
training_m5_1/.env.example                             # already had HF doc from research_v2 merge

training_m5_5/configs/m5_5_research_paper.yaml         # ckpt fix
training_m5_5/configs/m5_5_research_paper_2xa100.yaml  # ckpt + DTensor TP=2
training_m5_5/configs/m5_smoke.yaml                    # ckpt-enabled
training_m5_5/configs/m5_smoke_2xa100.yaml             # NEW (C-1)
training_m5_5/scripts/sbatch_m5_5.sh                   # same template as m5_1
training_m5_5/scripts/sbatch_m5_5_2xa100.sh            # same template as m5_1 2xa100
training_m5_5/scripts/run.sh                           # new smoke_2xa100 mode case
training_m5_5/scripts/upload_ckpts_watcher.sh          # NEW (porte from m5_1, -m5_5_ prefix)
training_m5_5/.env                                     # symlinked to training_m5_1/.env

training_m5_6/* (mirror of training_m5_5 with m5_6 substitutions)

docs/milestone_5/HARDWARE_FIXES_2026-05-12.md          # THIS doc
```

Branch: `experiment_1_alice`. Commits 8b8525d → 596afe6 (12 commits today).

## 10. References

- [`../report/RESULTS_SMOKE_m5.md`](../report/RESULTS_SMOKE_m5.md) — research_v2's a1 / a2 postmortem (ckpt fix originates here, §7)
- [`../setup/HF_CHECKPOINT_UPLOAD.md`](../setup/HF_CHECKPOINT_UPLOAD.md) — HF watcher operator doc (research_v2)
- [`../setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md) §3 — 2× A100 wall-clock projection
- [`MILESTONE_5.md`](MILESTONE_5.md) — M5 narrative + run history
