---
title: RESULTS — M5.1-prod-a4 (Qwen3.5-0.8B GRPO on MuSiQue, H200 Spheron + persistent volume)
tags: [results, m5.1, h200, production, a4]
source: internal
created: 2026-05-15
updated: 2026-05-15
status: live — v11 running on patched native Triton GDN; step 2 landed
---

# M5.1-prod-a4 — Production training on Spheron H200 with persistent volume

> **Status**: smoke launched 2026-05-15 ~11:18 UTC after host migration to H200 + virtiofs persistent volume. Follows 4 prior M5.1 losses (a1/a1b/a2/a3) — see [`RESULTS_M5_1_B200.md` §10](RESULTS_M5_1_B200.md) for the a3 incident report. This run uses the new bulletproof uploader + external monitor + dual-repo backup + persistent volume to make host-loss recovery structurally possible.
>
> **Key changes vs a3**:
> - Hardware: 1× H200 SXM 141GB (vs B200 192GB) on Spheron with **persistent volume** (`miletone5`, 600 GB, virtiofs)
> - All artifacts (model, corpus, indexes, venv, ckpts, rollouts) live on `/mnt/miletone5/workspace/` — survives host preemption
> - Uploader rewritten in Python ([`upload_a4_to_hf.py`](../../training_m5_1/scripts/upload_a4_to_hf.py)) with heartbeat, dual-repo, ckpt priority, pre-flight
> - External monitor running on user's local Mac, polls HF every 5 min
> - Pre-launch checklist enforced ([`LAUNCH_CHECKLIST_A4.md`](../../training_m5_1/scripts/LAUNCH_CHECKLIST_A4.md))
>
> **Live trajectory**: this doc is the canonical narrative. Cadence checkpoints every 10 steps. Auto-committed to `experiment_1_h200` + pushed to GitHub + HF Hub. Pulled from a3's documented mistakes — no "uploader ✓" claims without verification, cadence triggers fire on step-file landing not on user prompt.

## 1. Run identity

| Field | Value |
|---|---|
| Run name | `qwen3.5-0.8b-musique-h200-a4-seed42` |
| W&B project | `reason_over_search_h200` |
| HF Hub repo (primary) | [`pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42`](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42) |
| HF Hub repo (backup) | [`pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-backup`](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-backup) |
| Git branch | `experiment_1_h200` (forked from `experiment_1_b200` at commit `178d5db`) |
| Started commit | `48d9a64` (h200 yaml updates) |
| Launch time (UTC) | 2026-05-15 19:24 (v11, after 8 prior prod-launch iterations; see §7.5) |
| Process pid (container) | 58135 → 60161 (DTensorPolicyWorkerV2) + 59638 (VllmGenerationWorker) |
| Container | `h200-a4` on host `204.12.169.243` |
| Seed | 42 |
| Hardware | 1× NVIDIA H200 SXM 141GB, Spheron, persistent volume |
| Driver / CUDA | 580.126.09 / 13.0 (pre-installed by Spheron image) |
| Persistent volume | `/mnt/miletone5` → `/workspace` (virtiofs, 600 GB) |
| Container image | `pantomiman/reason-over-search-v1:v2` (46.4 GB) |
| Training framework | NeMo-RL v0.6.0, torch 2.10.0+cu129, vllm 0.17.1 |
| Model | `Qwen/Qwen3.5-0.8B` (hybrid; GatedDeltaNet + attention) |

## 2. Config — diff vs a3

Branch `experiment_1_h200` diverges from `experiment_1_b200` only in the W&B/swanlab/mlflow project + run names (B200 → H200 a4). The training hyperparameters (config knobs) are unchanged from a3 (which was producing higher reward than A100 reference).

| # | Knob | a3 (B200) | a4 (H200) | Why |
|---|---|---|---|---|
| 1 | `logger.wandb.project` | `reason_over_search_b200` | `reason_over_search_h200` | New W&B project |
| 2 | `logger.wandb.name` | `qwen3.5-0.8b-musique-b200-a3-seed42` | `qwen3.5-0.8b-musique-h200-a4-seed42` | New run name |
| 3 | `policy.train_micro_batch_size` | 2 | **2** (unchanged) | a3 producing higher reward than A100 ref — keep it |
| 4 | `policy.generation.vllm_cfg.gpu_memory_utilization` | 0.8 | **0.8** (unchanged) | 0.8 × 141 GB = 113 GB vLLM cache — plenty for 0.8B model |
| 5 | `grpo.max_num_steps` | 622 | 622 | 2 epochs of MuSiQue at 64 prompts/step |
| 6 | `checkpointing.save_period` | 50 | 50 | 12 ckpts at steps 50/100/.../600 |

All other paper-faithful hyperparameters (lr=1e-6, kl_penalty=0.001, ratio_clip=0.2, temperature=1.0, max_total_sequence_length=8192) are unchanged.

## 3. Persistent volume layout (`/mnt/miletone5/workspace/` on host = `/workspace/` in container)

```
/workspace/
├── reason_over_search/         # Git clone of experiment_1_h200 (931 MB; LFS pulled)
│   ├── training_m5_1/
│   │   └── nemo_rl/.venv/      # ✓ 12 GB built on volume (torch 2.10, vllm 0.17, nemo_rl)
│   ├── data/training/musique/
│   │   └── train.parquet       # ✓ 1.59 MB (LFS-pulled, valid parquet)
│   ├── local_retriever/
│   │   ├── corpus/             # ✓ 14 GB wiki-18 corpus
│   │   ├── indexes/            # ✓ 16 GB IVF-SQ8 FAISS index
│   │   └── models/e5-base-v2/  # ✓ ~0.5 GB encoder
│   ├── logs/                   # rollouts land here as exp_NNN/train_data_stepN.jsonl
│   └── results/grpo/m5_*/      # checkpoints land here
├── hf_cache/                    # ✓ HuggingFace model cache (Qwen3.5-0.8B, 2B, e5-base-v2, ...)
└── state/                       # uploader state files
```

**Persistence test passed**: a test file written on the previous H200 instance (`computeinstance-u00pxp20247myv137q`) is visible on this instance (`computeinstance-u00v4wrfbry6nx616p`) → **the volume survives host swaps**.

## 4. Pre-launch safety nets (the new infrastructure)

### 4.1 Bulletproof uploader (`upload_a4_to_hf.py`)
- Python (not bash) — proper exception handling, no silent-failure paths
- **Heartbeat every cycle** in `uploader.log` (proves the loop is iterating)
- **Checkpoint folder priority** — `step_N/` pushes before logs/rollouts
- **Aggressive retry** with exponential backoff (5/15/45/90/180 s on transient errors)
- **Pre-flight canary** — uploads a tiny test file at startup, exits 2 if HF unreachable
- **Dual-repo** — uploads to both primary and backup namespaces simultaneously
- **5-second filesystem scan** for new `step_N/` folders (vs 60s in a3's bash uploader)
- **JSON state file** with atomic writes (vs plain-text append in a3)

### 4.2 External monitor (`external_monitor.py`)
- Runs on user's **local Mac**, not the training instance
- Polls HF every 5 min, cross-references with W&B step count
- Reports STALE if no commit in 10 min
- Reports LAG if W&B step is >10 steps ahead of HF rollouts
- Tested against the dead a3 run: would have caught the failure at minute 11, not 11 hours

### 4.3 Launch checklist (`LAUNCH_CHECKLIST_A4.md`)
Explicit gate-by-gate verification. Every box must be checked before training:
- HF_TOKEN + HF_REPO_ID + HF_BACKUP_REPO_ID set
- Both repos created on HF
- Uploader pre-flight passes on both repos
- External monitor running on user's local Mac
- Volume mount verified persistent (destroy + recreate test)

### 4.4 Pre-launch gates passed (for this run)
| Gate | Status |
|---|---|
| HF token set | ✓ in `.env` |
| Both repos created | ✓ (h200-a4-seed42 + h200-a4-backup) |
| Pre-flight uploads canary to both | ✓ both repos have `.preflight_canary.txt` |
| External monitor reports HEALTHY | ✓ verified at 11:15 UTC |
| Persistent volume mount tested | ✓ (file from previous host visible here) |
| NeMo-RL venv functional | ✓ torch 2.10 + cuda True + H200 visible |
| Retriever healthy | ✓ port 3005, 8/8 workers available |
| `.env` populated (WANDB_API_KEY, HF_TOKEN) | ✓ |
| Data: `train.parquet` is valid LFS-pulled parquet | ✓ (caught corrupt LFS pointer on first smoke attempt; fixed) |

## 5. Checkpoint plan

Saves every 50 steps. With `max_num_steps=622`, the canonical "final" checkpoint is **step_600** (the last save before max).

| Step | Save lands at (ETA from prod launch) | Cumulative spend (H200 @ $15.15/h) |
|---|---|---:|
| 50 | first ckpt, ~5 h from prod start | ~$76 |
| 100 | ~10 h | ~$152 |
| 200 | ~20 h | ~$303 |
| 311 (1 epoch) | ~30 h | ~$455 |
| 400 | ~40 h | ~$606 |
| 600 (last save) | ~60 h = 2.5 d | ~$909 |
| 622 (run end) | ~62 h = 2.6 d | ~$939 |

These are estimates based on smoke timing — will be revised after step 1 lands.

## 6. Cadence checkpoints

Every 10 steps. Each cadence:
1. Pull rollouts from W&B + HF
2. Run `analyze_rollouts.py` for window
3. Pick BEST / WORST / MEAN trajectories from the cadence-end step
4. Hand-analyze each (2-3 sentence commentary)
5. Append to this doc; commit to `experiment_1_h200`; push to GitHub
6. README on both HF repos updated
7. **Before ANY of the above**: verify uploader is healthy by checking heartbeat in `uploader.log` AND running external_monitor.py

## 7. Smoke verification status (DONE)

Smoke `qwen3.5-0.8b-musique-h200-a4-smoke-seed42` ran 4 steps in 17 min wall on 2026-05-15 ~14:30 UTC. Both checkpoints (step_2 + step_4, 6.4 GB each) landed on both HF repos within 15 s of step completion. Throughput 28-46 K input toks/s, 1.1-3.3 K output toks/s. All 7 gates above passed.

## 7.5. The v3-v11 prod incident chain (2026-05-15)

Between the smoke landing and step 1 of prod, we burned a half-day iterating through 8 prod-launch attempts. Each taught us something load-bearing; chain captured here so the next time agent can skip it.

| Attempt | Time | Config delta | Outcome | Lesson |
|---|---|---|---|---|
| **v3** | ~14:50 | async_engine=true, async venv freshly built by NeMo-RL | `ModuleNotFoundError: vllm` | NeMo-RL env-builder marks `STARTED_ENV_BUILDER` before `uv pip install` finishes torch; Ray actors then import too early. → Hardcopy sync venv (P2). |
| **v4 → v5** | ~15:20 | async venv hardcopy of sync venv | Step 1 ran 80 s gen / 1137 s total | Hardcopy workaround verified. v4 = proven async + mb=2 + ckpt=true on H200. |
| **v6** | ~16:00 | mb=2 → 4 + ckpt=true → false | OOM at 135.74 GB during policy training | H200 ceiling is real; mb=4 + ckpt=false too aggressive. |
| **v7** | ~16:30 | mb=4 → 2 (kept ckpt=false) | OOM at 13.85 GiB log_softmax | Qwen3.5's 248K vocab × fp32 logits = ~14 GB transient. ckpt=true is mandatory to free activation memory. |
| **v8** | ~17:00 | mb=2 + ckpt=true + async=true (= v4 + ckpt back on) | Step 1 + step 2 landed: 1282 s + 222 s gen | v4 baseline reaffirmed. async + Triton-or-FlashInfer path works at smoke + early-prod scales. |
| **v9** | ~17:46 | mb=2 → 4 + ckpt stays true + async=false | OOM 30.19 GiB log_softmax @ 25 GiB free | mb=4 + ckpt=true also too aggressive on H200 (b/c log_softmax + 248K vocab dominates anyway). mb=2 is the locked ceiling. |
| **v10b** | ~18:15 | mb=4 → 2 + ckpt=true + **async=false (sync)** | **STUCK 60+ min in FlashInfer GDN kernel** | THE critical learning. See below. |
| **v11** | ~19:24 | Same as v10b BUT patched `qwen3_next.py:156` to force `forward_native` (Triton) GDN path | **Step 1 = 1101.82 s; step 2 = 1123.09 s** | Patch works. Sync + Triton-GDN is the v11 production config. |

### 7.5.1 The FlashInfer GDN kernel deadlock (P12)

vLLM 0.17 hard-codes a Hopper-only fast path at [`qwen3_next.py:156`](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/model_executor/models/qwen3_next.py#L156):

```python
if current_platform.is_cuda() and current_platform.is_device_capability(90):
    self._forward_method = self.forward_cuda  # → flashinfer.gdn_prefill.chunk_gated_delta_rule
else:
    self._forward_method = self.forward_native  # → Triton kernel
```

- B200 (sm_100): never matches the if; always Triton. Why all prior B200 runs worked.
- A100 (sm_80): never matches the if; always Triton. Why M2 training never hit it.
- H200 (sm_90): matches the if; calls FlashInfer GDN. On small batches (smoke shape, async continuous-batching micro-batches) the kernel finishes. On **prod-scale sync prefill (309 prompts post-retrieval-injection with 3-6 K context each)** the kernel deadlocks inside CUDA. GPU stays at 100 % util; the Python decode loop never returns from `forward_cuda`.

**Symptom**: prod.log + vLLM worker .out both frozen for 30+ min while vLLM proc shows R state, 180-190 % CPU, GPU at 100 % util. No traceback. No tqdm refresh. py-spy stack dump (run from host with `sudo /tmp/py-spy dump --pid <host_pid>` after `docker cp h200-a4:/workspace/.../bin/py-spy /tmp/py-spy`) confirms:

```
gdn_prefill (flashinfer/gdn_prefill.py:63)
chunk_gated_delta_rule (flashinfer/gdn_prefill.py:207)
fi_chunk_gated_delta_rule (vllm/.../qwen3_next.py:138)
forward_cuda (vllm/.../qwen3_next.py:176)
... → execute_model → step → generate
```

**Fix**: `sed -i "s/if current_platform.is_cuda() and current_platform.is_device_capability(90):/if False:  # PATCHED .../"` on both the sync and async venvs' `qwen3_next.py`. Native Triton GDN prefill is ~10× slower per call than FlashInfer (turn 0 of 320 prompts: 62 s vs ~6 s), but it actually returns. Full patch instructions in [`docs/spheron/SETUP_SPHERON.md` §9.1](../spheron/SETUP_SPHERON.md). Pitfall narrative in [P12](../spheron/SETUP_SPHERON.md).

### 7.5.2 Why this was missed for so long

1. All prior B200 + A100 runs were on hardware that doesn't match the Hopper-only if.
2. Smoke shape uses smaller batches (kernel happy).
3. v8 used async engine; continuous batching submits 1-10 prompts at a time, never the full prod prefill batch.
4. The "v4 proven" config was tested on async + smoke shape, not sync + prod batch. The proof didn't transfer.

**Cost of the v10b stall**: ~75 min wall + a wrong "system stable" claim mid-session. Logged so it doesn't happen again.

## 8. Live trajectory

| Step | Wall (s) | Phase breakdown | Notes |
|---:|---:|---|---|
| 1 | 1101.82 | gen (incl. multi-turn) ~5-7 min · reward+adv ~10 s · logprob ~2-3 min · train ~8-10 min | First step on the patched native Triton GDN path. GPU peak 140 GiB (close to 141 GB ceiling). |
| 2 | 1123.09 | similar | Stable; +21 s vs step 1 likely from rollout-distribution noise. |
| 3 | in-flight (as of 2026-05-15 20:09 UTC) | | Rollout phase active. |

**Untrained-model rollout cost**: turn 0 starts at 320 prompts; ~10 prompts emit `<answer>` per turn, so turn 1 = 310, turn 2 = 294, etc. Max turns = 10 (uncapped per user direction). Step time will decrease as the model learns to emit `<answer>` earlier (typically within first 50-100 steps based on b200 trajectory).

## 9. Cost / wall-clock estimate

H200 SXM 141GB at Spheron at $15.15/h. H200 is roughly:
- 0.6× B200's memory bandwidth (4.8 vs 8 TB/s)
- 0.44× B200's BF16 dense compute (989 vs 2250 TFLOPS)
- 1.76× A100's memory bandwidth (4.8 vs 2 TB/s)

Expected per-step wall vs measured anchors:
- A100 mean (a3 reference): ~24 min/step → ~250 h total
- B200 mean (a3 measured, 56 steps): ~9 min/step → projected ~95 h total
- H200 projected (BW-scaled from B200): ~14-16 min/step → ~150-165 h total ≈ 6-7 days

But: H200's 141 GB VRAM may allow `gpu_memory_utilization` headroom that helps, and our actual workload is bandwidth-bound not compute-bound — so H200 may perform closer to B200 than the compute ratio suggests.

**Estimate range for full run (622 steps)**: **2.5-7 days, $900-2500**. Revised after step 1.

## 10. The four prior losses (recap)

| # | When | Cost | Cause | Mitigation in a4 |
|---|---|---|---|---|
| a1 step-50 ckpt crash | 2026-05-11 | ~$30 | `metric_name: "train/loss/mean"` violated NeMo-RL assertion | `metric_name: null` + smoke-verified |
| a1 rollout-corpus deletion | 2026-05-11 | 196 MB data loss | `rm -rf logs/exp_010` (prod) bundled with cleanup | Rollouts uploaded to HF (now to TWO repos) so single-machine deletion can't kill them |
| a2 zombie GPU misdiagnosis kill | 2026-05-12 | ~$21 | Misread `[Not Found]` in nvidia-smi | Kill-on-evidence rule |
| **a3 Spheron T3 spot preemption** | **2026-05-15** | **~$36 + step_50 ckpt** | **Spot host preempted ~step 56 + uploader silently dropped rollouts past step 18** | **Persistent volume + bulletproof uploader + external monitor + dual-repo backup** |

**Cumulative loss across M5.1**: ~$108 + 196 MB + 1 trained checkpoint.

## 11. Pointers

- W&B project: [`reason_over_search_h200`](https://wandb.ai/gaurisankarj1996-leiden-university/reason_over_search_h200)
- HF Hub primary: [`pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42`](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42)
- HF Hub backup: [`pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-backup`](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-backup)
- Prior run (crashed): [a3 RESULTS](RESULTS_M5_1_B200.md) (W&B run [h68uskz6](https://wandb.ai/gaurisankarj1996-leiden-university/reason_over_search_b200/runs/h68uskz6))
- Config: [`training_m5_1/configs/m5_1_research_paper.yaml`](../../training_m5_1/configs/m5_1_research_paper.yaml)
- Smoke config: [`training_m5_1/configs/m5_smoke.yaml`](../../training_m5_1/configs/m5_smoke.yaml)
- Uploader: [`training_m5_1/scripts/upload_a4_to_hf.py`](../../training_m5_1/scripts/upload_a4_to_hf.py)
- External monitor: [`training_m5_1/scripts/external_monitor.py`](../../training_m5_1/scripts/external_monitor.py)
- Launch checklist: [`training_m5_1/scripts/LAUNCH_CHECKLIST_A4.md`](../../training_m5_1/scripts/LAUNCH_CHECKLIST_A4.md)
- Hardware comparison: [`../setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md)
