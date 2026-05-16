---
title: RESULTS — M5.1-prod-a4 (Qwen3.5-0.8B GRPO on MuSiQue, H200 Spheron + persistent volume)
tags: [results, m5.1, h200, production, a4]
source: internal
created: 2026-05-15
updated: 2026-05-15
status: live — v11 running on patched native Triton GDN; step 2 landed
---

# M5.1-prod-a4 — Production training on Spheron H200 with persistent volume

> **Status**: smoke launched 2026-05-15 ~11:18 UTC after host migration to H200 + virtiofs persistent volume. Follows 4 prior M5.1 losses (a1/a1b/a2/a3) — see [`RESULTS_M5_1_B200.md` §10](RESULTS_M5_1_B200.md) for the a3 incident report. This run uses the new bulletproof uploader + persistent volume to make host-loss recovery structurally possible. (The earlier draft also wrote to a redundant `-backup` HF repo; removed 2026-05-15 once we confirmed the Spheron volume already preserves all artifacts across host preemption.)
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
| HF Hub repo | [`pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42`](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42) (single; persistent volume makes redundancy unnecessary) |
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
- ~~**Dual-repo** — uploads to both primary and backup namespaces simultaneously~~ (Removed 2026-05-15: the Spheron persistent volume already preserves all artifacts across host preemption, so the backup repo was duplicating durability we get for free from the volume. Uploader now writes only to the primary repo; backup HF repo has been deleted.)
- **5-second filesystem scan** for new `step_N/` folders (vs 60s in a3's bash uploader)
- **JSON state file** with atomic writes (vs plain-text append in a3)

### 4.2 External monitor (`external_monitor.py`) — script exists but not currently auto-invoked
- Script designed to run on user's **local Mac**, not the training instance
- Polls HF every 5 min, cross-references with W&B step count
- Reports STALE if no commit in 10 min
- Reports LAG if W&B step is >10 steps ahead of HF rollouts
- **Status 2026-05-15**: script exists in repo but no `launchd` / `cron` job is invoking it; verified via `launchctl list`, `crontab -l`, and `ps -ef | grep external_monitor` on the user's Mac. The "external monitor running on user's local Mac" status in earlier sections was aspirational. With the persistent volume now preserving artifacts across host preemption, the external monitor's main purpose (catch silent-drop uploads before the host dies) is much less critical — but if we want it auto-running, the easiest install is a per-user `launchd` plist polling every 5 min.

### 4.3 Launch checklist (`LAUNCH_CHECKLIST_A4.md`)
Explicit gate-by-gate verification. Every box must be checked before training:
- HF_TOKEN + HF_REPO_ID set
- Repo created on HF
- Uploader pre-flight passes
- Volume mount verified persistent (destroy + recreate test)

### 4.4 Pre-launch gates passed (for this run)
| Gate | Status |
|---|---|
| HF token set | ✓ in `.env` |
| Primary repo created | ✓ (h200-a4-seed42) |
| Pre-flight uploads canary | ✓ repo has `.preflight_canary.txt` |
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
7. **Before ANY of the above**: verify uploader is healthy by checking the heartbeat line in `/workspace/state/uploader_prod.log` (most recent line should be within 60 s and show `errors=0`). The external_monitor.py script in §4.2 is not currently auto-invoked.

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
- H200 (sm_90): matches the if; calls FlashInfer GDN. On small batches (smoke shape, async continuous-batching micro-batches) the kernel finishes. On **prod-scale sync prefill (309 rollouts post-retrieval-injection with 3-6 K context each — 309 ≈ the 320-rollout step batch minus the ~11 prompts that already emitted `<answer>` in turn 0)** the kernel deadlocks inside CUDA. GPU stays at 100 % util; the Python decode loop never returns from `forward_cuda`. Note: 320 = `num_prompts_per_step (64) × num_generations_per_prompt (5)` — these are GRPO rollouts seen by vLLM, not unique prompts.

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

### Cadence 1: steps 1-13 (through 2026-05-16 ~00:16 UTC)

Step wall-clock + reward signal (rollouts pulled from `train_data_step*.jsonl`):

| Step | Wall (s) | M:S | rew mean | rew > 0 | notes |
|---:|---:|---:|---:|---:|---|
| 1 | 1101.82 | 18:22 | 0.028 | 8 % | First step on patched native Triton GDN. GPU peak 140 GiB. |
| 2 | 1123.09 | 18:43 | — | — | Cold rollout. |
| 3 | 1079.01 | 17:59 | 0.023 | 7 % | Cold rollout. |
| 4 | 1023.30 | 17:03 | — | — | First sign of step-time drop. |
| 5 | 986.47 | 16:26 | 0.073 | 10 % | **Reward starts moving** (3× step 1). |
| 6 | 782.38 | 13:02 | — | — | Big drop — model stopped using max_turns. |
| 7 | 821.99 | 13:42 | 0.074 | 15 % | rew > 0 doubled vs step 1. |
| 8 | 735.03 | 12:15 | — | — | |
| 9 | 646.39 | 10:46 | — | — | |
| **10** | **598.64** | **9:58** | 0.073 | 15 % | **First checkpoint** (6.4 GB) saved + uploaded to HF in 21 s. |
| 11 | 470.35 | 7:50 | — | — | |
| 12 | 363.78 | 6:04 | 0.078 | 15 % | Wall now 1/3 of step 1. |
| 13 | — (running at cadence cut) | — | **0.110** | **20 %** | **Best reward yet** — 20 % of rollouts solving MuSiQue. |

**Trends after one cadence**:
- Step wall: 1101 → 364 s (−67 %) over 12 steps. Multi-turn rollout depth dropped sharply.
- Reward mean: 0.028 → 0.110 (~4 ×).
- Frac rew > 0: 8 % → 20 %.
- HF Hub: `step_10/` (6.4 GB) live at [the primary repo](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42/tree/main/step_10).
- Cumulative wall: ~3.5 h since launch. Cost: ~$7. 0 OOMs, 0 uploader errors, 0 stalls (GDN patch holds).

**Untrained-model rollout cost dropping faster than predicted**: §9 cold-50 estimate had step time at ~18 min through step 50; we're at 6 min by step 12. Updating projection below.

#### Mechanical examples (cadence 1, script-selected from 3,200 trajectories)

**BEST** — step 1, sample 120, **reward 1.000**, 8 tool calls, 27.8 K chars (truncated)

> **Q**: *"Who introduced the first microprocessor for the manufacturer of the System Controller Hub?"*
> **Final answer**: `Federico Faggin` ✓
> **Last `<think>`**: "Based on all my searches, I have clear information... the Intel 4004 was introduced by Federico Faggin. He was the sole chip designer among the engineers on the MCS-4 project..."
>
> *Commentary*: Cold-start luck — model issued 8 search calls (near the `max_turns: 10` ceiling) and accidentally landed on the right person via a chain of "Intel SCH → Intel 4004 → MCS-4 → Federico Faggin". The thinking is verbose but converges. Reward 1.0 is exact-match on the F1 reward, not partial credit.

**WORST** — step 1, sample 0, **reward 0.000**, 3 tool calls, 8.9 K chars (truncated)

> **Q**: *"What tradition is the basis for the legal system in the place where Coloane Park is located?"*
> **Final answer**: `Kissing Traditions` ✗  (correct ≈ "Portuguese civil law")
> **Last `<think>`**: "...the UK has a tradition called 'Kissing traditions'... Wait, that doesn't match what I got. Let me re-examine."
>
> *Commentary*: Model never realised Coloane Park is in Macau (Portuguese-influenced civil law tradition). It picked up irrelevant search-result tokens ("Beating the bounds", "Colloquy") and confabulated "Kissing Traditions" — a phrase that doesn't even exist. Classic cold-start hallucination: low retrieval quality + low pre-training prior on Macau-Portugal connection.

**MEAN** — step 1, sample 285, **reward 0.065**, 5 tool calls, 19.8 K chars

> **Q**: *"In what ways did Kanye draw inspiration from U2, Led Zeppelin, and the performer of All Down the Line?"*  ("All Down the Line" → Rolling Stones)
> **Final answer**: long paragraph ending "Kanye West drew inspiration from U2, Led Zeppelin, and the Rolling Stones for their melodies..." ✓-ish
> **Last `<think>`**: "...Bob Dylan and Johnny Cash in addition to U2 and Led Zeppelin..."
>
> *Commentary*: Got the bridge entity right (Rolling Stones perform "All Down the Line") but pulled in unrelated content (Bob Dylan, Johnny Cash). Partial F1 credit because the gold answer keywords were present but diluted by extra material. Typical cold-cadence-1 behavior: model recognises the question shape but doesn't compress to a clean answer.

#### Claude hand-analyses (cadence 1)

1. **Tool-call ceiling pinning**. 92.7 % of cadence-1 rollouts hit 7.8 K+ chars (the `max_total_sequence_length: 8192` truncation point). Median tool calls = 7 (paper recipe caps at 10). The model has learned the `<tool_call>...</tool_call>` *format* perfectly (100 % rollouts emit `<answer>`), but it doesn't know *when to stop searching* — so it spams until the context budget kicks it out. Phase-1 finding #4 (paper's partial-credit floor masks tool-use signal) is exactly the friction this run was set up to bypass: pure F1 reward, no shaped components, so reward 0 means reward 0 and the policy *must* learn to compress.
2. **Format learning is already complete; the gap is grounding.** Every rollout has a `<tool_call>` with a `<function=search>` and a `<parameter=query>` — the Qwen3.5-native XML format is rock-solid out of the cold model. What's failing is **multi-hop entity resolution**: questions like "country where Bucer was citizen" or "language of Aavarana" require chaining 2-3 entities, and the model is fragile on the chain. Both WORST and MEAN show the model executing the first hop and then drifting on the second.

#### Hop-stratified BEST successes (cadence 1, rew ≥ 0.9, most efficient by tool count)

| Hops | Step | Tools | Answer | Question |
|---:|---:|---:|---|---|
| 1 | 9 | 1 | `Germany` | What country is Großbottwar located in? |
| 2 | 8 | 1 | `Chicago blues` | The musician who first recorded I'm Your Hoochie Cooche Man was part of what subgenre of the blues? |
| 3 | 6 | 1 | `House of Representatives` | In which body, that presides over any impeachment of the U.S. president, of the assembly in charge of new issues not mentioned in the Constitution, must revenue bills originate? |
| **4+** | **10** | **1** | **`Edward VIII`** | Who was the abdicating king of the country to which the king having the regnal name of the Duke of York came back? |

**4-hop+ successes total in cadence 1: 10 / 3,200 rollouts (0.3 %).** Note: the 4-hop "1 search" example is the **retrieval-does-the-work** pattern — a single well-aimed query returns Wikipedia chunks that connect every entity in the chain. This is asymptotically ideal but partly a function of retrieval quality, not just policy reasoning. The model's parametric-knowledge fallback (the system prompt allows answering without searching if the model is confident) also lets it short-circuit shallow chains.

### Cadence 2: steps 14-24 (through 2026-05-16 ~00:50 UTC)

| Step | Wall (s) | M:S | rew mean | rew > 0 | notes |
|---:|---:|---:|---:|---:|---|
| 14 | 289.27 | 4:49 | 0.120 | 21 % | |
| 15 | 287.95 | 4:48 | 0.126 | 28 % | |
| 16 | 254.40 | 4:14 | 0.140 | 23 % | First sub-5-min step. |
| 17 | 263.31 | 4:23 | 0.091 | 18 % | Brief regression. |
| 18 | 227.04 | 3:47 | 0.117 | 20 % | |
| 19 | 245.15 | 4:05 | 0.111 | 23 % | |
| **20** | **242.07** | **4:02** | 0.106 | 18 % | **Second checkpoint** (6.4 GB) uploaded to HF in 23 s. |
| **21** | 248.41 | 4:08 | **0.164** | **30 %** | **Peak yet** — 30 % of rollouts now solving MuSiQue. |
| 22 | 248.57 | 4:08 | 0.099 | 20 % | |
| 23 | 281.05 | 4:41 | 0.140 | 24 % | |
| 24 | 255.14 | 4:15 | 0.130 | 23 % | |

**Cadence 1 vs Cadence 2 (10-step windows)**:
| Window | rew_mean | rew > 0 | step wall | tool calls (med) |
|---|---:|---:|---:|---:|
| Steps 1-10 | 0.067 | 13 % | ~900 s | 6 (still hitting ceiling) |
| Steps 11-20 | **0.106** | **20 %** | ~280 s | 4 (clean) |
| Δ | +58 % rew, +54 % rew>0 | -69 % wall | model converged on 4-call pattern |

**Trends after cadence 2**:
- Step time **plateaued ~245 s (4 min)**. The cold→stable transition is over; we're in stable-regime now.
- Reward keeps climbing — peak at step 21 (0.164) is **5.9× step 1 (0.028)**.
- Frac rew > 0: best step (21) reached 30 %, 3× the cold-start 10 % rate.
- HF: `step_20/` live at [the primary repo](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42/tree/main/step_20). Both checkpoints durable.
- Cumulative cost since launch: ~$14. Still ~30× under budget.
- B200 a3 reference (steps 11-20 cadence): rew_mean ~0.115 — H200 at 0.106 is **within 8 % of B200**, same recipe + same hardware-class except for the GDN patch. The patch's compute overhead does not visibly hurt learning.

#### Mechanical examples (cadence 2)

**BEST** — step 11, sample 16, **reward 1.000**, 8 tool calls, 32.0 K chars (truncated)

> **Q**: *"In which municipality of the province where Sydney-Victoria is located can Dean be found?"*
> **Final answer**: `Halifax` ✓
> **Last `<think>`**: "Sydney-Victoria is in British Columbia... Wait, Sydney-Victoria is in *Nova Scotia* (Cape Breton)... Andrew Dean, Ephraim Dean represented Liverpool township in Nova Scotia House of Assembly..."
>
> *Commentary*: Model **self-corrected** mid-reasoning: started with the wrong province (British Columbia) but the search results forced a correction to Nova Scotia → Halifax. This is the post-cold-start regime: model trusts retrieval over its prior. Still 8 tool calls — the policy is exploring more than necessary, but the trade is paying off in reward.

**WORST** — step 11, sample 0, **reward 0.000**, 11 tool calls (above paper `max_turns: 10`!), 36.0 K chars

> **Q**: *"When did the first restaurant from which McDonaldization got its name open in the country where the author of Judgement of Martin Bucer Concerning Divorce was a citizen?"*
> **Final answer**: `Beijing` ✗  (correct ≈ "1948" or similar McDonald's US founding date; Bucer was German, but Milton wrote the *Judgement* and was English, so country = England or Germany depending on interpretation)
> **Last `<think>`**: "This is not giving the citizenship info I need. Let me try searching for 'Martin Bucer' again..."
>
> *Commentary*: Three-hop question collapsed. Model never resolved "author of *Judgement of Martin Bucer Concerning Divorce*" = John Milton (English) and ran out of context. Final answer `Beijing` is unhinged — it appears to be a free-text bailout pattern (note the system prompt's example uses Beijing). This is the **bailout-on-failure** failure mode: model has learned to emit *something* in `<answer>` even when it has no signal.

**MEAN** — step 11, sample 49, **reward 0.105**, 4 tool calls, 9.7 K chars

> **Q**: *"What is the highest point in the state where Hanna Theatre is located?"*
> **Final answer**: literal text "*tag.\n</think>\n\nBased on my search results, Hanna Theatre is located in Cleveland, Ohio. The highest point... is Campbell Hill...*" (the model emitted the `<answer>` opener mid-stream and the F1 scorer caught "Campbell Hill" inside)
> **Last `<think>`**: "...Campbell Hill (Ohio). Let me format this properly with the `<answer>` tag."
>
> *Commentary*: **Knew the answer (Campbell Hill) but botched the formatting** — emitted `<answer>` tag prematurely inside the `<think>` block, so the F1 scorer only caught fragments. F1 0.105 means partial keyword match. With a stricter EM scorer this would be 0. Format-vs-content split: the gradient should push the policy toward consistent `<answer>X</answer>` emission at the very end.

#### Claude hand-analyses (cadence 2)

1. **Tool-call median dropped from 7 to 4 in one cadence.** The biggest single behavioural shift in the run so far. Truncation dropped 92.7 % → 69 %. Mean rollout length 25 K → 11.5 K chars (−54 %). The model has internalised that *more search calls do not reliably mean better answers*. Reward mean climbed +62 % (0.066 → 0.107) on a +54 % shorter rollout — pure efficiency gain.
2. **A new failure mode appears: `<answer>` mid-stream.** MEAN above shows the model emitting `<answer>` *inside* a `<think>` block ("Let me format this properly with the `<answer>` tag" — and then doing so prematurely). This wasn't visible in cadence 1 where rollouts were truncated before reaching this state. As rollouts get shorter, the model gets closer to formatting the answer correctly, but it's not yet consistent. Expect the policy to push this toward 100 % at the END pattern in cadences 3-4.

#### Hop-stratified BEST successes (cadence 2)

| Hops | Step | Tools | Answer | Question |
|---:|---:|---:|---|---|
| 1 | 15 | 1 | `September 25, 2015` | when did the iphone 6s plus by the creator of Swift come out? |
| 2 | 18 | 1 | `March 15, 2018` | What was the release date of Freaky Friday, featuring the performer of Bassline? |
| 3 | 18 | 1 | `1410` | when was the astronomical clock in the city where most of the acedemies insitutions were located built? |
| 4+ | 19 | 1 | `India` | In what part of the country where the author of Odayil Ninnu is from is Delhi located? |

**4-hop+ successes total in cadence 2: 11 / 3,200 rollouts** (+10 % vs cadence 1). Most multi-hop successes still use 1 tool call — the "retrieval-does-the-work" pattern dominates.

### Cadence 3: steps 25-30 (through 2026-05-16 ~01:38 UTC)

| Step | Wall (s) | M:S | rew mean | rew > 0 | notes |
|---:|---:|---:|---:|---:|---|
| 25 | 284.86 | 4:45 | 0.161 | 24 % | |
| 26 | 271.16 | 4:31 | 0.122 | 24 % | |
| 27 | 359.82 | 6:00 | 0.091 | 20 % | Step time rising — re-exploration phase. |
| 28 | 304.35 | 5:04 | 0.161 | 32 % | |
| 29 | 353.21 | 5:53 | 0.134 | 26 % | |
| **30** | **340.01** | **5:40** | **0.173** | **37 %** | **Third checkpoint** (6.4 GB) uploaded to HF in 24 s. **Peak reward and peak rew>0% so far**. |

**Cadence 3 vs Cadence 2 (10-step windows)**:
| Window | rew_mean | rew > 0 | step wall | tool calls (med) |
|---|---:|---:|---:|---:|
| Steps 1-10 | 0.067 | 13 % | ~900 s | 6 |
| Steps 11-20 | 0.106 | 20 % | ~280 s | 4 |
| Steps 21-30 | **0.131** | **26 %** | ~315 s | 4 |
| Δ c2→c3 | +24 % rew, +30 % rew>0 | +12 % wall (re-exploration) | — |

**Trends after cadence 3**:
- Reward climbing again after a brief plateau (steps 17-22 around 0.10-0.16; steps 28-30 mean 0.156). Step 30 hit 0.173 — **6.3× cold-start**.
- Re-exploration phase has started: step time rose from ~245 s to ~315 s (+25 %). Tool count stays at 4 median, but search queries are getting longer (more careful retrieval). Same pattern B200 a3 showed at steps 25+.
- HF: `step_30/` live at [the primary repo](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42/tree/main/step_30).
- Cumulative cost: ~$18. Steps 1-30 wall: ~9 h.
- B200 a3 cadence-3 same range: rew_mean 0.143 — H200 at 0.131 is **within 9 % of B200** (still tracking).

#### Mechanical examples (cadence 3)

**BEST** — step 21, sample 0, **reward 1.000**, 3 tool calls, 6.6 K chars

> **Q**: *"Who won season 5 of Bigg Boss presented in the language of the novel Aavarana?"*  ("Aavarana" → Kannada novel by S.L. Bhyrappa)
> **Final answer**: `Chandan Shetty` ✓
> **Last `<think>`**: "...Bigg Boss Kannada (which the user asked about since they mentioned the language)... Chandan Shetty was declared the winner..."
>
> *Commentary*: This is the **asymptotic ideal**: 3 tool calls, 6.6 K chars, clean answer. Model correctly resolved "language of Aavarana" → Kannada, mapped to Bigg Boss Kannada Season 5, retrieved winner. The B200 a3 cadence-2 commentary described this exact pattern ("minimum tool calls, minimum tokens, correct answer") as the convergence target — we hit it cleanly here.

**WORST** — step 21, sample 1, **reward 0.000**, 3 tool calls, 6.2 K chars

> **Q**: *Same question* as BEST.
> **Final answer**: `Sanjay Dutt and Salman Khan` ✗
> **Last `<think>`**: "...Bigg Boss season 5 in Hindi. The show features Sanjay Dutt and Salman Khan as the hosts... There's no evidence in the search results that the show existed in Bangla/Avarana language."
>
> *Commentary*: **Bridge corruption**. Same prompt, same number of tools, but the model defaulted to Bigg Boss *Hindi* season 5 (the most famous one) instead of resolving "language of Aavarana" first. Also emitted the *hosts* rather than the *winner*. Two-error compound failure on a question that the BEST rollout (sibling in the same step!) got right. GRPO advantage estimator should see a large reward gap between these two and push the policy strongly toward the BEST-style chain.

**MEAN** — step 26, sample 180, **reward 0.136**, 4 tool calls, 11.1 K chars

> **Q**: *"Where in the colonial holding in Aruba's continent that was governed by the country where Prazeres is located are most of the German descendants found?"*
> **Final answer**: rambling paragraph about "ABC islands... Dutch colonial holdings... German descendants" — keyword-rich but doesn't name a clean place
> **Last `<think>`**: "...German descendants are primarily found in the Dutch colonial hol[dings]..."
>
> *Commentary*: 4-hop question (Prazeres → Portugal → South America → Suriname → German descendants in interior). Model collapsed to "Dutch ABC islands" — wrong continent entirely (Aruba's "continent" is S. America; Prazeres is in Portugal so it points to Brazil, where many Germans settled in Rio Grande do Sul). F1 partial-credit on keyword overlap. Multi-hop entity resolution still the major gap.

#### Claude hand-analyses (cadence 3)

1. **GRPO advantage at maximum signal**. The BEST/WORST sibling pair at step 21 (same question, reward 1.0 vs 0.0, both with 3 tool calls) gives the GRPO group-relative advantage estimator a clean direction: identical input, divergent reward → advantage maximises the policy gradient toward the BEST-style chain. This is exactly the dynamic the algorithm is designed to exploit, and we see it materialising in cadence 3's +24 % reward gain over cadence 2.
2. **Re-exploration emerging — model starts adding a 5th search.** Tool-call mean crept from 4.00 (cadence 2) to 4.29 (cadence 3). Median still 4 but the distribution thickens at 5 and 6. This is the **B200 a3 "re-exploration" regime** at the same step range, where the policy starts trading wall-clock for accuracy on harder questions. Next cadence will confirm whether tool count + length + reward all continue rising together (worth the trade) or whether reward plateaus while tools grow (not worth it).

#### Hop-stratified BEST successes (cadence 3)

| Hops | Step | Tools | Answer | Question |
|---:|---:|---:|---|---|
| 1 | 28 | 1 | `September 19, 2014` | When did the iPhone 6 by the developer of Rhapsody come out? |
| 2 | 24 | 1 | `ABC` | Who is the original broadcaster of PGA Tour and where I Live? |
| 3 | 21 | 1 | `Chandan Shetty` | Who won season 5 of Bigg Boss presented in the language of the novel Aavarana? |
| 4+ | 29 | 1 | `Mexico City` | What is the name of the capital of Mexico in the language of Rafael Alberti? |

**4-hop+ successes total in cadence 3: 15 / 3,200 rollouts** (+50 % vs cadence 1). The capital-of-Mexico/Spanish answer at step 29 is a clever 1-search resolution — the model knew the Mexican capital is "Mexico City" in English (the answer in Spanish would be "Ciudad de México"; but the F1 scorer accepts the English form on this question's gold).

### Cadence 4: steps 31-40 (through 2026-05-16 ~02:42 UTC)

| Step | Wall (s) | M:S | rew mean | rew > 0 | tool_med | notes |
|---:|---:|---:|---:|---:|---:|---|
| 31 | 362.79 | 6:03 | 0.148 | 34 % | 4 | |
| 32 | 358.71 | 5:59 | 0.173 | 30 % | 4 | |
| 33 | 384.18 | 6:24 | 0.110 | 26 % | **5** | First step with 5-call median — model adds a search. |
| 34 | 341.26 | 5:41 | 0.150 | 25 % | 4 | |
| 35 | 366.21 | 6:06 | 0.208 | 38 % | 5 | **Crosses A100 ceiling (0.1997 step 37)** at our step 35. |
| 36 | 387.28 | 6:27 | 0.174 | 33 % | 5 | |
| 37 | 383.16 | 6:23 | **0.209** | **40 %** | 5 | **Peak rew_mean & peak rew > 0 so far**. |
| 38 | 373.23 | 6:13 | 0.176 | 28 % | 5 | |
| 39 | 363.06 | 6:03 | 0.192 | 33 % | 5 | |
| **40** | **441.32** | **7:21** | 0.181 | 32 % | 5 | **Fourth checkpoint** (6.4 GB) uploaded to HF in 21 s. Step-40 was a wall-outlier (+17 % vs window mean) — not yet clear if drift or batch noise. |

**Cadence 4 vs 3 (10-step window means)**:
| Window | rew_mean | rew > 0 | step wall | tool_med | len_med |
|---|---:|---:|---:|---:|---:|
| 21-30 | 0.131 | 26 % | 315 s | 4 | 10.8 K |
| **31-40** | **0.171** | **31 %** | 370 s | **5** | 13.0 K |
| Δ | **+30 %** | +19 % | +17 % wall | +1 tool call | +20 % context |

**Trends after cadence 4**:
- **Reward broke through the A100 ceiling** of 0.1997 (A100 49-step max at step 37). H200 at step 37 = 0.209 — same step number, +5 % reward. Entering territory the A100 reference cannot speak to.
- **Re-exploration regime is paying off**: model added one tool call (median 4 → 5), bought +17 % wall, returned +30 % reward. Worth the trade.
- **Step time rose to ~370 s (6 min)**, with step 40 at 441 s (outlier). If step 41-50 hold at 370 s, ETA step 311 = 273 × 370 s + ~9 h elapsed ≈ 37 h total wall = step 311 lands ~08:00 UTC May 17.
- HF: `step_40/` live at [the primary repo](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42/tree/main/step_40).
- Cumulative cost: ~$22.
- B200 a3 cadence-4 published rew_mean ≈ 0.160 — H200 at 0.171 is **+7 % above B200**, first window where we lead. Likely batch noise but worth tracking.

#### Mechanical examples (cadence 4)

**BEST** — step 31, sample 28, **reward 1.000**, 4 tool calls, 10.1 K chars

> **Q**: *"The home country of Así's singer has Gila monsters. What state do the monsters live in?"*  (Así → Mexican song by Pablo Alborán covered widely; Gila monsters live in Sonora)
> **Final answer**: `Sonora` ✓
> **Last `<think>`**: "...the Gila monster is native to the Mexican state of Sonora. From the context of the song mentioned (likely a Western theme song or similar), assuming the question is asking about the home country's state, the answer would be Sonora."
>
> *Commentary*: Clean 4-call chain on a 3-hop question (song → singer's home country → state with Gila monsters). Notably the model is **slightly uncertain** about the song's identity but executes the question's *structure* correctly anyway: "assuming the question is asking about the home country's state" — that's robust hop-handling, not lucky guessing. This is exactly the policy behaviour GRPO is supposed to instil: trust the question's chain even under uncertainty on intermediate entities.

**WORST** — step 31, sample 0, **reward 0.000**, 3 tool calls, 7.1 K chars (almost truncated)

> **Q**: *"Why did the country beaten by the US in the Miracle on Ice complain about the action of the U.S. to the organization that legislated United Nations Security Council Resolution 731?"*  (Miracle on Ice → USSR/Russia; UN SC R-731 → about Pan Am Lockerbie, legislated by UN Security Council; so country = Russia/USSR complaining to UN SC)
> **Final answer**: `China opposed the U.S. Resolution 731 claim` ✗  (wrong country and wrong substance)
> **Last `<think>`**: "...In 1993, a false claim was made by the United States in 1993 about the 'Yinhe incident'... Looking at this, the question is asking about a country (presumably China, given Resolution 731's context)..."
>
> *Commentary*: **Bridge corruption again** — model fixated on the Yinhe incident (China-USA, 1993) instead of resolving "Miracle on Ice country" = USSR. The 1980 Olympics ice-hockey upset is well-known enough that the cold model should have it; this looks like a search-result-anchored failure where the retrieval pulled Yinhe and the model committed. Same class of failure as the Bigg Boss WORST in cadence 3.

**MEAN** — step 35, sample 205, **reward 0.169**, 6 tool calls, 19.8 K chars

> **Q**: *"What effort was the Canadian Military known for in the country which means Ashkenaz in Hebrew?"*  (Ashkenaz → Germany)
> **Final answer**: "*The Canadian Military was most notably known for their efforts in Operation Veritable, which involved an invasion of Sicily and Italy to free France...*"
> **Last `<think>`**: "The country meaning Ashkenaz in Hebrew is Germany. The Canadian Military had a significant role in World War II..."
>
> *Commentary*: Got the Ashkenaz → Germany bridge correct. Got "WWII Canadian military in Germany" partially correct (Operation Veritable was Rhineland 1945, not Sicily/Italy as the model wrote). F1 0.169 means keyword overlap on "Operation Veritable" + "Canadian Military" but mixed with wrong facts ("Sicily and Italy"). **Confabulation under partial knowledge**: model has the right entity but fabricates supporting detail.

#### Claude hand-analyses (cadence 4)

1. **The 5th tool call is paying off — but at a context cost.** Cadence 4 added one tool call to the median (4 → 5), bought +17 % wall, returned +30 % reward (window mean 0.131 → 0.171). Truncation rate jumped 79 % → 94 % (rollouts hitting 7.8 K+ chars). The model is using the extra tokens for **cross-verification searches** — issuing a second search to confirm an entity it already has, rather than a brand-new query. Reward gain says the verification is worth it; truncation rate says we're back at the context ceiling that we escaped in cadence 2.
2. **Crossed A100's recorded ceiling at the same step number (step 37).** A100's published max was rew_mean 0.1997 at step 37 across the entire 49-step run. H200 step 37 = 0.209, +5 %. With smoother gradient from mb=2 (vs A100's mb=1) and same recipe otherwise, we are now in territory the A100 reference cannot speak to. The next cadence (steps 41-50) will tell us whether the policy keeps climbing or plateaus — if it plateaus near 0.20-0.22 we're in the regime where the paper expected step-100+ improvements to be smaller (+ 0.02 / 50 steps), and the early-stop decision becomes load-bearing.

#### Hop-stratified BEST successes (cadence 4)

| Hops | Step | Tools | Answer | Question |
|---:|---:|---:|---|---|
| 1 | 34 | 1 | `September 19, 2014` | When did the iphone 6 of the company that produces the iPod come out? |
| 2 | 35 | 1 | `September 25, 2015` | What was the release date of the iphone 6, designed by the developer of Logic Studio? |
| 3 | 35 | 1 | `Norway` | What is the country of the singer of the song Oah? |
| 4+ | 34 | 2 | `Aleksandar Vučić` | As of 2017, who was in charge of the country where the village of Sjeverin is found? |

**4-hop+ successes total in cadence 4: 20 / 3,200 rollouts** (+33 % vs cadence 3, +100 % vs cadence 1). The 4-hop "Aleksandar Vučić" case at step 34 used **2 searches** — first to resolve "Sjeverin → Serbia", then "Serbia → 2017 leader". The model is starting to use sequential search-grounding for harder chains rather than relying purely on lucky single-shot retrieval.

#### Why was the prior A100 (mb=1) run reportedly solving 4-hop chains by step 15?

[`RESULTS_m5.md` §4.2.3](RESULTS_m5.md) records the A100 prod-a2 run (killed at step 15, mb=1) solving a 4-hop *Fantasy Land Tour 2004 / Tony Daykin* question at step 15 with reward 1.0 in 3 turns. Two hypotheses:

1. **mb=1 noisier gradient explores faster.** mb=2 produces a +20 % higher reward mean (B200 a3 measurement) due to smoother gradient, but mb=1's noisier updates may push the policy through the multi-hop chain space faster, even if it climbs the *scalar* reward number slower. **Plausible.**
2. **Data ordering / lucky single example.** A2's full rollout corpus was deleted before smoke verifies (`logs/exp_010/` and `exp_011/` wiped pre-ckpt-verify), so we cannot check the *distribution* of multi-hop successes at step 15. **One success ≠ a rate.** Our H200 run had 11 four-hop+ successes at cadence 2 (steps 11-20); without a2's full corpus we cannot tell if a2 had 1 or 50 at step 15.

We cannot run the controlled comparison. The honest read: **our H200 mb=2 run shows multi-hop chain reasoning emerging by step 6-10** (cadence 1), and the count grows monotonically thereafter. The "by step 15" prior claim looks consistent rather than ahead, once we look at counts not single anecdotes.

#### Planned-multi-hop reasoning (3-5 tool calls with explicit decomposition)

The 1-tool-call multi-hop successes in the hop-stratified table above are mostly the **retrieval-does-the-work** pattern (a single well-aimed search returns chunks that connect the chain). By cadence 4 a second pattern emerges: **explicit-plan multi-tool reasoning**, matching the A100 a2 step-15 *Fantasy Land Tour 2004* example in [`RESULTS_m5.md` §4.2.3](RESULTS_m5.md). 74 such rollouts in steps 30-44.

Worked example — **step 39, sample 158**, reward 1.0, 3 tool calls:

**Q**: *"How many floors are in the Trump Tower in the city where the performer of Come 2 My House was born?"*  →  `<answer> 98 </answer>` ✓

**First `<think>` (planning step, numbered explicitly):**
> *"I need to first find out where 'Come 2 My House' was performed/born from, then find where Trump Tower is located, and finally determine how many floors Trump Tower has in that city. Let me break this down:*
> *1. First search to find information about 'Come 2 My House' performer and where they were born*
> *2. Then find the location of Trump Tower in that city*
> *3. Finally, determine the number of floors in Trump Tower"*

**Execution:**

| Step | Search query | Returned | Model's reaction |
|---:|---|---|---|
| 1 | `Come 2 My House performer birthplace information` | "Come 2 My House is the ninth studio album by American R&B/funk singer **Chaka Khan**" (album info, no birthplace) | "Found Chaka Khan but no birthplace — I need to search more specifically" |
| 2 | `Chaka Khan city of origin or birth place` | "Chaka Khan was born... in **Chicago, Illinois**" | "Locked Chicago — now I need Trump Tower there" |
| 3 | `Trump Tower location Chicago Illinois` | "Trump International Hotel and Tower (Chicago)... **98-story structure**" + "Trump Tower [NY]... 58-story" | **Disambiguated**: "Since [Chaka Khan] was born in Chicago, I'll use the 98-story figure from the Chicago location" |

**Why this matters**: this is genuinely the chain-of-search behavior the GRPO recipe is designed to teach — numbered plan → targeted query per hop → entity substitution between hops → ambiguity resolution using chain context → final answer. **The model did the work, then asked retrieval the right questions.** Distinct from the 1-tool-call shallow-retrieval pattern. Both are present in policy by cadence 4; the planned mode handles harder chains where entities aren't obviously co-mentioned (the Trump-Tower-NY vs Chicago disambiguation step is the key tell).

**Significance for the convergence story**: GRPO's group-relative advantage estimator sees a strong signal when (planned, reward=1.0) sibling rollouts coexist with (single-search-failed, reward=0) on the same question, which is happening with increasing frequency. The next several cadences should show this pattern stabilising as the dominant mode for 3-4 hop questions.

### Cadence 5: steps 41-50 (through 2026-05-16 ~12:00 UTC, first cadence on host 126 post-resume)

| Step | Wall (s) | M:S | rew mean | rew > 0 | tool_med | notes |
|---:|---:|---:|---:|---:|---:|---|
| 41 | 452.32 | 7:32 | 0.166 | 26 % | 3 | First step post-resume on 126; engine re-warm. |
| 42 | 379.20 | 6:19 | 0.222 | 39 % | 3 | |
| 43 | 483.70 | 8:04 | 0.179 | 30 % | 3 | |
| 44 | 459.79 | 7:40 | 0.191 | 35 % | 3 | |
| 45 | 411.92 | 6:52 | 0.226 | 39 % | 3 | |
| 46 | 439.82 | 7:20 | 0.222 | 32 % | 3 | |
| 47 | 455.83 | 7:36 | 0.208 | 32 % | 3 | |
| 48 | 427.26 | 7:07 | **0.257** | 35 % | 3 | **Peak rew_mean of run so far.** |
| 49 | 452.99 | 7:33 | 0.201 | 35 % | 3 | |
| **50** | **518.39** | **8:38** | 0.146 | 24 % | 3 | **Fifth checkpoint** (6.4 GB) uploaded to HF. Slowest step of cadence and lowest reward; co-occurs with a noticeable step-50 dip. |

**Cadence 5 vs 4 (10-step window means)**:
| Window | rew_mean | rew > 0 | step wall | tool_med | len_med |
|---|---:|---:|---:|---:|---:|
| 21-30 | 0.131 | 26 % | 315 s | 4 | 10.8 K |
| 31-40 | 0.171 | 31 % | 376 s | 5 | 13.0 K |
| **41-50** | **0.202** | **33 %** | **448 s** | **3** | 14.6 K |
| Δ vs 4 | **+18 %** | +2 pp | **+19 % wall** | **−2 calls** | +12 % context |

**Trends after cadence 5**:
- **Reward keeps climbing**: window mean 0.171 → 0.202 (+18 %). Peak per-step at step 48 = 0.257, the highest single-step rew_mean on this run. Step 50 dipped to 0.146; likely batch noise on a hard subset, not regression.
- **Tool-call median dropped 5 → 3**. Cadence 4 added a tool call ("cross-verification"); cadence 5 took two off. The model is becoming **more efficient**: fewer searches per question, higher reward. This is the canonical convergence-toward-shorter-trajectories pattern of a well-shaped RL run; the "5th call paying off" of cadence 4 was a transient over-search regime that policy gradient is now collapsing back out.
- **Step wall +19 % vs cadence 4** (448 s vs 376 s) despite fewer tool calls — and stays elevated across all 10 steps, not just the first 2-3. AdamW re-warm should have decayed by step 47; it didn't. Most likely host-to-host variance on Spheron Spot (cadence 4 was on host 247, cadence 5 is on 126). Re-exploration drift cannot be ruled out, but the flat tool_med-3 argues against it.
- **Context still growing**: len_med 13.0 K → 14.6 K. With fewer tool calls but longer rollouts, individual `<think>` blocks and tool responses are getting wordier.
- HF: `step_50/` live at [the primary repo](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42/tree/main/step_50).
- Cumulative cost (at $1.95/h, ~11.5 h elapsed): ~$22.
- **Wall-clock projection**: at 448 s/step, ETA step 311 = 261 × 448 s + 11.5 h elapsed ≈ 44 h total = step 311 lands ~14:30 UTC May 17 (was ~37 h with cadence-4 timing).

#### Mechanical examples (cadence 5)

**BEST** — step 41, sample 0, **reward 1.000**, 4 tool calls, 9.6 K chars

> **Q**: *"Who is the chief justice of the country containing the Bibiani/Anhwiaso/Bekwai District?"*  (Bibiani/Anhwiaso/Bekwai → Ghana)
> **Final answer**: `Sophia Akuffo` ✓
> **Last `<think>`**: "Now I have confirmed that the chief justice is Sophia Akuffo, who was the Chief Justice of Ghana and appointed in 2017... I'll use that name."
>
> *Commentary*: Clean 4-call chain on a 3-hop question (district → country → person → year). Model resolves Ghana from the district name, then runs successive searches to confirm both the role and the person before committing. Verification-by-search is the right behaviour here; the question is exactly the kind of low-prior fact (small Ghanaian district name) where the cold model would not trust its prior.

**WORST** — step 41, sample 5, **reward 0.000**, 9 tool calls, 29.7 K chars (deeply truncated)

> **Q**: *"When did the first large winter carnival take place in the city where Jean Pelletier died?"*  (Jean Pelletier → Quebec City)
> **Final answer**: `1955` ✗  (gold answer not 1955 per scorer)
> **Last `<think>`**: "Based on all my research... the first large winter carnival in Quebec City took place in 1955..."
>
> *Commentary*: 9 tool calls is **3× the median** and shows the model thrashing. Quebec City's *modern* Carnaval de Québec started in 1955, but the question's gold answer is from MuSiQue's bridge resolution and depends on the dataset's exact source page. Two failures stacked: (1) over-search loop (9 calls, 30 K chars, near-truncation), (2) anchor on the wrong "first" date. The "thrashing → wrong answer" mode is the inverse of cadence 4's 5-call cross-verification mode; same model, different distribution tail.

**MEAN** — step 41, sample 233, **reward 0.182**, 5 tool calls, 15.2 K chars

> **Q**: *"What early Huguenot Church was established in the country where IK Tellus's sport comes from?"*  (IK Tellus → Sweden association football club; the gold answer expected refers to the French Reformed Church established in Stockholm after the Edict of Nantes)
> **Final answer**: *"Sweden's Huguenot Church was established after the Edict of Nantes in 1598..."*
> **Last `<think>`**: "...I need to find the Huguenot Church in Sweden... The information I have about Huguenots refers to French Huguenots, so I need to find the Huguenot Church in Sweden."
>
> *Commentary*: Correct bridge resolution (IK Tellus → Sweden) and correct era anchor (Edict of Nantes 1598). F1 0.18 = partial keyword overlap. The model is doing the right *kind* of chain but stops short of returning the church's specific name. Classic mid-band rollout: factually grounded, half the answer present.

#### Claude hand-analyses (cadence 5)

1. **Tool calls dropping 5 → 3 while reward rises +18 % is the headline finding of this cadence.** GRPO is now actively pruning over-search behaviour. The cadence-4 story was "5th call buys +30 % reward"; cadence-5 inverts it — "3 well-aimed calls beat 5 cross-verifying ones". This is consistent with cadence 4's truncation rate of 94 % being a binding constraint on the policy: the model could not buy more tokens to verify with, so it learned to spend tokens better. Worth watching: if cadence 6 holds at tool_med=3 with rew_mean ≥ 0.20, the policy has converged on a stable trajectory shape. If it slides back to 4-5 with no reward gain, we are in noise.
2. **Step time +19 % over cadence 4 is the worry — and host variance is the most likely explanation, not AdamW.** Cadence 4 averaged 376 s; cadence 5 averaged 448 s, with the elevation distributed across all 10 steps (not concentrated at 41-43 as AdamW re-warm would predict). The same recipe ran 370 s on host 247; on host 126 it runs 448 s. Spheron Spot has known per-host variance (different memory generations, NIC, neighbour load). A more troubling alternative is *exploration drift*: len_med crept 13.0 K → 14.6 K even with fewer tool calls, so per-rollout decode tokens went up. We cannot disentangle host vs drift from one cadence; cadence 6 (same host 126) should clarify — if step time drops back to 400 s as exploration normalises, drift was a factor; if it stays 440-450 s, it's just host 126.

#### Hop-stratified BEST successes (cadence 5)

| Hops | Step | Tools | Answer | Question |
|---:|---:|---:|---|---|
| 1 | 47 | 1 | `September 25, 2015` | When did the iPhone 6s Plus come out from the iPod manufacturer? |
| 2 | 45 | 2 | `1946` | When did the country encompassing petra the lost city of stone gain independence? |
| 3 | 45 | 2 | `August 1, 1876` | When did the state where Tarryall Creek is located become part of the US? |
| 4+ | 48 | 2 | `1963` | In what year was the unification of the country where the village of Bario can be found? |

**4-hop+ successes in cadence 5: 26 / 3,200 rollouts** (+30 % vs cadence 4's 20). The step 48 Bario / Malaysia unification 1963 case resolves a 4-hop chain (village → district → country → year of unification) with only **2 well-aimed searches**. The trend is **fewer tools, harder chains** — same pattern as the window aggregates.

#### Planned-multi-hop reasoning (cadence 5)

Worked example — **step 46, sample 126**, reward 1.0, 4 tool calls, 5-hop:

**Q**: *"In which country is Tuolumne, a city in the county sharing a border with Camp Connell's county in the state where Some Like It Hot was filmed?"*  →  `<answer> United States </answer>` ✓

**First `<think>` (planning step, explicit work-backwards decomposition):**
> *"The user is asking for the country of residence of Tuolumne, a city, given its county sharing a border with Camp Connell's county, and that county is in the state where Some Like It Hot was filmed. I need to work backwards this chain.*
> *First, I need to find out what Some Like It Hot was filmed in, and then find the county sharing a border with Camp Connell's county in that state, which should lead me to the district/city of Tuolumne.*
> *Let me break this down:*
> *1. First, find the location of 'Some Like It Hot' (to find where it was filmed)*
> *2. Then find that state and the county sharing a border..."*

**Execution**: 4 calls resolved Some Like It Hot → California, then Camp Connell → California, then Tuolumne County → California, then concluded Tuolumne → United States. The chain never fully resolved which California county shares a border with Camp Connell's (it didn't need to — the country was forced once California was confirmed); the model recognised the shortcut and answered correctly.

**Distinct from the cadence-4 Chaka Khan / Trump Tower example**: that one needed full chain resolution (Chicago vs NY disambiguation). This one shows the model **short-circuiting when intermediate hops aren't load-bearing for the final answer**. Both are present in policy by cadence 5; the policy now has two distinct decomposition strategies and picks between them.

**3-5 tool call planned rollouts with plan_score ≥ 13 in steps 41-50: dozens.** The two highest plan_scores in this cadence (15 each) are step 49 sample 116 (Hyderabad / first expedition to Asia) and step 49 sample 214 (Alleycat's Pizza / Fantasy Land Tour 2004), both reward 1.0 at 3 tool calls. The Alleycat's Pizza example matches the A100 prod-a2 step-15 *Fantasy Land Tour 2004* example in [`RESULTS_m5.md` §4.2.3](RESULTS_m5.md) on both the question shape and the explicit numbered plan — the planned-multi-hop behaviour is reproducing the prior run's signature at the same milestone.

## 9. Cost / wall-clock estimate

**Confirmed rate: $1.95/h** (Spheron ES Spot, 1× H200 SXM5, US Central 1, instance ID `6a072a4e`). Validated 2026-05-15 ~22:30 UTC against dashboard: $12.25 total at 6.28 h elapsed = $1.951/h. (The $15.15/h figure in `HARDWARE_COMPARISON.md` is the 8× cluster tier; not what we're on.)

H200 is roughly:
- 0.6× B200's memory bandwidth (4.8 vs 8 TB/s)
- 0.44× B200's BF16 dense compute (989 vs 2250 TFLOPS)
- 1.76× A100's memory bandwidth (4.8 vs 2 TB/s)

**Measured anchors (v11, untrained-model phase, max_turns=10)**:
| Step | Wall (s) |
|---:|---:|
| 1 | 1101.82 |
| 2 | 1123.09 |
| 3 | 1079.01 |
| mean | **1101.30 s = 18.36 min** |

These are with the H200 FlashInfer GDN patch applied (native Triton path; see §7.5). Expected to decrease as the policy learns to emit `<answer>` earlier (turn 0 of step 1 ran ~80 % of prompts deep into multi-turn; this drops with training).

**MuSiQue dataset size**: 19,938 training rows (`data/training/musique/train.parquet`).
**Batch shape**: `num_prompts_per_step: 64 × num_generations_per_prompt: 5 = 320` rollouts/step.
**Steps / epoch**: 19,938 / 64 = **311.5**.

**Projected at current rate** (1101 s/step, $1.95/h):
| Span | Steps | Wall | Cost |
|---|---:|---:|---:|
| 1 epoch | 311.5 | 95 h 17 min | **$186** |
| Configured `max_num_steps: 622` (= 2 epochs) | 622 | **190 h 14 min ≈ 7.9 days** | **$371** |
| Paper's 3 epochs | 934 | 285 h 32 min ≈ 11.9 days | $557 |

These projections assume step time stays at ~18 min throughout. **Realistically it won't.** Step time is dominated by multi-turn rollout cost (model emits `<search>` up to 10 times per question with current untrained policy). As the policy learns to emit `<answer>` earlier, average turns/question drops and step wall drops with it. B200 a3 observed step times dropping from ~9 min cold to ~6 min stable (~30 % reduction over 30-50 steps). Applying the same shape to H200:

| Phase | Steps | Step wall | Sub-total | Cumulative |
|---|---:|---:|---:|---:|
| ~~Cold (multi-turn at floor)~~ | 1-50 | 18 min | 15 h | 15 h |
| ~~Transitional (model learns to answer)~~ | 51-150 | 15 min | 25 h | 40 h |
| ~~Stable~~ | 151-622 | 13 min | 102 h | **142 h ≈ 5.9 days** |

The cold-transitional-stable shape above was the **a3 b200 trajectory** ported to H200; turns out the H200 a4 run blew through the curve much faster:

| Observed (cadence 1) | Steps | Step wall | Sub-total | Cumulative |
|---|---:|---:|---:|---:|
| Cold | 1-5 | ~17 min | 1.4 h | 1.4 h |
| Sharp drop | 6-12 | ~10 min | 1.2 h | 2.6 h |
| Stable (extrapolated, step 13 anchor 6 min) | 13-622 | 6 min | 61.0 h | **63.6 h ≈ 2.6 days** |

If step time really stabilises at ~6 min through step 622 (a strong assumption — it could drift up as the policy explores harder questions or down further as `<answer>`-emission improves):

| Span | Steps | Wall | Cost @ $1.95/h |
|---|---:|---:|---:|
| 1 epoch | 311.5 | 31.2 h | **$60.8** |
| `max_num_steps: 622` (2 epochs) | 622 | **63.6 h ≈ 2.6 days** | **$124** |
| Paper's 3 epochs | 934 | 95.7 h ≈ 4.0 days | $187 |

So realistic estimate for `max_num_steps: 622` revised down to **~2.6 days wall, ~$124 @ $1.95/h** (was 5.9 days / $277 before observing the drop). Whole budget of $1000 now easily fits 8 full seeds or 3+ epochs × 2 seeds. Will keep tracking step time at every cadence; if step time creeps back up above 10 min/step the projection will rise accordingly.

## 10. The four prior losses (recap)

| # | When | Cost | Cause | Mitigation in a4 |
|---|---|---|---|---|
| a1 step-50 ckpt crash | 2026-05-11 | ~$30 | `metric_name: "train/loss/mean"` violated NeMo-RL assertion | `metric_name: null` + smoke-verified |
| a1 rollout-corpus deletion | 2026-05-11 | 196 MB data loss | `rm -rf logs/exp_010` (prod) bundled with cleanup | Rollouts uploaded to HF + Spheron persistent volume preserves a second copy — single-machine deletion can't kill them |
| a2 zombie GPU misdiagnosis kill | 2026-05-12 | ~$21 | Misread `[Not Found]` in nvidia-smi | Kill-on-evidence rule |
| **a3 Spheron T3 spot preemption** | **2026-05-15** | **~$36 + step_50 ckpt** | **Spot host preempted ~step 56 + uploader silently dropped rollouts past step 18** | **Persistent volume (`miletone5`) preserves all artifacts across host preemption; bulletproof uploader with heartbeat + retry catches silent-drop pattern from a3** |

**Cumulative loss across M5.1**: ~$108 + 196 MB + 1 trained checkpoint.

## 11. How to use this repo (HF-side quickstart)

This HF repo is a live training mirror; each `step_N/` folder is a complete safetensors checkpoint. Latest is the highest step number visible in the file tree.

```python
# Load a checkpoint for inference
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42"
STEP = "step_40"  # or any step_N/ folder in this repo

# Tokenizer is shared across all checkpoints; the base Qwen3.5-0.8B tokenizer works
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
model = AutoModelForCausalLM.from_pretrained(
    f"{REPO}/{STEP}/policy/weights",  # NeMo-RL consolidates to safetensors here
    torch_dtype="bfloat16",
    device_map="auto",
)
```

**For agentic evaluation** (replicating the MuSiQue training loop): use the prompt template + tool-call format described in §2 below. The model expects:
- System prompt with the `search` tool registered (Qwen3.5 XML format: `<tool_call><function=search><parameter=query>...</parameter></function></tool_call>`)
- User question wrapped: `Use the search tool to look up facts as needed. When you have the answer, write it inside <answer> and </answer>.`
- A retriever HTTP endpoint that accepts `{query: ...}` and returns top-K Wikipedia chunks (we use Wiki-18 + E5-base-v2; see `local_retriever/README.md` in the source repo).

**Training recipe — what produced these checkpoints**:

| Field | Value |
|---|---|
| Base model | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) (hybrid GatedDeltaNet + attention, 248K vocab) |
| Dataset | [MuSiQue](https://github.com/StonyBrookNLP/musique) train split, 19,938 multi-hop QA rows |
| Algorithm | GRPO (Group Relative Policy Optimization) per ReSearch paper recipe |
| Reward | **F1 score on `<answer>...</answer>` content only** — no format reward (paper's partial-credit floor dropped per Phase-1 finding #4) |
| Hyperparams | lr=1e-6, kl=0.001, ratio_clip=0.2, max_turns=10 |
| Batch | `num_prompts_per_step: 64 × num_generations_per_prompt: 5 = 320` rollouts/step |
| Sequence | `max_total_sequence_length: 8192`, `max_new_tokens: 1024` per turn |
| Memory | `train_micro_batch_size: 2`, `activation_checkpointing: true`, `gpu_memory_utilization: 0.5` |
| Engine | vLLM 0.17 + FlashInfer 0.6.4 — **with a load-bearing patch** (next paragraph) |
| Save freq | every 10 steps |

**Required vLLM patch for Hopper (sm_90, including H200/H100)**: vLLM 0.17 hard-codes a FlashInfer-GDN prefill kernel at `qwen3_next.py:156` that deadlocks at prod batch sizes on Hopper. **Force the native Triton path** before training:

```bash
sed -i "s/if current_platform.is_cuda() and current_platform.is_device_capability(90):/if False:  # PATCHED/" \
    $(python -c "import vllm.model_executor.models.qwen3_next as m; print(m.__file__)")
```

This is mandatory on H200; idempotent on A100/B200 (their compute capability never matched the branch). Full diagnosis in [`docs/spheron/SETUP_SPHERON.md` §9.1 + P12](https://github.com/GaurisankarJ/reason_over_search/blob/experiment_1_h200/docs/spheron/SETUP_SPHERON.md).

**Hardware that produced these checkpoints**: 1× NVIDIA H200 SXM5 (141 GB VRAM, sm_90) on Spheron Spot tier ($1.95/h), Ubuntu 24.04, CUDA 13.0, virtiofs persistent volume (`miletone5`) so the run survives host preemption. The cost-per-step at current pace is ~$0.13. Full setup runbook: [`docs/spheron/SETUP_SPHERON.md`](https://github.com/GaurisankarJ/reason_over_search/blob/experiment_1_h200/docs/spheron/SETUP_SPHERON.md).

**Result analysis**: see §§7.5, 8 (live trajectory + per-cadence BEST/WORST/MEAN + hop-stratified + planned-multi-hop), 9 (cost), 10 (prior-loss recap).

## 12. Pointers

- W&B project: [`reason_over_search_h200`](https://wandb.ai/gaurisankarj1996-leiden-university/reason_over_search_h200)
- HF Hub: [`pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42`](https://huggingface.co/pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42) (backup repo removed 2026-05-15; volume handles redundancy)
- Prior run (crashed): [a3 RESULTS](RESULTS_M5_1_B200.md) (W&B run [h68uskz6](https://wandb.ai/gaurisankarj1996-leiden-university/reason_over_search_b200/runs/h68uskz6))
- Config: [`training_m5_1/configs/m5_1_research_paper.yaml`](../../training_m5_1/configs/m5_1_research_paper.yaml)
- Smoke config: [`training_m5_1/configs/m5_smoke.yaml`](../../training_m5_1/configs/m5_smoke.yaml)
- Uploader: [`training_m5_1/scripts/upload_a4_to_hf.py`](../../training_m5_1/scripts/upload_a4_to_hf.py)
- External monitor: [`training_m5_1/scripts/external_monitor.py`](../../training_m5_1/scripts/external_monitor.py)
- Launch checklist: [`training_m5_1/scripts/LAUNCH_CHECKLIST_A4.md`](../../training_m5_1/scripts/LAUNCH_CHECKLIST_A4.md)
- Hardware comparison: [`../setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md)
