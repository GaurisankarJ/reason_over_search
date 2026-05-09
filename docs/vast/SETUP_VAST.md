---
title: SETUP VAST
tags: [setup, vast, runbook]
source: internal
created: 2026-05-09
updated: 2026-05-09
---

# SETUP_VAST.md — total setup guide for a fresh Vast.ai instance

> **Audience**: a human operator OR a Claude agent given SSH access to a freshly
> booted Vast.ai instance running `pantomiman/reason-over-search-v1:v1`. Follow
> this end-to-end and you (or the agent) will be able to run **eval** (M1
> reproduction) and **training** (M2 GRPO + recipe-ablation runs) without
> needing any other doc.
>
> Sister doc for the ALICE HPC cluster: [`training/scripts/bootstrap_alice.sh`](../../training/scripts/bootstrap_alice.sh)
> (Apptainer-based, same image, ZFS-aware paths). Use this one for Vast.

The fastest path is **prebuilt docker image + bootstrap script**. Total cold-to-running time is ~10 min on the HF fast path, ~35 min on the compile fallback.

---

## 0. Pre-flight (do this before booting the instance)

| Resource | Minimum | Recommended | Why |
|---|---|---|---|
| GPU | 1× 24 GB (4090) | 1× 80 GB (A100 / H100 / H200) | Training needs ≥40 GB; eval is fine on 24 GB |
| Host RAM | 32 GB (IVF-SQ8 only, 1 worker) | 150 GB (IVF-SQ8, 8 workers for training) | 8 retriever workers each load ~16 GB index; flat IP needs ~65 GB |
| Disk | 60 GB (eval-only) | 150 GB (training + eval) | Image ~30 GB, corpus 14 GB, IVF-SQ8 16 GB, models ~40 GB, headroom |
| Public ports | none required | 3000, 3005 | Optional for external SGLang / retriever; default workflow is local-only |

Have ready:
- A Vast.ai account with billing set up
- Your SSH public key registered with Vast (Account > Keys)
- (Optional) A `WANDB_API_KEY` for live training curves
- (Optional) A `HF_TOKEN` if you plan to push artifacts back to HF (downloads are public)

See [`docs/setup/HARDWARE.md`](../setup/HARDWARE.md) for the full accelerator comparison and [`docs/setup/VAST_AI_PLAN_A.md`](../setup/VAST_AI_PLAN_A.md) for cost-optimised eval fleets.

---

## 1. Launch the instance

Vast template fields:

| Field | Value |
|---|---|
| Image | `pantomiman/reason-over-search-v1:v1` |
| Disk space | 150 GB (drop to 60 GB for eval-only boxes) |
| Launch mode | Default Vast SSH/Jupyter mode |
| GPU filter | A100-80GB or H100-80GB for training; any 24 GB+ for eval |
| On-start script | leave blank; we bootstrap manually |
| Open ports | 3000 (SGLang), 3005 (retriever); only if you need external access |

The image bundles:
- Two conda envs: `retriever` (faiss-cpu, FastAPI) and `evaluation_search_r1` (sglang, FlashRAG)
- App code at `/app` (read-only baseline; we work under `/workspace`)
- A boot hook at `/etc/vast_boot.d/10-fix-ssh-perms.sh` that fixes `/root/.ssh` permissions before sshd starts (avoids "bad ownership" SSH errors)
- A pre-warmed uv wheel cache for the NeMo-RL training venv (~13 GB at `/.uv/cache`); the bootstrap script uses this

Source: [`docker/reason-over-search-v1/`](../../docker/reason-over-search-v1/). Rebuild instructions in [`docker/reason-over-search-v1/README.md`](../../docker/reason-over-search-v1/README.md).

---

## 2. Connect

```bash
# Vast UI gives the line; format is:
ssh -o StrictHostKeyChecking=accept-new -p <PORT> root@ssh<HOST>.vast.ai
```

If you see `bad ownership or modes for file /root/.ssh/authorized_keys`, the boot hook didn't run. Open Vast's web shell and run:

```bash
chown root:root /root/.ssh /root/.ssh/authorized_keys
chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
```

---

## 3. Clone the repo + sanity (≈30 s)

The image stages app code at `/app`, but `/workspace` is the persistent volume; clone there so your work survives image rebuilds.

```bash
cd /workspace
git clone https://github.com/<your-org>/reason_over_search.git
cd reason_over_search
git checkout <branch>   # research_v2, develop, main, or whichever is current

# Sanity
pwd && which uv && conda env list 2>/dev/null | head && \
  nvidia-smi --query-gpu=name --format=csv,noheader && \
  df -h /workspace
```

Expect: cwd is the repo root; `uv` resolves; conda envs include `retriever` and `evaluation_search_r1`; one A100/H100/etc.; ≥30 GB free.

If anything is missing, **STOP**. Don't try to repair the image; it's a docker-rebuild job, not a runtime fix. Confirm `pantomiman/reason-over-search-v1:v1` (or rebuilt v1+) was used.

---

## 4. Bootstrap (≈10 min cold HF path; ≈35 min compile fallback; <1 min warm)

One command. Idempotent: re-running prints "already done" for steps that are.

```bash
bash training/scripts/bootstrap.sh
```

Watch for the final line `▶ Bootstrap complete.`. If it errors out, read the error and stop; `bootstrap.sh` checks for the conditions it needs and fails loud.

What it does, in order (so you know what to expect):

1. **Sanity** (envs, disk, RAM, GPU). Instant.
2. **Git LFS pull** if `data/training/nq_hotpotqa_train/train.parquet` is missing (or smaller than 1 KB → still an LFS pointer). ~30 s.
3. **Download Qwen3.5-2B-Base + Qwen3.5-2B** to `$HF_HOME=/workspace/hf_cache` if not cached (~4 min, ~8 GB).
4. **`uv sync --extra vllm`** to materialize `training/nemo_rl/.venv` (~2 min from the pre-warmed wheel cache).
5. **Download the pre-built v2/automodel uv venv** (~5 GB) from `pantomiman/reason-over-search-v1-venvs` on HuggingFace Hub and extract it to `training/nemo_rl/venvs/.../DTensorPolicyWorkerV2/` (~3 min, fast path). Falls back to host-shell compile (~25 min) if HF download fails. **This step cannot run inside a Ray actor** because `nv-grouped-gemm`'s setup.py calls `torch.cuda.init()` at install time and the actor has no GPU.
6. **Start the IVF-SQ8 retriever** with 8 workers on port 3005, wait until `Uvicorn running` lands in `/tmp/retriever.log`, smoke-check the `/health` endpoint.

Override flags:

```bash
SKIP_V2_BUILD=1 bash training/scripts/bootstrap.sh         # skip the long compile/download (eval-only)
SKIP_RETRIEVER=1 bash training/scripts/bootstrap.sh        # don't auto-start retriever
V2_BUILD_FROM_SOURCE=1 bash training/scripts/bootstrap.sh  # force compile, skip HF tarball
```

After bootstrap finishes:

```bash
# Confirm retriever is healthy
curl -sS http://127.0.0.1:3005/health
# {"status":"healthy"}

# Confirm models cached
ls $HF_HOME/hub/ | grep Qwen3.5-2B

# Confirm v2 venv exists (only if you didn't pass SKIP_V2_BUILD=1)
ls training/nemo_rl/venvs/*/DTensorPolicyWorkerV2/bin/python
```

---

## 5. Configure W&B (optional but recommended for training)

If you have a `WANDB_API_KEY`:

```bash
echo "WANDB_API_KEY=<your_key>" >> training/.env
echo "WANDB_PROJECT=reason_over_search_2b_v1" >> training/.env  # or your project name
```

If you don't, prepend `WANDB_MODE=disabled` to any training launch command (or set it once in `training/.env`).

---

## 6. Eval path (M1 reproduction; SGLang + retriever)

The retriever started by `bootstrap.sh` uses the IVF-SQ8 index, which is faster but approximate. The published M1 numbers in [`docs/report/RESULTS_m1.md`](../report/RESULTS_m1.md) used the **flat IP** index for paper-fidelity recall.

### 6a. (Optional) Switch retriever to flat IP for paper-fidelity eval

Only do this if you specifically want to reproduce M1 numbers exactly. Needs ~65 GB host RAM.

```bash
# Download flat index (~60 GB after merge + gunzip)
cd /workspace/reason_over_search/local_retriever
huggingface-cli download PeterJinGo/wiki-18-e5-index --repo-type dataset --local-dir indexes
cat indexes/part_aa indexes/part_ab > indexes/wiki18_100w_e5_flat_inner.index.gz
gunzip -f indexes/wiki18_100w_e5_flat_inner.index.gz
rm -f indexes/part_aa indexes/part_ab

# Stop IVF retriever, restart on flat
pkill -f retriever_serving.py; sleep 2
nohup /venv/retriever/bin/python retriever_serving.py \
  --config retriever_config.yaml --num_retriever 4 --port 3005 \
  --index ./indexes/wiki18_100w_e5_flat_inner.index \
  > /tmp/retriever.log 2>&1 &
until curl -sf http://127.0.0.1:3005/health; do sleep 5; done
```

### 6b. Download GRPO checkpoints (~27 GB)

```bash
cd /workspace/reason_over_search/evaluation_search_r1
huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo \
  --local-dir search_r1_base_model
huggingface-cli download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo \
  --local-dir search_r1_instruct_model

# Verify sha256 (catches partial / wrong-checkpoint downloads)
sha256sum search_r1_base_model/model-00001-of-00003.safetensors
# Expected: 7ac54e1b9762c3c6d639da28a2cca177fe7db092ff5cf6e5a9a7849a36a9dabf
sha256sum search_r1_instruct_model/model-00001-of-00003.safetensors
# Expected: 3d787062256210d1cc6c7c666a0ab0ac83a7a5d0296281b4811df72c968ccd35
```

Full sha256 + sizes table: [`docs/eval/REPRODUCIBILITY.md#models—confirmed-grpo`](../eval/REPRODUCIBILITY.md#models--confirmed-grpo).

### 6c. Start SGLang on port 3000

```bash
cd /workspace/reason_over_search
scripts/manage_sglang.sh switch instruct   # or 'base'
# Logs at /tmp/sglang_<variant>.log; first-time JIT compile is ~3 to 5 min

# Sanity
curl -sS http://127.0.0.1:3000/get_model_info | grep -o instruct
```

The script kills any running SGLang, launches the requested variant on `127.0.0.1:3000` with the canonical flags (`--context-length 8192 --dtype bfloat16 --tp 1 --trust-remote-code`), and waits for `/get_model_info` to come up.

**Constraint**: GPU FAISS + SGLang **cannot share a 24 GB 4090** (16 GB index + 22 GB SGLang > 24 GB VRAM). On 4090 keep retriever on CPU. On 80 GB cards (A100/H100/H200) GPU FAISS + SGLang fit, but CPU FAISS is fine and one less moving part.

### 6d. Smoke-eval (1 dataset × 1 seed; ~6 min on 4090, ~2 min on H100)

```bash
cd /workspace/reason_over_search
scripts/run_one.sh instruct bamboogle 1 > /tmp/smoke.log 2>&1
LATEST=$(ls -dt evaluation_search_r1/results/bamboogle/bamboogle_*_search_r1_instruct_seed1 | head -1)
grep -E "^(em|f1):" "$LATEST/metric_score.txt"
```

Expected: EM ≈ 0.36, F1 ≈ 0.45 on Bamboogle/instruct (n=125, greedy). If EM drops to 0.05–0.10 the eval is broken; check `apply_chat`/template/parser surface in [`docs/eval/PAPER_VS_OURS_AUDIT.md`](../eval/PAPER_VS_OURS_AUDIT.md) before touching anything else.

`run_one.sh` is **resume-aware**: any `(variant, dataset, seed)` cell with a `metric_score.txt` is skipped on re-run. Force a re-run via:

```bash
rm -rf evaluation_search_r1/results/bamboogle/bamboogle_*_search_r1_instruct_seed1
```

### 6e. Full sweep

Plan B reduced (1 k subsamples, both variants × 7 datasets × 5 seeds) lands in ~30 min on H100, ~3 h on 4090:

```bash
nohup bash scripts/sweep_b_reduced.sh > /workspace/logs/sweep_b.log 2>&1 & disown
tail -f /workspace/logs/sweep_b.log
```

Plan A (full splits × 5 seeds): see [`docs/setup/VAST_AI_PLAN_A.md`](../setup/VAST_AI_PLAN_A.md) for the multi-instance fleet recipe.

Aggregate after a sweep:

```bash
python scripts/aggregate.py
# Writes evaluation_search_r1/results/_aggregate/<variant>_summary.json
```

---

## 7. Training path (M2 GRPO + recipe ablations)

Pre-reqs: bootstrap finished without `SKIP_V2_BUILD=1` (you need the `DTensorPolicyWorkerV2` venv) and the retriever is up on 3005.

### 7a. Pick a combo

| Goal | Variant | Arm | Notes |
|---|---|---|---|
| Smoke-test the full pipeline (~5 min × 4 combos) | base + hybrid | qwen_native + paper | 2 outer steps × 4 prompts × group=5 |
| Recipe-ablation (`C-minimal`, `+MC-GRPO`, etc.) | base | qwen_native | 1005 steps; ~11–17 d on 1× A100, ~5–8.5 d on 1× H100 |
| One-off `full custom` | (you choose) | (you choose) | compose your own Hydra args |

The recipe-ablation plan supersedes the original 6-run plan. See [`docs/TODO_2026-05-04.md`](../TODO_2026-05-04.md) for the prioritised list (systems-only → C-minimal → +MC-GRPO → +S-GRPO → +E2H curriculum) and [`docs/milestone_2/PHASE_2_RUNBOOK.md`](../milestone_2/PHASE_2_RUNBOOK.md) for the full operational runbook.

### 7b. Required Hydra overrides

**Smoke knobs** (add to any smoke run):

```
grpo.max_num_steps=2 grpo.num_prompts_per_step=4 policy.train_global_batch_size=20
```

**Always-required Qwen3.5 overrides** (smoke OR full):

```
policy.sequence_packing.enabled=false
policy.dynamic_batching.enabled=true
policy.train_micro_batch_size=2
```

Without these, GatedDeltaNet/Mamba layers crash with `CUDA illegal memory access` during `get_logprobs`. See [`training/fix/CHANGES.md`](../../training/fix/CHANGES.md) for the diagnostic.

### 7c. Launch

Single combo (smoke):

```bash
cd /workspace/reason_over_search
export HF_HOME=/workspace/hf_cache
ulimit -n 65536
rm -rf /tmp/ray   # clear stale Ray cluster state

bash training/scripts/run_grpo_1xa100.sh \
  --variant base --seed 42 --arm qwen_native \
  -- \
  policy.sequence_packing.enabled=false \
  policy.dynamic_batching.enabled=true \
  policy.train_micro_batch_size=2 \
  grpo.max_num_steps=2 grpo.num_prompts_per_step=4 policy.train_global_batch_size=20
```

Single combo (full 1005-step run):

```bash
nohup bash training/scripts/run_grpo_1xa100.sh \
  --variant base --seed 42 --arm qwen_native \
  -- \
  policy.sequence_packing.enabled=false \
  policy.dynamic_batching.enabled=true \
  policy.train_micro_batch_size=2 \
  > /workspace/logs/m2_full.log 2>&1 &
disown
tail -f /workspace/logs/m2_full.log
```

Watch for `wandb: 🚀 View run at https://wandb.ai/...` in the launcher output; that's your live curve.

For all 4 smoke combos sequentially:

```bash
for variant in base hybrid; do
  for arm in qwen_native paper; do
    bash training/scripts/run_grpo_1xa100.sh \
      --variant $variant --seed 42 --arm $arm \
      -- \
      policy.sequence_packing.enabled=false \
      policy.dynamic_batching.enabled=true \
      policy.train_micro_batch_size=2 \
      grpo.max_num_steps=2 grpo.num_prompts_per_step=4 policy.train_global_batch_size=20
    # Rename trace dir so the four don't collide
    mv logs/exp_$(ls -1t logs/ | grep -E '^exp_[0-9]+$' | head -1 | sed 's/exp_//') \
       logs/smoke_${variant}_${arm}
  done
done
```

### 7d. Health checks

- A successful step prints `========================= Step N/M =========================` followed by `Logged data to logs/exp_NNN/train_data_step{N}.jsonl`.
- For smokes, after step 2 the run exits cleanly with `EXIT=0`.
- For full runs, **checkpointing is disabled by default** in the YAML (`checkpointing.enabled: false`). Re-enable in `training/configs/grpo_qwen3.5_2b_1xa100.yaml` if you want to resume; see [`docs/training/NEMO_RL_KNOBS.md`](../training/NEMO_RL_KNOBS.md).
- **Step-100 tripwire**: at step 100 (~1 h H100, ~2.5 h A100), check W&B `train/reward_mean`. If flat at ~0, abort and debug rather than burning the rest of the run.

### 7e. Sample extraction (smoke only)

After running all 4 smoke combos, consolidate samples and reward distributions:

```bash
python3 training/scripts/extract_smoke_samples.py
# Writes docs/training/SMOKE_RESULTS.md
# Rename to SMOKE_RESULTS_<UTC-DATE>.md before committing; previous dated runs live under docs/archive/training/
```

---

## 8. Background process discipline

Vast SSH connections drop. Always run long jobs under `nohup … & disown` (logs to `/tmp/` or `/workspace/logs/`) or under `tmux`.

```bash
mkdir -p /workspace/logs

# Option A: nohup
nohup bash scripts/sweep_b_reduced.sh > /workspace/logs/sweep_b.log 2>&1 & disown
tail -f /workspace/logs/sweep_b.log

# Option B: tmux
tmux new -s eval
# inside tmux: run anything
# Ctrl-b d   to detach
# tmux attach -t eval   to reattach
```

---

## 9. Tear-down checklist (before destroying the instance)

If results matter, exfiltrate first:

```bash
# Eval results
tar czf /tmp/results-$(date +%F).tar.gz \
  /workspace/reason_over_search/evaluation_search_r1/results
scp -P <PORT> root@ssh<HOST>.vast.ai:/tmp/results-*.tar.gz .

# Training traces
tar czf /tmp/training-traces-$(date +%F).tar.gz \
  /workspace/reason_over_search/logs

# Or sync to object storage
rclone sync /workspace/reason_over_search/evaluation_search_r1/results \
            r2:reason-over-search/results-$(date +%F)
```

W&B keeps the live curves; per-step JSONL on disk has the raw trajectories. Both are worth pulling before destroy.

Then destroy from the Vast UI; persistent-volume billing stops on destroy.

---

## Time + cost expectations (level-set the user)

| Phase | Wall-clock | Cost @ $1.20/h on 1× A100 |
|---|---|---|
| Bootstrap, fresh box, HF fast path | ~10 min | ~$0.20 |
| Bootstrap, fresh box, compile fallback | ~35 min | ~$0.70 |
| Bootstrap, re-used box | <1 min | ~$0.02 |
| Smoke combo (2 steps × 4 prompts) | ~5 min | ~$0.10 |
| Eval Plan B reduced (full 1k×7×2×5) | ~3 h on 4090, ~30 min on H100 | ~$3 to $5 |
| Eval Plan A (full split × 5 seeds × 7 × 2) | ≤24 h fleet | ~$58 to $108 (multi-instance, see VAST_AI_PLAN_A.md) |
| Full Phase-2 GRPO run (1005 steps × 510 trajectories) on 1× A100 | ~11 to 17 d | ~$300 to $490 |
| Same on 1× H100 80 GB SXM | ~5 to 8.5 d | ~$240 to $410 |
| Same on 1× H200 141 GB SXM | ~4 to 7 d | ~$270 to $470 |

Recommended hardware for full training runs: **1× H100 80 GB SXM** for best $/run. See [`docs/training/SMOKE_RESULTS_2026-05-06.md` "Full-training wall-clock + cost"](../training/SMOKE_RESULTS_2026-05-06.md#full-training-wall-clock--cost-phase-2-real-config).

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `bootstrap.sh: Conda env 'retriever' missing` | wrong docker image | Confirm `pantomiman/reason-over-search-v1:v1` (or rebuilt v1+) |
| `bootstrap.sh: nvidia-smi not found` | not a GPU instance | Vast template selection issue; pick a GPU machine |
| Retriever times out during rollouts | flat IP fallback by mistake | Confirm `local_retriever/retriever_config.yaml`'s `index_path` is the IVF; restart with `--num_retriever 8` (`bootstrap.sh` does this) |
| `KeyError: 'task_name'` in DataLoader | running pre-fix code | `git pull`; fix is in `training/src/datasets/search_r1.py` |
| `No actor environment registered for ... SearchR1Environment` | running pre-fix registry | `git pull`; fix is in `training/src/registry.py` |
| `unknown arg: --` from launcher | running pre-fix wrapper | `git pull`; fix is in `training/scripts/run_grpo_1xa100.sh` |
| `RuntimeError: No CUDA GPUs are available` (during uv install) | v2 venv being built inside Ray actor | bootstrap.sh handles this from host shell; if you bypass bootstrap, run the v2 build step manually per `training/fix/CHANGES.md` |
| `AttributeError: 'Qwen3_5Model' object has no attribute 'layers'` | `_v2: false` was overridden | Use the YAML default (`_v2: true`); ensure the v2 venv exists |
| `CUDA error: an illegal memory access` in `torch_chunk_gated_delta_rule` | sequence packing enabled with Qwen3.5 | Add `policy.sequence_packing.enabled=false` (always required) |
| Ray cluster startup timeout | stale `/tmp/ray` | `rm -rf /tmp/ray` and relaunch |
| `OSError: model.safetensors not found` from SGLang | GRPO download incomplete | Re-run `huggingface-cli download`; `du -sh search_r1_*_model` should be ~14 GB each |
| `EM` wildly off paper numbers | wrong checkpoint or `apply_chat=False` for base | Verify sha256 vs [`docs/eval/REPRODUCIBILITY.md`](../eval/REPRODUCIBILITY.md); check [`scripts/run_one.sh:35`](../../scripts/run_one.sh#L35) |
| `bad ownership or modes for file /root/.ssh/authorized_keys` | boot hook didn't run | See § 2 manual fix |
| `cuda out of memory` from SGLang | another process holds VRAM | `nvidia-smi` to find culprit; `pkill -f <name>` |

---

## Quick reference — paths after setup

```
/workspace/reason_over_search/
├── data/                                          # 7 eval datasets (jsonl, LFS)
├── data_subsample/                                # Plan B 1k subsamples (LFS)
├── data/training/nq_hotpotqa_train/               # Training parquets (LFS)
├── local_retriever/
│   ├── corpus/wiki18_100w.jsonl                   # 14 GB (downloaded by bootstrap)
│   ├── indexes/wiki18_100w_e5_ivf4096_sq8.index   # 16 GB (downloaded by bootstrap)
│   ├── indexes/wiki18_100w_e5_flat_inner.index    # 65 GB (optional, paper-fidelity eval)
│   └── models/e5-base-v2/                         # 0.5 GB encoder
├── evaluation_search_r1/
│   ├── search_r1_base_model/                      # 14 GB (download in § 6b)
│   ├── search_r1_instruct_model/                  # 14 GB (download in § 6b)
│   └── results/                                   # eval outputs land here
└── training/
    ├── nemo_rl/.venv/                             # main training venv
    ├── nemo_rl/venvs/.../DTensorPolicyWorkerV2/   # the v2 worker venv
    └── .env                                       # WANDB_API_KEY etc.

$HF_HOME = /workspace/hf_cache                     # Qwen3.5-2B + Qwen3.5-2B-Base cached here
```

---

## See also

- [`training/scripts/bootstrap.sh`](../../training/scripts/bootstrap.sh) — the actual script
- [`training/scripts/bootstrap_alice.sh`](../../training/scripts/bootstrap_alice.sh) — same flow on ALICE HPC via Apptainer
- [`docs/setup/BOOTSTRAP_NEW_INSTANCE.md`](../setup/BOOTSTRAP_NEW_INSTANCE.md) — generic bootstrap (any host, not Vast-specific); leans more on docker pull / docker run
- [`docs/setup/VAST_INSTANCE_SETUP.md`](../setup/VAST_INSTANCE_SETUP.md) — older Vast-specific eval-only walkthrough; partly superseded by this doc
- [`docs/setup/VAST_AI_PLAN_A.md`](../setup/VAST_AI_PLAN_A.md) — multi-instance fleet design for cost-optimal Plan A eval
- [`docs/setup/HARDWARE.md`](../setup/HARDWARE.md) — accelerator comparison
- [`docs/milestone_2/PHASE_2_RUNBOOK.md`](../milestone_2/PHASE_2_RUNBOOK.md) — operational runbook for Phase-2 training
- [`docs/training/CONVERSATION_CONTEXT.md`](../training/CONVERSATION_CONTEXT.md) — current state of training-side work
- [`docs/training/NEMO_RL_KNOBS.md`](../training/NEMO_RL_KNOBS.md) — every NeMo-RL knob and our recommended values
