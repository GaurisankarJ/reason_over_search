---
title: Spheron H200 setup runbook (M5.1 prod)
tags: [spheron, setup, runbook, h200, m5_1]
source: hands-on setup 2026-05-15 (h200-a4 experiment, replacing b200-a3 after spot-host preemption at step 56)
created: 2026-05-15
updated: 2026-05-15
---

# Spheron H200 setup runbook

Step-by-step procedure to provision a Spheron 1× H200 instance with persistent volume, bootstrap the container, and reach a verified smoke run + prod launch for the M5.1 GRPO training experiment (Qwen3.5-0.8B on MuSiQue, ReSearch-paper recipe).

A future Claude Code agent (or human) following this in order should reach "first prod step running on H200" in roughly **45 to 60 min** wall, of which ~15-25 min is one-time docker pull on a fresh host. Subsequent host re-provisions (same volume reattached) skip the heavy bootstrap and reach prod in ~20-30 min.

The runbook is anchored on the H200 hardware reality we discovered the hard way; treat all yaml knob values below as load-bearing for this specific GPU.

## Why H200 (vs B200, vs A100)

| Class | VRAM | RAM | Reason it matters for us |
|---|---:|---:|---|
| Spheron 1× A100 | 80 GB | 196 GB | Too slow at our budget; baseline for cost comparison |
| Spheron 1× B200 (a3 host, preempted 2026-05-15) | 192 GB | 196 GB | More VRAM headroom; loss-prone if no persistent volume |
| Spheron 1× **H200 (a4, this runbook)** | **141 GB** | **196 GB** | Cheaper ($2/h vs $15/h cluster); 50 GB less VRAM than B200 so several yaml knobs need adjustment |

The 50 GB VRAM gap between B200 and H200 is the single biggest reason why a yaml that "just worked" on B200 can OOM on H200; see §6 "Yaml knob deltas for H200" and §8 "Pitfalls" for the specifics.

## Prerequisites

1. Spheron account with active compute deployment for 1× H200 + persistent volume named `miletone5` (600 GB) attached.
2. SSH access to the deployment's IP (key set up in Spheron console).
3. Both HF model repos provisioned and empty (only README + config_snapshot + .gitattributes):
   - `pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42` (primary)
   - `pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-backup` (backup)
4. HF dataset `pantomiman/reason-over-search-v1-venvs` has `dtensor_policy_worker_v2.tar.gz` (6.5 GB pre-built DTensor V2 venv) accessible with your HF token.
5. WANDB_API_KEY in your environment for the run.
6. Branch `experiment_1_h200` exists locally on the volume (fork from `experiment_1_b200`); commit `48d9a64` is the canonical h200-rename commit.

## Stage 0: Provision a new H200 instance

In the Spheron console:
1. Deploy a new compute instance: 1× H200, region matching the `miletone5` volume.
2. Attach the `miletone5` volume (this is critical; without it the run state is host-bound and will be lost on spot preemption like the B200 a3 incident).
3. Wait for ssh to be ready. The instance hostname will be `computeinstance-<random>`.
4. Note the instance IP (e.g. `204.12.169.243`).

The deployment image type doesn't matter; we override its entrypoint. The Vast-style images (`pantomiman/reason-over-search-v1:v2`) work fine when re-entered with bash.

## Stage 1: Mount the persistent volume

SSH in as `ubuntu`:

```bash
ssh ubuntu@<INSTANCE_IP>
```

Mount the virtiofs volume (tag is `miletone5`):

```bash
sudo mkdir -p /mnt/miletone5
sudo mount -t virtiofs miletone5 /mnt/miletone5
echo "miletone5 /mnt/miletone5 virtiofs defaults 0 0" | sudo tee -a /etc/fstab
```

**Verify same volume** (proves we got the right disk; this file was written during initial setup on a prior host):

```bash
cat /mnt/miletone5/.mount_test.txt
# Expected: "mount-test 2026-05-15T10:38:53Z host=computeinstance-u00pxp20247myv137q"
ls /mnt/miletone5/workspace/
# Expected: corpus/  hf_cache/  indexes/  models/  reason_over_search/  results/  state/  smoke.log
df -h /mnt/miletone5
# Expected: ~493 GB free at first H200 reuse, more after smoke artifacts cleaned
```

If the test file is missing, **stop**; we attached the wrong volume. Open a support ticket with Spheron to confirm the volume attachment.

## Stage 2: Pull Docker image + start container

The Docker image cache does NOT survive host preemption. On a new host we pull from scratch. Image is ~46 GB and takes ~7-15 min to pull on Spheron's network.

```bash
# Add ubuntu to docker group (need a new session for it to take effect)
sudo usermod -aG docker ubuntu

# Pull image (run in background, monitor)
nohup sudo docker pull pantomiman/reason-over-search-v1:v2 > /tmp/docker_pull.log 2>&1 &
# Wait for completion
until ! ps aux | grep "docker pull" | grep -qv grep; do
    sleep 15
    sudo tail -1 /tmp/docker_pull.log
done
sudo docker images  # expect: 46.4 GB
```

Once pulled, start the container with the volume bind-mounted to `/workspace`. Override the Vast entrypoint (which requires `VAST_TCP_PORT_8080` and isn't suitable for an interactive bootstrap):

```bash
sudo docker run -d \
  --name h200-a4 \
  --gpus all \
  --network=host \
  --ipc=host \
  --shm-size=32g \
  --entrypoint /bin/bash \
  -v /mnt/miletone5/workspace:/workspace \
  -w /workspace/reason_over_search \
  -e VAST_TCP_PORT_8080=8080 \
  -e VAST_TCP_PORT_22=22 \
  -e HF_HOME=/workspace/hf_cache \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  pantomiman/reason-over-search-v1:v2 \
  -c "tail -f /dev/null"
```

**Verify** (all three must pass):

```bash
sudo docker ps | grep h200-a4
sudo docker exec h200-a4 nvidia-smi --query-gpu=name --format=csv,noheader
# Expected: NVIDIA H200
sudo docker exec h200-a4 ls /workspace/reason_over_search/.git
```

## Stage 3: Branch + permissions

```bash
sudo docker exec h200-a4 bash -c '
cd /workspace/reason_over_search
git config --global --add safe.directory /workspace/reason_over_search
git checkout -B experiment_1_h200
git log -1 --oneline
'
# Expected: 48d9a64 config(m5_1/h200): rename project/run to h200-a4-seed42; fork from b200
```

If any `.git/` files are owned by root (from prior container runs), fix:

```bash
sudo find /mnt/miletone5/workspace/reason_over_search/.git -user root -exec chown ubuntu:ubuntu {} \;
```

## Stage 4: Worker venvs (the load-bearing fix)

NeMo-RL's Ray env-builder creates per-worker-class venvs at `training_m5_1/nemo_rl/venvs/`. Three of these are needed for M5.1 prod:

| Worker | Path under `venvs/` | Source |
|---|---|---|
| DTensor V2 (policy) | `nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2` | Pre-built tarball on HF |
| vLLM sync (smoke + sync-engine prod) | `nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker` | Built by Ray on first smoke launch (~10 min) |
| vLLM async (paper-recipe prod, `async_engine: true`) | `nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker` | **Hardcopy of sync venv** (see §4.3) |

If the volume already has all three, skip to Stage 5. On a fresh volume (or after a wipe), populate them in this order:

### 4.1 Restore DTensor V2 venv from HF tarball

```bash
sudo docker exec h200-a4 bash -c '
set -a; . /workspace/reason_over_search/training_m5_1/.env; set +a
/venv/main/bin/python <<EOF
import os
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id="pantomiman/reason-over-search-v1-venvs",
    filename="dtensor_policy_worker_v2.tar.gz",
    repo_type="dataset",
    local_dir="/tmp/v2_tar",
    token=os.environ["HF_TOKEN"],
)
print(f"downloaded: {p}")
EOF
mkdir -p /workspace/reason_over_search/training_m5_1/nemo_rl/venvs
tar -xzf /tmp/v2_tar/dtensor_policy_worker_v2.tar.gz \
  -C /workspace/reason_over_search/training_m5_1/nemo_rl/venvs
'
```

Verify:

```bash
sudo docker exec h200-a4 ls -la \
  /workspace/reason_over_search/training_m5_1/nemo_rl/venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python
# Expected: a symlink to /.uv/python_install/cpython-3.13-linux-x86_64-gnu/bin/python3
```

### 4.2 Build sync vLLM venv

The sync venv is built automatically when Ray first instantiates a `VllmGenerationWorker` actor (i.e. during smoke). The build runs `uv pip install` for ~10 min on first launch and persists on the volume thereafter.

Trigger by running the smoke (Stage 7). After smoke completes, the sync venv is permanently cached on the volume.

### 4.3 Async vLLM venv: hardcopy of sync (CRITICAL — bypasses Ray race)

NeMo-RL's env-builder for the **async** worker has a race condition: it marks the venv as built (`STARTED_ENV_BUILDER` file) before `uv pip install` finishes installing torch, then Ray actors try to `import torch._utils` from the partially-installed venv → `ModuleNotFoundError` → orchestrator dies. We hit this 3 times in the 2026-05-15 setup before deciding to bypass it.

The workaround: hardcopy the sync venv to the async path. They share Python 3.13.13, the same Python home, and identical torch 2.10.0+cu129; functionally they are identical except for the wrapping NeMo-RL worker class.

```bash
sudo docker exec h200-a4 bash -c '
SYNC=/workspace/reason_over_search/training_m5_1/nemo_rl/venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
ASYNC=/workspace/reason_over_search/training_m5_1/nemo_rl/venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker
rm -rf $ASYNC
time cp -a $SYNC $ASYNC
$ASYNC/bin/python -c "import torch, vllm, flashinfer; print(torch.__version__, vllm.__version__, flashinfer.__version__)"
'
```

Expected output:

```
real    10m49s   # on virtiofs; expect 8-12 min depending on instance
2.10.0+cu129 0.17.1 0.6.4
```

Time cost: ~10-12 min on virtiofs (the 12 GB copy is many small files). One-time per volume.

Make sure no `STARTED_ENV_BUILDER` marker remains in the async venv after copy:

```bash
sudo docker exec h200-a4 ls /workspace/reason_over_search/training_m5_1/nemo_rl/venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/STARTED_ENV_BUILDER 2>&1
# Expected: No such file or directory
```

If a marker is present, delete it (`rm`). Ray reading a marker on an already-complete venv may trigger a rebuild attempt and the same race.

## Stage 5: Retriever startup

The retriever is a separate Python process serving `/batch_search` over HTTP at port 3005. It must be up before training. Run inside the container.

For **smoke**: `--num_retriever 4` (~60 GB RAM, more cautious during cold-start).

For **prod**: `--num_retriever 8` (~120 GB RAM, paper-target throughput; OK once kernels are cached and no co-occurring nvcc/cicc compile spikes).

```bash
# Smoke retriever
sudo docker exec -d h200-a4 bash -c '
cd /workspace/reason_over_search/local_retriever
/venv/retriever/bin/python retriever_serving.py \
    --config retriever_config.yaml \
    --num_retriever 4 \
    --port 3005 \
    > /tmp/retriever.log 2>&1
'
```

**Verify** (poll every 10s up to 5 min; loads 16 GB IVF-SQ8 index × N workers):

```bash
until sudo docker exec h200-a4 curl -sS -m 2 http://127.0.0.1:3005/health 2>/dev/null | grep -q '"available":4'; do
  sleep 10
  sudo docker exec h200-a4 tail -1 /tmp/retriever.log
done
sudo docker exec h200-a4 curl -sS http://127.0.0.1:3005/health
# Expected: {"status":"healthy","retrievers":{"total":4,"available":4}}
```

**Memory check after retriever ready**:

```bash
sudo docker exec h200-a4 free -h | head -2
# Expected with 4 workers: ~64 GB used, ~131 GB available
# Expected with 8 workers: ~125 GB used, ~70 GB available
```

When upgrading from smoke (4) to prod (8) retriever:
```bash
sudo docker exec h200-a4 pkill -f retriever_serving
# Wait a beat, then re-launch with --num_retriever 8
```

## Stage 6: Uploader startup

The bulletproof Python uploader (`training_m5_1/scripts/upload_a4_to_hf.py`) watches the checkpoint dir, rollout dir, and prod.log; pushes each new artifact to both primary + backup HF repos with retry and heartbeats. This was authored on 2026-05-15 after the prior bash uploader (a3) silently dropped uploads past step 18 and lost the step 50 ckpt when the host was preempted.

### 6.1 Pre-flight HF access

```bash
sudo docker exec h200-a4 bash -c '
set -a; . /workspace/reason_over_search/training_m5_1/.env; set +a
/venv/main/bin/python - <<EOF
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for repo in [
    "pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42",
    "pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-backup",
]:
    info = api.repo_info(repo, repo_type="model")
    files = api.list_repo_files(repo, repo_type="model")
    print(f"OK {repo}")
    print(f"  files: {files}")
EOF
'
```

Both repos should be reachable; files should be exactly `['.gitattributes', 'README.md', 'config_snapshot.yaml']` if cleaned.

### 6.2 Pre-mark stale rollouts in state file

The volume holds rollout jsonls from prior runs under `/workspace/reason_over_search/logs/exp_NNN/*.jsonl`. If the uploader starts fresh with an empty state file, it will scan and try to upload all of these (~74 files of historical pollution). Pre-mark them as uploaded:

```bash
sudo docker exec h200-a4 /venv/main/bin/python <<EOF
import json
from pathlib import Path
ROLLOUT_DIR = Path("/workspace/reason_over_search/logs")
STATE_FILE = Path("/workspace/state/.uploaded_artifacts.json")
uploaded = set()
for p in ROLLOUT_DIR.rglob("train_data_step*.jsonl"):
    rel = p.relative_to(ROLLOUT_DIR)
    uploaded.add(f"rollout:{rel}")
state = {
    "uploaded": sorted(uploaded),
    "last_log_size": 0,
    "last_timings_size": 0,
    "last_heartbeat": 0,
    "cycle": 0,
}
STATE_FILE.write_text(json.dumps(state, indent=2))
print(f"pre-marked {len(uploaded)} stale jsonls")
EOF
```

### 6.3 Launch uploader

For smoke, point at the smoke ckpt dir and use the smoke state file:

```bash
sudo docker exec -d h200-a4 bash -c '
set -a; . /workspace/reason_over_search/training_m5_1/.env; set +a
/venv/main/bin/python /workspace/reason_over_search/training_m5_1/scripts/upload_a4_to_hf.py \
  --repo-id pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42 \
  --backup-repo-id pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-backup \
  --checkpoint-dir /workspace/reason_over_search/results/grpo/m5_smoke/seed42 \
  --rollout-dir /workspace/reason_over_search/logs \
  --prod-log /workspace/smoke.log \
  --state-file /workspace/state/.uploaded_smoke.json \
  --log-file /workspace/state/uploader_smoke.log \
  > /workspace/state/uploader_smoke.stdout 2>&1
'
```

For prod, swap `m5_smoke` → `m5_prod`, `smoke.log` → `prod.log`, and use `.uploaded_artifacts.json`/`uploader_prod.log`.

**Verify pre-flight canary uploaded to both repos**:

```bash
sudo docker exec h200-a4 tail -15 /workspace/state/uploader_smoke.log
# Expect to see "preflight: ✓ canary uploaded" for both repos
```

## Stage 7: Smoke launch + verification

The smoke uses `training_m5_1/configs/m5_smoke.yaml` (20 traj/step × 4 steps) and validates the entire pipeline end-to-end including HF ckpt upload.

```bash
sudo docker exec -d h200-a4 bash -c '
set -a; . /workspace/reason_over_search/training_m5_1/.env; set +a
export TORCH_CUDA_ARCH_LIST=9.0
export RAY_object_store_memory=10737418240
cd /workspace/reason_over_search
exec bash training_m5_1/scripts/smoke.sh > /workspace/smoke.log 2>&1
'
```

### Important env vars

- `TORCH_CUDA_ARCH_LIST=9.0` is REQUIRED on H200. Without it, Ray's `_env_builder` subprocess (which has no GPU access at build time) fails kernel builds with "No CUDA GPUs are available". Hopper is sm_90.
- `RAY_object_store_memory=10737418240` (10 GB) caps Ray's plasma store. Default is 30% × 196 GB = 59 GB reserved upfront; we measured plasma at ~50 MiB actual usage in our pipeline (NeMo-RL uses CUDA-IPC for weight sync, not plasma). The 10 GB cap is free insurance against the 234 GB worst-case-OOM scenario when retriever runs at 8 workers.

### Verify smoke completion (all 4 steps + 2 ckpts + HF upload)

Wait ~17 min wall, then:

```bash
# All 4 steps logged
sudo docker exec h200-a4 grep -E "Max number of steps has been reached|Step [0-9]+/" /workspace/smoke.log | tail -5

# Both checkpoints on volume (smoke saves at step 2 and step 4)
sudo docker exec h200-a4 ls /workspace/reason_over_search/results/grpo/m5_smoke/seed42/
# Expected: step_2/  step_4/

# Each ckpt is ~6.4 GB
sudo docker exec h200-a4 du -sh /workspace/reason_over_search/results/grpo/m5_smoke/seed42/step_*/

# Both ckpts on BOTH HF repos
sudo docker exec h200-a4 grep "step_4 (folder)" /workspace/state/uploader_smoke.log
# Expected: 2 lines (one per repo)

# Cross-check HF API
sudo docker exec h200-a4 bash -c '
set -a; . /workspace/reason_over_search/training_m5_1/.env; set +a
/venv/main/bin/python - <<EOF
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for repo in [
    "pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42",
    "pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-backup",
]:
    files = api.list_repo_files(repo, repo_type="model")
    print(f"{repo}:")
    for f in files:
        print(f"  {f}")
EOF
'
```

Each repo should have at minimum: `step_2/` + `step_4/` (with safetensors inside), `logs/train_data/exp_NNN/train_data_step{1,2,3,4}.jsonl`, README, config_snapshot.yaml.

Smoke wall-clock observed (2026-05-15): **~17 min total**:
- Setup (vLLM + DTensor init): ~4.7 min
- Step 1 (cold turn-1 thinking blocks): ~9 min (mostly the model's `<think>` block at temp=1.0)
- Steps 2-3-4 (warm): ~2 min total
- Both checkpoints uploaded to both HF repos: ~15s after step 4 closes

If any of these gates fail, see §8 Pitfalls before retrying. **Do not launch prod until smoke is verified end-to-end.**

## Stage 8: Pre-prod cleanup + HF wipe

Smoke artifacts (step_2, step_4, train_data jsonls) pollute the production HF repos. Clean them so prod ckpts land in a fresh tree.

### 8.1 Stop smoke retriever + uploader

```bash
sudo docker exec h200-a4 pkill -f retriever_serving
sudo docker exec h200-a4 pkill -9 -f "uploaded_smoke.json"
```

### 8.2 Wipe HF repos to baseline (keep README + config_snapshot + .gitattributes)

```bash
sudo docker exec h200-a4 bash -c '
set -a; . /workspace/reason_over_search/training_m5_1/.env; set +a
/venv/main/bin/python - <<EOF
import os
from huggingface_hub import HfApi, CommitOperationDelete
api = HfApi(token=os.environ["HF_TOKEN"])
KEEP = {".gitattributes", "README.md", "config_snapshot.yaml"}
for repo in [
    "pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42",
    "pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-backup",
]:
    files = api.list_repo_files(repo, repo_type="model")
    to_del = [f for f in files if f not in KEEP]
    if to_del:
        ops = [CommitOperationDelete(path_in_repo=f) for f in to_del]
        api.create_commit(repo_id=repo, repo_type="model", operations=ops,
                          commit_message="cleanup: remove smoke artifacts pre-prod")
        print(f"deleted {len(to_del)} from {repo}")
EOF
'
```

### 8.3 Re-init prod uploader state (pre-mark all existing rollouts)

```bash
sudo docker exec h200-a4 /venv/main/bin/python <<EOF
import json
from pathlib import Path
ROLLOUT_DIR = Path("/workspace/reason_over_search/logs")
STATE_FILE = Path("/workspace/state/.uploaded_artifacts.json")
uploaded = set()
for p in ROLLOUT_DIR.rglob("train_data_step*.jsonl"):
    uploaded.add(f"rollout:{p.relative_to(ROLLOUT_DIR)}")
STATE_FILE.write_text(json.dumps({
    "uploaded": sorted(uploaded),
    "last_log_size": 0, "last_timings_size": 0, "last_heartbeat": 0, "cycle": 0,
}, indent=2))
print(f"pre-marked {len(uploaded)} jsonls in prod state file")
EOF
```

### 8.4 Restart retriever with 8 workers + start prod uploader

```bash
# Retriever 8w
sudo docker exec -d h200-a4 bash -c '
cd /workspace/reason_over_search/local_retriever
/venv/retriever/bin/python retriever_serving.py \
    --config retriever_config.yaml --num_retriever 8 --port 3005 \
    > /tmp/retriever.log 2>&1
'
# Wait for ready
until sudo docker exec h200-a4 curl -sS -m 2 http://127.0.0.1:3005/health | grep -q '"available":8'; do
    sleep 10
done

# Prod uploader
sudo docker exec -d h200-a4 bash -c '
set -a; . /workspace/reason_over_search/training_m5_1/.env; set +a
/venv/main/bin/python /workspace/reason_over_search/training_m5_1/scripts/upload_a4_to_hf.py \
  --repo-id pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-seed42 \
  --backup-repo-id pantomiman/qwen3.5-0.8b-grpo-musique-h200-a4-backup \
  --checkpoint-dir /workspace/reason_over_search/results/grpo/m5_prod/seed42 \
  --rollout-dir /workspace/reason_over_search/logs \
  --prod-log /workspace/prod.log \
  --state-file /workspace/state/.uploaded_artifacts.json \
  --log-file /workspace/state/uploader_prod.log \
  > /workspace/state/uploader_prod.stdout 2>&1
'
```

## Stage 9: Yaml knob deltas for H200 prod

These are the differences vs B200 a3's yaml; **all are mandatory for H200**. Apply via `sed` to `training_m5_1/configs/m5_1_research_paper.yaml`. Values reflect the final state after iterating v5 through v11 on 2026-05-15.

| Line | Knob | B200 default | H200 value | Reason |
|---|---|---|---|---|
| 151 | `save_period` | 50 | **10** | a3 lost everything before first ckpt at step 50 (host preemption); 10 caps blast radius to 10 steps |
| 187 | `train_micro_batch_size` | 2 | **2** | v6 mb=4 + ckpt=false OOM'd at 135 GB; v9 mb=4 + ckpt=true OOM'd in log_softmax (needed 30 GiB, 25 free). H200's 141 GB VRAM ceiling forces mb=2 |
| 202 | `dtensor_cfg.activation_checkpointing` | true | **true** | v7 with ckpt=false OOM'd at log_softmax (Qwen3.5 248K vocab × fp32 logits ≈ 14 GB transient). Keep ckpt=true to free activation memory; B200 could afford ckpt=false, H200 cannot |
| 385 | `generation.vllm_cfg.async_engine` | true | **true** | v8 (async) ran step 1 + 2 successfully; v10b (sync) deadlocked in FlashInfer GDN kernel (P12). Async is mandatory on H200, not optional. See P12 |
| 394 | `generation.vllm_cfg.gpu_memory_utilization` | 0.8 | **0.5** | H200 has 141 GB vs B200's 192 GB; 0.8 leaves no room for DTensor 84 GB + vLLM KV 17 GB. 0.5 fits with margin |

Verify after applying:

```bash
sudo docker exec h200-a4 grep -nE "^  save_period|^  train_micro_batch_size|^    activation_checkpointing|^      async_engine|^      gpu_memory_utilization" \
  /workspace/reason_over_search/training_m5_1/configs/m5_1_research_paper.yaml
```

Expected output:

```
151:  save_period: 10
187:  train_micro_batch_size: 2
202:    activation_checkpointing: true
224:    activation_checkpointing: false       # this is megatron_cfg (disabled), ignore
385:      async_engine: true
394:      gpu_memory_utilization: 0.5
```

### Stage 9.1: FlashInfer GDN prefill patch (Hopper-only)

**Load-bearing**: this patch prevents the deadlock observed in v10b (60+ min stall in `chunk_gated_delta_rule`; py-spy confirmed). See P12 for the full diagnosis.

vLLM 0.17 hard-codes a Hopper-only fast path for Qwen3.5's GatedDeltaNet prefill at `qwen3_next.py:156`. Force the native Triton path:

```bash
sudo docker exec h200-a4 bash -c '
VENV=/workspace/reason_over_search/training_m5_1/nemo_rl/venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
F=$VENV/lib/python3.13/site-packages/vllm/model_executor/models/qwen3_next.py
cp $F $F.bak
sed -i "s/if current_platform.is_cuda() and current_platform.is_device_capability(90):/if False:  # PATCHED: force forward_native, FlashInfer GDN kernel deadlocks at large prefill batches on H200/" $F
grep -n "PATCHED" $F
'
# Repeat for the async venv path:
sudo docker exec h200-a4 bash -c '
VENV=/workspace/reason_over_search/training_m5_1/nemo_rl/venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker
F=$VENV/lib/python3.13/site-packages/vllm/model_executor/models/qwen3_next.py
cp $F $F.bak
sed -i "s/if current_platform.is_cuda() and current_platform.is_device_capability(90):/if False:  # PATCHED: force forward_native, FlashInfer GDN kernel deadlocks at large prefill batches on H200/" $F
grep -n "PATCHED" $F
'
```

Cost: native (Triton) GDN prefill is ~15-25 % slower per-token than FlashInfer's optimised kernel, but it actually returns. Without this patch, prod-scale (320-prompt) batches in **sync mode** deadlock on Hopper. The async engine (continuous batching, small per-step prefill) does not appear to trigger the stall, but apply the patch to both venvs anyway for safety; B200 (sm_100) and A100 (sm_80) ignore the patch (they never matched the if-condition).

## Stage 10: Prod launch

The full pre-launch checklist must all pass. Run each check; do not relaunch if any gate fails.

### 10.1 Pre-launch gates (9 gates)

```bash
# Gate 1: no live ray/training procs
sudo docker exec h200-a4 bash -c 'ps -eo pid,state,comm | awk "/(ray|run_grpo|raylet|gcs|VllmGen|DTensor)/ && \$2 != \"Z\""' | head
# Expected: empty

# Gate 2: GPU clean (only retriever)
sudo docker exec h200-a4 nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv
# Expected: only retriever python with ~2.85 GB

# Gate 3: Retriever healthy with 8 workers
sudo docker exec h200-a4 curl -sS http://127.0.0.1:3005/health
# Expected: {"status":"healthy","retrievers":{"total":8,"available":8}}

# Gate 4: Prod uploader alive
sudo docker exec h200-a4 bash -c 'ps aux | grep upload_a4 | grep uploader_prod.log | grep -v grep'
# Expected: 1-2 lines (bash wrapper + python)

# Gate 5: Yaml knobs (all 5 in correct state per §9)
sudo docker exec h200-a4 grep -nE "^  save_period|^  train_micro_batch_size|^    activation_checkpointing|^      async_engine|^      gpu_memory_utilization" \
  /workspace/reason_over_search/training_m5_1/configs/m5_1_research_paper.yaml

# Gate 6: Async venv present + has vllm
sudo docker exec h200-a4 bash -c '
ASYNC=/workspace/reason_over_search/training_m5_1/nemo_rl/venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker
[ -d "$ASYNC" ] && [ ! -f "$ASYNC/STARTED_ENV_BUILDER" ] && $ASYNC/bin/python -c "import vllm; print(vllm.__version__)"
'
# Expected: 0.17.1

# Gate 7: Sync vLLM + DTensor V2 venvs present
sudo docker exec h200-a4 ls /workspace/reason_over_search/training_m5_1/nemo_rl/venvs/
# Expected: 3 dirs (sync, async, dtensor)

# Gate 8: Volume free (need ≥ 430 GB for 63 ckpts × 6.4 GB)
sudo docker exec h200-a4 df -h /workspace | tail -1

# Gate 9: HF repos clean (3 files only: README + config_snapshot + .gitattributes)
# (same script as 6.1, expect exactly 3 files each)
```

### 10.2 Launch prod

ONE command (background-detach, redirect to prod.log, set all env vars):

```bash
sudo docker exec -d h200-a4 bash -c '
set -a; . /workspace/reason_over_search/training_m5_1/.env; set +a
export TORCH_CUDA_ARCH_LIST=9.0
export RAY_object_store_memory=10737418240
cd /workspace/reason_over_search
exec bash training_m5_1/scripts/run.sh --mode prod --seed 42 > /workspace/prod.log 2>&1
'
```

**Critical**: do NOT launch twice. The shell exit code 2 (e.g. from an `echo` with parens) does NOT prevent `docker exec -d` from spawning the process. Always verify exactly 1 (or 2 for parent+child) run_grpo.py processes after launch:

```bash
sleep 8
sudo docker exec h200-a4 bash -c 'ps aux | grep run_grpo.py | grep -v grep | wc -l'
# Expected: 1 or 2 (parent + ray child). NOT MORE.
```

If `> 2`, kill the duplicates immediately and relaunch ONE clean copy.

### 10.3 Setup completion + step 1 timing

Setup completion gates (~3-5 min after launch on warm caches):

```bash
# vLLM done with cache hit
sudo docker exec h200-a4 grep "Directly load the compiled graph(s)" /workspace/prod.log
# CUDA graphs captured
sudo docker exec h200-a4 grep "Graph capturing finished" /workspace/prod.log
# DTensor V2 init done
sudo docker exec h200-a4 grep "lm_policy workers" /workspace/prod.log | tail -3
```

Step 1 lands at:

```
========================= Step 1/311 =========================
```

Per-step phase breakdown observed on H200 (with all H200 knobs applied):

| Phase | Step 1 time | % |
|---|---:|---:|
| Setup (once) | ~4-5 min | n/a |
| Rollout (generation) | ~1-2 min | 7 % |
| Reward + logprob + advantage | ~30 s | 3 % |
| Policy training (backward + optimizer) | ~9-12 min | 75-85 % |
| **Total Step 1** | **~12-15 min** | 100 % |

Steps 2+ should match the same ~12-15 min wall.

First checkpoint at step 10 lands at: **~2.5 h after launch** (10 steps × ~14 min/step + 4 min setup).

Full 311-step run wall-clock estimate: **~70-75 h ≈ $140-150 at $2/h**.

## §8 Pitfalls (the ones that cost us time on 2026-05-15)

### P1: gpu_memory_utilization 0.8 fits B200 but OOMs H200

B200 has 192 GB; H200 has 141 GB. At gpu_memory_utilization 0.8, vLLM tries to grab 112 GB on H200; combined with DTensor V2 at ~84 GB + retriever 2.5 GB + KV alloc 17 GB, this overflows. **Symptom: `torch.OutOfMemoryError: Tried to allocate 17.19 GiB. Free: 16.15 GiB` during vLLM KV cache init.** Fix: drop to 0.5. Verified working on H200.

### P2: Async venv race in NeMo-RL env-builder

Ray's env-builder marks the async venv with `STARTED_ENV_BUILDER` before `uv pip install` finishes installing torch. When Ray actors then try to `import torch._utils`, they fail. **Symptom: `ModuleNotFoundError: No module named 'torch._utils'` during IsolatedWorkerInitializer init.** Fix: hardcopy the (working) sync venv to the async path; do NOT let Ray rebuild from scratch. See §4.3.

### P3: nvcc/cicc compilation peak during first smoke

On a fresh host with `num_retriever=8`, the retriever loads 122 GB of FAISS indexes. Concurrently, vLLM's first init triggers ~10 nvcc/cicc kernel compilations (~33 GB transient). Combined peak hits 188 GB / 196 GB → Ray OOM kills vLLM worker. **Symptom: `Ray killed this worker because it was the most recently scheduled task` near step 0 of the first smoke.** Fix: for smoke, use `num_retriever 4` (60 GB instead of 122 GB) giving 58 GB extra headroom. Subsequent prod runs can use 8 retrievers because kernels are now cached on the volume.

### P4: Spheron docker image cache is host-local

When a Spheron spot instance is preempted, the docker image (~46 GB) is lost with the host. Persistent volume only preserves what's mounted at `/workspace`. Always run docker pull from scratch on a new host (Stage 2).

### P5: Vast-style entrypoint requires VAST_TCP_PORT vars

The `pantomiman/reason-over-search-v1:v2` image inherits a Vast.ai-style entrypoint that requires env vars like `VAST_TCP_PORT_8080`. Without an entrypoint override, `docker run` fails with `VAST_TCP_PORT_8080: unbound variable`. Always launch with `--entrypoint /bin/bash` and provide stub env vars.

### P6: Sequence packing OFF for Qwen3.5 hybrid

`sequence_packing: false` is correct and load-bearing. Qwen3.5's GatedDeltaNet kernel crashes when sequences are packed. Do not enable.

### P7: HF dataset has only DTensor V2 tarball, not vLLM venvs

`pantomiman/reason-over-search-v1-venvs` contains only `dtensor_policy_worker_v2.tar.gz`. Both vLLM venvs (sync and async) must be built locally. Building sync (~10 min via Ray on first smoke) is unavoidable; building async (~10-12 min via hardcopy from sync) is the workaround for P2.

### P8: Repeated `docker exec -d` launches can stack silently

If a `docker exec -d ... run.sh ...` command's wrapping shell fails after the exec (e.g. a syntax error in a downstream `echo`), the process still launched inside the container and competes with later launches. **Always verify exactly 1 (or 2 parent+child) run_grpo.py processes immediately after launch**, before assuming a relaunch was clean.

### P9: vLLM tqdm progress hides real generation activity

vLLM's "Processed prompts: 0%" tqdm output uses `\r` to overwrite a single line. From outside, the log appears stalled at 0 % for several minutes during the cold first turn (Qwen3.5 `<think>` blocks at temp=1.0 can generate thousands of internal tokens before any visible output). The training is NOT hung. Confirmation: check that `ray::VllmGenerationWorker` is in `R` state with growing CPU time, and that the orchestrator log file `mtime` is advancing.

### P10: Spheron sshd can hang under fork pressure

Aggressively killing many processes (`pkill -9 -f run_grpo`, `pkill -9 ray`, etc.) can leave the host's sshd in a wedged state for ~10 min as zombies are reaped. If `ssh ubuntu@<ip>` hangs at banner exchange even though `ping` and `nc -zv :22` succeed, **wait or restart the instance via Spheron console**. Volume stays attached during a restart.

### P11: NeMo-RL loads from `training/nemo_rl/` not `training_m5_1/nemo_rl/` sometimes

Error trace paths may show `/workspace/reason_over_search/training/nemo_rl/...` (the M2 reference scaffold) instead of the M5.1 path. This is because Python's import path finds whichever copy of `nemo_rl/` is first in sys.path. Both copies have the same content for our purposes; this is cosmetic, not functional.

### P12: FlashInfer GDN prefill kernel deadlocks at large prefill batches on Hopper (sm_90)

**Cost**: 60+ min stuck on v10b before kill. Cost on prior days: zero, because all prior "working" runs used either smoke batch sizes (small enough that GDN finished) or the async engine (continuous batching, never sends 309-prompt prefill in one call). The bug only manifests on H200 + sync engine + prod-scale rollout batch.

**Hardware/code path**: Qwen3.5-0.8B is a hybrid model (GatedDeltaNet + attention). vLLM 0.17's `qwen3_next.py:156` hard-codes:

```python
if current_platform.is_cuda() and current_platform.is_device_capability(90):
    self._forward_method = self.forward_cuda  # → FlashInfer GDN kernel
else:
    self._forward_method = self.forward_native  # → Triton kernel
```

- **B200 (sm_100)**: never matches the if, always uses native Triton. This is why prior B200 runs never hit the bug.
- **A100 (sm_80)**: never matches the if, always uses native Triton. M2 training was on A100; never hit the bug.
- **H200 (sm_90)**: matches the if, calls `flashinfer.gdn_prefill.chunk_gated_delta_rule`. On small batches (smoke, or async-fed micro-batches) it returns fast. On the **full sync rollout batch of 309 prompts with multi-turn growing context (~3-6 K tokens each after retrieval injection)**, the kernel deadlocks inside CUDA. GPU stays at 100 % util; the Python decode loop never returns from `forward_cuda`.

**Symptom**: vLLM proc is `R`, GPU at 100 % util, 180-190 % CPU, but neither prod.log nor the vLLM worker's `.out` log advances for 30+ min. tqdm never refreshes (because the decode loop is parked inside one CUDA call). No errors, no traceback; just a frozen log with growing CPU time.

**Diagnostic** (py-spy on host, since the container lacks `CAP_SYS_PTRACE`):

```bash
# Copy py-spy out of the container venv to the host:
docker cp h200-a4:/workspace/reason_over_search/training_m5_1/nemo_rl/.venv/bin/py-spy /tmp/py-spy
chmod +x /tmp/py-spy

# Find the host PID for the stuck vLLM worker (use the container-side PID
# you see in `ps -ef` inside docker, then map via docker top):
docker top h200-a4 -eo pid,ppid,cmd | grep VllmGenerationWorker.generate
# host PID column is what you pass below; pass --pid and run with sudo
sudo /tmp/py-spy dump --pid <HOST_PID>
```

Stack confirming the deadlock looks like:

```
Thread X (active): "MainThread"
    gdn_prefill (flashinfer/gdn_prefill.py:63)
    chunk_gated_delta_rule (flashinfer/gdn_prefill.py:207)
    fi_chunk_gated_delta_rule (vllm/model_executor/models/qwen3_next.py:138)
    forward_cuda (vllm/model_executor/models/qwen3_next.py:176)
    ...
    execute_model (vllm/v1/worker/gpu_model_runner.py:3639)
    step (vllm/v1/engine/llm_engine.py:302)
    generate (vllm/entrypoints/llm.py:489)
```

**Fix**: apply the Stage 9.1 patch (force `forward_native` always, ignoring the device-capability check). This is mandatory for H200; idempotent on other GPUs (their device capability never matched anyway).

**Why we missed it for so long**: every smoke run used small batches (kernel happy), and v8 used async (continuous batching submits 1-10 prompts at a time, kernel happy). The moment we switched to sync engine at prod batch size (v10b), we were the first one to hit the bug. The earlier "v4 proven config" was on the same hardware but on async + smoke batch; the proof did not transfer to sync + prod batch.

## Appendix A: Volume contents map (for sanity-checking)

When the volume `miletone5` is correctly attached and the smoke has completed once, expect this layout under `/mnt/miletone5/workspace/`:

```
/workspace/
├── .mount_test.txt                            (≈ 80 B, proves same volume)
├── corpus/                                    (legacy, empty stub)
├── hf_cache/                                  (~40 GB, Qwen3.5-0.8B + Qwen3.5-0.8B-Base)
│   ├── datasets/
│   ├── hub/
│   ├── models--Qwen--Qwen3.5-0.8B/            (~1.7 GB)
│   └── models--Qwen--Qwen3.5-0.8B-Base/       (~1.7 GB)
├── indexes/                                   (legacy, empty stub)
├── models/                                    (legacy, empty stub)
├── reason_over_search/                        (the repo, ~30 GB)
│   ├── .git/
│   ├── data/training/musique/train.parquet    (1.6 MB, post-LFS pull)
│   ├── local_retriever/
│   │   ├── corpus/wiki18_100w.jsonl           (~16 GB, FAISS corpus)
│   │   ├── indexes/wiki18_100w_e5_ivf4096_sq8.index  (16 GB, FAISS index)
│   │   └── retriever_serving.py
│   ├── training_m5_1/
│   │   ├── configs/m5_1_research_paper.yaml   (the prod yaml)
│   │   ├── nemo_rl/
│   │   │   ├── .venv/                         (orchestrator, ~10 GB)
│   │   │   └── venvs/                         (Ray actor venvs, ~33 GB total)
│   │   │       ├── dtensor_policy_worker_v2.../  (~10 GB, from HF tarball)
│   │   │       ├── vllm_worker.../               (~12 GB, built by Ray on smoke)
│   │   │       └── vllm_worker_async.../         (~12 GB, hardcopy of sync)
│   │   └── scripts/
│   │       ├── run.sh
│   │       ├── smoke.sh
│   │       └── upload_a4_to_hf.py
│   └── (rest of repo)
├── results/grpo/m5_prod/seed42/step_NNN/      (ckpts as they land, ~6.4 GB each)
└── state/
    ├── .uploaded_artifacts.json               (prod uploader state)
    ├── .uploaded_smoke.json                   (smoke uploader state)
    ├── uploader_prod.log
    └── uploader_smoke.log
```

Total volume usage after smoke + 10 prod ckpts: ~155 GB. Reserve ≥430 GB for full 63-ckpt run.

## Appendix B: Saving the async venv to HF (future optimization)

Currently the hardcopy step in §4.3 takes 10-12 min on every fresh volume. Future agents could upload a pre-built `vllm_async_generation_worker.tar.gz` to `pantomiman/reason-over-search-v1-venvs` (mirroring the existing DTensor V2 tarball), and download + extract on setup instead of hardcopying. This would save the 10-12 min on first-time setup.

To create the tarball after first successful smoke:

```bash
sudo docker exec h200-a4 bash -c '
cd /workspace/reason_over_search/training_m5_1/nemo_rl/venvs
tar czf /tmp/vllm_async_generation_worker.tar.gz \
  nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker
'
# Upload via HfApi as a dataset file
```

Same applies to the sync vLLM venv (~12 GB tarball).

## References

- Live state log: `docs/log.md` entry under 2026-05-15
- Smoke run details: `docs/report/RESULTS_M5_1_H200.md` §6 (smoke verification)
- Incident on B200 a3 (the host preemption that drove this setup): `docs/report/RESULTS_M5_1_B200.md` §10
- Pre-built venv tarballs: HF dataset `pantomiman/reason-over-search-v1-venvs`
- M5.1 milestone narrative: `docs/milestone_5/MILESTONE_5.md`
- Paper-vs-ours hyperparameter mapping: `docs/milestone_5/PAPER_VS_OURS_M5.md`
