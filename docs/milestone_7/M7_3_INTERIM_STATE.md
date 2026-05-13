# M7.3-short100 interim state snapshot

**Snapshot taken: 2026-05-13 ~11:00 CEST.** Run still in progress.

This file is a VM-crash recovery aid — captures the experiment state, file
paths, and run-resume instructions in one place so a fresh VM can pick up
exactly where this one left off.

## Live run

- **Run name:** `qwen3.5-0.8b-base-musique-m7_m73_short100-seed42-20260513T0709Z`
- **W&B URL:** https://wandb.ai/gaurisankarj1996-leiden-university/reason_over_search_m7_1/
- **Training PID:** 973518 (alive at snapshot time)
- **Watcher PID:** 976571
- **Started:** 2026-05-13 07:09 UTC
- **Mode:** `--mode m73_short100` → `training_m7_1/configs/m7_3_short100.yaml`
- **Prompt file:** `training_m7_1/src/prompts/m7_3_qwen35_base_user.txt` (170 tokens, hard imperative)
- **Ckpt dir:** `results/grpo/m7_m73_short100/seed42/` (cosmetic m7_m73 prefix from `m7_${MODE}` template in run.sh)

## Trajectory through step 7

| step | rew_mean | %tc | %ans | wall (s) |
|---:|---:|---:|---:|---:|
| 1 | 0.0064 | 42.2% | 30.9% | 909 |
| 2 | 0.0038 | 36.9% | 30.0% | 944 |
| 3 | 0.0064 | 40.3% | 36.9% | 857 |
| 4 | 0.0015 | 35.0% | 38.1% | 880 |
| 5 | 0.0028 | 34.1% | 31.9% | 799 |
| 6 | 0.0032 | 34.4% | 36.9% | 879 |
| 7 | 0.0003 | 39.1% | 36.6% | 828 |

**Key observations vs M7.1 same-step:**
- Tool-call rate held at 34-42% (M7.1 was 9-15% by step 7) — imperative working
- Reward flat at noise floor 0.0003-0.0064 (M7.1 was already rising to 0.016 by step 7) — fewer completed answers
- Avg trajectory chars ~4200 (vs M7.1's 2966) — longer multi-turn rollouts
- Pace ~14.5 min/step vs M7.1's ~5 min/step at steady state

## Hypothesis status

OPEN. Two outcomes possible by step 50-100:
- (A) Model learns to wrap up tool calls → emit answer → reward starts rising. M7.3 = success.
- (B) Model stays stuck producing endless tool calls → reward stays at zero → GRPO drifts. Imperative prevented collapse but didn't enable learning.

## Critical artifacts (all in git as of HEAD = 2cf4a68)

- Prompts: `training_m7_1/src/prompts/m7_3_qwen35_base_user.txt`
- Config: `training_m7_1/configs/m7_3_short100.yaml`
- Launcher: `training_m7_1/scripts/run.sh` (`--mode m73_short100`)
- Watcher: `training_m7_1/scripts/watch_m7_3.sh`
- Eval template: `evaluation_qwen35/flashrag/search_r1/templates.py` (`QWEN35_M7_3_NO_SYSTEM_TEMPLATE`)
- Pipeline registry: `evaluation_qwen35/flashrag/pipeline/active_pipeline.py` (`qwen35_m7_3_no_system` registered)
- Plan + design: `docs/milestone_7/MILESTONE_7_3.md`

## Resume instructions if this VM crashes

```bash
cd /workspace/reason_over_search   # or your fresh clone path
git checkout research_v2
git pull

# Sanity: verify the M7.3 ckpt dir if any saves landed:
ls -la results/grpo/m7_m73_short100/seed42/

# Relaunch from scratch (if no ckpt) or from latest ckpt:
nohup env TMPDIR=/workspace/tmp_build RAY_TMPDIR=/workspace/ray_tmp \
        TORCHINDUCTOR_CACHE_DIR=/workspace/torchinductor_cache \
        HF_HOME=/workspace/hf_cache VLLM_CACHE_ROOT=/workspace/vllm_cache \
    bash training_m7_1/scripts/run.sh --mode m73_short100 \
    > logs/m7_3_short100.log 2>&1 &

# Watcher
nohup bash training_m7_1/scripts/watch_m7_3.sh $! > /dev/null 2>&1 &
```

Note: NeMo-RL auto-resumes from the highest `step_N` ckpt found in `checkpoint_dir`.
If a step 50 ckpt exists, the run resumes from there. Otherwise it starts fresh.

## Cost ledger to date (revised — actual $0.806/hr)

| segment | cost |
|---|---:|
| M7.1 (smokes + short100 + idle + extend) | ~$10 |
| M7.3 short100 so far (7 steps, 1h 55min) | ~$1.50 |
| **Total project spend** | **~$11.50** |
| M7.3 projected to step 100 | +$20 |
