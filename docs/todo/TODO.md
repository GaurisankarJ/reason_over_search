---
title: TODO
tags: [todo]
source: internal
created: 2026-05-10
updated: 2026-05-12
---

# TODO

Evergreen pending-work tracker. New TODOs land here; close them by deleting the line (recent context lives in [`log.md`](../log.md), not here). For the current catch-up doc, see [`TODO_2026-05-12.md`](TODO_2026-05-12.md); older catch-up docs at [`TODO_2026-05-11.md`](TODO_2026-05-11.md) (superseded), [`TODO_2026-05-10.md`](TODO_2026-05-10.md), [`TODO_2026-05-04.md`](TODO_2026-05-04.md). Folder index: [`README.md`](README.md).

## Infra

- **Docker `:v1` from-scratch rebuild is broken.** NeMo-RL upstream made `automodel`+`vllm` extras mutually exclusive after 2026-05-02; the original [`Dockerfile`](../../docker/reason-over-search-v1/Dockerfile) `uv sync vllm,automodel` step now fails. Workaround in place: [`Dockerfile.v2`](../../docker/reason-over-search-v1/Dockerfile.v2) layers FROM `:v1` (no rebuild needed). Real fix: pin `NEMO_RL_REF` to a pre-2026-05-02 SHA OR drop one extra (`automodel` is a bootstrap speedup, not a correctness requirement). Not urgent: `:v2` pull works; [`training/scripts/bootstrap.sh`](../../training/scripts/bootstrap.sh) idempotently upgrades transformers, so `:v1` boxes self-heal.

## M4 (Qwen3.5-0.8B baseline eval)

- **M4.4 prompt search at n=300** — additive sub-phase to close (or rule out) the M3-vs-M4.2 cross-family gap (avg Δ −0.042 EM in favour of the smaller, older-family Qwen3-0.6B). Plan in [`milestone_4/MILESTONE_4.md` §M4.4](../milestone_4/MILESTONE_4.md): Phase 1 = 5 candidates × 7 datasets × n=300 (~2 h optimised stack), acceptance bar +0.025 mean EM over the M4.2 lock; Phase 2 ablate winner; Phase 3 full sweep + lock. Companion comment file: [`milestone_4/M4_PROMPTS_SCRATCH.md`](../milestone_4/M4_PROMPTS_SCRATCH.md). Implementation gap: only `qwen35_recall_port` (Phase-2 backup) is wired in [`templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py); Phase-1 templates B/C/D/E and pipeline branches not yet authored. `scripts/orchestrate_m4_4.sh` not yet authored. Base variant stays at `qwen35_minimal_no_system` per the M4.3 lock.
- **One-seed full-data confirmation** for both base + hybrid (the 2026-05-10 M4.2 close-out is single-seed).

## M5 (Qwen3.5-0.8B GRPO training)

- ~~Create `training_m5_1/` scaffold~~ — **DONE** (2026-05-09, [`training_m5_1/`](../../training_m5_1/) live).
- ~~Author `milestone_5/PAPER_VS_OURS_M5.md`~~ — **DONE** (2026-05-10; status "spine only" — fill in remaining TBD cells once M5.1 a3 lands).
- ~~M5 smoke v1-v6~~ — **DONE** (2026-05-10, v6 success: 93.1 s/step ex-warmup; [`logs/exp_006/`](../../logs/exp_006/) committed in `accf98c`).
- ~~M5.1 production-shape smoke~~ — **DONE** (2026-05-10 v7 baseline; established 25 d projection at micro=1).
- **Launch `M5.1-prod-a3`** — awaits user authorization. Prior attempts: a1 crashed at step-50 ckpt save ([§7](../report/RESULTS_SMOKE_m5.md#7-critical-postmortem--step-50-checkpoint-save-crash-2026-05-11)); a2 killed at step 15 on misdiagnosis ([§7.8](../report/RESULTS_SMOKE_m5.md#78-companion-postmortem--the-zombie-gpu-memory-misdiagnosis-2026-05-12)). Fix verified by two smokes. See [`TODO_2026-05-12.md`](TODO_2026-05-12.md) for current state.
- Hardware decision for a3 (Vast A100 vs port to B200/H100/Spheron) — see [`setup/HARDWARE_COMPARISON.md`](../setup/HARDWARE_COMPARISON.md) (status caveat: anchored on a1's step-38-42 data, revisit when a3 reaches steady-state).
- Re-fill `PAPER_VS_OURS_M5.md` TBD cells once a3 lands a clean ckpt (some cells e.g. `num_prompts_per_step: TBD → 64` decision are recorded in other docs but not yet ported back into the mapping table).

## M1 (housekeeping; Jose-owned)

- Format-validity / length-truncation table per (dataset, variant) from v1 sweep JSONs. Extend `aggregate.py` to surface `</answer>` close-rate.
- One-seed full-data runs for base + instruct (~4 h × 2 on a 4090) to confirm v1 config converges at full scale.
- Plan A on Vast.ai: 5 seeds × 7 datasets × 2 variants per [`setup/VAST_AI_PLAN_A.md`](../setup/VAST_AI_PLAN_A.md).
- Untrained Qwen3.5-2B baselines per [`milestone_1/MILESTONE_1.1_QWEN_BASELINES.md`](../milestone_1/MILESTONE_1.1_QWEN_BASELINES.md) (needs eval pipeline qwen_native arm; port from `training/src/environments/parsers.py`).
