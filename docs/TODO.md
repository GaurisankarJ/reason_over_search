---
title: TODO
tags: [todo]
source: internal
created: 2026-05-10
updated: 2026-05-10
---

# TODO

Evergreen pending-work tracker. New TODOs land here; close them by deleting the line (recent context lives in [`log.md`](log.md), not here). For the active ablation path narrative, see [`TODO_2026-05-04.md`](TODO_2026-05-04.md) (frozen catch-up doc).

## Infra

- **Docker `:v1` from-scratch rebuild is broken.** NeMo-RL upstream made `automodel`+`vllm` extras mutually exclusive after 2026-05-02; the original [`Dockerfile`](../docker/reason-over-search-v1/Dockerfile) `uv sync vllm,automodel` step now fails. Workaround in place: [`Dockerfile.v2`](../docker/reason-over-search-v1/Dockerfile.v2) layers FROM `:v1` (no rebuild needed). Real fix: pin `NEMO_RL_REF` to a pre-2026-05-02 SHA OR drop one extra (`automodel` is a bootstrap speedup, not a correctness requirement). Not urgent: `:v2` pull works; [`training/scripts/bootstrap.sh`](../training/scripts/bootstrap.sh) idempotently upgrades transformers, so `:v1` boxes self-heal.

## M4 (Qwen3.5-0.8B baseline eval)

- **Full-sweep close-out**: hybrid 2wikimultihopqa + musique re-running on Vast (the 2026-05-09 run died at 2wiki cell 12/14). Populate [`report/RESULTS_m4.md`](report/RESULTS_m4.md) and the M3-vs-M4 cross-family table once both finish.
- **One-seed full-data confirmation** for both base + hybrid (still listed in [`milestone_4/MILESTONE_4.md`](milestone_4/MILESTONE_4.md) "What's left").

## M5 (Qwen3.5-0.8B GRPO training)

- Create `training_m5_1/` scaffold (copy of [`training/`](../training/) with MuSiQue dataset adapter + parser re-exported from `evaluation_qwen35`). Per [`milestone_5/MILESTONE_5.md`](milestone_5/MILESTONE_5.md).
- Author `milestone_5/PAPER_VS_OURS_M5.md` (clause-by-clause ReSearch-paper-to-YAML mapping); blocks the M5.1 config.
- M5 smoke (50 steps, MuSiQue 200-row subsample) → populate [`report/RESULTS_SMOKE_m5.md`](report/RESULTS_SMOKE_m5.md) v1 row.
- M5.1 production-shape smoke (F1-only reward on `<answer>`, no `\boxed{}` wrapper) → v2 row.

## M1 (housekeeping; Jose-owned)

- Format-validity / length-truncation table per (dataset, variant) from v1 sweep JSONs. Extend `aggregate.py` to surface `</answer>` close-rate.
- One-seed full-data runs for base + instruct (~4 h × 2 on a 4090) to confirm v1 config converges at full scale.
- Plan A on Vast.ai: 5 seeds × 7 datasets × 2 variants per [`setup/VAST_AI_PLAN_A.md`](setup/VAST_AI_PLAN_A.md).
- Untrained Qwen3.5-2B baselines per [`milestone_1/MILESTONE_1.1_QWEN_BASELINES.md`](milestone_1/MILESTONE_1.1_QWEN_BASELINES.md) (needs eval pipeline qwen_native arm; port from `training/src/environments/parsers.py`).
