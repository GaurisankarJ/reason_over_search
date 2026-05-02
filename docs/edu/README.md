# Educational deep-dives

Explainers that go deeper than the operational docs — for the moments where you want to understand *why* a particular knob exists, not just what to set it to.

These are not load-bearing for running the pipeline. The runbook + configs are the source of truth for "what to do"; these docs are "what's actually happening".

## Index

| File | Topic |
|---|---|
| [SEED.md](SEED.md) | What `--seed` controls, where it propagates in the pipeline, what it does *not* fix, and why we run 3 seeds × 2 variants in Phase 2. |
| [GPU_MEMORY.md](GPU_MEMORY.md) | What lives in VRAM during GRPO training — always-resident state vs. rollout-phase vs. training-phase, with concrete numbers for our 1× A100 80GB config. |

## See also

- [`docs/training/README.md`](../training/README.md) — overlay architecture + step-5 audit summary
- [`docs/training/PAPER_VS_OURS_TRAINING.md`](../training/PAPER_VS_OURS_TRAINING.md) — every paper hyperparameter cross-checked
- [`docs/milestone_two/PHASE_2_RUNBOOK.md`](../milestone_two/PHASE_2_RUNBOOK.md) — concrete run sequence
