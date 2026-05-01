# Training

NeMo-RL setup for Search-R1-style GRPO training of Qwen3.5-2B (base + hybrid).

Scoped in [docs/milestone_two/MILESTONE_2.md](../docs/milestone_two/MILESTONE_2.md).

## Layout (planned)

```
training/
├── nemo_rl/         # cloned from NVIDIA-NeMo/RL, .git removed, modified locally
├── configs/         # GRPO + memory-tuning configs (kept separate)
├── scripts/         # 1× A100 and 2× A100 launch scripts
├── src/             # Search-R1-style reward / chat template / retrieval wiring
└── .env             # W&B key (gitignored)
```

## Status

Not yet set up. See the milestone doc for the step-by-step.
