# Batch math — `gbs`, `num_prompts_per_step`, `num_generations_per_prompt`

> Educational deep-dive on the three batch-size knobs in our GRPO config and how they interact. The "convention" `gbs == num_prompts_per_step × num_generations_per_prompt` shows up in upstream NeMo-RL configs and in our YAMLs without much explanation — this doc is the explanation.

## The three knobs

| Knob | Where | Our value | What it counts |
|---|---|---|---|
| `grpo.num_prompts_per_step` | dataloader | 102 | distinct prompts pulled from the parquet per "training step" |
| `grpo.num_generations_per_prompt` (G) | rollout (vLLM) | 5 | vLLM completions sampled per prompt — the "G" in GRPO |
| `policy.train_global_batch_size` (gbs) | training loop | 510 | trajectories consumed per `optimizer.step()` |

The first two are **rollout-side**: they describe how many trajectories the policy generates per step. The third is **training-side**: how many trajectories the gradient loop consumes between optimizer updates.

## Data flow per "training step"

```
                num_prompts_per_step (= 102)
                     │
                     ▼
              [dataloader yields 102 prompts]
                     │
                     ▼ for each prompt, vLLM samples G traces
                num_generations_per_prompt (G = 5)
                     │
                     ▼
              [510 trajectories produced this step]
                     │
                     ▼ gradient loop consumes gbs trajectories per update
                train_global_batch_size (gbs = 510)
                     │
                     ▼
              num_optimizer_steps_this_step = 510 / gbs
                     │
              ┌──────┴──────┐
              ▼             ▼
        gbs == 510      gbs < 510 (e.g., 51)
        ONE update      TEN updates per step
        (our config)    (verl's pattern)
```

So `prompts × gen` is the **trajectory count produced** per rollout. `gbs` is **trajectories consumed per gradient update**. They don't have to match — but the relationship between them controls how on-policy your training is.

## The two regimes

### A) `gbs == prompts × gen` (our config — 510 == 102 × 5)

- All 510 trajectories enter **one** `optimizer.step()`.
- **Pure on-policy GRPO**: every trajectory was sampled from the *same* policy snapshot that the gradient is now updating. No drift between sampling and learning.
- 1 update per training step. Over 1005 steps, **~1k total gradient updates**.

This is what NeMo-RL's example configs do (e.g., [`grpo_math_1B.yaml`](../../training/nemo_rl/examples/configs/grpo_math_1B.yaml): `32 × 16 = 512 = gbs`). It's the simplest, safest mode.

### B) `gbs < prompts × gen` (verl pattern)

verl's published Search-R1 setup:
- `train_batch_size=512, n_agent=5` → 2560 trajectories per step
- `ppo_mini_batch_size=256` → 2560/256 = **10 gradient updates per step**

Each mini-batch triggers an `optimizer.step()`. The policy parameters change **between** mini-batches inside one step. The first mini-batch (post first update) sees trajectories that are now slightly off-policy: they were sampled by π₀, but the loss is being computed against π₁, π₂, …, π₉.

PPO/GRPO's importance-sampling ratio:

```
ratio = π_current(token) / π_rollout(token)
clipped_ratio = clip(ratio, 1-ε, 1+ε)    # ε = 0.2 in our config
loss = -clipped_ratio × advantage
```

bounds how far the loss can get pulled by off-policy data — that's why PPO/GRPO can tolerate up to a few mini-batches per step at all. With `clip_ratio=0.2`, the per-step movement is capped, but **doing 10 small clipped steps moves the parameters further than 1 single clipped step** on the same data.

Over 1005 verl-style steps: **~10k total gradient updates**, ~10× ours.

## What `force_on_policy_ratio` does

`loss_fn.force_on_policy_ratio: true` is an opt-in *assertion* that runs at config-load:

```python
# nemo_rl/algorithms/grpo.py
if loss_fn_config["force_on_policy_ratio"]:
    assert (
        grpo_config["num_prompts_per_step"]
        * grpo_config["num_generations_per_prompt"]
        == policy_config["train_global_batch_size"]
    ), "force_on_policy_ratio requires train_global_batch_size == num_prompts_per_step × num_generations_per_prompt"
```

It enforces regime A by refusing to start training in regime B. With `force_on_policy_ratio: false` (our default — same as upstream), the assertion is skipped and you're free to deviate.

We *follow* the convention because it's the cleanest mode, but we're not *required* to.

## How NeMo-RL implements gradient accumulation in regime A

Even within regime A's "1 update per step", the 510 trajectories don't fit on the GPU at once — that would need ~80 GB of activation memory. So NeMo-RL chunks the work via **gradient accumulation**:

```
optimizer.zero_grad()
for micro_batch in chunk(510_trajectories, train_micro_batch_size=4):
    loss = forward_backward(micro_batch)
    # gradients accumulate into .grad attributes; no optimizer.step() yet
optimizer.step()       # ONE gradient update averaged over all 510
```

Number of forward+backward passes = 510 / 4 ≈ 128 micro-batches. With sequence packing on (`policy.sequence_packing.enabled: true`), the 4 in `train_micro_batch_size` is interpreted as a *token budget* (`4 × max_seq_length = 16384 tokens`) rather than a sample count, so the 510-non-divisible-by-4 issue is sidestepped — see [`make_microbatch_iterator_for_packable_sequences`](../../training/nemo_rl/nemo_rl/distributed/batched_data_dict.py#L786).

**Gradient accumulation ≠ multiple optimizer.step()**. It's chunking work for memory; the gradient produced is mathematically equivalent to a single forward+backward over all 510 (modulo float-summation order). One optimizer.step() at the end. Regime A.

## Concrete: why the verl-vs-ours difference is real

Identical `lr=1e-6, β=0.001, ε=0.2` between the two setups, but:

| | verl / paper | NeMo-RL / ours |
|---|---|---|
| Trajectories per step | 2560 | 510 |
| Gradient updates per step | 10 | 1 |
| **Total grad updates over 1005 steps** | **~10k** | **~1k** |
| Total trajectories over 1005 steps | 2.57M | 514k |

Same per-update LR, but verl applies it ~10× more often. With clip-bounded steps, the policy moves farther in the same wall-clock budget. So:

- **At step 1005, verl has done ~10× the optimization work we have.**
- **Per-trajectory data efficiency is ~10× higher in verl** (same data is reused 10 times via the mini-batch loop).

This is one of the open questions for our first-pass run: how much does this gap matter empirically? If `train/reward_mean` rises sanely over 1005 NeMo-RL steps, we may not need parity. If it plateaus, we have three options (in order of effort):

1. **More steps**: bump `grpo.max_num_steps` from 1005 → 10000 to match total update count.
2. **Mini-batch within a step**: collect 2560 trajectories per step (`num_prompts_per_step=512`), set `gbs=256` → 10 updates per step like verl. Breaks the upstream convention; need `force_on_policy_ratio: false` (already off).
3. **Accept the divergence**: document it, note that our setup is "less optimization per dollar" but still in the right ballpark.

For first-pass we go with option 3 (do nothing). Decide between 1 and 2 after observing the first run.

## TL;DR

- `prompts × gen` = how many trajectories rollout produces per step.
- `gbs` = how many trajectories one gradient update consumes.
- **`gbs == prompts × gen`** (our config): 1 update per step, pure on-policy. Simplest mode.
- **`gbs < prompts × gen`** (verl): N updates per step, slight off-policy drift bounded by PPO clip. Higher data efficiency.
- `force_on_policy_ratio` is just a config-time assertion to enforce the convention if you want it.
- Our 1k updates vs verl's 10k is the **biggest unmodelled gap** between our setup and the paper's training trajectory. Open question for first-pass.

## See also

- [`SEED.md`](SEED.md) — RNG-side variability across runs (orthogonal to batch math)
- [`GPU_MEMORY.md`](GPU_MEMORY.md) — how the 510 trajectories actually flow through the GPU within one step
- [`docs/training/PAPER_VS_OURS_TRAINING.md §6`](../training/PAPER_VS_OURS_TRAINING.md#6-hyperparameters) — full hyperparameter audit table
- [`docs/training/NEMO_RL_KNOBS.md §3`](../training/NEMO_RL_KNOBS.md) — GRPO algorithm knobs reference
