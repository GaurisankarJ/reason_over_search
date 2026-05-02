# Paper vs Ours — Training

Training-side counterpart to [`docs/eval/PAPER_VS_OURS_AUDIT.md`](../eval/PAPER_VS_OURS_AUDIT.md). Documents every knowing departure from Search-R1's training setup, plus the rationale.

Sources for paper numbers: [Search-R1 arXiv 2503.09516](https://arxiv.org/abs/2503.09516) (Appendix B.2), [Search-R1 GitHub](https://github.com/PeterGriffinJin/Search-R1).

---

## 1. Model

| | Paper | Ours |
|---|---|---|
| Family | Qwen2.5 | **Qwen3.5** |
| Size | 3B (also 7B) | **2B** |
| Variants | base, instruct | **base, hybrid** (Qwen3.5-2B + soft-switch reasoning, no separate Instruct) |
| Pre-training tool format | none consistent | `<tool_call>` / `<tool_response>` (Hermes-style) |
| Native reasoning tag | none | `<think>` (with `enable_thinking` toggle) |

**Why we differ:** Qwen3.5 is the current state-of-the-art for the Qwen family at this scale and supports tool use natively; verl does not support Qwen3.5, which is why we port to NeMo-RL. The 2B size keeps a single A100 80GB sufficient for 1-GPU training; the smaller param count is offset by Qwen3.5's stronger pre-training.

## 2. Framework

| | Paper | Ours |
|---|---|---|
| RL library | [`verl`](https://github.com/volcengine/verl) | [`NeMo-RL`](https://github.com/NVIDIA-NeMo/RL) |
| Rollout backend | vLLM | vLLM |
| Distributed backend | FSDP + CPU offload | DTensor (default) — Megatron available as alternative |

**Why we differ:** verl does not have Qwen3.5 support as of May 2026. NeMo-RL supports GRPO + RLVR for Qwen3.5 out of the box ([NeMo-RL releases](https://github.com/NVIDIA-NeMo/RL/releases)).

## 3. Reward function

**Paper:** rule-based outcome-only reward, exact-match against gold answer:

$$r_\phi(x, y) = \mathrm{EM}(a_\text{pred}, a_\text{gold})$$

No format penalty, no process reward, no learned reward model. ([arXiv 2503.09516, §3.4](https://arxiv.org/html/2503.09516)).

**Ours (Milestone 2 baseline):** **identical to the paper.** We do not modify the reward function in Milestone 2 — the goal is to verify the training pipeline reproduces paper-like dynamics on Qwen3.5 first. Reward ablations (process reward, format reward, retrieval-quality reward) move to **Milestone 3**.

The EM scorer is **byte-identical** to [`evaluation_search_r1/flashrag/search_r1/reward.py`](../../evaluation_search_r1/flashrag/search_r1/reward.py) (the SQuAD-canonical normaliser used in Milestone 1 eval): port at [`training/src/rewards/search_r1.py`](../../training/src/rewards/search_r1.py). Verified by [`training/tests/test_reward_parity.py`](../../training/tests/test_reward_parity.py) (15 tests, including exhaustive case coverage of all branches in `compute_search_r1_reward`). Training-time reward and post-training EM are computed by the same code.

> **Qwen-native arm caveat.** The reward's `is_valid_sequence` and `is_retrieval_correct` walkers are keyed on the paper's tag set (`<think>`, `<search>`, `<information>`, `<answer>`). For the `qwen_native` chat-template arm those checks always fail (different tags), so the reward collapses to **EM-only** via the `score - structure_format_score = 0.8` branch. That's the M2 baseline by design — paper reward unchanged; tag-aware variants move to M3.

## 4. Chat template

| | Paper | Ours |
|---|---|---|
| Retrieval call | `<search>query</search>` | **`<tool_call><function=search><parameter=query>...</parameter></function></tool_call>`** (Qwen3.5 XML-nested) |
| Retrieved doc | `<information>...</information>` | **`<tool_response>...</tool_response>`** in a synthetic user turn |
| Reasoning | `<think>...</think>` | `<think>...</think>` (identical) |
| Final answer | `<answer>...</answer>` | `<answer>...</answer>` (identical — required by EM scorer) |

Full rationale + verbatim jinja in [`CHAT_TEMPLATE.md`](CHAT_TEMPLATE.md).

## 5. Validation set & cadence

> ⚠️ **Currently disabled (first-pass training)**: both YAMLs have `grpo.val_period: 0` and `data.validation: null`; checkpointing is also off. Mechanics-verification first; re-enable per [`VALIDATION.md §7`](VALIDATION.md#7-re-enabling-validation-planned-not-active).

See [`VALIDATION.md`](VALIDATION.md) for the full plan. Summary (the values once re-enabled):

| | Paper / verl yaml | Ours |
|---|---|---|
| Val datasets | NQ + HotpotQA (held-out portions) — used as in-loop validation | Match paper |
| Save cadence | every 100 steps (`trainer.save_freq=100`) | match (`checkpointing.save_period: 100`) |
| **Test cadence** | **every 100 steps (`trainer.test_freq=100`)** — confirmed in [`Search-R1/scripts/nq_hotpotqa/v0.2/train_grpo.sh:69`](https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/nq_hotpotqa/v0.2/train_grpo.sh) | match (validation co-runs with each checkpoint save) |
| **Total steps** | **1005** (`trainer.total_training_steps=1005` in v0.2 verl yaml — supersedes the paper text's "500" claim, which appears to be an Appendix B.2 summary inconsistent with the published-checkpoint training run) | match (`grpo.max_num_steps: 1005`) |
| Val-before-train | `+trainer.val_before_train=true` (run val at step 0 as baseline) | match — log a step-0 val point for sanity |
| Logged metrics | EM, reward, format-validity | EM, reward, format-validity, length-truncation, mean answer tokens |

## 6. Hyperparameters

Paper values from Appendix B.2; verl values from [`Search-R1/scripts/nq_hotpotqa/v0.2/train_grpo.sh`](https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/nq_hotpotqa/v0.2/train_grpo.sh) (the EM-only baseline that produced the published GRPO checkpoints).

| Hyperparameter | Paper / verl | Ours | Notes |
|---|---|---|---|
| Learning rate | `1e-6` | `1e-6` | match |
| Warmup ratio | `0.285` | `0.285` (286 LinearLR iters of 1005) | match (unusually high; revisit after first run) |
| **Total training steps** | **`1005`** (verl `total_training_steps=1005`; supersedes paper text's "500") | **`1005`** | match — see §5 |
| Trajectories per step | `512` (`train_batch_size=512` × n_agent=1 prompt-traj? — verl semantics; we use 510) | `510` (102 prompts × 5 generations) | **2-traj rounding** from `5 × 102 = 510` factorization. Harmless: `force_on_policy_ratio: false`, mismatch isn't asserted. |
| PPO mini/micro batch | `256` / `64` (verl-specific abstraction) | n/a in NeMo-RL | NeMo-RL uses one gradient update per step over `train_global_batch_size` trajectories (with grad accumulation across `train_micro_batch_size`-sized chunks) |
| `train_global_batch_size` | n/a | `510` | matches `num_prompts_per_step × num_generations_per_prompt` (upstream convention; see grpo_math_1B.yaml's `32×16=512`) |
| `train_micro_batch_size` | n/a | `4` (1× and 2× A100 — same; DDP doesn't reduce per-GPU memory) | between upstream's tested `1.5B@seq=512@micro=4` and `8B@seq=4096@micro=1`; with `activation_checkpointing: true` should fit. Drop to 2 if OOM. |
| GRPO group size G | `5` | `5` (`num_generations_per_prompt`) | match |
| KL coef (β) | `0.001` | `0.001` (`reference_policy_kl_penalty`) | match |
| **KL estimator** | `low_var_kl` (Schulman 2020 k3) | `k3` (NeMo-RL default) | **byte-identical formula**; see [`VERL_REFERENCE.md`](VERL_REFERENCE.md) §2 |
| Clip ratio (ε) | `0.2` | `0.2` (`ratio_clip_min=ratio_clip_max=0.2`) | match |
| Max sequence length | `4096` | `4096` (`max_total_sequence_length`) | match |
| Max response length | `500` | `500` (`generation.max_new_tokens`) | match |
| **Max obs length** | `500` tokens per `<information>` block | `2000` chars (`env.search_r1.max_obs_chars`) | **char proxy** — pure-Python parsers, no tokenizer; ~4 char/token for English Wiki text |
| Rollout temperature / top-p | `1.0` / `1.0` | `1.0` / `1.0` | match |
| Max search turns | `4` (`max_turns=4`) | `4` (`env.search_r1.max_turns`, `grpo.max_rollout_turns`) | match |
| **State masking** | `True` — mask `<information>` from policy gradient | automatic via role-based `token_loss_mask` ([grpo.py:1685-1693](../../training/nemo_rl/nemo_rl/algorithms/grpo.py#L1685-L1693)) | **equivalent**, no config knob; env emits `role: tool` → loss=0 |
| GAE λ / γ | `1` / `1` | n/a — GRPO uses leave-one-out, not GAE | paper Appendix lists these but PPO-only |
| Optimizer / Precision | Adam / bf16 | AdamW / bf16 | match (AdamW = Adam with decoupled weight decay; same gradient direction) |
| Gradient checkpointing | enabled | enabled (`activation_checkpointing: true`) | match |
| FSDP CPU offload | enabled (verl-side memory budget tight on 8× H100) | off (`dtensor_cfg.cpu_offload: false`) | DTensor on A100 80GB has more headroom; flip to `true` if OOM persists |
| GPU memory utilization | `0.6` (verl) | `0.6` on 1× A100 / `0.7` on 2× A100 | conservative first-run defaults; raise after observing W&B `gpu_monitoring` |

## 7. Compute

> **High uncertainty until first run.** The projections below are derived from upstream NeMo-RL math benchmarks scaled by model-size + sequence-length + multi-turn factors, with a wide range to reflect the unknowns (retrieval HTTP latency, DTensor-vs-FSDP throughput, multi-turn batched generation efficiency). **Replace with observed numbers after the first 100 training steps complete on real Vast.ai hardware** — that's enough signal to project accurately.

| | Paper | Ours — projected (1× A100) | Ours — projected (2× A100) | Ours — observed |
|---|---|---|---|---|
| Hardware | 1 node × 8× H100 | 1× A100 80GB | 2× A100 80GB | TBD |
| Training steps | 1005 | 1005 | 1005 | TBD |
| Trajectories / step | 512 (verl) | 510 (102 × 5) | 510 | — |
| Total trajectories | ~515k | ~513k | ~513k | — |
| **Wall-clock / run** | ~24 h | **~50–150 h** | **~30–90 h** | **TBD** |
| GPU-hours / run | ~192 GPU-h | 50–150 GPU-h | 60–180 GPU-h | TBD |
| Vast.ai A100 80GB rate | n/a | ~$1–2 / GPU-h | ~$1–2 / GPU-h | check at launch |
| **$ / run** | n/a | **$50–300** | **$60–360** | TBD |
| Multi-seed plan | single-seed | 3 seeds × {base, hybrid} = 6 runs | same | — |
| **Total Phase-2 budget** | n/a | **$300–1800** | **$360–2160** | — |

### Derivation

**Paper baseline.** 8× H100 takes ~24 h to do 1005 steps with 512 trajectories per step (≈ 24 GPU-h × 8 = 192 GPU-h, rough estimate from arXiv Fig. 4). Per trajectory: ~1.3 GPU-sec.

**1× A100 scaling.** A100 bf16 is ~1.6× slower than H100. Single-GPU loses parallelism vs the paper's 8 GPUs but doesn't pay communication overhead. Net: **~10–12× slower per trajectory**. 1005 steps × 510 trajectories × ~13 sec/traj ≈ 1.8M GPU-sec ≈ 500 GPU-h … but that's worst-case. Realistic with vLLM batched generation + sequence packing: **50–150 h**. The wide range covers retrieval HTTP latency (CPU-bound on the same instance) and rollout-generation efficiency for the multi-turn loop.

**2× A100 scaling.** DDP across 2 GPUs cuts per-step train time ~2×; TP=2 vLLM rollout cuts per-step rollout ~1.5–2×. Net: ~1.7× wall-clock speedup over 1× A100, but 2× GPU-hours per wall-clock hour. Projection: **30–90 h × 2 = 60–180 GPU-h**.

### First-run gate

Run **one seed on 1× A100** first. After step 100 (first validation point):
- If wall-clock so far ≤ 5 h → projects ~50 h end-to-end, proceed with 6-run plan on 1× A100.
- If wall-clock 5–15 h → projects 50–150 h, evaluate cost-vs-time trade-off; consider 2× A100 for the remaining 5 runs.
- If > 15 h or `train/reward_mean` is flat at ~0 (model not finding any rewardable trajectories — first-pass has no `val/*` to lean on) → abort, debug.

Update this table from the W&B run summary once the first run completes.

**Throughput delta:** the paper's 8× H100 setup is roughly 4–6× our 1× A100 (rough rule of thumb: H100 ≈ 1.6× A100 for bf16, × 8 GPUs / 1 GPU). Expect our wall-clock per run to be ~4–6× the paper's. The 500-step training is short enough that this is acceptable; if it's not, we'll move to 2× A100 or rent H100 fleet on Vast.ai.

## 8. Data

| | Paper | Ours |
|---|---|---|
| Training corpus | NQ-train + HotpotQA-train (mixed) — released as [`PeterJinGo/nq_hotpotqa_train`](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train) (170k train + 51.7k test) | **Identical source**; prompt rewritten during conversion to use Qwen3.5's `<tool_call>` template (the dataset's `prompt[0].content` ships with paper's `<search>` tags) |
| Format | parquet (verl's expected format) | parquet → NeMo-RL row format (mapping pinned in [`TRAINING_DATA.md`](TRAINING_DATA.md)) |
| Retrieval index | E5-base-v2 + Wiki-18 FAISS Flat IP | **Identical** — same index from Milestone 1 |

## 9. Divergence summary (one place to glance)

Knowing departures from the paper, in priority order:

1. **Model family**: Qwen3.5-2B (vs. Qwen2.5-3B). Forced by NeMo-RL's Qwen3.5 support. Net: probably comparable; Qwen3.5's stronger base mostly offsets the 1B param drop.
2. **Chat template**: Qwen3.5 native `<tool_call>` (vs. paper's `<search>`). Deliberate; rationale in [`CHAT_TEMPLATE.md`](CHAT_TEMPLATE.md).
3. **Variant naming**: "hybrid" (= Qwen3.5-2B with soft-switch reasoning) replaces "instruct". The Qwen3.5 family does not ship an Instruct variant.
4. **Hardware**: 1×–2× A100 80GB (vs. 8× H100). Wall-clock will be longer; results should be unchanged.
5. **RL framework**: NeMo-RL (vs. verl). Core algorithm (GRPO with leave-one-out advantage, KL to ref, clip 0.2, β 0.001) is identical; engineering details (DTensor vs. FSDP) differ.

Everything else — reward function, group size, learning rate, warmup, batch sizes, sequence lengths, sampling temperature, training steps, validation set — is matched to the paper.
