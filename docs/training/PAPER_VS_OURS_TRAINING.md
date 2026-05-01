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

See [`VALIDATION.md`](VALIDATION.md) for the full plan. Summary:

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
| Warmup ratio | `0.285` | `0.285` | match (unusually high; revisit after first run) |
| **Total training steps** | **`1005`** (verl `total_training_steps=1005`; supersedes paper text's "500") | **`1005`** | match — see §5 note |
| Global batch size | `512` | `512` | match |
| PPO mini-batch | `256` | n/a in NeMo-RL — different abstraction | NeMo-RL uses gradient accumulation differently |
| PPO micro-batch | `64` | `train_micro_batch_size: 4` (NeMo default) — raise post-first-run | tune for memory |
| GRPO group size | `5` | `5` | match (`grpo.num_generations_per_prompt: 5`) |
| KL coef (β) | `0.001` | `0.001` | match (`loss.reference_policy_kl_penalty`) |
| **KL estimator** | **`low_var_kl`** (Schulman 2020 k3) | **`k3`** (NeMo-RL default; identical formula) | see [`VERL_REFERENCE.md`](VERL_REFERENCE.md) §2 |
| Clip ratio (ε) | `0.2` | `0.2` | match (`loss.clip_ratio`) |
| Max sequence length | `4096` | `4096` | match (`policy.max_total_sequence_length`) |
| Max response length | `500` | `500` | match (`policy.generation.max_new_tokens`) |
| **Max retrieved content length** | **`500`** (verl `max_obs_length=500`, per-`<information>` block) | **must enforce in env** | **TODO step 6**: cap retrieval response in `format_docs_*` (currently uncapped). |
| Max start length | `2048` (verl `max_start_length`) | n/a — NeMo-RL composes max_seq from prompt + completions | verl-specific composition |
| Rollout temperature | `1.0` | `1.0` | match |
| Rollout top-p | `1.0` | `1.0` | match |
| Max search turns | `4` (verl `max_turns=4` in v0.2) | `4` (`env.search_r1.max_turns: 4`) | match (set in step 4 env config) |
| **State masking** | **`True`** (verl) — mask `<information>` blocks from policy gradient | **automatic** in NeMo-RL via role-based `token_loss_mask` | see [`VERL_REFERENCE.md`](VERL_REFERENCE.md) §2 |
| GAE λ | `1` | n/a — GRPO uses leave-one-out, not GAE | (paper Appendix lists λ=1, γ=1 but those are PPO-only) |
| GAE γ | `1` | n/a — same as above |
| Optimizer | Adam | Adam | match |
| Precision | bf16 | bf16 | match |
| Gradient checkpointing | enabled | enabled (`activation_checkpointing: true`) | match |
| FSDP CPU offload | enabled | n/a — using DTensor | DTensor handles memory differently; revisit if OOM |

## 7. Compute

| | Paper | Ours (planned) |
|---|---|---|
| Hardware | 1 node × 8× H100 | **1× A100 80GB** (or 2× A100 80GB) |
| Training time | not stated explicitly in paper | **TBD** — record on first run |
| GPU-hours per run | not stated | **TBD** |
| $/run (Vast.ai) | n/a | **TBD** — Vast.ai A100 80GB ≈ $1–2/h depending on availability; multiply by GPU-hours |
| Multi-seed plan | (paper does single-seed training) | **3 seeds × {base, hybrid} = 6 runs** |

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
