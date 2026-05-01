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

The EM scorer must be byte-identical to [`flashrag/search_r1/reward.py`](../../evaluation_search_r1/flashrag/search_r1/reward.py) (the SQuAD-canonical normaliser used in Milestone 1 eval), so training-time reward and post-training EM are computed the same way.

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

| | Paper | Ours |
|---|---|---|
| Val datasets | NQ + HotpotQA (held-out portions) — used as in-loop validation | Match paper |
| Save cadence | every 100 steps | match (`checkpointing.save_period: 100`) |
| Total steps | 500 | match initially |
| Logged metrics | EM, reward, format-validity | EM, reward, format-validity, length-truncation, mean answer tokens |

## 6. Hyperparameters

All paper values from Appendix B.2.

| Hyperparameter | Paper | Ours | Notes |
|---|---|---|---|
| Learning rate | `1e-6` | `1e-6` | match |
| Warmup ratio | `0.285` | `0.285` | match (this is unusually high; revisit after first run) |
| Total training steps | `500` | `500` | match |
| Global batch size | `512` | `512` | match |
| Mini-batch size | `256` | n/a in NeMo-RL — replaced by `num_prompts_per_step` × `num_generations_per_prompt` | NeMo-RL uses a different batching abstraction |
| Micro-batch size | `64` | `train_micro_batch_size: 4` (NeMo-RL default) — raise if memory allows | NeMo-RL default; tune post-first-run |
| GRPO group size | `5` | `5` | match (`num_generations_per_prompt: 5`) |
| KL coef (β) | `0.001` | `0.001` | match (`loss.kl_coef`) |
| Clip ratio (ε) | `0.2` | `0.2` | match (`loss.clip_ratio`) |
| Max sequence length | `4096` | `4096` | match (`policy.max_total_sequence_length`) |
| Max response length | `500` | `500` | match (`policy.generation.max_new_tokens`) |
| Max retrieved content length | `500` | `500` | match — must be enforced in the retrieval-tool wrapper |
| Rollout temperature | `1.0` | `1.0` | match |
| Rollout top-p | `1.0` | `1.0` | match |
| GAE λ | `1` | n/a — GRPO uses leave-one-out, not GAE | (paper note: λ=1, γ=1 listed but PPO-only — GRPO ignores) |
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
