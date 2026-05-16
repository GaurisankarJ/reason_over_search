---
title: Open questions
tags: [meta, questions]
source: internal
created: 2026-05-07
updated: 2026-05-16
---

# Open questions

Running log of research questions and uncertainties to chase down later. One heading per question. When the user says "register this as a question" (or similar phrasing), append a new entry at the bottom following the template below; do not paraphrase the question, capture it as the user states it plus enough context for future-me to pick up cold.

To resolve a question, leave the entry in place and add a `**Resolution**` block beneath it (date, one-paragraph answer, and links to the source / PR / run / commit). Flip `**Status**` to `resolved`. Do not delete resolved entries; they double as a learning trail.

Cross-references: [`CONVERSATION_CONTEXT.md`](CONVERSATION_CONTEXT.md) lists the five "key tensions and unknowns" that drive the M2 ablation plan; this file is broader and lower-priority by default (things to look into when there is time, not things blocking a decision).

## Template

```
## Q<n>. <one-line question>

**Status**: open
**Filed**: YYYY-MM-DD
**Tags**: <comma-separated; e.g. systems, training, eval>

<Body: what is unknown, why it matters, and what to chase. Pointers to code paths, papers, runs, or framework versions where relevant.>
```

---

## Q1. Sequence packing in verl and NeMo-RL: works with Qwen3, not Qwen3.5

**Status**: open (preliminary answer filed; wall-clock follow-up pending)
**Filed**: 2026-05-07
**Tags**: systems, training, frameworks

**Question**: What is sequence packing in verl and NeMo-RL, and why does it work with the Qwen3 architecture but break with Qwen3.5?

**Context**: Verl not supporting Qwen3.5 is the load-bearing reason M2 ports to NeMo-RL ([CLAUDE.md "Gotchas"](../../claude/CLAUDE.md)). The 2026-05-06 smoke ran with packing OFF. If the dynamic-batching fallback turns out 1.5x to 2x slower than packing on the recipe-ablation shape, our 11 to 17 day wall-clock and 2 to 3 run budget shift.

**Answer**: Sequence packing concatenates multiple short sequences into one long tensor (no padding) and passes cumulative length offsets (`cu_seqlens`) to FlashAttention's varlen path; each segment only attends to itself. Verl exposes it as `use_remove_padding` ("RMPad"); NeMo-RL exposes it as `policy.sequence_packing`, supported on DTensor + Megatron since [PR #651](https://github.com/NVIDIA-NeMo/RL/pull/651) ([design doc](https://docs.nvidia.com/nemo/rl/latest/design-docs/sequence-packing-and-dynamic-batching.html); background: [HF blog](https://huggingface.co/blog/packing-with-FA2)).

Qwen3 dense models (incl. Phase 1's Qwen3-0.6B) are pure self-attention; the standard FA-varlen path handles them. **Qwen3.5 is hybrid**, interleaving Mamba-style GatedDeltaNet `linear_attention` layers with `full_attention` ([vLLM Qwen3.5](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html); interleaving recorded in [`training/fix/CHANGES.md:44-46`](../../training/fix/CHANGES.md)). The PyTorch chunked kernel `torch_chunk_gated_delta_rule` does not honour `cu_seqlens`; with packing on it raises `CUDA: illegal memory access` at `policy.get_logprobs()`. Mitigation in the active config: `policy.sequence_packing.enabled: false` + `policy.dynamic_batching.enabled: true` ([`grpo_qwen3.5_2b_1xa100.yaml:271-282`](../../training/configs/grpo_qwen3.5_2b_1xa100.yaml); confirmed in [`SMOKE_RESULTS_2026-05-06.md:25-29`](../training/SMOKE_RESULTS_2026-05-06.md)).

Verl stacks two extra blockers on top of the GDN issue: the RMPad allowlist historically excluded Qwen3 (added later, hence Phase 1's Qwen3-0.6B run on `verl_latest` worked; [verl#1305](https://github.com/volcengine/verl/issues/1305), [`RESULTS_m0_b.md:60`](../report/RESULTS_m0_b.md)); and Qwen3.5 is not loadable in verl's bundled transformers, while upgrading transformers breaks the vLLM rollout ([verl#5937](https://github.com/verl-project/verl/issues/5937)). Net: hard error, not silent drift; workaround is dynamic batching; NeMo-RL inherits the same GDN root cause but gives a clean off-switch.

**Open follow-ups**: quantify wall-clock cost of packing-off + dynamic batching on a real ablation shape; watch for an upstream GDN varlen fix ([Megatron-Bridge#1776](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/1776) is a related bug to track).

## Q2. Latent-space rollouts for GRPO: can we plan in a JEPA-style learned dynamics model instead of running the full transformer?

**Status**: open (idea-stage; not on any roadmap)
**Filed**: 2026-05-16
**Tags**: training, systems, world-models, future-work

**Question (user's words, 2026-05-16)**: "In the joint embedding system, there is a model to predict the next embedding instead of the actual raw information; so in a LoRA setting, in a transformer, can we not train something to predict the next LoRA activations instead of running the whole transformer, this could be a way to do simulations without running the whole transformer like MCTS in a latent space."

**Context (how the idea arose)**: I'd just web-searched JEPA (Yann LeCun's Joint Embedding Predictive Architecture; predicts representations of held-out input parts in latent space, not raw pixels / tokens). The user asked whether the same trick applies to a transformer: train a cheap predictor of "the next hidden state" (they said LoRA activations, see below) so search procedures like MCTS could expand many candidates in latent space without paying for full forward passes through the deep stack.

**Translation into the right vocabulary**: this is a *learned latent dynamics model* (a "world model") for an LLM, used for planning. Cousin literature: **MuZero** (Schrittwieser et al, 2020; MCTS in a learned latent without ever simulating the real environment), **Dreamer** series (Hafner et al; latent world models for RL), **Coconut** (Hao et al, Meta, late 2024; chain of thought in continuous space, feeds last hidden state back as next input embedding so reasoning skips the token bottleneck), **speculative decoding / Medusa / EAGLE** (cheap draft model proposes tokens, big model verifies; already in production), **pause / filler / thought tokens** (Goyal et al, Pfau et al; extra forward passes of internal computation before committing to a token). The user's idea pushes speculative-decoding further: do *search* in draft space, not just linear drafting.

**Why the LoRA framing is the wrong handle even though the instinct is right**: LoRA is a low-rank delta to *weights*, not a representation of activations. If you train "a LoRA on the base model" to predict the next hidden state, you still flow input through every transformer layer with the delta applied; you have not skipped compute. To actually skip the stack the predictor must be a *separately parameterized small network* (tiny MLP / shallow transformer / RNN) that takes the current hidden state or a KV-cache summary and outputs the predicted next hidden state, bypassing the deep stack. LoRA could still be useful as a *predictability prior*: fine-tune the base so its hidden state is easier for the predictor to model. But the predictor itself is not a LoRA.

**Hardest open problem**: compounding error. Each latent step drifts from where the real model would have gone. MuZero tolerates this by training dynamics jointly with the value head on real trajectories, so errors that do not move value get ignored. Pure next-hidden-state regression collapses to trivial copies (identity) or high-variance noise. Any LLM version of this needs a reward-relevant objective, not pixel-style reconstruction in hidden-state space.

**Where this connects to the thesis (the version worth pitching)**: GRPO spends most wall-clock on G rollouts per prompt. A latent rollout predictor that is 10x cheaper and ~80 % correlated with the real return could replace some real rollouts inside the advantage estimator; call it *model-based GRPO*. As far as I can find, this is novel for RLVR-with-search. **Catch**: Search-R1 / ReSearch rollouts contain *retriever calls* (a discrete branch into Wikipedia FAISS); you cannot latent-simulate the retriever. So a real instantiation would be a hybrid: real forward pass at decision points (token sampling, `<search>` emission, retriever call), latent extrapolation only across the cheap reasoning span between tool calls. That cuts the addressable wall-clock to whatever fraction of rollout time is spent in reasoning vs in retrieval + tool-call boundaries.

**Honest take**: the family of ideas is a real and active research direction; the framing to adopt is "learned latent dynamics + MuZero-style search," not "predict the next LoRA." The thesis-shaped version is model-based rollout amortization for GRPO with tool use. This is a much bigger bite than the 2026-06-10 experimentation deadline can absorb (would need a separate predictor architecture + a joint training objective + careful instrumentation against drift), so it stays as future-work / a thesis-extension or follow-up-paper hook, not an M6 candidate.

**Concrete pointers to chase if revisited**:
- MuZero ([Schrittwieser et al, 2020](https://arxiv.org/abs/1911.08265)) for the canonical "MCTS in learned latent" template.
- Coconut ([Hao et al, 2024](https://arxiv.org/abs/2412.06769)) for the closest LLM-side precedent (latent-space reasoning).
- I-JEPA ([arxiv 2301.08243](https://arxiv.org/abs/2301.08243)), V-JEPA ([Meta blog](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)), VL-JEPA ([arxiv 2512.10942](https://arxiv.org/abs/2512.10942)) for the JEPA family the user was inspired by.
- EAGLE / Medusa speculative decoding for the "cheap draft + verify" baseline the search-in-latent idea generalises.
