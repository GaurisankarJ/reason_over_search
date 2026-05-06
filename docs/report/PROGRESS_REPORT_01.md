---
title: PROGRESS REPORT 01
tags: []
source: internal
created: 2026-05-05
updated: 2026-05-06
---

Leiden University
Leiden Institute of Advanced Computer Science (LIACS)

# Progress report 1

Gaurisankar Jayadas

| | |
|---|---|
| **Date:** | 2026-05-04 |
| **Subject:** | Resource-constrained post-training of small LMs for search-augmented multi-hop QA: Search-R1 reproduction and pivot to Qwen3.5-2B on NeMo-RL |
| **Mentor:** | Prof. dr. Aske Plaat |
| **Co-supervisor:** | Dr. Álvaro Serra Gomez |
| **Project term:** | February 2026 – July 2026 |
| **E-mail:** | gaurisankarj1996@gmail.com |
| **Window covered:** | 2026-04-23 – 2026-05-04 |

## Outcome of previous meeting

- Run a Search-R1 [1] evaluation baseline analysis: re-evaluate the published `Qwen2.5-3B-Search-R1` checkpoints on the paper's seven QA datasets so we have a clean numerical anchor for downstream training comparisons.
- Keep the original project framing (Plan B), extending RLVR to non-verifiable domains via tool-use as a surrogate objective, as the working scaffold for the thesis.

## Results during the last weeks

- **Phase-1 Qwen3-0.6B training synthesised.** Two training blocks completed earlier on the ALICE cluster (1× A100) were aggregated. v0 (Apr 3 to Apr 9, 14 runs, paper `<search>` / `<result>` tags) tested base vs. hybrid plus a nine-prompt ablation on hybrid varying rules verbosity and the few-shot example. v1 (Apr 12 to Apr 19, 15 runs, in-distribution `<tool_call>` tag format) tested three new prompts on hybrid plus five fresh base-model cold-start attempts. Headline findings:
  - **Base model cannot follow the tool-use format from cold start.** Across the five v1 base-model runs (longest 2300 steps), tool-call rate stayed at 0 throughout. Without instruction tuning or an SFT warm-start, GRPO does not bootstrap the structured format on a 0.6B base. Decision: focus on the hybrid checkpoint.
  - **Hybrid does learn, slowly.** All nine prompt-ablation runs on hybrid showed monotonic reward gains; final means clustered at 0.18 to 0.22 over up to 2300 steps. Training is numerically stable on 1× A100-40GB at the constrained sequence budget.
  - **Prompt drives behavior more than reward.** With training config and reward held constant, prompt phrasing alone moved end-of-run tool-call rate from 0 to 2 per episode and response length from ~480 to ~2050 tokens; reward only moved ±3 pp across the same prompts.
  - **The paper's partial-credit reward creates a floor at 0.1** for any well-formatted but wrong answer. Even runs that abandoned tool use entirely finished at 0.16 mean reward; tool-using runs at 0.18 to 0.22. The 3 to 6 pp gap between tool-using and no-tool behaviors is too small to clearly drive learning.
  - **The `<tool_call>` in-distribution tag format costs nothing at equal step count.** v1's best instruct prompt at 884 steps reached reward 0.179, matching v0's mid-pack at the same step count; v0's higher peak (0.215) just reflects 2.6× more training steps.

- **Search-R1 evaluation baseline reproduced within paper margin.** No eval pipeline was provided upstream; one was built from scratch and locked after a 3-fix audit (`apply_chat=True` for base, restored prompt example sentence, removed runtime `add_special_tokens`). 1000-row stratified subsamples for the five large datasets (NQ, TriviaQA, PopQA, HotpotQA, 2Wiki), plus full Bamboogle (125) and full MuSiQue (2417); single seed, greedy decode. Results across the seven datasets: base **0.292** EM (paper 0.312, Δ −2.0 pp), instruct **0.361** EM (paper 0.336, Δ +2.5 pp); within ±5 pp of paper on 6/7 datasets. Format-validity (`</answer>` close-rate) ≥99.6 % on base and ≥91.4 % on instruct.

- **Pivot to Qwen3.5-2B on NeMo-RL.** Qwen3-0.6B training is too slow on 1× A100 to support reward-function ablation in the remaining timeline, so training pivoted to Qwen3.5-2B (closer to paper's Qwen2.5-3B in capacity, same hybrid soft-switch reasoning toggle). `verl` does not support Qwen3.5, so the Search-R1 training pipeline was ported to NeMo-RL (NVIDIA's GRPO/RLVR framework with Qwen3.5 support): custom Ray-actor environment for the retriever, byte-identical port of the paper's reward function with 15 parity tests passing, and unchanged paper hyperparameters.

- **Wall-clock observed from Vast.ai 1× A100 80GB smoke.** \~57 s/step for 20 trajectories. Scaled to our target batch shape (510 trajectories/step, 1005 steps; 1/5th the paper's 2560 traj/step, because we use 102 prompts/step vs the paper's 512), this projects to ~11 to 17 days per run on 1× A100 80GB (\~\$300 to \$490 / run on Vast). The paper's exact batch shape on 1× A100 would be infeasible: ~55 to 85 days (~\$1600 to \$2400) per run due to the 5× larger rollout per step. On 1× H100 80GB SXM: ~5 to 8.5 days, \$240 to \$410 / run. On 2× A100: ~6.5 to 9.5 days, \$370 to \$550 / run.

- **Reframed research question.** Original RQ1 to RQ4 (domain expansion approaches, reward-function modeling, meta-reasoning, curriculum-based training) require a sweep that the remaining compute budget cannot afford. New framing: *"Is it feasible to post-train a small LM to Search-R1-level results under realistic resource constraints, and what is the optimised training recipe?"* The artifacts already produced (prompt-sensitivity, partial-credit floor, reproduced eval baseline, ported NeMo-RL pipeline) become the primary evidence; answerable with one or two trained runs rather than a sweep.

## Plans for next weeks

The proposed training recipe is three drop-in additions to a single Search-R1 GRPO baseline run on Qwen3.5-2B. All three were chosen because they target the "1× A100 budget" reframing and are cheap to add.

- **E2H data-level curriculum** [2]: train NQ (1-hop) → HotpotQA (2-hop) → MuSiQue (3-hop), ~300 steps per stage, fading out earlier stages. Proven that vanilla GRPO fails on hard tasks for 1.5 to 3B LLMs and that curriculum scheduling recovers them across multiple domains. Cost to add: data scheduler only, no GRPO code change.
- **S-GRPO** [3]: compute the policy-gradient loss on 30 to 50 % of tokens per rollout (informativeness-sampled). Proven on Qwen2-1.5B + LoRA: vanilla GRPO produced 0 gain over the base model; S-GRPO reached +24 pp on SVAMP. Roughly 2× backprop saving per step. Cost: one change in the loss path.
- **MC-GRPO** [4]: replace the mean advantage baseline with the group median (G+1 rollouts, exclude pivot from backprop). Proven to close the G=2 to G=8 quality gap to within 1 % across model families. Cost: one-line baseline swap.

Combined, these are the candidate answer to *"what is the optimised training recipe?"*. They stack on a single GRPO baseline run, fit on 1× A100, and are testable within 2 to 3 affordable runs.

**Required control.** Per JustRL [5], stacking curriculum / length-penalty / dynamic-sampling tricks on top of plain GRPO can *degrade* OOD performance by collapsing exploration. A plain-GRPO control will be run alongside the recipe; if the minimal control beats the stack, the stack is hurting.

## References

[1] *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning.* arXiv [2503.09516](https://arxiv.org/abs/2503.09516), 2025.

[2] Y. Lin et al. *Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning.* arXiv [2506.06632](https://arxiv.org/abs/2506.06632), 2026.

[3] *Token-Efficient RL for LLM Reasoning (S-GRPO).* arXiv [2504.20834](https://arxiv.org/abs/2504.20834), 2026.

[4] *MC-GRPO: Median-Centered Group Relative Policy Optimization for Small-Rollout RL.* arXiv [2601.22582](https://arxiv.org/abs/2601.22582), 2026.

[5] *JustRL: Simple RL Recipe for 1.5B Models.* arXiv [2512.16649](https://arxiv.org/abs/2512.16649), 2025.
