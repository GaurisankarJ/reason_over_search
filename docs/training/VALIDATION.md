# Validation Set & Cadence

In-loop validation during GRPO training. Mirrors Search-R1's validation setup so the training dynamics are comparable to the paper.

Sources: [Search-R1 arXiv 2503.09516](https://arxiv.org/abs/2503.09516), [Search-R1 GitHub](https://github.com/PeterGriffinJin/Search-R1).

> **Status**: paper-side numbers and conventions are settled. The exact verl `test_freq` value (validation cadence in steps) was not visible from the paper or the GitHub README excerpt; **flagged in §6 as a "verify before first run" item.**

---

## 1. Training corpus (for context)

Search-R1 trains on a **mix of NQ-train and HotpotQA-train**, both from the standard FlashRAG / Search-R1 distribution. We use the same files in [`data/`](../../data/) (already present from Milestone 1).

Total training questions: ~170k. With paper's 500 steps × global batch 512, the loop sees ~256k question instances — i.e., roughly 1.5 epochs over the training mix.

## 2. Validation dataset

**Paper:** Search-R1 logs in-loop validation on **the same NQ + HotpotQA splits** they use for the final test (held out from training). Specifically the Milestone 1 jsonl files at:

- [`data/nq.jsonl`](../../data/nq/test.jsonl) (NQ-test)
- [`data/hotpotqa.jsonl`](../../data/hotpotqa/dev.jsonl) (HotpotQA-dev)

These are the in-distribution benchmarks; out-of-distribution datasets (Bamboogle, MuSiQue, 2Wiki, PopQA, TriviaQA) are eval-time only — *not* used during training validation.

**Ours:** match the paper. Use the same two jsonl files. To avoid overfitting validation choice to a single sample, we'll use a **deterministic 1k subsample** ([`data_subsample/nq_1k.jsonl`](../../data_subsample/nq/test.jsonl), [`data_subsample/hotpotqa_1k.jsonl`](../../data_subsample/hotpotqa/dev.jsonl)) for fast in-loop validation; the full sets are reserved for post-training eval.

## 3. Cadence

**Paper:** model checkpoints saved **every 100 training steps** ([Appendix B.2](https://arxiv.org/html/2503.09516)). Validation appears tied to checkpointing (validate when checkpointing) but the exact `test_freq` value is not stated in the paper text.

**Ours:** match — `checkpointing.save_period: 100` in [`NEMO_RL_KNOBS.md`](NEMO_RL_KNOBS.md). Validation runs alongside each checkpoint (every 100 steps).

With paper's 500-step training, that gives **5 validation points** per run plus step 0 (sanity baseline) = **6 total**.

## 4. Metrics logged to W&B

At every validation step:

| Metric | Source |
|---|---|
| `val/em` | Per-dataset exact-match using the existing scorer in [`flashrag/search_r1/reward.py`](../../evaluation_search_r1/flashrag/search_r1/reward.py) |
| `val/format_validity` | Fraction of traces where the final response contains a closed `<answer>...</answer>` |
| `val/length_truncation` | Fraction of traces hitting `max_new_tokens` (stop_reason ≠ stop/eos) |
| `val/mean_answer_tokens` | Mean token count of the answer span |
| `val/mean_search_calls` | Average number of `<tool_call>` invocations per trace (Qwen3.5 native template) |
| `val/mean_response_tokens` | Mean total response length |

Per-step training metrics (already standard in NeMo-RL):

- `train/reward_mean`, `train/reward_std`
- `train/kl`, `train/policy_loss`, `train/clip_fraction`
- `train/lr`, `train/grad_norm`
- `train/rollout_throughput` (tokens/s)

## 5. Stopping criteria

**Paper:** fixed schedule of 500 steps; no early stopping ([Appendix B.2](https://arxiv.org/html/2503.09516)).

**Ours:** match. Run all 500 steps. Save the best-by-`val/em` checkpoint via NeMo-RL's `keep_top_k: 3` mechanism; that's the candidate we feed into the Milestone 1 eval pipeline.

If reward is collapsing or training is unstable (NaN loss, KL spike), abort manually rather than auto-stop — we want the failure mode visible in W&B for diagnosis, not silently truncated.

## 6. Open questions

1. **Exact `test_freq` from upstream verl config.** We're matching the paper's 100-step checkpoint cadence and assuming validation runs alongside; verify this against the actual verl yaml in [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) before the first training run. If upstream uses a different `test_freq`, decide whether to match or stick with our checkpoint-aligned cadence.
2. **Validation subsample size.** 1k per validation point × 6 validations × {NQ, HotpotQA} × 5 GRPO rollouts (group=5) = 60k validation rollouts per run. If this dominates wall-clock, drop to 500-question subsamples.
3. **Validation rollout sampling.** Paper does not state whether validation rollouts use temp=0 or temp=1. Eval is greedy (Milestone 1 confirmed); training rollouts are temp=1. We default validation to **temp=1, single sample** to match the *training* distribution (the point of in-loop val is to track what the policy actually does during training). Final post-training eval uses the Milestone 1 pipeline at temp=0.
